// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/InsertSync/LifecycleSynthesis.h"
#include "PTO/Transforms/InsertSync/InsertSyncAnalysis.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/MoveSyncState.h"
#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::insert_sync_frontier;

namespace {
using Slice = InsertSyncLifecyclePlan::Slice;
using Channel = InsertSyncLifecyclePlan::Channel;

std::optional<Slice> exactSlice(const BaseMemInfo* info)
{
    if (
        !info || info->scope == AddressSpace::GM || info->aliasesUnknownRange || !info->hasKnownPhysicalAddresses ||
        info->baseAddresses.size() != 1 || !info->allocateSize ||
        info->baseAddresses[0] > std::numeric_limits<uint64_t>::max() - info->allocateSize) {
        return std::nullopt;
    }
    return Slice{info->scope, info->baseAddresses[0], info->allocateSize};
}

bool mayOverlap(const BaseMemInfo* info, const Slice& slice)
{
    if (
        !info || info->scope != slice.space) {
        return false;
    }
    if (
        info->aliasesUnknownRange || !info->hasKnownPhysicalAddresses || info->baseAddresses.empty() ||
        !info->allocateSize) {
        return true;
    }
    for (uint64_t address : info->baseAddresses) {
        if (
            !disjointBytes(address, info->allocateSize, slice.begin, slice.bytes)) {
            return true;
        }
    }
    return false;
}

std::optional<unsigned> localLane(PIPE pipe, bool cube)
{
    if (
        pipe == PIPE::PIPE_MTE2) {
        return 0;
    }
    if (
        cube) {
        if (
            pipe == PIPE::PIPE_MTE1) {
            return 1;
        }
        if (
            pipe == PIPE::PIPE_M) {
            return 2;
        }
        if (
            pipe == PIPE::PIPE_FIX) {
            return 3;
        }
    } else {
        if (
            pipe == PIPE::PIPE_V) {
            return 1;
        }
        if (
            pipe == PIPE::PIPE_MTE3) {
            return 2;
        }
    }
    return std::nullopt;
}
PIPE physicalLane(unsigned lane, bool cube)
{
    if (
        lane == 0) {
        return PIPE::PIPE_MTE2;
    }
    if (
        cube) {
        if (
            lane == 1) {
            return PIPE::PIPE_MTE1;
        }
        if (
            lane == 2) {
            return PIPE::PIPE_M;
        }
        return PIPE::PIPE_FIX;
    }
    return lane == 1 ? PIPE::PIPE_V : PIPE::PIPE_MTE3;
}

// The two reviewed bidirectional domains for each execution context. No new
// cross-core, GM-visibility, M/FIX, scalar, or arbitrary FIFO target rule.
bool supportedDirection(unsigned producer, unsigned consumer)
{
    return (producer == 0 && consumer == 1) || (producer == 1 && consumer == 2);
}

std::optional<Channel> makeChannel(const InsertSyncLifecycleStructure& structure, ArrayRef<Slice> slices)
{
    Channel channel;
    channel.members.append(slices.begin(), slices.end());
    auto& spec = channel.logical.spec;
    spec.members = slices.size();
    spec.phases.resize(structure.phases.size());
    std::optional<unsigned> producer, consumer;
    for (unsigned p = 0; p < structure.phases.size(); ++p) {
        const auto* phase = structure.phases[p];
        auto add = [&](const BaseMemInfo* info, bool write) {
            for (unsigned m = 0; m < slices.size(); ++m) {
                if (
                    !mayOverlap(info, slices[m])) {
                    continue;
                }
                // Unknown/partial/multiple-address overlap vetoes this candidate,
                // rather than pretending the access selects a different slot.
                auto exact = exactSlice(info);
                if (
                    !exact || !(*exact == slices[m])) {
                    return false;
                }
                auto& lane = write ? producer : consumer;
                unsigned actual = structure.program.phaseLane[p];
                if (
                    lane && *lane != actual) {
                    return false;
                }
                lane = actual;
                (write ? spec.phases[p].writes : spec.phases[p].reads) |= 1u << m;
            }
            return true;
        };
        for (const auto* read : phase->useVec) {
            if (
                !add(read, false)) {
                return std::nullopt;
            }
        }
        for (const auto* write : phase->defVec) {
            if (
                !add(write, true)) {
                return std::nullopt;
            }
        }
    }
    if (
        !producer || !consumer || !supportedDirection(*producer, *consumer)) {
        return std::nullopt;
    }
    spec.producerLane = *producer;
    spec.consumerLane = *consumer;
    return channel;
}

InsertSyncLifecyclePlan discover(
    const InsertSyncLifecycleStructure& structure, Budget& budget, unsigned& attempted, bool& invalid)
{
    InsertSyncLifecyclePlan plan;
    plan.cube = structure.cube;
    plan.lifetimeScope = structure.lifetimeScope;
    std::vector<Slice> slices;
    for (unsigned p = 0; p < structure.phases.size(); ++p) {
        const auto* phase = structure.phases[p];
        plan.phaseIds[phase->elementOp] = p;
        plan.phases.push_back(phase->elementOp);
        for (const auto* write : phase->defVec) {
            auto slice = exactSlice(write);
            if (
                !slice || (slice->space != AddressSpace::VEC && slice->space != AddressSpace::MAT &&
                           slice->space != AddressSpace::LEFT && slice->space != AddressSpace::RIGHT)) {
                continue;
            }
            if (
                std::find(slices.begin(), slices.end(), *slice) == slices.end()) {
                slices.push_back(*slice);
            }
        }
    }
    std::sort(slices.begin(), slices.end(), [](const Slice& a, const Slice& b) {
        return std::tie(a.space, a.begin, a.bytes) < std::tie(b.space, b.begin, b.bytes);
    });
    if (
        slices.size() > 32) {
        return plan;
    }
    std::vector<std::optional<Channel>> singles;
    for (const auto& slice : slices) {
        singles.push_back(makeChannel(structure, ArrayRef<Slice>(&slice, 1)));
    }
    std::vector<bool> selected(slices.size(), false);
    auto qualify = [&](Channel& channel) {
        ++attempted;
        channel.logical.certificate = recognizeLifecycle(structure.program, channel.logical.spec, budget);
        invalid |= channel.logical.certificate.status == LifecycleCertificate::Status::InvalidInput;
        // Do not add a primed round trip to a statically non-reused one-shot
        // buffer. General insertion already handles its availability.
        return channel.logical.certificate.status == LifecycleCertificate::Status::Complete &&
               channel.logical.certificate.mayReuse;
    };
    // LEFT/RIGHT are bundled only when every consumer reads both members. A
    // consumer of just one side vetoes bundling; grouping never postpones an
    // independent consumer merely to save an event stream.
    if (
        structure.cube) {
        for (unsigned a = 0; a < slices.size(); ++a) {
            if (
                slices[a].space != AddressSpace::LEFT || !singles[a]) {
                continue;
            }
            for (unsigned b = 0; b < slices.size(); ++b) {
                if (
                    selected[b] || slices[b].space != AddressSpace::RIGHT || !singles[b]) {
                    continue;
                }
                bool same = true, any = false;
                for (unsigned p = 0; p < structure.phases.size(); ++p) {
                    bool x = singles[a]->logical.spec.phases[p].reads != 0;
                    bool y = singles[b]->logical.spec.phases[p].reads != 0;
                    same &= x == y;
                    any |= x;
                }
                if (
                    !same || !any) {
                    continue;
                }
                SmallVector<Slice, 2> members{slices[a], slices[b]};
                auto bundle = makeChannel(structure, members);
                if (
                    bundle && qualify(*bundle)) {
                    selected[a] = selected[b] = true;
                    plan.channels.push_back(std::move(*bundle));
                    break;
                }
            }
        }
    }
    for (unsigned i = 0; i < singles.size(); ++i) {
        if (
            selected[i] || !singles[i]) {
            continue;
        }
        if (
            qualify(*singles[i])) {
            plan.channels.push_back(std::move(*singles[i]));
        }
    }
    for (unsigned i = 0; i < plan.channels.size(); ++i) {
        plan.channels[i].logical.identity = i;
    }
    return plan;
}

// Capture original IR identities before emission. Added operations are restricted
// to synchronization by this constructor; codegen retains its existing contract.
struct OriginalOperation {
    Operation* op;
    DictionaryAttr attributes;
    SmallVector<Value> operands;
    SmallVector<Type> results;
};
std::vector<OriginalOperation> capture(func::FuncOp function)
{
    std::vector<OriginalOperation> result;
    function.getBody().walk([&](Operation* op) {
        result.push_back(
            {op, op->getAttrDictionary(), SmallVector<Value>(op->getOperands().begin(), op->getOperands().end()),
             SmallVector<Type>(op->getResultTypes().begin(), op->getResultTypes().end())});
    });
    return result;
}
bool preserved(func::FuncOp function, const std::vector<OriginalOperation>& original)
{
    llvm::SmallPtrSet<Operation*, 32> identities;
    for (const auto& record : original) {
        identities.insert(record.op);
    }
    std::vector<Operation*> now;
    function.getBody().walk([&](Operation* op) {
        if (
            identities.contains(op)) {
            now.push_back(op);
        }
    });
    if (
        now.size() != original.size()) {
        return false;
    }
    for (unsigned i = 0; i < now.size(); ++i) {
        const auto& record = original[i];
        if (
            now[i] != record.op || record.attributes != now[i]->getAttrDictionary() ||
            !llvm::equal(record.operands, now[i]->getOperands()) ||
            !llvm::equal(record.results, now[i]->getResultTypes())) {
            return false;
        }
    }
    return true;
}

void materialize(func::FuncOp function, const InsertSyncLifecyclePlan& plan, const LifecycleAllocation& allocation)
{
    auto make = [&](OpBuilder& builder, bool set, unsigned source, unsigned target, unsigned id, Location loc) {
        auto src = PipeAttr::get(function.getContext(), physicalLane(source, plan.cube));
        auto dst = PipeAttr::get(function.getContext(), physicalLane(target, plan.cube));
        auto event = EventAttr::get(function.getContext(), static_cast<EVENT>(id));
        if (
            set) {
            builder.create<SetFlagOp>(loc, src, dst, event);
        } else {
            builder.create<WaitFlagOp>(loc, src, dst, event);
        }
    };
    OpBuilder entry(function.getContext());
    Block& scope = plan.lifetimeScope->getRegion(0).front();
    entry.setInsertionPointToStart(&scope);
    for (unsigned c = 0; c < plan.channels.size(); ++c) {
        const auto& spec = plan.channels[c].logical.spec;
        make(entry, true, spec.consumerLane, spec.producerLane, allocation.free[c], function.getLoc());
    }
    for (unsigned p = 0; p < plan.phases.size(); ++p) {
        Operation* op = plan.phases[p];
        OpBuilder before(op), after(function.getContext());
        after.setInsertionPointAfter(op);
        for (unsigned c = 0; c < plan.channels.size(); ++c) {
            const auto& logical = plan.channels[c].logical;
            const auto& role = logical.certificate.roles[p];
            unsigned producer = logical.spec.producerLane, consumer = logical.spec.consumerLane;
            if (
                role.acquireFree) {
                make(before, false, consumer, producer, allocation.free[c], op->getLoc());
            }
            if (
                role.acquireReady) {
                make(before, false, producer, consumer, allocation.ready[c], op->getLoc());
            }
            if (
                role.publishReady) {
                make(after, true, producer, consumer, allocation.ready[c], op->getLoc());
            }
            if (
                role.publishFree) {
                make(after, true, consumer, producer, allocation.free[c], op->getLoc());
            }
        }
    }
    // Drain only at the real function lifetime exit, after original suffix work.
    // Preserve the existing automatic ALL as the final synchronization boundary.
    Operation* drainPoint = nullptr;
    for (Operation& operation : llvm::reverse(scope)) {
        if (
            operation.hasTrait<OpTrait::IsTerminator>()) {
            drainPoint = &operation;
        }
        if (
            auto barrier = dyn_cast<BarrierOp>(operation)) {
            if (
                barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
                drainPoint = &operation;
            }
        }
        if (
            isa<OpPipeInterface>(operation) || operation.getNumRegions()) {
            break;
        }
    }
    OpBuilder exit(function.getContext());
    if (
        drainPoint) {
        exit.setInsertionPoint(drainPoint);
    } else {
        exit.setInsertionPointToEnd(&scope);
    }
    for (unsigned c = 0; c < plan.channels.size(); ++c) {
        const auto& spec = plan.channels[c].logical.spec;
        make(exit, false, spec.consumerLane, spec.producerLane, allocation.free[c], function.getLoc());
    }
}

bool reconstruct(
    const InsertSyncLifecycleStructure& concrete, const InsertSyncLifecyclePlan& plan,
    const LifecycleAllocation& allocation, Budget& budget)
{
    for (unsigned c = 0; c < plan.channels.size(); ++c) {
        const auto& channel = plan.channels[c];
        const auto& spec = channel.logical.spec;
        std::vector<ReconstructedLifecycleNode> nodes(concrete.program.nodes.size());
        for (unsigned n = 0; n < nodes.size(); ++n) {
            nodes[n].next = concrete.program.nodes[n].next;
            Operation* anchor = concrete.anchors[n];
            const auto& node = concrete.program.nodes[n];
            if (
                node.kind == Node::Kind::Issue) {
                if (
                    node.phase >= concrete.phases.size()) {
                    return false;
                }
                const auto* phase = concrete.phases[node.phase];
                LifecycleTouch touch;
                auto recover = [&](const BaseMemInfo* access, bool write) {
                    for (unsigned member = 0; member < channel.members.size(); ++member) {
                        if (
                            !mayOverlap(access, channel.members[member])) {
                            continue;
                        }
                        auto slice = exactSlice(access);
                        if (
                            !slice || !(*slice == channel.members[member])) {
                            return false;
                        }
                        (write ? touch.writes : touch.reads) |= 1u << member;
                    }
                    return true;
                };
                for (const auto* a : phase->useVec) {
                    if (
                        !recover(a, false)) {
                        return false;
                    }
                }
                for (const auto* a : phase->defVec) {
                    if (
                        !recover(a, true)) {
                        return false;
                    }
                }
                if (
                    touch.reads && touch.writes) {
                    return false;
                }
                if (
                    (touch.writes && concrete.program.phaseLane[node.phase] != spec.producerLane) ||
                    (touch.reads && concrete.program.phaseLane[node.phase] != spec.consumerLane)) {
                    return false;
                }
                if (
                    touch.writes) {
                    nodes[n].action = LifecycleAction::Write;
                    nodes[n].members = touch.writes;
                }
                if (
                    touch.reads) {
                    nodes[n].action = LifecycleAction::Read;
                    nodes[n].members = touch.reads;
                }
            } else if (node.kind == Node::Kind::Signal || node.kind == Node::Kind::Wait) {
                PIPE source, target;
                unsigned id;
                if (
                    auto set = dyn_cast_or_null<SetFlagOp>(anchor)) {
                    source = set.getSrcPipe().getPipe();
                    target = set.getDstPipe().getPipe();
                    id = unsigned(set.getEventId().getEvent());
                } else if (auto wait = dyn_cast_or_null<WaitFlagOp>(anchor)) {
                    source = wait.getSrcPipe().getPipe();
                    target = wait.getDstPipe().getPipe();
                    id = unsigned(wait.getEventId().getEvent());
                } else {
                    return false;
                }
                auto s = localLane(source, plan.cube), t = localLane(target, plan.cube);
                if (
                    !s || !t) {
                    return false;
                }
                if (
                    *s == spec.producerLane && *t == spec.consumerLane && id == allocation.ready[c]) {
                    nodes[n].action =
                        node.kind == Node::Kind::Signal ? LifecycleAction::PublishReady : LifecycleAction::AcquireReady;
                } else if (*s == spec.consumerLane && *t == spec.producerLane && id == allocation.free[c]) {
                    // Prime/drain are recognized by their placement, not a tag.
                    bool atFunction = anchor->getBlock() == &plan.lifetimeScope->getRegion(0).front();
                    bool beforePhysical = atFunction;
                    if (
                        atFunction) {
                        for (Operation* prev = anchor->getPrevNode(); prev; prev = prev->getPrevNode()) {
                            if (
                                isa<OpPipeInterface>(prev) || prev->getNumRegions()) {
                                beforePhysical = false;
                            }
                        }
                    }
                    bool afterPhysical = atFunction;
                    if (
                        atFunction) {
                        for (Operation* next = anchor->getNextNode(); next; next = next->getNextNode()) {
                            if (
                                isa<OpPipeInterface>(next) || next->getNumRegions()) {
                                afterPhysical = false;
                            }
                        }
                    }
                    if (
                        node.kind == Node::Kind::Signal) {
                        nodes[n].action = beforePhysical ? LifecycleAction::Prime : LifecycleAction::PublishFree;
                    } else {
                        nodes[n].action = afterPhysical ? LifecycleAction::Drain : LifecycleAction::AcquireFree;
                    }
                }
            }
        }
        if (
            !verifyReconstructedLifecycle(nodes, spec.members, budget)) {
            return false;
        }
    }
    return true;
}
} // namespace

void InsertSyncLifecyclePlan::removeSuppliedDependencies(
    CompoundInstanceElement* source, CompoundInstanceElement* target, DepBaseMemInfoPairVec& pairs) const
{
    if (
        !source || !target) {
        return;
    }
    auto s = phaseIds.find(source->elementOp), t = phaseIds.find(target->elementOp);
    if (
        s == phaseIds.end() || t == phaseIds.end()) {
        return;
    }
    pairs.erase(
        std::remove_if(
            pairs.begin(), pairs.end(),
            [&](const auto& pair) {
                auto a = exactSlice(pair.first), b = exactSlice(pair.second);
                if (
                    !a || !b || !(*a == *b)) {
                    return false;
                }
                for (const auto& channel : channels) {
                    for (unsigned m = 0; m < channel.members.size(); ++m) {
                        if (
                            !(channel.members[m] == *a)) {
                            continue;
                        }
                        auto supply =
                            channel.logical.certificate.supplies(channel.logical.spec, s->second, t->second, m);
                        if (
                            supply != LifecycleCertificate::Supply::None) {
                            ++suppliedPairs;
                            return true;
                        }
                    }
                }
                return false;
            }),
        pairs.end());
}

InsertSyncLifecycleResult mlir::pto::tryInsertSyncLifecycleSynthesis(
    func::FuncOp function, const InsertSyncOptions& options)
{
    InsertSyncLifecycleResult result;
    auto parent = function->getParentOfType<ModuleOp>();
    if (
        !parent || !llvm::hasSingleElement(function.getBody())) {
        result.reason = "unchanged: lifecycle synthesis needs one structured function body";
        return result;
    }
    // User-authored barriers and event protocols remain fixed. Production also
    // bypasses explicit flags before this entry point; this is defense in depth.
    bool fixed = false;
    function.walk([&](Operation* op) {
        // The transactional clone contains only this function. Symbol users such
        // as func.constant would lose sibling definitions and make verification
        // fail even though ordinary InsertSync accepts the original module.
        fixed |= isa<BarrierOp, SetFlagOp, WaitFlagOp, RecordEventOp, WaitEventOp, func::CallOp,
                     func::ConstantOp>(op) ||
                 static_cast<bool>(getSyncMacroModel(op));
    });
    if (
        fixed) {
        result.reason = "unchanged: fixed input synchronization or helper/macro contract outside this constructor";
        return result;
    }
    OwningOpRef<ModuleOp> holder(ModuleOp::create(function.getLoc()));
    (*holder)->setAttrs(parent->getAttrs());
    IRMapping mapping;
    auto cloned = cast<func::FuncOp>(function->clone(mapping));
    holder->getBody()->push_back(cloned.getOperation());
    auto original = capture(cloned);
    auto gm = cloned->getAttrOfType<StringAttr>("pto.gm_alias");
    auto mode = gm && gm.getValue() == "assume-disjoint-arguments" ? InsertSyncGMAliasMode::DisjointArguments :
                                                                     InsertSyncGMAliasMode::MayAlias;
    MemoryDependentAnalyzer memory;
    memory.setGMContract(cloned, mode);
    SyncIRs ir;
    SyncOperations syncs;
    Buffer2MemInfoMap buffers;
    PTOIRTranslator translator(ir, memory, buffers, cloned, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    auto covered = inspectInsertSyncEffectCoverage(cloned, ir, options.effectCoverage == "strict");
    if (
        failed(covered)) {
        result.status = InsertSyncLifecycleResult::Status::InputError;
        result.reason = "requested contract/coverage validation failed before lifecycle construction";
        return result;
    }
    if (
        !*covered) {
        result.reason = "unchanged: effect coverage gap; ordinary report-mode translation retained";
        return result;
    }
    cloned->setAttr("pto.insert_sync.effect_coverage", StringAttr::get(cloned.getContext(), "complete"));
    Budget budget;
    auto structure = buildInsertSyncLifecycleStructure(cloned, ir, budget);
    if (
        structure.status != StorageFrontierSnapshot::Status::Complete) {
        if (
            structure.status == StorageFrontierSnapshot::Status::InternalError) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
        }
        result.reason = "structural import: " + structure.reason;
        return result;
    }
    bool invalidLifecycle = false;
    auto plan = discover(structure, budget, result.attempted, invalidLifecycle);
    if (
        invalidLifecycle) {
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "malformed selected lifecycle domain";
        return result;
    }
    if (
        !budget.left) {
        result.reason = "unchanged: lifecycle recognition budget";
        return result;
    }
    if (
        plan.channels.empty()) {
        result.reason = "unchanged: no complete exact-slot lifecycle";
        return result;
    }
    result.selected = plan.channels.size();
    result.logicalStreams = 2 * result.selected;

    // The shared logical plan exists BEFORE dependency insertion. Legacy
    // traversal retains and repairs every dependency not explicitly supplied.
    InsertSyncAnalysis analysis(ir, memory, syncs, cloned, SyncAnalysisMode::NORMALSYNC);
    analysis.setLifecycleSupply(&plan);
    analysis.Run(true, options.deferSamePipe, options.mmadChains);
    MoveSyncState move(ir, syncs);
    move.Run();
    RemoveRedundantSync redundant(ir, syncs, SyncAnalysisMode::NORMALSYNC);
    redundant.Run();
    SyncEventIdAllocation allocator(ir, syncs);
    allocator.Allocate();

    // Keep complete protocols out of legacy motion/redundancy/widening. Their
    // keys are allocated jointly with residual use, conservatively reserving
    // a whole-function key per stream. A failed realization rolls back all
    // omitted repairs and reruns ordinary InsertSync on the untouched original.
    ReservedLifecycleKeys occupied;
    for (const auto& group : syncs) {
        for (const auto& item : group) {
            const auto* sync = item.get();
            if (
                sync->uselessSync) {
                continue;
            }
            if (
                sync->isBarrierType()) {
                if (
                    sync->GetActualSrcPipe() == PipelineType::PIPE_ALL && !sync->IsAutoSyncTailBarrier()) {
                    result.reason = "unchanged: residual allocation required serialization";
                    return result;
                }
                continue;
            }
            if (
                !sync->isSyncSetType() && !sync->isSyncWaitType()) {
                continue;
            }
            auto source = localLane(static_cast<PIPE>(sync->GetActualSrcPipe()), plan.cube);
            auto target = localLane(static_cast<PIPE>(sync->GetActualDstPipe()), plan.cube);
            if (
                !source || !target || sync->eventIds.empty()) {
                result.reason = "unchanged: residual event realization unresolved";
                return result;
            }
            for (int id : sync->eventIds) {
                if (
                    id < 0 || id >= 8) {
                    result.reason = "unchanged: residual key out of range";
                    return result;
                }
                occupied[{*source, *target}] |= 1u << id;
            }
        }
    }
    std::vector<LogicalLifecycle> logical;
    for (const auto& channel : plan.channels) {
        logical.push_back(channel.logical);
    }
    auto allocation = allocateLifecycles(logical, occupied);
    if (
        allocation.status != LifecycleAllocation::Status::Complete) {
        result.reason = "unchanged: " + allocation.reason;
        return result;
    }
    SyncCodegen codegen(ir, cloned, SyncAnalysisMode::NORMALSYNC);
    codegen.Run();
    unsigned allCount = 0;
    bool bodyAll = false;
    cloned.walk([&](BarrierOp barrier) {
        if (
            barrier.getPipe().getPipe() != PIPE::PIPE_ALL) {
            return;
        }
        ++allCount;
        bodyAll |= static_cast<bool>(barrier->getParentOfType<scf::ForOp>()) ||
                   static_cast<bool>(barrier->getParentOfType<scf::IfOp>());
    });
    if (
        bodyAll || allCount > 1) {
        result.reason = "unchanged: residual output contains a broad body cut";
        return result;
    }
    materialize(cloned, plan, allocation);
    if (
        !preserved(cloned, original) || failed(verify(cloned))) {
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "lifecycle construction changed original payload or produced invalid MLIR";
        return result;
    }
    // Re-import actual emitted events. No planner-supplied coverage bit is used
    // to check event matching, current-slot state or consumption-before-rearm.
    SyncIRs checkedIR;
    Buffer2MemInfoMap checkedBuffers;
    PTOIRTranslator checked(checkedIR, memory, checkedBuffers, cloned, SyncAnalysisMode::NORMALSYNC);
    checked.Build();
    auto concrete = buildInsertSyncLifecycleStructure(cloned, checkedIR, budget);
    if (
        concrete.status != StorageFrontierSnapshot::Status::Complete) {
        if (
            concrete.status == StorageFrontierSnapshot::Status::InternalError) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
        }
        result.reason = "concrete event reconstruction: " + concrete.reason;
        return result;
    }
    // Dedicated keys let the new protocols carry their own causal/rearm proof.
    // Check them without relying on residual acknowledgements, then check the
    // combined event plan as well. Unproved composition abandons the optional
    // candidate; it never turns into rejection of the original input.
    Program protocolEvents = concrete.program;
    for (unsigned n = 0; n < protocolEvents.nodes.size(); ++n) {
        auto& node = protocolEvents.nodes[n];
        if (
            node.kind != Node::Kind::Signal && node.kind != Node::Kind::Wait) {
            continue;
        }
        Operation* op = concrete.anchors[n];
        PIPE source, target;
        unsigned id;
        if (
            auto set = dyn_cast_or_null<SetFlagOp>(op)) {
            source = set.getSrcPipe().getPipe();
            target = set.getDstPipe().getPipe();
            id = static_cast<unsigned>(set.getEventId().getEvent());
        } else if (auto wait = dyn_cast_or_null<WaitFlagOp>(op)) {
            source = wait.getSrcPipe().getPipe();
            target = wait.getDstPipe().getPipe();
            id = static_cast<unsigned>(wait.getEventId().getEvent());
        } else {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "concrete event anchor missing";
            return result;
        }
        auto from = localLane(source, plan.cube), to = localLane(target, plan.cube);
        bool owned = false;
        for (unsigned c = 0; from && to && c < plan.channels.size(); ++c) {
            const auto& spec = plan.channels[c].logical.spec;
            owned |= (*from == spec.producerLane && *to == spec.consumerLane && id == allocation.ready[c]) ||
                     (*from == spec.consumerLane && *to == spec.producerLane && id == allocation.free[c]);
        }
        if (
            !owned) {
            node.kind = Node::Kind::Pass;
            node.phase = node.key = kInvalid;
        }
    }
    auto events = completion(protocolEvents, Bits(protocolEvents.nodes.size()), budget);
    if (
        events.status != CompletionResult::Status::Complete || !events.eventsProved) {
        if (
            events.status == CompletionResult::Status::InvalidInput) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
        }
        result.reason = "new protocol consumption-before-rearm proof unresolved";
        return result;
    }
    if (
        !reconstruct(concrete, plan, allocation, budget)) {
        if (
            !budget.left) {
            result.reason = "unchanged: reconstruction budget";
            return result;
        }
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "emitted lifecycle does not implement its complete protocol";
        return result;
    }
    Budget auditBudget;
    auditBudget.left = 1000000;
    auto whole = completion(concrete.program, Bits(concrete.program.nodes.size()), auditBudget);
    if (
        whole.status == CompletionResult::Status::InvalidInput) {
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "malformed concrete graph during combined event audit";
        return result;
    }
    result.combinedEventAuditProved = whole.status == CompletionResult::Status::Complete && whole.eventsProved;
    if (
        !result.combinedEventAuditProved) {
        result.reason =
            "unchanged: combined residual event transfer unproved at node " + std::to_string(whole.unprovedEventNode);
        return result;
    }
    result.suppliedPairs = plan.suppliedPairs;
    function.getBody().takeBody(cloned.getBody());
    function->setAttrs(cloned->getAttrs());
    result.status = InsertSyncLifecycleResult::Status::Applied;
    result.reason = "pre-insertion exact-slot lifecycle plan plus legacy residuals; no added scarcity ordering";
    return result;
}
