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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    return LocalStorageRequirements::exact(info);
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

std::string channelIdentity(ArrayRef<Slice> slices)
{
    std::string result;
    for (const auto& slice : slices) {
        if (
            !result.empty()) {
            result += "+";
        }
        result += std::to_string(static_cast<unsigned>(slice.space)) + ":[" + std::to_string(slice.begin) + "," +
                  std::to_string(slice.begin + slice.bytes) + ")";
    }
    return result;
}

std::optional<Channel> makeChannel(
    const InsertSyncLifecycleStructure& structure, ArrayRef<Slice> slices, bool generations = false)
{
    Channel channel;
    channel.bufferGenerations = generations;
    channel.members.append(slices.begin(), slices.end());
    channel.identity = channelIdentity(slices);
    auto& spec = channel.logical.spec;
    spec.members = slices.size();
    spec.phases.resize(structure.phases.size());
    if (!structure.requirements) return std::nullopt;
    for (unsigned m = 0; m < slices.size(); ++m) {
        const auto* projection = structure.requirements->project(slices[m]);
        if (!projection || !projection->complete) return std::nullopt;
        for (unsigned p = 0; p < spec.phases.size(); ++p) {
            if (projection->readers.test(p)) spec.phases[p].reads |= 1u << m;
            if (projection->writers.test(p)) spec.phases[p].writes |= 1u << m;
        }
    }
    std::optional<unsigned> producer, consumer;
    for (unsigned p = 0; p < structure.phases.size(); ++p) {
        if (spec.phases[p].writes) {
            unsigned lane = structure.program.phaseLane[p];
            if (producer && *producer != lane) return std::nullopt;
            producer = lane;
        }
    }
    if (!producer) return std::nullopt;
    for (unsigned p = 0; p < spec.phases.size(); ++p) {
        auto& t = spec.phases[p];
        if (!t.reads) continue;
        unsigned lane = structure.program.phaseLane[p];
        if (generations && lane == *producer && t.reads == t.writes) {
            t.updates = t.reads;
            t.reads = 0;
            continue;
        }
        if (consumer && *consumer != lane) return std::nullopt;
        consumer = lane;
    }
    if (!consumer || !(supportedDirection(*producer, *consumer) ||
                        (generations && structure.cube && *producer == 2 && *consumer == 3)))
        return std::nullopt;
    spec.producerLane = *producer;
    spec.consumerLane = *consumer;
    return channel;
}

InsertSyncLifecyclePlan discover(
    const InsertSyncLifecycleStructure& structure, Budget& budget, unsigned& attempted, bool& invalid,
    func::FuncOp function, const std::set<std::string>& excluded, std::vector<LifecycleDiagnostic>& diagnostics,
    bool generations)
{
    InsertSyncLifecyclePlan plan;
    plan.cube = structure.cube;
    plan.lifetimeScope = structure.lifetimeScope;
    plan.requirements.local = structure.requirements;
    std::vector<Slice> slices;
    for (unsigned p = 0; p < structure.phases.size(); ++p) {
        const auto* phase = structure.phases[p];
        plan.phaseIds[phase->elementOp] = p;
        plan.phases.push_back(phase->elementOp);
        for (const auto* write : phase->defVec) {
            auto slice = exactSlice(write);
            if (
                !slice || (slice->space != AddressSpace::VEC && slice->space != AddressSpace::MAT &&
                           slice->space != AddressSpace::LEFT && slice->space != AddressSpace::RIGHT &&
                           !(generations && slice->space == AddressSpace::ACC))) {
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
        diagnostics.push_back({"function", "candidate-limit", "more than 32 exact physical slices", false});
        return plan;
    }
    std::vector<std::optional<Channel>> singles;
    for (const auto& slice : slices) {
        auto single = makeChannel(structure, ArrayRef<Slice>(&slice, 1), generations);
        if (
            !single) {
            diagnostics.push_back(
                {channelIdentity(ArrayRef<Slice>(&slice, 1)), "physical-recovery",
                 "partial/unknown overlap, incomplete participant set, or unsupported pipe direction", false});
        }
        singles.push_back(std::move(single));
    }
    std::vector<bool> selected(slices.size(), false);
    auto qualify = [&](Channel& channel) {
        if (
            excluded.count(channel.identity)) {
            diagnostics.push_back(
                {channel.identity, "retry-exclusion", "candidate omitted; its dependencies are restored", false});
            return false;
        }
        ++attempted;
        std::string reason;
        bool accepted = qualifyInsertSyncLifecycleBoundaries(structure, channel, function, budget, reason);
        invalid |= channel.logical.certificate.status == LifecycleCertificate::Status::InvalidInput;
        if (generations) {
            reason += "; read-associations=" + std::to_string(channel.generations.reads.size()) +
                      "; readerless-returns=" + std::to_string(channel.generations.readerlessReturns);
            if (channel.generations.unprovedNode != kInvalid)
                reason += "; unproved-node=" + std::to_string(channel.generations.unprovedNode);
        }
        diagnostics.push_back({channel.identity, generations ? "buffer-generations" : "recognition-and-guards",
                               reason, accepted});
        return accepted;
    };
    // Co-consumed slots can share a handoff only if every actual consumer
    // needs every member, with the same producer/consumer pipeline domain.
    // Generation checking must additionally prove compatible episodes.
    if (generations) {
        for (unsigned a = 0; a < slices.size(); ++a) {
            if (selected[a] || !singles[a]) continue;
            SmallVector<Slice, 4> members{slices[a]};
            SmallVector<unsigned, 4> indices{a};
            const auto& first = singles[a]->logical.spec;
            for (unsigned b = a + 1; b < slices.size() && members.size() < 4; ++b) {
                if (selected[b] || !singles[b]) continue;
                const auto& other = singles[b]->logical.spec;
                bool same = first.producerLane == other.producerLane && first.consumerLane == other.consumerLane;
                for (unsigned p = 0; p < first.phases.size(); ++p)
                    same &= bool(first.phases[p].reads) == bool(other.phases[p].reads) &&
                            !first.phases[p].updates && !other.phases[p].updates;
                if (same) { members.push_back(slices[b]); indices.push_back(b); }
            }
            if (members.size() < 2) continue;
            auto bundle = makeChannel(structure, members, true);
            if (bundle && qualify(*bundle)) {
                for (unsigned i : indices) selected[i] = true;
                plan.channels.push_back(std::move(*bundle));
            }
        }
    }
    // LEFT/RIGHT are bundled only when every consumer reads both members. A
    // consumer of just one side vetoes bundling; grouping never postpones an
    // independent consumer merely to save an event stream.
    if (
        structure.cube && !generations) {
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

Value materializeGuard(OpBuilder& builder, Location loc, const LifecyclePlacement& placement)
{
    Value disjunction;
    for (const auto& term : placement.guard.alternatives) {
        if (
            term.empty()) {
            return {}; // unconditional true
        }
        Value conjunction;
        for (const auto& literal : term) {
            const auto& test = placement.tests[literal.test];
            Value predicate;
            if (
                test.kind == LifecycleGuardTest::Kind::ValueEquals) {
                Value constant;
                if (
                    test.expression.getType().isIndex()) {
                    constant = builder.create<arith::ConstantIndexOp>(loc, test.value);
                } else {
                    constant = builder.create<arith::ConstantOp>(
                        loc, test.expression.getType(), builder.getIntegerAttr(test.expression.getType(), test.value));
                }
                predicate = builder.create<arith::CmpIOp>(
                    loc, literal.truth ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne, test.expression,
                    constant);
            } else {
                auto loop = cast<scf::ForOp>(test.loop);
                if (
                    test.kind == LifecycleGuardTest::Kind::First) {
                    predicate = builder.create<arith::CmpIOp>(
                        loc, literal.truth ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne,
                        loop.getInductionVar(), loop.getLowerBound());
                } else if (test.kind == LifecycleGuardTest::Kind::Last) {
                    Value remaining =
                        builder.create<arith::SubIOp>(loc, loop.getUpperBound(), loop.getInductionVar());
                    predicate = builder.create<arith::CmpIOp>(
                        loc, literal.truth ? arith::CmpIPredicate::sle : arith::CmpIPredicate::sgt, remaining,
                        loop.getStep());
                } else {
                    predicate = builder.create<arith::CmpIOp>(
                        loc, literal.truth ? arith::CmpIPredicate::sge : arith::CmpIPredicate::slt,
                        loop.getLowerBound(), loop.getUpperBound());
                }
            }
            conjunction = conjunction ? builder.create<arith::AndIOp>(loc, conjunction, predicate).getResult() :
                                        predicate;
        }
        disjunction =
            disjunction ? builder.create<arith::OrIOp>(loc, disjunction, conjunction).getResult() : conjunction;
    }
    return disjunction;
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
    Block& scope = plan.lifetimeScope->getRegion(0).front();
    // Save the real tail BEFORE adding guarded cleanup. A new scf.if must not
    // stop a backwards scan and move cleanup behind the original terminal ALL.
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
    OpBuilder entry(function.getContext());
    entry.setInsertionPointToStart(&scope);
    for (unsigned c = 0; c < plan.channels.size(); ++c) {
        const auto& spec = plan.channels[c].logical.spec;
        make(entry, true, spec.consumerLane, spec.producerLane, allocation.free[c], function.getLoc());
    }
    auto emit = [&](OpBuilder& builder, unsigned c, const LifecyclePlacement& place) {
        if (
            place.guard.alternatives.empty()) {
            return;
        }
        Location loc = place.anchor ? place.anchor->getLoc() : function.getLoc();
        OpBuilder::InsertionGuard restore(builder);
        Value condition = materializeGuard(builder, loc, place);
        if (
            condition) {
            auto branch = builder.create<scf::IfOp>(loc, condition, false);
            builder.setInsertionPoint(branch.thenBlock()->getTerminator());
        }
        const auto& spec = plan.channels[c].logical.spec;
        unsigned producer = spec.producerLane, consumer = spec.consumerLane;
        switch (place.role) {
            case LifecyclePlacement::Role::BypassReady:
                make(builder, false, producer, consumer, allocation.ready[c], loc);
                make(builder, true, consumer, producer, allocation.free[c], loc);
                break;
            case LifecyclePlacement::Role::AcquireFree:
                make(builder, false, consumer, producer, allocation.free[c], loc);
                break;
            case LifecyclePlacement::Role::AcquireReady:
                make(builder, false, producer, consumer, allocation.ready[c], loc);
                break;
            case LifecyclePlacement::Role::PublishBefore:
            case LifecyclePlacement::Role::PublishReady:
                make(builder, true, producer, consumer, allocation.ready[c], loc);
                break;
            case LifecyclePlacement::Role::ReleaseBefore:
            case LifecyclePlacement::Role::PublishFree:
                make(builder, true, consumer, producer, allocation.free[c], loc);
                break;
        }
    };
    // Establish a deterministic original-operation traversal BEFORE adding new
    // branches. Bypasses precede acquisition at the same overwrite point.
    SmallVector<Operation*> originalOps;
    SmallVector<Block*> originalBlocks;
    function.getBody().walk([&](Operation* op) { originalOps.push_back(op); });
    originalBlocks.push_back(&function.getBody().front());
    for (Operation* op : originalOps) {
        for (Region& region : op->getRegions()) {
            for (Block& block : region) {
                originalBlocks.push_back(&block);
            }
        }
    }
    for (Operation* op : originalOps) {
        OpBuilder before(op), after(function.getContext());
        after.setInsertionPointAfter(op);
        if (
            op->hasTrait<OpTrait::IsTerminator>() && op->getBlock() == &scope && drainPoint) {
            before.setInsertionPoint(drainPoint);
        }
        for (unsigned c = 0; c < plan.channels.size(); ++c) {
            for (const auto& place : plan.channels[c].placements) {
                if (
                    place.anchor == op) {
                    emit(place.after ? after : before, c, place);
                }
            }
        }
    }
    for (Block* block : originalBlocks) {
        OpBuilder atEnd(function.getContext());
        if (
            block == &scope && drainPoint) {
            atEnd.setInsertionPoint(drainPoint);
        } else if (!block->empty() && block->back().hasTrait<OpTrait::IsTerminator>()) {
            atEnd.setInsertionPoint(&block->back());
        } else {
            atEnd.setInsertionPointToEnd(block);
        }
        for (unsigned c = 0; c < plan.channels.size(); ++c) {
            for (const auto& place : plan.channels[c].placements) {
                if (
                    place.blockEnd == block) {
                    emit(atEnd, c, place);
                }
            }
        }
    }
    // Readerless returns at physical-scope exit must precede this final Free
    // consumption. Keep any original terminal ALL as the final boundary.
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
                            !LocalStorageRequirements::mayOverlap(access, channel.members[member])) {
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
                if (touch.reads && touch.writes) {
                    if (!channel.bufferGenerations || touch.reads != touch.writes ||
                        concrete.program.phaseLane[node.phase] != spec.producerLane)
                        return false;
                    touch.updates = touch.reads;
                    touch.reads = 0;
                }
                if (
                    (touch.writes && concrete.program.phaseLane[node.phase] != spec.producerLane) ||
                    (touch.reads && concrete.program.phaseLane[node.phase] != spec.consumerLane)) {
                    return false;
                }
                if (
                    touch.writes) {
                    nodes[n].action = touch.updates ? LifecycleAction::Update : LifecycleAction::Write;
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
                    // Recover prime/release and acquire/drain by the actual
                    // per-key state machine, not by tags or "no later region".
                    nodes[n].action =
                        node.kind == Node::Kind::Signal ? LifecycleAction::FreeSignal : LifecycleAction::FreeWait;
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

// The mixed view interprets actual logical endpoints, not the singleton member
// certificates. Protocol publications include preceding effects on their source
// lane, including GM effects of a TSTORE whose selected member is a local read.
void InsertSyncLifecyclePlan::refreshCompletionSupply(const SyncIRs& ir, const SyncOperations& syncs) const
{
    completionSupply = LifecycleCompletionSupply{};
    importedResidualHandoffs = 0;
    if (completionStructure.nodes.empty()) {
        completionReason = "no read-only lifecycle completion structure";
        return;
    }
    std::vector<LogicalLifecycle> logical;
    for (const auto& channel : channels) {
        logical.push_back(channel.logical);
    }
    Budget budget;
    const uint64_t startingWork = budget.left;
    auto selected = analyzeLifecycleCompletion(completionStructure, logical, {}, budget);
    completionWork += startingWork - budget.left;
    if (selected.status == LifecycleCompletionSupply::Status::InvalidInput) {
        completionInternalError = true;
        completionReason = selected.reason;
        return;
    }
    if (selected.status != LifecycleCompletionSupply::Status::Complete) {
        completionReason = "protocol-prefix query unavailable: " + selected.reason;
        return; // Existing exact-member certificates still remain available.
    }
    completionSupply = std::move(selected);
    completionReason = "selected-protocol completion";

    std::vector<ResidualCompletionHandoff> direct;
    for (const auto& group : syncs) {
        if (
            group.size() != 2) {
            continue;
        }
        const auto *set = group[0].get(), *wait = group[1].get();
        if (!set || !wait || set->uselessSync || wait->uselessSync ||
            set->GetType() != SyncOperation::TYPE::SET_EVENT || wait->GetType() != SyncOperation::TYPE::WAIT_EVENT ||
            set->GetSyncIndex() != wait->GetSyncIndex() || set->GetForEndIndex() || wait->GetForEndIndex() ||
            set->isCompensation || wait->isCompensation || set->eventIdNum != 1 || wait->eventIdNum != 1 ||
            set->slotSSAExpr || wait->slotSSAExpr || set->GetSyncIRIndex() >= ir.size() ||
            wait->GetSyncIRIndex() >= ir.size()) {
            continue;
        }
        auto* from = dyn_cast<CompoundInstanceElement>(ir[set->GetSyncIRIndex()].get());
        auto* to = dyn_cast<CompoundInstanceElement>(ir[wait->GetSyncIRIndex()].get());
        if (!from || !to || !from->elementOp || !to->elementOp || from->elementOp == to->elementOp ||
            from->elementOp->getBlock() != to->elementOp->getBlock() ||
            !from->elementOp->isBeforeInBlock(to->elementOp)) {
            continue;
        }
        auto source = phaseIds.find(from->elementOp), target = phaseIds.find(to->elementOp);
        if (
            source == phaseIds.end() || target == phaseIds.end() || set->GetActualSrcPipe() != from->kPipeValue ||
            set->GetActualDstPipe() != to->kPipeValue || wait->GetActualSrcPipe() != from->kPipeValue ||
            wait->GetActualDstPipe() != to->kPipeValue || from->kPipeValue == to->kPipeValue) {
            continue;
        }
        auto setAt = std::find(from->pipeAfter.begin(), from->pipeAfter.end(), set);
        auto waitAt = std::find(to->pipeBefore.begin(), to->pipeBefore.end(), wait);
        if (
            setAt == from->pipeAfter.end() || waitAt == to->pipeBefore.end() ||
            std::count(from->pipeAfter.begin(), from->pipeAfter.end(), set) != 1 ||
            std::count(to->pipeBefore.begin(), to->pipeBefore.end(), wait) != 1) {
            continue;
        }
        direct.push_back(
            {source->second, target->second, static_cast<unsigned>(std::distance(from->pipeAfter.begin(), setAt)),
             static_cast<unsigned>(std::distance(to->pipeBefore.begin(), waitAt))});
    }
    if (direct.empty()) {
        return;
    }
    Budget jointBudget;
    const uint64_t jointStart = jointBudget.left;
    auto mixed = analyzeLifecycleCompletion(completionStructure, logical, direct, jointBudget);
    completionWork += jointStart - jointBudget.left;
    if (mixed.status == LifecycleCompletionSupply::Status::InvalidInput) {
        completionInternalError = true;
        completionReason = mixed.reason;
        return;
    }
    if (mixed.status == LifecycleCompletionSupply::Status::Complete) {
        completionSupply = std::move(mixed);
        importedResidualHandoffs = direct.size();
        completionReason = "selected protocols plus qualified same-block residual handoffs";
    } else {
        // A repeated direct stream may need an acknowledgement outside this
        // deliberately restricted import. Do not use an unproved token world.
        // Retain the separately proved selected-protocol view instead.
        completionReason = "selected-protocol completion; residual overlay unproved: " + mixed.reason;
    }
}

void InsertSyncLifecyclePlan::removeSuppliedDependencies(
    CompoundInstanceElement* source, CompoundInstanceElement* target, DepBaseMemInfoPairVec& pairs) const
{
    if (!source || !target) {
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
                if (requirements.local) {
                    auto required = requirements.local->required(source->elementOp, pair.second,
                                                                  target->elementOp, pair.first);
                    if (required && !*required) return true;
                }
                auto a = exactSlice(pair.first), b = exactSlice(pair.second);
                if (a && b && *a == *b) {
                    for (const auto& channel : channels) {
                        for (unsigned m = 0; m < channel.members.size(); ++m) {
                            if (!(channel.members[m] == *a)) {
                                continue;
                            }
                            auto supply =
                                channel.logical.certificate.supplies(channel.logical.spec, s->second, t->second, m);
                            if (supply != LifecycleCertificate::Supply::None) {
                                requirements.retain({SyncRequirement::Kind::Lifecycle, source->elementOp,
                                                     target->elementOp, pair.second->baseBuffer, pair.first->baseBuffer,
                                                     false, unsigned(&channel - channels.data())});
                                ++suppliedPairs;
                                return true;
                            }
                        }
                    }
                }
                // A physical completion certificate is stronger than exact member
                // ownership. For example, Store(g)->Free->Compute(g+1)->Ready->Store(g+1)
                // also orders the store's GM WAW, although the channel owns only its UB
                // input. Restrict the new use to same-pipe memory repair: completion
                // never becomes a new cross-pipe GM visibility or ACC exception.
                if (pair.first && pair.second && source->kPipeValue == target->kPipeValue &&
                    completionSupply.proves(s->second, t->second)) {
                    ++suppliedPairs;
                    ++completionSuppliedPairs;
                    // DepBetween records (target access, source access). Retain original
                    // identities for checking the ACTUAL emitted plan after allocation.
                    requirements.retain({SyncRequirement::Kind::FullCompletion,
                                         source->elementOp, target->elementOp,
                                         pair.second->baseBuffer, pair.first->baseBuffer});
                    return true;
                }
                return false;
            }),
        pairs.end());
}

namespace {
// Replace one COMPLETE readiness stream with already-established physical
// completion. Reclamation remains a separate obligation and keeps its stream.
// This is a checked alternative realization, not deletion of arbitrary actions
// inside an opaque protocol or a claim that two streams have equal recurrence.
void reuseReadinessSupply(func::FuncOp function, const SyncIRs& ir, InsertSyncLifecyclePlan& plan,
                         const LifecycleAllocation& allocation, InsertSyncLifecycleResult& result)
{
    Budget budget;
    auto snapshot = analyzeInsertSyncStorageFrontiers(function, ir, false, budget, true);
    if (snapshot.status != StorageFrontierSnapshot::Status::Complete ||
        !completionAtExits(snapshot.program, snapshot.supply).proved || !plan.requirements.local) return;
    const auto& input = *plan.requirements.local;
    auto identitiesMatch = [&](const StorageFrontierSnapshot& view) {
        if (view.program.phaseLane.size() != input.phases.size() ||
            view.anchors.size() != view.program.nodes.size()) return false;
        for (unsigned n = 0; n < view.program.nodes.size(); ++n) {
            const auto& node = view.program.nodes[n];
            if (node.kind == Node::Kind::Issue &&
                (node.phase >= input.phases.size() || view.anchors[n] != input.phases[node.phase])) return false;
        }
        return true;
    };
    if (!identitiesMatch(snapshot)) return;
    auto accepted = snapshot.program;
    auto supply = snapshot.supply;
    SmallVector<Operation*> removed;
    unsigned streams = 0, sets = 0, waits = 0;
    std::set<unsigned> suppliedLanes;
    SmallVector<unsigned> suppliedChannels;
    for (unsigned c = 0; c < plan.channels.size() && budget.left; ++c) {
        const auto& channel = plan.channels[c];
        const auto& spec = channel.logical.spec;
        SmallVector<Operation*> actions;
        function.walk([&](Operation* op) {
            if (!plan.protocolActions.contains(op)) return;
            PIPE source, target;
            unsigned id;
            if (auto set = dyn_cast<SetFlagOp>(op)) {
                source = set.getSrcPipe().getPipe(); target = set.getDstPipe().getPipe();
                id = unsigned(set.getEventId().getEvent());
            } else if (auto wait = dyn_cast<WaitFlagOp>(op)) {
                source = wait.getSrcPipe().getPipe(); target = wait.getDstPipe().getPipe();
                id = unsigned(wait.getEventId().getEvent());
            } else return;
            if (source == physicalLane(spec.producerLane, plan.cube) &&
                target == physicalLane(spec.consumerLane, plan.cube) && id == allocation.ready[c]) actions.push_back(op);
        });
        if (actions.empty()) continue;
        auto trial = accepted;
        for (unsigned n = 0; n < snapshot.anchors.size(); ++n)
            if (llvm::is_contained(actions, snapshot.anchors[n])) trial.nodes[n].kind = Node::Kind::Pass;
        auto facts = completion(trial, Bits(trial.nodes.size()), budget);
        if (!completionAtExits(trial, facts).proved) continue;
        // Source-prefix publication can serve more than its motivating member.
        // Recheck EVERY memory obligation sourced on this physical lane, from
        // shared input requirements AND independent emitted effect extraction.
        // Full phase completion is deliberately stronger than a recipe's own
        // generation claim; unknown correspondence refuses this replacement.
        std::vector<Requirement> needs;
        for (const auto& q : snapshot.requirements)
            if (trial.phaseLane[q.source] == spec.producerLane) needs.push_back(q);
        for (const auto& q : input.obligations)
            if (trial.phaseLane[q.source] == spec.producerLane)
                needs.push_back({q.source, q.target, Requirement::Kind::Conservative});
        if (!covers(trial, facts, needs).proved) continue;
        bool same = true;
        for (unsigned n = 0; n < trial.nodes.size() && same; ++n) {
            if (trial.nodes[n].kind != Node::Kind::Issue || !supply.before[n]) continue;
            same = facts.before[n] && supply.before[n]->known == facts.before[n]->known;
        }
        if (!same) continue;
        accepted = std::move(trial);
        supply = std::move(facts);
        for (Operation* op : actions) {
            sets += isa<SetFlagOp>(op);
            waits += isa<WaitFlagOp>(op);
            removed.push_back(op);
        }
        ++streams;
        suppliedLanes.insert(spec.producerLane);
        suppliedChannels.push_back(c);
    }
    if (!removed.empty()) {
        // Detach transactionally, then translate actual emitted effects and
        // reconstruct its keys/guards independently of the trial graph.
        SmallVector<std::pair<Operation*, Operation*>> detached;
        for (Operation* op : removed) {
            detached.push_back({op, op->getNextNode()});
            op->remove();
        }
        MemoryDependentAnalyzer memory;
        auto mode = function->getAttrOfType<StringAttr>("pto.gm_alias");
        memory.setGMContract(function, mode && mode.getValue() == "assume-disjoint-arguments" ?
            InsertSyncGMAliasMode::DisjointArguments : InsertSyncGMAliasMode::MayAlias);
        SyncIRs rebuilt;
        Buffer2MemInfoMap buffers;
        PTOIRTranslator translator(rebuilt, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        translator.enableGenerationFlow();
        translator.Build();
        auto concrete = analyzeInsertSyncStorageFrontiers(function, rebuilt, false, budget, true);
        bool proved = concrete.status == StorageFrontierSnapshot::Status::Complete && identitiesMatch(concrete) &&
                      completionAtExits(concrete.program, concrete.supply).proved;
        std::vector<Requirement> needs;
        if (proved) {
            for (const auto& q : concrete.requirements)
                if (suppliedLanes.count(concrete.program.phaseLane[q.source])) needs.push_back(q);
            for (const auto& q : input.obligations)
                if (suppliedLanes.count(concrete.program.phaseLane[q.source]))
                    needs.push_back({q.source, q.target, Requirement::Kind::Conservative});
            proved = covers(concrete.program, concrete.supply, needs).proved;
        }
        // Deleting event nodes must preserve the supported guarded payload
        // occurrences, and every completion prefix at each such occurrence.
        SmallVector<unsigned> oldIssues, newIssues;
        for (unsigned n = 0; n < snapshot.program.nodes.size(); ++n)
            if (snapshot.program.nodes[n].kind == Node::Kind::Issue) oldIssues.push_back(n);
        for (unsigned n = 0; n < concrete.program.nodes.size(); ++n)
            if (concrete.program.nodes[n].kind == Node::Kind::Issue) newIssues.push_back(n);
        proved &= oldIssues.size() == newIssues.size() && snapshot.guardDomains.size() == concrete.guardDomains.size();
        for (unsigned g = 0; proved && g < snapshot.guardDomains.size(); ++g) {
            const auto& a = snapshot.guardDomains[g]; const auto& b = concrete.guardDomains[g];
            proved = a.expression == b.expression && a.invocationScope == b.invocationScope &&
                     a.tripShapeOf == b.tripShapeOf && a.possibleValues == b.possibleValues;
        }
        for (unsigned i = 0; proved && i < oldIssues.size(); ++i) {
            unsigned a = oldIssues[i], b = newIssues[i];
            const auto& oldFact = snapshot.supply.before[a];
            const auto& newFact = concrete.supply.before[b];
            proved = snapshot.anchors[a] == concrete.anchors[b] &&
                     !(snapshot.guards[a] < concrete.guards[b]) && !(concrete.guards[b] < snapshot.guards[a]) &&
                     bool(oldFact) == bool(newFact) && (!oldFact || oldFact->known == newFact->known);
        }
        proved &= succeeded(verify(function));
        if (!proved) {
            for (const auto& item : llvm::reverse(detached))
                item.second->getBlock()->getOperations().insert(Block::iterator(item.second), item.first);
            streams = sets = waits = 0;
            result.diagnostics.push_back({"mixed-plan", "shared-readiness",
                "actual emitted reconstruction unproved; complete original streams retained", false});
        } else {
            for (Operation* op : removed) { plan.protocolActions.erase(op); op->destroy(); }
            for (unsigned c : suppliedChannels)
                result.diagnostics.push_back({plan.channels[c].identity, "shared-readiness",
                    "complete readiness supplied by remaining handoffs; fresh effects, guarded payload prefixes, "
                    "source-lane requirements, exit completion and actual event participation rechecked", true});
        }
    }
    auto i64 = IntegerType::get(function.getContext(), 64);
    function->setAttr("pto.insert_sync.generation_ready_streams_supplied", IntegerAttr::get(i64, streams));
    function->setAttr("pto.insert_sync.generation_ready_sets_removed", IntegerAttr::get(i64, sets));
    function->setAttr("pto.insert_sync.generation_ready_waits_removed", IntegerAttr::get(i64, waits));
}

// Recolor residual keys using the COMPLETE logical plan before reserving
// dedicated lifecycle keys. Sharing never changes an endpoint or adds a wait.
unsigned compactResidualAllocation(func::FuncOp function, SyncOperations& syncs,
                                   const InsertSyncLifecyclePlan& plan)
{
    std::map<std::pair<unsigned, unsigned>, std::set<int>> domains;
    for (const auto& group : syncs) for (const auto& sync : group)
        if (!sync->uselessSync && (sync->isSyncSetType() || sync->isSyncWaitType()))
            for (int id : sync->eventIds)
                domains[{unsigned(sync->GetActualSrcPipe()), unsigned(sync->GetActualDstPipe())}].insert(id);
    bool possible = false;
    for (const auto& [domain, keys] : domains) possible |= keys.size() > 1;
    if (!possible) return 0;
    SyncIRs residualIR;
    Buffer2MemInfoMap buffers;
    MemoryDependentAnalyzer memory;
    PTOIRTranslator translator(residualIR, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
    translator.enableGenerationFlow(); translator.Build();
    Budget budget;
    auto structure = buildInsertSyncLifecycleStructure(function, residualIR, budget);
    if (structure.status != StorageFrontierSnapshot::Status::Complete) return 0;
    llvm::DenseMap<Operation*, unsigned> phases;
    for (unsigned p = 0; p < structure.phases.size(); ++p) phases[structure.phases[p]->elementOp] = p;
    std::vector<LogicalLifecycle> channels;
    for (const auto& old : plan.channels) {
        auto channel = old;
        channel.placements.clear();
        channel.logical.spec.phases.assign(structure.phases.size(), {});
        for (unsigned p = 0; p < old.logical.spec.phases.size(); ++p) {
            auto current = phases.find(plan.phases[p]);
            if (current == phases.end()) return 0;
            channel.logical.spec.phases[current->second] = old.logical.spec.phases[p];
        }
        std::string reason;
        if (!qualifyInsertSyncLifecycleBoundaries(structure, channel, function, budget, reason)) return 0;
        channels.push_back(std::move(channel.logical));
    }
    auto projected = projectLifecycleCompletion(structure.program, channels, {}, budget, true);
    if (projected.status != LifecycleCompletionProjection::Status::Complete) return 0;
    Bits eligible(projected.program.keys.size());
    for (unsigned k = projected.protocolKeys; k < projected.program.keys.size(); ++k) eligible.set(k);
    auto sharing = shareEventKeys(projected.program, eligible, budget);
    if (!sharing.merged) return 0;
    std::vector<unsigned> physicalIds(structure.program.keys.size(), kInvalid);
    llvm::DenseMap<Operation*, unsigned> eventKeys;
    for (unsigned n = 0; n < structure.program.nodes.size(); ++n) {
        const auto& node = structure.program.nodes[n];
        if (node.kind != Node::Kind::Signal && node.kind != Node::Kind::Wait) continue;
        Operation* op = structure.anchors[n];
        if (auto set = dyn_cast_or_null<SetFlagOp>(op)) physicalIds[node.key] = unsigned(set.getEventId().getEvent());
        else if (auto wait = dyn_cast_or_null<WaitFlagOp>(op)) physicalIds[node.key] = unsigned(wait.getEventId().getEvent());
        else return 0;
        eventKeys[op] = node.key;
    }
    using PhysicalKey = std::tuple<unsigned, unsigned, int>;
    std::map<PhysicalKey, int> replacement;
    for (unsigned k = 0; k < physicalIds.size(); ++k) {
        unsigned representative = sharing.representative[projected.protocolKeys + k] - projected.protocolKeys;
        if (representative >= physicalIds.size() || physicalIds[representative] == kInvalid) return 0;
        const auto& key = structure.program.keys[k];
        replacement[{unsigned(physicalLane(key.source, plan.cube)), unsigned(physicalLane(key.target, plan.cube)),
                     int(physicalIds[k])}] = physicalIds[representative];
    }
    for (const auto& [op, key] : eventKeys) {
        unsigned representative = sharing.representative[projected.protocolKeys + key] - projected.protocolKeys;
        auto id = EventAttr::get(function.getContext(), static_cast<EVENT>(physicalIds[representative]));
        if (auto set = dyn_cast<SetFlagOp>(op)) set.setEventIdAttr(id);
        else cast<WaitFlagOp>(op).setEventIdAttr(id);
    }
    for (auto& group : syncs) for (auto& sync : group)
        if (!sync->uselessSync && (sync->isSyncSetType() || sync->isSyncWaitType()))
            for (int& id : sync->eventIds) {
                auto where = replacement.find({unsigned(sync->GetActualSrcPipe()), unsigned(sync->GetActualDstPipe()), id});
                if (where != replacement.end()) id = where->second;
            }
    return sharing.merged;
}

struct RetryRequest {
    std::string identity;
};
InsertSyncLifecycleResult attemptLifecycleSynthesis(
    func::FuncOp function, const InsertSyncOptions& options, const std::set<std::string>& excluded,
    RetryRequest& retryRequest)
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
    translator.enableGenerationFlow(options.bufferGenerations);
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
    auto structure = buildInsertSyncLifecycleStructure(cloned, ir, budget, options.bufferGenerations);
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
    auto plan = discover(
        structure, budget, result.attempted, invalidLifecycle, cloned, excluded, result.diagnostics, options.bufferGenerations);
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
    for (const auto& channel : plan.channels) {
        result.guardedActions += channel.guardedActions;
        result.consumerRegions += channel.consumerRegions;
    }
    auto retry = [&](const std::string& stage, const std::string& reason, unsigned source = kInvalid,
                     unsigned target = kInvalid, unsigned failedIdentity = kInvalid) {
        result.reason = reason;
        if (failedIdentity != kInvalid) {
            auto failed = std::find_if(plan.channels.begin(), plan.channels.end(), [&](const auto& channel) {
                return channel.logical.identity == failedIdentity;
            });
            if (
                failed == plan.channels.end()) {
                result.status = InsertSyncLifecycleResult::Status::InternalError;
                result.reason = "resource assignment reported an unknown lifecycle candidate";
                return;
            }
            retryRequest.identity = failed->identity;
            result.diagnostics.push_back({retryRequest.identity, stage, reason, false});
            return;
        }
        std::vector<LogicalLifecycle> selected;
        for (const auto& channel : plan.channels) {
            selected.push_back(channel.logical);
        }
        auto identity = chooseLifecycleRetry(selected, source, target);
        if (
            !identity && source != kInvalid) {
            identity = chooseLifecycleRetry(selected);
        }
        if (
            identity) {
            retryRequest.identity = plan.channels[*identity].identity;
            result.diagnostics.push_back({retryRequest.identity, stage, reason, false});
        }
    };

    // The logical completion view is refreshed by each ordinary repair stage.
    // It is based on immutable physical/control facts, not the barriers that
    // stage is about to create. Candidate retry builds this again from scratch.
    plan.completionStructure = structure.program;
    // The shared logical plan exists BEFORE dependency insertion. Legacy
    // traversal retains and repairs every dependency not explicitly supplied.
    InsertSyncAnalysis analysis(ir, memory, syncs, cloned, SyncAnalysisMode::NORMALSYNC);
    analysis.setLifecycleSupply(&plan);
    analysis.setRequirements(&plan.requirements);
    if (options.bufferGenerations) analysis.setStorageFlow(&structure.storageFlow);
    analysis.Run(true, options.deferSamePipe, options.mmadChains);
    if (plan.completionInternalError) {
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "invalid mixed logical completion view: " + plan.completionReason;
        return result;
    }
    result.diagnostics.push_back(
        {"mixed-plan", "completion-supply",
         std::to_string(plan.completionSuppliedPairs) + " same-pipe access pairs supplied; " +
             std::to_string(plan.importedResidualHandoffs) + " residual handoffs; " + plan.completionReason,
         true});
    MoveSyncState move(ir, syncs);
    move.Run();
    RemoveRedundantSync redundant(ir, syncs, SyncAnalysisMode::NORMALSYNC);
    redundant.Run();
    SyncEventIdAllocation allocator(ir, syncs);
    allocator.Allocate();

    bool residualEmitted = false;
    if (options.bufferGenerations) {
        SyncCodegen codegen(ir, cloned, SyncAnalysisMode::NORMALSYNC, &plan.requirements);
        codegen.Run(); residualEmitted = true;
        unsigned shared = compactResidualAllocation(cloned, syncs, plan);
        cloned->setAttr("pto.insert_sync.generation_residual_keys_shared",
                       IntegerAttr::get(IntegerType::get(cloned.getContext(), 64), shared));
    }

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
                    retry("residual-allocation", "residual allocation required serialization");
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
                retry("residual-allocation", "residual event realization unresolved");
                return result;
            }
            for (int id : sync->eventIds) {
                if (
                    id < 0 || id >= 8) {
                    result.status = InsertSyncLifecycleResult::Status::InternalError;
                    result.reason = "residual key out of range";
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
        allocation.status == LifecycleAllocation::Status::InvalidInput) {
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "invalid logical channel allocation input";
        return result;
    }
    if (
        allocation.status != LifecycleAllocation::Status::Complete) {
        retry(
            "resource-assignment", allocation.reason, allocation.failedSource,
            allocation.failedTarget, allocation.failedIdentity);
        return result;
    }
    if (!residualEmitted) {
        SyncCodegen codegen(ir, cloned, SyncAnalysisMode::NORMALSYNC, &plan.requirements);
        codegen.Run();
    }
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
        retry("residual-emission", "residual output contains a broad body cut");
        return result;
    }
    materialize(cloned, plan, allocation);
    cloned.walk([&](Operation* op) {
        if (isa<SetFlagOp, WaitFlagOp>(op) && !plan.requirements.owns(op)) plan.protocolActions.insert(op);
    });
    if (options.frontierRefinement || options.frontierPlacement || options.bufferGenerations) {
        SmallVector<Operation*> residualBarriers;
        cloned.walk([&](BarrierOp barrier) {
            if (plan.requirements.owns(barrier.getOperation()) && barrier.getPipe().getPipe() != PIPE::PIPE_ALL)
                residualBarriers.push_back(barrier.getOperation());
        });
        auto refinement = refineInsertSyncStorageFrontiers(
            cloned, ir, residualBarriers, options.mmadChains, options.frontierPlacement, {}, true, &plan.requirements);
        result.diagnostics.push_back({"residual-plan", "frontier-refinement", refinement.reason, !refinement.internalError});
        cloned->setAttr("pto.insert_sync.generation_storage_barriers_removed",
                       IntegerAttr::get(IntegerType::get(cloned.getContext(), 64), refinement.removed));
        cloned->setAttr("pto.insert_sync.generation_storage_barriers_guarded",
                       IntegerAttr::get(IntegerType::get(cloned.getContext(), 64), refinement.guarded));
        if (refinement.internalError) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "invalid residual frontier refinement";
            return result;
        }
    }
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
    checked.enableGenerationFlow(options.bufferGenerations);
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
        if (!plan.protocolActions.contains(op)) {
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
        if (
            whole.status == CompletionResult::Status::LimitExceeded) {
            result.reason = "combined event analysis budget exhausted; not event scarcity";
            return result;
        }
        unsigned source = kInvalid, target = kInvalid;
        if (
            whole.unprovedEventNode < concrete.program.nodes.size()) {
            unsigned key = concrete.program.nodes[whole.unprovedEventNode].key;
            if (
                key < concrete.program.keys.size()) {
                source = concrete.program.keys[key].source;
                target = concrete.program.keys[key].target;
            }
        }
        retry(
            "combined-event-proof",
            "combined residual event transfer unproved at node " + std::to_string(whole.unprovedEventNode), source,
            target);
        return result;
    }
    // Revalidate EVERY new full-completion exemption against actual emitted
    // actions after legacy motion, cleanup and allocation. The proof cannot
    // silently rely on a direct handoff later removed/widened by those stages.
    llvm::DenseMap<Operation*, unsigned> concretePhaseIds;
    for (unsigned p = 0; p < concrete.phases.size(); ++p) {
        concretePhaseIds[concrete.phases[p]->elementOp] = p;
    }
    // Every exact-member omission uses the same payload/SSA identities as direct
    // repairs. Recovered channel actions were independently reconstructed above;
    // now verify that the retained access still names that channel's member.
    for (const auto& witness : plan.requirements.all()) {
        if (witness.kind != SyncRequirement::Kind::Lifecycle) continue;
        if (witness.channel >= plan.channels.size() || !concretePhaseIds.count(witness.source) ||
            !concretePhaseIds.count(witness.target)) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "lost lifecycle requirement ownership";
            return result;
        }
        const auto& channel = plan.channels[witness.channel];
        auto memberAt = [&](Operation* operation, Value access) -> std::optional<Slice> {
            auto phase = concrete.phases[concretePhaseIds.lookup(operation)];
            for (auto info : phase->useVec) if (info->baseBuffer == access) return exactSlice(info);
            for (auto info : phase->defVec) if (info->baseBuffer == access) return exactSlice(info);
            return std::nullopt;
        };
        auto source = memberAt(witness.source, witness.sourceAccess);
        auto target = memberAt(witness.target, witness.targetAccess);
        if (!source || !target || !(*source == *target) || !llvm::is_contained(channel.members, *source)) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "changed physical member in lifecycle requirement";
            return result;
        }
    }
    std::vector<Requirement> retainedCompletionRequirements;
    for (const auto& witness : plan.requirements.all()) {
        if (witness.kind != SyncRequirement::Kind::FullCompletion) continue;
        auto source = concretePhaseIds.find(witness.source), target = concretePhaseIds.find(witness.target);
        if (
            source == concretePhaseIds.end() || target == concretePhaseIds.end()) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "lost original phase in mixed-completion witness reconstruction";
            return result;
        }
        retainedCompletionRequirements.push_back({source->second, target->second, Requirement::Kind::Conservative});
    }
    auto supplied = covers(concrete.program, whole, retainedCompletionRequirements);
    if (!supplied.proved) {
        // Legacy motion/cleanup/allocation is not part of this proof. A failure
        // of the conservative final query is an unproved optional realization,
        // not a demonstrated device race. Never commit it: R7 retries from the
        // original input and restores every omitted residual dependency.
        retry(
            "completion-realization", "emitted plan does not establish the required full-completion witness: phase " +
                                          std::to_string(supplied.source) + " -> " + std::to_string(supplied.target));
        return result;
    }
    auto i64 = IntegerType::get(cloned.getContext(), 64);
    cloned->setAttr("pto.insert_sync.lifecycle_completion_pairs", IntegerAttr::get(i64, plan.completionSuppliedPairs));
    cloned->setAttr(
        "pto.insert_sync.lifecycle_completion_witnesses", IntegerAttr::get(i64, plan.requirements.count(SyncRequirement::Kind::FullCompletion)));
    cloned->setAttr("pto.insert_sync.lifecycle_completion_work", IntegerAttr::get(i64, plan.completionWork));
    unsigned diagnosticWitnesses = 0;
    for (const auto& witness : plan.requirements.all()) {
        if (witness.kind != SyncRequirement::Kind::FullCompletion) continue;
        if (diagnosticWitnesses++ == 64) {
            result.diagnostics.push_back(
                {"mixed-plan", "completion-recheck",
                 "individual witness diagnostics capped at 64; all " +
                     std::to_string(plan.requirements.count(SyncRequirement::Kind::FullCompletion)) + " requirements rechecked",
                 true});
            break;
        }
        result.diagnostics.push_back(
            {"phase-" + std::to_string(plan.phaseIds.lookup(witness.source)) + "-to-" +
                 std::to_string(plan.phaseIds.lookup(witness.target)),
             "completion-recheck", "all prior source occurrences complete at every target; emitted plan rechecked",
             true});
    }
    if (!recheckInsertSyncGenerationRequirements(cloned, plan.requirements)) {
        result.status = InsertSyncLifecycleResult::Status::InternalError;
        result.reason = "emitted generation-qualified requirements could not be reconstructed";
        return result;
    }
    cloned->setAttr("pto.insert_sync.generation_mmad_witnesses",
                   IntegerAttr::get(i64, plan.requirements.count(SyncRequirement::Kind::MmadOrder)));
    cloned->setAttr("pto.insert_sync.generation_global_witnesses",
                   IntegerAttr::get(i64, plan.requirements.count(SyncRequirement::Kind::GlobalDisjoint)));
    cloned->setAttr("pto.insert_sync.generation_slot_witnesses",
                   IntegerAttr::get(i64, plan.requirements.count(SyncRequirement::Kind::SlotDisjoint)));
    result.suppliedPairs = plan.suppliedPairs;
    if (options.bufferGenerations) {
        SmallVector<Operation*> candidates;
        cloned.walk([&](BarrierOp barrier) { candidates.push_back(barrier.getOperation()); });
        Budget cleanupBudget;
        auto cleanup = refineInsertSyncCompletion(cloned, checkedIR, candidates, cleanupBudget);
        cloned->setAttr("pto.insert_sync.generation_barriers_removed", IntegerAttr::get(i64, cleanup.removed));
        result.diagnostics.push_back({"mixed-plan", "completion-refinement", cleanup.reason, true});
        if (failed(verify(cloned))) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "completion refinement produced invalid MLIR";
            return result;
        }
        Budget placementBudget;
        auto placement = refineInsertSyncPublications(cloned, checkedIR, plan.requirements, placementBudget);
        cloned->setAttr("pto.insert_sync.generation_publications_advanced", IntegerAttr::get(i64, placement.signalsAdvanced));
        cloned->setAttr("pto.insert_sync.generation_publications_guarded", IntegerAttr::get(i64, placement.guarded));
        cloned->setAttr("pto.insert_sync.generation_publication_edges_removed", IntegerAttr::get(i64, placement.occurrenceProofs));
        cloned->setAttr("pto.insert_sync.generation_publication_work", IntegerAttr::get(i64, placement.work));
        result.diagnostics.push_back({"mixed-plan", "shared-publication", placement.reason, true});
        reuseReadinessSupply(cloned, checkedIR, plan, allocation, result);
        if (!preserved(cloned, original) || failed(verify(cloned))) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "publication refinement changed original payload or produced invalid IR";
            return result;
        }
    }
    for (const auto& channel : plan.channels) {
        result.diagnostics.push_back(
            {channel.identity, "commit", "complete emitted lifecycle and residuals committed", true});
    }
    function.getBody().takeBody(cloned.getBody());
    function->setAttrs(cloned->getAttrs());
    result.status = InsertSyncLifecycleResult::Status::Applied;
    result.reason = "guarded lifecycle plan plus rebuilt legacy residuals; no added scarcity ordering";
    return result;
}

} // namespace

InsertSyncLifecycleResult mlir::pto::tryInsertSyncLifecycleSynthesis(
    func::FuncOp function, const InsertSyncOptions& options)
{
    constexpr unsigned maximumAttempts = 8;
    std::set<std::string> excluded;
    std::vector<LifecycleDiagnostic> diagnostics;
    unsigned candidatesAttempted = 0;
    InsertSyncLifecycleResult result;
    for (unsigned attempt = 0; attempt < maximumAttempts; ++attempt) {
        RetryRequest retry;
        // Each attempt clones function afresh. No omitted dependency, event ID,
        // or mutable SyncIR from the failed attempt can leak into this one.
        result = attemptLifecycleSynthesis(function, options, excluded, retry);
        candidatesAttempted += result.attempted;
        for (auto diagnostic : result.diagnostics) {
            diagnostic.stage = "attempt-" + std::to_string(attempt + 1) + "/" + diagnostic.stage;
            diagnostics.push_back(std::move(diagnostic));
        }
        result.planningAttempts = attempt + 1;
        result.retries = attempt;
        result.attempted = candidatesAttempted;
        result.diagnostics = diagnostics;
        if (
            result.status != InsertSyncLifecycleResult::Status::Applied) {
            // These counters describe committed changes, never rolled-back trials.
            result.selected = result.logicalStreams = result.guardedActions = result.consumerRegions = 0;
            result.suppliedPairs = 0;
        }
        if (
            result.status != InsertSyncLifecycleResult::Status::Unchanged || retry.identity.empty()) {
            return result;
        }
        if (
            !excluded.insert(retry.identity).second) {
            result.status = InsertSyncLifecycleResult::Status::InternalError;
            result.reason = "retry did not remove a distinct optional candidate";
            return result;
        }
    }
    result.reason = "bounded optional-candidate retry exhausted; original input retained (not a scarcity proof)";
    return result;
}
