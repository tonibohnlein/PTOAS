// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/InsertSync/StorageFrontierAnalysis.h"
#include "StorageFrontierAccess.h"
#include "PTO/Transforms/InsertSync/MmadChainAnalysis.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::insert_sync_frontier;

namespace {
using Tails = std::vector<unsigned>;
std::optional<unsigned> laneFor(PIPE pipe, bool cube)
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

// Validate the existing translator's payload mapping, not a second opcode
// semantics table. Explicit synchronization is interpreted separately below.
// This also lets the development pass inspect manually synchronized fixtures
// without sending flag-resource effects through the payload-only coverage gate.
bool mappedPayloadEffects(Operation* op, const CompoundInstanceElement* compound)
{
    auto interface = dyn_cast<MemoryEffectOpInterface>(op);
    if (
        !interface) {
        return false;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    for (const auto& effect : effects) {
        Value value = effect.getValue();
        bool write = isa<MemoryEffects::Write>(effect.getEffect());
        if (
            (!write && !isa<MemoryEffects::Read>(effect.getEffect())) || !value) {
            return false;
        }
        if (
            !isa<TileBufType, TensorViewType, PartitionTensorViewType, PtrType, BaseMemRefType>(value.getType())) {
            // By-value scalar operands have no payload region. An asynchronous
            // producer of that value is still visited and rejected by this adapter
            // if it is not one of the modeled physical phases.
            if (
                write || !isa<IntegerType, IndexType, FloatType>(value.getType())) {
                return false;
            }
            continue;
        }
        const auto& entries = write ? compound->defVec : compound->useVec;
        if (
            !llvm::any_of(entries, [&](const BaseMemInfo* entry) { return entry && entry->baseBuffer == value; })) {
            return false;
        }
    }
    return true;
}

class NativeGraph {
public:
    Program program;
    std::vector<PhaseInfo> phases;
    std::vector<Operation*> anchors;
    struct PhysicalAtom {
        AddressSpace space;
        uint64_t begin, end;
    };
    std::vector<PhysicalAtom> physicalAtoms;
    std::vector<BoundGuard> guards;
    std::string failure;
    unsigned occurrenceProofs = 0;
    bool cube = false;
    bool internalError = false;

    NativeGraph(func::FuncOp f, const SyncIRs& syncIR) : function(f)
    {
        auto module = f->getParentOfType<ModuleOp>();
        auto arch = module ? module->getAttrOfType<StringAttr>("pto.target_arch") : StringAttr();
        auto core = f->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
        if (
            !arch || (arch.getValue() != "a2" && arch.getValue() != "a3") || !core ||
            (core.getKernelKind() != FunctionKernelKind::Cube && core.getKernelKind() != FunctionKernelKind::Vector)) {
            failure = "requires one explicit A2/A3 physical Cube or vector context";
            return;
        }
        cube = core.getKernelKind() == FunctionKernelKind::Cube;
        program.lanes = cube ? 4 : 3;
        for (const auto& e : syncIR) {
            auto* c = dyn_cast<CompoundInstanceElement>(e.get());
            if (
                !c || !c->elementOp) {
                continue;
            }
            if (
                compounds.count(c->elementOp)) {
                failure = "multi-phase operation requires its complete semantic adapter";
                return;
            }
            compounds[c->elementOp] = c;
        }
        add({}, Node{}, nullptr);
        region(f.getBody(), {0}, 0);
        if (
            failure.empty() && !valid(program)) {
            internalError = true;
            failure = "internal error: malformed immutable frontier graph";
        }
    }

    std::vector<Atom> atoms(Budget& budget)
    {
        std::map<AddressSpace, std::set<uint64_t>> boundaries;
        for (const auto& phase : phases) {
            for (const auto& a : phase.accesses) {
                const auto& m = *a.legacy;
                if (
                    m.scope == AddressSpace::GM) {
                    continue;
                }
                for (uint64_t begin : m.baseAddresses) {
                    boundaries[m.scope].insert(begin);
                    boundaries[m.scope].insert(begin + m.allocateSize); // validated at import
                }
            }
        }
        std::vector<Atom> result;
        unsigned totalBoundaries = 0;
        for (const auto& entry : boundaries) {
            totalBoundaries += entry.second.size();
        }
        if (
            totalBoundaries > 512) {
            failure = "physical boundary budget";
            return {};
        }
        for (const auto& [space, points] : boundaries) {
            if (
                points.empty()) {
                continue;
            }
            for (auto i = points.begin(), j = std::next(i); j != points.end(); ++i, ++j) {
                if (
                    !budget.spend(phases.size() * 16 + 1)) {
                    failure = "atom partition budget";
                    return {};
                }
                Atom atom{Bits(phases.size()), Bits(phases.size())};
                for (unsigned p = 0; p < phases.size(); ++p) {
                    for (const auto& a : phases[p].accesses) {
                        const auto& m = *a.legacy;
                        if (
                            m.scope != space) {
                            continue;
                        }
                        for (uint64_t begin : m.baseAddresses) {
                            if (
                                begin < *j && *i < begin + m.allocateSize) {
                                (a.write ? atom.writes : atom.reads).set(p);
                            }
                        }
                    }
                }
                if (
                    !atom.reads.empty() || !atom.writes.empty()) {
                    physicalAtoms.push_back({space, *i, *j});
                    result.push_back(std::move(atom));
                }
                if (
                    result.size() > 256) {
                    failure = "physical atom budget";
                    return {};
                }
            }
        }
        return result;
    }

    // Independent non-atomized overlap check also remains an acceptance gate.
    // Lifecycle requirements are used for planning/explanation, not trusted as
    // the sole proof that no older reader/writer disappeared at a join.
    std::vector<Requirement> allPairs(bool useIntrinsic, std::vector<Bits>& intrinsic, Budget& budget)
    {
        auto mode = function->getAttrOfType<StringAttr>("pto.gm_alias");
        bool disjointArguments = mode && mode.getValue() == "assume-disjoint-arguments";
        std::optional<MmadChainAnalysis> matrix;
        if (
            useIntrinsic) {
            matrix.emplace(function);
        }
        std::vector<Requirement> result;
        intrinsic.assign(phases.size(), Bits(phases.size()));
        for (unsigned s = 0; s < phases.size(); ++s) {
            for (unsigned t = 0; t < phases.size(); ++t) {
                DepBaseMemInfoPairVec dependencies;
                for (const auto& a : phases[s].accesses) {
                    for (const auto& b : phases[t].accesses) {
                        if (
                            !budget.spend(1 + a.legacy->baseAddresses.size() * b.legacy->baseAddresses.size())) {
                            failure = "independent access-pair budget";
                            return {};
                        }
                        if (
                            !a.write && !b.write) {
                            continue;
                        }
                        if (
                            a.legacy->scope != b.legacy->scope) {
                            continue;
                        }
                        if (
                            a.legacy->scope == AddressSpace::GM) {
                            Value ra = a.global.root ? a.global.root : a.legacy->rootBuffer;
                            Value rb = b.global.root ? b.global.root : b.legacy->rootBuffer;
                            if (
                                disjointArguments && disjointInsertSyncGMRoots(function, ra, rb)) {
                                continue;
                            }
                            auto proof = compareFrontierOccurrences(
                                a.global, phases[s].legacy->elementOp, b.global, phases[t].legacy->elementOp);
                            if (
                                proof.disjoint) {
                                ++occurrenceProofs;
                                for (auto guard : proof.guards) {
                                    addGuard(guard);
                                }
                                continue;
                            }
                            if (
                                program.phaseLane[s] != program.phaseLane[t]) {
                                failure =
                                    "unqualified cross-pipe GM publication/ordered effect; original plan retained";
                                return {};
                            }
                        } else if (!localFrontierOverlap(*a.legacy, *b.legacy)) {
                            continue;
                        }
                        dependencies.push_back({a.legacy, b.legacy});
                    }
                }
                if (
                    dependencies.empty()) {
                    continue;
                }
                if (
                    matrix && matrix->discharges(phases[s].legacy, phases[t].legacy, dependencies)) {
                    intrinsic[t].set(s); // Requirement-specific ACC order, NOT lane completion.
                    continue;
                }
                result.push_back({s, t, Requirement::Kind::Conservative, kInvalid});
            }
        }
        return result;
    }

private:
    func::FuncOp function;
    unsigned visited = 0;
    unsigned fragments = 0;
    llvm::DenseMap<Operation*, const CompoundInstanceElement*> compounds;
    std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> eventKeys;

    void addGuard(BoundGuard guard)
    {
        for (auto& existing : guards) {
            if (
                existing.upper == guard.upper) {
                existing.limit = std::min(existing.limit, guard.limit);
                return;
            }
        }
        guards.push_back(guard);
    }

    Tails add(Tails previous, Node node, Operation* anchor)
    {
        if (
            !failure.empty()) {
            return {};
        }
        if (
            program.nodes.size() >= 2048) {
            failure = "structured node budget";
            return {};
        }
        unsigned id = program.nodes.size();
        program.nodes.push_back(std::move(node));
        anchors.push_back(anchor);
        for (unsigned p : previous) {
            program.nodes[p].next.push_back(id);
        }
        return {id};
    }

    bool importAccess(PhaseInfo& phase, const BaseMemInfo* info, bool write)
    {
        if (
            !info || info->scope == AddressSpace::Zero || info->aliasesUnknownRange || phase.accesses.size() >= 16) {
            return false;
        }
        fragments += info->baseAddresses.size();
        if (
            fragments > 2048) {
            return false;
        }
        if (
            info->scope != AddressSpace::GM) {
            if (
                !info->hasKnownPhysicalAddresses || !info->allocateSize || info->baseAddresses.empty() ||
                info->baseAddresses.size() > 64) {
                return false;
            }
            for (uint64_t address : info->baseAddresses) {
                if (
                    address > std::numeric_limits<uint64_t>::max() - info->allocateSize) {
                    return false;
                }
            }
        }
        AccessInfo access{info, write, {}};
        if (
            info->scope == AddressSpace::GM) {
            access.global = recoverFrontierGlobalSlice(info->baseBuffer, phase.legacy->elementOp, function);
        }
        phase.accesses.push_back(std::move(access));
        return true;
    }

    Tails region(Region& r, Tails tails, unsigned depth)
    {
        if (
            !failure.empty()) {
            return {};
        }
        if (
            depth > 24 || !llvm::hasSingleElement(r)) {
            failure = "unsupported region or nesting budget";
            return {};
        }
        tails = add(std::move(tails), {}, nullptr);
        if (
            !failure.empty()) {
            return {};
        }
        unsigned regionId = program.regions.size();
        program.regions.push_back({tails.front(), tails.front()});
        for (Operation& op : r.front()) {
            if (
                ++visited > 8192) {
                failure = "IR visitation budget";
                return {};
            }
            if (
                !failure.empty()) {
                return {};
            }
            if (
                isa<scf::YieldOp>(op)) {
                continue;
            }
            if (
                isa<func::ReturnOp>(op)) {
                tails = add(std::move(tails), {Node::Kind::Exit}, &op);
                continue;
            }
            if (
                auto choice = dyn_cast<scf::IfOp>(op)) {
                IntegerAttr constant;
                if (
                    matchPattern(choice.getCondition(), m_Constant(&constant))) {
                    if (
                        !constant.getValue().isZero()) {
                        tails = region(choice.getThenRegion(), std::move(tails), depth + 1);
                    } else if (!choice.getElseRegion().empty()) {
                        tails = region(choice.getElseRegion(), std::move(tails), depth + 1);
                    }
                } else {
                    Tails a = region(choice.getThenRegion(), tails, depth + 1);
                    Tails b = choice.getElseRegion().empty() ? tails : region(choice.getElseRegion(), tails, depth + 1);
                    a.insert(a.end(), b.begin(), b.end());
                    tails = add(std::move(a), {}, &op);
                }
                continue;
            }
            if (
                auto loop = dyn_cast<scf::ForOp>(op)) {
                // One symbolic backedge plus bypass, no trip-count unrolling. Dynamic
                // predicates are conservative alternatives; no assumed parity history.
                Tails header = add(std::move(tails), {}, &op);
                if (
                    !failure.empty()) {
                    return {};
                }
                Tails body = region(loop.getRegion(), header, depth + 1);
                if (
                    !failure.empty()) {
                    return {};
                }
                for (unsigned end : body) {
                    program.nodes[end].next.push_back(header.front());
                }
                tails = header;
                continue;
            }
            if (
                op.getNumRegions() || getSyncMacroModel(&op)) {
                failure = "unmodeled physical region or macro; no partial optimization";
                return {};
            }
            auto flag = [&](PIPE sp, PIPE tp, unsigned id, Node::Kind kind) {
                auto source = laneFor(sp, cube), target = laneFor(tp, cube);
                if (
                    !source || !target || *source == *target || id >= 8) {
                    failure = "unsupported event domain";
                    return;
                }
                auto key = std::make_tuple(*source, *target, id);
                auto [where, inserted] = eventKeys.emplace(key, eventKeys.size());
                if (
                    inserted) {
                    if (
                        program.keys.size() >= 64) {
                        failure = "event-key analysis budget";
                        return;
                    }
                    program.keys.push_back({*source, *target});
                }
                tails =
                    add(std::move(tails),
                        {kind, kind == Node::Kind::Signal ? *source : *target, kInvalid, where->second, {}}, &op);
            };
            if (
                auto set = dyn_cast<SetFlagOp>(op)) {
                flag(
                    set.getSrcPipe().getPipe(), set.getDstPipe().getPipe(),
                    static_cast<unsigned>(set.getEventId().getEvent()), Node::Kind::Signal);
                continue;
            }
            if (
                auto wait = dyn_cast<WaitFlagOp>(op)) {
                flag(
                    wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe(),
                    static_cast<unsigned>(wait.getEventId().getEvent()), Node::Kind::Wait);
                continue;
            }
            if (
                auto barrier = dyn_cast<BarrierOp>(op)) {
                if (
                    barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
                    tails = add(std::move(tails), {Node::Kind::All}, &op);
                } else if (auto lane = laneFor(barrier.getPipe().getPipe(), cube)) {
                    tails = add(std::move(tails), {Node::Kind::Barrier, *lane}, &op);
                } else {
                    failure = "unsupported barrier resource";
                }
                continue;
            }
            if (
                auto physical = dyn_cast<OpPipeInterface>(op)) {
                auto lane = laneFor(physical.getPipe(), cube);
                auto found = compounds.find(&op);
                if (
                    !lane || found == compounds.end() ||
                    !isa<TLoadOp, TStoreOp, TAbsOp, TAddOp, TExtractOp, TMovOp, TMatmulOp, TMatmulAccOp>(op)) {
                    failure = "physical phase has no qualified frontier adapter";
                    return {};
                }
                if (
                    !mappedPayloadEffects(&op, found->second)) {
                    failure = "translated payload effect is incomplete; original plan retained";
                    return {};
                }
                PhaseInfo phase;
                phase.legacy = found->second;
                for (const auto* read : phase.legacy->useVec) {
                    if (
                        !importAccess(phase, read, false)) {
                        failure = "unproved read footprint";
                        return {};
                    }
                }
                for (const auto* write : phase.legacy->defVec) {
                    if (
                        !importAccess(phase, write, true)) {
                        failure = "unproved write footprint";
                        return {};
                    }
                }
                if (
                    phase.accesses.empty() || phases.size() >= 256) {
                    failure = "empty physical summary or phase budget";
                    return {};
                }
                unsigned id = phases.size();
                phases.push_back(std::move(phase));
                program.phaseLane.push_back(*lane);
                tails = add(std::move(tails), {Node::Kind::Issue, *lane, id}, &op);
            } else if (!isa<AllocTileOp>(op) && !isMemoryEffectFree(&op)) {
                failure = "unmodeled non-payload effect; original translation retained";
                return {};
            }
        }
        tails = add(std::move(tails), {}, nullptr);
        if (
            !failure.empty()) {
            return {};
        }
        program.regions[regionId].exit = tails.front();
        return tails;
    }
};

} // namespace

StorageFrontierRefinementResult mlir::pto::refineInsertSyncStorageFrontiers(
    func::FuncOp function, const SyncIRs& syncIR, ArrayRef<Operation*> candidates, bool useMmadChains)
{
    StorageFrontierRefinementResult result;
    if (
        candidates.empty()) {
        result.reason = "no owned named-barrier candidates";
        return result;
    }
    NativeGraph graph(function, syncIR);
    if (
        !graph.failure.empty()) {
        result.internalError = graph.internalError;
        result.reason = result.internalError ? graph.failure : "unchanged: " + graph.failure;
        return result;
    }
    Budget budget;
    const uint64_t initialBudget = budget.left;
    auto atoms = graph.atoms(budget);
    result.work = initialBudget - budget.left;
    if (
        !graph.failure.empty()) {
        result.reason = "unchanged: " + graph.failure;
        return result;
    }
    auto life = lifecycles(graph.program, atoms, budget);
    result.work = initialBudget - budget.left;
    result.atoms = atoms.size();
    if (
        !life.complete) {
        result.reason = "unchanged: lifecycle transfer budget";
        return result;
    }
    std::vector<Bits> intrinsic;
    auto dense = graph.allPairs(useMmadChains, intrinsic, budget);
    result.work = initialBudget - budget.left;
    result.occurrenceProofs = graph.occurrenceProofs;
    if (
        !graph.failure.empty()) {
        result.reason = "unchanged: " + graph.failure;
        return result;
    }
    auto requirements = dense;
    for (const auto& requirement : life.requirements) {
        if (
            !intrinsic[requirement.target].test(requirement.source)) {
            requirements.push_back(requirement);
        }
    }
    result.requirements = requirements.size();
    if (
        isInsertSyncDebugEnabled(InsertSyncDebugLevel::Trace)) {
        llvm::errs() << "[StorageFrontier] " << graph.program.regions.size() << " regions, " << life.boundaries.size()
                     << " region/atom transfers\n";
        for (unsigned atom = 0; atom < graph.physicalAtoms.size(); ++atom) {
            const auto& a = graph.physicalAtoms[atom];
            llvm::errs() << "  atom=" << atom << " space=" << static_cast<unsigned>(a.space) << " bytes=[" << a.begin
                         << "," << a.end << ")\n";
        }
        for (unsigned phase = 0; phase < graph.phases.size(); ++phase) {
            llvm::errs() << "  phase=" << phase << " lane=" << graph.program.phaseLane[phase]
                         << " op=" << graph.phases[phase].legacy->elementOp->getName() << "\n";
        }
        for (const auto& q : life.requirements) {
            llvm::errs() << "  atom=" << q.storageAtom << " source=" << q.source << " target=" << q.target
                         << " role=" << static_cast<unsigned>(q.kind) << "\n";
        }
    }
    llvm::SmallPtrSet<Operation*, 32> owned;
    owned.insert(candidates.begin(), candidates.end());
    Bits selectable(graph.program.nodes.size());
    for (unsigned n = 0; n < graph.program.nodes.size(); ++n) {
        const auto& node = graph.program.nodes[n];
        if (
            node.kind != Node::Kind::Barrier || !owned.contains(graph.anchors[n])) {
            continue;
        }
        // MMAD's target-specific decision remains its own opt-in optimization.
        // This pass never deletes M barriers or ABI/PIPE_ALL/fixed input barriers.
        if (
            graph.cube && node.lane == 2) {
            continue;
        }
        selectable.set(n);
    }
    auto plan = refine(graph.program, requirements, selectable, budget);
    result.work = initialBudget - budget.left;
    if (
        plan.status == Refinement::Status::InvalidInput) {
        result.internalError = true;
        result.reason = "internal error: inconsistent frontier refinement input";
        return result;
    }
    if (
        plan.status != Refinement::Status::Complete) {
        result.reason = plan.status == Refinement::Status::AnalysisLimit ?
                            "unchanged: bounded frontier proof exhausted (not event scarcity)" :
                            "unchanged: baseline completion/participation not established by optional model";
        if (
            plan.failure.source != kInvalid) {
            result.reason += "; outstanding phase " + std::to_string(plan.failure.source) + " before " +
                             std::to_string(plan.failure.target);
        }
        return result;
    }
    unsigned total = 0;
    for (unsigned n = 0; n < graph.program.nodes.size(); ++n) {
        total += plan.omitted.test(n);
    }
    if (
        !total) {
        result.reason = "no redundant owned barriers proved";
        return result;
    }

    // All overflow predicates refer to immutable function arguments. Their common
    // conjunction selects either the fully proved fast synchronization plan or
    // the original one for the ENTIRE execution, not per-iteration assumptions.
    IRMapping mapping;
    // Preserve the module's target/data-layout context while verifying the
    // detached candidate. This adapter admits no symbol-referencing helpers.
    OwningOpRef<ModuleOp> stagingModule(ModuleOp::create(function.getLoc()));
    (*stagingModule)->setAttrs(function->getParentOfType<ModuleOp>()->getAttrs());
    auto staged = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->getBody()->push_back(staged.getOperation());
    Value slow;
    if (
        !graph.guards.empty()) {
        // The bound is an immutable function argument. Compute the common scalar
        // guard once; do not insert repeated comparisons in a hot inner loop.
        OpBuilder guardBuilder(staged.getContext());
        guardBuilder.setInsertionPointToStart(&staged.getBody().front());
        for (const auto& guard : graph.guards) {
            Value upper = mapping.lookup(guard.upper);
            Value limit = guardBuilder.create<arith::ConstantIndexOp>(staged.getLoc(), guard.limit);
            Value tooLarge =
                guardBuilder.create<arith::CmpIOp>(staged.getLoc(), arith::CmpIPredicate::sgt, upper, limit);
            slow = slow ? guardBuilder.create<arith::OrIOp>(staged.getLoc(), slow, tooLarge).getResult() : tooLarge;
        }
    }
    for (unsigned n = 0; n < graph.program.nodes.size(); ++n) {
        if (
            !plan.omitted.test(n)) {
            continue;
        }
        Operation* barrier = mapping.lookup(graph.anchors[n]);
        if (
            graph.guards.empty()) {
            barrier->erase();
            ++result.removed;
            continue;
        }
        OpBuilder builder(barrier);
        auto branch = builder.create<scf::IfOp>(barrier->getLoc(), slow, false);
        branch->setAttr("pto.insert_sync.frontier_overflow_guard", builder.getUnitAttr());
        barrier->moveBefore(branch.thenBlock()->getTerminator());
        ++result.guarded;
    }
    if (
        failed(verify(staged.getOperation()))) {
        staged.erase();
        result.removed = 0;
        result.guarded = 0;
        result.internalError = true;
        result.reason = "internal error: staged frontier transformation failed MLIR verification";
        return result;
    }
    function.getBody().takeBody(staged.getBody());
    staged.erase();
    result.reason = "storage requirements and lane completion proved; event positions/IDs unchanged";
    if (
        result.guarded) {
        result.reason += "; original barriers retained on invariant overflow-risk path";
    }
    return result;
}

namespace {
// Test/development adapter: only explicitly marked barriers are owned. It does
// NOT change production InsertSync's explicit-synchronization bypass behavior.
struct PTORefineStorageFrontiersPass : PassWrapper<PTORefineStorageFrontiersPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORefineStorageFrontiersPass)
    StringRef getArgument() const final { return "pto-refine-storage-frontiers"; }
    StringRef getDescription() const final { return "Refine marked synchronization using storage and lane frontiers"; }
    void runOnOperation() override
    {
        auto function = getOperation();
        if (
            function.isDeclaration()) {
            return;
        }
        auto contract = resolveInsertSyncGMAlias(function, "");
        if (
            failed(contract)) {
            signalPassFailure();
            return;
        }
        MemoryDependentAnalyzer memory;
        memory.setGMContract(function, *contract);
        SyncIRs syncIR;
        Buffer2MemInfoMap buffers;
        PTOIRTranslator translator(syncIR, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        translator.Build();
        SmallVector<Operation*> candidates;
        function.walk([&](BarrierOp barrier) {
            if (
                barrier->hasAttr("pto.insert_sync.frontier_candidate")) {
                candidates.push_back(barrier.getOperation());
            }
        });
        auto result = refineInsertSyncStorageFrontiers(function, syncIR, candidates);
        function.emitRemark("InsertSync frontier refinement: ")
            << result.removed << " removed, " << result.guarded << " guarded; " << result.reason;
        if (
            result.internalError) {
            signalPassFailure();
        }
    }
};
PassRegistration<PTORefineStorageFrontiersPass> registerStorageFrontiers;
} // namespace
