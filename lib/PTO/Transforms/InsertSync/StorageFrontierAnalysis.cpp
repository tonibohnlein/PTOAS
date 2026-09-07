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
#include "PTO/Transforms/InsertSync/StorageFrontierControl.h"
#include "PTO/Transforms/InsertSync/StorageFrontierQueries.h"
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
#include "llvm/Support/raw_ostream.h"
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
    for (
        const auto& effect : effects) {
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
    GuardControl control;
    std::vector<std::vector<unsigned>> rawCopies;
    std::vector<unsigned> rawOrigins;
    std::vector<RequirementWitness> requirementWitnesses;
    std::vector<GuardEnvironment> guardEnvironments;

    std::vector<StorageFrontierGuardRecord> describeGuardDomains() const
    {
        std::vector<StorageFrontierGuardRecord> result(control.variables.size());
        for (
            unsigned i = 0; i < result.size(); ++i) {
            result[i].identity = i;
            result[i].possibleValues = control.variables[i].finiteValues;
            unsigned scope = control.variables[i].scope;
            if (
                scope != kInvalid && scope < loops.size()) {
                result[i].invocationScope = loops[scope];
            }
        }
        for (
            auto [value, id] : predicateIds) {
            result[id].expression = value;
        }
        for (
            auto [loop, id] : loopShapes) {
            result[id].tripShapeOf = loop;
        }
        return result;
    }

    bool partition(Budget& budget)
    {
        control.forgetAtNode.resize(program.nodes.size());
        control.updatesAtNode.resize(program.nodes.size());
        for (
            auto [node, loop] : resets) {
            for (
                unsigned v = 0; v < control.variables.size(); ++v) {
                unsigned scope = control.variables[v].scope;
                if (
                    scope != kInvalid && scope < loops.size() &&
                    (loops[scope] == loop || loop->isAncestor(loops[scope]))) {
                    auto tracked = residueVariables.find(v);
                    if (
                        scope < loops.size() && loops[scope] == loop && tracked != residueVariables.end() &&
                        resetKind.count(node) && resetKind[node] != 0) {
                        auto carrier = cast<scf::ForOp>(loop);
                        bool first = resetKind[node] == 1;
                        int64_t value = first ? *constant(carrier.getLowerBound()) : *constant(carrier.getStep());
                        control.updatesAtNode[node].push_back(
                            {v, first ? GuardUpdate::Kind::Assign : GuardUpdate::Kind::AddModulo,
                             first ? int64_t(residue(value, tracked->second)) : value, tracked->second});
                    } else {
                        control.forgetAtNode[node].push_back(v);
                    }
                }
            }
        }
        auto partitioned = partitionGuards(program, control, budget);
        if (
            partitioned.status != GuardPartitionResult::Status::Complete) {
            failure = partitioned.status == GuardPartitionResult::Status::AnalysisLimit ?
                          "guard-state partition budget" :
                          "invalid guard partition input";
            internalError = partitioned.status == GuardPartitionResult::Status::InvalidInput;
            return false;
        }
        auto old = anchors;
        anchors.clear();
        for (
            unsigned origin : partitioned.origins) {
            anchors.push_back(old[origin]);
        }
        rawCopies = std::move(partitioned.copies);
        rawOrigins = std::move(partitioned.origins);
        guardEnvironments = std::move(partitioned.environments);
        program = std::move(partitioned.program);
        return true;
    }

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
        for (
            const auto& e : syncIR) {
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
        // The shape of each loop invocation is finite: empty, one, or >=2 trips.
        // This describes all trip counts; it is not a bounded execution sample.
        f.walk([&](scf::ForOp loop) {
            loopIds[loop.getOperation()] = loops.size();
            loops.push_back(loop.getOperation());
        });
        for (
            Operation* operation : loops) {
            auto parent = operation->getParentOfType<scf::ForOp>();
            unsigned scope = parent ? loopIds.lookup(parent.getOperation()) : kInvalid;
            unsigned key = control.variables.size();
            GuardVariable domain{scope, {0, 1, 2}};
            auto loop = cast<scf::ForOp>(operation);
            auto lower = constant(loop.getLowerBound()), upper = constant(loop.getUpperBound()),
                 step = constant(loop.getStep());
            if (
                lower && upper && step && *step > 0 && *step <= std::numeric_limits<int32_t>::max() &&
                !loop->hasAttr("unsignedCmp")) {
                const __int128 difference = __int128(*upper) - *lower;
                domain.finiteValues = {difference <= 0 ? 0 : (difference <= *step ? 1 : 2)};
            }
            control.variables.push_back(std::move(domain));
            loopShapes[operation] = key;
        }
        f.walk([&](Operation* op) {
            Value dividend, divisor;
            if (
                auto rem = dyn_cast<arith::RemUIOp>(op)) {
                dividend = rem.getLhs();
                divisor = rem.getRhs();
            } else if (auto rem = dyn_cast<arith::RemSIOp>(op)) {
                dividend = rem.getLhs();
                divisor = rem.getRhs();
            } else {
                return;
            }
            auto argument = dyn_cast<BlockArgument>(dividend);
            auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
            auto modulus = constant(divisor);
            if (
                loop && dividend == loop.getInductionVar() && supportsBoundaryClasses(loop) &&
                op->getParentOfType<scf::ForOp>() == loop && modulus && *modulus > 0 && *modulus <= 16) {
                residueVariables[variable(op->getResult(0))] = static_cast<unsigned>(*modulus);
            }
        });
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
        for (
            const auto& phase : phases) {
            for (
                const auto& a : phase.accesses) {
                const auto& m = *a.legacy;
                if (
                    m.scope == AddressSpace::GM) {
                    continue;
                }
                for (
                    uint64_t begin : m.baseAddresses) {
                    boundaries[m.scope].insert(begin);
                    boundaries[m.scope].insert(begin + m.allocateSize); // validated at import
                }
            }
        }
        std::vector<Atom> result;
        unsigned totalBoundaries = 0;
        for (
            const auto& entry : boundaries) {
            totalBoundaries += entry.second.size();
        }
        if (
            totalBoundaries > 512) {
            failure = "physical boundary budget";
            return {};
        }
        for (
            const auto& [space, points] : boundaries) {
            if (
                points.empty()) {
                continue;
            }
            for (
                auto i = points.begin(), j = std::next(i); j != points.end(); ++i, ++j) {
                if (
                    !budget.spend(phases.size() * 16 + 1)) {
                    failure = "atom partition budget";
                    return {};
                }
                Atom atom{Bits(phases.size()), Bits(phases.size())};
                for (
                    unsigned p = 0; p < phases.size(); ++p) {
                    for (
                        const auto& a : phases[p].accesses) {
                        const auto& m = *a.legacy;
                        if (
                            m.scope != space) {
                            continue;
                        }
                        for (
                            uint64_t begin : m.baseAddresses) {
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
    std::vector<Requirement> allPairs(
        bool useIntrinsic, std::vector<Bits>& intrinsic, Budget& budget, bool allowArithmeticGuard = true)
    {
        auto mode = function->getAttrOfType<StringAttr>("pto.gm_alias");
        bool disjointArguments = mode && mode.getValue() == "assume-disjoint-arguments";
        std::optional<MmadChainAnalysis> matrix;
        if (
            useIntrinsic) {
            matrix.emplace(function);
        }
        std::vector<Requirement> result;
        requirementWitnesses.clear();
        intrinsic.assign(phases.size(), Bits(phases.size()));
        for (
            unsigned s = 0; s < phases.size(); ++s) {
            for (
                unsigned t = 0; t < phases.size(); ++t) {
                DepBaseMemInfoPairVec dependencies;
                std::vector<RequirementWitness> pairWitnesses;
                for (
                    const auto& a : phases[s].accesses) {
                    for (
                        const auto& b : phases[t].accesses) {
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
                                proof.disjoint && (allowArithmeticGuard || proof.guards.empty())) {
                                ++occurrenceProofs;
                                for (
                                    auto guard : proof.guards) {
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
                        } else {
                            // Shared access-relation API: local ranges are absolute and qualified
                            // by the translator. Unknown symbolic/large-address cases retain the
                            // independent legacy interval query; no missing result proves disjoint.
                            bool definitelySeparate = true;
                            for (
                                uint64_t beginA : a.legacy->baseAddresses) {
                                for (
                                    uint64_t beginB : b.legacy->baseAddresses) {
                                    AccessSlice x, y;
                                    x.space = static_cast<unsigned>(a.legacy->scope);
                                    y.space = static_cast<unsigned>(b.legacy->scope);
                                    x.read = !a.write;
                                    x.write = a.write;
                                    y.read = !b.write;
                                    y.write = b.write;
                                    x.known = y.known = true;
                                    x.extent = a.legacy->allocateSize;
                                    y.extent = b.legacy->allocateSize;
                                    if (
                                        beginA > uint64_t(std::numeric_limits<int64_t>::max()) ||
                                        beginB > uint64_t(std::numeric_limits<int64_t>::max())) {
                                        definitelySeparate = false;
                                        continue;
                                    }
                                    x.byteStart = {static_cast<int64_t>(beginA), {}, true};
                                    y.byteStart = {static_cast<int64_t>(beginB), {}, true};
                                    auto relation = compareAccesses(x, y, {}, false, budget);
                                    definitelySeparate &= relation.kind == AccessRelationResult::Kind::Disjoint;
                                }
                            }
                            if (
                                definitelySeparate || !localFrontierOverlap(*a.legacy, *b.legacy)) {
                                continue;
                            }
                        }
                        dependencies.push_back({a.legacy, b.legacy});
                        RequirementWitness witness;
                        witness.obligation = {
                            s, t,
                            a.write ? (b.write ? Requirement::Kind::WriteOrder : Requirement::Kind::Availability) :
                                      Requirement::Kind::Reclamation,
                            kInvalid};
                        witness.sourceAccess = a.identity;
                        witness.targetAccess = b.identity;
                        witness.relation.kind = OccurrenceRelation::Kind::OrderedUnknown;
                        pairWitnesses.push_back(std::move(witness));
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
                requirementWitnesses.insert(requirementWitnesses.end(), pairWitnesses.begin(), pairWitnesses.end());
                result.push_back({s, t, Requirement::Kind::Conservative, kInvalid});
            }
        }
        return result;
    }

private:
    func::FuncOp function;
    unsigned visited = 0;
    unsigned fragments = 0;
    unsigned nextAccessIdentity = 0;
    llvm::DenseMap<Operation*, const CompoundInstanceElement*> compounds;
    std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> eventKeys;
    llvm::DenseMap<Operation*, unsigned> phaseIds, loopIds, loopShapes;
    llvm::DenseMap<Value, unsigned> predicateIds;
    std::vector<Operation*> loops;
    std::map<unsigned, Operation*> resets;
    std::map<unsigned, unsigned> resetKind; // 0 unknown; 1 first; 2 successor
    std::map<unsigned, unsigned> residueVariables;
    std::map<Operation*, std::pair<bool, bool>> iterationClass;

    static std::optional<int64_t> constant(Value value)
    {
        IntegerAttr attr;
        if (
            !value || !matchPattern(value, m_Constant(&attr)) || !attr.getValue().isSignedIntN(64)) {
            return std::nullopt;
        }
        return attr.getValue().getSExtValue();
    }
    // These are exact scalar identities. Tags on generated guards are diagnostic
    // only and are never accepted in place of inspecting the actual expression.
    std::optional<bool> classifiedCondition(Value value)
    {
        if (
            auto c = constant(value)) {
            return *c != 0;
        }
        auto cmp = value.getDefiningOp<arith::CmpIOp>();
        if (
            !cmp) {
            return std::nullopt;
        }
        for (
            const auto& [operation, stage] : iterationClass) {
            auto loop = cast<scf::ForOp>(operation);
            bool forward = cmp.getLhs() == loop.getInductionVar() && cmp.getRhs() == loop.getLowerBound();
            bool reverse = cmp.getRhs() == loop.getInductionVar() && cmp.getLhs() == loop.getLowerBound();
            if (
                forward || reverse) {
                if (
                    cmp.getPredicate() == arith::CmpIPredicate::eq) {
                    return stage.first;
                }
                if (
                    cmp.getPredicate() == arith::CmpIPredicate::ne) {
                    return !stage.first;
                }
            }
            auto remaining = cmp.getLhs().getDefiningOp<arith::SubIOp>();
            if (
                remaining && remaining.getLhs() == loop.getUpperBound() &&
                remaining.getRhs() == loop.getInductionVar() && cmp.getRhs() == loop.getStep()) {
                if (
                    cmp.getPredicate() == arith::CmpIPredicate::sle) {
                    return stage.second;
                }
                if (
                    cmp.getPredicate() == arith::CmpIPredicate::sgt) {
                    return !stage.second;
                }
            }
        }
        return std::nullopt;
    }
    unsigned variable(Value value)
    {
        if (
            auto i = predicateIds.find(value); i != predicateIds.end()) {
            return i->second;
        }
        Operation* definition = value.getDefiningOp();
        auto argument = dyn_cast<BlockArgument>(value);
        Operation* owner = definition ? definition : (argument ? argument.getOwner()->getParentOp() : nullptr);
        scf::ForOp scope;
        if (
            argument && owner && isa<scf::ForOp>(owner)) {
            scope = cast<scf::ForOp>(owner);
        } else if (owner) {
            scope = owner->getParentOfType<scf::ForOp>();
        }
        unsigned id = control.variables.size();
        GuardVariable domain;
        domain.scope = scope ? loopIds.lookup(scope.getOperation()) : kInvalid;
        if (
            value.getType().isInteger(1)) {
            domain.finiteValues = {0, 1};
        }
        Value remainderDivisor;
        if (
            auto rem = value.getDefiningOp<arith::RemUIOp>()) {
            remainderDivisor = rem.getRhs();
        }
        if (
            auto rem = value.getDefiningOp<arith::RemSIOp>()) {
            auto argument = dyn_cast<BlockArgument>(rem.getLhs());
            auto loop = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp();
            if (
                loop && rem.getLhs() == loop.getInductionVar() && supportsBoundaryClasses(loop)) {
                remainderDivisor = rem.getRhs();
            }
        }
        if (
            remainderDivisor) {
            auto modulus = constant(remainderDivisor);
            if (
                modulus && *modulus > 0 && *modulus <= 16) {
                for (
                    int64_t i = 0; i < *modulus; ++i) {
                    domain.finiteValues.push_back(i);
                }
            }
        }
        control.variables.push_back(std::move(domain));
        predicateIds[value] = id;
        return id;
    }
    GuardFact condition(Value value, bool truth)
    {
        if (
            auto cmp = value.getDefiningOp<arith::CmpIOp>()) {
            for (
                Operation* op : loops) {
                auto loop = cast<scf::ForOp>(op);
                if (
                    cmp.getLhs() == loop.getLowerBound() && cmp.getRhs() == loop.getUpperBound()) {
                    bool empty = cmp.getPredicate() == arith::CmpIPredicate::sge;
                    bool nonempty = cmp.getPredicate() == arith::CmpIPredicate::slt;
                    if (
                        empty || nonempty) {
                        return {loopShapes.lookup(op), kInvalid, 0, truth == empty};
                    }
                }
            }
            if (
                cmp.getPredicate() == arith::CmpIPredicate::eq || cmp.getPredicate() == arith::CmpIPredicate::ne) {
                bool equal = truth == (cmp.getPredicate() == arith::CmpIPredicate::eq);
                if (
                    auto literal = constant(cmp.getRhs())) {
                    return {variable(cmp.getLhs()), kInvalid, *literal, equal};
                }
                if (
                    auto literal = constant(cmp.getLhs())) {
                    return {variable(cmp.getRhs()), kInvalid, *literal, equal};
                }
            }
        }
        return {variable(value), kInvalid, truth ? 1 : 0, true};
    }
    Tails constrained(Tails tails, GuardFact fact)
    {
        Tails from = tails;
        Tails head = add(std::move(tails), {}, nullptr);
        if (
            head.empty()) {
            return {};
        }
        for (
            unsigned p : from) {
            control.assumptions.push_back({p, head.front(), fact.expression, fact.value, fact.equal});
        }
        return head;
    }
    Tails refresh(Tails tails, Operation* loop, unsigned kind = 0)
    {
        Tails head = add(std::move(tails), {}, nullptr);
        if (
            !head.empty()) {
            resets[head.front()] = loop;
            resetKind[head.front()] = kind;
        }
        return head;
    }
    bool supportsBoundaryClasses(scf::ForOp loop)
    {
        auto lower = constant(loop.getLowerBound()), step = constant(loop.getStep());
        return lower && *lower >= 0 && *lower <= std::numeric_limits<int32_t>::max() && step && *step > 0 &&
               *step <= std::numeric_limits<int32_t>::max() && !loop->hasAttr("unsignedCmp") &&
               loop.getInductionVar().getType().isIndex();
    }
    Tails classifiedLoop(scf::ForOp loop, Tails tails, unsigned depth)
    {
        unsigned shape = loopShapes.lookup(loop.getOperation());
        auto result = structuredLoopTransfer(
            std::move(tails),
            [&](Tails entry, unsigned which) {
                return constrained(std::move(entry), {shape, kInvalid, int64_t(which), true});
            },
            [&](Tails entry, IterationClass which) {
                iterationClass[loop.getOperation()] = {which.first, which.last};
                return region(
                    loop.getRegion(), refresh(std::move(entry), loop.getOperation(), which.first ? 1 : 2), depth + 1);
            },
            [&](Tails entry) { return add(std::move(entry), {}, loop.getOperation()); },
            [&](const Tails& ends, const Tails& header) {
                if (
                    !failure.empty() || header.empty()) {
                    return;
                }
                for (
                    unsigned end : ends) {
                    program.nodes[end].next.push_back(header.front());
                }
            });
        iterationClass.erase(loop.getOperation());
        return result;
    }

    void addGuard(BoundGuard guard)
    {
        for (
            auto& existing : guards) {
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
        for (
            unsigned p : previous) {
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
            for (
                uint64_t address : info->baseAddresses) {
                if (
                    address > std::numeric_limits<uint64_t>::max() - info->allocateSize) {
                    return false;
                }
            }
        }
        AccessInfo access{info, write, {}, nextAccessIdentity++};
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
        RegionScope scope{tails.front(), tails.front()};
        if (
            isa<scf::ForOp>(r.getParentOp())) {
            scope.kind = RegionScope::Kind::Loop;
        } else if (isa<scf::IfOp>(r.getParentOp())) {
            scope.kind = RegionScope::Kind::Choice;
        } else if (isa<func::FuncOp>(r.getParentOp())) {
            scope.kind = RegionScope::Kind::Function;
        }
        program.regions.push_back(scope);
        for (
            Operation& op : r.front()) {
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
                if (
                    auto known = classifiedCondition(choice.getCondition())) {
                    if (
                        *known) {
                        tails = region(choice.getThenRegion(), std::move(tails), depth + 1);
                    } else if (!choice.getElseRegion().empty()) {
                        tails = region(choice.getElseRegion(), std::move(tails), depth + 1);
                    }
                } else {
                    Tails a = constrained(tails, condition(choice.getCondition(), true));
                    Tails b = constrained(tails, condition(choice.getCondition(), false));
                    a = region(choice.getThenRegion(), std::move(a), depth + 1);
                    if (
                        !choice.getElseRegion().empty()) {
                        b = region(choice.getElseRegion(), std::move(b), depth + 1);
                    }
                    a.insert(a.end(), b.begin(), b.end());
                    tails = add(std::move(a), {}, &op);
                }
                continue;
            }
            if (
                auto loop = dyn_cast<scf::ForOp>(op)) {
                if (
                    supportsBoundaryClasses(loop)) {
                    tails = classifiedLoop(loop, std::move(tails), depth);
                    continue;
                }
                // Otherwise retain a conservative backedge and reset loop-local facts.
                Tails header = refresh(std::move(tails), &op);
                if (
                    !failure.empty()) {
                    return {};
                }
                Tails body = region(loop.getRegion(), header, depth + 1);
                if (
                    !failure.empty()) {
                    return {};
                }
                for (
                    unsigned end : body) {
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
                unsigned id = 0;
                if (
                    auto cached = phaseIds.find(&op); cached != phaseIds.end()) {
                    id = cached->second;
                } else {
                    PhaseInfo phase;
                    phase.legacy = found->second;
                    for (
                        const auto* read : phase.legacy->useVec) {
                        if (
                            !importAccess(phase, read, false)) {
                            failure = "unproved read footprint";
                            return {};
                        }
                    }
                    for (
                        const auto* write : phase.legacy->defVec) {
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
                    id = phases.size();
                    phaseIds[&op] = id;
                    phases.push_back(std::move(phase));
                    program.phaseLane.push_back(*lane);
                }
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

StorageFrontierSnapshot mlir::pto::analyzeInsertSyncStorageFrontiers(
    func::FuncOp function, const SyncIRs& syncIR, bool useMmadChains, Budget& budget)
{
    StorageFrontierSnapshot snapshot;
    NativeGraph graph(function, syncIR);
    if (
        graph.failure.empty()) {
        graph.partition(budget);
    }
    if (
        !graph.failure.empty()) {
        snapshot.reason = graph.failure;
        snapshot.status = graph.internalError ? StorageFrontierSnapshot::Status::InternalError :
                                                (budget.left == 0 || graph.failure.find("budget") != std::string::npos ?
                                                     StorageFrontierSnapshot::Status::AnalysisLimit :
                                                     StorageFrontierSnapshot::Status::Unsupported);
        return snapshot;
    }
    snapshot.atoms = graph.atoms(budget);
    for (
        const auto& a : graph.physicalAtoms) {
        snapshot.physicalAtoms.push_back({a.space, a.begin, a.end});
    }
    if (
        !graph.failure.empty()) {
        snapshot.reason = graph.failure;
        snapshot.status = graph.failure.find("budget") != std::string::npos || !budget.left ?
                              StorageFrontierSnapshot::Status::AnalysisLimit :
                              StorageFrontierSnapshot::Status::Unsupported;
        return snapshot;
    }
    snapshot.lifecycle = lifecycles(graph.program, snapshot.atoms, budget);
    snapshot.generations = generationFrontiers(graph.program, snapshot.atoms, budget);
    if (
        !snapshot.lifecycle.complete || snapshot.generations.status != GenerationFrontiers::Status::Complete) {
        snapshot.status = snapshot.generations.status == GenerationFrontiers::Status::InvalidInput ?
                              StorageFrontierSnapshot::Status::InternalError :
                              StorageFrontierSnapshot::Status::AnalysisLimit;
        snapshot.reason = "storage frontier fixed-point unavailable";
        return snapshot;
    }
    std::vector<Bits> intrinsic;
    snapshot.requirements = graph.allPairs(useMmadChains, intrinsic, budget, false);
    if (
        !graph.failure.empty()) {
        snapshot.reason = graph.failure;
        snapshot.status = graph.failure.find("budget") != std::string::npos || !budget.left ?
                              StorageFrontierSnapshot::Status::AnalysisLimit :
                              StorageFrontierSnapshot::Status::Unsupported;
        return snapshot;
    }
    snapshot.witnesses = std::move(graph.requirementWitnesses);
    for (
        const auto& requirement : snapshot.lifecycle.requirements) {
        if (
            !intrinsic[requirement.target].test(requirement.source)) {
            snapshot.requirements.push_back(requirement);
        }
    }
    snapshot.supply = completion(graph.program, Bits(graph.program.nodes.size()), budget);
    if (
        snapshot.supply.status != CompletionResult::Status::Complete) {
        snapshot.status = snapshot.supply.status == CompletionResult::Status::LimitExceeded ?
                              StorageFrontierSnapshot::Status::AnalysisLimit :
                              StorageFrontierSnapshot::Status::InternalError;
        snapshot.reason = "completion fixed-point unavailable";
        return snapshot;
    }
    snapshot.lanes = projectLanes(graph.program);
    snapshot.physicalOwner = function.getOperation();
    snapshot.cubeContext = graph.cube;
    snapshot.guardDomains = graph.describeGuardDomains();
    for (
        unsigned p = 0; p < graph.phases.size(); ++p) {
        for (
            const auto& a : graph.phases[p].accesses) {
            const auto& m = *a.legacy;
            snapshot.accesses.push_back(
                {a.identity, p, m.baseBuffer, m.rootBuffer, m.scope, m.baseAddresses, m.allocateSize, a.write, true});
        }
    }
    snapshot.program = std::move(graph.program);
    snapshot.anchors = std::move(graph.anchors);
    snapshot.guards = std::move(graph.guardEnvironments);
    snapshot.status = StorageFrontierSnapshot::Status::Complete;
    snapshot.reason = "immutable access witnesses, guarded occurrences, lane supply and generation frontiers";
    return snapshot;
}

static StorageFrontierRefinementResult refineStorageBarriers(
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
    if (
        !graph.partition(budget)) {
        result.work = initialBudget - budget.left;
        result.internalError = graph.internalError;
        result.reason = "unchanged: " + graph.failure;
        return result;
    }
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
    for (
        const auto& requirement : life.requirements) {
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
        for (
            unsigned atom = 0; atom < graph.physicalAtoms.size(); ++atom) {
            const auto& a = graph.physicalAtoms[atom];
            llvm::errs() << "  atom=" << atom << " space=" << static_cast<unsigned>(a.space) << " bytes=[" << a.begin
                         << "," << a.end << ")\n";
        }
        for (
            unsigned phase = 0; phase < graph.phases.size(); ++phase) {
            llvm::errs() << "  phase=" << phase << " lane=" << graph.program.phaseLane[phase]
                         << " op=" << graph.phases[phase].legacy->elementOp->getName() << "\n";
        }
        for (
            const auto& q : life.requirements) {
            llvm::errs() << "  atom=" << q.storageAtom << " source=" << q.source << " target=" << q.target
                         << " role=" << static_cast<unsigned>(q.kind) << "\n";
        }
    }
    llvm::SmallPtrSet<Operation*, 32> owned;
    owned.insert(candidates.begin(), candidates.end());
    std::map<Operation*, BarrierGroup> grouped;
    for (
        unsigned n = 0; n < graph.program.nodes.size(); ++n) {
        const auto& node = graph.program.nodes[n];
        if (
            node.kind != Node::Kind::Barrier || !owned.contains(graph.anchors[n]) || (graph.cube && node.lane == 2)) {
            continue;
        }
        grouped[graph.anchors[n]].nodes.push_back(n);
    }
    std::vector<BarrierGroup> groups;
    // Preserve lexical discovery order, not pointer-address tie breaking.
    for (
        Operation* candidate : candidates) {
        if (
            grouped.count(candidate)) {
            groups.push_back(grouped[candidate]);
        }
    }
    auto plan = refineGroups(graph.program, requirements, groups, budget);
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
    llvm::SmallPtrSet<Operation*, 32> omittedOperations;
    for (
        unsigned n = 0; n < graph.program.nodes.size(); ++n) {
        if (
            plan.omitted.test(n) && omittedOperations.insert(graph.anchors[n]).second) {
            ++total;
        }
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
        for (
            const auto& guard : graph.guards) {
            Value upper = mapping.lookup(guard.upper);
            Value limit = guardBuilder.create<arith::ConstantIndexOp>(staged.getLoc(), guard.limit);
            Value tooLarge =
                guardBuilder.create<arith::CmpIOp>(staged.getLoc(), arith::CmpIPredicate::sgt, upper, limit);
            slow = slow ? guardBuilder.create<arith::OrIOp>(staged.getLoc(), slow, tooLarge).getResult() : tooLarge;
        }
    }
    for (
        Operation* original : candidates) {
        if (
            !omittedOperations.contains(original)) {
            continue;
        }
        Operation* barrier = mapping.lookup(original);
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
constexpr llvm::StringLiteral workIdentity = "pto.insert_sync.frontier_work_id";
constexpr llvm::StringLiteral boundaryRole = "pto.insert_sync.frontier_boundary";

std::optional<int64_t> identity(Operation* op);

struct CheckedFrontiers {
    bool proved = false, internal = false, analysisLimit = false;
    std::string reason;
    unsigned generations = 0;
    std::set<int64_t> publicationAnchors, acquisitionAnchors;
    uint64_t work = 0;
};

// Reconstruct the actual candidate, including its first/last/empty guards. No
// saved placement certificate or diagnostic tag is accepted as evidence.
CheckedFrontiers checkPlacement(func::FuncOp function, bool useMmad, Budget& budget)
{
    CheckedFrontiers result;
    auto mode = resolveInsertSyncGMAlias(function, "");
    if (
        failed(mode)) {
        result.internal = true;
        result.reason = "invalid existing GM contract";
        return result;
    }
    MemoryDependentAnalyzer memory;
    memory.setGMContract(function, *mode);
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    auto snapshot = analyzeInsertSyncStorageFrontiers(function, ir, useMmad, budget);
    if (
        snapshot.status != StorageFrontierSnapshot::Status::Complete) {
        result.reason = snapshot.reason;
        result.internal = snapshot.status == StorageFrontierSnapshot::Status::InternalError;
        result.analysisLimit = snapshot.status == StorageFrontierSnapshot::Status::AnalysisLimit;
        return result;
    }
    result.generations = snapshot.generations.generations.size();
    auto remember = [&](unsigned node, std::set<int64_t>& anchors) {
        if (
            node == kInvalid || node >= snapshot.anchors.size()) {
            return;
        }
        if (
            auto id = identity(snapshot.anchors[node])) {
            anchors.insert(*id);
        }
    };
    for (
        const auto& generation : snapshot.generations.generations) {
        remember(generation.producerNode, result.publicationAnchors);
        for (
            unsigned node : generation.finalReaders) {
            remember(node, result.publicationAnchors);
        }
        for (
            unsigned node : generation.firstReaders) {
            remember(node, result.acquisitionAnchors);
        }
        for (
            unsigned node : generation.nextOverwrites) {
            remember(node, result.acquisitionAnchors);
        }
    }
    auto coverage = covers(snapshot.program, snapshot.supply, snapshot.requirements);
    if (
        !coverage.proved) {
        result.reason =
            snapshot.supply.eventsProved ? "outstanding storage requirement" : "unproved event generation/progress";
        return result;
    }
    unsigned explanations = 0;
    for (
        const auto& q : snapshot.requirements) {
        for (
            unsigned n = 0; n < snapshot.program.nodes.size() && explanations < 8; ++n) {
            if (
                snapshot.program.nodes[n].kind != Node::Kind::Issue || snapshot.program.nodes[n].phase != q.target) {
                continue;
            }
            auto witness = completionBefore(snapshot.program, snapshot.supply, q.source, n, budget);
            if (
                witness.status == CompletionWitness::Status::AnalysisLimit) {
                result.analysisLimit = true;
                result.reason = "completion witness budget";
                return result;
            }
            if (
                witness.status != CompletionWitness::Status::Proved) {
                result.internal = true;
                result.reason = "inconsistent completion witness";
                return result;
            }
            ++explanations;
        }
        if (
            explanations >= 8) {
            break;
        }
    }
    result.proved = true;
    result.reason = "all storage requirements and token generations proved";
    return result;
}

Operation* findIdentity(func::FuncOp f, int64_t id)
{
    Operation* found = nullptr;
    f.walk([&](Operation* op) {
        auto a = op->getAttrOfType<IntegerAttr>(workIdentity);
        if (
            a && a.getInt() == id) {
            found = op;
        }
    });
    return found;
}
std::optional<int64_t> identity(Operation* op)
{
    auto attr = op ? op->getAttrOfType<IntegerAttr>(workIdentity) : IntegerAttr();
    if (
        !attr) {
        return std::nullopt;
    }
    return attr.getInt();
}
bool localSync(Operation* op) { return isa<SetFlagOp, WaitFlagOp, BarrierOp, RecordEventOp, WaitEventOp>(op); }
std::optional<int64_t> scalarLiteral(Value v)
{
    IntegerAttr a;
    if (
        !v || !matchPattern(v, m_Constant(&a)) || !a.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return a.getValue().getSExtValue();
}
bool boundaryLoop(scf::ForOp loop)
{
    auto lb = scalarLiteral(loop.getLowerBound()), step = scalarLiteral(loop.getStep());
    // On every executing iteration 0<=iv<ub, so ub-iv is representable. No
    // overflow-prone iv+step test is introduced by the last-use construction.
    return lb && *lb >= 0 && *lb <= std::numeric_limits<int32_t>::max() && step && *step > 0 &&
           *step <= std::numeric_limits<int32_t>::max() && !loop->hasAttr("unsignedCmp") &&
           loop.getInductionVar().getType().isIndex();
}

// Compare original operations, attributes, types, and SSA operand identities.
// Newly introduced untagged operations are synchronization-control only. This
// check is in addition to (not a replacement for) MLIR and memory/event checks.
std::vector<std::string> originalTrace(func::FuncOp f)
{
    std::vector<std::string> trace;
    f.walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (
            localSync(op)) {
            return;
        }
        auto id = identity(op);
        if (
            !id) {
            return;
        }
        std::string text;
        llvm::raw_string_ostream out(text);
        out << *id << ':' << op->getName().getStringRef();
        for (
            NamedAttribute a : op->getAttrs()) {
            if (
                a.getName().getValue() == workIdentity) {
                continue;
            }
            out << "|attr:" << a.getName().getValue() << '=' << a.getValue();
        }
        for (
            Type type : op->getResultTypes()) {
            out << "|type:" << type;
        }
        for (
            Value operand : op->getOperands()) {
            out << "|operand:" << operand.getType() << ':';
            if (
                auto result = dyn_cast<OpResult>(operand)) {
                auto owner = identity(result.getOwner());
                out << "result:" << (owner ? *owner : -1) << ':' << result.getResultNumber();
            } else if (auto argument = dyn_cast<BlockArgument>(operand)) {
                auto owner = identity(argument.getOwner()->getParentOp());
                out << "argument:" << (owner ? *owner : -1) << ':' << argument.getArgNumber();
            } else {
                out << "unclassified";
            }
        }
        out.flush();
        trace.push_back(std::move(text));
    });
    return trace;
}

enum class MotionKind { EarlierSignal, LaterWait, LastUseSignal, FirstUseWait };
struct Motion {
    MotionKind kind;
    int64_t event, anchor;
    // For a boundary move, 'anchor' is an operation directly inside this loop.
    int64_t loop = -1;
    bool deferEmptyToExit = false;
};

bool materializeMotion(func::FuncOp f, const Motion& motion)
{
    Operation *event = findIdentity(f, motion.event), *anchor = findIdentity(f, motion.anchor);
    if (
        !event || !anchor) {
        return false;
    }
    if (
        motion.kind == MotionKind::EarlierSignal) {
        if (
            !isa<SetFlagOp>(event) || event->getBlock() != anchor->getBlock()) {
            return false;
        }
        event->moveAfter(anchor);
        return true;
    }
    if (
        motion.kind == MotionKind::LaterWait) {
        if (
            !isa<WaitFlagOp>(event) || event->getBlock() != anchor->getBlock()) {
            return false;
        }
        event->moveBefore(anchor);
        return true;
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(findIdentity(f, motion.loop));
    if (
        !loop || !boundaryLoop(loop) || anchor->getBlock() != loop.getBody() || event->getBlock() != loop->getBlock()) {
        return false;
    }
    const bool publication = motion.kind == MotionKind::LastUseSignal;
    if (
        publication != isa<SetFlagOp>(event)) {
        return false;
    }
    OpBuilder builder(f.getContext());
    Location loc = event->getLoc();
    if (
        publication) {
        builder.setInsertionPointAfter(anchor);
    } else {
        builder.setInsertionPoint(anchor);
    }
    Value condition;
    if (
        publication) {
        Value remaining = builder.create<arith::SubIOp>(loc, loop.getUpperBound(), loop.getInductionVar());
        condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, remaining, loop.getStep());
    } else {
        condition =
            builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, loop.getInductionVar(), loop.getLowerBound());
    }
    auto inside = builder.create<scf::IfOp>(loc, condition, false);
    inside->setAttr(boundaryRole, builder.getStringAttr(publication ? "last-use" : "first-use"));
    // Preserve one actual action on every original execution, including bypass.
    builder.setInsertionPointAfter(loop);
    Value empty =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, loop.getLowerBound(), loop.getUpperBound());
    if (
        !publication && motion.deferEmptyToExit) {
        Operation* cleanup = loop->getBlock()->getTerminator();
        // Preserve an existing terminal ALL as the final synchronization boundary.
        // The empty-path acquisition follows independent suffix work, not the drain.
        for (
            Operation* next = loop->getNextNode(); next; next = next->getNextNode()) {
            if (
                auto barrier = dyn_cast<BarrierOp>(next)) {
                if (
                    barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
                    cleanup = next;
                }
            }
        }
        builder.setInsertionPoint(cleanup);
    }
    auto bypass = builder.create<scf::IfOp>(loc, empty, false);
    bypass->setAttr(boundaryRole, builder.getStringAttr("zero-trip-token"));
    Operation* copy = event->clone();
    copy->removeAttr(workIdentity);
    bypass.thenBlock()->getOperations().insert(bypass.thenBlock()->getTerminator()->getIterator(), copy);
    event->moveBefore(inside.thenBlock()->getTerminator());
    // There are now two mutually exclusive static sites for ONE dynamic action.
    // No additional event key is allocated and no body is drained.
    event->removeAttr(workIdentity);
    return true;
}

std::vector<Motion> proposeMotions(func::FuncOp f, int64_t eventId, const CheckedFrontiers& frontiers)
{
    std::vector<Motion> result;
    Operation* event = findIdentity(f, eventId);
    if (
        !event || !isa<SetFlagOp, WaitFlagOp>(event)) {
        return result;
    }
    bool signal = isa<SetFlagOp>(event);
    PIPE pipe = signal ? cast<SetFlagOp>(event).getSrcPipe().getPipe() : cast<WaitFlagOp>(event).getDstPipe().getPipe();
    if (
        signal) {
        for (
            Operation* op = event->getPrevNode(); op; op = op->getPrevNode()) {
            if (
                localSync(op) || op->getNumRegions()) {
                break;
            }
            auto physical = dyn_cast<OpPipeInterface>(op);
            auto id = identity(op);
            if (
                op == event->getPrevNode() || !physical || physical.getPipe() != pipe || !id) {
                continue;
            }
            if (
                frontiers.publicationAnchors.count(*id)) {
                result.push_back({MotionKind::EarlierSignal, eventId, *id});
            }
        }
        // Producer and per-lane final-reader frontiers, earliest first. Every
        // candidate still has to satisfy ALL requirements, not just this lifecycle.
        std::reverse(result.begin(), result.end());
    } else {
        bool crossedPhysical = false;
        for (
            Operation* op = event->getNextNode(); op; op = op->getNextNode()) {
            // Another acquisition can be crossed only as a trial: the reconstructed
            // memory and token proof decides whether its ordering is independent.
            // Do not cross publications, barriers, or structured control here.
            if (
                (localSync(op) && !isa<WaitFlagOp>(op)) || op->getNumRegions()) {
                break;
            }
            auto physical = dyn_cast<OpPipeInterface>(op);
            auto id = identity(op);
            if (
                crossedPhysical && physical && physical.getPipe() == pipe && id &&
                frontiers.acquisitionAnchors.count(*id)) {
                result.push_back({MotionKind::LaterWait, eventId, *id});
            }
            crossedPhysical |= static_cast<bool>(physical);
            if (
                op->hasTrait<OpTrait::IsTerminator>()) {
                break;
            }
        }
        std::reverse(result.begin(), result.end());
    }
    // A separate construction crosses a loop boundary; never move one action
    // into a repeated body without its first/last and zero-trip counterparts.
    Operation* neighbor = signal ? event->getPrevNode() : event->getNextNode();
    while (
        neighbor && !neighbor->getNumRegions() && !localSync(neighbor) && !isa<OpPipeInterface>(neighbor) &&
           isMemoryEffectFree(neighbor)) {
        neighbor = signal ? neighbor->getPrevNode() : neighbor->getNextNode();
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(neighbor);
    if (
        !loop || !boundaryLoop(loop)) {
        return result;
    }
    auto loopId = identity(loop.getOperation());
    if (
        !loopId) {
        return result;
    }
    std::vector<Motion> boundaries;
    for (
        Operation& op : *loop.getBody()) {
        auto physical = dyn_cast<OpPipeInterface>(op);
        auto id = identity(&op);
        if (
            !physical || physical.getPipe() != pipe || !id) {
            continue;
        }
        if (
            signal && !frontiers.publicationAnchors.count(*id)) {
            continue;
        }
        if (
            !signal && !frontiers.acquisitionAnchors.count(*id)) {
            continue;
        }
        if (
            signal) {
            boundaries.push_back({MotionKind::LastUseSignal, eventId, *id, *loopId, false});
        } else {
            boundaries.push_back({MotionKind::FirstUseWait, eventId, *id, *loopId, true});
            boundaries.push_back({MotionKind::FirstUseWait, eventId, *id, *loopId, false});
        }
    }
    if (
        !signal) {
        // Prefer the latest first-use anchor, and at that anchor prefer zero-trip
        // token consumption after independent suffix work when the proof allows it.
        std::stable_sort(boundaries.begin(), boundaries.end(), [](const Motion& a, const Motion& b) {
            if (
                a.anchor != b.anchor) {
                return a.anchor > b.anchor;
            }
            return a.deferEmptyToExit > b.deferEmptyToExit;
        });
    }
    result.insert(result.end(), boundaries.begin(), boundaries.end());
    return result;
}
} // namespace

StorageFrontierRefinementResult mlir::pto::refineInsertSyncStorageFrontiers(
    func::FuncOp function, const SyncIRs& syncIR, ArrayRef<Operation*> candidates, bool useMmadChains,
    bool placeFrontiers, ArrayRef<Operation*> ownedEvents)
{
    if (
        !placeFrontiers || ownedEvents.empty()) {
        return refineStorageBarriers(function, syncIR, candidates, useMmadChains);
    }
    StorageFrontierRefinementResult result;
    bool collision = false;
    function.walk([&](Operation* op) { collision |= op->hasAttr(workIdentity); });
    if (
        collision) {
        result.reason = "unchanged: pre-existing private frontier work IDs";
        return result;
    }
    Budget budget;
    budget.left = 64000000;
    const uint64_t initial = budget.left;
    auto abandon = [&]() {
        result.removed = result.guarded = 0;
        result.signalsAdvanced = result.waitsDelayed = result.boundaryHandoffs = 0;
        result.work = initial - budget.left;
        return result;
    };
    auto baseline = checkPlacement(function, useMmadChains, budget);
    if (
        !baseline.proved) {
        result.reason = "unchanged: placement baseline " + baseline.reason;
        result.internalError = baseline.internal;
        return abandon();
    }
    result.generations = baseline.generations;
    IRMapping mapping;
    OwningOpRef<ModuleOp> stage(ModuleOp::create(function.getLoc()));
    (*stage)->setAttrs(function->getParentOfType<ModuleOp>()->getAttrs());
    auto working = cast<func::FuncOp>(function->clone(mapping));
    stage->getBody()->push_back(working.getOperation());
    std::vector<int64_t> eventIds, barrierIds;
    int64_t id = 0;
    llvm::SmallPtrSet<Operation*, 32> events, barriers;
    events.insert(ownedEvents.begin(), ownedEvents.end());
    barriers.insert(candidates.begin(), candidates.end());
    function.walk<WalkOrder::PreOrder>([&](Operation* op) {
        Operation* copy = op == function.getOperation() ? working.getOperation() : mapping.lookup(op);
        copy->setAttr(workIdentity, IntegerAttr::get(IntegerType::get(function.getContext(), 64), id));
        if (
            events.contains(op)) {
            eventIds.push_back(id);
        }
        if (
            barriers.contains(op)) {
            barrierIds.push_back(id);
        }
        ++id;
    });
    const auto trace = originalTrace(working);
    // Rebind generation frontiers to private clone identities. These facts select
    // proposals; each trial is then checked against independently reconstructed
    // all-access requirements and event generations.
    auto frontierFacts = checkPlacement(working, useMmadChains, budget);
    if (
        !frontierFacts.proved) {
        result.internalError = frontierFacts.internal;
        result.reason = "unchanged: cloned frontier analysis " + frontierFacts.reason;
        return abandon();
    }
    for (
        int64_t eventId : eventIds) {
        // Once-only identity per original event. New guarded variants are not
        // recursively respecialized in this rollout.
        auto proposals = proposeMotions(working, eventId, frontierFacts);
        for (
            const Motion& motion : proposals) {
            if (
                result.placementAttempts >= 96 || !budget.left) {
                result.reason = "unchanged: whole placement transaction exhausted its budget";
                return abandon();
            }
            ++result.placementAttempts;
            IRMapping trialMapping;
            auto trial = cast<func::FuncOp>(working->clone(trialMapping));
            OwningOpRef<ModuleOp> trialModule(ModuleOp::create(function.getLoc()));
            (*trialModule)->setAttrs((*stage)->getAttrs());
            trialModule->getBody()->push_back(trial.getOperation());
            if (
                !materializeMotion(trial, motion)) {
                trial.erase();
                continue;
            }
            if (
                originalTrace(trial) != trace || failed(verify(trial.getOperation()))) {
                trial.erase();
                result.internalError = true;
                result.reason = "internal error: frontier placement changed payload or produced invalid IR";
                return abandon();
            }
            auto checked = checkPlacement(trial, useMmadChains, budget);
            if (
                checked.internal) {
                trial.erase();
                result.internalError = true;
                result.reason = checked.reason;
                return abandon();
            }
            if (
                checked.analysisLimit || !budget.left) {
                trial.erase();
                result.reason = "unchanged: placement analysis limit: " + checked.reason;
                return abandon();
            }
            if (
                !checked.proved) {
                trial.erase();
                continue;
            }
            working.erase();
            trial->remove();
            stage->getBody()->push_back(trial.getOperation());
            working = trial;
            frontierFacts = std::move(checked);
            if (
                motion.kind == MotionKind::EarlierSignal) {
                ++result.signalsAdvanced;
            } else if (motion.kind == MotionKind::LaterWait) {
                ++result.waitsDelayed;
            } else {
                ++result.boundaryHandoffs;
                if (
                    motion.kind == MotionKind::LastUseSignal) {
                    ++result.signalsAdvanced;
                } else {
                    ++result.waitsDelayed;
                }
            }
            break;
        }
    }
    // Re-import after placement; no stale SyncIR/Value pointers cross a clone.
    auto alias = resolveInsertSyncGMAlias(working, "");
    if (
        failed(alias)) {
        result.internalError = true;
        result.reason = "invalid staged alias contract";
        return abandon();
    }
    MemoryDependentAnalyzer memory;
    memory.setGMContract(working, *alias);
    SyncIRs updated;
    Buffer2MemInfoMap buffers;
    PTOIRTranslator translator(updated, memory, buffers, working, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    SmallVector<Operation*> barrierCandidates;
    for (
        int64_t barrierId : barrierIds) {
        if (
            auto op = findIdentity(working, barrierId)) {
            barrierCandidates.push_back(op);
        }
    }
    auto refinement = refineStorageBarriers(working, updated, barrierCandidates, useMmadChains);
    if (
        refinement.internalError) {
        result.internalError = true;
        result.reason = refinement.reason;
        return abandon();
    }
    result.removed = refinement.removed;
    result.guarded = refinement.guarded;
    result.atoms = refinement.atoms;
    result.requirements = refinement.requirements;
    result.occurrenceProofs = refinement.occurrenceProofs;
    result.work = initial - budget.left + refinement.work;
    working.walk([&](Operation* op) { op->removeAttr(workIdentity); });
    if (
        failed(verify(working.getOperation()))) {
        result.internalError = true;
        result.reason = "invalid final frontier IR";
        return abandon();
    }
    if (
        result.signalsAdvanced || result.waitsDelayed || result.removed || result.guarded) {
        function.getBody().takeBody(working.getBody());
    }
    result.reason = "storage/guard/generation proofs retained; static event keys unchanged; no scarcity serialization";
    return result;
}

namespace {
// Test/development adapter: only explicitly marked barriers are owned. It does
// NOT change production InsertSync's explicit-synchronization bypass behavior.
struct PTORefineStorageFrontiersPass : PassWrapper<PTORefineStorageFrontiersPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORefineStorageFrontiersPass)
    StringRef getArgument() const final { return "pto-refine-storage-frontiers"; }
    StringRef getDescription() const final { return "Refine marked synchronization using storage and lane frontiers"; }
    Option<bool> placement{
        *this, "placement", llvm::cl::init(false),
        llvm::cl::desc("Test complete guarded placement of explicitly marked events")};
    PTORefineStorageFrontiersPass() = default;
    PTORefineStorageFrontiersPass(const PTORefineStorageFrontiersPass& other) : PassWrapper(other)
    {
        placement = other.placement;
    }
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
        SmallVector<Operation*> events;
        if (
            placement) {
            function.walk([&](Operation* op) {
                if (
                    isa<SetFlagOp, WaitFlagOp>(op) && op->hasAttr("pto.insert_sync.frontier_candidate")) {
                    events.push_back(op);
                }
            });
        }
        auto result = refineInsertSyncStorageFrontiers(function, syncIR, candidates, false, placement, events);
        function.emitRemark("InsertSync frontier refinement: ")
            << result.removed << " removed, " << result.guarded << " guarded; " << result.signalsAdvanced
            << " signals advanced, " << result.waitsDelayed << " waits delayed, " << result.boundaryHandoffs
            << " complete boundary constructions; " << result.reason;
        if (
            result.internalError) {
            signalPassFailure();
        }
    }
};
PassRegistration<PTORefineStorageFrontiersPass> registerStorageFrontiers;
} // namespace
