// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include "mlir/IR/Dominance.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> SequenceAnalysisState::prepare(ArrayRef<scf::ForOp> enclosing)
{
    childPreparationOperations = 0;
    crossingPreparationOperations = 0;
    if (ArrayRef<scf::ForOp>(requiredOuterLoops) != enclosing) {
        fail("subregion endpoint binding requires its enclosing repeat coordinates"); return failure();
    }
    if (relationalResult) {
        FailureOr<std::unique_ptr<PreparedLogicalPlan>> result = failure();
        if (relationalResult->prepareWithVisits) { result = relationalResult->prepareWithVisits(enclosing); }
        else if (enclosing.empty() && relationalResult->prepare) { result = relationalResult->prepare(); }
        if (failed(result)) { fail("symbolic sequence endpoint adapter unavailable at the requested cuts"); }
        else { (*result)->completeInvocation = completeInvocation; }
        return result;
    }
    for (const auto& child : children) {
        if (!child.regional.capabilities.endpointRecipes ||
            (!child.regional.prepare && !child.regional.prepareWithVisits)) {
            fail("regional child has no endpoint recipe"); return failure();
        }
        if (llvm::any_of(child.regional.outerLoops, [](const auto& frame) { return !frame.empty(); }) &&
            !child.regional.prepareWithVisits) {
            fail("nested child has no contextual endpoint recipe"); return failure();
        }
    }
    auto result = std::make_unique<PreparedLogicalPlan>(0);
    result->completeInvocation = completeInvocation &&
        llvm::any_of(children, [](const Child& child) { return !child.anchors.empty(); });
    result->groupedFamilies = true;
    result->independentPieces = true;
    result->regionalAllocation = std::make_shared<RegionalAllocationSummary>();
    bool allocationAvailable = true;
    std::vector<uint32_t> typeOffsets;
    uint32_t nextType = 0;
    uint32_t nextRecord = 0;
    int64_t nextPiece = 0;
    for (auto& child : children) {
        typeOffsets.push_back(nextType);
        for (auto loop : child.regional.occurrenceLoops) {
            if (!loop) { continue; }
            auto step = sequenceInteger(loop.getStep());
            if (!step || *step <= 0 || !loop.getInductionVar().getType().isIndex() ||
                loop->getParentOfType<func::FuncOp>() != function) {
                fail("regional endpoint loop requires an available constant positive index step"); return failure();
            }
        }
        auto supplied = child.regional.prepareWithVisits ? child.regional.prepareWithVisits(enclosing) :
                                                          child.regional.prepare();
        if (failed(supplied)) {
            fail("regional child endpoint preparation failed");
            if (!expressions.error().empty()) { error += ": " + expressions.error(); }
            return failure();
        }
        auto prepared = std::move(*supplied);
        if (child.regional.endpointSiteGuard || child.regional.endpointInvocationGuard) {
            auto active = yes();
            if (child.regional.endpointSiteGuard) { active = both(active, *child.regional.endpointSiteGuard); }
            if (child.regional.endpointInvocationGuard) {
                active = both(active, *child.regional.endpointInvocationGuard);
            }
            std::map<Operation*, RegionExpressions::CutEmission> contexts;
            for (auto& endpoint : prepared->endpoints) {
                auto& block = prepared->addPreparation(endpoint.before);
                OpBuilder builder(function.getContext());
                builder.setInsertionPointToEnd(&block);
                auto predicate = expressions.emitContextual(active,
                    builder, endpoint.before, contexts[endpoint.before]);
                if (failed(predicate)) { fail("phase site predicate unavailable at original cut"); return failure(); }
                auto no = builder.create<arith::ConstantIntOp>(endpoint.before->getLoc(), 0, 1);
                endpoint.guard = builder.create<arith::SelectOp>(
                    endpoint.before->getLoc(), *predicate, endpoint.guard, no);
            }
        }
        for (const auto& stage : prepared->preparation) {
            childPreparationOperations += stage.code->getOperations().size();
        }
        result->nestedIdentities |= prepared->nestedIdentities;
        if (!prepared->regionalAllocation) {
            prepared->regionalAllocation = finiteRegionalAllocation(child.regional, *prepared);
        }
        if (!prepared->regionalAllocation) { allocationAvailable = false; }
        else {
            for (auto group : prepared->regionalAllocation->groups) {
                exportConstantRegionalLanes(group);
                for (auto& member : group.members) {
                    if (member.record > UINT32_MAX - nextRecord ||
                        member.firstSource.type > UINT32_MAX - nextType ||
                        member.lastTarget.type > UINT32_MAX - nextType) {
                        fail("regional allocation identity overflow"); return failure();
                    }
                    member.record += nextRecord;
                    member.firstSource.type += nextType;
                    member.lastTarget.type += nextType;
                }
                for (auto& lane : group.lanes) {
                    for (auto* selectors : {&lane.firstSources, &lane.lastTargets}) {
                        for (auto& selector : *selectors) {
                            if (selector.event.type > UINT32_MAX - nextType) {
                                fail("regional lane identity overflow"); return failure();
                            }
                            selector.event.type += nextType;
                        }
                    }
                }
                result->regionalAllocation->groups.push_back(std::move(group));
            }
        }
        if (child.anchors.size() > UINT32_MAX - nextType) {
            fail("regional allocation type overflow"); return failure();
        }
        nextType += child.anchors.size();
        uint32_t maximum = 0;
        bool any = false;
        for (const auto& family : prepared->families) {
            maximum = std::max(maximum, family.id);
            for (const auto& member : family.members) { maximum = std::max(maximum, member.record); any = true; }
        }
        if (any && maximum >= UINT32_MAX - nextRecord) {
            fail("sequence record identity overflow"); return failure();
        }
        for (auto& family : prepared->families) {
            family.id += nextRecord;
            for (auto& member : family.members) {
                maximum = std::max(maximum, member.record);
                any = true;
                member.record += nextRecord;
            }
            result->families.push_back(std::move(family));
        }
        for (auto& stage : prepared->preparation) { result->preparation.push_back(std::move(stage)); }
        for (auto& endpoint : prepared->endpoints) {
            endpoint.record += nextRecord;
            endpoint.piece = nextPiece++;
            if (!prepared->independentPieces) {
                endpoint.records.push_back(static_cast<uint32_t>(endpoint.record - nextRecord));
                if (endpoint.kind != LogicalCommandKind::Barrier) {
                    OpBuilder builder(function.getContext());
                    builder.setInsertionPointToEnd(&result->addPreparation(endpoint.before));
                    endpoint.memberCoordinates.push_back(builder.create<arith::ConstantIndexOp>(
                        endpoint.before->getLoc(), endpoint.record - nextRecord));
                }
            }
            for (auto& record : endpoint.records) { record += nextRecord; }
            if (!endpoint.memberCoordinates.empty() && nextRecord) {
                OpBuilder builder(function.getContext());
                builder.setInsertionPointToEnd(&result->addPreparation(endpoint.before));
                auto shift = builder.create<arith::ConstantIndexOp>(endpoint.before->getLoc(), nextRecord);
                endpoint.memberCoordinates[0] = builder.create<arith::AddIOp>(endpoint.before->getLoc(),
                    endpoint.memberCoordinates[0], shift);
            }
            result->endpoints.push_back(std::move(endpoint));
        }
        if (any) {
            if (maximum >= UINT32_MAX - nextRecord) { fail("sequence record identity overflow"); return failure(); }
            nextRecord += maximum + 1;
        }
    }
    std::map<Operation*, std::unique_ptr<DenseMap<Expr, Value>>> memos;
    std::map<Operation*, Block*> stages;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    const bool contextual = llvm::any_of(children, [](const Child& child) {
        return child.regional.capabilities.contextualGuards;
    });
    auto emit = [&](Expr expression, Operation* cut) -> FailureOr<Value> {
        auto& memo = memos[cut];
        if (!memo) { memo = std::make_unique<DenseMap<Expr, Value>>(); stages[cut] = &result->addPreparation(cut); }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(stages[cut]);
        return contextual ? expressions.emitContextual(expression,builder,cut,contexts[cut]) :
                            expressions.emit(expression,builder,cut,*memo);
    };
    auto endpointGuard = [&](const Port& point, Expr guard) {
        const auto& child = children[point.child];
        if (child.regional.endpointSiteGuard) {
            guard = expressions.select(guard, *child.regional.endpointSiteGuard, no());
        }
        if (child.regional.endpointEventGuard) {
            auto value = child.regional.endpointEventGuard(point.event());
            if (!value) { fail("regional phase endpoint predicate unavailable"); return no(); }
            guard = expressions.select(guard, *value, no());
        }
        const auto& anchor = child.anchors[point.type];
        auto loop = child.regional.occurrenceLoops[point.type];
        if (loop) {
            auto ordinal = expressions.div(expressions.sub(expressions.input(loop.getInductionVar()),
                expressions.input(loop.getLowerBound())), c(*sequenceInteger(loop.getStep())));
            guard = expressions.select(guard, expressions.eq(ordinal, point.ordinal), no());
        }
        if (!child.regional.outerLoops.empty()) {
            const auto& loops = child.regional.outerLoops[point.type];
            for (std::size_t i = 0; i < loops.size(); ++i) {
                auto coordinate = loops[i];
                auto step = sequenceInteger(coordinate.getStep());
                if (!step || *step <= 0) { fail("nested endpoint requires a constant positive step"); return no(); }
                auto ordinal = expressions.div(expressions.sub(expressions.input(coordinate.getInductionVar()),
                    expressions.input(coordinate.getLowerBound())), c(*step));
                if (!child.regional.outerDivisors.empty()) {
                    ordinal = expressions.div(ordinal, c(child.regional.outerDivisors[point.type][i]));
                }
                guard = expressions.select(guard, expressions.eq(ordinal, point.visits[i]), no());
            }
        }
        for (auto coordinate : anchor.coordinates) {
            guard = expressions.select(guard, expressions.eq(expressions.input(coordinate.loop.getInductionVar()),
                                               c(coordinate.induction)), no());
        }
        return guard;
    };
    struct SharedPreparation {
        Block* block = nullptr;
        DenseMap<Expr, Value> values;
        RegionExpressions::CutEmission context;
    };
    std::map<Operation*, SharedPreparation> shared;
    DominanceInfo dominance(function);
    auto shareAt = [&](Expr expression, Operation* cut) -> FailureOr<Value> {
        auto found = shared.find(cut);
        DenseMap<Expr, Value> values;
        RegionExpressions::CutEmission context;
        if (found != shared.end()) {
            values = found->second.values;
            context = found->second.context;
        }
        // Availability can fail at an ancestor cut even when endpoint-local
        // emission succeeds. Work transactionally: failed attempts leave no
        // operations or memo values in the accepted detached preparation.
        Block scratch;
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(&scratch);
        auto value = contextual ? expressions.emitContextual(expression, builder, cut, context) :
                                  expressions.emit(expression, builder, cut, values);
        if (failed(value)) { return failure(); }
        auto& stage = shared[cut];
        if (!stage.block) {
            stage.block = &result->addPreparation(cut);
            // Shared stages use original SSA values and their own preceding
            // operations only. They never borrow an endpoint's detached code,
            // so placing them before endpoint stages preserves SSA availability.
            std::rotate(result->preparation.begin(), std::prev(result->preparation.end()),
                        result->preparation.end());
        }
        stage.block->getOperations().splice(stage.block->end(), scratch.getOperations());
        stage.values = std::move(values);
        stage.context = std::move(context);
        stage.values[expression] = *value;
        return *value;
    };
    auto shareEndpointValues = [&](Expr retained, const Port& source, const Port& target,
                                   Operation* sourceCut, Operation* targetCut) {
        SmallVector<Expr> roots{retained, source.ordinal, target.ordinal};
        llvm::append_range(roots, source.visits);
        llvm::append_range(roots, target.visits);
        // Reference-forward siblings need not have the same leaf loop. Their
        // common original dominator can still compute immutable boundary maps
        // once, including bulk-to-loop and loop-to-bulk handoffs.
        SmallVector<Operation*> candidates;
        for (auto* candidate = sourceCut; candidate && candidate != function.getOperation();
             candidate = candidate->getParentOp()) { candidates.push_back(candidate); }
        // Prefer the outermost available cut, so an invariant does not move
        // from one loop-body endpoint to another and still execute every trip.
        for (auto* candidate : llvm::reverse(candidates)) {
            auto dominates = [&](Operation* endpoint) {
                return candidate == endpoint || dominance.properlyDominates(candidate, endpoint);
            };
            if (!dominates(sourceCut) || !dominates(targetCut)) { continue; }
            if (failed(shareAt(retained, candidate))) { continue; }
            for (auto root : roots) { (void)shareAt(root, candidate); }
            const auto& stage = shared.at(candidate);
            for (auto* cut : {sourceCut, targetCut}) {
                auto& memo = memos[cut];
                if (!memo) {
                    memo = std::make_unique<DenseMap<Expr, Value>>();
                    stages[cut] = &result->addPreparation(cut);
                }
                auto seed = [&](const auto& values) {
                    for (const auto& item : values) {
                        (*memo)[item.first] = item.second;
                        contexts[cut].values[item.first] = item.second;
                        contexts[cut].cofactors[item.first] = item.first;
                    }
                };
                seed(stage.values);
                seed(stage.context.values);
            }
            break;
        }
        // Ordinary endpoint emission below clears failed optional-attempt
        // diagnostics and remains authoritative for guard availability.
    };
    auto addDemand = [&](const Port& a, const Port& b, Expr retained) -> bool {
        if (expressions.constantValue(retained) == 0) { return true; }
        const auto& ca = children[a.child];
        const auto& cb = children[b.child];
        // Keep the uniform semantic guard in allocation summaries. The
        // invocation masks below bind executable endpoints to the current
        // outer visit; repeated allocation clips its lifetime spans separately.
        const auto allocationActive = retained;
        if (ca.regional.endpointInvocationGuard) { retained = both(retained, *ca.regional.endpointInvocationGuard); }
        if (cb.regional.endpointInvocationGuard) { retained = both(retained, *cb.regional.endpointInvocationGuard); }
        const auto& aa = ca.anchors[a.type];
        const auto& ab = cb.anchors[b.type];
        auto p = static_cast<uint32_t>(aa.phase->kPipeValue);
        auto q = static_cast<uint32_t>(ab.phase->kPipeValue);

        shareEndpointValues(retained, a, b, aa.after.before, ab.before.before);

        if (nextRecord == UINT32_MAX) { return fail("sequence endpoint identity overflow"); }
        const auto record = nextRecord++;
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = aa.after; family.targetCut = ab.before;
        family.members.push_back({record, a.type, b.type, aa.coordinates, ab.coordinates});
        result->families.push_back(std::move(family));
        auto targetGuard = emit(endpointGuard(b, retained), ab.before.before);
        if (failed(targetGuard)) { return fail("sequence WAIT/barrier predicate unavailable at its cut"); }
        if (p == q) {
            PreparedLogicalEndpoint endpoint{ab.before.before, LogicalCommandKind::Barrier, p, q,
                                              record, *targetGuard, {}};
            endpoint.records.push_back(record); endpoint.piece = nextPiece++;
            result->endpoints.push_back(std::move(endpoint));
            return true;
        }
        // Each crossing selects one source and one consumer occurrence in
        // this region invocation. Guarded port choices change that pair, never
        // its multiplicity; enclosing repetition must drop this certificate.
        result->regionalAllocation->groups.push_back({p, q, 1, {{record, 0, 0,
            {typeOffsets[a.child] + a.type, a.ordinal, PeriodicEventKind::Start, a.visits},
            {typeOffsets[b.child] + b.type, b.ordinal, PeriodicEventKind::Completion, b.visits},
            allocationActive, {}, true}}});
        exportConstantRegionalLanes(result->regionalAllocation->groups.back());
        auto sourceGuard = emit(endpointGuard(a, retained), aa.after.before);
        auto sourceIdentity = emit(c(0), aa.after.before);
        auto targetIdentity = emit(c(0), ab.before.before);
        auto sourceMember = emit(c(record), aa.after.before);
        auto targetMember = emit(c(record), ab.before.before);
        if (failed(sourceGuard) || failed(sourceIdentity) || failed(targetIdentity) ||
            failed(sourceMember) || failed(targetMember)) {
            return fail("sequence SET predicate or matching identity unavailable at its cut");
        }
        PreparedLogicalEndpoint set{aa.after.before, LogicalCommandKind::Set, p, q, record,
                                     *sourceGuard, *sourceIdentity};
        set.memberCoordinates.push_back(*sourceMember); set.records.push_back(record); set.piece = nextPiece++;
        PreparedLogicalEndpoint wait{ab.before.before, LogicalCommandKind::Wait, p, q, record,
                                      *targetGuard, *targetIdentity};
        wait.memberCoordinates.push_back(*targetMember); wait.records.push_back(record); wait.piece = nextPiece++;
        result->endpoints.push_back(std::move(set)); result->endpoints.push_back(std::move(wait));
        return true;
    };
    for (const auto& edge : crossings) {
        if (!addDemand(ports[edge.source], ports[edge.target], edge.guard)) { return failure(); }
    }
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    std::map<uint32_t, Namespace> recordNamespace;
    for (const auto& family : result->families) {
        Namespace key{family.sourcePipe, family.targetPipe, family.displacement};
        for (const auto& member : family.members) {
            auto [it, added] = namespaces.emplace(key, member.record);
            it->second = std::min(it->second, member.record);
            recordNamespace.emplace(member.record, key);
        }
    }
    for (auto& endpoint : result->endpoints) {
        endpoint.record = namespaces.at(recordNamespace.at(endpoint.records.front()));
    }
    if (!expressions.error().empty()) { fail(expressions.error()); return failure(); }
    if (!allocationAvailable) { result->regionalAllocation.reset(); }
    uint64_t preparationOperations = 0;
    for (const auto& stage : result->preparation) {
        preparationOperations += stage.code->getOperations().size();
    }
    // Includes the small identity/namespace adapters added while importing the
    // children, as well as the preparation for newly selected crossing edges.
    crossingPreparationOperations = preparationOperations - childPreparationOperations;
    return result;
}
} // namespace mlir::pto::frontiersynch
