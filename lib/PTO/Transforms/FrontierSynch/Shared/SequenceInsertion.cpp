// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "OccurrenceBinding.h"
#include "MixedFamilyInsertion.h"
#include "OriginalCutFacts.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalLaneExports.h"
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
#include "mlir/IR/Dominance.h"
#include <set>
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> SequenceAnalysisState::prepare(
    std::string& diagnostic, ArrayRef<scf::ForOp> enclosing,
    const SequenceEndpointResolver& resolver)
{
    return prepareWithTypeBases(diagnostic, enclosing, {}, resolver);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> SequenceAnalysisState::prepareWithTypeBases(
    std::string& diagnostic, ArrayRef<scf::ForOp> enclosing, ArrayRef<uint32_t> typeBases,
    const SequenceEndpointResolver& resolver)
{
    diagnostic.clear();
    // A failed executable recipe must not poison retained mathematics or a
    // later realization. Do not roll back the shared expression arena: child
    // query caches can retain IDs appended during an unsuccessful attempt.
    auto fail = [&](StringRef message) {
        if (!diagnostic.empty()) { diagnostic += ": "; }
        diagnostic += message.str();
        return false;
    };
    if (!error.empty() || !expressions.constructionError().empty()) {
        fail(!error.empty() ? error : expressions.constructionError());
        return failure();
    }
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    const bool unsupportedWidth = bits.isScalable() || bits.getFixedValue() != 64;
    if (unsupportedWidth) {
        fail("sequence endpoint arithmetic requires a 64-bit index representation"); return failure();
    }
    if (!typeBases.empty() && typeBases.size() != children.size()) {
        fail("regional endpoint owner type-base count differs"); return failure();
    }
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
        if (!expressions.constructionError().empty()) {
            fail(expressions.constructionError()); return failure();
        }
        return result;
    }
    for (const auto& child : children) {
        if (resolver && child.originalNode) { continue; }
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
    struct ChildAllocation {
        std::shared_ptr<PreparedLogicalPlan> view;
        uint32_t typeBase;
        uint32_t recordBase;
    };
    std::vector<ChildAllocation> allocationChildren;
    std::vector<RegionalAllocationGroup> allocationCrossings;
    std::vector<uint32_t> typeOffsets;
    uint32_t nextType = 0;
    uint32_t nextRecord = 0;
    int64_t nextPiece = 0;
    for (auto& child : children) {
        const uint32_t typeBase = typeBases.empty() ? nextType : typeBases[typeOffsets.size()];
        typeOffsets.push_back(typeBase);
        for (auto loop : child.regional.occurrenceLoops) {
            if (!loop) { continue; }
            auto step = sequenceInteger(loop.getStep());
            if (!step || *step <= 0 || !loop.getInductionVar().getType().isIndex() ||
                loop->getParentOfType<func::FuncOp>() != function) {
                fail("regional endpoint loop requires an available constant positive index step"); return failure();
            }
        }
        FailureOr<std::unique_ptr<PreparedLogicalPlan>> supplied = failure();
        if (resolver && child.originalNode) {
            supplied = resolver(*child.originalNode, child.regional, enclosing);
            if (child.originalArmGuard) {
                supplied = maskRegionalArmPlan(std::move(supplied), arena, *child.originalArmGuard);
            }
        } else {
            supplied = child.regional.prepareWithVisits ? child.regional.prepareWithVisits(enclosing) :
                                                         child.regional.prepare();
        }
        if (failed(supplied)) {
            fail("regional child endpoint preparation failed");
            if (!expressions.constructionError().empty()) { fail(expressions.constructionError()); }
            return failure();
        }
        auto prepared = std::move(*supplied);
        if (child.regional.endpointSiteGuard || child.regional.endpointInvocationGuard) {
            auto active = yes();
            if (child.regional.endpointSiteGuard) { active = both(active, *child.regional.endpointSiteGuard); }
            if (child.regional.endpointInvocationGuard) {
                active = both(active, *child.regional.endpointInvocationGuard);
            }
            if (prepared->familyRecipe) {
                auto masked = maskFamilyActivity(prepared->familyRecipe, arena, active);
                if (failed(masked)) { fail("phase family predicate unavailable at original cut"); return failure(); }
                prepared->familyRecipe = std::move(*masked);
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
        result->nestedIdentities |= prepared->nestedIdentities;
        allocationChildren.push_back({takeRegionalAllocationView(*prepared, true), typeBase, nextRecord});
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
        if (prepared->familyRecipe) {
            auto joined = appendFamilyEndpointRecipe(result->familyRecipe, prepared->familyRecipe,
                                                     nextRecord, result->endpoints.size());
            if (failed(joined)) { fail("regional family recipe translation overflow"); return failure(); }
            result->familyRecipe = std::move(*joined);
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
    // Attribute namespace/identity adapters to child import, not to newly
    // constructed crossings. Their work remains in the total preparation count.
    for (const auto& stage : result->preparation) {
        childPreparationOperations += stage.code->getOperations().size();
    }
    ContextualFamilyExpressions familyExpressions(arena);
    DominanceInfo dominance(function);
    const bool contextual = llvm::any_of(children, [](const Child& child) {
        return child.regional.capabilities.contextualGuards;
    });
    auto available = [&](Expr expression, Operation* cut) -> LogicalResult {
        // Retain the incumbent protocol policy: an ordinary provider does not
        // acquire scalar replay merely because its lowering is now deferred.
        if (expression >= expressions.size()) { return failure(); }
        if (!contextual && llvm::any_of(expressions.referencedInputs(expression), [&](const auto& input) {
            return !detail::originalValueAvailable(input.second, cut, dominance);
        })) { return failure(); }
        return success(succeeded(familyExpressions.prepare(expression, cut)));
    };
    std::vector<ContextualEndpointFamily> crossingFamilies;
    SmallVector<uint8_t> numericalEndpoints;
    std::vector<ContextualSharedExpression> shared;
    std::set<std::pair<Operation*, Expr>> sharedRoots;
    auto endpointGuard = [&](const Port& point, Expr guard) {
        const auto& child = children[point.child];
        if (child.regional.endpointSiteGuard) {
            guard = expressions.select(guard, *child.regional.endpointSiteGuard, no());
        }
        std::string error;
        auto conditions = regionalOccurrenceConstraints(child.regional, point.event(), error);
        if (failed(conditions)) { fail(error); return no(); }
        for (auto condition : *conditions) { guard = expressions.select(guard, condition, no()); }
        return guard;
    };
    auto shareAt = [&](Expr expression, Operation* cut) -> LogicalResult {
        if (failed(available(expression, cut))) { return failure(); }
        if (sharedRoots.emplace(cut, expression).second) { shared.push_back({expression, cut}); }
        return success();
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
            break;
        }
        // Required family availability remains authoritative after these
        // optional common-cut probes. No scalar code is constructed here.
    };
    auto addDemand = [&](const Port& a, const Port& b, Expr retained, bool numerical) -> bool {
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

        if (nextRecord == UINT32_MAX) { return fail("sequence endpoint identity overflow"); }
        const auto record = nextRecord++;
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = aa.after; family.targetCut = ab.before;
        family.members.push_back({record, a.type, b.type, aa.coordinates, ab.coordinates});
        result->families.push_back(std::move(family));
        auto append = [&](TemplateEndpointCut cut, EndpointKind kind, Expr guard,
                          ArrayRef<TemplateCoordinate> coordinates) {
            ContextualEndpointFamily endpoint;
            endpoint.piece.cut = cut;
            endpoint.piece.namespaceId = record;
            endpoint.piece.sourcePipe = p; endpoint.piece.targetPipe = q;
            endpoint.piece.kind = kind;
            endpoint.piece.members.push_back({record, {coordinates.begin(), coordinates.end()}});
            endpoint.activity = guard;
            endpoint.ordinal = c(0);
            crossingFamilies.push_back(std::move(endpoint));
            numericalEndpoints.push_back(numerical);
        };
        if (numerical) {
            if (p != q) {
                append(aa.after, EndpointKind::Set, endpointGuard(a, retained), aa.coordinates);
                allocationCrossings.push_back({p, q, 1, {{record, 0, 0,
                    {typeOffsets[a.child] + a.type, a.ordinal, PeriodicEventKind::Start, a.visits},
                    {typeOffsets[b.child] + b.type, b.ordinal, PeriodicEventKind::Completion, b.visits},
                    allocationActive, {}, true}}, {}, false});
            }
            append(ab.before, p == q ? EndpointKind::Barrier : EndpointKind::Wait,
                   endpointGuard(b, retained), ab.coordinates);
            return diagnostic.empty();
        }
        shareEndpointValues(retained, a, b, aa.after.before, ab.before.before);
        auto targetGuard = endpointGuard(b, retained);
        if (p == q) {
            if (failed(available(targetGuard, ab.before.before))) {
                return fail("sequence barrier predicate unavailable at its cut");
            }
            append(ab.before, EndpointKind::Barrier, targetGuard, ab.coordinates);
            return true;
        }
        // Settle selective versus early publication and terminal cleanup from
        // exact expressions. The allocation view owns this protocol before any
        // detached scalar arithmetic or command operands are materialized.
        auto sourcePresence = regionalPresence(ca.regional, a.event());
        if (!sourcePresence) { return fail("sequence publication requires exact source presence"); }
        auto sourceGuard = endpointGuard(a, retained);
        auto selectedSource = endpointGuard(a, *sourcePresence);
        if (!diagnostic.empty()) { return false; }
        auto publication = preparePublicationRecipe(function, expressions, aa.after, ab.before,
            sourceGuard, retained, targetGuard, available, diagnostic,
            SelectedPublication{*sourcePresence, selectedSource});
        if (failed(publication)) {
            llvm::raw_string_ostream detail(diagnostic);
            detail << "; crossing record " << record << " from " << aa.phase->elementOp->getLoc()
                   << " to " << ab.phase->elementOp->getLoc();
            return false;
        }
        const auto& protocol = **publication;
        if (protocol.cleanupCut) {
            auto choices = qualifyTerminalCleanupCuts(
                {ab.before, {protocol.cleanupCut->getBlock(), protocol.cleanupCut}}, arena, targetGuard);
            if (!choices) { return fail("sequence cleanup requires a singleton consumer recipe"); }
            result->families.back().targetChoices = std::move(choices);
        }
        allocationCrossings.push_back({p, q, 1, {{record, 0, 0,
            {typeOffsets[a.child] + a.type, a.ordinal, PeriodicEventKind::Start, a.visits},
            {typeOffsets[b.child] + b.type, b.ordinal, PeriodicEventKind::Completion, b.visits},
            protocol.cleanupCut ? protocol.publicationGuard : allocationActive, {}, true}}, {},
            bool(protocol.cleanupCut)});
        append(aa.after, EndpointKind::Set, protocol.sourceGuard, aa.coordinates);
        append(ab.before, EndpointKind::Wait, targetGuard, ab.coordinates);
        if (protocol.cleanupCut) {
            append({protocol.cleanupCut->getBlock(), protocol.cleanupCut}, EndpointKind::Wait,
                   protocol.cleanupGuard, {});
        }
        return true;
    };
    std::vector<uint8_t> qualified;
    auto finite = prepareFiniteCrossings(*result, allocationCrossings, typeOffsets, nextRecord, qualified);
    if (failed(finite)) { fail("finite crossing family preparation failed"); return failure(); }
    if (!*finite) {
        for (const auto& edge : crossings) {
            if (!addDemand(ports[edge.source], ports[edge.target], edge.guard,
                           qualified[&edge - crossings.data()])) { return failure(); }
        }
    }
    if (!crossingFamilies.empty()) {
        auto recipe = buildMixedFamilyEndpointRecipe(
            familyExpressions, crossingFamilies, numericalEndpoints, {}, shared);
        if (failed(recipe)) {
            recipe = buildContextualFamilyEndpointRecipe(familyExpressions, crossingFamilies, shared);
        }
        if (failed(recipe)) { fail("sequence crossing expressions unavailable at original cuts"); return failure(); }
        auto joined = appendFamilyEndpointRecipe(result->familyRecipe, *recipe, 0, result->endpoints.size());
        if (failed(joined)) { fail("sequence crossing recipe composition failed"); return failure(); }
        result->familyRecipe = std::move(*joined);
    }
    const auto namespaces = familyRecordNamespaces(result->families);
    for (auto& endpoint : result->endpoints) {
        endpoint.record = namespaces.at(endpoint.records.front());
    }
    if (!expressions.constructionError().empty()) { fail(expressions.constructionError()); }
    // Optional ancestor-cut probes do not govern acceptance. Required recipe
    // failures and occurrence-binding failures remain explicit diagnostics.
    if (!diagnostic.empty()) { return failure(); }
    result->allocationPreparation = [owner = shared_from_this(),
        allocationChildren = std::move(allocationChildren),
        allocationCrossings = std::move(allocationCrossings)](PreparedLogicalPlan& plan) {
        auto summary = std::make_shared<RegionalAllocationSummary>();
        for (std::size_t i = 0; i < allocationChildren.size(); ++i) {
            const auto& child = allocationChildren[i];
            prepareAllocationSupport(*child.view, plan.allocationProfile);
            auto allocation = child.view->regionalAllocation;
            if (!allocation) { allocation = finiteRegionalAllocation(owner->children[i].regional, *child.view); }
            if (!allocation) { return; }
            for (auto group : allocation->groups) {
                exportConstantRegionalLanes(group);
                for (auto& member : group.members) {
                    if (member.record > UINT32_MAX - child.recordBase ||
                        member.firstSource.type > UINT32_MAX - child.typeBase ||
                        member.lastTarget.type > UINT32_MAX - child.typeBase) { return; }
                    member.record += child.recordBase;
                    member.firstSource.type += child.typeBase;
                    member.lastTarget.type += child.typeBase;
                }
                for (auto& lane : group.lanes) {
                    for (auto* selectors : {&lane.firstSources, &lane.lastTargets}) {
                        for (auto& selector : *selectors) {
                            if (selector.event.type > UINT32_MAX - child.typeBase) { return; }
                            selector.event.type += child.typeBase;
                        }
                    }
                }
                summary->groups.push_back(std::move(group));
            }
        }
        for (auto group : allocationCrossings) {
            exportConstantRegionalLanes(group);
            summary->groups.push_back(std::move(group));
        }
        plan.regionalAllocation = std::move(summary);
    };
    uint64_t preparationOperations = 0;
    for (const auto& stage : result->preparation) {
        preparationOperations += stage.code->getOperations().size();
    }
    // This boundary measures only newly selected crossing construction;
    // imported child arithmetic and its namespace adapters were charged above.
    crossingPreparationOperations = preparationOperations - childPreparationOperations;
    return result;
}
} // namespace mlir::pto::frontiersynch
