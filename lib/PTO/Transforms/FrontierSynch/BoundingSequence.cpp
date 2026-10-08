// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/BoundingSequence.h"
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
namespace {
bool sameFrame(const RegionalAnalysis& a, const RegionalAnalysis& b)
{
    if (a.expressions != b.expressions || a.accessModel != b.accessModel || a.gmAliasPolicy != b.gmAliasPolicy ||
        a.anchors.size() != b.anchors.size() || a.occurrenceLoops != b.occurrenceLoops ||
        a.outerLoops != b.outerLoops || a.outerDivisors != b.outerDivisors) { return false; }
    for (std::size_t i = 0; i < a.anchors.size(); ++i) {
        const auto& x = a.anchors[i];
        const auto& y = b.anchors[i];
        if (x.phase != y.phase || x.before.block != y.before.block || x.before.before != y.before.before ||
            x.after.block != y.after.block || x.after.before != y.after.before) { return false; }
    }
    return true;
}
std::optional<RegionalAnalysis> childExport(const BoundingSequenceChild& child, bool upper, std::string& error)
{
    const auto& view = upper ? child.bounds.upper : child.bounds.lower;
    const auto& supplied = upper ? child.upperExports : child.lowerExports;
    if (!view || !supplied) { error = "selected child query or sign-specific boundary export unavailable"; return {}; }
    const auto& canonical = view->regional();
    if (!sameFrame(canonical, *supplied)) {
        error = "sign-specific export has a different occurrence frame"; return {};
    }
    // Sequence's current pipe directory requires concrete original phase anchors.
    // Abstract balanced slots keep their mathematics until such an adapter exists.
    if (llvm::any_of(canonical.anchors, [](const auto& anchor) { return !anchor.phase; })) {
        error = "sequence pipe directory for abstract child slots unavailable"; return {};
    }
    auto result = *supplied;
    result.anchors = canonical.anchors;
    result.presence = canonical.presence;
    result.referenceBefore = canonical.referenceBefore;
    result.firstOrdinal = canonical.firstOrdinal;
    result.endpointSiteGuard = canonical.endpointSiteGuard;
    result.endpointInvocationGuard = canonical.endpointInvocationGuard;
    result.endpointEventGuard = canonical.endpointEventGuard;
    result.reachability = canonical.reachability;
    result.numerical = canonical.numerical;
    result.capabilities.exactQueries = true;
    return result;
}
bool validCrossing(const BoundingSequenceCrossing& edge, const std::vector<RegionalAnalysis>& children)
{
    if (edge.source.child >= children.size() || edge.target.child >= children.size() ||
        edge.source.child >= edge.target.child || edge.source.kind != PeriodicEventKind::Completion ||
        edge.target.kind != PeriodicEventKind::Start) { return false; }
    const auto& arena = *children[edge.source.child].expressions;
    RegionalEvent a{edge.source.type, edge.source.ordinal, edge.source.kind, edge.source.visits};
    RegionalEvent b{edge.target.type, edge.target.ordinal, edge.target.kind, edge.target.visits};
    return edge.active < arena.size() && arena.isBoolean(edge.active) &&
        validRegionalEvent(children[edge.source.child], a) && validRegionalEvent(children[edge.target.child], b);
}
void refreshOccurrences(SequenceAnalysis& result)
{
    const auto& state = *result.state;
    result.occurrences.clear();
    for (const auto& port : state.ports) {
        const auto& child = state.children[port.child];
        result.occurrences.push_back({port.child, port.type, child.anchors[port.type],
            child.regional.occurrenceLoops[port.type], port.ordinal, port.visits});
    }
    result.cost = state.costs;
    result.cost.children = state.children.size(); result.cost.cells = state.cells.size();
    result.cost.ports = state.ports.size(); result.cost.crossings = state.crossings.size();
    result.cost.expressionNodes = state.expressions.size();
}
void clearStorage(RegionalAnalysis& regional)
{
    regional.storageBoundary.clear(); regional.accessBoundary.clear(); regional.deferredAccessBoundary.clear();
    regional.storageSelectors = {}; regional.symbolicStorageEffects.clear();
}
BoundingSequenceSign composeSign(func::FuncOp function, const BoundingSequenceInput& specification, bool upper)
{
    BoundingSequenceSign result;
    const auto& selected = upper ? specification.upper : specification.lower;
    if (selected.bridges != BoundingSequenceBridges::PhysicalSelectors &&
        selected.bridges != BoundingSequenceBridges::SuppliedCrossings) {
        result.exportError = "invalid sequence storage bridge mode"; return result;
    }
    std::vector<RegionalAnalysis> children;
    for (const auto& child : specification.children) {
        auto regional = childExport(child, upper, result.exportError);
        if (!regional) { return result; }
        if (selected.bridges == BoundingSequenceBridges::SuppliedCrossings) {
            // This local adapter composes a native-only storage specification.
            // The qualified caller supplies its complete storage crossing set.
            clearStorage(*regional);
            regional->capabilities.completeStorageModel = true;
            regional->capabilities.exactSelectors = true;
        }
        children.push_back(std::move(*regional));
    }
    const auto& crossings = selected.crossings;
    for (const auto& edge : crossings) {
        if (!validCrossing(edge, children)) {
            result.exportError = "invalid supplied sequence crossing"; return result;
        }
    }
    auto arena = specification.children.front().bounds.context->expressions();
    auto sequence = composeRegionalSequenceWithin(function, arena, std::move(children),
        specification.reconstructPrerequisites, false, specification.enclosing);
    if (!sequence.error.empty()) { result.exportError = sequence.error; return result; }
    auto& state = *sequence.state;
    if (!crossings.empty()) {
        // Existing reduction preserves closure. It is safe to add requirements
        // to its retained generators and reduce again, before exporting queries.
        // No cached index/query for the old graph may survive this mutation.
        state.numerical.reset(); state.numericalTree.reset();
        state.numericalChainKeys.clear();
        state.numericalQueryCost = {}; state.reachabilityCache.clear();
        state.crossingIds.clear();
        if (state.crossings.size() > UINT32_MAX) {
            result.exportError = "sequence crossing identity overflow"; return result;
        }
        for (uint32_t i = 0; i < state.crossings.size(); ++i) {
            const auto& edge = state.crossings[i]; state.crossingIds[{edge.source, edge.target}] = i;
        }
        // The numerical specialization releases nativeValueCrossings after
        // incorporating them in its index. Reconstruct the ORIGINAL value links
        // from the unchanged PhaseIndex before clearing/rebuilding incoming.
        // Native links remain native; they never become software commands.
        if (specification.reconstructPrerequisites) {
            state.nativeValueCrossings.clear();
            if (!state.valueBridges()) { result.exportError = state.error; return result; }
        }
        for (const auto& edge : crossings) {
            auto a = state.port(edge.source.child, edge.source.type, edge.source.ordinal, edge.source.visits);
            auto b = state.port(edge.target.child, edge.target.type, edge.target.ordinal, edge.target.visits);
            auto active = state.both(edge.active, state.both(state.present(a), state.present(b)));
            state.crossing({a, active}, {b, active});
        }
        if (!state.error.empty() || !state.closure() || !arena->constructionError().empty()) {
            result.exportError = !state.error.empty() ? state.error : arena->constructionError(); return result;
        }
        refreshOccurrences(sequence);
    }
    result.regional = sequenceRegionalResult(sequence);
    if (selected.bridges == BoundingSequenceBridges::SuppliedCrossings) {
        clearStorage(*result.regional);
        result.regional->capabilities.completeStorageModel = false;
        result.regional->capabilities.exactSelectors = false;
    }
    result.sequence = std::move(sequence);
    return result;
}
} // namespace
BoundingSequenceResult composeBoundingSequence(
    func::FuncOp function, const SyncInput& input, BoundingSequenceInput specification)
{
    BoundingSequenceResult result;
    result.original = std::make_shared<const BoundingSequenceInput>(std::move(specification));
    const auto& original = *result.original;
    if (original.children.empty() || original.children.size() > UINT32_MAX ||
        (original.guarantee != InputOrderGuarantee::InputOrderEquivalent &&
         original.guarantee != InputOrderGuarantee::InputOrderCovering)) {
        result.error = "invalid bounding sequence specification"; return result;
    }
    OrderContext first;
    for (const auto& child : original.children) {
        if (failed(validateBoundingRegionalResult(child.bounds, result.error))) { return result; }
        const auto& context = child.bounds.context;
        if (&context->input() != &input || (first && (first->expressions() != context->expressions() ||
            first->accessModel() != context->accessModel() || first->gmAliasPolicy() != context->gmAliasPolicy()))) {
            result.error = "bounding sequence children have different shared input contexts"; return result;
        }
        if (!first) { first = context; }
    }
    result.lower = composeSign(function, original, false);
    result.upper = composeSign(function, original, true);
    auto* domain = result.upper.regional ? &*result.upper.regional :
        (result.lower.regional ? &*result.lower.regional : nullptr);
    if (!domain) { return result; }
    auto context = captureRegionalOrderContext(input, *domain, result.error);
    if (failed(context)) { return result; }
    result.bounds.context = *context;
    auto exportView = [&](BoundingSequenceSign& sign, std::optional<RegionalOrderView>& destination) {
        if (!sign.regional) { return; }
        auto view = makeRegionalOrderView(*context,
            {sign.regional->reachability, sign.regional->numerical}, sign.exportError);
        if (succeeded(view)) { destination = std::move(*view); }
    };
    exportView(result.lower, result.bounds.lower); exportView(result.upper, result.bounds.upper);
    result.bounds.guarantee = original.guarantee;
    result.bounds.reduction = ReductionQuality::Covers;
    result.placementMayStrengthen = false;
    for (const auto& child : original.children) {
        result.placementMayStrengthen |= child.placementMayStrengthen;
        if (child.bounds.reduction == ReductionQuality::Generators) {
            result.bounds.reduction = ReductionQuality::Generators;
        } else if (child.bounds.reduction == ReductionQuality::Partial &&
                   result.bounds.reduction == ReductionQuality::Covers) {
            result.bounds.reduction = ReductionQuality::Partial;
        }
        if (child.bounds.guarantee != InputOrderGuarantee::InputOrderEquivalent) {
            result.bounds.guarantee = InputOrderGuarantee::InputOrderCovering;
        }
    }
    if (result.upper.sequence) {
        const auto& state = *result.upper.sequence->state;
        for (const auto& edge : state.crossings) {
            if (state.expressions.constantValue(edge.guard) != 0 &&
                state.pipe(edge.source) == state.pipe(edge.target)) {
                result.placementMayStrengthen = true;
            }
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
