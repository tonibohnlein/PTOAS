// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Capture one occurrence context; selected graph views inherit only that domain.
#include "PTO/Transforms/FrontierSynch/BoundingRegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
bool validFrames(const RegionalAnalysis& domain)
{
    const auto count = domain.anchors.size();
    if (domain.occurrenceLoops.size() != count) { return false; }
    if (domain.outerLoops.empty()) { return domain.outerDivisors.empty(); }
    if (domain.outerLoops.size() != count ||
        (!domain.outerDivisors.empty() && domain.outerDivisors.size() != count)) { return false; }
    for (std::size_t type = 0; type < count; ++type) {
        if (llvm::any_of(domain.outerLoops[type], [](scf::ForOp loop) { return !loop; })) { return false; }
        if (domain.outerDivisors.empty()) { continue; }
        const auto& divisors = domain.outerDivisors[type];
        if (divisors.size() != domain.outerLoops[type].size() ||
            llvm::any_of(divisors, [](uint64_t divisor) { return divisor == 0; })) { return false; }
    }
    return true;
}
bool validDomainRoots(const RegionalAnalysis& domain)
{
    const auto& arena = *domain.expressions;
    if (domain.firstOrdinal && (*domain.firstOrdinal >= arena.size() ||
        arena.isBoolean(*domain.firstOrdinal))) { return false; }
    for (auto guard : {domain.endpointSiteGuard, domain.endpointInvocationGuard}) {
        if (guard && (*guard >= arena.size() || !arena.isBoolean(*guard))) { return false; }
    }
    return true;
}
void clearSelectedGraph(RegionalAnalysis& domain)
{
    domain.reachability = {};
    domain.numerical.reset();
    domain.arithmeticRelations.reset(); domain.relations.reset();
    domain.symbolicStorage.reset();
    domain.prepare = {};
    domain.prepareWithVisits = {};
    domain.prepareFiltered = {};
    domain.capabilities.exactQueries = false;
    domain.capabilities.endpointRecipes = false;
    domain.cost = {};
}
bool validDirectory(const NumericalChainInterface& index, std::size_t size)
{
    if (size > UINT32_MAX || index.chains.size() > UINT32_MAX || index.chain.size() != size ||
        index.rank.size() != size || index.forward.size() != size || index.reverse.size() != size) { return false; }
    auto remaining = size;
    for (const auto& chain : index.chains) {
        if (chain.size() > remaining) { return false; }
        remaining -= chain.size();
    }
    if (remaining) { return false; }
    // Each event maps back to its unique slot. Equal total slot/event counts
    // then exclude duplicate, missing or unreferenced chain entries.
    for (std::size_t id = 0; id < size; ++id) {
        auto chain = index.chain[id], rank = index.rank[id];
        if (chain >= index.chains.size() || rank >= index.chains[chain].size() ||
            index.chains[chain][rank] != id) { return false; }
    }
    return true;
}
bool validNumerical(const RegionalAnalysis& domain, const RegionalNumericalInterface& numerical)
{
    if (!numerical.index || !numerical.index->error.empty() || !numerical.query || !numerical.thresholds) {
        return false;
    }
    const auto& index = *numerical.index;
    if (!validDirectory(index, numerical.events.size()) ||
        numerical.chainKeys.size() != index.chains.size()) { return false; }
    for (std::size_t id = 0; id < numerical.events.size(); ++id) {
        if (!validRegionalEvent(domain, numerical.events[id]) || index.forward[id].size() != index.chains.size() ||
            index.reverse[id].size() != index.chains.size()) { return false; }
        for (std::size_t chain = 0; chain < index.chains.size(); ++chain) {
            if (index.forward[id][chain] > index.chains[chain].size() ||
                index.reverse[id][chain] > index.chains[chain].size()) { return false; }
        }
    }
    return true;
}
bool validQuality(const BoundingRegionalResult& result)
{
    if (result.guarantee != InputOrderGuarantee::InputOrderEquivalent &&
        result.guarantee != InputOrderGuarantee::InputOrderCovering) { return false; }
    return result.reduction == ReductionQuality::Covers || result.reduction == ReductionQuality::Partial ||
           result.reduction == ReductionQuality::Generators;
}
} // namespace
RegionalOrderContext::RegionalOrderContext(const SyncInput& input, RegionalAnalysis domain)
    : source(&input), canonical(std::move(domain)) {}

FailureOr<OrderContext> captureRegionalOrderContext(
    const SyncInput& input, const RegionalAnalysis& domain, std::string& error)
{
    error.clear();
    if (!domain.expressions || !domain.expressions->constructionError().empty() ||
        domain.accessModel != &input.accesses() ||
        domain.gmAliasPolicy != input.memory().gmPolicy()) {
        error = "order context requires a valid arena and the unchanged shared access model and alias policy";
        return failure();
    }
    if (!validFrames(domain) || !validDomainRoots(domain)) {
        error = "order context has malformed occurrence coordinates or domain expressions";
        return failure();
    }
    llvm::DenseSet<const CompoundInstanceElement*> phases(input.instructions().begin(), input.instructions().end());
    for (const auto& anchor : domain.anchors) {
        if (anchor.phase && !phases.contains(anchor.phase)) {
            error = "order context anchor is outside the unchanged shared input";
            return failure();
        }
    }
    auto canonical = domain;
    clearSelectedGraph(canonical);
    // The private constructor prevents uncaptured contexts. shared_ptr owns the
    // allocation immediately, including control-block allocation failure.
    return OrderContext(new RegionalOrderContext(input, std::move(canonical)));
}
RegionalOrderView::RegionalOrderView(OrderContext context, std::shared_ptr<const RegionalAnalysis> regional)
    : owner(std::move(context)), analysis(std::move(regional)) {}

FailureOr<RegionalOrderView> makeRegionalOrderView(
    OrderContext context, RegionalOrderQueries queries, std::string& error)
{
    error.clear();
    if (!context || !context->expressions()->constructionError().empty() || !queries.reachability) {
        error = "selected order view requires a valid context arena and an exact selected-graph query";
        return failure();
    }
    if (queries.numerical && !validNumerical(context->domain(), *queries.numerical)) {
        error = "selected numerical order export has an invalid event interface";
        return failure();
    }
    auto regional = std::make_shared<RegionalAnalysis>(context->domain());
    regional->reachability = std::move(queries.reachability);
    regional->numerical = std::move(queries.numerical);
    regional->capabilities.exactQueries = true;
    return RegionalOrderView(std::move(context), std::move(regional));
}
LogicalResult validateBoundingRegionalResult(const BoundingRegionalResult& result, std::string& error)
{
    error.clear();
    if (!result.context || !validQuality(result)) {
        error = "mathematical order bounds require a context and valid order/reduction tags";
        return failure();
    }
    if ((result.lower && result.lower->context() != result.context) ||
        (result.upper && result.upper->context() != result.context) ||
        (result.provenance && result.provenance->context() != result.context)) {
        error = "order views and provenance must share the unchanged occurrence context";
        return failure();
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
