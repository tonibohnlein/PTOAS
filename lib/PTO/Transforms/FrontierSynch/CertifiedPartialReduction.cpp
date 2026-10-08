// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CertifiedPartialReduction.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
class Checker {
public:
    explicit Checker(CertifiedPartialReduction& result)
        : result(result), graph(*result.graph), domain(graph.actualPaths.context()->domain()),
          arena(*domain.expressions) {}
    bool prepare();
    bool attempt(const PartialReductionAttempt& proof);
private:
    bool fail(const char* message) { result.error = message; return false; }
    bool boolean(Id id) const { return id < arena.size() && arena.isBoolean(id); }
    bool event(const RegionalEvent& value) const { return validRegionalEvent(domain, value); }
    std::optional<Id> present(const RegionalEvent& value);
    std::optional<Id> before(const RegionalEvent& a, const RegionalEvent& b);
    std::optional<Id> path(const RegionalOrderView& view, const RegionalEvent& a, const RegionalEvent& b);
    std::optional<Id> checked(std::optional<Id> value);
    std::optional<Id> intermediate(const PartialReductionGenerator& edge, const RegionalEvent& witness);
    CertifiedPartialReduction& result;
    const CertifiedPartialGraph& graph;
    const RegionalAnalysis& domain;
    RegionExpressions& arena;
    std::vector<Id> live;
};
std::optional<Id> Checker::checked(std::optional<Id> value)
{
    if (value && !boolean(*value)) {
        fail("partial reduction query returned an invalid Boolean expression");
        return std::nullopt;
    }
    return value;
}
std::optional<Id> Checker::present(const RegionalEvent& value)
{
    ++result.cost.presenceQueries;
    return checked(regionalPresence(domain, value));
}
std::optional<Id> Checker::before(const RegionalEvent& a, const RegionalEvent& b)
{
    ++result.cost.orderQueries;
    return checked(regionalReferenceBefore(domain, a, b));
}
std::optional<Id> Checker::path(const RegionalOrderView& view, const RegionalEvent& a, const RegionalEvent& b)
{
    ++result.cost.pathQueries;
    return checked(view.regional().reachability(a, b));
}
bool Checker::prepare()
{
    llvm::DenseSet<std::pair<uint64_t, uint64_t>> records;
    for (const auto& edge : graph.generators) {
        if (!records.insert({edge.record.producer, edge.record.originalRecord}).second) {
            return fail("partial reduction original record key is duplicated");
        }
        if (!boolean(edge.active) || !event(edge.source) || !event(edge.target) ||
            edge.source.kind != PeriodicEventKind::Completion || edge.target.kind != PeriodicEventKind::Start) {
            return fail("partial reduction generator has malformed endpoints or guard");
        }
        auto source = present(edge.source), target = present(edge.target);
        auto order = before(edge.source, edge.target);
        if (!result.error.empty()) { return false; }
        auto active = source && target ? arena.land(edge.active, arena.land(*source, *target)) : arena.boolean(false);
        bool forward = false;
        if (source && target && order) {
            ++result.cost.implicationChecks;
            forward = arena.implies(active, *order);
            if (arena.constantValue(active) == 1 && arena.constantValue(*order) == 0) {
                return fail("partial reduction input contradicts reference-forward order");
            }
        }
        // The producer supplies forward generators. An incomplete sufficient
        // proof only disables this reducer's attempts, preserving the input.
        live.push_back(forward ? active : arena.boolean(false));
        result.retained.push_back(edge.active);
        result.removed.push_back(arena.boolean(false));
    }
    return true;
}
std::optional<Id> Checker::intermediate(const PartialReductionGenerator& edge, const RegionalEvent& witness)
{
    auto exists = present(witness);
    auto first = before(edge.source, witness), second = before(witness, edge.target);
    if (!exists || !first || !second || !result.error.empty()) { return std::nullopt; }
    auto interior = arena.land(*exists, arena.land(*first, *second));
    if (arena.constantValue(interior) == 0) { return interior; }
    auto prefix = path(graph.actualPaths, edge.source, witness);
    auto suffix = path(graph.actualPaths, witness, edge.target);
    if (!prefix || !suffix || !result.error.empty()) { return std::nullopt; }
    return arena.land(interior, arena.land(*prefix, *suffix));
}
bool Checker::attempt(const PartialReductionAttempt& proof)
{
    ++result.cost.attempts;
    if (proof.graph != result.graph || proof.generator >= graph.generators.size() || !boolean(proof.guard)) {
        return fail("partial reduction attempt has stale graph, record index or guard");
    }
    if (proof.intermediate && !event(*proof.intermediate)) {
        return fail("partial reduction attempt has a malformed intermediate event");
    }
    if (arena.constantValue(live[proof.generator]) == 0) { return true; }
    const auto& edge = graph.generators[proof.generator];
    std::optional<Id> witness;
    if (proof.intermediate) { witness = intermediate(edge, *proof.intermediate); }
    else if (graph.nativePaths) { witness = path(*graph.nativePaths, edge.source, edge.target); }
    if (!result.error.empty()) { return false; }
    if (!witness) { return true; }
    auto guard = arena.land(live[proof.generator], arena.land(proof.guard, *witness));
    auto& removed = result.removed[proof.generator];
    removed = arena.lor(removed, guard);
    // Other attempts always query the original snapshot, never these outputs.
    result.retained[proof.generator] = arena.land(edge.active, arena.lnot(removed));
    return true;
}
bool validContext(const PartialGraphSnapshot& graph)
{
    if (!graph || !graph->actualPaths.context()) { return false; }
    const auto& context = graph->actualPaths.context();
    const auto& domain = context->domain();
    if (!domain.expressions || !domain.expressions->constructionError().empty() || !domain.presence ||
        !graph->actualPaths.regional().reachability ||
        graph->actualPaths.regional().expressions != domain.expressions) {
        return false;
    }
    if (graph->nativePaths && (graph->nativePaths->context() != context ||
        !graph->nativePaths->regional().reachability ||
        graph->nativePaths->regional().expressions != domain.expressions)) {
        return false;
    }
    return !graph->provenance || graph->provenance->context() == context;
}
} // namespace
CertifiedPartialReduction reduceCertifiedPartial(
    PartialGraphSnapshot graph, llvm::ArrayRef<PartialReductionAttempt> attempts)
{
    CertifiedPartialReduction result;
    result.graph = std::move(graph);
    if (!validContext(result.graph)) {
        result.error = "partial reduction graph, arena or provenance context is invalid";
        return result;
    }
    auto& arena = *result.graph->actualPaths.context()->expressions();
    const auto initialSize = arena.size();
    Checker checker(result);
    if (checker.prepare()) {
        for (const auto& attempt : attempts) { if (!checker.attempt(attempt)) { break; } }
    }
    if (result.error.empty() && !arena.constructionError().empty()) { result.error = arena.constructionError(); }
    result.cost.expressionNodes = arena.size() - initialSize;
    if (!result.error.empty()) { result.retained.clear(); result.removed.clear(); }
    return result;
}
} // namespace mlir::pto::frontiersynch
