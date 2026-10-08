// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic selected graphs on original occurrence anchors; no synthetic lower
// fact in this oracle is asserted to follow from the fixture's storage effects.
#include "PTO/Transforms/FrontierSynch/BoundingSequence.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Expr = fs::RegionExpressions::Id;
using Kind = fs::PeriodicEventKind;
using Graph = std::vector<std::vector<bool>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
}
fs::RegionalAnalysis leaf(const pto::SyncInput& input, const pto::CompoundInstanceElement* phase,
    std::shared_ptr<fs::RegionExpressions> arena, uint64_t trips, bool upper)
{
    fs::RegionalAnalysis result;
    result.expressions = arena; result.accessModel = &input.accesses();
    result.gmAliasPolicy = input.memory().gmPolicy(); result.capabilities = {true, true, true, false};
    auto* operation = phase->elementOp;
    result.anchors.push_back({phase, {}, {operation->getBlock(), operation},
                                       {operation->getBlock(), operation->getNextNode()}});
    result.occurrenceLoops.push_back(operation->getParentOfType<scf::ForOp>());
    auto active = arena->boolean(trips != 0);
    auto pipe = static_cast<uint32_t>(phase->kPipeValue);
    result.firstPayloads[pipe].push_back({{0, arena->constant(0), Kind::Start}, active});
    result.lastPayloads[pipe].push_back({{0, arena->constant(trips ? trips - 1 : 0), Kind::Start}, active});
    result.presence = [arena, trips](fs::RegionalEvent event) -> std::optional<Expr> {
        return arena->lt(event.ordinal, arena->constant(trips));
    };
    result.reachability = [arena, trips, upper](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<Expr> {
        auto present = arena->land(arena->lt(a.ordinal, arena->constant(trips)),
                                   arena->lt(b.ordinal, arena->constant(trips)));
        auto ordered = arena->le(a.ordinal, b.ordinal);
        if (a.kind == Kind::Completion && b.kind == Kind::Start) {
            ordered = upper ? arena->lt(a.ordinal, b.ordinal) : arena->boolean(false);
        }
        return arena->land(present, ordered);
    };
    return result;
}
std::optional<fs::BoundingSequenceInput> specification(const pto::SyncInput& input,
    std::shared_ptr<fs::RegionExpressions> arena, const std::array<uint64_t, 3>& trips, unsigned mask)
{
    fs::BoundingSequenceInput result;
    result.reconstructPrerequisites = false;
    for (uint32_t i = 0; i < 3; ++i) {
        auto lower = leaf(input, input.instructions()[i], arena, trips[i], false);
        auto upper = leaf(input, input.instructions()[i], arena, trips[i], true);
        std::string error;
        auto context = fs::captureRegionalOrderContext(input, lower, error);
        if (failed(context)) { return {}; }
        auto lo = fs::makeRegionalOrderView(*context, {lower.reachability, {}}, error);
        auto hi = fs::makeRegionalOrderView(*context, {upper.reachability, {}}, error);
        if (failed(lo) || failed(hi)) { return {}; }
        fs::BoundingSequenceChild child;
        child.bounds.context = *context; child.bounds.lower = *lo; child.bounds.upper = *hi;
        child.bounds.reduction = fs::ReductionQuality::Covers;
        child.lowerExports = std::move(lower); child.upperExports = std::move(upper);
        child.mathematicalOwner = std::make_shared<const unsigned>(i);
        child.placementMayStrengthen = false;
        result.children.push_back(std::move(child));
    }
    auto edge = [&](uint32_t a, uint32_t b, unsigned owner) {
        return fs::BoundingSequenceCrossing{{a, 0, arena->constant(trips[a] ? trips[a] - 1 : 0), Kind::Completion},
            {b, 0, arena->constant(0), Kind::Start}, arena->boolean(true), {{17, owner}}};
    };
    if (mask & 1) { result.lower.crossings.push_back(edge(0, 2, 0)); }
    result.upper = result.lower;
    if (mask & 2) {
        result.upper.crossings.push_back(edge(0, 1, 1)); result.upper.crossings.push_back(edge(1, 2, 2));
    }
    return result;
}
Graph oracle(const std::array<uint64_t, 3>& trips, const fs::BoundingSequenceSpecification& selected,
             fs::RegionExpressions& arena, bool upper)
{
    Graph graph(18, std::vector<bool>(18));
    for (uint32_t a = 0; a < 9; ++a) {
        if (a % 3 >= trips[a / 3]) { continue; }
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (uint32_t b = a + 1; b < 9; ++b) {
            if (b % 3 >= trips[b / 3]) { continue; }
            graph[2*a][2*b] = graph[2*a+1][2*b+1] = true;
            if (upper && a / 3 == b / 3) { graph[2*a+1][2*b] = true; }
        }
    }
    for (const auto& edge : selected.crossings) {
        auto a = *arena.constantValue(edge.source.ordinal), b = *arena.constantValue(edge.target.ordinal);
        if (a < trips[edge.source.child] && b < trips[edge.target.child]) {
            graph[2*(3*edge.source.child+a)+1][2*(3*edge.target.child+b)] = true;
        }
    }
    close(graph); return graph;
}
bool checkCase(func::FuncOp function, const pto::SyncInput& input,
               const std::array<uint64_t, 3>& trips, unsigned mask, uint64_t& checked)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto spec = specification(input, arena, trips, mask);
    if (!spec) { return false; }
    auto result = fs::composeBoundingSequence(function, input, *spec);
    if (!result.error.empty() || !result.lower.exportError.empty() || !result.upper.exportError.empty() ||
        !result.bounds.lower || !result.bounds.upper || result.bounds.context == spec->children[0].bounds.context ||
        result.original->children[0].mathematicalOwner != spec->children[0].mathematicalOwner ||
        result.upper.regional->anchors.size() != 3 || result.bounds.upper->regional().prepare) {
        llvm::errs() << result.error << "; " << result.lower.exportError << "; " << result.upper.exportError << "\n";
        return false;
    }
    auto event = [&](uint32_t id) {
        return fs::RegionalEvent{id / 6, arena->constant((id / 2) % 3), id % 2 ? Kind::Completion : Kind::Start};
    };
    for (bool upper : {false, true}) {
        auto graph = oracle(trips, upper ? spec->upper : spec->lower, *arena, upper);
        const auto& view = upper ? *result.bounds.upper : *result.bounds.lower;
        for (uint32_t a = 0; a < 18; ++a) {
            for (uint32_t b = 0; b < 18; ++b) {
                auto query = fs::regionalReachability(view.regional(), event(a), event(b));
                if (!query || arena->constantValue(*query) != uint64_t(graph[a][b])) { return false; }
                ++checked;
            }
        }
    }
    // Missing one sign does not erase the other, original records, or owners.
    auto missing = *spec; missing.children[1].lowerExports.reset();
    auto partial = fs::composeBoundingSequence(function, input, std::move(missing));
    if (!partial.error.empty() || partial.lower.exportError.empty() || partial.bounds.lower ||
        !partial.bounds.upper || partial.original->upper.crossings.size() != spec->upper.crossings.size()) {
        return false;
    }
    auto malformed = *spec;
    malformed.upper.crossings.push_back({{2, 0, arena->constant(0), Kind::Completion},
        {0, 0, arena->constant(0), Kind::Start}, arena->boolean(true), {}});
    auto rejected = fs::composeBoundingSequence(function, input, std::move(malformed));
    return rejected.upper.exportError.size() && !rejected.bounds.upper && rejected.bounds.lower;
}
bool checkGuards(func::FuncOp function, const pto::SyncInput& input)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto spec = specification(input, arena, {2, 0, 2}, 1);
    if (!spec) { return false; }
    auto guard = arena->input(function.getArgument(1));
    spec->lower.crossings[0].active = guard;
    spec->upper = spec->lower;
    auto result = fs::composeBoundingSequence(function, input, *spec);
    if (!result.error.empty() || !result.bounds.lower || !result.bounds.upper || !result.placementMayStrengthen) {
        return false;
    }
    auto query = fs::regionalReachability(result.bounds.lower->regional(),
        {0, arena->constant(1), Kind::Completion}, {2, arena->constant(0), Kind::Start});
    if (!query) { return false; }
    for (bool enabled : {false, true}) {
        const std::pair<Expr, Expr> binding{guard, arena->boolean(enabled)};
        fs::RegionExpressions::Substitution substitution(binding);
        if (arena->constantValue(arena->substitute(*query, substitution)) != uint64_t(enabled)) { return false; }
    }
    // Every sign gets the canonical occurrence callbacks, even if a stale
    // query-only sidecar had an unrelated presence closure.
    spec->children[0].upperExports->presence = [arena](fs::RegionalEvent) -> std::optional<Expr> {
        return arena->boolean(false);
    };
    auto canonical = fs::composeBoundingSequence(function, input, *spec);
    if (!canonical.bounds.upper) { return false; }
    auto present = fs::regionalPresence(canonical.bounds.upper->regional(), {0, arena->constant(0), Kind::Start});
    if (!present || arena->constantValue(*present) != 1) { return false; }
    auto supplied = *spec;
    supplied.lower.bridges = supplied.upper.bridges = fs::BoundingSequenceBridges::SuppliedCrossings;
    for (auto& child : supplied.children) {
        for (auto* sidecar : {&*child.lowerExports, &*child.upperExports}) {
            sidecar->capabilities.completeStorageModel = false;
            sidecar->capabilities.exactSelectors = false;
            sidecar->symbolicStorageEffects = {0};
        }
    }
    auto explicitBridges = fs::composeBoundingSequence(function, input, std::move(supplied));
    if (!explicitBridges.bounds.upper || !explicitBridges.bounds.lower ||
        explicitBridges.upper.regional->capabilities.completeStorageModel ||
        explicitBridges.upper.regional->capabilities.exactSelectors ||
        !explicitBridges.upper.regional->symbolicStorageEffects.empty() ||
        explicitBridges.original->children[0].upperExports->symbolicStorageEffects.empty()) { return false; }
    // Context identity within each child is fixed; different child occurrences
    // may have distinct contexts, but may never switch expression arenas.
    auto otherArena = std::make_shared<fs::RegionExpressions>();
    auto foreign = specification(input, otherArena, {2, 0, 2}, 0);
    if (!foreign) { return false; }
    spec->children[1] = foreign->children[1];
    return !fs::composeBoundingSequence(function, input, *spec).error.empty();
}

bool checkNative(func::FuncOp function, const pto::SyncInput& input)
{
    if (input.instructions().size() != 2) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::BoundingSequenceInput spec;
    for (auto* phase : input.instructions()) {
        auto region = leaf(input, phase, arena, 1, false);
        std::string error;
        auto context = fs::captureRegionalOrderContext(input, region, error);
        if (failed(context)) { return false; }
        auto view = fs::makeRegionalOrderView(*context, {region.reachability, {}}, error);
        if (failed(view)) { return false; }
        fs::BoundingSequenceChild child;
        child.bounds.context = *context; child.bounds.lower = *view; child.bounds.upper = *view;
        child.lowerExports = region; child.upperExports = region;
        spec.children.push_back(std::move(child));
    }
    // Even a false extra requirement forces the index mutation path. The
    // original native tgetval -> texpands C->I link must survive reclosure.
    spec.upper.crossings.push_back({{0, 0, arena->constant(0), Kind::Completion},
        {1, 0, arena->constant(0), Kind::Start}, arena->boolean(false), {}});
    auto result = fs::composeBoundingSequence(function, input, std::move(spec));
    if (!result.error.empty() || !result.upper.exportError.empty() || !result.bounds.upper ||
        result.upper.sequence->cost.numericalMerges < 2 || result.upper.sequence->cost.crossings) { return false; }
    auto query = fs::regionalReachability(result.bounds.upper->regional(),
        {0, arena->constant(0), Kind::Completion}, {1, arena->constant(0), Kind::Start});
    return query && arena->constantValue(*query) == 1;
}

} // namespace
int runBoundingSequenceChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (function->hasAttr("test.bounding_native")) {
        if (!checkNative(function, input)) { llvm::errs() << "bounding native reclosure check failed\n"; return 1; }
        return 0;
    }
    if (input.instructions().size() != 3) {
        llvm::errs() << "bounding sequence fixture requires three phases\n"; return 1;
    }
    uint64_t checked = 0;
    for (uint64_t a = 0; a < 3; ++a) {
        for (uint64_t b = 0; b < 3; ++b) {
            for (uint64_t c = 0; c < 3; ++c) {
                for (unsigned mask = 0; mask < 4; ++mask) {
                    if (!checkCase(function, input, {a, b, c}, mask, checked)) {
                        llvm::errs() << "bounding sequence oracle failed: " << a << "," << b << "," << c
                                     << " mask=" << mask << "\n";
                        return 1;
                    }
                }
            }
        }
    }
    if (!checkGuards(function, input)) { llvm::errs() << "bounding sequence guard checks failed\n"; return 1; }
    llvm::outs() << "bounding sequence checked " << checked << " independent event pairs\n";
    return 0;
}
