// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic selected graphs on original occurrence anchors; no synthetic lower
// fact in this oracle is asserted to follow from the fixture's storage effects.
#include "PTO/Transforms/FrontierSynch/BoundingRepetition.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
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
fs::RegionalAnalysis body(const pto::SyncInput& input, std::shared_ptr<fs::RegionExpressions> arena,
                          uint64_t inner, bool upper)
{
    fs::RegionalAnalysis result;
    result.expressions = arena; result.accessModel = &input.accesses();
    result.gmAliasPolicy = input.memory().gmPolicy(); result.capabilities = {true, true, true, false};
    for (uint32_t type = 0; type < 2; ++type) {
        auto* phase = input.instructions()[type]; auto* operation = phase->elementOp;
        result.anchors.push_back({phase, {}, {operation->getBlock(), operation},
                                             {operation->getBlock(), operation->getNextNode()}});
        result.occurrenceLoops.push_back(operation->getParentOfType<scf::ForOp>());
        result.firstSitePayloads[type] = {{{type, arena->constant(0), Kind::Start}, arena->boolean(inner != 0)}};
    }
    const auto pipe = static_cast<uint32_t>(input.instructions()[0]->kPipeValue);
    result.firstPayloads[pipe] = {{{0, arena->constant(0), Kind::Start}, arena->boolean(inner != 0)}};
    result.lastPayloads[pipe] = {{{1, arena->constant(inner ? inner-1 : 0), Kind::Start},
                                 arena->boolean(inner != 0)}};
    result.presence = [arena, inner](fs::RegionalEvent event) -> std::optional<Id> {
        return arena->lt(event.ordinal, arena->constant(inner));
    };
    result.referenceBefore = [arena](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<Id> {
        return arena->lor(arena->lt(a.ordinal, b.ordinal),
            arena->land(arena->eq(a.ordinal, b.ordinal), arena->boolean(a.type < b.type)));
    };
    auto before = result.referenceBefore;
    result.reachability = [arena, inner, upper, before](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<Id> {
        auto present = arena->land(arena->lt(a.ordinal, arena->constant(inner)),
                                   arena->lt(b.ordinal, arena->constant(inner)));
        auto order = arena->lnot(*before(b, a));
        if (a.kind == Kind::Completion && b.kind == Kind::Start) {
            order = upper ? *before(a, b) : arena->boolean(false);
        }
        return arena->land(present, order);
    };
    return result;
}
std::optional<fs::BoundingRepetitionInput> specification(func::FuncOp function, const pto::SyncInput& input,
    std::shared_ptr<fs::RegionExpressions> arena, uint64_t inner, uint64_t visits, uint64_t begin, uint32_t phases)
{
    fs::BoundingRepetitionInput result;
    result.trips = arena->constant(visits); result.begin = arena->constant(begin);
    result.lower.bridges = result.upper.bridges = fs::BoundingSequenceBridges::SuppliedCrossings;
    for (uint32_t phase = 0; phase < phases; ++phase) {
        auto lo = body(input, arena, inner, false), hi = body(input, arena, inner, true);
        std::string error;
        auto context = fs::captureRegionalOrderContext(input, lo, error);
        if (failed(context)) { return {}; }
        auto lower = fs::makeRegionalOrderView(*context, {lo.reachability, {}}, error);
        auto upper = fs::makeRegionalOrderView(*context, {hi.reachability, {}}, error);
        if (failed(lower) || failed(upper)) { return {}; }
        fs::BoundingSequenceChild child;
        child.bounds.context = *context; child.bounds.lower = *lower; child.bounds.upper = *upper;
        child.bounds.reduction = fs::ReductionQuality::Covers;
        child.lowerExports = std::move(lo); child.upperExports = std::move(hi);
        child.mathematicalOwner = std::make_shared<const uint32_t>(phase);
        child.placementMayStrengthen = false;
        result.phases.push_back(std::move(child));
    }
    auto edge = [&](uint32_t source, uint32_t target, uint64_t ordinal, uint64_t distance, Id guard) {
        return fs::BoundingRepeatedCrossing{{{source, arena->constant(ordinal), Kind::Completion},
            {target, arena->constant(0), Kind::Start}, guard, false, distance}, {{19, source}}};
    };
    result.lower.crossings.push_back(edge(1, 0, inner ? inner-1 : 0, 3, arena->input(function.getArgument(2))));
    result.upper = result.lower;
    result.upper.crossings.push_back(edge(0, 1, inner ? inner-1 : 0, 2, arena->boolean(true)));
    // At inner=1 these records denote the same semantic endpoints.
    result.upper.crossings.push_back(edge(0, 1, 0, 2, arena->boolean(true)));
    return result;
}
Graph oracle(const fs::BoundingRepetitionInput& spec, fs::RegionExpressions& arena, uint64_t inner,
             uint64_t visits, uint64_t begin, uint32_t phases, bool upper, bool enabled)
{
    const uint64_t count = 2*inner*visits;
    Graph graph(2*count, std::vector<bool>(2*count));
    if (!inner) { return graph; }
    for (uint64_t a = 0; a < count; ++a) {
        if (a/(2*inner) < begin) { continue; }
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (uint64_t b = a+1; b < count; ++b) {
            graph[2*a][2*b] = graph[2*a+1][2*b+1] = true;
            if (upper && a/(2*inner) == b/(2*inner)) { graph[2*a+1][2*b] = true; }
        }
    }
    const auto& edges = upper ? spec.upper.crossings : spec.lower.crossings;
    for (const auto& record : edges) {
        const auto& edge = record.edge;
        if (arena.constantValue(edge.guard) != 1 && !enabled) { continue; }
        auto ordinal = *arena.constantValue(edge.source.ordinal);
        if (ordinal >= inner) { continue; }
        for (uint64_t source = begin; source < visits; ++source) {
            if (source%phases != edge.source.type/2) { continue; }
            auto target = phases*(source/phases+edge.displacement)+edge.target.type/2;
            if (target >= visits) { continue; }
            auto a = 2*inner*source+2*ordinal+edge.source.type%2;
            auto b = 2*inner*target+edge.target.type%2;
            graph[2*a+1][2*b] = true;
        }
    }
    close(graph); return graph;
}
bool checkCase(func::FuncOp function, scf::ForOp loop, const pto::SyncInput& input,
               uint64_t inner, uint64_t visits, uint64_t begin, uint32_t phases, uint64_t& queries)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto spec = specification(function, input, arena, inner, visits, begin, phases);
    if (!spec) { return false; }
    auto result = fs::repeatBoundingRegion(function, loop, input, *spec);
    if (!result.error.empty() || !result.bounds.lower || !result.bounds.upper ||
        result.lower.query->ports().size() != result.upper.query->ports().size() ||
        result.bounds.upper->regional().prepare || result.upper.repeated->regional.capabilities.completeStorageModel ||
        result.original->phases[0].mathematicalOwner != spec->phases[0].mathematicalOwner ||
        (inner && result.upper.constructionCost.deletionTests < 2)) {
        llvm::errs() << result.error << "; " << result.lower.exportError << "; " << result.upper.exportError << "\n";
        return false;
    }
    auto event = [&](uint64_t id) {
        const auto occurrence = id/2, visit = occurrence/(2*inner), site = occurrence%2;
        return fs::RegionalEvent{static_cast<uint32_t>(2*(visit%phases)+site),
            arena->constant((occurrence/2)%inner), id%2 ? Kind::Completion : Kind::Start,
            {arena->constant(visit/phases)}};
    };
    for (bool upper : {false, true}) {
        for (bool enabled : {false, true}) {
            const auto graph = oracle(*spec, *arena, inner, visits, begin, phases, upper, enabled);
            const auto& view = upper ? *result.bounds.upper : *result.bounds.lower;
            const std::pair<Id, Id> binding{arena->input(function.getArgument(2)), arena->boolean(enabled)};
            fs::RegionExpressions::Substitution substitution(binding);
            for (uint64_t a = 0; a < graph.size(); ++a) {
                for (uint64_t b = 0; b < graph.size(); ++b) {
                    auto answer = fs::regionalReachability(view.regional(), event(a), event(b));
                    if (!answer || arena->constantValue(arena->substitute(*answer, substitution)) !=
                                   uint64_t(graph[a][b])) {
                        llvm::errs() << "repeated mismatch K=" << inner << " T=" << visits << " begin=" << begin
                                     << " q=" << phases << " upper=" << upper << " guard=" << enabled
                                     << " pair=" << a << "," << b << "\n";
                        return false;
                    }
                    ++queries;
                }
            }
        }
    }
    auto missing = *spec; missing.phases[0].lowerExports.reset();
    auto partial = fs::repeatBoundingRegion(function, loop, input, std::move(missing));
    if (partial.bounds.lower || !partial.bounds.upper || partial.lower.exportError.empty() ||
        partial.original->lower.crossings.size() != spec->lower.crossings.size()) { return false; }
    auto malformed = *spec; malformed.upper.crossings[0].edge.displacement = 0;
    auto rejected = fs::repeatBoundingRegion(function, loop, input, std::move(malformed));
    return rejected.bounds.lower && !rejected.bounds.upper && !rejected.upper.exportError.empty();
}
bool checkWideDistances()
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::RegionalAnalysis body; body.expressions = arena;
    body.presence = [arena](fs::RegionalEvent event) -> std::optional<Id> {
        return event.type < 3 ? std::optional<Id>(arena->boolean(true)) : std::nullopt;
    };
    body.reachability = [arena](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<Id> {
        return arena->boolean(a.type == b.type && (a.kind == Kind::Start || b.kind == Kind::Completion));
    };
    std::vector<fs::RegionalEvent> ports;
    std::vector<fs::RepeatedCrossing> edges;
    for (uint32_t type = 0; type < 3; ++type) {
        for (auto kind : {Kind::Start, Kind::Completion}) {
            fs::RegionalEvent event{type, arena->constant(0), kind}; ports.push_back(event);
            edges.push_back({event, event, arena->boolean(true), true, 1});
        }
    }
    edges.push_back({ports[1], ports[2], arena->boolean(true), false, UINT64_MAX});
    edges.push_back({ports[3], ports[4], arena->boolean(true), false, 1});
    std::string error;
    auto query = fs::buildBoundingRepeatedQuery(body, ports, edges, error);
    if (!query || !error.empty() || query->cost().bodyQueries || query->cost().relaxations) { return false; }
    // Unit proofs must not initialize P^3 closure. Sibling snapshots reuse
    // immutable body answers, but never each other's crossing-dependent result.
    auto unit = query->across(ports[3], ports[4], arena->constant(1));
    if (!unit || arena->constantValue(*unit) != 1 || query->cost().relaxations != edges.size() ||
        !query->cost().bodyQueries) { return false; }
    auto sibling = query->withCrossings(edges, error);
    if (!sibling || !error.empty()) { return false; }
    auto same = sibling->across(ports[3], ports[4], arena->constant(1));
    if (!same || arena->constantValue(*same) != 1 || sibling->cost().bodyQueries) { return false; }
    auto removed = edges; removed.back().guard = arena->boolean(false);
    auto different = sibling->withCrossings(std::move(removed), error);
    sibling.reset(); // Shared body memo ownership survives its originating query.
    if (!different || !error.empty()) { return false; }
    auto absent = different->across(ports[3], ports[4], arena->constant(1));
    if (!absent || arena->constantValue(*absent) != 0 || different->cost().bodyQueries) { return false; }
    auto finite = query->distance(ports[1], ports[2]), huge = query->distance(ports[1], ports[4]);
    auto unreachable = query->distance(ports[5], ports[0]);
    auto yes = query->across(ports[1], ports[2], arena->constant(UINT64_MAX));
    auto no = query->across(ports[1], ports[4], arena->constant(UINT64_MAX));
    if (!finite || !huge || !unreachable || !yes || !no ||
        arena->constantValue(finite->value) != UINT64_MAX || arena->constantValue(finite->representable) != 1 ||
        arena->constantValue(huge->reachable) != 1 || arena->constantValue(huge->representable) != 0 ||
        arena->constantValue(unreachable->reachable) != 0 || arena->constantValue(*yes) != 1 ||
        arena->constantValue(*no) != 0) { return false; }
    auto invalid = ports[0]; invalid.ordinal = arena->boolean(false);
    if (query->distance(invalid, ports[0])) { return false; }
    edges[0].displacement = 0;
    return !fs::buildBoundingRepeatedQuery(body, ports, edges, error) && !error.empty();
}
} // namespace
int runBoundingRepetitionChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (input.instructions().size() != 2 || function.getNumArguments() != 3 ||
        input.instructions()[0]->kPipeValue != input.instructions()[1]->kPipeValue) { return 1; }
    auto inner = input.instructions()[0]->elementOp->getParentOfType<scf::ForOp>();
    auto loop = inner ? inner->getParentOfType<scf::ForOp>() : scf::ForOp{};
    if (!loop || !checkWideDistances()) { llvm::errs() << "weighted distance checks failed\n"; return 1; }
    uint64_t queries = 0;
    for (uint32_t phases : {1u, 2u}) {
        for (uint64_t innerCount : {0u, 1u, 2u}) {
            for (uint64_t visits : {0u, 1u, 2u, 4u, 5u}) {
                for (uint64_t begin : {0u, 1u}) {
                    if (!checkCase(function, loop, input, innerCount, visits, begin, phases, queries)) { return 1; }
                }
            }
        }
    }
    llvm::outs() << "bounding repetition checks: " << queries << " exact nested event pairs\n";
    return 0;
}
