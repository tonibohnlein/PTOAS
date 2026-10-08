// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ControlOriginDistances.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
using namespace mlir::pto::frontiersynch;
namespace {
// Independent exact oracle over a weight-product graph. This is not a sampled
// trip prefix: a walk of weight >=V contains a positive cycle. Conversely every
// descendant of a reachable positive cycle has a witness of weight <=3V:
// take a simple prefix, repeat one simple positive cycle until the weight first
// reaches V (now <2V), then take a simple exit (<V). Finite maxima are <V.
// Thus all vertices' min/max/infinite answers follow from these finite states,
// without sharing shortest-path, SCC or condensation code with production.
std::vector<OriginDistanceInterval> oracle(const ControlOriginGraph& graph, const ControlStorageOrigin& origin)
{
    std::vector<ControlOriginEdge> edges = graph.edges;
    for (uint32_t id = 0; id < graph.payloads.size(); ++id) {
        if (std::find(origin.definiteKillPayloads.begin(), origin.definiteKillPayloads.end(), id) !=
            origin.definiteKillPayloads.end()) { continue; }
        const auto& payload = graph.payloads[id];
        edges.push_back({payload.input, payload.output, 0});
    }
    const auto bound = 3 * graph.vertices;
    std::vector<std::vector<bool>> reached(graph.vertices, std::vector<bool>(bound + 1, false));
    const auto source = graph.payloads[origin.payload].output;
    reached[source][0] = true;
    std::vector<std::pair<uint32_t, uint32_t>> pending{{source, 0}};
    for (std::size_t next = 0; next < pending.size(); ++next) {
        const auto [vertex, weight] = pending[next];
        for (const auto& edge : edges) {
            if (edge.source != vertex || weight + edge.weight > bound || reached[edge.target][weight + edge.weight]) {
                continue;
            }
            reached[edge.target][weight + edge.weight] = true;
            pending.push_back({edge.target, weight + edge.weight});
        }
    }
    std::vector<OriginDistanceInterval> result(graph.vertices);
    for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
        for (uint32_t weight = 0; weight <= bound; ++weight) {
            if (!reached[vertex][weight]) { continue; }
            if (!result[vertex].reachable) { result[vertex].reachable = true; result[vertex].minimum = weight; }
            result[vertex].maximum = weight;
        }
        if (result[vertex].maximum && *result[vertex].maximum >= graph.vertices) { result[vertex].maximum.reset(); }
    }
    return result;
}
bool same(const OriginDistanceInterval& a, const OriginDistanceInterval& b)
{
    return a.reachable == b.reachable && (!a.reachable || (a.minimum == b.minimum && a.maximum == b.maximum));
}
bool compare(const ControlOriginGraph& graph, const ControlStorageOrigin& origin, uint64_t& cases)
{
    auto expected = oracle(graph, origin);
    auto result = computeControlOriginDistances(graph, {origin});
    const auto& ownInput = expected[graph.payloads[origin.payload].input];
    const bool invalidReturn = ownInput.reachable && ownInput.minimum == 0;
    if (invalidReturn) {
        if (result.error.empty() || !result.distances.empty()) { return false; }
    } else {
        if (!result.error.empty() || result.distances.size() != 1 || result.distances[0].size() != graph.vertices) {
            llvm::errs() << result.error << "\n";
            return false;
        }
        for (uint32_t vertex = 0; vertex < graph.vertices; ++vertex) {
            if (!same(result.distances[0][vertex], expected[vertex])) { return false; }
        }
        const auto work = uint64_t(graph.vertices) + graph.edges.size() + graph.payloads.size();
        if (result.cost.origins != 1 || result.cost.vertexVisits > 32 * work || result.cost.edgeVisits > 32 * work) {
            return false;
        }
    }
    ++cases;
    return true;
}
bool exhaustive(uint64_t& cases)
{
    const std::array<std::pair<uint32_t, uint32_t>, 4> pairs{{{1, 0}, {1, 2}, {3, 0}, {3, 2}}};
    for (unsigned encoded = 0; encoded < 81; ++encoded) {
        ControlOriginGraph graph{4, {{0, 1}, {2, 3}}, {}};
        auto digits = encoded;
        for (auto [source, target] : pairs) {
            const auto choice = digits % 3;
            digits /= 3;
            if (choice) { graph.edges.push_back({source, target, static_cast<uint8_t>(choice - 1)}); }
        }
        for (uint32_t source = 0; source < 2; ++source) {
            for (unsigned mask = 0; mask < 4; ++mask) {
                ControlStorageOrigin origin{source, mask, {}};
                for (uint32_t killed = 0; killed < 2; ++killed) {
                    if (mask & (1U << killed)) { origin.definiteKillPayloads.push_back(killed); }
                }
                if (!compare(graph, origin, cases)) {
                    llvm::errs() << "control exhaustive case " << encoded << " source " << source << " kill " << mask
                                 << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}
bool cycles(uint64_t& cases)
{
    // Zero-weight SCCs do not make maxima infinite. Positive internal edges
    // do; include zero-weight exits and a disconnected positive self-cycle.
    for (uint8_t positive : {uint8_t(0), uint8_t(1)}) {
        ControlOriginGraph graph{7, {{0, 1}, {2, 3}},
            {{1, 4, 0}, {4, 5, positive}, {5, 4, 0}, {5, 2, 0}, {6, 6, 1}}};
        for (const auto& kills : std::vector<std::vector<uint32_t>>{{}, {0}, {1}, {0, 1}}) {
            if (!compare(graph, {0, 0, kills}, cases) || !compare(graph, {1, 1, kills}, cases)) { return false; }
        }
        graph.edges.push_back({3, 0, 1});
        if (!compare(graph, {0, 2, {}}, cases)) { return false; }
    }
    // Parallel unequal-weight arcs are finite alternatives, not a cycle.
    ControlOriginGraph parallel{4, {{0, 1}, {2, 3}}, {{1, 2, 0}, {1, 2, 1}}};
    if (!compare(parallel, {0, 0, {}}, cases)) { return false; }
    auto finite = computeControlOriginDistances(parallel, {{0, 0, {}}});
    return finite.error.empty() && finite.distances[0][2].minimum == 0 && finite.distances[0][2].maximum == 1;
}
bool overwrites()
{
    ControlOriginGraph graph{4, {{0, 1}, {2, 3}}, {{1, 2, 0}, {3, 0, 1}}};
    auto result = computeControlOriginDistances(graph, {{0, 4, {}}, {0, 5, {1}}, {0, 6, {0}}});
    if (!result.error.empty() || result.distances.size() != 3) { return false; }
    const auto& unbounded = result.distances[0];
    const auto& killedTarget = result.distances[1];
    const auto& killedSource = result.distances[2];
    return unbounded[0].reachable && unbounded[0].minimum == 1 && !unbounded[0].maximum &&
        killedTarget[2].reachable && killedTarget[2].minimum == 0 && killedTarget[2].maximum == 0 &&
        !killedTarget[3].reachable && !killedTarget[0].reachable &&
        killedSource[0].reachable && killedSource[0].minimum == 1 && killedSource[0].maximum == 1 &&
        killedSource[1].minimum == 0 && killedSource[1].maximum == 0;
}
bool rejects(const ControlOriginGraph& graph, llvm::ArrayRef<ControlStorageOrigin> origins)
{
    auto result = computeControlOriginDistances(graph, origins);
    return !result.error.empty() && result.distances.empty();
}
bool validation()
{
    auto empty = computeControlOriginDistances({}, {});
    if (!empty.error.empty() || !empty.distances.empty() || empty.cost.vertices || empty.cost.origins) { return false; }
    const ControlOriginGraph graph{4, {{0, 1}, {2, 3}}, {{1, 2, 0}, {3, 0, 1}}};
    if (!rejects({}, {{0, 0, {}}}) || !rejects(graph, {{2, 0, {}}}) ||
        !rejects(graph, {{0, 0, {2}}}) || !rejects(graph, {{0, 0, {1, 1}}})) { return false; }
    // A late invalid origin must not publish earlier successful rows.
    if (!rejects(graph, {{0, 0, {}}, {2, 0, {}}})) { return false; }
    for (unsigned defect = 0; defect < 6; ++defect) {
        auto bad = graph;
        switch (defect) {
            case 0: bad.edges[0].weight = 2; break;
            case 1: bad.edges[0].target = 4; break;
            case 2: bad.edges.push_back({0, 1, 0}); break;
            case 3: bad.payloads[1].input = 0; break;
            case 4: bad.payloads[0].output = 0; break;
            default: bad.edges.push_back({1, 0, 0}); break;
        }
        if (!rejects(bad, {{0, 0, {}}})) { return false; }
    }
    return true;
}
bool linearCost()
{
    constexpr uint32_t Vertices = 4096;
    ControlOriginGraph graph{Vertices, {{0, 1}, {Vertices - 2, Vertices - 1}}, {}};
    for (uint32_t vertex = 1; vertex < Vertices - 2; ++vertex) {
        graph.edges.push_back({vertex, vertex + 1, 1});
    }
    auto result = computeControlOriginDistances(graph, {{0, 0, {}}});
    if (!result.error.empty() || result.distances.size() != 1) { return false; }
    const auto& last = result.distances[0].back();
    const auto work = uint64_t(Vertices) + graph.edges.size() + graph.payloads.size();
    return last.reachable && last.minimum == Vertices - 3 && last.maximum == Vertices - 3 &&
        result.cost.vertexVisits <= 32 * work && result.cost.edgeVisits <= 32 * work;
}
} // namespace
int runControlOriginDistanceChecks()
{
    uint64_t cases = 0;
    if (!exhaustive(cases) || !cycles(cases) || !overwrites() || !validation() || !linearCost()) {
        llvm::errs() << "control origin distance oracle failed\n";
        return 1;
    }
    llvm::outs() << "control origin distances: " << cases << " exact product-graph cases passed\n";
    return 0;
}
