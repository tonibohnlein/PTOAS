// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Conditional parallel-edge deduplication, sparse pipe frontiers and final-edge tests.
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
#include <algorithm>
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Threshold = GuardedPeriodicThreshold;
constexpr std::size_t noGroup = std::numeric_limits<std::size_t>::max();
struct Edge {
    uint32_t source = 0, target = 0;
    Threshold weight;
    std::size_t group = noGroup;
};
struct Group {
    uint32_t source = 0, target = 0;
    Threshold weight;
    std::vector<std::size_t> records;
};
const char* validateRecords(const RegionExpressions& expressions,
                           llvm::ArrayRef<GuardedPeriodicPayload> payloads,
                           llvm::ArrayRef<GuardedPeriodicRecord> records, uint64_t& maxWeight)
{
    for (const auto& payload : payloads) {
        if (!expressions.isBoolean(payload.presence)) {
            return "guarded quotient presence must be a Boolean expression";
        }
    }
    for (const auto& record : records) {
        if (record.source >= payloads.size() || record.target >= payloads.size() ||
            record.displacement >= expressions.size() || expressions.isBoolean(record.displacement) ||
            !expressions.isBoolean(record.active)) {
            return "guarded quotient has an invalid endpoint or expression";
        }
        const auto enabled = expressions.constantValue(record.active);
        const auto distance = expressions.constantValue(record.displacement);
        if ((!enabled || *enabled) && distance &&
            (*distance > record.maxDisplacement || (*distance == 0 && record.source >= record.target))) {
            return "guarded quotient violates a distance bound or forward-order premise";
        }
        maxWeight = std::max(maxWeight, record.maxDisplacement);
    }
    return nullptr;
}
const char* validate(const RegionExpressions& expressions,
                     llvm::ArrayRef<GuardedPeriodicPayload> payloads,
                     llvm::ArrayRef<GuardedPeriodicRecord> records)
{
    if (!expressions.constructionError().empty()) {
        return "guarded quotient received an invalid expression arena";
    }
    if (payloads.size() > std::numeric_limits<uint32_t>::max() / 2) {
        return "guarded quotient event identity overflow";
    }
    const uint64_t vertices = 2 * uint64_t(payloads.size());
    uint64_t maxWeight = 1; // Every present native chain includes a whole-period wrap.
    if (const char* error = validateRecords(expressions, payloads, records, maxWeight)) {
        return error;
    }
    std::map<uint32_t, uint64_t> counts;
    uint64_t largestPipe = 0;
    for (const auto& payload : payloads) { largestPipe = std::max(largestPipe, ++counts[payload.pipe]); }
    // Shifted frontier scores are h-r+h*distance. V-1 rounds and one
    // final-edge candidate use at most V weights; all intermediates fit u64.
    const auto maximum = std::numeric_limits<uint64_t>::max();
    if (largestPipe && (maxWeight > (maximum - (largestPipe - 1)) / largestPipe / vertices)) {
        return "guarded quotient frontier arithmetic bound exceeds uint64";
    }
    return nullptr;
}
class Builder {
public:
    explicit Builder(GuardedPeriodicQuotient& result)
        : output(result), expressions(*result.expressions), vertices(2 * result.payloads.size()),
          zero(expressions.constant(0)), absent{expressions.boolean(false), zero} {}
    void run();
private:
    GuardedPeriodicQuotient& output;
    RegionExpressions& expressions;
    std::size_t vertices;
    Id zero;
    Threshold absent;
    std::vector<Edge> edges;
    std::vector<Group> groups;
    std::vector<std::vector<std::size_t>> incoming;
    std::vector<Id> winners;
    Threshold minimum(Threshold a, Threshold b);
    Threshold sum(Threshold a, Threshold b);
    Threshold guarded(Id active, Id distance);
    void addEdge(uint32_t source, uint32_t target, Threshold weight, std::size_t group = noGroup);
    void buildNative();
    void buildGroups();
    void selectWinners(const Group& group);
    void buildRanks();
    Id scale(Id value, uint32_t count);
    std::vector<Threshold> frontier(uint32_t row, bool completion, ArrayRef<Threshold> weights);
    void retain(uint32_t row, ArrayRef<Threshold> weights);
    void computeFrontiers();
};
Threshold Builder::guarded(Id active, Id distance)
{
    return {active, expressions.select(active, distance, zero)};
}
Threshold Builder::minimum(Threshold a, Threshold b)
{
    const auto take = expressions.land(b.reachable,
        expressions.lor(expressions.lnot(a.reachable), expressions.lt(b.distance, a.distance)));
    return {expressions.lor(a.reachable, b.reachable), expressions.select(take, b.distance, a.distance)};
}
Threshold Builder::sum(Threshold a, Threshold b)
{
    return guarded(expressions.land(a.reachable, b.reachable), expressions.add(a.distance, b.distance));
}
void Builder::addEdge(uint32_t source, uint32_t target, Threshold weight, std::size_t group)
{
    incoming[target].push_back(edges.size());
    edges.push_back({source, target, weight, group});
}
void Builder::buildNative()
{
    for (const auto& edge : output.nativePrerequisites) {
        auto active = expressions.land(edge.active, expressions.land(
            output.payloads[edge.source].presence, output.payloads[edge.target].presence));
        addEdge(2*edge.source+1, 2*edge.target, guarded(active, edge.displacement));
    }
    std::map<uint32_t, std::vector<uint32_t>> pipes;
    for (uint32_t a = 0; a < output.payloads.size(); ++a) {
        pipes[output.payloads[a].pipe].push_back(a);
        addEdge(2 * a, 2 * a + 1, guarded(output.payloads[a].presence, zero));
    }
    // Potential native positions are latent wires, including absent sites.
    // Contracting absent positions gives exactly the actual native chain:
    // start/start and completion/completion wires only advance reference order;
    // every I->C or supplied/demand edge still requires its endpoint presence.
    // A singleton present site retains the wrap through absent positions.
    // Only present sites are seeded, and query endpoints are presence-gated.
    for (const auto& [pipe, sites] : pipes) {
        for (std::size_t i = 0; i < sites.size(); ++i) {
            const auto from = sites[i], to = sites[(i + 1) % sites.size()];
            const auto weight = guarded(expressions.boolean(true), expressions.constant(i + 1 == sites.size()));
            addEdge(2 * from, 2 * to, weight);
            addEdge(2 * from + 1, 2 * to + 1, weight);
        }
    }
}
void Builder::selectWinners(const Group& group)
{
    auto selected = expressions.boolean(false);
    for (auto id : group.records) {
        const auto& record = output.records[id];
        auto active = expressions.land(record.active, expressions.land(
            output.payloads[record.source].presence, output.payloads[record.target].presence));
        auto attains = expressions.select(active, expressions.eq(record.displacement, group.weight.distance),
                                          expressions.boolean(false));
        winners[id] = expressions.land(attains, expressions.lnot(selected));
        selected = expressions.lor(selected, attains);
    }
}
void Builder::buildGroups()
{
    std::map<std::pair<uint32_t, uint32_t>, std::size_t> index;
    for (std::size_t id = 0; id < output.records.size(); ++id) {
        const auto& record = output.records[id];
        auto [entry, added] = index.emplace(std::make_pair(record.source, record.target), groups.size());
        if (added) { groups.push_back({record.source, record.target, absent, {}}); }
        auto& group = groups[entry->second];
        group.records.push_back(id);
        auto active = expressions.land(record.active, expressions.land(
            output.payloads[record.source].presence, output.payloads[record.target].presence));
        group.weight = minimum(group.weight, guarded(active, record.displacement));
    }
    for (std::size_t id = 0; id < groups.size(); ++id) {
        const auto& group = groups[id];
        selectWinners(group);
        addEdge(2 * group.source + 1, 2 * group.target, group.weight, id);
    }
}
void Builder::buildRanks()
{
    std::map<uint32_t, uint32_t> rows;
    for (const auto& payload : output.payloads) {
        auto [found, added] = rows.emplace(payload.pipe, rows.size());
        if (added) { output.frontiers.push_back({payload.pipe, 0, {}, {}}); }
        output.sourceRows.push_back(found->second);
        output.localRanks.push_back(++output.frontiers[found->second].count);
    }
}
Id Builder::scale(Id value, uint32_t count)
{
    auto result = zero;
    while (count) {
        if (count & 1U) { result = expressions.add(result, value); ++output.scalingAdditions; }
        count >>= 1U;
        if (count) { value = expressions.add(value, value); ++output.scalingAdditions; }
    }
    return result;
}
std::vector<Threshold> Builder::frontier(uint32_t row, bool completion, ArrayRef<Threshold> weights)
{
    std::vector<Threshold> scores(vertices, absent);
    const auto count = output.frontiers[row].count;
    for (uint32_t a = 0; a < output.payloads.size(); ++a) {
        if (output.sourceRows[a] == row) {
            scores[2 * a + unsigned(completion)] = guarded(output.payloads[a].presence,
                expressions.constant(count - output.localRanks[a]));
        }
    }
    // Synchronous Bellman-Ford circuits: a shortest nonnegative-weight walk
    // has a simple representative. Absent native positions remain wire nodes.
    for (std::size_t round = 1; round < vertices; ++round) {
        auto previous = scores;
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            scores[edge.target] = minimum(scores[edge.target], sum(previous[edge.source], weights[i]));
            ++output.relaxationCandidates;
        }
    }
    return scores;
}
void Builder::retain(uint32_t row, ArrayRef<Threshold> weights)
{
    // Native order makes ancestors on this pipe a prefix, including gaps for
    // omitted sites. Excluding one FINAL edge, rho<=h*d+h-r exactly says the
    // candidate source is in that prefix. A prefix cannot improperly use the
    // excluded edge: that would require a nonempty zero-displacement cycle.
    // Prefix/suffix minima exclude edge identities, preserving equal scores.
    const auto& frontier = output.frontiers[row];
    for (uint32_t target = 0; target < output.payloads.size(); ++target) {
        const auto& list = incoming[2 * target];
        std::vector<Threshold> values, prefix{absent};
        values.reserve(list.size()); prefix.reserve(list.size() + 1);
        for (auto id : list) {
            values.push_back(sum(frontier.completions[edges[id].source], weights[id]));
            prefix.push_back(minimum(prefix.back(), values.back()));
            ++output.exclusionCandidates;
        }
        auto suffix = absent;
        for (std::size_t i = list.size(); i > 0; --i) {
            const auto id = list[i - 1];
            const auto& edge = edges[id];
            if (edge.group != noGroup && output.sourceRows[groups[edge.group].source] == row) {
                const auto& group = groups[edge.group];
                auto alternative = minimum(prefix[i - 1], suffix);
                auto limit = expressions.add(weights[id].distance,
                    expressions.constant(frontier.count - output.localRanks[group.source]));
                auto redundant = expressions.land(alternative.reachable,
                    expressions.le(alternative.distance, limit));
                for (auto record : group.records) {
                    output.retained[record] = expressions.land(winners[record], expressions.lnot(redundant));
                }
            }
            suffix = minimum(values[i - 1], suffix);
        }
    }
}
void Builder::computeFrontiers()
{
    for (uint32_t row = 0; row < output.frontiers.size(); ++row) {
        std::vector<Threshold> weights;
        weights.reserve(edges.size());
        for (const auto& edge : edges) {
            weights.push_back(guarded(edge.weight.reachable, scale(edge.weight.distance, output.frontiers[row].count)));
        }
        output.frontiers[row].starts = frontier(row, false, weights);
        output.frontiers[row].completions = frontier(row, true, weights);
        retain(row, weights);
    }
}
void Builder::run()
{
    output.retained.assign(output.records.size(), expressions.boolean(false));
    winners.resize(output.records.size(), expressions.boolean(false));
    incoming.resize(vertices);
    buildRanks(); buildNative(); buildGroups(); computeFrontiers();
    output.graphEdges = edges.size();
}
} // namespace
GuardedPeriodicQuotient analyzeGuardedPeriodicQuotient(
    std::shared_ptr<RegionExpressions> expressions,
    llvm::ArrayRef<GuardedPeriodicPayload> payloads, llvm::ArrayRef<GuardedPeriodicRecord> records,
    llvm::ArrayRef<GuardedPeriodicRecord> nativePrerequisites)
{
    GuardedPeriodicQuotient result;
    result.expressions = std::move(expressions);
    if (!result.expressions) {
        result.error = "guarded quotient requires an expression arena";
        return result;
    }
    std::vector<GuardedPeriodicRecord> all(records.begin(), records.end());
    all.insert(all.end(), nativePrerequisites.begin(), nativePrerequisites.end());
    if (const char* error = validate(*result.expressions, payloads, all)) {
        result.error = error;
        return result;
    }
    result.payloads.assign(payloads.begin(), payloads.end());
    result.records.assign(records.begin(), records.end());
    result.nativePrerequisites.assign(nativePrerequisites.begin(), nativePrerequisites.end());
    Builder(result).run();
    if (!result.expressions->constructionError().empty()) {
        result.error = result.expressions->constructionError();
        result.retained.clear();
        result.frontiers.clear();
        result.sourceRows.clear(); result.localRanks.clear();
    }
    return result;
}
std::optional<GuardedPeriodicThreshold> GuardedPeriodicQuotient::eventThreshold(
    PeriodicEvent source, PeriodicEvent target) const
{
    const auto validKind = [](PeriodicEventKind kind) {
        return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
    };
    if (!error.empty() || !expressions || source.type >= payloads.size() || target.type >= payloads.size() ||
        !validKind(source.kind) || !validKind(target.kind)) {
        return std::nullopt;
    }
    if (source.type >= sourceRows.size() || source.type >= localRanks.size() ||
        sourceRows[source.type] >= frontiers.size()) { return std::nullopt; }
    auto& e = *expressions;
    const auto& row = frontiers[sourceRows[source.type]];
    const auto& scores = source.kind == PeriodicEventKind::Start ? row.starts : row.completions;
    const auto vertex = 2 * std::size_t(target.type) + unsigned(target.kind == PeriodicEventKind::Completion);
    if (!row.count || !localRanks[source.type] || localRanks[source.type] > row.count ||
        vertex >= scores.size()) { return std::nullopt; }
    const auto score = scores[vertex];
    auto active = e.land(score.reachable, e.land(payloads[source.type].presence, payloads[target.type].presence));
    // The last ancestor rank at target period j is h*j+h-rho. For a present
    // source of rank r at period i, inclusion is h*(j-i)>=rho+r-h, hence the
    // nonnegative threshold floor((rho+r-1)/h). Fixed potential ranks also
    // handle gaps from omitted sites. Split the sum to avoid u64 overflow.
    auto count = e.constant(row.count), zero = e.constant(0);
    auto carry = e.le(count, e.add(e.rem(score.distance, count), e.constant(localRanks[source.type] - 1)));
    auto distance = e.add(e.div(score.distance, count), e.select(carry, e.constant(1), zero));
    if (!e.constructionError().empty()) { return std::nullopt; }
    return Threshold{active, e.select(active, distance, zero)};
}
} // namespace mlir::pto::frontiersynch
