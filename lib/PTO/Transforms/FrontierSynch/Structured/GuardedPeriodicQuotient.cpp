// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Conditional parallel-edge deduplication, sparse pipe frontiers and final-edge tests.
#include "PTO/Transforms/FrontierSynch/Structured/GuardedPeriodicQuotient.h"
#include "PTO/Transforms/FrontierSynch/Structured/GuardedPeriodicWork.h"
#include <algorithm>
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Threshold = GuardedPeriodicThreshold;
using Score = GuardedPeriodicScore;
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
                     llvm::ArrayRef<GuardedPeriodicRecord> records, bool coreChains)
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
    // Period components count edge displacements; residues stay below the
    // source-pipe length. Reserve wrapped scan candidates and the final query
    // carry without multiplying distances by that length.
    const auto maximum = std::numeric_limits<uint64_t>::max();
    const auto pathWeights = vertices + (coreChains ? 2 : 0);
    const bool exceedsBound = vertices && maxWeight > (maximum - 1) / pathWeights;
    if (exceedsBound) {
        return "guarded quotient frontier arithmetic bound exceeds uint64";
    }
    return nullptr;
}
class Builder {
public:
    explicit Builder(GuardedPeriodicQuotient& result)
        : output(result), expressions(*result.expressions), vertices(2 * result.payloads.size()),
          zero(expressions.constant(0)), absent{expressions.boolean(false), zero},
          noScore{absent.reachable, zero, zero} {}
    void run();
private:
    GuardedPeriodicQuotient& output;
    RegionExpressions& expressions;
    std::size_t vertices;
    Id zero;
    Threshold absent;
    Score noScore;
    std::vector<Edge> edges;
    std::vector<Group> groups;
    std::vector<std::vector<std::size_t>> incoming;
    std::vector<Id> winners;
    std::vector<std::vector<uint32_t>> pipeSites;
    Threshold minimum(Threshold a, Threshold b);
    Score minimum(Score a, Score b);
    Score sum(Score a, Threshold b);
    Score guardedScore(Id active, Id period, Id residue);
    Id lessEqual(Score a, Score b);
    Threshold guarded(Id active, Id distance);
    void addEdge(uint32_t source, uint32_t target, Threshold weight, std::size_t group = noGroup);
    void buildNative();
    void buildGroups();
    void selectWinners(const Group& group);
    void buildRanks();
    std::vector<Score> frontier(uint32_t row, bool completion, ArrayRef<Threshold> weights);
    std::vector<Score> scanNative(ArrayRef<Score> scores,
                                      bool completion, bool strict);
    std::vector<Score> closeNative(ArrayRef<Score> scores);
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
Score Builder::guardedScore(Id active, Id period, Id residue)
{
    return {active, expressions.select(active, period, zero), expressions.select(active, residue, zero)};
}
Id Builder::lessEqual(Score a, Score b)
{
    return expressions.lor(expressions.lt(a.period, b.period),
        expressions.land(expressions.eq(a.period, b.period), expressions.le(a.residue, b.residue)));
}
Score Builder::minimum(Score a, Score b)
{
    const auto take = expressions.land(b.reachable,
        expressions.lor(expressions.lnot(a.reachable), lessEqual(b, a)));
    return {expressions.lor(a.reachable, b.reachable),
        expressions.select(take, b.period, a.period), expressions.select(take, b.residue, a.residue)};
}
Score Builder::sum(Score a, Threshold b)
{
    return guardedScore(expressions.land(a.reachable, b.reachable),
        expressions.add(a.period, b.distance), a.residue);
}
void Builder::addEdge(uint32_t source, uint32_t target, Threshold weight, std::size_t group)
{
    incoming[target].push_back(edges.size());
    edges.push_back({source, target, weight, group});
}
void Builder::buildNative()
{
    if (output.nativePrerequisites.empty()) { return; }
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
        if (added) { output.frontiers.push_back({payload.pipe, 0, {}, {}}); pipeSites.emplace_back(); }
        pipeSites[found->second].push_back(output.sourceRows.size());
        output.sourceRows.push_back(found->second);
        output.localRanks.push_back(++output.frontiers[found->second].count);
    }
}
// All native paths in one scan pair. Strict start scans supply alternative
// final native edges: earlier starts without a wrap, current/later with one.
std::vector<Score> Builder::scanNative(ArrayRef<Score> scores,
                                          bool completion, bool strict)
{
    std::vector<Score> result(output.payloads.size(), noScore);
    const auto wrap = guarded(expressions.boolean(true), expressions.constant(1));
    auto input = [&](uint32_t site) {
        const auto value = completion ? minimum(scores[2 * site], scores[2 * site + 1]) : scores[2 * site];
        return guardedScore(expressions.land(output.payloads[site].presence, value.reachable),
                            value.period, value.residue);
    };
    for (const auto& sites : pipeSites) {
        auto prefix = noScore;
        for (auto site : sites) {
            const auto value = input(site);
            result[site] = strict ? prefix : minimum(prefix, value);
            prefix = minimum(prefix, value);
            ++output.nativeScanVisits;
        }
        auto suffix = noScore;
        for (auto it = sites.rbegin(); it != sites.rend(); ++it) {
            const auto value = input(*it);
            if (strict) { suffix = minimum(suffix, value); }
            result[*it] = minimum(result[*it], sum(suffix, wrap));
            if (!strict) { suffix = minimum(suffix, value); }
            ++output.nativeScanVisits;
        }
    }
    return result;
}
std::vector<Score> Builder::closeNative(ArrayRef<Score> scores)
{
    auto starts = scanNative(scores, false, false);
    auto completions = scanNative(scores, true, false);
    std::vector<Score> result(vertices, noScore);
    for (std::size_t site = 0; site < output.payloads.size(); ++site) {
        const auto present = output.payloads[site].presence;
        result[2 * site] = guardedScore(expressions.land(present, starts[site].reachable),
                                        starts[site].period, starts[site].residue);
        result[2 * site + 1] = guardedScore(expressions.land(present, completions[site].reachable),
                                            completions[site].period, completions[site].residue);
    }
    return result;
}
std::vector<Score> Builder::frontier(uint32_t row, bool completion, ArrayRef<Threshold> weights)
{
    std::vector<Score> scores(vertices, noScore);
    const auto count = output.frontiers[row].count;
    for (uint32_t a = 0; a < output.payloads.size(); ++a) {
        if (output.sourceRows[a] == row) {
            scores[2 * a + unsigned(completion)] = guardedScore(output.payloads[a].presence, zero,
                expressions.constant(count - output.localRanks[a]));
        }
    }
    const auto rounds = *guardedPeriodicRounds(
        vertices, output.frontiers.size(), output.nativePrerequisites.empty());
    if (output.nativePrerequisites.empty()) {
        scores = closeNative(scores);
        for (uint64_t round = 0; round < rounds; ++round) {
            const auto previous = scores;
            for (std::size_t i = 0; i < edges.size(); ++i) {
                const auto& edge = edges[i];
                scores[edge.target] = minimum(scores[edge.target], sum(previous[edge.source], weights[i]));
                ++output.relaxationCandidates;
            }
            scores = closeNative(scores);
            ++output.demandRounds;
        }
        return scores;
    }
    // Synchronous Bellman-Ford circuits: a shortest nonnegative-weight walk
    // has a simple representative. Absent native positions remain wire nodes.
    for (uint64_t round = 0; round < rounds; ++round) {
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
    const auto native = output.nativePrerequisites.empty() ?
        scanNative(frontier.completions, false, true) : std::vector<Score>();
    for (uint32_t target = 0; target < output.payloads.size(); ++target) {
        const auto& list = incoming[2 * target];
        std::vector<Score> values, prefix{native.empty() ? noScore : native[target]};
        values.reserve(list.size()); prefix.reserve(list.size() + 1);
        for (auto id : list) {
            values.push_back(sum(frontier.completions[edges[id].source], weights[id]));
            prefix.push_back(minimum(prefix.back(), values.back()));
            ++output.exclusionCandidates;
        }
        auto suffix = noScore;
        for (std::size_t i = list.size(); i > 0; --i) {
            const auto id = list[i - 1];
            const auto& edge = edges[id];
            if (edge.group != noGroup && output.sourceRows[groups[edge.group].source] == row) {
                const auto& group = groups[edge.group];
                auto alternative = minimum(prefix[i - 1], suffix);
                const Score limit{weights[id].reachable, weights[id].distance,
                    expressions.constant(frontier.count - output.localRanks[group.source])};
                auto redundant = expressions.land(alternative.reachable,
                    lessEqual(alternative, limit));
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
            weights.push_back(edge.weight);
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
    if (const char* error = validate(*result.expressions, payloads, all, nativePrerequisites.empty())) {
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
    // ceil((rho-(h-r))/h) is period plus a carry exactly when the
    // residue exceeds h-r. Potential ranks preserve absent-site gaps.
    auto zero = e.constant(0);
    auto carry = e.lt(e.constant(row.count - localRanks[source.type]), score.residue);
    auto distance = e.add(score.period, e.select(carry, e.constant(1), zero));
    if (!e.constructionError().empty()) { return std::nullopt; }
    return Threshold{active, e.select(active, distance, zero)};
}
} // namespace mlir::pto::frontiersynch
