// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Conditional parallel-edge deduplication, shared closure and final-edge tests.
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
#include <algorithm>
#include <limits>
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
    const auto capacity = std::vector<Threshold>().max_size();
    if (vertices && vertices > capacity / vertices) {
        return "guarded quotient matrix size overflow";
    }
    uint64_t maxWeight = 1; // Every present native chain includes a whole-period wrap.
    if (const char* error = validateRecords(expressions, payloads, records, maxWeight)) {
        return error;
    }
    // Shortest paths have simple representatives. Floyd candidates add two
    // paths of at most V-1 edges; final-edge candidates have at most V edges.
    const uint64_t terms = vertices ? std::max(vertices, 2 * (vertices - 1)) : 0;
    if (terms && maxWeight > std::numeric_limits<uint64_t>::max() / terms) {
        return "guarded quotient distance arithmetic bound exceeds uint64";
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
    Threshold& cell(uint32_t source, uint32_t target);
    void addEdge(uint32_t source, uint32_t target, Threshold weight, std::size_t group = noGroup);
    void buildNative();
    void buildGroups();
    void selectWinners(const Group& group);
    void close();
    void retain(uint32_t source, uint32_t target);
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
Threshold& Builder::cell(uint32_t source, uint32_t target)
{
    return output.thresholds[std::size_t(source) * vertices + target];
}
void Builder::addEdge(uint32_t source, uint32_t target, Threshold weight, std::size_t group)
{
    incoming[target].push_back(edges.size());
    edges.push_back({source, target, weight, group});
    cell(source, target) = minimum(cell(source, target), weight);
}
void Builder::buildNative()
{
    for (const auto& edge : output.nativePrerequisites) {
        auto active = expressions.land(edge.active, expressions.land(
            output.payloads[edge.source].presence, output.payloads[edge.target].presence));
        addEdge(2*edge.source+1, 2*edge.target, guarded(active, edge.displacement));
    }
    for (uint32_t a = 0; a < output.payloads.size(); ++a) {
        const auto& payload = output.payloads[a];
        const auto identity = guarded(payload.presence, zero);
        cell(2 * a, 2 * a) = identity;
        cell(2 * a + 1, 2 * a + 1) = identity;
        addEdge(2 * a, 2 * a + 1, identity);
        // Pairwise native edges preserve the actual same-pipe chain when any
        // intermediate site is absent. a==b supplies the singleton self-wrap.
        for (uint32_t b = 0; b < output.payloads.size(); ++b) {
            if (payload.pipe != output.payloads[b].pipe) { continue; }
            auto active = expressions.land(payload.presence, output.payloads[b].presence);
            auto weight = guarded(active, expressions.constant(a < b ? 0 : 1));
            addEdge(2 * a, 2 * b, weight);
            addEdge(2 * a + 1, 2 * b + 1, weight);
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
    const auto count = output.payloads.size();
    std::vector<std::size_t> index(count * count, noGroup);
    for (std::size_t id = 0; id < output.records.size(); ++id) {
        const auto& record = output.records[id];
        auto& entry = index[std::size_t(record.source) * count + record.target];
        if (entry == noGroup) {
            entry = groups.size();
            groups.push_back({record.source, record.target, absent, {}});
        }
        auto& group = groups[entry];
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
void Builder::close()
{
    for (uint32_t via = 0; via < vertices; ++via) {
        // Explicit previous-stage storage keeps the recurrence independent of
        // host update order while expression nodes remain shared across stages.
        auto previous = output.thresholds;
        for (uint32_t source = 0; source < vertices; ++source) {
            for (uint32_t target = 0; target < vertices; ++target) {
                // Present vertices already have zero reflexive distance; all
                // weights are nonnegative. These candidates cannot improve it
                // or a path by adding a reflexive segment at either endpoint.
                if (source == via || target == via || source == target) { continue; }
                auto candidate = sum(previous[std::size_t(source) * vertices + via],
                                     previous[std::size_t(via) * vertices + target]);
                cell(source, target) = minimum(previous[std::size_t(source) * vertices + target], candidate);
            }
        }
    }
}
void Builder::retain(uint32_t source, uint32_t target)
{
    const auto& list = incoming[target];
    std::vector<Threshold> values, prefix{absent};
    values.reserve(list.size());
    prefix.reserve(list.size() + 1);
    for (auto id : list) {
        const auto& edge = edges[id];
        values.push_back(sum(cell(2 * source + 1, edge.source), edge.weight));
        prefix.push_back(minimum(prefix.back(), values.back()));
    }
    auto suffix = absent;
    for (std::size_t i = list.size(); i > 0; --i) {
        const auto& edge = edges[list[i - 1]];
        if (edge.group != noGroup && groups[edge.group].source == source) {
            auto alternative = minimum(prefix[i - 1], suffix);
            auto redundant = expressions.land(alternative.reachable,
                expressions.le(alternative.distance, edge.weight.distance));
            for (auto record : groups[edge.group].records) {
                output.retained[record] = expressions.land(winners[record], expressions.lnot(redundant));
            }
        }
        suffix = minimum(values[i - 1], suffix);
    }
}
void Builder::run()
{
    output.thresholds.assign(vertices * vertices, absent);
    output.retained.assign(output.records.size(), expressions.boolean(false));
    winners.resize(output.records.size(), expressions.boolean(false));
    incoming.resize(vertices);
    buildNative();
    buildGroups();
    close();
    for (uint32_t source = 0; source < output.payloads.size(); ++source) {
        for (uint32_t target = 0; target < output.payloads.size(); ++target) {
            retain(source, 2 * target);
        }
    }
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
        result.thresholds.clear();
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
    const auto vertex = [](PeriodicEvent event) {
        return 2 * std::size_t(event.type) + (event.kind == PeriodicEventKind::Completion ? 1 : 0);
    };
    const auto index = vertex(source) * (2 * payloads.size()) + vertex(target);
    return index < thresholds.size() ? std::optional<Threshold>(thresholds[index]) : std::nullopt;
}
} // namespace mlir::pto::frontiersynch
