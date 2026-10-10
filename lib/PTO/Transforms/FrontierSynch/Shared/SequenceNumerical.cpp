// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical specialization only; symbolic guards/order keep the general path.
#include "SequenceAnalysisInternal.h"
#include "ChainInterfaceInternal.h"
#include "llvm/ADT/ScopeExit.h"
namespace mlir::pto::frontiersynch {
namespace {
using EventIdentity = std::tuple<uint32_t, uint64_t, PeriodicEventKind, std::vector<uint64_t>>;
using ChainKey = std::pair<uint32_t, PeriodicEventKind>;
std::shared_ptr<const RegionalNumericalInterface> leafInterface(const RegionalAnalysis& region,
    const std::vector<RegionalEvent>& events, uint64_t& queries, uint64_t& operations)
{
    if (region.referenceBefore) { return {}; }
    auto result = std::make_shared<RegionalNumericalInterface>();
    result->events = events;
    std::map<ChainKey, std::vector<uint32_t>> lists;
    for (uint32_t id = 0; id < events.size(); ++id) {
        const auto& event = events[id];
        auto* phase = region.anchors[event.type].phase;
        if (!phase) { return {}; }
        lists[{static_cast<uint32_t>(phase->kPipeValue), event.kind}].push_back(id);
    }
    std::vector<std::vector<uint32_t>> chains;
    for (auto& [key, list] : lists) {
        std::stable_sort(list.begin(), list.end(), [&](uint32_t a, uint32_t b) {
            const auto& x = events[a]; const auto& y = events[b];
            return std::make_pair(*region.expressions->constantValue(x.ordinal), x.type) <
                   std::make_pair(*region.expressions->constantValue(y.ordinal), y.type);
        });
        for (std::size_t i = 1; i < list.size(); ++i) {
            const auto& a = events[list[i - 1]]; const auto& b = events[list[i]];
            if (a.type == b.type && a.ordinal == b.ordinal) { return {}; }
        }
        result->chainKeys.push_back(key); chains.push_back(std::move(list));
    }
    auto index = buildNumericalChainInterface(std::move(chains), [&](uint32_t a, uint32_t b) -> std::optional<bool> {
        auto answer = regionalReachability(region, events[a], events[b]);
        auto value = answer ? region.expressions->constantValue(*answer) : std::nullopt;
        return value ? std::optional<bool>(*value != 0) : std::nullopt;
    });
    accumulateCost(queries, index.queries); accumulateCost(operations, index.operations);
    if (!index.error.empty()) { return {}; }
    result->index = std::make_shared<const NumericalChainInterface>(std::move(index));
    result->query = [region](RegionalEvent a, RegionalEvent b, NumericalChainQueryCost& cost) -> std::optional<bool> {
        if (!validRegionalEvent(region, a) || !validRegionalEvent(region, b)) { return std::nullopt; }
        ++cost.leafQueries;
        auto answer = region.reachability(a, b);
        auto value = answer ? region.expressions->constantValue(*answer) : std::nullopt;
        return value ? std::optional<bool>(*value != 0) : std::nullopt;
    };
    result->thresholds = [region, events, index = result->index](RegionalEvent event, bool reverse,
        NumericalChainQueryCost& cost) -> std::optional<std::vector<uint32_t>> {
        if (!validRegionalEvent(region, event)) { return std::nullopt; }
        return numericalLeafThresholds(*index, [&](uint32_t port, bool back) -> std::optional<bool> {
            auto answer = back ? regionalReachability(region, events[port], event) :
                                 regionalReachability(region, event, events[port]);
            auto value = answer ? region.expressions->constantValue(*answer) : std::nullopt;
            return value ? std::optional<bool>(*value != 0) : std::nullopt;
        }, reverse, cost);
    };
    return result;
}
bool childSelection(const RegionalNumericalInterface& index, const std::vector<RegionalEvent>& events,
                    std::vector<uint32_t>& selected, RegionExpressions& expressions)
{
    if (!index.index || !index.index->error.empty() || index.events.size() != index.index->chain.size() ||
        index.chainKeys.size() != index.index->chains.size() || !index.thresholds || !index.query) { return false; }
    auto identity = [&](const RegionalEvent& event) -> std::optional<EventIdentity> {
        auto ordinal = expressions.constantValue(event.ordinal);
        if (!ordinal) { return {}; }
        std::vector<uint64_t> visits;
        for (auto coordinate : event.visits) {
            auto value = expressions.constantValue(coordinate);
            if (!value) { return {}; }
            visits.push_back(*value);
        }
        return EventIdentity{event.type, *ordinal, event.kind, std::move(visits)};
    };
    std::map<EventIdentity, uint32_t> ids;
    for (uint32_t id = 0; id < index.events.size(); ++id) {
        auto key = identity(index.events[id]);
        if (!key || !ids.emplace(*key, id).second) { return false; }
    }
    for (const auto& event : events) {
        auto key = identity(event);
        if (!key) { return false; }
        auto found = ids.find(*key);
        if (found == ids.end()) { return false; }
        selected.push_back(found->second);
    }
    return true;
}
} // namespace
// The flat regional children and original endpoint IDs remain authoritative.
// This separate balanced tree owns only numerical indices and query routing.
struct SequenceNumericalNode {
    uint32_t begin = 0, end = 0;
    std::shared_ptr<const RegionalNumericalInterface> leaf;
    std::array<std::shared_ptr<SequenceNumericalNode>, 2> children;
    std::shared_ptr<NumericalChainMerge> merge;
    std::shared_ptr<const NumericalChainInterface> index;
    std::vector<ChainKey> keys;
    // Original sequence event ID -> this node's selected/index event ID.
    std::map<uint32_t, uint32_t> events;
    std::optional<std::vector<uint32_t>> thresholds(
        SequenceEvent event, bool reverse, NumericalChainQueryCost& cost) const
    {
        if (event.child < begin || event.child >= end) { return {}; }
        RegionalEvent local{event.type, event.ordinal, event.kind, event.visits};
        if (leaf) { return leaf->thresholds(local, reverse, cost); }
        const auto side = event.child < children[0]->end ? 0U : 1U;
        auto selected = children[side]->thresholds(event, reverse, cost);
        return selected ? merge->propagate(side, *selected, reverse, cost) : std::nullopt;
    }
    std::optional<bool> query(SequenceEvent a, SequenceEvent b, NumericalChainQueryCost& cost) const
    {
        if (a.child < begin || b.child >= end || a.child > b.child) { return false; }
        if (leaf) {
            return leaf->query({a.type, a.ordinal, a.kind, a.visits},
                               {b.type, b.ordinal, b.kind, b.visits}, cost);
        }
        const auto left = a.child < children[0]->end ? 0U : 1U;
        const auto right = b.child < children[0]->end ? 0U : 1U;
        if (left == right) { return children[left]->query(a, b, cost); }
        auto from = children[0]->thresholds(a, false, cost);
        auto to = children[1]->thresholds(b, true, cost);
        return from && to ? merge->crosses(*from, *to, cost) : std::nullopt;
    }
};
bool SequenceAnalysisState::preferNumericalCrossings()
{
    AnalysisCostEstimate numeric, symbolic;
    EstimatedCount leafWork = 0, leafSize = 0, callbackCost = 0;
    SmallVector<uint64_t> events(children.size(), 0);
    std::set<ChainKey> chains;
    for (const auto& port : ports) {
        if (port.child >= children.size() || port.type >= children[port.child].anchors.size()) { continue; }
        accumulateCost(events[port.child], 2);
        auto* phase = children[port.child].anchors[port.type].phase;
        if (!phase) { continue; }
        const auto pipe = static_cast<uint32_t>(phase->kPipeValue);
        chains.emplace(pipe, PeriodicEventKind::Start);
        chains.emplace(pipe, PeriodicEventKind::Completion);
    }
    for (std::size_t child = 0; child < children.size(); ++child) {
        const auto& region = children[child].regional;
        // Existing indices have constant-time local pair queries. A callback
        // without a circuit-size description has unknown construction cost.
        EstimatedCount queryCost = region.numerical ? EstimatedCount(1) :
            (region.cost.expressionNodes ? EstimatedCount(region.cost.expressionNodes) : std::nullopt);
        if (events[child]) {
            if (!callbackCost || !queryCost) { callbackCost = std::nullopt; }
            else { callbackCost = std::max(*callbackCost, *queryCost); }
        }
        if (!region.numerical) {
            const auto pairs = estimatedMultiply(events[child], events[child]);
            leafWork = estimatedAdd(leafWork, estimatedMultiply(pairs, queryCost));
            leafSize = estimatedAdd(leafSize, pairs);
        }
    }
    EstimatedCount links = 0;
    for (const auto& [child, entries] : incoming) { links = estimatedAdd(links, entries.size()); }
    uint64_t levels = 0;
    for (auto count = children.size(); count > 1; count = count / 2 + count % 2) { ++levels; }
    const auto vertices = estimatedMultiply(ports.size(), 2);
    const auto chainPairs = estimatedMultiply(chains.size(), chains.size());
    const auto routing = estimatedMultiply(estimatedAdd(vertices, links), levels);
    numeric.generatorPieces = links; numeric.ports = vertices;
    numeric.numericalWindow = vertices; numeric.circuitNodes = numeric.relationConversion = 0;
    numeric.work = estimatedAdd(leafWork, estimatedMultiply(routing, chainPairs));
    numeric.representation = estimatedAdd(leafSize,
        estimatedMultiply(estimatedMultiply(vertices, chains.size()), levels));
    const auto folding = estimatedMultiply(estimatedMultiply(ports.size(), ports.size()), 2);
    const auto tests = estimatedMultiply(crossings.size(), links);
    const auto alternateLinks = links && *links ? EstimatedCount(*links - 1) : links;
    const auto callbacks = estimatedMultiply(estimatedMultiply(crossings.size(), alternateLinks), 2);
    symbolic.generatorPieces = links; symbolic.ports = vertices;
    symbolic.numericalWindow = symbolic.relationConversion = 0;
    symbolic.circuitNodes = estimatedAdd(folding,
        estimatedAdd(estimatedMultiply(vertices, estimatedAdd(links, 1)), estimatedMultiply(tests, 2)));
    symbolic.work = estimatedAdd(symbolic.circuitNodes, estimatedMultiply(callbacks, callbackCost));
    symbolic.representation = symbolic.circuitNodes;
    crossingMethods = {{{0, 0, "sequence-crossings-numerical", numeric},
                        {0, 0, "sequence-crossings-symbolic", symbolic}}};
    // Work first, representation second, existing numerical order on ties.
    return !estimatedCostLess(symbolic, numeric);
}
bool SequenceAnalysisState::numericalCrossingReduction()
{
    if (children.size() < 2 || children.size() > UINT32_MAX ||
        ports.size() > UINT32_MAX / 2 || !portChoices.empty()) { return false; }
    // Qualify all guards before constructing any leaf or merge index.
    for (const auto& edge : crossings) {
        if (!expressions.constantValue(edge.guard)) { return false; }
    }
    using Pair = std::pair<uint32_t, uint32_t>;
    std::set<Pair> links, retained;
    for (const auto& [target, entries] : incoming) {
        for (const auto& link : entries) {
            auto active = expressions.constantValue(link.guard);
            if (!active || link.source >= 2 * ports.size() || link.target >= 2 * ports.size() ||
                ports[link.source / 2].child >= ports[link.target / 2].child) { return false; }
            if (*active) { links.emplace(link.source, link.target); }
        }
    }
    std::vector<std::vector<RegionalEvent>> events(children.size());
    std::vector<std::vector<uint32_t>> originals(children.size());
    for (uint32_t p = 0; p < ports.size(); ++p) {
        const auto& port = ports[p];
        if (port.child >= children.size() || !expressions.constantValue(port.ordinal) ||
            (!port.visits.empty() && !children[port.child].regional.numerical) ||
            (!children[port.child].regional.numerical && expressions.constantValue(present(p)) != 1)) { return false; }
        for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
            originals[port.child].push_back(2 * p + (kind == PeriodicEventKind::Completion));
            events[port.child].push_back(port.event(kind));
        }
    }
    uint64_t leafQueries = 0, operations = 0, reused = 0, merges = 0;
    auto charge = llvm::make_scope_exit([&]() {
        accumulateCost(costs.numericalLeafQueries, leafQueries);
        accumulateCost(costs.numericalIndexOperations, operations);
        accumulateCost(costs.numericalReusedChildren, reused);
        accumulateCost(costs.numericalMerges, merges);
    });
    std::vector<std::shared_ptr<SequenceNumericalNode>> leaves;
    for (uint32_t child = 0; child < children.size(); ++child) {
        auto node = std::make_shared<SequenceNumericalNode>();
        node->begin = child; node->end = child + 1;
        node->leaf = children[child].regional.numerical;
        if (node->leaf) { accumulateCost(reused, 1); }
        else { node->leaf = leafInterface(children[child].regional, events[child], leafQueries, operations); }
        std::vector<uint32_t> selection;
        if (!node->leaf || !childSelection(*node->leaf, events[child], selection, expressions)) { return false; }
        // Distinct syntactic port IDs must not duplicate one semantic child
        // index entry. General guarded identity handling remains the fallback.
        if (std::set<uint32_t>(selection.begin(), selection.end()).size() != selection.size()) { return false; }
        node->index = node->leaf->index; node->keys = node->leaf->chainKeys;
        for (std::size_t i = 0; i < selection.size(); ++i) { node->events.emplace(originals[child][i], selection[i]); }
        leaves.push_back(std::move(node));
    }
    std::function<std::shared_ptr<SequenceNumericalNode>(uint32_t, uint32_t, const std::vector<Pair>&)> build =
        [&](uint32_t begin, uint32_t end, const std::vector<Pair>& candidates)
            -> std::shared_ptr<SequenceNumericalNode> {
        if (end - begin == 1) { return leaves[begin]; }
        const auto middle = begin + (end - begin) / 2;
        std::array<std::vector<Pair>, 2> interior;
        std::vector<Pair> boundary;
        // Partition once per level: O(r log(children)) routing work beyond
        // the numerical merge bounds, with no rescans of unrelated links.
        for (auto pair : candidates) {
            accumulateCost(operations, 1);
            if (ports[pair.second / 2].child < middle) { interior[0].push_back(pair); }
            else if (ports[pair.first / 2].child >= middle) { interior[1].push_back(pair); }
            else { boundary.push_back(pair); }
        }
        auto left = build(begin, middle, interior[0]), right = build(middle, end, interior[1]);
        if (!left || !right) { return {}; }
        auto node = std::make_shared<SequenceNumericalNode>();
        node->begin = begin; node->end = end; node->children = {left, right};
        std::vector<NumericalCrossing> crossings;
        std::vector<Pair> identities;
        // Each original crossing belongs to exactly one split. Links can skip
        // any number of children; they are not restricted to adjacent siblings.
        for (auto pair : boundary) {
            auto a = left->events.find(pair.first), b = right->events.find(pair.second);
            if (a == left->events.end() || b == right->events.end()) { return {}; }
            crossings.push_back({a->second, b->second}); identities.push_back(pair);
        }
        uint64_t reductionOperations = 0;
        auto covers = chain::reduce(*left->index, *right->index, crossings, reductionOperations);
        accumulateCost(operations, reductionOperations);
        if (!covers) { return {}; }
        for (std::size_t i = 0; i < covers->size(); ++i) { if ((*covers)[i]) { retained.insert(identities[i]); } }
        std::map<uint32_t, NumericalChainSelection> selected;
        for (unsigned side = 0; side < 2; ++side) {
            for (auto [original, local] : node->children[side]->events) {
                selected.emplace(original, NumericalChainSelection{side, local});
            }
        }
        std::vector<NumericalChainSelection> selection;
        for (const auto& [original, local] : selected) {
            node->events.emplace(original, selection.size()); selection.push_back(local);
        }
        std::map<ChainKey, uint32_t> directory;
        for (const auto& child : node->children) {
            for (const auto& key : child->keys) { directory.emplace(key, 0); }
        }
        for (auto& [key, id] : directory) { id = node->keys.size(); node->keys.push_back(key); }
        std::array<std::vector<uint32_t>, 2> maps;
        for (unsigned side = 0; side < 2; ++side) {
            for (const auto& key : node->children[side]->keys) { maps[side].push_back(directory.at(key)); }
        }
        auto merge = buildNumericalChainMerge(left->index, right->index, crossings, selection, maps[0], maps[1]);
        accumulateCost(operations, merge.index.operations);
        accumulateCost(merges, 1);
        if (!merge.index.error.empty()) { return {}; }
        node->merge = std::make_shared<NumericalChainMerge>(std::move(merge));
        node->index = std::shared_ptr<const NumericalChainInterface>(node->merge, &node->merge->index);

        return node;
    };
    auto tree = build(0, children.size(), std::vector<Pair>(links.begin(), links.end()));
    if (!tree || tree->events.size() != 2 * ports.size()) { return false; }
    std::set<Pair> nativePairs;
    for (const auto& edge : nativeValueCrossings) {
        if (expressions.constantValue(edge.guard) == 1) { nativePairs.emplace(2 * edge.source + 1, 2 * edge.target); }
    }
    std::vector<Crossing> kept;
    for (const auto& edge : crossings) {
        const Pair pair{2 * edge.source + 1, 2 * edge.target};
        const auto active = expressions.constantValue(edge.guard);
        if (!active || (*active && !links.count(pair))) { return false; }
        if (*active && retained.count(pair) && !nativePairs.count(pair)) { kept.push_back(edge); }
    }
    // Publish atomically: unsupported numerical inputs retain the exact general
    // reducer, original incoming records, and original endpoint ownership.
    numericalTree = std::move(tree); numerical = numericalTree->merge;
    numericalChainKeys = numericalTree->keys;
    for (auto& [target, entries] : incoming) {
        std::set<Pair> seen;
        llvm::erase_if(entries, [&](const auto& link) {
            const Pair pair{link.source, link.target};
            return expressions.constantValue(link.guard) == 0 || !retained.count(pair) || !seen.insert(pair).second;
        });
    }
    reachabilityCache.clear(); crossings.swap(kept);
    crossingIds.clear(); std::vector<Crossing>().swap(nativeValueCrossings);
    return true;
}
std::optional<std::vector<uint32_t>> SequenceAnalysisState::numericalThresholds(
    SequenceEvent event, bool reverse, NumericalChainQueryCost& cost)
{
    if (!numerical || !numericalTree || event.child >= children.size()) { return std::nullopt; }
    auto port = portIds.find({event.child, event.type, event.ordinal, event.visits});
    if (port != portIds.end()) {
        auto id = 2 * port->second + (event.kind == PeriodicEventKind::Completion);
        ++cost.indexOperations;
        return reverse ? numerical->index.reverse[id] : numerical->index.forward[id];
    }
    return numericalTree->thresholds(event, reverse, cost);
}
std::optional<Expr> SequenceAnalysisState::numericalReachability(
    SequenceEvent source, SequenceEvent target, NumericalChainQueryCost& cost)
{
    if (!numerical || !numericalTree || source.child >= children.size() || target.child >= children.size()) {
        return std::nullopt;
    }
    if (source.child > target.child) { return no(); }
    auto aPort = portIds.find({source.child, source.type, source.ordinal, source.visits});
    auto bPort = portIds.find({target.child, target.type, target.ordinal, target.visits});
    if (aPort != portIds.end() && bPort != portIds.end()) {
        ++cost.indexOperations;
        return expressions.boolean(numerical->index.reaches(
            2 * aPort->second + (source.kind == PeriodicEventKind::Completion),
            2 * bPort->second + (target.kind == PeriodicEventKind::Completion)));
    }
    auto answer = numericalTree->query(source, target, cost);
    return answer ? std::optional<Expr>(expressions.boolean(*answer)) : std::nullopt;
}
} // namespace mlir::pto::frontiersynch
