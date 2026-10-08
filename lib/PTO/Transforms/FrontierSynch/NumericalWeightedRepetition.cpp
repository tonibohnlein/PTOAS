// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/NumericalWeightedRepetition.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <queue>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Wide = llvm::APInt;
Wide wide(uint64_t value) { return Wide(128, value); }
std::optional<Wide> add(const Wide& a, const Wide& b)
{
    bool overflow = false;
    auto result = a.uadd_ov(b, overflow);
    return overflow ? std::nullopt : std::optional<Wide>(std::move(result));
}
std::optional<Wide> scaled(uint64_t count, uint64_t value)
{
    bool overflow = false;
    auto result = wide(count).umul_ov(wide(value), overflow);
    return overflow ? std::nullopt : std::optional<Wide>(std::move(result));
}
struct Edge { uint32_t target; uint64_t weight; };
struct HeapEntry { Wide distance; uint32_t port; };
struct Later {
    bool operator()(const HeapEntry& a, const HeapEntry& b) const
    {
        return a.distance == b.distance ? a.port > b.port : a.distance.ugt(b.distance);
    }
};
struct Candidate { Wide score; uint32_t record; };
bool earlier(const Candidate& a, const Candidate& b)
{
    return a.score == b.score ? a.record < b.record : a.score.ult(b.score);
}
struct BestTwo {
    std::optional<Candidate> first, second;
    void insert(Candidate candidate)
    {
        if (first && first->record == candidate.record) { return; }
        if (second && second->record == candidate.record) { return; }
        if (!first || earlier(candidate, *first)) { second = first; first = std::move(candidate); }
        else if (!second || earlier(candidate, *second)) { second = std::move(candidate); }
    }
    const Candidate* excluding(uint32_t record) const
    {
        if (first && first->record != record) { return &*first; }
        return second ? &*second : nullptr;
    }
};
struct Prefix { uint32_t rank; BestTwo best; };
class Builder {
public:
    Builder(const NumericalChainInterface& body, const std::vector<NumericalWeightedCrossing>& input)
        : body(body), input(input) {}
    NumericalWeightedRepetition run()
    {
        if (!validate() || !canonicalize() || !graph()) { return failed(); }
        auto index = std::make_shared<NumericalWeightedRepetitionIndex>();
        index->body = body;
        index->rho.resize(body.chains.size());
        std::vector<uint32_t> ordered(out.crossings.size());
        std::iota(ordered.begin(), ordered.end(), 0);
        std::sort(ordered.begin(), ordered.end(), [&](uint32_t a, uint32_t b) {
            auto x = out.crossings[a].target, y = out.crossings[b].target;
            return std::tie(body.chain[x], body.rank[x], a) < std::tie(body.chain[y], body.rank[y], b);
        });
        out.retained.assign(out.crossings.size(), true);
        for (uint32_t c = 0; c < body.chains.size(); ++c) {
            if (!dijkstra(c, index->rho[c]) || !covers(c, index->rho[c], ordered)) { return failed(); }
        }
        out.index = std::move(index);
        return std::move(out);
    }
private:
    const NumericalChainInterface& body;
    const std::vector<NumericalWeightedCrossing>& input;
    NumericalWeightedRepetition out;
    std::vector<std::vector<Edge>> edges;
    bool fail(const char* message) { out.error = message; return false; }
    NumericalWeightedRepetition failed()
    {
        out.index.reset(); out.crossings.clear(); out.retained.clear();
        out.originalToCanonical.clear(); out.representatives.clear();
        return std::move(out);
    }
    bool charge(uint64_t& counter, uint64_t amount = 1)
    {
        if (amount > UINT64_MAX - counter) { return fail("weighted repetition cost counter overflow"); }
        counter += amount; return true;
    }
    bool validate()
    {
        const auto p = body.chain.size(), c = body.chains.size();
        if (!body.error.empty() || p > UINT32_MAX || c > UINT32_MAX || input.size() > UINT32_MAX ||
            body.rank.size() != p || body.forward.size() != p || body.reverse.size() != p) {
            return fail("invalid weighted repetition child index dimensions");
        }
        uint64_t slots = 0;
        for (uint32_t chain = 0; chain < c; ++chain) {
            const auto& members = body.chains[chain];
            if (members.empty() || members.size() > p - slots) {
                return fail("weighted repetition chains must be nonempty and partition the ports");
            }
            slots += members.size();
            for (uint32_t rank = 0; rank < members.size(); ++rank) {
                if (!charge(out.cost.validation)) { return false; }
                auto port = members[rank];
                if (port >= p || body.chain[port] != chain || body.rank[port] != rank) {
                    return fail("invalid weighted repetition chain directory");
                }
            }
        }
        if (slots != p) { return fail("weighted repetition chain directory is incomplete"); }
        for (uint32_t port = 0; port < p; ++port) {
            if (body.forward[port].size() != c || body.reverse[port].size() != c) {
                return fail("invalid weighted repetition threshold row");
            }
            for (uint32_t chain = 0; chain < c; ++chain) {
                if (!charge(out.cost.validation)) { return false; }
                if (body.forward[port][chain] > body.chains[chain].size() ||
                    body.reverse[port][chain] > body.chains[chain].size()) {
                    return fail("weighted repetition threshold outside its chain");
                }
            }
            if (body.forward[port][body.chain[port]] != body.rank[port] ||
                body.reverse[port][body.chain[port]] != body.rank[port] + 1) {
                return fail("weighted repetition child lacks exact reflexive native order");
            }
        }
        for (uint32_t source = 0; source < c; ++source) {
            const auto& members = body.chains[source];
            for (uint32_t target = 0; target < c; ++target) {
                for (uint32_t rank = 1; rank < members.size(); ++rank) {
                    if (!charge(out.cost.validation)) { return false; }
                    if (body.forward[members[rank - 1]][target] > body.forward[members[rank]][target]) {
                        return fail("weighted repetition child thresholds violate native chain order");
                    }
                }
                uint32_t cursor = 0;
                for (uint32_t rank = 0; rank < body.chains[target].size(); ++rank) {
                    while (cursor < members.size() && body.forward[members[cursor]][target] <= rank) {
                        ++cursor;
                        if (!charge(out.cost.validation)) { return false; }
                    }
                    if (!charge(out.cost.validation)) { return false; }
                    if (body.reverse[body.chains[target][rank]][source] != cursor) {
                        return fail("weighted repetition forward/reverse thresholds disagree");
                    }
                }
            }
        }
        return true;
    }
    bool canonicalize()
    {
        std::vector<uint32_t> order(input.size());
        std::iota(order.begin(), order.end(), 0);
        for (const auto& edge : input) {
            if (!charge(out.cost.validation)) { return false; }
            if (edge.source >= body.chain.size() || edge.target >= body.chain.size() || !edge.displacement) {
                return fail("weighted repetition crossings require valid ports and positive distances");
            }
        }
        const auto key = [](const auto& e) { return std::tie(e.source, e.target, e.displacement); };
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return key(input[a]) == key(input[b]) ? a < b : key(input[a]) < key(input[b]);
        });
        out.originalToCanonical.resize(input.size());
        std::vector<bool> carried(body.chains.size(), false);
        for (auto id : order) {
            const auto& edge = input[id];
            if (out.crossings.empty() || key(out.crossings.back()) != key(edge)) {
                out.crossings.push_back(edge); out.representatives.push_back(id);
            } else if (edge.native && !out.crossings.back().native) {
                out.crossings.back().native = true; out.representatives.back() = id;
            }
            out.originalToCanonical[id] = static_cast<uint32_t>(out.crossings.size() - 1);
            const auto c = body.chain[edge.source];
            if (edge.native && edge.displacement == 1 && edge.source == body.chains[c].back() &&
                edge.target == body.chains[c].front()) { carried[c] = true; }
        }
        if (std::find(carried.begin(), carried.end(), false) != carried.end()) {
            return fail("weighted repetition requires explicit native unit carry on every chain");
        }
        return true;
    }
    bool graph()
    {
        const auto p = body.chain.size();
        edges.resize(p);
        std::vector<uint64_t> indegree(p, 0);
        auto internal = [&](uint32_t a, uint32_t b) {
            if (a == b) { return true; }
            edges[a].push_back({b, 0});
            return charge(indegree[b]) && charge(out.cost.internalEdges);
        };
        for (const auto& chain : body.chains) {
            for (std::size_t rank = 1; rank < chain.size(); ++rank) {
                if (!internal(chain[rank - 1], chain[rank])) { return false; }
            }
        }
        for (uint32_t source = 0; source < p; ++source) {
            for (uint32_t c = 0; c < body.chains.size(); ++c) {
                const auto rank = body.forward[source][c];
                if (rank < body.chains[c].size() && !internal(source, body.chains[c][rank])) { return false; }
            }
        }
        // Nontrivial zero cycles mean distinct supplied identities are aliases
        // or the child order is invalid. Deduplication belongs to the caller.
        std::vector<uint32_t> ready;
        for (uint32_t port = 0; port < p; ++port) { if (!indegree[port]) { ready.push_back(port); } }
        for (std::size_t next = 0; next < ready.size(); ++next) {
            for (const auto& edge : edges[ready[next]]) {
                if (!charge(out.cost.validation)) { return false; }
                if (!--indegree[edge.target]) { ready.push_back(edge.target); }
            }
        }
        if (ready.size() != p) { return fail("weighted repetition requires deduplicated zero-order identities"); }
        for (const auto& edge : out.crossings) {
            edges[edge.source].push_back({edge.target, edge.displacement});
            if (!charge(out.cost.crossingEdges)) { return false; }
        }
        return true;
    }
    bool dijkstra(uint32_t c, std::vector<std::optional<Wide>>& distances)
    {
        const uint64_t h = body.chains[c].size();
        distances.resize(body.chain.size());
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, Later> heap;
        for (uint32_t rank = 0; rank < h; ++rank) {
            auto port = body.chains[c][rank];
            distances[port] = wide(h - 1 - rank);
            heap.push({*distances[port], port});
            if (!charge(out.cost.heapPushes)) { return false; }
        }
        while (!heap.empty()) {
            auto current = heap.top(); heap.pop();
            if (!charge(out.cost.heapPops)) { return false; }
            if (current.distance != *distances[current.port]) { continue; }
            for (const auto& edge : edges[current.port]) {
                if (!charge(out.cost.relaxations)) { return false; }
                auto weight = scaled(h, edge.weight);
                auto next = weight ? add(current.distance, *weight) : std::nullopt;
                if (!next) { return fail("weighted repetition scaled distance overflow"); }
                auto& old = distances[edge.target];
                if (!old || next->ult(*old)) {
                    old = std::move(next); heap.push({*old, edge.target});
                    if (!charge(out.cost.heapPushes)) { return false; }
                }
            }
        }
        return true;
    }
    bool covers(uint32_t c, const std::vector<std::optional<Wide>>& rho,
                const std::vector<uint32_t>& ordered)
    {
        const uint64_t h = body.chains[c].size();
        std::vector<std::vector<Prefix>> prefixes(body.chains.size());
        for (auto id : ordered) {
            const auto& edge = out.crossings[id];
            if (!charge(out.cost.prefixEntries)) { return false; }
            if (!rho[edge.source]) { continue; }
            auto weight = scaled(h, edge.displacement);
            auto score = weight ? add(*rho[edge.source], *weight) : std::nullopt;
            if (!score) { return fail("weighted repetition exclusion score overflow"); }
            auto& group = prefixes[body.chain[edge.target]];
            BestTwo best = group.empty() ? BestTwo{} : group.back().best;
            best.insert({std::move(*score), id});
            group.push_back({body.rank[edge.target], std::move(best)});
        }
        for (uint32_t id = 0; id < out.crossings.size(); ++id) {
            const auto& edge = out.crossings[id];
            if (edge.native || body.chain[edge.source] != c) { continue; }
            auto weight = scaled(h, edge.displacement);
            auto limit = weight ? add(*weight, wide(h - 1 - body.rank[edge.source])) : std::nullopt;
            if (!limit) { return fail("weighted repetition cover threshold overflow"); }
            for (uint32_t d = 0; d < body.chains.size(); ++d) {
                if (!charge(out.cost.coverQueries)) { return false; }
                const auto& group = prefixes[d];
                const auto stop = body.reverse[edge.target][d];
                auto found = std::lower_bound(group.begin(), group.end(), stop,
                    [](const Prefix& prefix, uint32_t rank) { return prefix.rank < rank; });
                if (found == group.begin()) { continue; }
                const auto* best = std::prev(found)->best.excluding(id);
                if (best && best->score.ule(*limit)) { out.retained[id] = false; break; }
            }
        }
        return true;
    }
};
} // namespace
NumericalWeightedRepetition buildNumericalWeightedRepetition(
    const NumericalChainInterface& body, const std::vector<NumericalWeightedCrossing>& crossings)
{
    return Builder(body, crossings).run();
}
std::optional<NumericalWeightedDistance> NumericalWeightedRepetitionIndex::distance(
    uint32_t source, uint32_t target) const
{
    if (source >= body.chain.size() || target >= body.chain.size()) { return std::nullopt; }
    const auto c = body.chain[source];
    if (!rho[c][target]) { return NumericalWeightedDistance{}; }
    const uint64_t h = body.chains[c].size();
    // ceil((rho + rank(source) - (h-1))/h), clamped to zero.
    const auto base = wide(h - 1 - body.rank[source]);
    Wide result = wide(0);
    if (rho[c][target]->ugt(base)) {
        const auto numerator = *rho[c][target] - base;
        const auto divisor = wide(h);
        result = numerator.udiv(divisor);
        if (!numerator.urem(divisor).isZero()) { ++result; }
    }
    return NumericalWeightedDistance{true, result.getActiveBits() > 64 ? std::nullopt :
        std::optional<uint64_t>(result.getZExtValue())};
}
std::optional<bool> NumericalWeightedRepetitionIndex::query(uint32_t source, uint32_t target, uint64_t gap) const
{
    auto result = distance(source, target);
    if (!result) { return std::nullopt; }
    return result->reachable && result->displacement && *result->displacement <= gap;
}
std::optional<bool> NumericalWeightedRepetitionIndex::across(const std::vector<uint32_t>& forward,
    const std::vector<uint32_t>& reverse, uint64_t gap, uint64_t& probes) const
{
    const auto count = body.chains.size();
    if (!gap || forward.size() != count || reverse.size() != count) { return std::nullopt; }
    for (std::size_t c = 0; c < count; ++c) {
        if (forward[c] > body.chains[c].size() || reverse[c] > body.chains[c].size()) { return std::nullopt; }
    }
    for (std::size_t c = 0; c < count; ++c) {
        if (forward[c] == body.chains[c].size()) { continue; }
        for (std::size_t d = 0; d < count; ++d) {
            if (!reverse[d]) { continue; }
            if (probes == UINT64_MAX) { return std::nullopt; }
            ++probes;
            auto result = query(body.chains[c][forward[c]], body.chains[d][reverse[d] - 1], gap);
            if (!result || *result) { return result; }
        }
    }
    return false;
}
} // namespace mlir::pto::frontiersynch
