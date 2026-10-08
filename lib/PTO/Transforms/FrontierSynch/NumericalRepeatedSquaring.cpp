// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/NumericalRepeatedSquaring.h"
#include "ChainInterfaceInternal.h"
#include <algorithm>
#include <numeric>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
bool increment(uint64_t& value, uint64_t amount = 1)
{
    if (amount > UINT64_MAX - value) { return false; }
    value += amount; return true;
}
bool validThresholds(const NumericalChainInterface& index, const std::vector<uint32_t>& values)
{
    if (values.size() != index.chains.size()) { return false; }
    for (std::size_t c = 0; c < values.size(); ++c) {
        if (values[c] > index.chains[c].size()) { return false; }
    }
    return true;
}
bool validLeaf(const NumericalChainInterface& index)
{
    const auto p = index.chain.size(), c = index.chains.size();
    if (!index.error.empty() || p > UINT32_MAX / 2 || c > p || index.rank.size() != p ||
        index.forward.size() != p || index.reverse.size() != p) { return false; }
    uint64_t seen = 0;
    for (uint32_t chain = 0; chain < c; ++chain) {
        const auto& members = index.chains[chain];
        if (members.empty() || members.size() > p - seen) { return false; }
        seen += members.size();
        for (uint32_t rank = 0; rank < members.size(); ++rank) {
            auto id = members[rank];
            if (id >= p || index.chain[id] != chain || index.rank[id] != rank) { return false; }
        }
    }
    if (seen != p) { return false; }
    for (uint32_t id = 0; id < p; ++id) {
        if (!validThresholds(index, index.forward[id]) || !validThresholds(index, index.reverse[id]) ||
            index.forward[id][index.chain[id]] != index.rank[id] ||
            index.reverse[id][index.chain[id]] != index.rank[id] + 1) { return false; }
    }
    return true;
}
// Existing chain routines increment local uint64 counters. This conservative
// bound dominates their scans, reductions and query propagation, so those
// counters cannot wrap before their checked accumulation into the DAG ledger.
bool boundedWork(uint64_t chains, uint64_t ports, uint64_t links)
{
    uint64_t bound = 128;
    for (auto factor : {chains + 1, chains + 1, ports + links + 1}) {
        if (factor > UINT64_MAX / bound) { return false; }
        bound *= factor;
    }
    return true;
}
class Builder {
public:
    Builder(std::shared_ptr<const NumericalChainInterface> leaf,
            const std::vector<NumericalRepeatedLink>& links) : supplied(std::move(leaf)), input(links) {}
    NumericalRepeatedSquaring run(uint64_t copies)
    {
        if (!supplied || !validLeaf(*supplied) || input.size() > UINT32_MAX ||
            !boundedWork(supplied->chains.size(), 2 * supplied->chain.size(), input.size())) {
            return fail("invalid or unrepresentable numerical repetition leaf interface");
        }
        out.leaf = std::make_shared<const NumericalChainInterface>(*supplied);
        if (!canonicalize()) { return fail("numerical repetition requires valid links and explicit native carries"); }
        auto leaf = std::make_shared<NumericalRepeatedNode>();
        leaf->copies = copies ? 1 : 0;
        leaf->index = copies ? out.leaf : std::make_shared<const NumericalChainInterface>();
        if (copies) {
            for (uint32_t port = 0; port < out.leaf->chain.size(); ++port) {
                leaf->ports.push_back({0, port}); leaf->first.push_back(port); leaf->last.push_back(port);
            }
        }
        out.cost.definitions = 1; out.cost.boundaryPorts = leaf->ports.size();
        if (!copies) { out.root = std::move(leaf); return std::move(out); }
        std::vector<std::shared_ptr<const NumericalRepeatedNode>> powers{std::move(leaf)};
        while (powers.back()->copies <= copies / 2) {
            auto next = merge(powers.back(), powers.back());
            if (!next) { return fail("numerical repetition merge or cost overflow"); }
            powers.push_back(std::move(next));
        }
        for (auto i = powers.size(); i; --i) {
            const auto& power = powers[i - 1];
            if (!(copies & power->copies)) { continue; }
            out.root = out.root ? merge(out.root, power) : power;
            if (!out.root) { return fail("numerical repetition concatenation or cost overflow"); }
        }
        return std::move(out);
    }
private:
    std::shared_ptr<const NumericalChainInterface> supplied;
    const std::vector<NumericalRepeatedLink>& input;
    NumericalRepeatedSquaring out;
    NumericalRepeatedSquaring fail(const char* error)
    {
        out.error = error; out.root.reset(); out.leaf.reset(); out.links.clear();
        out.originalToCanonical.clear(); out.representatives.clear(); return std::move(out);
    }
    bool canonicalize()
    {
        std::vector<uint32_t> order(input.size()); std::iota(order.begin(), order.end(), 0);
        for (const auto& link : input) {
            if (link.source >= out.leaf->chain.size() || link.target >= out.leaf->chain.size()) { return false; }
        }
        auto key = [](const auto& link) { return std::tie(link.source, link.target); };
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return key(input[a]) == key(input[b]) ? a < b : key(input[a]) < key(input[b]);
        });
        std::vector<bool> carry(out.leaf->chains.size(), false);
        out.originalToCanonical.resize(input.size());
        for (auto id : order) {
            const auto& link = input[id];
            if (out.links.empty() || key(out.links.back()) != key(link)) {
                out.links.push_back(link); out.representatives.push_back(id);
            } else if (link.native && !out.links.back().native) {
                out.links.back().native = true; out.representatives.back() = id;
            }
            out.originalToCanonical[id] = static_cast<uint32_t>(out.links.size() - 1);
            const auto chain = out.leaf->chain[link.source];
            if (link.native && link.source == out.leaf->chains[chain].back() &&
                link.target == out.leaf->chains[chain].front()) { carry[chain] = true; }
        }
        return std::find(carry.begin(), carry.end(), false) == carry.end();
    }
    std::shared_ptr<const NumericalRepeatedNode> merge(
        std::shared_ptr<const NumericalRepeatedNode> left, std::shared_ptr<const NumericalRepeatedNode> right)
    {
        if (right->copies > UINT64_MAX - left->copies) { return {}; }
        auto node = std::make_shared<NumericalRepeatedNode>();
        node->copies = left->copies + right->copies; node->children = {left, right};
        std::vector<NumericalCrossing> links;
        for (const auto& link : out.links) { links.push_back({left->last[link.source], right->first[link.target]}); }
        std::vector<NumericalChainSelection> selection;
        for (uint32_t port = 0; port < out.leaf->chain.size(); ++port) {
            node->first.push_back(static_cast<uint32_t>(selection.size()));
            selection.push_back({0, left->first[port]}); node->ports.push_back({0, port});
        }
        for (uint32_t port = 0; port < out.leaf->chain.size(); ++port) {
            node->last.push_back(static_cast<uint32_t>(selection.size()));
            selection.push_back({1, right->last[port]}); node->ports.push_back({node->copies - 1, port});
        }
        std::vector<uint32_t> directory(out.leaf->chains.size());
        std::iota(directory.begin(), directory.end(), 0);
        uint64_t reductionWork = 0;
        auto retained = chain::reduce(*left->index, *right->index, links, reductionWork);
        if (!retained) { return {}; }
        auto merged = buildNumericalChainMerge(left->index, right->index, links, selection, directory, directory);
        if (!merged.index.error.empty()) { return {}; }
        for (uint32_t id = 0; id < out.links.size(); ++id) {
            const auto& link = out.links[id];
            if ((*retained)[id] || link.native) {
                node->seams.push_back({id, {left->copies - 1, link.source}, {left->copies, link.target}, link.native});
            }
        }
        if (!increment(out.cost.definitions) || !increment(out.cost.merges) ||
            !increment(out.cost.boundaryPorts, node->ports.size()) ||
            !increment(out.cost.indexOperations, reductionWork) ||
            !increment(out.cost.indexOperations, merged.index.operations) ||
            !increment(out.cost.candidateLinks, links.size()) ||
            !increment(out.cost.retainedLinks, node->seams.size())) {
            return {};
        }
        node->merge = std::make_shared<const NumericalChainMerge>(std::move(merged));
        node->index = std::shared_ptr<const NumericalChainInterface>(node->merge, &node->merge->index);
        return node;
    }
};
std::optional<std::vector<uint32_t>> lift(const NumericalRepeatedNode& node, uint64_t copy,
    const std::vector<uint32_t>& leafThresholds, bool reverse, NumericalSquaringQueryCost& cost)
{
    if (!increment(cost.nodes)) { return {}; }
    if (!node.merge) { return leafThresholds; }
    const auto child = copy < node.children[0]->copies ? 0U : 1U;
    auto local = copy - (child ? node.children[0]->copies : 0);
    auto thresholds = lift(*node.children[child], local, leafThresholds, reverse, cost);
    if (!thresholds) { return {}; }
    NumericalChainQueryCost work;
    auto result = node.merge->propagate(child, *thresholds, reverse, work);
    return increment(cost.indexOperations, work.indexOperations) ? result : std::nullopt;
}
} // namespace
NumericalRepeatedSquaring buildNumericalRepeatedSquaring(std::shared_ptr<const NumericalChainInterface> leaf,
    const std::vector<NumericalRepeatedLink>& links, uint64_t copies)
{
    return Builder(std::move(leaf), links).run(copies);
}
std::optional<std::vector<uint32_t>> NumericalRepeatedSquaring::thresholds(uint64_t copy,
    const std::vector<uint32_t>& values, bool reverse, NumericalSquaringQueryCost& cost) const
{
    if (!error.empty() || !root || !leaf || copy >= root->copies || !validThresholds(*leaf, values)) { return {}; }
    return lift(*root, copy, values, reverse, cost);
}
std::optional<bool> NumericalRepeatedSquaring::across(uint64_t source, const std::vector<uint32_t>& forward,
    uint64_t target, const std::vector<uint32_t>& reverse, NumericalSquaringQueryCost& cost) const
{
    if (!error.empty() || !root || !leaf || source >= root->copies || target >= root->copies || source == target ||
        !validThresholds(*leaf, forward) || !validThresholds(*leaf, reverse)) { return {}; }
    if (source > target) { return false; }
    const auto* node = root.get();
    while (node->merge) {
        if (!increment(cost.nodes)) { return {}; }
        const auto split = node->children[0]->copies;
        if (target < split) { node = node->children[0].get(); continue; }
        if (source >= split) { source -= split; target -= split; node = node->children[1].get(); continue; }
        auto a = lift(*node->children[0], source, forward, false, cost);
        auto b = lift(*node->children[1], target - split, reverse, true, cost);
        if (!a || !b) { return {}; }
        NumericalChainQueryCost work;
        auto result = node->merge->crosses(*a, *b, work);
        return increment(cost.indexOperations, work.indexOperations) ? result : std::nullopt;
    }
    return {}; // Distinct valid copy coordinates must split at a merge.
}
std::optional<bool> NumericalRepeatedSquaring::query(uint64_t sourceCopy, uint32_t source,
    uint64_t targetCopy, uint32_t target, NumericalSquaringQueryCost& cost) const
{
    if (!error.empty() || !root || !leaf || source >= leaf->chain.size() || target >= leaf->chain.size() ||
        sourceCopy >= root->copies || targetCopy >= root->copies) { return {}; }
    if (sourceCopy == targetCopy) {
        if (!increment(cost.nodes) || !increment(cost.indexOperations)) { return {}; }
        return leaf->reaches(source, target);
    }
    return across(sourceCopy, leaf->forward[source], targetCopy, leaf->reverse[target], cost);
}
} // namespace mlir::pto::frontiersynch
