// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Prefix/suffix rectangle unions on supplied exact lower and upper interfaces.
#include "PTO/Transforms/FrontierSynch/BoundaryExcess.h"
#include <algorithm>
#include <map>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
using Pair = std::pair<uint32_t, uint32_t>;
llvm::APInt wide(uint64_t value) { return llvm::APInt(256, value); }
bool add(llvm::APInt& sum, const llvm::APInt& value)
{
    if (value.getActiveBits() > 256) { return false; }
    bool overflow = false;
    auto next = sum.uadd_ov(value.zextOrTrunc(256), overflow);
    if (overflow) { return false; }
    sum = std::move(next);
    return true;
}
bool charge(uint64_t& sum, uint64_t value)
{
    if (value > UINT64_MAX - sum) { return false; }
    sum += value;
    return true;
}
bool validSelectors(const BoundaryExcessInput& input, const BoundaryExcessSelectors& selectors)
{
    if (selectors.prefix.size() != input.ports.size() || selectors.suffixExcluded.size() != input.ports.size()) {
        return false;
    }
    for (std::size_t id = 0; id < input.ports.size(); ++id) {
        if (selectors.prefix[id].size() != input.pipes || selectors.suffixExcluded[id].size() != input.pipes) {
            return false;
        }
        const auto& port = input.ports[id];
        for (uint32_t pipe = 0; pipe < input.pipes; ++pipe) {
            const auto count = input.regions[port.region].counts[pipe];
            if (selectors.prefix[id][pipe] > count || selectors.suffixExcluded[id][pipe] > count ||
                (!port.present && (selectors.prefix[id][pipe] || selectors.suffixExcluded[id][pipe] != count))) {
                return false;
            }
        }
    }
    return true;
}
bool initialize(const BoundaryExcessInput& input, BoundaryExcessResult& out)
{
    if (input.regions.size() > UINT32_MAX || input.ports.size() > UINT32_MAX ||
        (!input.pipes && !input.ports.empty())) {
        out.error = "boundary excess region/port identity overflow or empty pipe directory"; return false;
    }
    out.exactDifference = true;
    for (const auto& region : input.regions) {
        if (region.counts.size() != input.pipes || !add(out.internal, region.internalBound)) {
            out.error = "boundary excess count directory or internal certificate overflow"; return false;
        }
        out.exactDifference &= region.exactInternalDifference;
    }
    for (const auto& port : input.ports) {
        if (port.region >= input.regions.size()) { out.error = "boundary excess port has no region"; return false; }
    }
    if (!validSelectors(input, input.lower) || !validSelectors(input, input.upper)) {
        out.error = "invalid boundary excess rank selector shape, presence or bounds"; return false;
    }
    for (std::size_t id = 0; id < input.ports.size(); ++id) {
        for (uint32_t pipe = 0; pipe < input.pipes; ++pipe) {
            if (input.lower.prefix[id][pipe] > input.upper.prefix[id][pipe] ||
                input.lower.suffixExcluded[id][pipe] < input.upper.suffixExcluded[id][pipe]) {
                out.error = "boundary excess lower rank selectors exceed upper relation"; return false;
            }
        }
    }
    return true;
}
bool validClosures(const BoundaryExcessInput& input, const BoundaryPortClosure& lower,
                   const BoundaryPortClosure& upper)
{
    const auto size = input.ports.size();
    if (lower.size() != size || upper.size() != size) { return false; }
    for (std::size_t a = 0; a < size; ++a) {
        if (lower[a].size() != size || upper[a].size() != size ||
            lower[a][a] != input.ports[a].present || upper[a][a] != input.ports[a].present) { return false; }
        for (std::size_t b = 0; b < size; ++b) {
            if (lower[a][b] && !upper[a][b]) { return false; }
            if (upper[a][b] && (!input.ports[a].present || !input.ports[b].present ||
                input.ports[a].region > input.ports[b].region)) { return false; }
        }
    }
    return true;
}
bool rectangles(const BoundaryExcessInput& input, llvm::ArrayRef<Pair> pairs,
                const BoundaryExcessSelectors& selectors, uint32_t sourcePipe, uint32_t targetPipe,
                uint32_t targetRegion, BoundaryExcessResult& out, llvm::APInt& area)
{
    std::vector<BoundaryRankRectangle> values;
    values.reserve(pairs.size());
    for (auto [source, target] : pairs) {
        values.push_back({selectors.prefix[source][sourcePipe], selectors.suffixExcluded[target][targetPipe]});
    }
    auto counted = countBoundaryRectangles(values, input.regions[targetRegion].counts[targetPipe]);
    if (!counted.error.empty() || !charge(out.rectangles, values.size()) ||
        !charge(out.comparisons, counted.comparisons)) {
        out.error = counted.error.empty() ? "boundary rectangle work counter overflow" : counted.error;
        return false;
    }
    area = std::move(counted.pairs);
    return true;
}
bool difference(const BoundaryExcessInput& input, llvm::ArrayRef<Pair> lower, llvm::ArrayRef<Pair> upper,
                uint32_t targetRegion, BoundaryExcessResult& out)
{
    for (uint32_t p = 0; p < input.pipes; ++p) {
        for (uint32_t q = 0; q < input.pipes; ++q) {
            auto a = wide(0), b = wide(0);
            if (!rectangles(input, upper, input.upper, p, q, targetRegion, out, a) ||
                !rectangles(input, lower, input.lower, p, q, targetRegion, out, b)) { return false; }
            if (b.ugt(a) || !add(out.cross, a - b)) {
                out.error = "boundary excess containment or cross-count overflow"; return false;
            }
        }
    }
    return true;
}
void finish(BoundaryExcessResult& out)
{
    out.total = out.internal;
    if (!add(out.total, out.cross)) { out.error = "boundary excess total exceeds 256 bits"; }
}
bool validIndex(const NumericalChainInterface& index, std::size_t size, BoundaryExcessResult& out)
{
    const auto chains = index.chains.size();
    if (!index.error.empty() || chains > UINT32_MAX || index.chain.size() != size || index.rank.size() != size ||
        index.forward.size() != size || index.reverse.size() != size) { return false; }
    std::size_t slots = 0;
    for (const auto& chain : index.chains) {
        if (chain.size() > size - slots) { return false; }
        slots += chain.size();
    }
    if (slots != size) { return false; }
    // This representation-only bound charges validation before any counter can
    // wrap. Numerical payload counts never occur in work/allocation bounds.
    if ((wide(chains + 1) * (wide(size) + wide(chains) + 1) * 16).getActiveBits() > 64) { return false; }
    for (std::size_t id = 0; id < size; ++id) {
        const auto c = index.chain[id], r = index.rank[id];
        if (c >= chains || r >= index.chains[c].size() || index.chains[c][r] != id ||
            index.forward[id].size() != chains || index.reverse[id].size() != chains) { return false; }
        for (std::size_t d = 0; d < chains; ++d) {
            ++out.indexOperations;
            if (index.forward[id][d] > index.chains[d].size() || index.reverse[id][d] > index.chains[d].size()) {
                return false;
            }
        }
        if (index.forward[id][c] != r || index.reverse[id][c] != r + 1) { return false; }
    }
    // Invert monotone forward rows in linear work per chain pair, checking that
    // the supplied reverse index is the same relation, without all-port pairs.
    for (std::size_t c = 0; c < chains; ++c) {
        for (std::size_t d = 0; d < chains; ++d) {
            uint32_t last = 0;
            for (auto id : index.chains[c]) {
                ++out.indexOperations;
                if (index.forward[id][d] < last) { return false; }
                last = index.forward[id][d];
            }
            std::size_t cursor = 0;
            for (std::size_t rank = 0; rank < index.chains[d].size(); ++rank) {
                while (cursor < index.chains[c].size() && index.forward[index.chains[c][cursor]][d] <= rank) {
                    ++cursor; ++out.indexOperations;
                }
                ++out.indexOperations;
                if (index.reverse[index.chains[d][rank]][c] != cursor) { return false; }
            }
        }
    }
    return true;
}
bool binaryFrame(const BoundaryExcessInput& input, const NumericalChainInterface& lower,
                 const NumericalChainInterface& upper, std::vector<uint32_t>& left, BoundaryExcessResult& out)
{
    const auto workBound = (wide(input.pipes) + wide(upper.chains.size()) + 1) *
        (wide(input.ports.size()) + wide(upper.chains.size()) + 1) * 32;
    if (workBound.getActiveBits() > 64 || input.regions.size() != 2 ||
        upper.chains.size() > 2 * uint64_t(input.pipes) ||
        upper.chains != lower.chains || !validIndex(lower, input.ports.size(), out) ||
        !validIndex(upper, input.ports.size(), out)) { return false; }
    left.resize(upper.chains.size());
    for (std::size_t c = 0; c < upper.chains.size(); ++c) {
        std::optional<uint32_t> previous;
        for (auto id : upper.chains[c]) {
            const auto& port = input.ports[id];
            if (!port.present || (previous && input.ports[*previous].region > port.region)) { return false; }
            if (!port.region) { ++left[c]; }
            for (uint32_t p = 0; p < input.pipes && previous; ++p) {
                if (input.ports[*previous].region != port.region) { break; }
                for (const auto* selectors : {&input.lower, &input.upper}) {
                    ++out.indexOperations;
                    if (selectors->prefix[*previous][p] > selectors->prefix[id][p] ||
                        selectors->suffixExcluded[*previous][p] > selectors->suffixExcluded[id][p]) { return false; }
                }
            }
            previous = id;
        }
    }
    for (std::size_t id = 0; id < input.ports.size(); ++id) {
        for (std::size_t c = 0; c < upper.chains.size(); ++c) {
            ++out.indexOperations;
            if (lower.reverse[id][c] > upper.reverse[id][c] || lower.forward[id][c] < upper.forward[id][c] ||
                (!input.ports[id].region && upper.reverse[id][c] > left[c])) { return false; }
        }
    }
    return true;
}
std::vector<Pair> representatives(const BoundaryExcessInput& input, const NumericalChainInterface& index,
                                  llvm::ArrayRef<uint32_t> left, BoundaryExcessResult& out)
{
    std::vector<Pair> result;
    for (uint32_t target = 0; target < input.ports.size(); ++target) {
        if (!input.ports[target].region) { continue; }
        for (std::size_t c = 0; c < index.chains.size(); ++c) {
            ++out.indexOperations;
            const auto end = std::min(left[c], index.reverse[target][c]);
            if (end) { result.push_back({index.chains[c][end - 1], target}); }
        }
    }
    return result;
}
} // namespace
BoundaryRectangleCount countBoundaryRectangles(llvm::ArrayRef<BoundaryRankRectangle> rectangles, uint64_t consumers)
{
    BoundaryRectangleCount out;
    std::vector<BoundaryRankRectangle> sorted;
    for (auto rectangle : rectangles) {
        if (rectangle.targetExcluded > consumers) { out.error = "boundary rectangle rank exceeds count"; return out; }
        if (rectangle.sourcePrefix && rectangle.targetExcluded < consumers) { sorted.push_back(rectangle); }
    }
    bool overflow = false;
    std::sort(sorted.begin(), sorted.end(), [&out, &overflow](auto a, auto b) {
        if (!charge(out.comparisons, 1)) { overflow = true; }
        return a.sourcePrefix > b.sourcePrefix;
    });
    uint64_t excluded = consumers;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        excluded = std::min(excluded, sorted[i].targetExcluded);
        const uint64_t next = i + 1 < sorted.size() ? sorted[i + 1].sourcePrefix : 0;
        if (!add(out.pairs, wide(sorted[i].sourcePrefix - next) * wide(consumers - excluded))) { overflow = true; }
    }
    if (overflow) { out.error = "boundary rectangle arithmetic counter overflow"; }
    return out;
}
BoundaryExcessResult countBoundaryExcess(const BoundaryExcessInput& input,
    const BoundaryPortClosure& lower, const BoundaryPortClosure& upper)
{
    BoundaryExcessResult out;
    if (!initialize(input, out)) { return out; }
    if (!validClosures(input, lower, upper)) { out.error = "invalid common boundary port closures"; return out; }
    struct Group { std::vector<Pair> lower, upper; };
    std::map<Pair, Group> groups;
    for (uint32_t source = 0; source < input.ports.size(); ++source) {
        for (uint32_t target = 0; target < input.ports.size(); ++target) {
            ++out.portPairs;
            const auto a = input.ports[source].region, b = input.ports[target].region;
            if (a >= b || !upper[source][target]) { continue; }
            auto& group = groups[{a, b}];
            group.upper.emplace_back(source, target);
            if (lower[source][target]) { group.lower.emplace_back(source, target); }
        }
    }
    for (const auto& [regions, group] : groups) {
        if (!difference(input, group.lower, group.upper, regions.second, out)) { return out; }
    }
    finish(out);
    return out;
}
BoundaryExcessResult countBinaryBoundaryExcess(const BoundaryExcessInput& input,
    const NumericalChainInterface& lower, const NumericalChainInterface& upper)
{
    BoundaryExcessResult out;
    if (!initialize(input, out)) { return out; }
    std::vector<uint32_t> left;
    if (!binaryFrame(input, lower, upper, left, out)) {
        out.error = "invalid shared binary chain frame, rank monotonicity or lower containment"; return out;
    }
    const auto a = representatives(input, lower, left, out), b = representatives(input, upper, left, out);
    if (!difference(input, a, b, 1, out)) { return out; }
    finish(out);
    return out;
}
} // namespace mlir::pto::frontiersynch
