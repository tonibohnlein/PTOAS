// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RepeatedExcess.h"
#include <algorithm>
#include <iterator>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
llvm::APInt wide(uint64_t value) { return llvm::APInt(256, value); }
bool add(llvm::APInt& sum, const llvm::APInt& value)
{
    if (value.getActiveBits() > 256) { return false; }
    bool overflow = false;
    auto next = sum.uadd_ov(value.zextOrTrunc(256), overflow);
    if (!overflow) { sum = std::move(next); }
    return !overflow;
}
bool charge(uint64_t& sum, uint64_t value = 1)
{
    if (value > UINT64_MAX - sum) { return false; }
    sum += value; return true;
}
struct Skyline {
    const std::vector<uint64_t>& widths;
    std::map<std::size_t, uint64_t> corners;
    llvm::APInt area{256, 0};
    uint64_t updates = 0;
    void insert(std::size_t index, uint64_t height)
    {
        auto next = corners.lower_bound(index);
        if (next != corners.end() && next->second >= height) { return; }
        if (next != corners.end() && next->first == index) { ++next; }
        // The skyline heights strictly decrease with width. A newly inserted
        // corner erases a contiguous preceding suffix; every corner is erased
        // at most once over the entire threshold sweep, not once per query.
        while (next != corners.begin()) {
            auto old = std::prev(next);
            if (old->second > height) { break; }
            const uint64_t start = old == corners.begin() ? 0 : widths[std::prev(old)->first];
            const uint64_t following = next == corners.end() ? 0 : next->second;
            area -= wide(widths[old->first] - start) * wide(old->second - following);
            corners.erase(old); ++updates;
        }
        const uint64_t start = next == corners.begin() ? 0 : widths[std::prev(next)->first];
        const uint64_t following = next == corners.end() ? 0 : next->second;
        area += wide(widths[index] - start) * wide(height - following);
        corners.emplace_hint(next, index, height); ++updates;
    }
};
struct ActivatedPair { uint32_t source = 0, target = 0; uint64_t distance = 0; };
bool validSelectors(const RepeatedExcessInput& input)
{
    const auto p = input.present.size(), k = input.period.counts.size();
    for (const auto* rows : {&input.lower, &input.upper}) {
        if (rows->prefix.size() != p || rows->suffixExcluded.size() != p) { return false; }
        for (std::size_t a = 0; a < p; ++a) {
            if (rows->prefix[a].size() != k || rows->suffixExcluded[a].size() != k) { return false; }
            for (std::size_t pipe = 0; pipe < k; ++pipe) {
                if (rows->prefix[a][pipe] > input.period.counts[pipe] ||
                    rows->suffixExcluded[a][pipe] > input.period.counts[pipe] ||
                    (!input.present[a] && (rows->prefix[a][pipe] ||
                        rows->suffixExcluded[a][pipe] != input.period.counts[pipe]))) { return false; }
            }
        }
    }
    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t pipe = 0; pipe < k; ++pipe) {
            if (input.lower.prefix[a][pipe] > input.upper.prefix[a][pipe] ||
                input.lower.suffixExcluded[a][pipe] < input.upper.suffixExcluded[a][pipe]) { return false; }
        }
    }
    return true;
}
bool initialize(const RepeatedExcessInput& input, uint64_t periods, RepeatedExcessResult& result)
{
    const auto k = input.period.counts.size();
    if (k > UINT32_MAX || input.present.size() > UINT32_MAX || input.prefixes.empty() ||
        input.prefixes.size() > UINT32_MAX || (!k && !input.present.empty()) || !validSelectors(input) ||
        input.period.internalBound.getActiveBits() > 256) {
        result.error = "invalid repeated excess common period/rank frame"; return false;
    }
    bool overflow = false;
    const auto internal = input.period.internalBound.zextOrTrunc(256).umul_ov(wide(periods), overflow);
    if (overflow) { result.error = "repeated internal certificate overflow"; return false; }
    std::vector<uint64_t> previous(k, 0);
    for (std::size_t h = 0; h < input.prefixes.size(); ++h) {
        const auto& prefix = input.prefixes[h];
        if (prefix.counts.size() != k || (!h && !prefix.internalBound.isZero())) {
            result.error = "invalid repeated phase-prefix certificate"; return false;
        }
        for (std::size_t pipe = 0; pipe < k; ++pipe) {
            if (prefix.counts[pipe] < previous[pipe] || prefix.counts[pipe] > input.period.counts[pipe] ||
                (!h && prefix.counts[pipe])) { result.error = "invalid repeated phase-prefix counts"; return false; }
        }
        previous = prefix.counts;
        BoundaryExcessResult value;
        value.internal = internal;
        if (!add(value.internal, prefix.internalBound)) {
            result.error = "repeated internal/prefix certificate overflow"; return false;
        }
        value.exactDifference = input.period.exactInternalDifference && prefix.exactInternalDifference;
        result.prefixes.push_back(std::move(value));
    }
    return true;
}
bool absorb(RepeatedExcessResult& result, const ThresholdRectangleSum& sum, uint64_t rectangles)
{
    if (!sum.error.empty()) { result.error = sum.error; return false; }
    if (!charge(result.rectangles, rectangles) || !charge(result.comparisons, sum.comparisons) ||
        !charge(result.skylineUpdates, sum.skylineUpdates)) {
        result.error = "repeated sweep work counter overflow"; return false;
    }
    return true;
}
std::optional<llvm::APInt> sweep(const BoundaryExcessSelectors& selectors,
    llvm::ArrayRef<ActivatedPair> pairs, uint32_t sourcePipe, uint32_t targetPipe,
    uint64_t consumers, uint64_t periods, bool full, RepeatedExcessResult& result)
{
    std::vector<ThresholdRankRectangle> rectangles;
    for (const auto& edge : pairs) {
        const auto excluded = selectors.suffixExcluded[edge.target][targetPipe];
        const auto width = selectors.prefix[edge.source][sourcePipe];
        if (width && excluded < consumers) { rectangles.push_back({width, consumers - excluded, edge.distance}); }
    }
    auto summed = sumThresholdRectangles(rectangles, periods, full);
    if (!absorb(result, summed, rectangles.size())) { return std::nullopt; }
    return summed.weighted;
}
void accumulate(const RepeatedExcessInput& input, llvm::ArrayRef<ActivatedPair> lower,
    llvm::ArrayRef<ActivatedPair> upper, uint64_t periods, RepeatedExcessResult& result)
{
    const auto k = input.period.counts.size();
    for (uint32_t p = 0; p < k; ++p) {
        for (uint32_t q = 0; q < k; ++q) {
            llvm::APInt full(256, 0);
            for (std::size_t phase = 0; phase <= input.prefixes.size(); ++phase) {
                const bool complete = phase == 0;
                const auto consumers = complete ? input.period.counts[q] : input.prefixes[phase-1].counts[q];
                auto a = sweep(input.upper, upper, p, q, consumers, periods, complete, result);
                auto b = sweep(input.lower, lower, p, q, consumers, periods, complete, result);
                if (!a || !b) { return; }
                if (a->ult(*b)) { result.error = "lower repeated rectangle union exceeds upper"; return; }
                const auto difference = *a - *b;
                if (complete) { full = difference; }
                else if (!add(result.prefixes[phase-1].cross, full) ||
                         !add(result.prefixes[phase-1].cross, difference)) {
                    result.error = "repeated crossing certificate overflow"; return;
                }
            }
        }
    }
    for (auto& prefix : result.prefixes) {
        prefix.total = prefix.internal;
        if (!add(prefix.total, prefix.cross)) { result.error = "repeated total certificate overflow"; return; }
    }
}
} // namespace
ThresholdRectangleSum sumThresholdRectangles(llvm::ArrayRef<ThresholdRankRectangle> input,
                                             uint64_t periods, bool full)
{
    ThresholdRectangleSum result;
    std::vector<ThresholdRankRectangle> rectangles;
    std::vector<uint64_t> widths;
    const uint64_t limit = full ? (periods ? periods - 1 : 0) : periods;
    for (auto rectangle : input) {
        rectangle.activation = std::max(uint64_t(1), rectangle.activation);
        if (!rectangle.width || !rectangle.height || rectangle.activation > limit) { continue; }
        rectangles.push_back(rectangle); widths.push_back(rectangle.width);
    }
    if (rectangles.size() > UINT64_MAX / 2) { result.error = "rectangle skyline counter overflow"; return result; }
    bool countOK = true;
    std::sort(widths.begin(), widths.end(), [&](uint64_t a, uint64_t b) {
        countOK &= charge(result.comparisons); return a < b;
    });
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    std::sort(rectangles.begin(), rectangles.end(), [&](const auto& a, const auto& b) {
        countOK &= charge(result.comparisons); return a.activation < b.activation;
    });
    Skyline skyline{widths, {}, wide(0), 0};
    std::size_t at = 0;
    while (at < rectangles.size()) {
        const auto start = rectangles[at].activation;
        do {
            const auto& rectangle = rectangles[at++];
            const auto found = std::lower_bound(widths.begin(), widths.end(), rectangle.width,
                [&](uint64_t a, uint64_t b) { countOK &= charge(result.comparisons); return a < b; });
            skyline.insert(found - widths.begin(), rectangle.height);
        } while (at < rectangles.size() && rectangles[at].activation == start);
        const uint64_t end = at == rectangles.size() ? limit : rectangles[at].activation - 1;
        auto weight = wide(end - start + 1);
        if (full) { weight *= wide(periods - start) + wide(periods - end); weight = weight.udiv(wide(2)); }
        bool overflow = false;
        const auto term = skyline.area.umul_ov(weight, overflow);
        if (overflow || !add(result.weighted, term)) {
            result.error = "threshold rectangle weighted sum overflow"; return result;
        }
    }
    result.skylineUpdates = skyline.updates;
    if (!countOK) { result.error = "rectangle comparison counter overflow"; }
    return result;
}
RepeatedExcessResult countRepeatedExcess(const RepeatedExcessInput& input,
    const RepeatedPortDistances& lower, const RepeatedPortDistances& upper, uint64_t periods)
{
    RepeatedExcessResult result;
    if (!initialize(input, periods, result)) { return result; }
    const auto p = input.present.size();
    if (lower.size() != p || upper.size() != p) {
        result.error = "repeated distance row count mismatch"; return result;
    }
    std::vector<ActivatedPair> a, b;
    for (uint32_t source = 0; source < p; ++source) {
        if (lower[source].size() != p || upper[source].size() != p) {
            result.error = "repeated distance column count mismatch"; return result;
        }
        for (uint32_t target = 0; target < p; ++target) {
            if (!charge(result.thresholdEntries)) {
                result.error = "repeated distance counter overflow"; return result;
            }
            const auto lo = lower[source][target], hi = upper[source][target];
            if ((lo && (!hi || *hi > *lo)) || ((!input.present[source] || !input.present[target]) && (lo || hi)) ||
                (source == target && input.present[source] && (lo != 0 || hi != 0))) {
                result.error = "repeated distances violate presence, reflexivity or lower containment"; return result;
            }
            if (hi) { a.push_back({source, target, *hi}); }
            if (lo) { b.push_back({source, target, *lo}); }
        }
    }
    accumulate(input, b, a, periods, result);
    return result;
}
RepeatedExcessResult countChainRepeatedExcess(const RepeatedExcessInput& input,
    const RepeatedChainFrontiers& frontiers, uint64_t periods)
{
    RepeatedExcessResult result;
    if (!initialize(input, periods, result)) { return result; }
    const auto p = input.present.size(), k = input.period.counts.size(), c = frontiers.chains.size();
    if (c > 2 * uint64_t(k) || frontiers.lower.size() != c || frontiers.upper.size() != c ||
        std::find(input.present.begin(), input.present.end(), false) != input.present.end()) {
        result.error = "repeated compression requires a common present native-chain frame"; return result;
    }
    std::vector<bool> seen(p, false);
    std::vector<ActivatedPair> upper, lower;
    for (std::size_t chain = 0; chain < c; ++chain) {
        const auto& ports = frontiers.chains[chain];
        if (ports.empty() || frontiers.lower[chain].size() != p || frontiers.upper[chain].size() != p) {
            result.error = "invalid repeated chain frontier shape"; return result;
        }
        for (std::size_t rank = 0; rank < ports.size(); ++rank) {
            const auto port = ports[rank];
            if (port >= p || seen[port] || frontiers.lower[chain][port] != ports.size()-rank-1 ||
                frontiers.upper[chain][port] != ports.size()-rank-1) {
                result.error = "repeated native chain directory or reflexive frontier mismatch"; return result;
            }
            seen[port] = true;
            if (!rank) { continue; }
            for (const auto* rows : {&input.lower, &input.upper}) {
                for (std::size_t pipe = 0; pipe < k; ++pipe) {
                    if (rows->prefix[ports[rank-1]][pipe] > rows->prefix[port][pipe] ||
                        rows->suffixExcluded[ports[rank-1]][pipe] > rows->suffixExcluded[port][pipe]) {
                        result.error = "rank selectors are not ordered on the native chain"; return result;
                    }
                }
            }
        }
        for (uint32_t target = 0; target < p; ++target) {
            if (!charge(result.thresholdEntries)) {
                result.error = "repeated frontier counter overflow"; return result;
            }
            const auto lo = frontiers.lower[chain][target], hi = frontiers.upper[chain][target];
            if (lo && (!hi || *hi > *lo)) { result.error = "lower repeated frontier exceeds upper"; return result; }
            auto append = [&](std::optional<uint64_t> rho, std::vector<ActivatedPair>& edges) {
                if (!rho) { return; }
                const uint64_t distance = *rho / ports.size(), tail = *rho % ports.size();
                edges.push_back({ports[ports.size()-tail-1], target, distance});
                // tail!=0 implies width>1 and distance<UINT64_MAX.
                if (tail) { edges.push_back({ports.back(), target, distance+1}); }
            };
            append(hi, upper); append(lo, lower);
        }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        result.error = "repeated native chains omit a present port"; return result;
    }
    accumulate(input, lower, upper, periods, result);
    return result;
}
} // namespace mlir::pto::frontiersynch
