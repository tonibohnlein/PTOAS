// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Closed arithmetic counts; support is positive-length pipe reachability.
#include "PTO/Transforms/FrontierSynch/PeriodicExcessCertificate.h"
#include "PTO/Transforms/FrontierSynch/CompactOrderBounds.h"
#include "llvm/ADT/SmallString.h"
#include <algorithm>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
namespace {
using Support = std::vector<std::vector<bool>>;
llvm::APInt wide(uint64_t value) { return llvm::APInt(256, value); }
bool validRecords(llvm::ArrayRef<PeriodicRecord> records, std::size_t count)
{
    for (const auto& record : records) {
        if (record.source >= count || record.target >= count ||
            (!record.displacement && record.source >= record.target)) { return false; }
    }
    return true;
}
bool validShape(const PeriodicAnalysis& index)
{
    const auto m = index.payloads.size(), k = index.frontiers.size();
    if (!index.error.empty() || m > UINT32_MAX / 3 || k > m || index.pipeRows.size() != k ||
        index.sourceRows.size() != m || index.localRanks.size() != m ||
        !validRecords(index.generators, m) || !validRecords(index.nativePrerequisites, m)) { return false; }
    std::vector<uint32_t> counts(k);
    for (std::size_t row = 0; row < k; ++row) {
        const auto& frontier = index.frontiers[row];
        const auto found = index.pipeRows.find(frontier.pipe);
        if (!frontier.count || frontier.distances.size() != 2 * m || found == index.pipeRows.end() ||
            found->second != row) { return false; }
    }
    for (std::size_t type = 0; type < m; ++type) {
        const auto row = index.sourceRows[type];
        if (row >= k || index.frontiers[row].pipe != index.payloads[type].pipe ||
            index.localRanks[type] != ++counts[row]) { return false; }
    }
    for (std::size_t row = 0; row < k; ++row) {
        if (counts[row] != index.frontiers[row].count) { return false; }
    }
    uint64_t previous = 0;
    for (auto record : index.retained) {
        if (record >= index.generators.size() || uint64_t(record) + 1 <= previous) { return false; }
        previous = uint64_t(record) + 1;
    }
    return true;
}
bool sameRecords(llvm::ArrayRef<PeriodicRecord> a, llvm::ArrayRef<PeriodicRecord> b)
{
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].source != b[i].source || a[i].target != b[i].target ||
            a[i].displacement != b[i].displacement) { return false; }
    }
    return true;
}
bool sameBody(const PeriodicAnalysis& a, const PeriodicAnalysis& b)
{
    if (a.payloads.size() != b.payloads.size() || !sameRecords(a.nativePrerequisites, b.nativePrerequisites)) {
        return false;
    }
    for (std::size_t i = 0; i < a.payloads.size(); ++i) {
        if (a.payloads[i].pipe != b.payloads[i].pipe) { return false; }
    }
    return true;
}
Support support(const PeriodicAnalysis& index, uint64_t& steps)
{
    const auto k = index.frontiers.size();
    Support result(k, std::vector<bool>(k));
    auto add = [&result, &index](llvm::ArrayRef<PeriodicRecord> records) {
        for (const auto& record : records) {
            result[index.sourceRows[record.source]][index.sourceRows[record.target]] = true;
        }
    };
    add(index.generators);
    add(index.nativePrerequisites);
    // Deliberately no reflexive seed: native starts/completions cannot cross C->I.
    for (std::size_t middle = 0; middle < k; ++middle) {
        for (std::size_t source = 0; source < k; ++source) {
            for (std::size_t target = 0; target < k; ++target) {
                ++steps;
                result[source][target] = result[source][target] ||
                    (result[source][middle] && result[middle][target]);
            }
        }
    }
    return result;
}
llvm::APInt phi(uint32_t count, std::optional<uint64_t> distance, uint64_t trips)
{
    if (!distance) { return wide(0); }
    const uint64_t firstVisit = std::min(trips, *distance / count);
    if (firstVisit == trips) { return wide(0); }
    const auto first = wide(count) * (wide(firstVisit) + 1) - wide(*distance);
    const auto last = wide(count) * wide(trips) - wide(*distance);
    return (wide(trips - firstVisit) * (first + last)).udiv(wide(2));
}
bool ordered(const std::optional<uint64_t>& upper, const std::optional<uint64_t>& lower)
{
    return !lower || (upper && *upper <= *lower);
}
bool accumulate(const PeriodicAnalysis& upper, const PeriodicAnalysis& lower,
                const Support& upperSupport, const Support& lowerSupport, PeriodicExcessSummary& out)
{
    auto gamma = wide(0);
    out.sameSupport = true;
    out.equalFrontiers = true;
    for (std::size_t row = 0; row < upper.frontiers.size(); ++row) {
        const auto& a = upper.frontiers[row];
        const auto& b = lower.frontiers[lower.pipeRows.at(a.pipe)];
        for (std::size_t type = 0; type < upper.payloads.size(); ++type) {
            ++out.frontierEntries;
            const auto targetRow = upper.sourceRows[type];
            const auto lowerRow = lower.pipeRows.at(a.pipe);
            const auto lowerTarget = lower.sourceRows[type];
            const auto x = a.distances[2 * type], y = b.distances[2 * type];
            if (bool(x) != upperSupport[row][targetRow] || bool(y) != lowerSupport[lowerRow][lowerTarget] ||
                !ordered(x, y) || !ordered(a.distances[2 * type + 1], b.distances[2 * type + 1])) {
                return false;
            }
            out.equalFrontiers &= x == y;
            out.sameSupport &= bool(x) == bool(y);
            out.excess += phi(a.count, x, out.trips) - phi(b.count, y, out.trips);
            if (x && y) { gamma += wide(*y - *x); }
            if (x && !y) { out.quadraticPairs += wide(a.count); }
        }
    }
    if (out.sameSupport) { out.gamma = gamma; out.linearBound = gamma * wide(out.trips); }
    return true;
}
std::optional<uint64_t> counterpartDelta(const PeriodicAnalysis& upper, const PeriodicAnalysis& lower)
{
    std::unordered_map<uint64_t, uint64_t> earliest;
    auto key = [](const PeriodicRecord& record) { return (uint64_t(record.source) << 32) | record.target; };
    auto add = [&earliest, &key](llvm::ArrayRef<PeriodicRecord> records) {
        for (const auto& record : records) {
            auto inserted = earliest.emplace(key(record), record.displacement);
            if (!inserted.second) { inserted.first->second = std::min(inserted.first->second, record.displacement); }
        }
    };
    add(lower.generators);
    add(lower.nativePrerequisites);
    uint64_t delta = 0;
    for (const auto& record : upper.generators) {
        const auto found = earliest.find(key(record));
        if (found == earliest.end()) { return std::nullopt; }
        if (found->second > record.displacement) { delta = std::max(delta, found->second - record.displacement); }
    }
    return delta;
}
} // namespace
PeriodicExcessGraph::PeriodicExcessGraph(CompactFixedBody context, PeriodicAnalysis analysis,
                                       ReductionQuality reduction)
    : owner(std::move(context)), index(std::move(analysis)), quality(reduction) {}
const OrderContext& PeriodicExcessGraph::context() const { return owner->context(); }
PeriodicExcessSnapshot bindPeriodicExcessGraph(CompactFixedBody context, const PeriodicAnalysis& analysis,
                                             ReductionQuality reduction, std::string& error)
{
    error.clear();
    if (!context || !validShape(analysis) || context->payloads().size() != analysis.payloads.size() ||
        (reduction != ReductionQuality::Covers && reduction != ReductionQuality::Partial &&
         reduction != ReductionQuality::Generators)) {
        error = "invalid periodic excess snapshot context, index or reduction";
        return {};
    }
    for (std::size_t type = 0; type < analysis.payloads.size(); ++type) {
        if (analysis.payloads[type].pipe != context->payloads()[type].pipe) {
            error = "periodic excess snapshot does not match fixed body";
            return {};
        }
    }
    return PeriodicExcessSnapshot(new PeriodicExcessGraph(std::move(context), analysis, reduction));
}
PeriodicExcessSummary analyzePeriodicExcess(const PeriodicAnalysis& a, const PeriodicAnalysis& b, uint64_t trips)
{
    PeriodicExcessSummary out;
    out.trips = trips;
    if (!validShape(a) || !validShape(b) || !sameBody(a, b)) {
        out.error = "periodic excess bounds have invalid indices, different bodies or native prerequisites";
        return out;
    }
    const uint64_t k = a.frontiers.size();
    if (k && (k > UINT64_MAX / k || k * k > UINT64_MAX / (2 * k))) {
        out.error = "periodic excess support work exceeds representation";
        return out;
    }
    const auto upperSupport = support(a, out.supportSteps), lowerSupport = support(b, out.supportSteps);
    if (!accumulate(a, b, upperSupport, lowerSupport, out)) {
        out.error = "periodic excess frontier support or lower containment is inconsistent";
        return out;
    }
    out.counterpartDelta = counterpartDelta(a, b);
    if (out.counterpartDelta) {
        out.counterpartBound = wide(trips) * wide(a.payloads.size()) * wide(a.payloads.size()) *
            wide(k) * wide(*out.counterpartDelta);
    }
    return out;
}
PeriodicExcessCertificate certifyPeriodicExcess(PeriodicExcessSnapshot upper,
                                                PeriodicExcessSnapshot lower, uint64_t trips)
{
    PeriodicExcessCertificate out;
    out.upper = std::move(upper);
    out.lower = std::move(lower);
    out.trips = trips;
    if (!out.upper) { out.error = "missing periodic excess upper snapshot"; return out; }
    const auto& a = out.upper->analysis();
    if (!out.lower) {
        const auto native = analyzePeriodicDemands(a.payloads, {}, a.nativePrerequisites);
        out.lower = bindPeriodicExcessGraph(out.upper->binding(), native, ReductionQuality::Generators, out.error);
        if (!out.lower) { return out; }
    }
    if (out.upper->binding() != out.lower->binding()) {
        out.error = "periodic excess bounds have different fixed-body contexts";
        return out;
    }
    static_cast<PeriodicExcessSummary&>(out) = analyzePeriodicExcess(a, out.lower->analysis(), trips);
    if (out.error.empty() && out.equalFrontiers) {
        out.guarantee = InputOrderGuarantee::InputOrderEquivalent;
        out.exactOriginalCovers = out.upper->reduction() == ReductionQuality::Covers;
    }
    return out;
}
std::string periodicExcessDecimal(const llvm::APInt& value)
{
    llvm::SmallString<80> buffer;
    value.toString(buffer, 10, false);
    return std::string(buffer);
}
} // namespace mlir::pto::frontiersynch
