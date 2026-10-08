// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactBoundaryRanks.h"
#include "PTO/Transforms/FrontierSynch/CompactOrderBounds.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
bool sameNative(const PeriodicAnalysis& a, const PeriodicAnalysis& b)
{
    if (a.nativePrerequisites.size() != b.nativePrerequisites.size()) { return false; }
    for (std::size_t i = 0; i < a.nativePrerequisites.size(); ++i) {
        const auto& x = a.nativePrerequisites[i];
        const auto& y = b.nativePrerequisites[i];
        if (x.source != y.source || x.target != y.target || x.displacement != y.displacement) { return false; }
    }
    return true;
}
bool contained(const PeriodicAnalysis& upper, const PeriodicAnalysis& lower, uint64_t& entries)
{
    // Snapshot factories have already checked every row and reciprocal map.
    // A common qualified binding fixes the payload word and therefore ranks.
    for (const auto& row : upper.frontiers) {
        const auto& other = lower.frontiers[lower.pipeRows.at(row.pipe)];
        for (std::size_t vertex = 0; vertex < row.distances.size(); ++vertex) {
            ++entries;
            const auto a = row.distances[vertex], b = other.distances[vertex];
            if (b && (!a || *a > *b)) { return false; }
        }
    }
    return true;
}
bool makeSelectors(const PeriodicAnalysis& graph, uint64_t trips, llvm::ArrayRef<RegionalEvent> ports,
                   llvm::ArrayRef<uint64_t> ordinals, llvm::ArrayRef<uint64_t> counts,
                   llvm::ArrayRef<uint32_t> rows, BoundaryExcessSelectors& out, uint64_t& queries)
{
    out.prefix.assign(ports.size(), std::vector<uint64_t>(counts.size(), 0));
    out.suffixExcluded.assign(ports.size(), std::vector<uint64_t>(counts.begin(), counts.end()));
    for (std::size_t p = 0; p < ports.size(); ++p) {
        const auto ordinal = ordinals[p];
        if (ordinal >= trips) { continue; }
        const PeriodicEvent event{ports[p].type, ports[p].kind};
        for (uint32_t type = 0; type < graph.payloads.size(); ++type) {
            const auto row = rows[type];
            const uint64_t width = graph.frontiers[graph.sourceRows[type]].count;
            const uint64_t rank = graph.localRanks[type]; // One based in the word.
            auto before = graph.eventThreshold({type, PeriodicEventKind::Completion}, event);
            auto after = graph.eventThreshold(event, {type, PeriodicEventKind::Start});
            queries += 2; // Factory bounds 4*P*m before entering either sign.
            if (before.error != PeriodicQueryError::None || after.error != PeriodicQueryError::None) { return false; }
            if (before.displacement && *before.displacement <= ordinal) {
                // The largest reaching ordinal is ordinal-d. Native completion
                // order makes its rank cover every earlier completion on pipe.
                const auto latest = width * (ordinal - *before.displacement) + rank;
                out.prefix[p][row] = std::max(out.prefix[p][row], latest);
            }
            if (after.displacement && *after.displacement < trips - ordinal) {
                // This guard proves ordinal+d<T before addition; count=h*T
                // bounds every multiplication and the final rank-minus-one.
                const auto earliest = width * (ordinal + *after.displacement) + rank - 1;
                out.suffixExcluded[p][row] = std::min(out.suffixExcluded[p][row], earliest);
            }
        }
    }
    return true;
}
} // namespace
std::shared_ptr<const CompactBoundaryRanks> captureCompactBoundaryRanks(
    PeriodicExcessSnapshot upper, PeriodicExcessSnapshot lower, uint64_t trips,
    llvm::ArrayRef<RegionalEvent> ports, std::string& error)
{
    error.clear();
    if (!upper || !lower || upper->binding() != lower->binding() || !sameNative(upper->analysis(), lower->analysis())) {
        error = "compact boundary ranks require two snapshots on the same qualified body and native graph";
        return {};
    }
    auto result = std::shared_ptr<CompactBoundaryRanks>(new CompactBoundaryRanks());
    result->upperGraph = std::move(upper); result->lowerGraph = std::move(lower); result->tripCount = trips;
    const auto& a = result->upperGraph->analysis();
    const auto& b = result->lowerGraph->analysis();
    const auto m = a.payloads.size();
    if (ports.size() > UINT32_MAX || (m && ports.size() > UINT64_MAX / 4 / m)) {
        error = "compact boundary rank query count overflow"; return {};
    }
    auto& arena = *result->upperGraph->context()->expressions();
    if (!arena.constructionError().empty()) { error = arena.constructionError(); return {}; }
    if (!contained(a, b, result->entries)) { error = "compact lower frontier is not contained in upper"; return {}; }
    for (const auto& row : a.frontiers) {
        if (trips && row.count > UINT64_MAX / trips) {
            error = "compact boundary pipe count overflow"; return {};
        }
        result->labels.push_back(row.pipe); result->sizes.push_back(uint64_t(row.count) * trips);
    }
    std::vector<uint64_t> ordinals;
    for (const auto& port : ports) {
        if (port.type >= m || !validRegionalEvent(result->upperGraph->context()->domain(), port) ||
            !port.visits.empty()) {
            error = "compact boundary port is outside the fixed-body occurrence frame"; return {};
        }
        auto ordinal = arena.constantValue(port.ordinal);
        if (!ordinal) { error = "compact boundary port ordinal is not an evaluated constant"; return {}; }
        ordinals.push_back(*ordinal); result->presence.push_back(*ordinal < trips);
    }
    result->events.assign(ports.begin(), ports.end());
    // Canonical output columns use the upper snapshot's directory for both signs.
    std::vector<uint32_t> rows;
    for (const auto& payload : a.payloads) { rows.push_back(a.pipeRows.at(payload.pipe)); }
    if (!makeSelectors(a, trips, ports, ordinals, result->sizes, rows, result->upperSelectors, result->queries) ||
        !makeSelectors(b, trips, ports, ordinals, result->sizes, rows, result->lowerSelectors, result->queries)) {
        error = "compact boundary threshold query failed"; return {};
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
