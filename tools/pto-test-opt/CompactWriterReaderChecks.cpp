// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite graph oracle for compact circular distances, consolidation and placement.
#include "PTO/Transforms/FrontierSynch/CompactWriterReader.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <map>
#include <tuple>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
using Key = std::tuple<uint32_t, uint32_t, uint64_t>;
std::vector<Key> keys(llvm::ArrayRef<fs::PeriodicRecord> records)
{
    std::vector<Key> result;
    for (const auto& record : records) { result.emplace_back(record.source, record.target, record.displacement); }
    std::sort(result.begin(), result.end());
    return result;
}
Graph close(Graph graph)
{
    for (std::size_t middle = 0; middle < graph.size(); ++middle) {
        for (std::size_t source = 0; source < graph.size(); ++source) {
            if (!graph[source][middle]) { continue; }
            for (std::size_t target = 0; target < graph.size(); ++target) {
                graph[source][target] = graph[source][target] || graph[middle][target];
            }
        }
    }
    return graph;
}
Graph direct(llvm::ArrayRef<fs::PeriodicPayload> payloads, llvm::ArrayRef<fs::PeriodicRecord> records, uint32_t trips)
{
    const auto count = payloads.size() * trips;
    Graph graph(2 * count, std::vector<bool>(2 * count));
    for (std::size_t source = 0; source < count; ++source) {
        graph[2*source][2*source] = graph[2*source+1][2*source+1] = graph[2*source][2*source+1] = true;
        for (std::size_t target = source + 1; target < count; ++target) {
            if (payloads[source % payloads.size()].pipe == payloads[target % payloads.size()].pipe) {
                graph[2*source][2*target] = graph[2*source+1][2*target+1] = true;
            }
        }
    }
    for (const auto& record : records) {
        for (uint32_t iteration = 0; iteration < trips; ++iteration) {
            if (record.displacement >= trips - iteration) { continue; }
            const auto source = iteration * payloads.size() + record.source;
            const auto target = (iteration + record.displacement) * payloads.size() + record.target;
            graph[2*source+1][2*target] = true;
        }
    }
    return graph;
}
bool subset(const Graph& a, const Graph& b)
{
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a.size(); ++j) { if (a[i][j] && !b[i][j]) { return false; } }
    }
    return true;
}
std::vector<fs::PeriodicRecord> retained(const fs::PeriodicAnalysis& analysis)
{
    std::vector<fs::PeriodicRecord> result;
    for (auto id : analysis.retained) { result.push_back(analysis.generators[id]); }
    return result;
}
std::vector<fs::CompactSourceQuery> allQueries(llvm::ArrayRef<fs::CompactClassAccess> accesses)
{
    std::vector<fs::CompactSourceQuery> queries;
    for (uint32_t a = 0; a < accesses.size(); ++a) {
        for (uint32_t b = 0; b < accesses.size(); ++b) {
            if (accesses[a].storageClass != accesses[b].storageClass) { continue; }
            if (accesses[a].write && accesses[b].read) { queries.push_back({a, b, fs::StorageHazard::RAW}); }
            if (accesses[a].read && accesses[b].write) { queries.push_back({a, b, fs::StorageHazard::WAR}); }
            if (accesses[a].write && accesses[b].write) { queries.push_back({a, b, fs::StorageHazard::WAW}); }
        }
    }
    return queries;
}
bool distances(uint32_t sites, llvm::ArrayRef<fs::CompactClassAccess> accesses)
{
    auto index = fs::indexFixedBodyOverwrites(sites, accesses);
    if (!index.error.empty() || index.indexingSteps > 3 * accesses.size()) { return false; }
    for (uint32_t origin = 0; origin < accesses.size(); ++origin) {
        for (uint32_t target = 0; target < sites; ++target) {
            std::optional<uint64_t> minimum;
            bool killed = false;
            for (uint64_t step = 1; step <= sites; ++step) {
                const auto absolute = uint64_t(accesses[origin].site) + step;
                const auto site = absolute % sites;
                if (site == target) { minimum = absolute / sites; }
                for (const auto& access : accesses) {
                    killed |= access.site == site && access.storageClass == accesses[origin].storageClass &&
                              access.fullOverwrite;
                }
                if (killed) { break; }
            }
            auto actual = index.distances(origin, target);
            if (!actual || actual->reachable != bool(minimum)) { return false; }
            if (minimum && (actual->minimum != *minimum ||
                actual->maximum != (killed ? minimum : std::nullopt))) { return false; }
        }
    }
    return !index.distances(static_cast<uint32_t>(accesses.size()), 0);
}
std::vector<fs::PeriodicRecord> adjacent(llvm::ArrayRef<fs::PeriodicPayload> payloads,
                                       llvm::ArrayRef<fs::PeriodicRecord> records)
{
    std::vector<fs::PeriodicRecord> result(records.begin(), records.end());
    for (auto& record : result) {
        if (payloads[record.source].pipe != payloads[record.target].pipe) { continue; }
        std::optional<uint32_t> before, last;
        for (uint32_t site = 0; site < payloads.size(); ++site) {
            if (payloads[site].pipe != payloads[record.target].pipe) { continue; }
            last = site;
            if (site < record.target) { before = site; }
        }
        record.source = before ? *before : *last;
        record.displacement = before ? 0 : 1;
    }
    return result;
}
Graph conflicts(llvm::ArrayRef<fs::PeriodicPayload> payloads,
                llvm::ArrayRef<fs::CompactClassAccess> accesses, uint32_t trips)
{
    auto graph = direct(payloads, {}, trips);
    const auto count = payloads.size() * trips;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = a + 1; b < count; ++b) {
            for (const auto& source : accesses) {
                if (source.site != a % payloads.size()) { continue; }
                for (const auto& target : accesses) {
                    if (target.site == b % payloads.size() && source.storageClass == target.storageClass &&
                        (source.write || target.write)) { graph[2*a+1][2*b] = true; }
                }
            }
        }
    }
    return close(std::move(graph));
}
bool endpoints(const fs::LogicalEndpointPlan& plan, uint32_t trips)
{
    std::map<std::pair<uint32_t, uint64_t>, std::array<unsigned, 2>> matches;
    for (uint32_t recipe = 0; recipe < plan.recipes.size(); ++recipe) {
        for (uint32_t ordinal = 0; ordinal <= trips; ++ordinal) {
            auto value = plan.evaluate(recipe, trips, ordinal);
            if (value.error != fs::EndpointError::None || (ordinal == trips && value.active)) { return false; }
            if (!value.active || value.instance.kind == fs::EndpointKind::Barrier) { continue; }
            auto key = std::make_pair(value.instance.identity.record, value.instance.identity.sourceOrdinal);
            ++matches[key][value.instance.kind == fs::EndpointKind::Set ? 0 : 1];
        }
    }
    for (const auto& item : matches) { if (item.second != std::array<unsigned, 2>{{1, 1}}) { return false; } }
    return true;
}
bool compare(llvm::ArrayRef<fs::PeriodicPayload> payloads, llvm::ArrayRef<fs::CompactClassAccess> accesses,
             uint64_t& checked)
{
    if (!distances(static_cast<uint32_t>(payloads.size()), accesses)) { return false; }
    auto queries = allQueries(accesses);
    const auto indexed = fs::analyzeCompactWriterReader(payloads, accesses, queries);
    const auto coarse = fs::analyzeCoarseCompactWriterReader(payloads, accesses);
    if (!indexed.error.empty() || !coarse.error.empty() ||
        keys(indexed.consolidated) != keys(coarse.consolidated) ||
        keys(retained(indexed.upper)) != keys(retained(coarse.upper)) ||
        indexed.cost.distanceQueries != queries.size() || coarse.cost.distanceQueries != 0 ||
        indexed.cost.quotientPasses != 2 || coarse.cost.historyVisits > 5 * payloads.size() * accesses.size()) {
        llvm::errs() << "compact engine mismatch: " << indexed.error << " / " << coarse.error << "\n";
        return false;
    }
    std::vector<fs::PeriodicRecord> selected;
    for (const auto& candidate : indexed.candidates) {
        if (!candidate.distances.reachable) { continue; }
        if (!candidate.consolidated || *candidate.consolidated >= indexed.consolidated.size()) { return false; }
        const auto& winner = indexed.consolidated[*candidate.consolidated];
        if (winner.target != candidate.selected.target || payloads[winner.source].pipe !=
            payloads[candidate.selected.source].pipe || winner.displacement > candidate.selected.displacement ||
            (winner.displacement == candidate.selected.displacement && winner.source < candidate.selected.source)) {
            return false;
        }
        selected.push_back(candidate.selected);
    }
    for (uint32_t trips = 0; trips <= 3; ++trips) {
        auto all = close(direct(payloads, selected, trips));
        auto consolidated = close(direct(payloads, indexed.consolidated, trips));
        auto first = close(direct(payloads, retained(indexed.firstReduction), trips));
        auto upper = close(direct(payloads, retained(indexed.upper), trips));
        auto early = close(direct(payloads, adjacent(payloads, indexed.consolidated), trips));
        if (!subset(conflicts(payloads, accesses, trips), all) || all != consolidated || consolidated != first ||
            !subset(first, upper) || !subset(upper, early) || !endpoints(indexed.endpoints, trips)) { return false; }
        ++checked;
    }
    return true;
}
bool exhaustive(uint64_t& checked)
{
    for (unsigned modes = 0; modes < 64; ++modes) {
        for (unsigned assignments = 0; assignments < 8; ++assignments) {
            std::vector<fs::PeriodicPayload> payloads;
            std::vector<fs::CompactClassAccess> accesses;
            for (uint32_t site = 0; site < 3; ++site) {
                const auto mode = (modes >> (2 * site)) & 3U;
                payloads.push_back({(assignments >> site) & 1U});
                accesses.push_back({site, 71, mode == 0 || mode == 3, mode != 0, mode >= 2});
            }
            if (!compare(payloads, accesses, checked)) { return false; }
        }
    }
    // Independent sparse classes at a shared RMW position, and an empty body.
    return compare({{7}, {99}, {7}}, {{0, 1, true, false}, {1, 1, true, true, true},
        {1, UINT32_MAX, false, true}, {2, UINT32_MAX, true, false}}, checked) && compare({}, {}, checked);
}
bool specificAndStartup()
{
    auto filtered = fs::analyzeCompactWriterReader({{7}, {7}, {9}},
        {{0, 1, false, true}, {1, 1, false, true}, {2, 1, true, false}}, {{0, 2, fs::StorageHazard::RAW}});
    if (!filtered.error.empty() || keys(filtered.consolidated) != std::vector<Key>{{0, 2, 0}}) { return false; }
    // A;X;R;B: the relay removes A->B before X could become a stronger local source.
    auto delayed = fs::analyzeCompactWriterReader({{1}, {1}, {2}, {1}}, {}, {},
        {{0, 3, {true, 0, 0}}, {0, 2, {true, 0, 0}}, {2, 3, {true, 0, 0}}});
    if (!delayed.error.empty() || keys(retained(delayed.upper)) != std::vector<Key>{{0, 2, 0}, {2, 3, 0}}) {
        return false;
    }
    auto startup = fs::analyzeCompactWriterReader({{17}}, {}, {}, {{0, 0, {true, 7, 7}}});
    if (!startup.error.empty() || startup.adjacency.size() != 1 ||
        keys(retained(startup.firstReduction)) != std::vector<Key>{{0, 0, 7}} ||
        keys(retained(startup.upper)) != std::vector<Key>{{0, 0, 1}}) { return false; }
    auto native = fs::analyzeCompactWriterReader({{0}, {1}}, {}, {}, {{0, 1, {true, 0, 0}}}, {{0, 1, 0}});
    return native.error.empty() && native.upper.retained.empty() &&
           native.firstReduction.nativePrerequisites.size() == 1 &&
           native.upper.nativePrerequisites.size() == 1;
}
bool boundsAndInvalid()
{
    constexpr uint64_t huge = uint64_t(1) << 40;
    auto result = fs::analyzeCompactWriterReader({{0}, {1}},
        {{0, 4, false, true}, {1, 4, true, false}}, {{0, 1, fs::StorageHazard::RAW, {true, huge, huge}}});
    if (!result.error.empty() || keys(retained(result.upper)) != std::vector<Key>{{0, 1, huge}} ||
        result.endpoints.recipes.size() != 2) { return false; }
    for (uint32_t recipe = 0; recipe < 2; ++recipe) {
        const auto ordinal = result.endpoints.recipes[recipe].kind == fs::EndpointKind::Set ? 0 : huge;
        auto enabled = result.endpoints.evaluate(recipe, huge + 1, ordinal);
        auto absent = result.endpoints.evaluate(recipe, huge, ordinal);
        if (enabled.error != fs::EndpointError::None || !enabled.active || enabled.instance.identity.sourceOrdinal ||
            absent.error != fs::EndpointError::None || absent.active) { return false; }
    }
    auto killed = fs::analyzeCompactWriterReader({{0}, {1}},
        {{0, 4, false, true, true}, {1, 4, true, false}}, {{0, 1, fs::StorageHazard::RAW, {true, 1, 1}}});
    if (!killed.error.empty() || !killed.upper.retained.empty() || killed.candidates[0].distances.reachable) {
        return false;
    }
    if (fs::indexFixedBodyOverwrites(2, {{1, 0, true, false}, {0, 0, true, false}}).error.empty() ||
        fs::indexFixedBodyOverwrites(1, {{0, 0, true, false}, {0, 0, false, true}}).error.empty() ||
        fs::indexFixedBodyOverwrites(1, {{0, 0, true, false, true}}).error.empty()) { return false; }
    if (fs::analyzeCompactWriterReader({{0}}, {{0, 0, false, true}},
        {{0, 0, fs::StorageHazard::WAW, {true, 3, 2}}}).error.empty() ||
        fs::analyzeCompactWriterReader({{0}}, {{0, 0, false, true}},
        {{0, 0, fs::StorageHazard::Supplied}}).error.empty() ||
        fs::analyzeCompactWriterReader({}, {}, {}, {{0, 0}}).error.empty()) { return false; }
    auto overflow = fs::analyzeCompactWriterReader({{0}}, {}, {}, {{0, 0, {true, UINT64_MAX, UINT64_MAX}}});
    return !overflow.error.empty();
}
} // namespace
int runCompactWriterReaderChecks()
{
    uint64_t checked = 0;
    if (!exhaustive(checked)) { llvm::errs() << "compact writer/reader finite graph oracle failed\n"; return 1; }
    if (!specificAndStartup()) { llvm::errs() << "compact filtering/local startup check failed\n"; return 1; }
    if (!boundsAndInvalid()) { llvm::errs() << "compact interval/representation check failed\n"; return 1; }
    llvm::outs() << "compact writer/reader: " << checked << " finite graph envelopes passed\n";
    return 0;
}
