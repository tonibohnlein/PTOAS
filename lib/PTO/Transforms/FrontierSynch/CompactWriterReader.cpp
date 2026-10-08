// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactWriterReader.h"
#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
struct ClassEntries {
    std::vector<uint32_t> accesses, kills;
};
bool validInterval(const OriginDistanceInterval& interval)
{
    return !interval.reachable || !interval.maximum || interval.minimum <= *interval.maximum;
}
OriginDistanceInterval intersect(OriginDistanceInterval a, const OriginDistanceInterval& b)
{
    if (!a.reachable || !b.reachable) { return {}; }
    a.minimum = std::max(a.minimum, b.minimum);
    if (b.maximum) { a.maximum = a.maximum ? std::min(*a.maximum, *b.maximum) : b.maximum; }
    a.reachable = !a.maximum || a.minimum <= *a.maximum;
    return a;
}
uint64_t advance(uint32_t sites, uint32_t source, uint32_t target)
{
    return target > source ? uint64_t(target) - source : uint64_t(sites) - source + target;
}
bool validModes(const CompactClassAccess& source, const CompactClassAccess& target, StorageHazard hazard)
{
    if (source.storageClass != target.storageClass) { return false; }
    switch (hazard) {
        case StorageHazard::RAW: return source.write && target.read;
        case StorageHazard::WAR: return source.read && target.write;
        case StorageHazard::WAW: return source.write && target.write;
        default: return false;
    }
}
using RecordKey = std::tuple<uint32_t, uint32_t, uint64_t>;
RecordKey key(const PeriodicRecord& record) { return {record.source, record.target, record.displacement}; }
struct History {
    std::optional<uint32_t> access;
    unsigned body = 0;
};
class Constructor {
public:
    explicit Constructor(llvm::ArrayRef<PeriodicPayload> payloads) : payloads(payloads) {}
    CompactWriterReaderAnalysis result;
    bool initialize();
    bool indexed(const FixedBodyOverwriteIndex& index, llvm::ArrayRef<CompactSourceQuery> queries);
    bool coarse(const FixedBodyOverwriteIndex& index);
    bool finish(llvm::ArrayRef<CompactAdditionalRequirement> additional,
                llvm::ArrayRef<PeriodicRecord> nativePrerequisites);
private:
    bool fail(const char* message) { result.error = message; return false; }
    bool add(CompactSelectedCandidate candidate);
    bool extras(llvm::ArrayRef<CompactAdditionalRequirement> additional);
    bool reduce(llvm::ArrayRef<PeriodicRecord> nativePrerequisites);
    bool emitHistory(const FixedBodyOverwriteIndex& index, const History& entry, uint32_t target, StorageHazard hazard);
    llvm::ArrayRef<PeriodicPayload> payloads;
    std::vector<uint32_t> pipeRows;
    std::vector<std::optional<uint32_t>> winners;
    std::size_t pipes = 0;
};
bool Constructor::initialize()
{
    if (payloads.size() > UINT32_MAX / 3) { return fail("compact body exceeds periodic identity representation"); }
    std::unordered_map<uint32_t, uint32_t> rows;
    for (const auto& payload : payloads) {
        auto found = rows.emplace(payload.pipe, static_cast<uint32_t>(rows.size())).first;
        pipeRows.push_back(found->second);
    }
    pipes = rows.size();
    if (pipes && payloads.size() > winners.max_size() / pipes) {
        return fail("compact consolidation table exceeds representation");
    }
    winners.resize(payloads.size() * pipes);
    result.cost.consolidationSlots = winners.size();
    return true;
}
bool Constructor::add(CompactSelectedCandidate candidate)
{
    if (result.candidates.size() >= UINT32_MAX) { return fail("compact candidate identity overflow"); }
    const auto id = static_cast<uint32_t>(result.candidates.size());
    if (candidate.distances.reachable) {
        auto& selected = candidate.selected;
        selected.displacement = candidate.distances.minimum;
        if (selected.source >= payloads.size() || selected.target >= payloads.size() ||
            (!selected.displacement && selected.source >= selected.target)) {
            return fail("compact candidate is not reference-forward");
        }
        auto& winner = winners[std::size_t(selected.target) * pipes + pipeRows[selected.source]];
        if (!winner || selected.displacement < result.candidates[*winner].selected.displacement ||
            (selected.displacement == result.candidates[*winner].selected.displacement &&
             selected.source > result.candidates[*winner].selected.source)) { winner = id; }
    }
    result.candidates.push_back(std::move(candidate));
    ++result.cost.candidates;
    return true;
}
bool Constructor::indexed(const FixedBodyOverwriteIndex& index, llvm::ArrayRef<CompactSourceQuery> queries)
{
    if (queries.size() > UINT32_MAX) { return fail("compact query identity overflow"); }
    for (uint32_t id = 0; id < queries.size(); ++id) {
        const auto& query = queries[id];
        if (query.sourceAccess >= index.accesses.size() || query.targetAccess >= index.accesses.size() ||
            !validInterval(query.bounds)) { return fail("compact query has invalid access indices or bounds"); }
        const auto& source = index.accesses[query.sourceAccess];
        const auto& target = index.accesses[query.targetAccess];
        if (!validModes(source, target, query.hazard)) { return fail("compact query has incompatible class or modes"); }
        auto interval = index.distances(query.sourceAccess, target.site);
        ++result.cost.distanceQueries;
        if (!interval) { return fail("compact overwrite query is invalid"); }
        CompactSelectedCandidate candidate;
        candidate.query = id; candidate.sourceAccess = query.sourceAccess; candidate.targetAccess = query.targetAccess;
        candidate.hazard = query.hazard; candidate.distances = intersect(*interval, query.bounds);
        candidate.selected = {source.site, target.site, 0};
        if (!add(std::move(candidate))) { return false; }
    }
    return true;
}
bool Constructor::emitHistory(
    const FixedBodyOverwriteIndex& index, const History& entry, uint32_t target, StorageHazard hazard)
{
    ++result.cost.historyVisits;
    if (!entry.access) { return true; }
    const auto& source = index.accesses[*entry.access];
    auto interval = index.distances(*entry.access, index.accesses[target].site);
    if (!interval || !interval->reachable || interval->minimum != 1U - entry.body) {
        return fail("compact history disagrees with circular overwrite interval");
    }
    CompactSelectedCandidate candidate;
    candidate.sourceAccess = *entry.access; candidate.targetAccess = target; candidate.hazard = hazard;
    candidate.distances = *interval; candidate.selected = {source.site, index.accesses[target].site, 0};
    return add(std::move(candidate));
}
bool Constructor::coarse(const FixedBodyOverwriteIndex& index)
{
    if (pipes && index.accesses.size() > UINT64_MAX / (5 * uint64_t(pipes))) {
        return fail("compact history operation count overflow");
    }
    std::unordered_map<uint32_t, std::size_t> classes;
    std::vector<std::size_t> accessRows;
    for (const auto& access : index.accesses) {
        auto row = classes.emplace(access.storageClass, classes.size()).first->second;
        accessRows.push_back(row);
    }
    std::vector<History> writers, readers;
    if (pipes && classes.size() > writers.max_size() / pipes) { return fail("compact history size overflow"); }
    writers.resize(classes.size() * pipes); readers.resize(writers.size());
    for (unsigned body = 0; body < 2; ++body) {
        for (uint32_t id = 0; id < index.accesses.size(); ++id) {
            const auto& access = index.accesses[id];
            const auto row = accessRows[id] * pipes;
            for (std::size_t pipe = 0; pipe < pipes; ++pipe) {
                if (body) {
                    if (access.read && !emitHistory(index, writers[row + pipe], id, StorageHazard::RAW)) {
                        return false;
                    }
                    if (access.write && (!emitHistory(index, writers[row + pipe], id, StorageHazard::WAW) ||
                        !emitHistory(index, readers[row + pipe], id, StorageHazard::WAR))) { return false; }
                }
                if (access.fullOverwrite) { writers[row + pipe] = {}; readers[row + pipe] = {}; }
                ++result.cost.historyVisits;
            }
            const auto own = row + pipeRows[access.site];
            if (access.write) { writers[own] = {id, body}; }
            if (access.read) { readers[own] = {id, body}; }
        }
    }
    return true;
}
bool Constructor::extras(llvm::ArrayRef<CompactAdditionalRequirement> additional)
{
    if (additional.size() > UINT32_MAX) { return fail("compact prerequisite identity overflow"); }
    for (uint32_t id = 0; id < additional.size(); ++id) {
        const auto& requirement = additional[id];
        if (requirement.source >= payloads.size() || requirement.target >= payloads.size() ||
            !validInterval(requirement.bounds)) { return fail("compact prerequisite has invalid endpoints or bounds"); }
        CompactSelectedCandidate candidate;
        candidate.additional = id;
        candidate.distances = intersect(requirement.bounds,
            {true, requirement.source >= requirement.target ? 1U : 0U, std::nullopt});
        candidate.selected = {requirement.source, requirement.target, 0};
        if (!add(std::move(candidate))) { return false; }
    }
    return true;
}
bool Constructor::reduce(llvm::ArrayRef<PeriodicRecord> nativePrerequisites)
{
    result.firstReduction = analyzePeriodicDemands(payloads, result.consolidated, nativePrerequisites);
    if (!result.firstReduction.error.empty()) { result.error = result.firstReduction.error; return false; }
    std::map<RecordKey, uint32_t> firstIds;
    for (uint32_t id = 0; id < result.firstReduction.generators.size(); ++id) {
        firstIds.emplace(key(result.firstReduction.generators[id]), id);
    }
    for (const auto& record : result.consolidated) {
        auto found = firstIds.find(key(record));
        if (found == firstIds.end()) { return fail("first reduction lost a canonical candidate identity"); }
        result.consolidatedToFirst.push_back(found->second);
    }
    std::vector<uint32_t> previous(payloads.size()), last(pipes, UINT32_MAX);
    for (uint32_t site = 0; site < payloads.size(); ++site) {
        previous[site] = last[pipeRows[site]]; last[pipeRows[site]] = site;
    }
    std::vector<PeriodicRecord> adjacent;
    for (auto id : result.firstReduction.retained) {
        auto record = result.firstReduction.generators[id];
        if (pipeRows[record.source] == pipeRows[record.target]) {
            auto prior = previous[record.target];
            record.source = prior == UINT32_MAX ? last[pipeRows[record.target]] : prior;
            record.displacement = record.source >= record.target ? 1 : 0;
        }
        adjacent.push_back(record);
    }
    result.upper = analyzePeriodicDemands(payloads, adjacent, nativePrerequisites);
    if (!result.upper.error.empty()) { result.error = result.upper.error; return false; }
    std::map<RecordKey, uint32_t> upperIds;
    for (uint32_t id = 0; id < result.upper.generators.size(); ++id) {
        upperIds.emplace(key(result.upper.generators[id]), id);
    }
    for (std::size_t i = 0; i < adjacent.size(); ++i) {
        auto found = upperIds.find(key(adjacent[i]));
        if (found == upperIds.end()) { return fail("upper reduction lost an adjacent record identity"); }
        result.adjacency.push_back({result.firstReduction.retained[i], found->second});
    }
    result.endpoints = buildLogicalEndpoints(result.upper);
    result.cost.quotientPasses = 2;
    result.cost.quotientEdges = result.firstReduction.graphEdges + result.upper.graphEdges;
    if (!result.endpoints.error.empty()) { result.error = result.endpoints.error; return false; }
    return true;
}
bool Constructor::finish(llvm::ArrayRef<CompactAdditionalRequirement> additional,
                         llvm::ArrayRef<PeriodicRecord> nativePrerequisites)
{
    if (!extras(additional)) { return false; }
    std::vector<std::optional<uint32_t>> consolidated(winners.size());
    for (std::size_t slot = 0; slot < winners.size(); ++slot) {
        if (!winners[slot]) { continue; }
        consolidated[slot] = static_cast<uint32_t>(result.consolidated.size());
        result.consolidated.push_back(result.candidates[*winners[slot]].selected);
    }
    for (auto& candidate : result.candidates) {
        if (candidate.distances.reachable) {
            candidate.consolidated = consolidated[std::size_t(candidate.selected.target) * pipes +
                                                   pipeRows[candidate.selected.source]];
        }
    }
    return reduce(nativePrerequisites);
}
} // namespace
FixedBodyOverwriteIndex indexFixedBodyOverwrites(uint32_t sites, llvm::ArrayRef<CompactClassAccess> accesses)
{
    FixedBodyOverwriteIndex result;
    result.sites = sites;
    if (accesses.size() > UINT32_MAX) { result.error = "compact access identity overflow"; return result; }
    std::unordered_map<uint32_t, ClassEntries> classes;
    uint32_t previous = 0;
    for (uint32_t id = 0; id < accesses.size(); ++id) {
        const auto& access = accesses[id];
        if (access.site >= sites || (id && access.site < previous) || (!access.read && !access.write) ||
            (access.fullOverwrite && !access.write)) {
            result.error = "compact accesses require ordered valid sites, modes and certified writer kills";
            return result;
        }
        auto& entries = classes[access.storageClass];
        if (!entries.accesses.empty() && accesses[entries.accesses.back()].site == access.site) {
            result.error = "compact class access is duplicated at one site"; return result;
        }
        entries.accesses.push_back(id);
        if (access.fullOverwrite) { entries.kills.push_back(access.site); }
        previous = access.site; ++result.indexingSteps;
    }
    result.accesses.assign(accesses.begin(), accesses.end());
    result.nextOverwrite.resize(accesses.size());
    for (const auto& item : classes) {
        const auto& entries = item.second;
        std::size_t cursor = 0;
        for (auto id : entries.accesses) {
            ++result.indexingSteps;
            if (entries.kills.empty()) { continue; }
            const auto site = accesses[id].site;
            while (cursor < entries.kills.size() && entries.kills[cursor] <= site) { ++cursor; ++result.indexingSteps; }
            auto next = cursor < entries.kills.size() ? entries.kills[cursor] : entries.kills.front();
            result.nextOverwrite[id] = advance(sites, site, next);
        }
    }
    return result;
}
std::optional<OriginDistanceInterval> FixedBodyOverwriteIndex::distances(
    uint32_t originAccess, uint32_t querySite) const
{
    if (!error.empty() || originAccess >= accesses.size() || nextOverwrite.size() != accesses.size() ||
        querySite >= sites || accesses[originAccess].site >= sites) { return std::nullopt; }
    const auto source = accesses[originAccess].site;
    const auto kill = nextOverwrite[originAccess];
    if (kill && (!*kill || *kill > sites)) { return std::nullopt; }
    if (kill && advance(sites, source, querySite) > *kill) { return OriginDistanceInterval{}; }
    const uint64_t minimum = querySite <= source ? 1 : 0;
    return OriginDistanceInterval{true, minimum, kill ? std::optional<uint64_t>(minimum) : std::nullopt};
}
CompactWriterReaderAnalysis analyzeCompactWriterReader(llvm::ArrayRef<PeriodicPayload> payloads,
    llvm::ArrayRef<CompactClassAccess> accesses, llvm::ArrayRef<CompactSourceQuery> queries,
    llvm::ArrayRef<CompactAdditionalRequirement> additional, llvm::ArrayRef<PeriodicRecord> nativePrerequisites)
{
    Constructor builder(payloads);
    if (!builder.initialize()) { return std::move(builder.result); }
    auto index = indexFixedBodyOverwrites(static_cast<uint32_t>(payloads.size()), accesses);
    builder.result.cost.indexingSteps = index.indexingSteps;
    if (!index.error.empty()) { builder.result.error = index.error; return std::move(builder.result); }
    if (builder.indexed(index, queries)) { builder.finish(additional, nativePrerequisites); }
    return std::move(builder.result);
}
CompactWriterReaderAnalysis analyzeCoarseCompactWriterReader(llvm::ArrayRef<PeriodicPayload> payloads,
    llvm::ArrayRef<CompactClassAccess> accesses, llvm::ArrayRef<CompactAdditionalRequirement> additional,
    llvm::ArrayRef<PeriodicRecord> nativePrerequisites)
{
    Constructor builder(payloads);
    if (!builder.initialize()) { return std::move(builder.result); }
    auto index = indexFixedBodyOverwrites(static_cast<uint32_t>(payloads.size()), accesses);
    builder.result.cost.indexingSteps = index.indexingSteps;
    if (!index.error.empty()) { builder.result.error = index.error; return std::move(builder.result); }
    if (builder.coarse(index)) { builder.finish(additional, nativePrerequisites); }
    return std::move(builder.result);
}
} // namespace mlir::pto::frontiersynch
