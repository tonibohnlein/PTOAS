// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A reusable lane needs a return path through guaranteed-present occurrences.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicSharedAllocation.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PeriodicAnalysisInternal.h"
#include "BoundedLifetimeAllocationInternal.h"
#include "BoundedLifetimeInactiveRecords.h"
#include <algorithm>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
struct Link { PeriodicEvent event; uint64_t distance; };
struct Witness {
    PeriodicAnalysis mandatory;
    std::vector<std::vector<Link>> outgoing, incoming;
    std::optional<uint64_t> threshold(uint32_t from, uint32_t to) const
    {
        std::optional<uint64_t> best;
        for (const auto& a : outgoing[from]) {
            for (const auto& b : incoming[to]) {
                auto inner = mandatory.eventThreshold(a.event, b.event);
                uint64_t distance;
                if (inner.error != PeriodicQueryError::None || !inner.displacement ||
                    !periodic::add(a.distance, *inner.displacement, distance) ||
                    !periodic::add(distance, b.distance, distance)) { continue; }
                if (!best || distance < *best) { best = distance; }
            }
        }
        return best;
    }
};
bool conflict(RegionExpressions& e, const LifetimeWindowInput& w,
              const LifetimeAccess& a, const LifetimeAccess& b)
{
    if (a.payload >= b.payload || a.cell != b.cell ||
        (!w.operations.empty() && w.operations[a.payload] && w.operations[a.payload] == w.operations[b.payload])) {
        return false;
    }
    const auto p = w.payloads[a.payload].pipe, q = w.payloads[b.payload].pipe;
    if (w.storageProtection.protectsScalar(p, q)) { return false; }
    const bool ar = e.constantValue(a.read) == 1, aw = e.constantValue(a.write) == 1;
    const bool br = e.constantValue(b.read) == 1, bw = e.constantValue(b.write) == 1;
    return ((aw && (br || bw)) || (ar && bw)) &&
           !(aw && bw && hardwareProtectsConflict(p, a.protectionGroup, q, b.protectionGroup));
}
Witness witness(RegionExpressions& e, const LifetimeWindowInput& w, llvm::ArrayRef<uint8_t> always)
{
    Witness result;
    result.outgoing.resize(w.sites); result.incoming.resize(w.sites);
    std::vector<uint32_t> mapping(w.sites, UINT32_MAX);
    std::vector<PeriodicPayload> payloads;
    for (uint32_t i = 0; i < w.sites; ++i) {
        if (always[i]) { mapping[i] = payloads.size(); payloads.push_back({w.payloads[i].pipe}); }
    }
    // Native attachments use no fictitious execution of optional sites.
    for (uint32_t i = 0; i < w.sites; ++i) {
        for (uint32_t j = 0; j < w.sites; ++j) {
            if (always[j] && w.payloads[i].pipe == w.payloads[j].pipe) {
                result.outgoing[i].push_back({{mapping[j], PeriodicEventKind::Completion}, i <= j ? 0U : 1U});
                result.incoming[i].push_back({{mapping[j], PeriodicEventKind::Start}, j <= i ? 0U : 1U});
            }
        }
    }
    std::vector<PeriodicRecord> generators, native;
    for (const auto& a : w.accesses) {
        if (a.payload >= w.sites) { continue; }
        for (const auto& b : w.accesses) {
            if (!conflict(e, w, a, b)) { continue; }
            const auto from = a.payload, to = b.payload % w.sites;
            const auto d = b.payload / w.sites;
            if (always[to]) { result.outgoing[from].push_back({{mapping[to], PeriodicEventKind::Start}, d}); }
            if (always[from]) { result.incoming[to].push_back({{mapping[from], PeriodicEventKind::Completion}, d}); }
            if (always[from] && always[to]) { generators.push_back({mapping[from], mapping[to], d}); }
        }
    }
    auto append = [&](llvm::ArrayRef<GuardedRankEdge> edges, std::vector<PeriodicRecord>& output) {
        for (auto edge : edges) {
            if (edge.source < w.sites && edge.target < w.payloads.size() &&
                e.constantValue(edge.guard) == 1 && always[edge.source] && always[edge.target % w.sites]) {
                output.push_back({mapping[edge.source], mapping[edge.target % w.sites], edge.target / w.sites});
            }
        }
    };
    append(w.prerequisites, generators); append(w.nativePrerequisites, native);
    result.mandatory = analyzePeriodicDemands(payloads, generators, native);
    return result;
}
using Palette = detail::BoundedLifetimePalette;
// Optional occurrences cannot justify reuse through an absent phase. Check
// every pair that maps to the same numeric label, not just cycle successors.
bool certifyShared(const PeriodicReuseMatrix& costs, const PeriodicSharedAllocation& assignment)
{
    if (assignment.status != PeriodicSharedAllocationStatus::Success || assignment.budget > INT64_MAX ||
        assignment.phases.size() != costs.size()) { return false; }
    for (std::size_t i = 0; i < costs.size(); ++i) {
        const auto& a = assignment.phases[i];
        for (std::size_t j = 0; j < costs.size(); ++j) {
            const auto& b = assignment.phases[j];
            if (a.cycle != b.cycle) { continue; }
            if (!a.laneCount || a.laneCount != b.laneCount || !costs[i][j]) { return false; }
            // Offsets and widths are at most INT64_MAX, so this sum fits uint64.
            auto gap = (a.offset + a.laneCount - b.offset) % a.laneCount;
            if (!gap) {
                if (i != j && *costs[i][j] && (!costs[j][i] || *costs[j][i])) { return false; }
                gap = a.laneCount;
            }
            if (*costs[i][j] > gap) { return false; }
        }
    }
    return true;
}
std::optional<LifetimeWindowInput> normalizeAllocationModes(RegionExpressions& e, const LifetimeWindowInput& w)
{
    // Use the same occurrence/cell modes as demand generation. In particular,
    // separate read/write declarations of an RMW must not introduce a read
    // edge that bypasses the shared protected-writer rule.
    std::map<std::pair<uint32_t, uint32_t>, LifetimeAccess> modes;
    for (auto access : w.accesses) {
        if (access.payload >= w.payloads.size() || !e.isBoolean(access.read) || !e.isBoolean(access.write) ||
            !e.constantValue(access.read) || !e.constantValue(access.write)) { return {}; }
        auto [entry, added] = modes.emplace(std::make_pair(access.payload, access.cell), access);
        if (!added) {
            auto& prior = entry->second;
            prior.read = e.lor(prior.read, access.read); prior.write = e.lor(prior.write, access.write);
            if (prior.protectionGroup != access.protectionGroup) { prior.protectionGroup = 0; }
        }
    }
    auto normalized = w;
    normalized.accesses.clear();
    for (const auto& [key, access] : modes) { normalized.accesses.push_back(access); }
    return normalized;
}
bool buildLivePalettes(const LifetimeWindowInput& w, llvm::ArrayRef<GuardedRankEdge> demands,
    const Witness& proof, detail::BoundedLifetimeAllocationAssignment& result)
{
    auto& palettes = result.palettes;
    std::map<std::pair<uint32_t, uint32_t>, std::size_t> bySource;
    for (std::size_t r = 0; r < demands.size(); ++r) {
        const auto& edge = demands[r];
        if (edge.source >= w.sites || edge.target >= w.payloads.size() || r > INT64_MAX) { return false; }
        const auto p = w.payloads[edge.source].pipe, q = w.payloads[edge.target].pipe;
        if (p == q || result.inactiveRecords[r]) { continue; }
        const auto h = proof.threshold(edge.target % w.sites, edge.source);
        uint64_t gap;
        if (!h || !periodic::add(edge.target / w.sites, *h, gap) || gap > INT64_MAX) { return false; }
        auto [entry, added] = bySource.emplace(std::make_pair(edge.source, q), palettes.size());
        if (added) { palettes.push_back({p, q, edge.source, 1, {}}); }
        auto& palette = palettes[entry->second];
        palette.gap = std::max(palette.gap, gap);
        palette.records.push_back(r);
    }
    return true;
}
std::optional<uint64_t> paletteReturnCost(const LifetimeWindowInput& w, llvm::ArrayRef<GuardedRankEdge> demands,
    const Witness& proof, const detail::BoundedLifetimeAllocationAssignment& assignment, std::size_t i, std::size_t j)
{
    const auto& palettes = assignment.palettes;
    // An empty event family cannot overlap another notification. Zero here
    // certifies disjointness, not an invented event-order path. Its own mapping
    // still requires one valid eligible label.
    if (palettes[i].inactive || palettes[j].inactive) { return i == j ? 1 : 0; }
    uint64_t maximum = 0;
    for (auto record : palettes[i].records) {
        if (assignment.inactiveRecords[record]) { continue; }
        const auto& edge = demands[record];
        const auto consumer = edge.target % w.sites, producer = palettes[j].site;
        // WAIT blocks consumer start; SET follows producer issue. Native start
        // order suffices on one pipe, including equality.
        auto h = w.payloads[consumer].pipe == w.payloads[producer].pipe ?
            std::optional<uint64_t>(consumer <= producer ? 0 : 1) : proof.threshold(consumer, producer);
        uint64_t gap;
        if (!h || !periodic::add(edge.target / w.sites, *h, gap) || gap > INT64_MAX) { return {}; }
        maximum = std::max(maximum, gap);
    }
    return maximum;
}
PeriodicReuseMatrix paletteReuseCosts(const LifetimeWindowInput& w, llvm::ArrayRef<GuardedRankEdge> demands,
    const Witness& proof, const detail::BoundedLifetimeAllocationAssignment& assignment)
{
    const auto n = assignment.palettes.size();
    PeriodicReuseMatrix costs(n, std::vector<std::optional<uint64_t>>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) { costs[i][j] = paletteReturnCost(w, demands, proof, assignment, i, j); }
    }
    return costs;
}
// Certified-empty records retain a same-direction physical formula without
// reserving another live palette. An entirely empty direction needs one valid
// dummy label; ordinary eligible-ID and macro-reservation checks still apply.
void appendInactivePalettes(const LifetimeWindowInput& w, llvm::ArrayRef<GuardedRankEdge> demands,
    detail::BoundedLifetimeAllocationAssignment& result)
{
    for (std::size_t r = 0; r < demands.size(); ++r) {
        if (!result.inactiveRecords[r]) { continue; }
        const auto& edge = demands[r];
        const auto p = w.payloads[edge.source].pipe, q = w.payloads[edge.target].pipe;
        if (p == q) { continue; }
        auto found = std::find_if(result.palettes.begin(), result.palettes.end(), [&](const auto& palette) {
            return palette.source == p && palette.target == q;
        });
        if (found == result.palettes.end()) {
            result.palettes.push_back({p, q, edge.source, 1, {static_cast<int64_t>(r)}, true});
        } else { found->records.push_back(static_cast<int64_t>(r)); }
    }
}
DictionaryAttr encode(func::FuncOp function, llvm::ArrayRef<Palette> palettes, int64_t plan,
                      const PeriodicSharedAllocation* shared = nullptr)
{
    Builder b(function.getContext());
    using Direction = std::pair<uint32_t, uint32_t>;
    std::map<Direction, std::set<int64_t>> reserved;
    function.walk([&](Operation* operation) {
        if (auto model = getSyncMacroModel(operation)) {
            for (const auto& event : model->hiddenEvents) {
                const auto source = static_cast<uint32_t>(event.srcPipe);
                const auto target = static_cast<uint32_t>(event.dstPipe);
                reserved[{source, target}].insert(event.eventIds.begin(), event.eventIds.end());
            }
        }
    });
    std::vector<std::vector<std::size_t>> domains;
    std::map<Direction, std::size_t> positions;
    for (std::size_t i = 0; i < palettes.size(); ++i) {
        if (shared) {
            auto entry = positions.emplace(Direction{palettes[i].source, palettes[i].target}, domains.size());
            if (entry.second) { domains.emplace_back(); }
            domains[entry.first->second].push_back(i);
        } else { domains.push_back({i}); }
    }
    SmallVector<Attribute> groups;
    for (std::size_t i = 0; i < domains.size(); ++i) {
        const auto& p = palettes[domains[i].front()];
        SmallVector<int64_t> conflicts, records, sources, targets;
        SmallVector<Attribute> rules;
        uint64_t paletteBudget = 0;
        for (std::size_t j = 0; j < i; ++j) {
            const auto& prior = palettes[domains[j].front()];
            if (p.source == prior.source && p.target == prior.target) { conflicts.push_back(j); }
        }
        for (auto index : domains[i]) {
            const auto& member = palettes[index];
            const uint64_t begin = shared ? shared->phases[index].laneBegin : 0;
            const uint64_t width = shared ? shared->phases[index].laneCount : member.gap;
            const uint64_t offset = shared ? shared->phases[index].offset : 0;
            paletteBudget = std::max(paletteBudget, begin + width);
            auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(1)),
                b.getNamedAttr("base", b.getI64IntegerAttr(begin)),
                b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr(
                    {0, 1, int64_t(offset), int64_t(width), 1})}))});
            for (auto record : member.records) {
                records.push_back(record); sources.push_back(member.source);
                targets.push_back(member.target); rules.push_back(rule);
            }
        }
        const auto& hidden = reserved[{p.source, p.target}];
        SmallVector<int64_t> forbidden(hidden.begin(), hidden.end());
        SmallVector<int64_t> zeros(records.size(), 0);
        groups.push_back(b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(p.source)),
            b.getNamedAttr("target", b.getI64IntegerAttr(p.target)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(paletteBudget)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr(records)),
            b.getNamedAttr("sources", b.getDenseI64ArrayAttr(sources)),
            b.getNamedAttr("targets", b.getDenseI64ArrayAttr(targets)),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("tuple_rules", b.getArrayAttr(rules)),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr(conflicts)),
            b.getNamedAttr("forbidden_ids", b.getDenseI64ArrayAttr(forbidden))}));
    }
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("kind", b.getStringAttr("finite")), b.getNamedAttr("plan", b.getI64IntegerAttr(plan)),
        b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
        b.getNamedAttr("macro_reservations", b.getUnitAttr()), b.getNamedAttr("groups", b.getArrayAttr(groups))});
}
} // namespace
std::optional<detail::BoundedLifetimeAllocationAssignment> detail::boundedLifetimeAllocationAssignmentWithInactive(
    RegionExpressions& e, const LifetimeWindowInput& w,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> demands,
    llvm::ArrayRef<uint8_t> invocationInactive)
{
    if (!invocationInactive.empty() && invocationInactive.size() != demands.size()) { return {}; }
    if (!w.sites || w.span >= UINT32_MAX || w.sites > UINT32_MAX / (w.span + 1) ||
        w.payloads.size() != w.sites * (w.span + 1) || unconditional.size() != w.sites ||
        (!w.operations.empty() && w.operations.size() != w.payloads.size())) { return {}; }
    auto normalized = normalizeAllocationModes(e, w);
    if (!normalized) { return {}; }
    for (const auto& payload : w.payloads) {
        if (!e.isBoolean(payload.present)) { return {}; }
    }
    for (const auto& edge : demands) {
        if (edge.source >= w.sites || edge.target >= w.payloads.size() || !e.isBoolean(edge.guard)) { return {}; }
    }
    auto proof = witness(e, *normalized, unconditional);
    if (!proof.mandatory.error.empty()) { return {}; }
    BoundedLifetimeAllocationAssignment result;
    BoundedInactiveRecordProof inactiveProof(e, w, unconditional);
    for (auto [record, edge] : llvm::enumerate(demands)) {
        result.inactiveRecords.push_back(inactiveProof.proves(edge.guard) ||
            (!invocationInactive.empty() && invocationInactive[record]));
    }
    auto& palettes = result.palettes;
    if (!buildLivePalettes(w, demands, proof, result)) { return {}; }
    appendInactivePalettes(w, demands, result);
    auto costs = paletteReuseCosts(w, demands, proof, result);
    std::vector<std::pair<uint32_t, uint32_t>> directions;
    for (const auto& palette : palettes) { directions.emplace_back(palette.source, palette.target); }
    auto shared = allocateDirectedPeriodic(costs, directions);
    if (certifyShared(costs, shared)) { result.shared = std::move(shared); }
    return result;
}
std::optional<detail::BoundedLifetimeAllocationAssignment> detail::boundedLifetimeAllocationAssignment(
    RegionExpressions& e, const LifetimeWindowInput& w, llvm::ArrayRef<uint8_t> unconditional,
    llvm::ArrayRef<GuardedRankEdge> demands)
{
    return boundedLifetimeAllocationAssignmentWithInactive(e, w, unconditional, demands, {});
}
DictionaryAttr boundedLifetimeAllocationCertificate(
    func::FuncOp function, RegionExpressions& e, const LifetimeWindowInput& w,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> demands, int64_t plan,
    llvm::ArrayRef<uint32_t> originalRecords)
{
    if (!originalRecords.empty() && (originalRecords.size() != demands.size() ||
        std::set<uint32_t>(originalRecords.begin(), originalRecords.end()).size() != demands.size())) { return {}; }
    auto assignment = detail::boundedLifetimeAllocationAssignment(e, w, unconditional, demands);
    if (!assignment) { return {}; }
    if (!originalRecords.empty()) {
        for (auto& palette : assignment->palettes) {
            for (auto& record : palette.records) { record = originalRecords[record]; }
        }
    }
    return encode(function, assignment->palettes, plan, assignment->shared ? &*assignment->shared : nullptr);
}
} // namespace mlir::pto::frontiersynch
