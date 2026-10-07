// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A reusable lane needs a return path through guaranteed-present occurrences.
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PeriodicAnalysisInternal.h"
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
struct Palette {
    uint32_t source, target, site;
    uint64_t gap = 1;
    SmallVector<int64_t> records;
};
DictionaryAttr encode(func::FuncOp function, llvm::ArrayRef<Palette> palettes, int64_t plan)
{
    Builder b(function.getContext());
    std::set<int64_t> reserved;
    function.walk([&](Operation* operation) {
        if (auto model = getSyncMacroModel(operation)) {
            for (const auto& event : model->hiddenEvents) {
                reserved.insert(event.eventIds.begin(), event.eventIds.end());
            }
        }
    });
    SmallVector<Attribute> groups;
    for (std::size_t i = 0; i < palettes.size(); ++i) {
        const auto& p = palettes[i];
        SmallVector<int64_t> conflicts, strides(p.records.size(), 0), phases(p.records.size(), 0);
        for (std::size_t j = 0; j < i; ++j) {
            conflicts.push_back(j);
        }
        // Version-four source tuple has one coordinate: the source ordinal.
        auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(1)),
            b.getNamedAttr("base", b.getI64IntegerAttr(0)),
            b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr({0, 1, 0, int64_t(p.gap), 1})}))});
        SmallVector<Attribute> rules(p.records.size(), rule);
        const auto& hidden = reserved;
        SmallVector<int64_t> forbidden(hidden.begin(), hidden.end());
        groups.push_back(b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(p.source)),
            b.getNamedAttr("target", b.getI64IntegerAttr(p.target)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(p.gap)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr(p.records)),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr(strides)),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr(phases)),
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
DictionaryAttr boundedLifetimeAllocationCertificate(
    func::FuncOp function, RegionExpressions& e, const LifetimeWindowInput& w,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> demands, int64_t plan)
{
    if (!w.sites || w.span >= UINT32_MAX || w.sites > UINT32_MAX / (w.span + 1) ||
        w.payloads.size() != w.sites * (w.span + 1) || unconditional.size() != w.sites ||
        (!w.operations.empty() && w.operations.size() != w.payloads.size())) { return {}; }
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
    auto proof = witness(e, normalized, unconditional);
    if (!proof.mandatory.error.empty()) { return {}; }
    std::vector<Palette> palettes;
    std::map<std::pair<uint32_t, uint32_t>, std::size_t> bySource;
    for (std::size_t r = 0; r < demands.size(); ++r) {
        const auto& edge = demands[r];
        if (edge.source >= w.sites || edge.target >= w.payloads.size() || r > INT64_MAX) { return {}; }
        const auto p = w.payloads[edge.source].pipe, q = w.payloads[edge.target].pipe;
        if (p == q) { continue; }
        const auto h = proof.threshold(edge.target % w.sites, edge.source);
        uint64_t gap;
        if (!h || !periodic::add(edge.target / w.sites, *h, gap) || gap > INT64_MAX) { return {}; }
        auto [entry, added] = bySource.emplace(std::make_pair(edge.source, q), palettes.size());
        if (added) { palettes.push_back({p, q, edge.source, 1, {}}); }
        auto& palette = palettes[entry->second];
        palette.gap = std::max(palette.gap, gap);
        palette.records.push_back(r);
    }
    return encode(function, palettes, plan);
}
} // namespace mlir::pto::frontiersynch
