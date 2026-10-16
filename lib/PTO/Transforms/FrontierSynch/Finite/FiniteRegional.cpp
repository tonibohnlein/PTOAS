// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical finite export wraps requested answers, never reconstructs a guarded
// analysis. Callbacks own the child and its original source context, not a parent.
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
namespace mlir::pto::frontiersynch {
namespace {
RegionalAnalysis concreteRegional(std::shared_ptr<FiniteAnalysis> owner, std::shared_ptr<RegionExpressions> arena)
{
    RegionalAnalysis out;
    const auto& result = *owner->concrete();
    const auto& input = owner->originalInput()->storage();
    out.expressions = arena;
    out.accessModel = &input.accesses();
    out.gmAliasPolicy = input.memory().gmPolicy();
    out.capabilities = {true, true, true, true};
    const auto zero = arena->constant(0), yes = arena->boolean(true);
    auto selected = [&](uint32_t type) { return RegionalSelector{{type, zero, PeriodicEventKind::Start}, yes}; };
    for (uint32_t type = 0; type < result.phases.size(); ++type) {
        auto* phase = result.phases[type];
        auto* op = phase->elementOp;
        arena->forbidRecomputation(op);
        out.anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        const auto selector = selected(type);
        out.firstSitePayloads[type].push_back(selector);
        const auto pipe = static_cast<uint32_t>(phase->kPipeValue);
        if (out.firstPayloads[pipe].empty()) { out.firstPayloads[pipe].push_back(selector); }
        out.lastPayloads[pipe] = {selector};
        for (auto effect : input.accesses().effectsFor(phase)) {
            if (llvm::is_contained(result.dischargedEffects, effect)) {
                out.deferredAccessBoundary.push_back({effect, selector, selector, false});
            } else {
                const auto& access = input.accesses().effects()[effect];
                out.accessBoundary.push_back({effect, selector, selector, access.rangesMaterialized});
                accumulateCost(out.cost.physicalFragments, access.ranges.size());
            }
        }
    }
    out.occurrenceLoops.resize(out.anchors.size());
    for (const auto& boundary : result.storageBoundary) {
        if (boundary.atom >= input.accesses().cells().size()) { return {}; }
        RegionalStorageBoundary cell;
        cell.cell = input.accesses().cells()[boundary.atom];
        if (boundary.firstWriter) { cell.firstWriters.push_back(selected(*boundary.firstWriter)); }
        if (boundary.lastWriter) { cell.lastWriters.push_back(selected(*boundary.lastWriter)); }
        for (const auto& [pipe, type] : boundary.firstReaders) { cell.firstReaders[pipe].push_back(selected(type)); }
        for (const auto& [pipe, type] : boundary.lastReaders) { cell.lastReaders[pipe].push_back(selected(type)); }
        accumulateCost(owner->work.boundaryEntries, cell.firstWriters.size() + cell.lastWriters.size());
        for (const auto& entry : cell.firstReaders) {
            accumulateCost(owner->work.boundaryEntries, entry.second.size());
        }
        for (const auto& entry : cell.lastReaders) {
            accumulateCost(owner->work.boundaryEntries, entry.second.size());
        }
        out.storageBoundary.push_back(std::move(cell));
    }
    out.cost.children = 1;
    out.cost.cells = out.storageBoundary.size();
    out.cost.ports = result.phases.size();
    auto present = [owner, arena, zero](RegionalEvent event) -> std::optional<RegionExpressions::Id> {
        if (event.type >= owner->concrete()->phases.size() || !event.visits.empty() ||
            event.ordinal >= arena->size() || arena->isBoolean(event.ordinal) ||
            (event.kind != PeriodicEventKind::Start && event.kind != PeriodicEventKind::Completion)) {
            return std::nullopt;
        }
        return arena->eq(event.ordinal, zero);
    };
    out.presence = present;
    out.reachability = [owner, arena, present](RegionalEvent a, RegionalEvent b)
        -> std::optional<RegionExpressions::Id> {
        auto pa = present(a), pb = present(b);
        if (!pa || !pb) { return std::nullopt; }
        accumulateCost(owner->work.numericalQueries, 1);
        auto answer = explicitEventPrecedes(*owner->concrete(), {a.type, a.kind}, {b.type, b.kind});
        if (!answer) { return std::nullopt; }
        return arena->land(arena->boolean(*answer), arena->land(*pa, *pb));
    };
    out.prepare = [owner]() { return owner->prepare(); };
    return out;
}
} // namespace
RegionalAnalysis finiteRegionalResult(std::shared_ptr<FiniteAnalysis> analysis,
    std::shared_ptr<RegionExpressions> expressions)
{
    if (!analysis || !analysis->error().empty() || !analysis->originalInput()) { return {}; }
    if (analysis->concrete()) {
        if (!expressions) { expressions = std::make_shared<RegionExpressions>(); }
        return concreteRegional(std::move(analysis), std::move(expressions));
    }
    auto out = finiteGuardedRegionalResult(*analysis->guarded());
    if (!out.expressions || (expressions && out.expressions != expressions)) { return {}; }
    // Preserve the guarded kernel's exact query semantics while retaining source
    // ownership even when a caller extracts just one callback from this export.
    auto present = out.presence;
    out.presence = [analysis, present](RegionalEvent event) { return present(event); };
    auto query = out.reachability;
    out.reachability = [analysis, query](RegionalEvent a, RegionalEvent b) { return query(a, b); };
    out.prepare = [analysis]() { return analysis->prepare(); };
    auto filtered = out.prepareFiltered;
    out.prepareFiltered = [analysis, filtered](const RegionalDemandFilter& filter) { return filtered(filter); };
    return out;
}
} // namespace mlir::pto::frontiersynch
