// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FINITE_REGIONAL_CHECKS_H
#define PTO_FINITE_REGIONAL_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include <set>
namespace mlir::pto::test {
using FiniteBoundaryKey = std::tuple<unsigned, uint32_t, const CompoundInstanceElement*>;
inline std::optional<std::set<FiniteBoundaryKey>> finiteBoundaryAt(
    const frontiersynch::RegionalAnalysis& region, const SyncStorageCell& point)
{
    std::set<FiniteBoundaryKey> result;
    auto add = [&](const auto& values, unsigned kind, uint32_t pipe) {
        for (const auto& selected : values) {
            const auto active = region.expressions->constantValue(selected.present);
            if (!active || selected.event.type >= region.anchors.size()) { return false; }
            if (*active) {
                if (region.expressions->constantValue(selected.event.ordinal) != 0 ||
                    !selected.event.visits.empty()) { return false; }
                result.emplace(kind, pipe, region.anchors[selected.event.type].phase);
            }
        }
        return true;
    };
    for (const auto& boundary : region.storageBoundary) {
        if (!sameStorageDomain(point, boundary.cell) || point.begin < boundary.cell.begin ||
            point.begin >= boundary.cell.end) { continue; }
        if (!add(boundary.firstWriters, 0, 0) || !add(boundary.lastWriters, 1, 0)) { return std::nullopt; }
        for (const auto& [pipe, values] : boundary.firstReaders) {
            if (!add(values, 2, pipe)) { return std::nullopt; }
        }
        for (const auto& [pipe, values] : boundary.lastReaders) {
            if (!add(values, 3, pipe)) { return std::nullopt; }
        }
    }
    return result;
}
inline bool sameFiniteAnswers(const frontiersynch::RegionalAnalysis& flat,
                              const frontiersynch::RegionalAnalysis& composed)
{
    namespace fs = frontiersynch;
    if (flat.expressions != composed.expressions || flat.anchors.size() != composed.anchors.size() ||
        !composed.reachability || !composed.presence) { return false; }
    auto& arena = *flat.expressions;
    SmallVector<uint32_t> mapped;
    for (const auto& anchor : flat.anchors) {
        auto found = llvm::find_if(composed.anchors, [&](const auto& other) { return anchor.phase == other.phase; });
        if (found == composed.anchors.end()) { return false; }
        mapped.push_back(static_cast<uint32_t>(found - composed.anchors.begin()));
    }
    const auto zero = arena.constant(0);
    for (uint32_t a = 0; a < flat.anchors.size(); ++a) {
        for (uint32_t b = 0; b < flat.anchors.size(); ++b) {
            for (auto ka : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                for (auto kb : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                    auto left = flat.reachability({a, zero, ka}, {b, zero, kb});
                    auto right = composed.reachability({mapped[a], zero, ka}, {mapped[b], zero, kb});
                    if (!left || !right || !arena.constantValue(*left) ||
                        arena.constantValue(*left) != arena.constantValue(*right)) { return false; }
                }
            }
        }
    }
    for (const auto* region : {&flat, &composed}) {
        for (const auto& boundary : region->storageBoundary) {
            const auto left = finiteBoundaryAt(flat, boundary.cell), right = finiteBoundaryAt(composed, boundary.cell);
            if (!left || !right || left != right) { return false; }
        }
    }
    return true;
}
inline bool checkFiniteRegionalPartitions(const std::shared_ptr<frontiersynch::FiniteAnalysis>& owner)
{
    namespace fs = frontiersynch;
    if (!owner || !owner->concrete() || !owner->originalInput()) { return false; }
    const auto input = owner->originalInput();
    const auto roots = input->roots();
    const auto function = input->function();
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto flat = fs::finiteRegionalResult(owner, arena);
    if (!flat.presence || !flat.reachability || owner->guarded()) { return false; }
    if (roots.size() > 2) {
        SmallVector<Operation*> invalid{roots.front(), roots.back()};
        if (fs::finiteRegionInput(input, invalid)) { return false; }
    }
    using Demand = std::tuple<Operation*, Operation*, uint32_t, uint32_t>;
    std::set<Demand> demands;
    for (const auto& edge : owner->concrete()->reduction.retained) {
        const auto& concrete = *owner->concrete();
        demands.emplace(concrete.phases[edge.source]->elementOp->getNextNode(),
            concrete.phases[edge.target]->elementOp, concrete.occurrences[edge.source].pipe,
            concrete.occurrences[edge.target].pipe);
    }
    for (std::size_t cut = 0; cut <= roots.size(); ++cut) {
        std::vector<fs::RegionalAnalysis> children;
        for (auto slice : {roots.take_front(cut), roots.drop_front(cut)}) {
            auto region = fs::analyzeFinite(fs::finiteRegionInput(input, slice), fs::FiniteStrategy::Concrete);
            if (!region || !region->error().empty() || region->guarded()) { return false; }
            auto exported = fs::finiteRegionalResult(region, arena);
            region.reset();
            // Export callbacks own their numerical result after its caller releases it.
            auto plan = exported.prepare();
            if (failed(plan) || (*plan)->completeInvocation) { return false; }
            children.push_back(std::move(exported));
        }
        auto composed = fs::composeRegionalSequence(function, arena, std::move(children), true, true,
                                                    input->index().sourceFacts());
        if (!composed.error.empty() || !sameFiniteAnswers(flat, fs::sequenceRegionalResult(composed))) { return false; }
        auto prepared = fs::prepareSequenceLogicalInsertion(composed);
        if (failed(prepared)) { return false; }
        std::set<Demand> realized;
        for (const auto& family : (*prepared)->families) {
            if (!realized.emplace(family.sourceCut.before, family.targetCut.before,
                                  family.sourcePipe, family.targetPipe).second) { return false; }
        }
        if (realized != demands) { return false; }
        auto exported = fs::sequenceRegionalResult(composed);
        composed = {};
        if (!sameFiniteAnswers(flat, exported)) { return false; }
    }
    return true;
}
} // namespace mlir::pto::test
#endif
