// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "StationaryCells.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
bool intersects(ArrayRef<std::size_t> first, ArrayRef<std::size_t> second)
{
    std::size_t a = 0, b = 0;
    while (a < first.size() && b < second.size()) {
        if (first[a] == second[b]) { return true; }
        if (first[a] < second[b]) { ++a; } else { ++b; }
    }
    return false;
}
struct SiteCells { SmallVector<std::size_t> reads, writes; };
void canonicalize(SmallVectorImpl<std::size_t>& cells)
{
    llvm::sort(cells);
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
}
} // namespace
FailureOr<std::shared_ptr<const StationaryCellInput>> StationaryCellInput::build(
    const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases, CostLedger& costs, std::string& reason)
{
    auto result = std::shared_ptr<StationaryCellInput>(new StationaryCellInput());
    SmallVector<SiteCells> sites(phases.size());
    SmallVector<RotatingFragment> fragments;
    {
        CostScope qualification(costs, CostStage::Effects);
        const auto& effects = input.storageBounds();
        if (phases.empty()) { reason = "no stationary payload sites"; return failure(); }
        auto observe = [&](ArrayRef<const BaseMemInfo*> memories, SmallVectorImpl<std::size_t>& cells) {
            for (const auto* memory : memories) {
                auto id = effects.memoryIdentity(memory);
                if (!id || effects.memory()[*id].bounds != SyncBoundStatus::ConstantPhysical) { return failure(); }
                llvm::append_range(cells, effects.memory()[*id].boundCells);
            }
            canonicalize(cells);
            return success();
        };
        for (auto [id, phase] : llvm::enumerate(phases)) {
            if (failed(observe(phase->useVec, sites[id].reads)) ||
                failed(observe(phase->defVec, sites[id].writes))) {
                reason = "stationary cells lack qualified constant local bounds"; return failure();
            }
        }
        // Fixed body order + invariant modeled locations means these static
        // predicates describe every inter-iteration occurrence pair. Compare
        // each hazard independently, including self/reverse site pairs. This
        // certificate is not a bounded execution sample or a pipe-filtered graph.
        for (auto [a, source] : llvm::enumerate(phases)) {
            for (auto [b, target] : llvm::enumerate(phases)) {
                DepBaseMemInfoPairVec raw, war, waw;
                bool expectedRAW = input.memory().DepBetween(source->defVec, target->useVec, raw);
                bool expectedWAR = input.memory().DepBetween(source->useVec, target->defVec, war);
                bool expectedWAW = input.memory().DepBetween(source->defVec, target->defVec, waw);
                ++result->checkedPairs;
                if (expectedRAW != intersects(sites[a].writes, sites[b].reads) ||
                    expectedWAR != intersects(sites[a].reads, sites[b].writes) ||
                    expectedWAW != intersects(sites[a].writes, sites[b].writes)) {
                    reason = "stationary cell hazards differ from authoritative shared alias model"; return failure();
                }
            }
        }
        for (const auto& site : sites) {
            llvm::append_range(result->sharedCells, site.reads);
            llvm::append_range(result->sharedCells, site.writes);
        }
        canonicalize(result->sharedCells);
        if (result->sharedCells.empty()) { reason = "no stationary storage cells"; return failure(); }
        DenseMap<std::size_t, std::size_t> atoms;
        for (auto [atom, cell] : llvm::enumerate(result->sharedCells)) { atoms.try_emplace(cell, atom); }
        for (auto [site, cells] : llvm::enumerate(sites)) {
            for (auto cell : cells.reads) { fragments.push_back({site, 0, atoms.lookup(cell), llvm::DynamicAPInt(0),
                                                               RotatingAccessMode::Read}); }
            for (auto cell : cells.writes) { fragments.push_back({site, 0, atoms.lookup(cell), llvm::DynamicAPInt(0),
                                                                RotatingAccessMode::Write}); }
        }
    }
    CostScope backend(costs, CostStage::Backend);
    RotatingFamily family{llvm::DynamicAPInt(1), llvm::DynamicAPInt(0), result->sharedCells.size()};
    if (failed(result->normalized.build(phases, {family}, fragments))) {
        reason = "stationary cell compact extraction obligation"; return failure();
    }
    return std::shared_ptr<const StationaryCellInput>(result);
}
} // namespace mlir::pto::frontiersynch
