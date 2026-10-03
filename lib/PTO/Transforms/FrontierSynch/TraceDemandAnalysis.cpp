// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared conflict queries over source phases; no alternate demand reducer.
#include "PTO/Transforms/FrontierSynch/TraceDemandAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::pto::frontiersynch {
namespace {
bool conflict(const SyncInput& input, const CompoundInstanceElement& a, const CompoundInstanceElement& b)
{
    DepBaseMemInfoPairVec witnesses;
    return input.memory().DepBetween(a.defVec, b.useVec, witnesses) ||
           input.memory().DepBetween(a.useVec, b.defVec, witnesses) ||
           input.memory().DepBetween(a.defVec, b.defVec, witnesses);
}
} // namespace
FailureOr<TraceDemandAnalysis> TraceDemandAnalysis::build(
    func::FuncOp function, const SyncInput& input)
{
    if (!function || function.isExternal()) { return failure(); }
    PhaseIndex index;
    if (failed(index.build(function, input))) { return failure(); }
    TraceDemandAnalysis result;
    for (const auto* phase : input.instructions()) {
        StructuredSite site;
        site.phase = phase;
        site.anchor = phase->elementOp;
        site.regions = index.controlPath(*phase);
        auto phases = index.phasesFor(site.anchor);
        site.localPhase = std::distance(phases.begin(), llvm::find(phases, phase));
        for (Region* enclosing : site.regions) {
            Operation* owner = enclosing->getParentOp();
            if (isa<LoopLikeOpInterface>(owner) && !llvm::is_contained(site.loops, owner)) {
                site.loops.push_back(owner);
            }
        }
        result.records.push_back(std::move(site));
    }
    bool hasNestedRegions = false;
    function.walk([&](Operation* operation) {
        hasNestedRegions |= operation != function && operation->getNumRegions() != 0;
    });
    if (function.getBody().hasOneBlock() && !hasNestedRegions) {
        result.explicitSequence.emplace();
        for (std::size_t i = 0; i < result.records.size(); ++i) {
            result.explicitSequence->push_back(i);
        }
    }
    for (const auto& site : result.records) {
        if (!llvm::is_contained(result.columns, site.phase->kPipeValue)) {
            result.columns.push_back(site.phase->kPipeValue);
        }
    }
    result.sharedInput = &input;
    return result;
}
ArrayRef<SmallVector<std::size_t>> TraceDemandAnalysis::conflicts() const
{
    if (!conflictsReady) {
        incoming.resize(records.size());
        if (sharedInput) {
            for (auto [b, target] : llvm::enumerate(records)) {
                for (auto [a, source] : llvm::enumerate(records)) {
                    if (conflict(*sharedInput, *source.phase, *target.phase)) {
                        incoming[b].push_back(a);
                    }
                }
            }
        }
        if (sharedInput && explicitSequence) {
            for (const auto& baseline : sharedInput->target().finiteDrains()) {
                auto source = llvm::find_if(records, [&](const auto& site) { return site.phase == baseline.previous; });
                auto target = llvm::find_if(records, [&](const auto& site) { return site.phase == baseline.consumer; });
                if (source != records.end() && target != records.end()) {
                    auto a = static_cast<std::size_t>(source - records.begin());
                    auto& row = incoming[target - records.begin()];
                    if (!llvm::is_contained(row, a)) { row.push_back(a); }
                }
            }
        }
        conflictsReady = true;
    }
    return incoming;
}
} // namespace mlir::pto::frontiersynch
