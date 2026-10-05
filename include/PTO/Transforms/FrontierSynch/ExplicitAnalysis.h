// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_EXPLICITANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_EXPLICITANALYSIS_H
#include "PTO/Transforms/FrontierSynch/ExplicitReduction.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include <optional>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
// Boundary identities refer to occurrences, not SSA allocation roots. Read-only
// accesses before the first/after the last writer are retained per pipe.
struct ExplicitCellBoundary {
    uint32_t atom = 0;
    std::optional<uint32_t> firstWriter;
    std::optional<uint32_t> lastWriter;
    std::unordered_map<uint32_t, uint32_t> firstReaders;
    std::unordered_map<uint32_t, uint32_t> lastReaders;
};
struct ExplicitAnalysis {
    std::string error;
    SmallVector<const CompoundInstanceElement*> phases;
    std::vector<ExplicitEffects> occurrences;
    StorageScanResult scan;
    ExplicitReduction reduction;
    std::vector<ExplicitCellBoundary> storageBoundary;
    SmallVector<std::size_t> dischargedEffects; // Globally independent root-block GM effects.
};
// Borrowed phases and cell IDs remain valid only while input/IR are unchanged.
// Input ranges must enumerate exact physical bytes; symbolic geometry is left
// for compact backends. Shared partitioning cost is outside this scan/reduction.
ExplicitAnalysis analyzeExplicit(Block& block, const PhaseIndex& index, const SyncInput& input);
// The span must contain every payload between its first and last anchors in
// one block, in reference order. Metadata between anchors is checked too.
// Optional GM discharge is confined to root-block occurrences with no possible
// conflict against any other phase in the complete input. It cannot certify
// an effect repeated by an enclosing loop.
ExplicitAnalysis analyzeExplicit(ArrayRef<const CompoundInstanceElement*> phases,
                                 const PhaseIndex& index, const SyncInput& input,
                                 bool dischargeIndependentGM = false);
// Here event.type is an occurrence index, not a periodic type. No periods are
// involved. Null means invalid input; the result is otherwise reflexive.
std::optional<bool> explicitEventPrecedes(const ExplicitAnalysis& analysis,
                                        PeriodicEvent source, PeriodicEvent target);
// Whole-function route only. Preparation is detached and failure changes no IR.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareExplicitInsertion(
    func::FuncOp function, const ExplicitAnalysis& analysis);
} // namespace mlir::pto::frontiersynch
#endif
