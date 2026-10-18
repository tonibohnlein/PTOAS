// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Read-only guarantee views of retained results; no capability is constructed.
#ifndef PTO_FRONTIERSYNCH_RESULTGUARANTEES_H
#define PTO_FRONTIERSYNCH_RESULTGUARANTEES_H
#include "PTO/Transforms/FrontierSynch/Shared/BoundingRegionalAnalysis.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
namespace mlir::pto::frontiersynch {
struct ExplicitAnalysis;
struct AnalysisOutcome;
struct PreparedLogicalPlan;
enum class RealizationOrder { Unestablished, CoversSelectedOrder, Equivalent, MayStrengthen, CompletionCoverage };
RealizationOrder explicitRealizationOrder(const ExplicitAnalysis& analysis);
struct ResultGuarantees {
    // The unchanged modeled effects and alias/admission contract, not a claim
    // that inferred access summaries equal all concrete original effects.
    std::shared_ptr<const SyncInput> input;
    std::optional<InputOrderGuarantee> modeledOrder;
    std::optional<ReductionQuality> reduction;
    RealizationOrder realization = RealizationOrder::Unestablished;
    bool selectedOrderBounds = false;
    bool completedExit = false;
    DictionaryAttr realizedLoss; // Missing is unavailable, never zero.
    bool completion = false;
    std::size_t completionClasses = 0;
    uint64_t eventFreeChildren = 0, completedChildren = 0, descendantClosedChildren = 0;
    uint64_t inlineChildren = 0, eventfulInlineChildren = 0, readinessCuts = 0;
};
// Mathematical evidence survives a failed executable/allocation export.
ResultGuarantees describeResultGuarantees(const AnalysisOutcome& outcome);
// Equality is reported only for a supported producer's realization premises.
// Capability readiness and a successful allocation never promote this field.
ResultGuarantees describeResultGuarantees(const PreparedLogicalPlan& plan);
DictionaryAttr encodeResultGuarantees(MLIRContext* context, const ResultGuarantees& guarantees);
void attachResultGuarantees(PreparedLogicalPlan& plan, MLIRContext* context);
} // namespace mlir::pto::frontiersynch
#endif
