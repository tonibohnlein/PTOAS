// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEOVERLAY_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEOVERLAY_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
struct FiniteOverlayDemand {
    RegionalEvent source, target; // Completion to start, in the unchanged base occurrence frame.
    RegionExpressions::Id guard = RegionExpressions::invalid;
};
using RegionalOrderQuery = std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)>;
struct FiniteOverlayCost {
    uint64_t ports = 0, baseQueries = 0, identityTests = 0, closureUpdates = 0;
};
struct FiniteOverlayState;
struct FiniteOverlayAnalysis {
    std::string error;
    RegionalAnalysis base;
    RegionalAnalysis regional; // Augmented exact queries; endpointRecipes remains false until prepared.
    std::vector<FiniteOverlayDemand> demands;
    std::vector<RegionExpressions::Id> retained; // First actual-endpoint representative among added records.
    // Refines an existing base cover; its original presence/retention guard must
    // also hold. Both endpoints must invoke this with the identical actual pair.
    RegionalOrderQuery retainBase;
    std::shared_ptr<FiniteOverlayCost> cost;
    std::shared_ptr<FiniteOverlayState> state;
};
// Each added record denotes AT MOST ONE actual edge per invocation context.
// An unbounded iteration-indexed edge family is not a finite overlay. The base
// has exact reflexive queries and minimum nonnative cover recipes on this same
// occurrence frame. Distinct type IDs identify distinct payload occurrences;
// equal type/ordinal/kind/visits tuples identify the same event. Absent endpoints
// disable records. referenceBefore must return exact strict occurrence order;
// acceptance proves active => forward using the arena's sufficient implication.
// The wrapper neither discovers exceptions nor establishes that the selected
// base and overlay cover all physical effects. Storage/presence exports stay
// unchanged. No IR mutation or endpoint preparation occurs in this function.
FiniteOverlayAnalysis analyzeFiniteOverlay(RegionalAnalysis base,
    std::vector<FiniteOverlayDemand> demands, RegionalOrderQuery referenceBefore);
} // namespace mlir::pto::frontiersynch
#endif
