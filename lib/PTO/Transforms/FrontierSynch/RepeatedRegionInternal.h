// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#ifndef PTO_FRONTIERSYNCH_REPEATEDREGIONINTERNAL_H
#define PTO_FRONTIERSYNCH_REPEATEDREGIONINTERNAL_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
void liftRepeatedSelectors(RegionalAnalysis& out, RegionExpressions::Id trips);
struct RepeatedRegionState {
    using Id = RegionExpressions::Id;
    func::FuncOp function;
    scf::ForOp loop;
    RegionalAnalysis body;
    Id trips;
    // q>1 uses the original loop's quotient coordinate; originalTrips bounds
    // the partial final period. Empty typePhases denotes invariant q=1.
    uint64_t phaseCount = 1;
    Id originalTrips = RegionExpressions::invalid;
    std::vector<uint32_t> typePhases;
    std::vector<RepeatedCrossing> crossings;
    // Raw bridge generators define Q; reduced guards are only endpoint recipes.
    std::vector<RepeatedCrossing> queryCrossings;
    std::vector<RegionalEvent> slots;
    std::vector<Id> distances;
    uint64_t infinity = 0;
    using EventKey = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
    std::map<std::pair<EventKey, EventKey>, std::optional<Id>> queryMemo;
    std::string error;
    RegionExpressions& e() { return *body.expressions; }
    bool buildBoundary();
    bool closePorts();
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepare(ArrayRef<scf::ForOp> enclosing);
    std::optional<Id> query(RegionalEvent source, RegionalEvent target);
    std::optional<Id> present(RegionalEvent event);
    std::optional<Id> computeQuery(RegionalEvent source, RegionalEvent target);
};
} // namespace mlir::pto::frontiersynch
#endif
