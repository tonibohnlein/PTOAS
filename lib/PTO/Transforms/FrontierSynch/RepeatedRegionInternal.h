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
    Id originalBegin = RegionExpressions::invalid;
    std::vector<uint32_t> typePhases;
    std::vector<RepeatedCrossing> crossings;
    // Raw bridge generators define Q; reduced guards are only endpoint recipes.
    std::vector<RepeatedCrossing> queryCrossings;
    // Optional certified weighted quotient, owned by its producing adapter.
    // The callback answers positive-gap body queries in this same arena/domain.
    std::function<std::optional<Id>(const RegionalEvent&, const RegionalEvent&, Id)> weightedAcross;
    // Only crossing targets are quotient vertices. A unit edge contracts one
    // complete within-body path followed by one crossing; no zero-edge body
    // closure is repeated in the quotient.
    std::vector<RegionalEvent> slots;
    std::vector<std::vector<std::size_t>> targetCrossings;
    // Base distances and higher recurrence levels are memoized only on demand.
    // Initializing the interface must not allocate a quadratic port table.
    std::map<std::tuple<std::size_t, std::size_t, std::size_t>, Id> distanceMemo;
    // Every across-visit path uses a boundary port. Factor Boolean conditions
    // necessary for any live port once, and restore them on across queries.
    Id portEnable = RegionExpressions::invalid;
    std::unique_ptr<RegionExpressions::Substitution> portCondition;
    uint64_t infinity = 0;
    using EventKey = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
    std::map<std::pair<EventKey, EventKey>, std::optional<Id>> queryMemo, bodyPortMemo;
    // Body coordinates only: R lazy distance columns and R first-crossing
    // costs. Each finite answer includes at least one crossing. Invalid means
    // not queried yet; every outer visit and consumer shares these entries.
    std::map<EventKey, std::vector<Id>> sourceDistances;
    std::string error;
    RegionExpressions& e() { return *body.expressions; }
    bool buildBoundary();
    bool buildNativeBoundary();
    bool closePorts();
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepare(ArrayRef<scf::ForOp> enclosing);
    std::optional<Id> query(RegionalEvent source, RegionalEvent target);
    std::optional<Id> present(RegionalEvent event);
    std::optional<Id> computeQuery(RegionalEvent source, RegionalEvent target);
    // Translation-invariant relation on body events, independent of finite
    // outer invocation bounds. Callers separately establish actual endpoints.
    std::optional<Id> relativeQuery(RegionalEvent source, RegionalEvent target, uint64_t gap);
    std::optional<Id> acrossQuery(const RegionalEvent& source, const RegionalEvent& target, Id gap);
    std::optional<Id> distanceFrom(const RegionalEvent& source, std::size_t column);
    std::optional<Id> crossingStep(const RegionalEvent& source, std::size_t target);
    Id underPortEnable(Id expression);
    std::optional<Id> portDistance(std::size_t source, std::size_t target);
    std::optional<Id> bodyPortQuery(const RegionalEvent& source, const RegionalEvent& target);
};
void attachNumericalRepeatedExports(const std::shared_ptr<RepeatedRegionState>& state, RegionalAnalysis& out);
// Export a validated state's body, coordinates, selectors and recipes. Callers
// establish its crossing graph (possibly empty for a certified singleton).
RepeatedRegionAnalysis exportRepeatedRegion(std::shared_ptr<RepeatedRegionState> state);
} // namespace mlir::pto::frontiersynch
#endif
