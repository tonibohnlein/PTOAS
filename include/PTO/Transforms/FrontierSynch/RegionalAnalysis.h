// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALANALYSIS_H
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include <functional>
#include <map>
namespace mlir::pto::frontiersynch {
struct RegionalCost {
    uint64_t children = 0;
    uint64_t cells = 0;
    uint64_t ports = 0;
    uint64_t crossings = 0;
    uint64_t physicalFragments = 0, rotatingResidues = 0, numericVisits = 0;
    uint64_t selectorComparisons = 0, crossingCandidates = 0, expressionNodes = 0;
    uint64_t implicationChecks = 0; // Crossing-generation proofs; O(A*G) per check.
};
struct RegionalEvent {
    uint32_t type = 0;
    RegionExpressions::Id ordinal = 0;
    PeriodicEventKind kind = PeriodicEventKind::Start;
};
struct RegionalSelector { RegionalEvent event; RegionExpressions::Id present = 0; };
struct RegionalStorageBoundary {
    SyncStorageCell cell;
    std::vector<RegionalSelector> firstWriters, lastWriters;
    std::map<uint32_t, std::vector<RegionalSelector>> firstReaders, lastReaders;
};
struct RegionalCapabilities {
    bool exactEffects = false, exactQueries = false, exactSelectors = false, endpointRecipes = false;
    bool contextualGuards = false; // Endpoint circuits may contain branch-local predicates.
};
struct RegionalAnalysis {
    std::shared_ptr<RegionExpressions> expressions;
    std::vector<TemplateEndpointAnchor> anchors;
    std::vector<scf::ForOp> occurrenceLoops; // Empty loop: ordinal must be zero.
    std::vector<RegionalStorageBoundary> storageBoundary;
    std::map<uint32_t, std::vector<RegionalSelector>> firstPayloads, lastPayloads;
    RegionalCapabilities capabilities;
    RegionalCost cost;
    // Queries include endpoint presence and reflexivity. Invalid input is nullopt;
    // a valid unreachable pair is the arena's false expression.
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent)> presence;
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)> reachability;
    // The producer owns exact internal demand recipes through this closure.
    // Preparation is detached, adds no boundary drains, and cannot invalidate
    // the query/selector result if an endpoint-placement obligation fails.
    std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>()> prepare;
};
} // namespace mlir::pto::frontiersynch
#endif
