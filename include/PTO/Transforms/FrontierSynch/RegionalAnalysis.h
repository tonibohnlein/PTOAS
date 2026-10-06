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
// This exporter materializes slot visits before selector and port-graph work.
// Bound that expansion independently of a compact modulus and runtime trips.
// The periodic quotient itself does not require slot enumeration.
inline constexpr uint64_t maxRegionalSlotVisits = 256;
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
    // Enclosing repeat ordinals, outermost first. The leaf ordinal remains
    // separate; explicit leaves use zero. Never flatten coordinates by trips.
    std::vector<RegionExpressions::Id> visits;
};
using RegionalDemandFilter = std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)>;
struct RegionalSelector { RegionalEvent event; RegionExpressions::Id present = 0; };
struct RegionalStorageBoundary {
    SyncStorageCell cell;
    std::vector<RegionalSelector> firstWriters, lastWriters;
    std::map<uint32_t, std::vector<RegionalSelector>> firstReaders, lastReaders;
};
struct RegionalAccessBoundary {
    std::size_t effect = 0;
    RegionalSelector first, last;
    bool representedByCells = false;
};
struct RegionalCapabilities {
    bool completeStorageModel = false, exactQueries = false, exactSelectors = false, endpointRecipes = false;
    bool contextualGuards = false; // Endpoint circuits may contain branch-local predicates.
};
struct RegionalAnalysis {
    std::shared_ptr<RegionExpressions> expressions;
    std::vector<TemplateEndpointAnchor> anchors;
    std::vector<scf::ForOp> occurrenceLoops; // Empty loop: ordinal must be zero.
    // Empty vector is the legacy flat interface. Otherwise one frame per type,
    // matching RegionalEvent::visits. Original loops also identify endpoint cuts.
    std::vector<std::vector<scf::ForOp>> outerLoops;
    // Strict payload reference order, including nested coordinates. Event kind
    // does not change occurrence order. Flat producers may use the default.
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)> referenceBefore;
    // Uniform first ordinal of a contiguous periodic slice. Absent for ordinary
    // explicit/whole-loop exports, whose occurrence numbering starts at zero.
    // Applies to every present payload type; expression belongs to expressions.
    std::optional<RegionExpressions::Id> firstOrdinal;
    std::vector<RegionalStorageBoundary> storageBoundary;
    const SyncStorageEffects* accessModel = nullptr; // Borrows the unchanged shared input.
    std::vector<RegionalAccessBoundary> accessBoundary;
    std::map<uint32_t, std::vector<RegionalSelector>> firstPayloads, lastPayloads;
    RegionalCapabilities capabilities;
    RegionalCost cost;
    // Exports, including discharged effects, belong to one unchanged invocation
    // and alias context. Composition must not reinterpret that context.
    // This field records the assumption on any exported based GM cells.
    GMAliasPolicy gmAliasPolicy = GMAliasPolicy::MayAlias;
    // Queries include endpoint presence and reflexivity. Invalid input is nullopt;
    // a valid unreachable pair is the arena's false expression.
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent)> presence;
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)> reachability;
    // The producer owns exact internal demand recipes through this closure.
    // Preparation is detached, adds no boundary drains, and cannot invalidate
    // the query/selector result if an endpoint-placement obligation fails.
    std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>()> prepare;
    // Optional overlay contract: refine each base cover before emitting either
    // endpoint. Both sides query the same actual completion/start pair, using
    // the inverse endpoint map at WAIT. Failure leaves original IR unchanged.
    std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(const RegionalDemandFilter&)> prepareFiltered;
};
// Validate coordinate shape before forwarding to a backend that might only
// understand flat events. Query-only flat producers may omit anchors and validate
// type IDs themselves. False presence is distinct from a malformed identity.
bool validRegionalEvent(const RegionalAnalysis& region, const RegionalEvent& event);
std::optional<RegionExpressions::Id> regionalPresence(const RegionalAnalysis& region, RegionalEvent event);
std::optional<RegionExpressions::Id> regionalReachability(
    const RegionalAnalysis& region, RegionalEvent source, RegionalEvent target);
std::optional<RegionExpressions::Id> regionalReferenceBefore(
    const RegionalAnalysis& region, RegionalEvent source, RegionalEvent target);
} // namespace mlir::pto::frontiersynch
#endif
