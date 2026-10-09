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
struct RegionalNumericalInterface;
struct RegionalCost {
    uint64_t repeatedRegions = 0, phaseDescriptions = 0;
    uint64_t children = 0;
    uint64_t cells = 0;
    uint64_t ports = 0;
    uint64_t crossings = 0;
    uint64_t physicalFragments = 0, rotatingResidues = 0, numericVisits = 0;
    uint64_t arithmeticRegions = 0;
    uint64_t boundaryBytes = 0; // Physical bytes represented by the certified boundary atoms.
    uint64_t selectorComparisons = 0, crossingCandidates = 0, expressionNodes = 0;
    // Owned analysis arenas distinct from the composing regional expression DAG.
    uint64_t retainedExpressionNodes = 0;
    // Proof requests, including cached requests. Each producer states its query
    // cost: guarded implication and numerical boundary proofs have different costs.
    uint64_t implicationChecks = 0;
    uint64_t numericalLeafQueries = 0, numericalIndexOperations = 0;
    uint64_t numericalMerges = 0, numericalReusedChildren = 0;
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
// Each first/last list contains guarded alternatives for its single extremum
// (readers separately per pipe). A macro occurrence may have simultaneous
// envelopes on distinct pipes; retain all of them without assuming internal
// completion order. An empty list proves that no such boundary exists.
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
struct RegionalByteAddress {
    AddressSpace space;
    Value base;
    RegionExpressions::Id offset = RegionExpressions::invalid;
};
struct RegionalStorageSelectors {
    std::vector<RegionalSelector> firstWriters, lastWriters;
    std::map<uint32_t, std::vector<RegionalSelector>> firstReaders, lastReaders;
};
// Optional algebraic storage evidence. Family reservation ownership and actual
// access membership are separate: unused holes have no storage selectors.
struct RegionalStorageOwner {
    RegionExpressions::Id present, visit, localByte;
};
enum class RegionalStorageFamilyKind { General, VisitOwned, SharedReadOnly };
struct RegionalStorageFamily {
    AddressSpace space = AddressSpace::GM;
    Value base;
    std::vector<std::size_t> effects;
    RegionalStorageFamilyKind kind = RegionalStorageFamilyKind::General;
    std::function<std::optional<RegionExpressions::Id>(RegionalByteAddress)> membership;
    std::function<std::optional<RegionalStorageOwner>(RegionalByteAddress)> owner;
};
struct RegionalSymbolicStorageCertificate {
    std::shared_ptr<RegionExpressions> expressions;
    const SyncStorageEffects* accessModel = nullptr;
    GMAliasPolicy gmAliasPolicy = GMAliasPolicy::MayAlias;
    std::vector<RegionalStorageFamily> families;
    // Complete uniform-selector atoms for the named effects. These are proved
    // for every byte of each atom, not sampled at a representative address.
    // Symbolic support not covered here keeps its per-cell query interface.
    std::vector<std::size_t> uniformEffects;
    std::vector<RegionalStorageBoundary> uniformBoundaries;
};
struct ArithmeticRegionalRelations;
struct RegionalRelations;
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
    // Parallel coordinate divisors. Empty means original ordinal in every frame.
    std::vector<std::vector<uint64_t>> outerDivisors;
    // Strict payload reference order, including nested coordinates. Event kind
    // does not change occurrence order. Flat producers may use the default.
    // Exact query producers include intrinsic start->completion and native
    // per-pipe start/start and completion/completion order. Every other edge
    // between distinct occurrences advances this reference order. A type and
    // its full coordinate tuple identify an occurrence, even when phase views
    // share the same original payload anchor.
    // Thus S->S, C->C and S->C on one pipe are decided by this order and presence.
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)> referenceBefore;
    // Uniform first ordinal of a contiguous periodic slice. Absent for ordinary
    // explicit/whole-loop exports, whose occurrence numbering starts at zero.
    // Applies to every present payload type; expression belongs to expressions.
    std::optional<RegionExpressions::Id> firstOrdinal;
    // Optional phase-view binding at original cuts. Site guard identifies this
    // phase; invocation guard also tests a counterpart in a partial period.
    std::optional<RegionExpressions::Id> endpointSiteGuard, endpointInvocationGuard;
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent)> endpointEventGuard;
    std::vector<RegionalStorageBoundary> storageBoundary;
    const SyncStorageEffects* accessModel = nullptr; // Borrows the unchanged shared input.
    std::vector<RegionalAccessBoundary> accessBoundary;
    // Discharged within this region, but retained for a later owner adapter.
    // Ordinary finite composition must not reinterpret these as uniform edges.
    std::vector<RegionalAccessBoundary> deferredAccessBoundary;
    // nullopt is unavailable; an empty selector set proves absence.
    std::function<std::optional<RegionalStorageSelectors>(RegionalByteAddress)> storageSelectors;
    std::vector<std::size_t> symbolicStorageEffects;
    std::shared_ptr<const RegionalSymbolicStorageCertificate> symbolicStorage;
    // Exact relational export is separate from callable query circuits. An
    // identity/presence/order transformation must remap it or clear it.
    std::shared_ptr<const ArithmeticRegionalRelations> arithmeticRelations;
    std::shared_ptr<const RegionalRelations> relations;
    std::map<uint32_t, std::vector<RegionalSelector>> firstPayloads, lastPayloads;
    // Optional first executed occurrence of each payload type. A present key
    // with an empty vector proves absence; a missing key means unavailable.
    // Unlike firstPayloads, these extrema are per site, not per pipe. Consumers
    // also apply regionalPresence, preserving any enclosing conditional mask.
    std::map<uint32_t, std::vector<RegionalSelector>> firstSitePayloads;
    RegionalCapabilities capabilities;
    RegionalCost cost;
    // Exact immutable numerical port order for this selected graph/context.
    // Graph, identity, or presence adapters must clear it unless transformed.
    std::shared_ptr<const RegionalNumericalInterface> numerical;
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
    // Bind a subtree in enclosing invariant repeats, preserving original cuts.
    // Each repeat prefixes its own matching coordinate after preparing its body.
    std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(ArrayRef<scf::ForOp>)> prepareWithVisits;
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
// Reuse from WAIT before consumer to SET after producer. Same-pipe command
// order is sufficient; other directions require completion-before-start order.
std::optional<RegionExpressions::Id> regionalHandoffReuse(
    const RegionalAnalysis& region, RegionalEvent consumer, RegionalEvent producer);
std::optional<RegionExpressions::Id> regionalReferenceBefore(
    const RegionalAnalysis& region, RegionalEvent source, RegionalEvent target);
} // namespace mlir::pto::frontiersynch
#endif
