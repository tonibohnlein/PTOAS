// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverCandidateIndex.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

using namespace mlir::pto;

static_assert(!std::is_move_constructible<SyncCoverGraph>::value,
              "moving a graph must not bypass generation tracking");
static_assert(!std::is_assignable<SyncCoverGraph &, SyncCoverGraph>::value,
              "assigning a graph must not bypass generation tracking");

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverGraphTest failure: " << message << '\n';
  }
  return condition;
}

bool check(const SyncCoverGraphResult &result, std::string_view message) {
  return check(static_cast<bool>(result), message);
}

std::size_t takeIndex(const SyncCoverGraphResult &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverEdge makeEdge(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  return edge;
}

SyncCoverDemand makeDemand(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  return demand;
}

bool testZeroDistanceDag() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add first node");
  const SyncCoverNodeId second =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add second node");
  const SyncCoverNodeId third =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add third node");
  passed &=
      check(graph.addEdge(makeEdge(first, second)), "first edge is valid");
  passed &=
      check(graph.addEdge(makeEdge(second, third)), "second edge is valid");
  passed &= check(graph.validate(), "forward zero-distance graph is acyclic");

  const std::size_t edgeCount = graph.getEdges().size();
  const SyncCoverGraphResult selfEdge = graph.addEdge(makeEdge(third, third));
  passed &= check(selfEdge.error == SyncCoverGraphError::ZeroDistanceSelfEdge,
                  "zero-distance self edge has a precise error");
  passed &= check(graph.getEdges().size() == edgeCount,
                  "failed edge insertion does not mutate the graph");
  passed &= check(graph.addEdge(makeEdge(third, first)).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "zero-distance edges must follow the global timeline");
  passed &= check(graph.addDemand(makeDemand(third, first)).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "zero-distance demands must follow the global timeline");
  passed &= check(graph.validate(), "ordered zero-distance graph validates");
  return passed;
}

bool testRecurrenceScopes() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, false, std::nullopt, true), passed, "add outer loop");
  const SyncCoverScopeId inner =
      takeIndex(graph.addScope(outer, false, std::nullopt, true), passed,
                "add inner loop");
  const SyncCoverScopeId sibling = takeIndex(
      graph.addScope(0, false, std::nullopt, true), passed, "add sibling loop");
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, inner, 0), passed, "add nested node");
  const SyncCoverNodeId second = takeIndex(graph.addNode(2, 1, inner, 1),
                                           passed, "add second nested node");
  const SyncCoverNodeId siblingNode =
      takeIndex(graph.addNode(3, 1, sibling, 6), passed, "add sibling node");
  const SyncCoverNodeId direct =
      takeIndex(graph.addNode(4, 1, outer, 2), passed, "add direct child node");
  const SyncCoverNodeId root =
      takeIndex(graph.addNode(5, 1, 0, 3), passed, "add root node");

  SyncCoverEdge recurrence = makeEdge(first, first);
  recurrence.kind = SyncCoverEdgeKind::CompletionSupply;
  recurrence.scope = outer;
  recurrence.distance = 2;
  passed &= check(graph.addEdge(recurrence),
                  "enclosing-loop self recurrence is valid");

  SyncCoverDemand nested = makeDemand(first, second);
  nested.scope = outer;
  nested.distance = 1;
  passed &= check(graph.addDemand(nested),
                  "enclosing loop may own nested recurrence demand");

  SyncCoverDemand directRecurrence = makeDemand(direct, direct);
  directRecurrence.scope = outer;
  directRecurrence.distance = 1;
  passed &= check(graph.addDemand(directRecurrence),
                  "positive-distance self-demand is a valid recurrence");

  SyncCoverDemand unrelated = makeDemand(first, siblingNode);
  unrelated.scope = inner;
  unrelated.distance = 1;
  passed &=
      check(unrelated.scope != sibling, "test uses unrelated sibling scopes");
  passed &= check(graph.addDemand(unrelated).error ==
                      SyncCoverGraphError::InvalidScope,
                  "unrelated scope cannot own a recurrence demand");

  SyncCoverEdge rootEndpoint = makeEdge(direct, root);
  rootEndpoint.scope = outer;
  passed &= check(graph.addEdge(rootEndpoint).error ==
                      SyncCoverGraphError::InvalidScope,
                  "nested scope cannot own an edge to a root node");

  SyncCoverEdge missingScope = makeEdge(first, second);
  missingScope.distance = 1;
  passed &= check(graph.addEdge(missingScope).error ==
                      SyncCoverGraphError::InvalidDistance,
                  "positive distance requires a non-root recurrence scope");
  const SyncCoverScopeId region =
      takeIndex(graph.addScope(), passed, "add non-loop region");
  const SyncCoverNodeId regionSource =
      takeIndex(graph.addNode(6, 1, region, 4), passed, "add region source");
  const SyncCoverNodeId regionTarget =
      takeIndex(graph.addNode(7, 1, region, 5), passed, "add region target");
  SyncCoverDemand nonLoopRecurrence = makeDemand(regionSource, regionTarget);
  nonLoopRecurrence.scope = region;
  nonLoopRecurrence.distance = 1;
  passed &= check(graph.addDemand(nonLoopRecurrence).error ==
                      SyncCoverGraphError::InvalidDistance,
                  "positive distance requires an explicitly modeled loop");
  passed &= check(graph.addDemand(makeDemand(direct, direct)).error ==
                      SyncCoverGraphError::ZeroDistanceSelfDemand,
                  "zero-distance self-demand is rejected");
  passed &= check(graph.validate(), "valid recurrences preserve body DAG");
  return passed;
}

bool testStructuredGuards() {
  bool passed = true;
  SyncCoverGuard unsorted{{{2, 1}, {1, 0}, {2, 1}}};
  SyncCoverGuard required{{{1, 0}}};
  SyncCoverGuard incompatible{{{1, 1}}};
  passed &= check(syncCoverGuardImplies(unsorted, required),
                  "guard implication normalizes its arguments");
  passed &= check(!syncCoverGuardsCompatible(unsorted, incompatible),
                  "guard compatibility normalizes its arguments");
  SyncCoverGuard malformed{{{1, 0}, {1, 1}}};
  passed &= check(!syncCoverGuardImplies(malformed, required),
                  "malformed implication input fails closed");

  SyncCoverGraph graph;
  takeIndex(graph.addControl(2), passed, "add branch control");
  takeIndex(graph.addControl(2), passed, "add nested control");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, 0, 0, {{{0, 0}}}), passed, "add guarded source");
  const SyncCoverNodeId target = takeIndex(
      graph.addNode(2, 1, 0, 1, {{{1, 1}}}), passed, "add guarded target");
  passed &= check(graph.addEdge(makeEdge(source, target)),
                  "compatible endpoint guards form an edge guard");
  const SyncCoverEdge &edge = graph.getEdges().back();
  passed &= check(edge.sourceGuard.literals ==
                          std::vector<SyncCoverGuardLiteral>{{0, 0}} &&
                      edge.targetGuard.literals ==
                          std::vector<SyncCoverGuardLiteral>{{1, 1}},
                  "edge preserves each endpoint execution condition");

  const SyncCoverNodeId alternative = takeIndex(
      graph.addNode(3, 1, 0, 2, {{{0, 1}}}), passed, "add alternative node");
  passed &= check(graph.addEdge(makeEdge(source, alternative)).error ==
                      SyncCoverGraphError::IncompatibleEndpoints,
                  "mutually exclusive endpoints are rejected");
  passed &= check(graph.addNode(4, 1, 0, 3, {{{2, 0}}}).error ==
                      SyncCoverGraphError::InvalidControl,
                  "unknown controls are rejected");
  passed &= check(graph.addNode(4, 1, 0, 3, {{{0, 2}}}).error ==
                      SyncCoverGraphError::InvalidControl,
                  "out-of-range alternatives are rejected");
  passed &=
      check(graph.addControl(0).error == SyncCoverGraphError::InvalidControl,
            "controls require at least one alternative");
  return passed;
}

bool testRecurrenceGuardContexts() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add guard recurrence loop");
  const SyncCoverControlId outside =
      takeIndex(graph.addControl(2), passed, "add outside control");
  const SyncCoverControlId inside =
      takeIndex(graph.addControl(2, loop), passed, "add inside control");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0, {{{outside, 0}, {inside, 0}}}),
                passed, "add recurrence source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 1, {{{outside, 0}, {inside, 1}}}),
                passed, "add recurrence target");
  SyncCoverEdge recurrence = makeEdge(source, target);
  recurrence.scope = loop;
  recurrence.distance = 1;
  passed &= check(graph.addEdge(recurrence),
                  "per-iteration alternatives may differ across recurrence");
  const SyncCoverEdge &stored = graph.getEdges().back();
  passed &= check(stored.sourceGuard.literals != stored.targetGuard.literals,
                  "recurrence keeps source and target occurrence guards");

  const SyncCoverNodeId incompatible =
      takeIndex(graph.addNode(3, 1, loop, 2, {{{outside, 1}, {inside, 0}}}),
                passed, "add outer-alternative node");
  SyncCoverEdge rejected = makeEdge(source, incompatible);
  rejected.scope = loop;
  rejected.distance = 1;
  passed &= check(graph.addEdge(rejected).error ==
                      SyncCoverGraphError::IncompatibleEndpoints,
                  "loop-invariant alternatives remain incompatible");

  const SyncCoverNodeId same =
      takeIndex(graph.addNode(4, 1, loop, 3, {{{outside, 0}, {inside, 0}}}),
                passed, "add same-alternative recurrence target");
  SyncCoverDemand sameAlternative = makeDemand(source, same);
  sameAlternative.scope = loop;
  sameAlternative.distance = 1;
  passed &= check(graph.addDemand(sameAlternative),
                  "same alternatives remain valid across occurrences");
  const SyncCoverDemand &storedDemand = graph.getDemands().back();
  passed &= check(!storedDemand.sourceGuard.literals.empty() &&
                      !storedDemand.targetGuard.literals.empty(),
                  "same guards remain attached to distinct occurrences");
  return passed;
}

bool testInvalidReferencesDoNotMutate() {
  bool passed = true;
  SyncCoverGraph graph;
  const std::size_t scopeCount = graph.getScopes().size();
  passed &= check(graph.addScope(7).error == SyncCoverGraphError::InvalidScope,
                  "invalid parent scope is diagnosed");
  passed &= check(graph.getScopes().size() == scopeCount,
                  "failed scope insertion does not mutate the graph");
  passed &= check(graph.addNode(1, 1, 7, 0).error ==
                      SyncCoverGraphError::InvalidScope,
                  "invalid node scope is diagnosed");
  passed &=
      check(graph.addControl(2, 7).error == SyncCoverGraphError::InvalidScope,
            "invalid control scope is diagnosed");
  passed &= check(graph.addDemand(makeDemand(0, 1)).error ==
                      SyncCoverGraphError::InvalidNode,
                  "invalid demand nodes are diagnosed");
  return passed;
}

bool testTimelineAnchorsAndCompletionCapabilities() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, SyncCoverTimelineInterval{2, 9}, true),
                passed, "add explicit loop timeline");
  const SyncCoverScopeId untimedChild =
      takeIndex(graph.addScope(loop), passed, "add untimed child scope");
  passed &=
      check(graph.addScope(untimedChild, false, SyncCoverTimelineInterval{1, 4})
                    .error == SyncCoverGraphError::InvalidTimeline,
            "child timeline must fit the nearest timelined ancestor");
  passed &= check(
      graph.addScope(untimedChild, false, SyncCoverTimelineInterval{3, 4}),
      "contained timeline may cross an untimed direct parent");
  const SyncCoverScopeId unknown =
      takeIndex(graph.addScope(), passed, "add unknown timeline scope");
  const SyncCoverNodeId node =
      takeIndex(graph.addNode(3, 1, loop, 1, {}, {7, 5, 7}), passed,
                "add node with completion destinations");
  passed &= check(graph.getNodes()[node].completionTargets ==
                      std::vector<std::uint32_t>{5, 7},
                  "completion destinations are normalized");
  passed &= check(syncCoverNodeCanProduceCompletion(graph, node, 5) &&
                      !syncCoverNodeCanProduceCompletion(graph, node, 6),
                  "completion capability is destination-specific");

  const SyncCoverAnchor before{SyncCoverAnchorKind::BeforeNode, node, 0};
  const SyncCoverAnchor after{SyncCoverAnchorKind::AfterNode, node, 0};
  const SyncCoverAnchor entry{SyncCoverAnchorKind::ScopeEntry, 0, loop};
  const SyncCoverAnchor exit{SyncCoverAnchorKind::ScopeExit, 0, loop};
  const SyncCoverAnchor point{SyncCoverAnchorKind::TimelinePoint, 0, loop, 6};
  passed &= check(resolveSyncCoverAnchor(graph, before) == 2 &&
                      resolveSyncCoverAnchor(graph, after) == 3 &&
                      resolveSyncCoverAnchor(graph, entry) == 2 &&
                      resolveSyncCoverAnchor(graph, exit) == 9 &&
                      resolveSyncCoverAnchor(graph, point) == 6,
                  "node, scope, and exact anchors share one global timeline");
  passed &= check(
      !resolveSyncCoverAnchor(
          graph, {SyncCoverAnchorKind::TimelinePoint, 0, loop, 10}),
      "timeline-point anchors must fit their occurrence scope");
  passed &= check(!resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::ScopeEntry, 0, unknown}),
                  "missing scope timeline fails anchor resolution closed");
  passed &= check(!resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::BeforeNode, node, loop}),
                  "unused anchor fields are rejected");
  passed &= check(graph.addNode(4, 1, loop, 5).error ==
                      SyncCoverGraphError::InvalidTimeline,
                  "node positions must fit every enclosing timeline");

  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  passed &= check(graph.addNode(4, 1, 0, maximum / 2 + 1).error ==
                      SyncCoverGraphError::InvalidTimeline,
                  "node timeline multiplication cannot overflow");
  passed &= check(graph.validate(), "timeline graph validates");
  return passed;
}

bool testScopeQueriesAndGeneration() {
  bool passed = true;
  SyncCoverGraph graph;
  const std::size_t initialGeneration = graph.getGeneration();
  passed &= check(graph.addScope(99).error == SyncCoverGraphError::InvalidScope,
                  "failed graph mutations preserve the generation");
  passed &= check(graph.getGeneration() == initialGeneration,
                  "failed scope insertion does not advance the generation");

  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, false, SyncCoverTimelineInterval{0, 20}, true), passed,
      "add scope-query outer loop");
  const SyncCoverScopeId inner =
      takeIndex(graph.addScope(outer, true), passed,
                "add must-execute inner scope");
  const SyncCoverScopeId nestedLoop = takeIndex(
      graph.addScope(inner, false, SyncCoverTimelineInterval{2, 18}, true),
      passed, "add nested loop scope");
  passed &= check(graph.getGeneration() == initialGeneration + 3,
                  "successful graph mutations advance the generation");
  passed &= check(graph.scopeContains(outer, nestedLoop) &&
                      !graph.scopeContains(nestedLoop, outer),
                  "scope containment follows the complete ancestor chain");
  passed &= check(graph.scopeMustExecuteWithin(outer, inner) &&
                      !graph.scopeMustExecuteWithin(outer, nestedLoop),
                  "must-execute queries stop at an optional descendant");
  passed &= check(graph.scopeExecutesWhen(nestedLoop, outer) &&
                      !graph.scopeExecutesWhen(outer, nestedLoop),
                  "scope execution implication is direction-sensitive");
  passed &= check(graph.getLowestCommonScope(inner, nestedLoop) == inner &&
                      graph.getLowestCommonScope(nestedLoop, outer) == outer &&
                      !graph.getLowestCommonScope(99, outer),
                  "lowest-common-scope queries handle nesting and errors");
  passed &= check(graph.getScopeLoopDepth(nestedLoop) == 2 &&
                      graph.getScopeLoopDepth(nestedLoop, false) == 1 &&
                      !graph.getScopeLoopDepth(99),
                  "shared loop-depth queries handle scope anchors and errors");

  const SyncCoverNodeId first = takeIndex(
      graph.addNode(1, 1, inner, 1), passed, "add timeline-order owner");
  static_cast<void>(first);
  passed &= check(graph.addNode(2, 1, inner, 1).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "node order is injective within one owning timeline");
  passed &= check(graph.addNode(2, 1, nestedLoop, 1),
                  "nested timelines have independent node-order spaces");
  return passed;
}

bool testStorageProvenanceContract() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add storage source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add storage target");
  const SyncCoverStorageDomainId domain = takeIndex(
      graph.addStorageDomain(), passed, "add local storage domain");
  const SyncCoverStorageDomainId otherDomain = takeIndex(
      graph.addStorageDomain(), passed, "add second storage domain");
  const SyncCoverStorageAccessId sourceWrite = takeIndex(
      graph.addStorageAccess(source, domain, 10, {0, 16},
                             SyncCoverStorageAccessMode::Write, 0),
      passed, "add exact source write");
  const SyncCoverStorageAccessId mergedSource = takeIndex(
      graph.addStorageAccess(source, domain, 10, {0, 16},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "merge modes of one logical access");
  passed &= check(sourceWrite == mergedSource &&
                      graph.getStorageAccesses()[sourceWrite].mode ==
                          SyncCoverStorageAccessMode::ReadWrite,
                  "identical access records merge read-write mode");
  const SyncCoverStorageAccessId reusedAddress = takeIndex(
      graph.addStorageAccess(source, domain, 11, {0, 16},
                             SyncCoverStorageAccessMode::Write, 0),
      passed, "retain a distinct logical access family");
  passed &= check(reusedAddress != sourceWrite,
                  "physical reuse does not erase logical family identity");
  const SyncCoverStorageAccessId targetRead = takeIndex(
      graph.addStorageAccess(target, domain, 12, {8, 24},
                             SyncCoverStorageAccessMode::Read, 1),
      passed, "add exact target read");
  const SyncCoverStorageAccessId adjacent = takeIndex(
      graph.addStorageAccess(target, domain, 13, {16, 32},
                             SyncCoverStorageAccessMode::Read, 2),
      passed, "add adjacent access");
  const SyncCoverStorageAccessId other = takeIndex(
      graph.addStorageAccess(target, otherDomain, 14, {8, 24},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "add access in another domain");
  const SyncCoverStorageWitnessId witness = takeIndex(
      graph.addStorageWitness(sourceWrite, targetRead), passed,
      "add exact overlap witness");
  passed &= check(graph.getStorageWitnesses()[witness].overlap ==
                      SyncCoverStorageInterval{8, 16},
                  "witness stores the exact strict intersection");
  passed &= check(graph.addStorageWitness(sourceWrite, adjacent).error ==
                      SyncCoverGraphError::InvalidStorageWitness,
                  "adjacent half-open intervals do not overlap");
  passed &= check(graph.addStorageWitness(sourceWrite, other).error ==
                      SyncCoverGraphError::InvalidStorageWitness,
                  "witness endpoints must share a storage domain");

  SyncCoverDemand raw = makeDemand(source, target);
  raw.kind = SyncCoverDemandKind::MemoryRAW;
  raw.storageProvenance = SyncCoverStorageProvenance::Complete;
  raw.storageWitnesses = {witness, witness};
  passed &= check(graph.addDemand(raw), "complete RAW provenance is valid");
  passed &= check(graph.getDemands().back().storageWitnesses.size() == 1,
                  "demand witness identities are normalized");

  SyncCoverDemand incomplete = makeDemand(source, target);
  incomplete.kind = SyncCoverDemandKind::MemoryWAR;
  incomplete.storageProvenance = SyncCoverStorageProvenance::Incomplete;
  passed &= check(graph.addDemand(incomplete),
                  "unknown memory provenance remains representable");
  SyncCoverDemand emptyComplete = incomplete;
  emptyComplete.storageProvenance = SyncCoverStorageProvenance::Complete;
  passed &= check(graph.addDemand(emptyComplete).error ==
                      SyncCoverGraphError::InvalidStorageProvenance,
                  "complete memory provenance requires a witness");
  SyncCoverDemand invalidRole = raw;
  invalidRole.kind = SyncCoverDemandKind::MemoryWAW;
  passed &= check(graph.addDemand(invalidRole).error ==
                      SyncCoverGraphError::InvalidStorageWitness,
                  "witness access modes must implement the hazard direction");
  SyncCoverDemand invalidSSA = makeDemand(source, target);
  invalidSSA.storageWitnesses = {witness};
  passed &= check(graph.addDemand(invalidSSA).error ==
                      SyncCoverGraphError::InvalidStorageProvenance,
                  "SSA demands cannot carry storage provenance");
  passed &= check(graph.validate(), "storage provenance graph validates");
  return passed;
}

bool testCandidateOpportunityRecurrenceContext() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 9}, true), passed,
      "add opportunity loop");
  const SyncCoverControlId branch =
      takeIndex(graph.addControl(2, loop), passed, "add opportunity branch");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(7, 1, loop, 2, {{{branch, 0}}}), passed,
      "add opportunity source");
  const SyncCoverNodeId target = takeIndex(
      graph.addNode(9, 1, loop, 4, {{{branch, 1}}}), passed,
      "add opportunity target");
  const SyncCoverStorageDomainId domain = takeIndex(
      graph.addStorageDomain(), passed, "add opportunity storage domain");
  const SyncCoverStorageAccessId sourceAccess = takeIndex(
      graph.addStorageAccess(source, domain, 10, {16, 32},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "add opportunity source access");
  const SyncCoverStorageAccessId targetAccess = takeIndex(
      graph.addStorageAccess(target, domain, 11, {16, 32},
                             SyncCoverStorageAccessMode::Write, 1),
      passed, "add opportunity target access");
  const SyncCoverStorageWitnessId witness = takeIndex(
      graph.addStorageWitness(sourceAccess, targetAccess), passed,
      "add opportunity recurrence witness");
  SyncCoverDemand recurrence = makeDemand(source, target);
  recurrence.kind = SyncCoverDemandKind::MemoryWAR;
  recurrence.scope = loop;
  recurrence.distance = 1;
  recurrence.storageProvenance = SyncCoverStorageProvenance::Complete;
  recurrence.storageWitnesses = {witness};
  passed &= check(graph.addDemand(recurrence),
                  "add opportunity recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze opportunity graph");

  const SyncCoverCandidateIndex index(graph);
  const auto opportunities = index.getOpportunities();
  const bool exactContext =
      opportunities && opportunities.value->size() == 1 &&
      opportunities.value->front().sourceResource == 7 &&
      opportunities.value->front().targetResource == 9 &&
      opportunities.value->front().scope == loop &&
      opportunities.value->front().distance == 1 &&
      opportunities.value->front().sourceGuard.literals ==
          std::vector<SyncCoverGuardLiteral>{{branch, 0}} &&
      opportunities.value->front().targetGuard.literals ==
          std::vector<SyncCoverGuardLiteral>{{branch, 1}} &&
      opportunities.value->front().slot &&
      opportunities.value->front().slot->sourceAddressOrdinal == 0 &&
      opportunities.value->front().slot->targetAddressOrdinal == 1;
  passed &= check(exactContext,
                  "candidate opportunities preserve recurrence context and "
                  "per-occurrence guards");
  return passed;
}

bool testStructuralFreezeAndCandidateIndex() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverControlId branch =
      takeIndex(graph.addControl(2), passed, "add index branch");
  const SyncCoverNodeId first = takeIndex(
      graph.addNode(1, 1, 0, 0, {{{branch, 0}}}), passed,
      "add first indexed node");
  const SyncCoverNodeId second = takeIndex(
      graph.addNode(1, 1, 0, 1, {{{branch, 0}}}), passed,
      "add second indexed node");
  const SyncCoverNodeId alternative = takeIndex(
      graph.addNode(1, 1, 0, 2, {{{branch, 1}}}), passed,
      "add alternative indexed node");
  const SyncCoverStorageDomainId domain = takeIndex(
      graph.addStorageDomain(), passed, "add indexed storage domain");
  const SyncCoverStorageAccessId firstAccess = takeIndex(
      graph.addStorageAccess(first, domain, 1, {0, 8},
                             SyncCoverStorageAccessMode::Write, 0),
      passed, "add indexed source access");
  const SyncCoverStorageAccessId secondAccess = takeIndex(
      graph.addStorageAccess(second, domain, 2, {0, 8},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "add indexed target access");
  const SyncCoverStorageWitnessId witness = takeIndex(
      graph.addStorageWitness(firstAccess, secondAccess), passed,
      "add indexed witness");
  SyncCoverDemand demand = makeDemand(first, second);
  demand.kind = SyncCoverDemandKind::MemoryRAW;
  demand.storageProvenance = SyncCoverStorageProvenance::Complete;
  demand.storageWitnesses = {witness};
  passed &= check(graph.addDemand(demand), "add indexed demand");
  passed &= check(graph.addDemand(makeDemand(first, second)),
                  "add slot-less indexed demand");
  SyncCoverDemand incomplete = makeDemand(first, second);
  incomplete.kind = SyncCoverDemandKind::MemoryRAW;
  incomplete.storageProvenance = SyncCoverStorageProvenance::Incomplete;
  incomplete.storageWitnesses = {witness};
  passed &= check(graph.addDemand(incomplete),
                  "add incomplete-storage indexed demand");

  const SyncCoverCandidateIndex unfrozen(graph);
  passed &= check(unfrozen.getError() ==
                      SyncCoverCandidateIndexError::StructureNotFrozen,
                  "candidate indexes require a frozen structural prefix");
  const std::size_t structuralGeneration = graph.getStructuralGeneration();
  passed &= check(graph.freezeStructure(), "freeze structural graph");
  passed &= check(graph.isStructureFrozen() &&
                      graph.getStructuralEdgeCount() == graph.getEdges().size() &&
                      graph.getStructuralGeneration() ==
                          structuralGeneration + 1,
                  "freeze records the immutable structural edge prefix");
  const SyncCoverCandidateIndex index(graph);
  passed &= check(static_cast<bool>(index),
                  "build candidate index over frozen graph");
  const auto timelines = index.getTimelines();
  passed &= check(timelines && timelines.value->size() == 1,
                  "one pipe timeline contains every structured alternative");
  const auto firstPosition = index.getNodePosition(first);
  const auto secondPosition = index.getNodePosition(second);
  const auto alternativePosition = index.getNodePosition(alternative);
  passed &= check(firstPosition && secondPosition && alternativePosition &&
                      firstPosition.value->timeline ==
                          secondPosition.value->timeline &&
                      alternativePosition.value->timeline ==
                          firstPosition.value->timeline &&
                      firstPosition.value->ordinal == 0 &&
                      secondPosition.value->ordinal == 1 &&
                      alternativePosition.value->ordinal == 2,
                  "index exposes complete ordered pipe timelines");
  const auto accesses = index.getDomainAccesses(domain);
  const auto witnesses = index.getDemandWitnesses(0);
  passed &= check(accesses && accesses.value->size() == 2 && witnesses &&
                      *witnesses.value ==
                          std::vector<SyncCoverStorageWitnessId>{witness},
                  "index exposes deterministic storage lookups");
  const auto overlapGroups = index.getWitnessOverlapGroups();
  passed &= check(overlapGroups && overlapGroups.value->size() == 1 &&
                      overlapGroups.value->front().witnesses ==
                          std::vector<SyncCoverStorageWitnessId>{witness} &&
                      overlapGroups.value->front().demands ==
                          std::vector<std::size_t>{0, 2},
                  "overlap groups retain witness-to-demand provenance");
  const auto opportunities = index.getOpportunities();
  const auto exactOpportunities = index.getDemandOpportunities(0);
  const auto slotlessOpportunities = index.getDemandOpportunities(1);
  const bool exactShape =
      opportunities && opportunities.value->size() == 3 &&
      exactOpportunities && exactOpportunities.value->size() == 1 &&
      opportunities.value->front().demand == 0 &&
      opportunities.value->front().source == first &&
      opportunities.value->front().target == second &&
      opportunities.value->front().sourceResource == 1 &&
      opportunities.value->front().targetResource == 1 &&
      opportunities.value->front().sourcePosition.ordinal == 0 &&
      opportunities.value->front().targetPosition.ordinal == 1 &&
      opportunities.value->front().slot &&
      opportunities.value->front().slot->domain == domain &&
      opportunities.value->front().slot->overlap ==
          SyncCoverStorageInterval{0, 8} &&
      opportunities.value->front().slot->sourceExtent ==
          SyncCoverStorageInterval{0, 8} &&
      opportunities.value->front().slot->targetExtent ==
          SyncCoverStorageInterval{0, 8} &&
      opportunities.value->front().slot->sourceFamily == 1 &&
      opportunities.value->front().slot->targetFamily == 2 &&
      opportunities.value->front().slot->sourceMode ==
          SyncCoverStorageAccessMode::Write &&
      opportunities.value->front().slot->targetMode ==
          SyncCoverStorageAccessMode::Read;
  passed &= check(exactShape,
                  "candidate opportunities retain context, positions, and "
                  "exact physical slots");
  passed &= check(
      slotlessOpportunities && slotlessOpportunities.value->size() == 1 &&
          !(*opportunities.value)[1].slot &&
          (*opportunities.value)[1].kind == SyncCoverDemandKind::SSA,
      "non-memory opportunities remain explicit without invented slots");
  const auto incompleteOpportunities = index.getDemandOpportunities(2);
  passed &= check(
      incompleteOpportunities && incompleteOpportunities.value->size() == 1 &&
          !opportunities.value->back().slot &&
          opportunities.value->back().storageProvenance ==
              SyncCoverStorageProvenance::Incomplete,
      "incomplete memory provenance remains slot-less and fail closed");
  passed &= check(index.getDemandOpportunities(3).error ==
                      SyncCoverCandidateIndexError::InvalidIndex,
                  "opportunity lookup rejects an unknown demand");
  passed &= check(graph.addNode(2, 1, 0, 3).error ==
                      SyncCoverGraphError::StructureFrozen &&
                      graph.addStorageDomain().error ==
                          SyncCoverGraphError::StructureFrozen,
                  "structural mutation is rejected after freezing");
  passed &= check(static_cast<bool>(index),
                  "rejected structural mutations do not stale the index");

  SyncCoverGraph contaminated;
  const SyncCoverNodeId contaminatedSource =
      takeIndex(contaminated.addNode(1, 1, 0, 0), passed,
                "add contaminated source");
  const SyncCoverNodeId contaminatedTarget =
      takeIndex(contaminated.addNode(2, 1, 0, 1), passed,
                "add contaminated target");
  SyncCoverEdge mechanismEdge =
      makeEdge(contaminatedSource, contaminatedTarget);
  mechanismEdge.kind = SyncCoverEdgeKind::CompletionSupply;
  mechanismEdge.mechanism = 7;
  passed &= check(contaminated.addEdge(mechanismEdge),
                  "low-level graph accepts a selectable test edge");
  passed &= check(contaminated.freezeStructure().error ==
                      SyncCoverGraphError::InvalidEdgeOwnership &&
                      !contaminated.isStructureFrozen(),
                  "freeze rejects mechanism edges in the structural prefix");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testZeroDistanceDag();
  passed &= testRecurrenceScopes();
  passed &= testStructuredGuards();
  passed &= testRecurrenceGuardContexts();
  passed &= testInvalidReferencesDoNotMutate();
  passed &= testTimelineAnchorsAndCompletionCapabilities();
  passed &= testScopeQueriesAndGeneration();
  passed &= testStorageProvenanceContract();
  passed &= testCandidateOpportunityRecurrenceContext();
  passed &= testStructuralFreezeAndCandidateIndex();
  return passed ? 0 : 1;
}
