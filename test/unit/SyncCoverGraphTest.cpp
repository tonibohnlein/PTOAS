// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <iostream>
#include <limits>
#include <string_view>

namespace {

using namespace mlir::pto;

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
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
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
                  "backward zero-distance edge is rejected immediately");
  passed &= check(graph.validate(), "rejected edge leaves graph valid");
  return passed;
}

bool testRecurrenceScopes() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true), passed,
      "add outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{0, 20}, true),
      passed, "add inner loop");
  const SyncCoverScopeId sibling = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true), passed,
      "add sibling loop");
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, inner, 0), passed, "add nested node");
  const SyncCoverNodeId second = takeIndex(graph.addNode(2, 1, inner, 1),
                                           passed, "add second nested node");
  const SyncCoverNodeId siblingNode =
      takeIndex(graph.addNode(3, 1, sibling, 2), passed, "add sibling node");
  const SyncCoverNodeId direct =
      takeIndex(graph.addNode(4, 1, outer, 3), passed, "add direct child node");
  const SyncCoverNodeId root =
      takeIndex(graph.addNode(5, 1, 0, 4), passed, "add root node");

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
  passed &= check(graph.addDemand(makeDemand(direct, direct)).error ==
                      SyncCoverGraphError::ZeroDistanceSelfDemand,
                  "zero-distance self-demand is rejected");

  const SyncCoverScopeId nonLoop =
      takeIndex(graph.addScope(), passed, "add non-loop scope");
  const SyncCoverNodeId nonLoopSource = takeIndex(
      graph.addNode(6, 1, nonLoop, 5), passed, "add non-loop source");
  const SyncCoverNodeId nonLoopTarget = takeIndex(
      graph.addNode(7, 1, nonLoop, 6), passed, "add non-loop target");
  SyncCoverEdge invalidRecurrence = makeEdge(nonLoopSource, nonLoopTarget);
  invalidRecurrence.scope = nonLoop;
  invalidRecurrence.distance = 1;
  passed &= check(graph.addEdge(invalidRecurrence).error ==
                      SyncCoverGraphError::InvalidDistance,
                  "non-loop scopes cannot own recurrence edges");
  SyncCoverDemand invalidDemand =
      makeDemand(nonLoopSource, nonLoopTarget);
  invalidDemand.scope = nonLoop;
  invalidDemand.distance = 1;
  passed &= check(graph.addDemand(invalidDemand).error ==
                      SyncCoverGraphError::InvalidDistance,
                  "non-loop scopes cannot own recurrence demands");
  passed &= check(graph.addScope(0, true, std::nullopt, true).error ==
                      SyncCoverGraphError::InvalidTimeline,
                  "loop scopes require explicit timeline boundaries");
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
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 16}, true), passed,
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

bool testScopeGuardsAreInherited() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverControlId outerControl =
      takeIndex(graph.addControl(2, 0), passed, "add outer guard control");
  const SyncCoverGuard outerGuard{{{outerControl, 1}}};
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(0, true, std::nullopt, false, outerGuard),
                passed, "add guarded outer scope");
  passed &= check(graph.addScope(outer, true).error ==
                      SyncCoverGraphError::InvalidGuard,
                  "child scope cannot drop an inherited guard");
  const SyncCoverScopeId child = takeIndex(
      graph.addScope(outer, true, std::nullopt, false, outerGuard), passed,
      "add child with inherited guard");
  passed &= check(graph.addNode(1, 1, child, 0).error ==
                      SyncCoverGraphError::InvalidGuard,
                  "node cannot drop its inherited scope guard");
  passed &= check(graph.addNode(1, 1, child, 0, outerGuard),
                  "node retains inherited scope condition");
  return passed && check(graph.validate(), "validate inherited scope guard");
}

bool testTimelineStorageAndFreezeContract() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{1, 20}, true), passed,
      "add timelined loop");
  const SyncCoverScopeId body =
      takeIndex(graph.addScope(loop, true), passed, "add loop body scope");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, body, 2, {}, {2, 2, 3}), passed,
      "add completion-capable source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, body, 4), passed, "add target");
  passed &= check(syncCoverNodeCanProduceCompletion(graph, source, 2),
                  "completion targets are normalized and queryable");
  passed &= check(graph.scopeMustExecuteWithin(loop, body),
                  "must-execute scope chain is explicit");
  passed &= check(graph.getScopeLoopDepth(body).value_or(0) == 1,
                  "loop depth follows the scope tree");

  const SyncCoverStorageDomainId storage =
      takeIndex(graph.addStorageDomain(), passed, "add storage domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(source, storage, 1, {0, 64},
                             SyncCoverStorageAccessMode::Write, 0, true),
      passed, "add write access");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(target, storage, 2, {32, 96},
                             SyncCoverStorageAccessMode::Read, 1, true),
      passed, "add read access");
  passed &= check(graph.getStorageAccesses()[write].exactPhysical &&
                      graph.getStorageAccesses()[read].exactPhysical,
                  "exact physical storage provenance is retained");
  const SyncCoverStorageWitnessId witness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add overlap witness");
  passed &= check(graph.getStorageWitnesses()[witness].overlap.begin == 32 &&
                      graph.getStorageWitnesses()[witness].overlap.end == 64,
                  "storage witness records the exact overlap");
  const SyncCoverStorageAccessId secondWrite = takeIndex(
      graph.addStorageAccess(source, storage, 3, {100, 160},
                             SyncCoverStorageAccessMode::Write, 2),
      passed, "add second write access");
  passed &= check(!graph.getStorageAccesses()[secondWrite].exactPhysical,
                  "conservative storage provenance is the default");
  const SyncCoverStorageAccessId secondRead = takeIndex(
      graph.addStorageAccess(target, storage, 4, {120, 180},
                             SyncCoverStorageAccessMode::Read, 3),
      passed, "add second read access");
  const SyncCoverStorageWitnessId secondWitness = takeIndex(
      graph.addStorageWitness(secondWrite, secondRead), passed,
      "add second overlap witness");

  SyncCoverDemand raw = makeDemand(source, target);
  raw.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  raw.storageWitnesses = {witness, witness};
  const SyncCoverGraphResult rawResult = graph.addDemand(raw);
  passed &= check(rawResult && rawResult.index &&
                      graph.getDemands().size() > *rawResult.index &&
                      graph.getDemands()[*rawResult.index]
                              .storageWitnesses.size() == 1,
                  "memory demand witness IDs are canonicalized");
  raw.storageWitnesses = {secondWitness};
  const SyncCoverGraphResult mergedRawResult = graph.addDemand(raw);
  passed &= check(mergedRawResult && rawResult &&
                      mergedRawResult.index == rawResult.index &&
                      graph.getDemands()[*rawResult.index]
                              .storageWitnesses ==
                          std::vector<SyncCoverStorageWitnessId>{witness,
                                                                 secondWitness},
                  "canonical demand merges witnesses across insertions");

  passed &= check(resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::BeforeNode, source}) == 4,
                  "before-node anchor uses doubled timeline coordinates");
  passed &= check(resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::AfterNode, source}) == 5,
                  "after-node anchor follows before-node anchor");
  passed &= check(resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::ScopeEntry, 0, loop}) == 1,
                  "scope entry resolves to the explicit loop boundary");
  passed &= check(resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::ScopeExit, 0, loop}) == 20,
                  "scope exit resolves to the explicit loop boundary");
  passed &= check(resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::ScopeEntry, 0, body}) == 4,
                  "structured scope entry resolves from its first node");
  passed &= check(resolveSyncCoverAnchor(
                      graph, {SyncCoverAnchorKind::ScopeExit, 0, body}) == 9,
                  "structured scope exit resolves from its last node");
  passed &= check(resolveSyncCoverAnchor(
                      graph,
                      {SyncCoverAnchorKind::TimelinePoint, 0, body, 10}) == 10,
                  "timeline point resolves through the nearest timeline");
  passed &= check(!resolveSyncCoverAnchor(
                       graph,
                       {SyncCoverAnchorKind::TimelinePoint, 0, body, 21}),
                  "out-of-range timeline point fails closed");
  passed &= check(graph.addNode(3, 1, body, 10).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "node after-anchor cannot escape its timeline");

  passed &= check(graph.freezeStructure(), "freeze validates the graph");
  passed &= check(graph.addNode(3, 1, body, 6).error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects structural mutation");
  passed &= check(graph.addScope().error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects scope mutation");
  passed &= check(graph.addControl(2).error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects control mutation");
  passed &= check(graph.addEdge(makeEdge(source, target)).error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects edge mutation");
  passed &= check(graph.addDemand(makeDemand(source, target)).error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects demand mutation");
  passed &= check(graph.addStorageDomain().error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects storage-domain mutation");
  passed &= check(graph.addStorageAccess(source, storage, 3, {0, 1},
                                         SyncCoverStorageAccessMode::Read)
                          .error == SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects storage-access mutation");
  passed &= check(graph.addStorageWitness(write, read).error ==
                      SyncCoverGraphError::StructureFrozen,
                  "frozen graph rejects storage-witness mutation");
  return passed;
}

bool testCanonicalRowsAndInvalidKinds() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add canonical source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(1, 1, 0, 1), passed, "add canonical target");
  const SyncCoverNodeId crossResource = takeIndex(
      graph.addNode(2, 1, 0, 2), passed, "add cross-resource target");

  SyncCoverEdge edge = makeEdge(source, target);
  edge.kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  const SyncCoverGraphResult firstEdge = graph.addEdge(edge);
  passed &= check(firstEdge, "add canonical edge");
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  const SyncCoverGraphResult strongerEdge = graph.addEdge(edge);
  passed &= check(strongerEdge && strongerEdge.index == firstEdge.index,
                  "duplicate edge returns its canonical row");
  passed &= check(graph.getEdges().size() == 1 &&
                      graph.getEdges()[0].kind ==
                          SyncCoverEdgeKind::CompletionSupply,
                  "duplicate edge retains the strongest semantics");
  edge.kind = SyncCoverEdgeKind::CompletionPreservingIssueOrder;
  passed &= check(graph.addEdge(edge).index == firstEdge.index &&
                      graph.getEdges()[0].kind ==
                          SyncCoverEdgeKind::CompletionSupply,
                  "weaker edge cannot demote canonical edge semantics");
  edge.kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  passed &= check(graph.addEdge(edge).index == firstEdge.index &&
                      graph.getEdges()[0].kind ==
                          SyncCoverEdgeKind::CompletionSupply,
                  "strongest edge merge is insertion-order stable");
  for (SyncCoverEdgeKind kind :
       {SyncCoverEdgeKind::CompletionPreservingIssueOrder,
        SyncCoverEdgeKind::NonCompletionPreservingIssueOrder}) {
    SyncCoverEdge crossIssue = makeEdge(source, crossResource);
    crossIssue.kind = kind;
    passed &= check(graph.addEdge(crossIssue).error ==
                        SyncCoverGraphError::InvalidEdgeKind,
                    "issue-order edge cannot cross physical issue streams");
  }

  const SyncCoverGraphResult firstDemand =
      graph.addDemand(makeDemand(source, target));
  const SyncCoverGraphResult duplicateDemand =
      graph.addDemand(makeDemand(source, target));
  passed &= check(firstDemand && duplicateDemand &&
                      duplicateDemand.index == firstDemand.index &&
                      graph.getDemands().size() == 1,
                  "duplicate demands share one canonical row");

  SyncCoverEdge invalidEdge = makeEdge(source, target);
  invalidEdge.kind = static_cast<SyncCoverEdgeKind>(255);
  passed &= check(graph.addEdge(invalidEdge).error ==
                      SyncCoverGraphError::InvalidEdgeKind,
                  "invalid edge enum fails closed");
  SyncCoverDemand invalidDemand = makeDemand(source, target);
  invalidDemand.provenanceKinds = {static_cast<SyncCoverDemandKind>(255)};
  passed &= check(graph.addDemand(invalidDemand).error ==
                      SyncCoverGraphError::InvalidDemandKind,
                  "invalid demand enum fails closed");
  return passed;
}

bool testStorageProvenanceFailures() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add storage source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add storage target");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add storage provenance domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(source, domain, 1, {0, 16},
                             SyncCoverStorageAccessMode::Write),
      passed, "add provenance write");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(target, domain, 2, {8, 24},
                             SyncCoverStorageAccessMode::Read),
      passed, "add provenance read");
  const SyncCoverStorageWitnessId witness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add provenance witness");

  SyncCoverDemand invalidWitness = makeDemand(source, target);
  invalidWitness.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  invalidWitness.storageWitnesses = {999};
  passed &= check(graph.addDemand(invalidWitness).error ==
                      SyncCoverGraphError::InvalidStorageWitness,
                  "invalid demand witness ID fails closed");

  SyncCoverDemand missingWitness = makeDemand(source, target);
  missingWitness.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  passed &= check(graph.addDemand(missingWitness).error ==
                      SyncCoverGraphError::InvalidStorageProvenance,
                  "memory demand without a witness fails closed");
  SyncCoverDemand wrongHazard = missingWitness;
  wrongHazard.provenanceKinds = {SyncCoverDemandKind::MemoryWAR};
  wrongHazard.storageWitnesses = {witness};
  passed &= check(graph.addDemand(wrongHazard).error ==
                      SyncCoverGraphError::InvalidStorageProvenance,
                  "witness access roles must match the hazard kind");
  SyncCoverDemand ssaWithWitness = makeDemand(source, target);
  ssaWithWitness.storageWitnesses = {witness};
  passed &= check(graph.addDemand(ssaWithWitness).error ==
                      SyncCoverGraphError::InvalidStorageProvenance,
                  "SSA demand cannot claim a storage witness");
  passed &= check(graph.addStorageAccess(
                      source, domain, 3, {0, 1},
                      static_cast<SyncCoverStorageAccessMode>(255))
                          .error == SyncCoverGraphError::InvalidStorageAccess,
                  "invalid storage access mode fails closed");

  const SyncCoverNodeId unrelated =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add unrelated node");
  const SyncCoverStorageAccessId unrelatedWrite = takeIndex(
      graph.addStorageAccess(unrelated, domain, 4, {0, 16},
                             SyncCoverStorageAccessMode::Write),
      passed, "add unrelated write");
  const SyncCoverStorageWitnessId wrongEndpointWitness = takeIndex(
      graph.addStorageWitness(unrelatedWrite, read), passed,
      "add wrong-endpoint witness");
  SyncCoverDemand wrongEndpoint = missingWitness;
  wrongEndpoint.storageWitnesses = {wrongEndpointWitness};
  passed &= check(graph.addDemand(wrongEndpoint).error ==
                      SyncCoverGraphError::InvalidStorageProvenance,
                  "witness endpoints must match demand endpoints");

  const SyncCoverStorageDomainId secondDomain = takeIndex(
      graph.addStorageDomain(), passed, "add second storage domain");
  const SyncCoverStorageAccessId crossDomain = takeIndex(
      graph.addStorageAccess(target, secondDomain, 5, {0, 16},
                             SyncCoverStorageAccessMode::Read),
      passed, "add cross-domain access");
  passed &= check(graph.addStorageWitness(write, crossDomain).error ==
                      SyncCoverGraphError::InvalidStorageWitness,
                  "cross-domain witness fails closed");
  const SyncCoverStorageAccessId disjoint = takeIndex(
      graph.addStorageAccess(target, domain, 6, {32, 48},
                             SyncCoverStorageAccessMode::Read),
      passed, "add disjoint access");
  passed &= check(graph.addStorageWitness(write, disjoint).error ==
                      SyncCoverGraphError::InvalidStorageWitness,
                  "non-overlap witness fails closed");

  const SyncCoverStorageAccessId sourceRead = takeIndex(
      graph.addStorageAccess(source, domain, 7, {64, 96},
                             SyncCoverStorageAccessMode::Read),
      passed, "add WAR source read");
  const SyncCoverStorageAccessId targetWrite = takeIndex(
      graph.addStorageAccess(target, domain, 8, {80, 112},
                             SyncCoverStorageAccessMode::Write),
      passed, "add WAR target write");
  const SyncCoverStorageWitnessId warWitness = takeIndex(
      graph.addStorageWitness(sourceRead, targetWrite), passed,
      "add WAR witness");
  SyncCoverDemand war = makeDemand(source, target);
  war.provenanceKinds = {SyncCoverDemandKind::MemoryWAR};
  war.storageWitnesses = {warWitness};
  passed &= check(graph.addDemand(war), "valid WAR provenance is accepted");

  const SyncCoverStorageAccessId sourceWrite = takeIndex(
      graph.addStorageAccess(source, domain, 9, {128, 160},
                             SyncCoverStorageAccessMode::Write),
      passed, "add WAW source write");
  const SyncCoverStorageAccessId secondTargetWrite = takeIndex(
      graph.addStorageAccess(target, domain, 10, {144, 176},
                             SyncCoverStorageAccessMode::Write),
      passed, "add WAW target write");
  const SyncCoverStorageWitnessId wawWitness = takeIndex(
      graph.addStorageWitness(sourceWrite, secondTargetWrite), passed,
      "add WAW witness");
  SyncCoverDemand waw = makeDemand(source, target);
  waw.provenanceKinds = {SyncCoverDemandKind::MemoryWAW};
  waw.storageWitnesses = {wawWitness};
  passed &= check(graph.addDemand(waw), "valid WAW provenance is accepted");
  passed &= check(graph.getDemands().size() == 1 &&
                      graph.getDemands()[0].provenanceKinds ==
                          std::vector<SyncCoverDemandKind>{
                              SyncCoverDemandKind::MemoryWAR,
                              SyncCoverDemandKind::MemoryWAW},
                  "equal WAR and WAW obligations share one canonical row");
  return passed;
}

bool testTimelineBoundariesAndOverflow() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 12}), passed,
      "add outer timeline");
  passed &= check(graph.addScope(outer, true,
                                 SyncCoverTimelineInterval{1, 10})
                          .error == SyncCoverGraphError::InvalidTimeline,
                  "nested timeline must fit nearest timelined ancestor");
  passed &= check(graph.addNode(1, 1, outer, 1),
                  "node anchors may exactly fit timeline entry");
  passed &= check(graph.addNode(2, 1, outer, 6).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "node after-anchor must fit timeline exit");

  SyncCoverGraph overflow;
  passed &= check(overflow
                          .addNode(1, 1, 0,
                                   std::numeric_limits<std::size_t>::max())
                          .error == SyncCoverGraphError::InvalidOrder,
                  "node anchor arithmetic fails closed on overflow");
  passed &= check(!resolveSyncCoverAnchor(
                       graph, {SyncCoverAnchorKind::TimelinePoint, 0, 999, 2}),
                  "timeline point rejects an invalid scope");
  return passed;
}

bool testPeriodicControlEvidence() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true), passed,
      "add periodic loop");
  const SyncCoverControlId control =
      takeIndex(graph.addControl(2, loop), passed, "add periodic control");
  SyncCoverControlPhaseRelation relation;
  relation.loopScope = loop;
  relation.initialPhase = 0;
  relation.nextPhase = {1, 0};
  relation.activeAlternative = {0, 1};
  passed &= check(graph.setControlPhaseRelation(control, relation),
                  "attach proven periodic relation");
  SyncCoverControlPhaseRelation invalid = relation;
  invalid.nextPhase[1] = 2;
  passed &= check(graph.setControlPhaseRelation(control, invalid).error ==
                      SyncCoverGraphError::InvalidControl,
                  "reject out-of-range successor phase");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(loop, true, SyncCoverTimelineInterval{2, 18}, true),
      passed, "add nested periodic loop");
  const SyncCoverControlId innerControl = takeIndex(
      graph.addControl(2, inner), passed, "add nested periodic control");
  passed &= check(graph.setControlPhaseRelation(innerControl, relation).error ==
                      SyncCoverGraphError::InvalidControl,
                  "phase evidence must name the nearest loop");
  relation.loopScope = inner;
  passed &= check(graph.setControlPhaseRelation(innerControl, relation),
                  "accept nearest-loop phase evidence");
  passed &= check(graph.freezeStructure(), "freeze periodic graph");
  const std::optional<SyncCoverControlPhaseRelation> &stored =
      graph.getControls()[control].phaseRelation;
  return check(stored.has_value(), "retain periodic relation") &&
         check(stored->nextPhase == std::vector<std::size_t>({1, 0}),
               "failed update leaves periodic relation unchanged") &&
         check(static_cast<bool>(graph.validate()),
               "validate periodic control evidence");
}

bool testCompletionDominanceContract() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first = takeIndex(
      graph.addNode(1, 1, 0, 0), passed, "add first completion node");
  const SyncCoverNodeId second = takeIndex(
      graph.addNode(1, 1, 0, 1), passed, "add second completion node");
  const SyncCoverNodeId third = takeIndex(
      graph.addNode(1, 1, 0, 2), passed, "add third completion node");
  const SyncCoverNodeId other = takeIndex(
      graph.addNode(2, 1, 0, 3), passed, "add other-resource node");
  passed &= check(graph.addCompletionDominance(first, second),
                  "record direct completion dominance");
  passed &= check(graph.addCompletionDominance(second, third),
                  "record transitive completion dominance");
  passed &= check(graph.completionDominates(third, first) &&
                      graph.completionDominates(second, first) &&
                      !graph.completionDominates(first, second),
                  "query completion dominance transitively");
  passed &= check(graph.addCompletionDominance(first, other).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "completion dominance cannot cross issue resources");
  passed &= check(graph.addCompletionDominance(third, second).error ==
                      SyncCoverGraphError::InvalidOrder,
                  "completion dominance cannot point backward");
  passed &= check(graph.freezeStructure(), "freeze completion graph");
  return passed && check(graph.validate(), "validate completion dominance");
}

} // namespace

int main() {
  bool passed = true;
  passed &= testZeroDistanceDag();
  passed &= testRecurrenceScopes();
  passed &= testStructuredGuards();
  passed &= testRecurrenceGuardContexts();
  passed &= testInvalidReferencesDoNotMutate();
  passed &= testScopeGuardsAreInherited();
  passed &= testTimelineStorageAndFreezeContract();
  passed &= testCanonicalRowsAndInvalidKinds();
  passed &= testStorageProvenanceFailures();
  passed &= testTimelineBoundariesAndOverflow();
  passed &= testPeriodicControlEvidence();
  passed &= testCompletionDominanceContract();
  return passed ? 0 : 1;
}
