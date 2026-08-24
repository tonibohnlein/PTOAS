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
  passed &=
      check(graph.addEdge(makeEdge(third, first)), "cycle edge is admitted");
  passed &=
      check(graph.validate().error == SyncCoverGraphError::ZeroDistanceCycle,
            "global zero-distance cycle is diagnosed by validate");
  return passed;
}

bool testRecurrenceScopes() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(), passed, "add outer loop");
  const SyncCoverScopeId inner =
      takeIndex(graph.addScope(outer), passed, "add inner loop");
  const SyncCoverScopeId sibling =
      takeIndex(graph.addScope(), passed, "add sibling loop");
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, inner, 0), passed, "add nested node");
  const SyncCoverNodeId second = takeIndex(graph.addNode(2, 1, inner, 1),
                                           passed, "add second nested node");
  const SyncCoverNodeId siblingNode =
      takeIndex(graph.addNode(3, 1, sibling, 0), passed, "add sibling node");
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
      takeIndex(graph.addScope(), passed, "add guard recurrence loop");
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

} // namespace

int main() {
  bool passed = true;
  passed &= testZeroDistanceDag();
  passed &= testRecurrenceScopes();
  passed &= testStructuredGuards();
  passed &= testRecurrenceGuardContexts();
  passed &= testInvalidReferencesDoNotMutate();
  return passed ? 0 : 1;
}
