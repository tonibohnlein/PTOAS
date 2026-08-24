// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverCoverageTest failure: " << message << '\n';
  }
  return condition;
}

bool check(const SyncCoverGraphResult &result, std::string_view message) {
  return check(static_cast<bool>(result), message);
}

std::size_t takeIndex(const SyncCoverGraphResult &result, bool &passed,
                      std::string_view message) {
  passed &=
      check(static_cast<bool>(result) && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverEdge makeEdge(SyncCoverNodeId source, SyncCoverNodeId target,
                       SyncCoverEdgeKind kind,
                       std::optional<SyncCoverMechanismId> mechanism = {}) {
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  edge.kind = kind;
  edge.mechanism = mechanism;
  return edge;
}

SyncCoverDemand makeDemand(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  return demand;
}

bool testSelectedCompletionAndCut() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add target");
  passed &= check(graph.addDemand(makeDemand(source, target)), "add demand");
  passed &= check(graph.addEdge(makeEdge(
                      source, target, SyncCoverEdgeKind::CompletionSupply, 7)),
                  "add selectable completion edge");

  SyncCoverCoverageOracle oracle(graph);
  const SyncCoverCoverageResult missing = oracle.checkDemand(0, {});
  passed &= check(missing && !missing.covered,
                  "unselected completion does not cover demand");
  passed &= check(missing.cutMechanisms == std::vector<SyncCoverMechanismId>{7},
                  "unselected crossing mechanism is reported in the cut");
  const SyncCoverCoverageResult selected = oracle.checkDemand(0, {7, 7});
  passed &=
      check(selected && selected.covered, "selected completion covers demand");
  passed &=
      check(selected.witnessMechanisms == std::vector<SyncCoverMechanismId>{7},
            "coverage returns a deterministic mechanism witness");
  return passed;
}

bool testCompletionTransitionKinds() {
  bool passed = true;
  SyncCoverGraph preserving;
  const SyncCoverNodeId first =
      takeIndex(preserving.addNode(1, 1, 0, 0), passed, "add first node");
  const SyncCoverNodeId second =
      takeIndex(preserving.addNode(1, 1, 0, 1), passed, "add second node");
  const SyncCoverNodeId third =
      takeIndex(preserving.addNode(2, 1, 0, 2), passed, "add third node");
  passed &= check(preserving.addDemand(makeDemand(first, third)),
                  "add transitive demand");
  passed &= check(
      preserving.addEdge(makeEdge(
          first, second, SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
      "add completion-preserving issue edge");
  passed &= check(preserving.addEdge(makeEdge(
                      second, third, SyncCoverEdgeKind::CompletionSupply, 3)),
                  "add completion supply");
  passed &=
      check(SyncCoverCoverageOracle(preserving).checkDemand(0, {3}).covered,
            "preserving issue path can reach a later completion supply");

  SyncCoverGraph issuing;
  const SyncCoverNodeId issueFirst =
      takeIndex(issuing.addNode(1, 1, 0, 0), passed, "add issue source");
  const SyncCoverNodeId issueSecond =
      takeIndex(issuing.addNode(1, 1, 0, 1), passed, "add issue middle");
  const SyncCoverNodeId issueThird =
      takeIndex(issuing.addNode(2, 1, 0, 2), passed, "add issue target");
  passed &= check(issuing.addDemand(makeDemand(issueFirst, issueThird)),
                  "add issue-only demand");
  passed &= check(issuing.addEdge(makeEdge(
                      issueFirst, issueSecond,
                      SyncCoverEdgeKind::NonCompletionPreservingIssueOrder)),
                  "add pure issue edge");
  passed &=
      check(issuing.addEdge(makeEdge(issueSecond, issueThird,
                                     SyncCoverEdgeKind::CompletionSupply, 3)),
            "add downstream completion supply");
  passed &= check(!SyncCoverCoverageOracle(issuing).checkDemand(0, {3}).covered,
                  "pure issue order cannot transfer completion provenance");
  return passed;
}

bool testRecurrenceCopiesAndGuards() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(), passed, "add recurrence loop");
  const SyncCoverControlId branch =
      takeIndex(graph.addControl(2, loop), passed, "add loop-local branch");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0, {{{branch, 0}}}), passed,
                "add guarded recurrence source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 1, {{{branch, 1}}}), passed,
                "add guarded recurrence target");
  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = loop;
  demand.distance = 2;
  passed &= check(graph.addDemand(demand), "add distance-two demand");
  SyncCoverEdge completion =
      makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, 9);
  completion.scope = loop;
  completion.distance = 2;
  passed &= check(graph.addEdge(completion), "add distance-two completion");

  const SyncCoverCoverageResult result =
      SyncCoverCoverageOracle(graph).checkDemand(0, {9});
  passed &= check(result.covered,
                  "copy-contextualized endpoint guards cover recurrence");
  passed &= check(result.reachableStates.back().copy == 2,
                  "oracle materializes only the required virtual window");
  return passed;
}

bool testIntermediateConditionalFailsClosed() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(), passed, "add conditional loop");
  const SyncCoverControlId branch =
      takeIndex(graph.addControl(2, loop), passed, "add conditional control");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add plain source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, loop, 1, {{{branch, 0}}}), passed,
                "add conditional middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, loop, 2), passed, "add plain target");
  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = loop;
  demand.distance = 2;
  passed &= check(graph.addDemand(demand), "add unconditional recurrence");

  SyncCoverEdge first = makeEdge(
      source, middle, SyncCoverEdgeKind::CompletionPreservingIssueOrder);
  first.scope = loop;
  first.distance = 1;
  passed &= check(graph.addEdge(first), "add guarded first half");
  SyncCoverEdge second =
      makeEdge(middle, target, SyncCoverEdgeKind::CompletionSupply, 11);
  second.scope = loop;
  second.distance = 1;
  passed &= check(graph.addEdge(second), "add guarded second half");
  passed &= check(!SyncCoverCoverageOracle(graph).checkDemand(0, {11}).covered,
                  "conditional intermediate cannot cover outer demand");
  return passed;
}

bool testComposedRecurrencePath() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(), passed, "add composed loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add composed source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, loop, 1), passed, "add composed middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, loop, 2), passed, "add composed target");
  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = loop;
  demand.distance = 2;
  passed &= check(graph.addDemand(demand), "add composed demand");

  SyncCoverEdge first =
      makeEdge(source, middle, SyncCoverEdgeKind::CompletionSupply, 12);
  first.scope = loop;
  first.distance = 1;
  passed &= check(graph.addEdge(first), "add first recurrence hop");
  SyncCoverEdge second = makeEdge(
      middle, target, SyncCoverEdgeKind::CompletionPreservingIssueOrder, 12);
  second.scope = loop;
  second.distance = 1;
  passed &= check(graph.addEdge(second), "add second recurrence hop");
  passed &= check(SyncCoverCoverageOracle(graph).checkDemand(0, {12}).covered,
                  "d+1 window composes two distance-one edges");
  return passed;
}

bool testExactRecurrenceScopeAndOuterControl() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(), passed, "add outer recurrence scope");
  const SyncCoverScopeId inner =
      takeIndex(graph.addScope(outer), passed, "add inner recurrence scope");
  const SyncCoverControlId outerControl =
      takeIndex(graph.addControl(2), passed, "add outer control");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, inner, 0, {{{outerControl, 1}}}), passed,
                "add controlled source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, inner, 1, {{{outerControl, 1}}}), passed,
                "add controlled target");
  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = outer;
  demand.distance = 1;
  passed &= check(graph.addDemand(demand), "add outer recurrence demand");

  SyncCoverEdge innerEdge =
      makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, 13);
  innerEdge.scope = inner;
  innerEdge.distance = 1;
  passed &= check(graph.addEdge(innerEdge), "add inner recurrence edge");
  SyncCoverCoverageOracle oracle(graph);
  passed &= check(!oracle.checkDemand(0, {13}).covered,
                  "inner recurrence cannot discharge outer recurrence");

  SyncCoverEdge outerEdge =
      makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, 14);
  outerEdge.scope = outer;
  outerEdge.distance = 1;
  passed &= check(graph.addEdge(outerEdge), "add outer recurrence edge");
  passed &= check(oracle.checkDemand(0, {14}).covered,
                  "shared outer control remains valid across recurrence");
  return passed;
}

bool nestedScopeCoverage(bool mustExecute, bool &passed) {
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(), passed, "add parent loop");
  const SyncCoverScopeId nested =
      takeIndex(graph.addScope(loop, mustExecute), passed, "add nested loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add nested source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, nested, 1), passed, "add nested middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, loop, 2), passed, "add nested target");
  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = loop;
  passed &= check(graph.addDemand(demand), "add nested-scope demand");
  SyncCoverEdge first = makeEdge(
      source, middle, SyncCoverEdgeKind::CompletionPreservingIssueOrder);
  first.scope = loop;
  passed &= check(graph.addEdge(first), "add nested path entry");
  SyncCoverEdge second =
      makeEdge(middle, target, SyncCoverEdgeKind::CompletionSupply);
  second.scope = loop;
  passed &= check(graph.addEdge(second), "add nested path exit");
  return SyncCoverCoverageOracle(graph).checkDemand(0, {}).covered;
}

bool testNestedExecutionAndExpansionLimit() {
  bool passed = true;
  passed &= check(!nestedScopeCoverage(false, passed),
                  "potentially zero-trip nested scope fails closed");
  passed &= check(nestedScopeCoverage(true, passed),
                  "proven must-execute nested scope may cover demand");

  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(), passed, "add expansion loop");
  const SyncCoverNodeId node =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add expansion node");
  SyncCoverDemand huge = makeDemand(node, node);
  huge.scope = loop;
  huge.distance = std::numeric_limits<unsigned>::max();
  passed &= check(graph.addDemand(huge), "add bounded-expansion probe");
  passed &= check(SyncCoverCoverageOracle(graph).checkDemand(0, {}).error ==
                      SyncCoverCoverageError::ExpansionLimitExceeded,
                  "oversized virtual expansion fails before allocation");
  return passed;
}

bool testAtomicBundleAndErrors() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add bundle source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add bundle middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add bundle target");
  passed &=
      check(graph.addDemand(makeDemand(first, target)), "add bundle demand");
  passed &= check(graph.addEdge(makeEdge(
                      first, middle, SyncCoverEdgeKind::CompletionSupply, 5)),
                  "add first bundle edge");
  passed &= check(graph.addEdge(makeEdge(
                      middle, target,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder, 5)),
                  "add second bundle edge");
  SyncCoverCoverageOracle oracle(graph);
  const SyncCoverCoverageResult covered = oracle.checkDemand(0, {5});
  passed &= check(covered.covered && covered.witnessMechanisms.size() == 1,
                  "one mechanism ID enables an atomic multi-edge bundle");
  passed &= check(oracle.checkDemand(1, {}).error ==
                      SyncCoverCoverageError::InvalidDemand,
                  "invalid demand index fails closed");
  passed &=
      check(oracle.checkAll({5}).size() == 1 && oracle.checkAll({5})[0].covered,
            "whole-graph coverage reuses one graph validation");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testSelectedCompletionAndCut();
  passed &= testCompletionTransitionKinds();
  passed &= testRecurrenceCopiesAndGuards();
  passed &= testIntermediateConditionalFailsClosed();
  passed &= testComposedRecurrencePath();
  passed &= testExactRecurrenceScopeAndOuterControl();
  passed &= testNestedExecutionAndExpansionLimit();
  passed &= testAtomicBundleAndErrors();
  return passed ? 0 : 1;
}
