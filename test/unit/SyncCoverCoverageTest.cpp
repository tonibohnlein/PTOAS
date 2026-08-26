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
#include <type_traits>
#include <vector>

namespace {

using namespace mlir::pto;

class LegacyCoverageOracle : public mlir::pto::SyncCoverCoverageOracle {
public:
  explicit LegacyCoverageOracle(const SyncCoverGraph &graph)
      : SyncCoverCoverageOracle(
            graph, SyncCoverCoverageBackend::LegacyPerContext) {}
};

static_assert(!std::is_copy_constructible<LegacyCoverageOracle>::value,
              "coverage caches must not be aliased by copying an oracle");
static_assert(!std::is_move_constructible<LegacyCoverageOracle>::value,
              "moving an oracle must not obscure cache ownership");

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

bool testSharedExpansionRequiresFrozenStructure() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add mutable source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add mutable target");
  passed &= check(graph.addDemand(makeDemand(source, target)),
                  "add mutable demand");
  mlir::pto::SyncCoverCoverageOracle shared(
      graph, SyncCoverCoverageBackend::SharedExpansion);
  passed &= check(shared.checkDemand(0, {}).error ==
                      SyncCoverCoverageError::InvalidGraph,
                  "shared expansion rejects an unfrozen structure");
  return passed;
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

  LegacyCoverageOracle oracle(graph);
  const SyncCoverCoverageResult missing = oracle.checkDemand(0, {});
  passed &= check(missing && !missing.covered,
                  "unselected completion does not cover demand");
  passed &= check(missing.cutMechanisms == std::vector<SyncCoverMechanismId>{7},
                  "unselected crossing mechanism is reported in the cut");
  const SyncCoverCoverageResult freshMissing =
      LegacyCoverageOracle(graph).checkDemand(0, {});
  passed &= check(missing.error == freshMissing.error &&
                      missing.covered == freshMissing.covered &&
                      missing.reachableStates == freshMissing.reachableStates &&
                      missing.cutMechanisms == freshMissing.cutMechanisms,
                  "cached and fresh uncovered cuts are equivalent");
  const SyncCoverCoverageResult selected = oracle.checkDemand(0, {7, 7});
  passed &=
      check(selected && selected.covered, "selected completion covers demand");
  passed &=
      check(selected.witnessMechanisms == std::vector<SyncCoverMechanismId>{7},
            "coverage returns a deterministic mechanism witness");
  const SyncCoverCoverageResult cached = oracle.checkDemand(0, {7});
  const SyncCoverCoverageResult uncached =
      LegacyCoverageOracle(graph).checkDemand(0, {7});
  passed &= check(cached.error == uncached.error &&
                      cached.covered == uncached.covered &&
                      cached.witnessMechanisms == uncached.witnessMechanisms &&
                      cached.reachableStates == uncached.reachableStates &&
                      cached.cutMechanisms == uncached.cutMechanisms,
                  "cached and fresh demand preparation are equivalent");
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
  passed &= check(
      !LegacyCoverageOracle(preserving).checkDemand(0, {3}).covered,
      "issue order cannot aggregate an uncompleted prefix into a later set");

  SyncCoverGraph carried;
  const SyncCoverNodeId carriedFirst =
      takeIndex(carried.addNode(1, 1, 0, 0), passed, "add carried source");
  const SyncCoverNodeId carriedSecond =
      takeIndex(carried.addNode(2, 1, 0, 1), passed, "add carried middle");
  const SyncCoverNodeId carriedThird =
      takeIndex(carried.addNode(2, 1, 0, 2), passed, "add carried target");
  passed &= check(carried.addDemand(makeDemand(carriedFirst, carriedThird)),
                  "add carried demand");
  passed &=
      check(carried.addEdge(makeEdge(carriedFirst, carriedSecond,
                                     SyncCoverEdgeKind::CompletionSupply, 4)),
            "add completion before issue order");
  passed &= check(carried.addEdge(makeEdge(
                      carriedSecond, carriedThird,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "add completion-carrying issue edge");
  passed &= check(LegacyCoverageOracle(carried).checkDemand(0, {4}).covered,
                  "issue order carries an already established completion fact");

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
  passed &= check(!LegacyCoverageOracle(issuing).checkDemand(0, {3}).covered,
                  "pure issue order cannot transfer completion provenance");
  return passed;
}

bool testRecurrenceCopiesAndGuards() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add recurrence loop");
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
      LegacyCoverageOracle(graph).checkDemand(0, {9});
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
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add conditional loop");
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
  passed &= check(!LegacyCoverageOracle(graph).checkDemand(0, {11}).covered,
                  "conditional intermediate cannot cover outer demand");
  return passed;
}

bool testEndpointImpliedOptionalScopes() {
  bool passed = true;
  SyncCoverGraph targetGuarded;
  const SyncCoverScopeId targetScope =
      takeIndex(targetGuarded.addScope(0, false), passed,
                "add target-implied optional scope");
  const SyncCoverControlId targetControl =
      takeIndex(targetGuarded.addControl(2), passed,
                "add target-implied control");
  const SyncCoverNodeId outerSource =
      takeIndex(targetGuarded.addNode(1, 1, 0, 0), passed,
                "add outer source");
  const SyncCoverNodeId targetMiddle = takeIndex(
      targetGuarded.addNode(2, 1, targetScope, 1, {{{targetControl, 0}}}),
      passed, "add target-implied middle");
  const SyncCoverNodeId guardedTarget = takeIndex(
      targetGuarded.addNode(2, 1, targetScope, 2, {{{targetControl, 0}}}),
      passed, "add guarded target");
  passed &= check(
      targetGuarded.addDemand(makeDemand(outerSource, guardedTarget)),
      "add target-implied demand");
  passed &= check(targetGuarded.addEdge(makeEdge(
                      outerSource, targetMiddle,
                      SyncCoverEdgeKind::CompletionSupply)),
                  "add target-implied completion");
  passed &= check(targetGuarded.addEdge(makeEdge(
                      targetMiddle, guardedTarget,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "add target-implied carry");
  passed &= check(
      LegacyCoverageOracle(targetGuarded).checkDemand(0, {}).covered,
      "target execution makes an intermediate in the same arm available");

  SyncCoverGraph sourceGuarded;
  const SyncCoverScopeId sourceScope =
      takeIndex(sourceGuarded.addScope(0, false), passed,
                "add source-implied optional scope");
  const SyncCoverControlId sourceControl =
      takeIndex(sourceGuarded.addControl(2), passed,
                "add source-implied control");
  const SyncCoverNodeId guardedSource = takeIndex(
      sourceGuarded.addNode(1, 1, sourceScope, 0, {{{sourceControl, 1}}}),
      passed, "add guarded source");
  const SyncCoverNodeId sourceMiddle = takeIndex(
      sourceGuarded.addNode(2, 1, sourceScope, 1, {{{sourceControl, 1}}}),
      passed, "add source-implied middle");
  const SyncCoverNodeId outerTarget =
      takeIndex(sourceGuarded.addNode(2, 1, 0, 2), passed,
                "add outer target");
  passed &= check(
      sourceGuarded.addDemand(makeDemand(guardedSource, outerTarget)),
      "add source-implied demand");
  passed &= check(sourceGuarded.addEdge(makeEdge(
                      guardedSource, sourceMiddle,
                      SyncCoverEdgeKind::CompletionSupply)),
                  "add source-implied completion");
  passed &= check(sourceGuarded.addEdge(makeEdge(
                      sourceMiddle, outerTarget,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "add source-implied carry");
  passed &= check(
      LegacyCoverageOracle(sourceGuarded).checkDemand(0, {}).covered,
      "source execution makes an intermediate in the same arm available");

  SyncCoverGraph nestedLoops;
  const SyncCoverScopeId outerLoop = takeIndex(
      nestedLoops.addScope(0, false, std::nullopt, true), passed,
      "add potentially zero-trip outer loop");
  const SyncCoverScopeId outerBody =
      takeIndex(nestedLoops.addScope(outerLoop, true), passed,
                "add outer loop body");
  const SyncCoverScopeId innerLoop = takeIndex(
      nestedLoops.addScope(outerBody, false, std::nullopt, true), passed,
      "add potentially zero-trip inner loop");
  const SyncCoverScopeId innerBody =
      takeIndex(nestedLoops.addScope(innerLoop, true), passed,
                "add inner loop body");
  const SyncCoverNodeId loopSource =
      takeIndex(nestedLoops.addNode(1, 1, 0, 0), passed,
                "add pre-loop source");
  const SyncCoverNodeId loopMiddle =
      takeIndex(nestedLoops.addNode(2, 1, outerBody, 1), passed,
                "add outer-loop middle");
  const SyncCoverNodeId loopTarget =
      takeIndex(nestedLoops.addNode(2, 1, innerBody, 2), passed,
                "add nested-loop target");
  passed &= check(nestedLoops.addDemand(makeDemand(loopSource, loopTarget)),
                  "add nested-loop demand");
  passed &= check(nestedLoops.addEdge(makeEdge(
                      loopSource, loopMiddle,
                      SyncCoverEdgeKind::CompletionSupply)),
                  "add nested-loop completion");
  passed &= check(nestedLoops.addEdge(makeEdge(
                      loopMiddle, loopTarget,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "add nested-loop carry");
  passed &= check(
      LegacyCoverageOracle(nestedLoops).checkDemand(0, {}).covered,
      "nested-loop target execution implies its enclosing loop scopes");
  return passed;
}

bool testEndpointImpliedScopeDoesNotCrossRecurrenceCopies() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add recurrence loop");
  const SyncCoverScopeId optional =
      takeIndex(graph.addScope(loop, false), passed,
                "add optional recurrence scope");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, optional, 0), passed,
                "add optional recurrence source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, optional, 1), passed,
                "add optional recurrence middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 2), passed,
                "add recurrence target");

  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = loop;
  demand.distance = 2;
  passed &= check(graph.addDemand(demand), "add optional recurrence demand");

  SyncCoverEdge first =
      makeEdge(source, middle, SyncCoverEdgeKind::CompletionSupply, 13);
  first.scope = loop;
  first.distance = 1;
  passed &= check(graph.addEdge(first), "add optional first recurrence hop");
  SyncCoverEdge second = makeEdge(
      middle, target, SyncCoverEdgeKind::CompletionPreservingIssueOrder, 13);
  second.scope = loop;
  second.distance = 1;
  passed &= check(graph.addEdge(second), "add optional second recurrence hop");

  passed &= check(
      !LegacyCoverageOracle(graph).checkDemand(0, {13}).covered,
      "endpoint-implied optional scope is unavailable in another copy");
  return passed;
}

bool testComposedRecurrencePath() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add composed loop");
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
  passed &= check(LegacyCoverageOracle(graph).checkDemand(0, {12}).covered,
                  "d+1 window composes two distance-one edges");
  return passed;
}

bool testExactRecurrenceScopeAndOuterControl() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add outer recurrence scope");
  const SyncCoverScopeId inner =
      takeIndex(graph.addScope(outer, false, std::nullopt, true), passed,
                "add inner recurrence scope");
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
  LegacyCoverageOracle oracle(graph);
  passed &= check(!oracle.checkDemand(0, {13}).covered,
                  "inner recurrence cannot discharge outer recurrence");

  SyncCoverEdge outerEdge =
      makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, 14);
  outerEdge.scope = outer;
  outerEdge.distance = 1;
  passed &= check(graph.addEdge(outerEdge), "add outer recurrence edge");
  passed &= check(!oracle.checkDemand(0, {14}).covered,
                  "an oracle epoch is isolated from later graph mutations");
  passed &= check(LegacyCoverageOracle(graph).checkDemand(0, {14}).covered,
                  "a new epoch observes the outer recurrence edge");
  return passed;
}

bool nestedScopeCoverage(bool mustExecute, bool &passed) {
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, false, std::nullopt, true), passed, "add parent loop");
  const SyncCoverScopeId nested =
      takeIndex(graph.addScope(loop, mustExecute, std::nullopt, true), passed,
                "add nested loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add nested source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, nested, 1), passed, "add nested middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, loop, 2), passed, "add nested target");
  SyncCoverDemand demand = makeDemand(source, target);
  demand.scope = loop;
  passed &= check(graph.addDemand(demand), "add nested-scope demand");
  SyncCoverEdge first =
      makeEdge(source, middle, SyncCoverEdgeKind::CompletionSupply);
  first.scope = loop;
  passed &= check(graph.addEdge(first), "add nested completion entry");
  SyncCoverEdge second = makeEdge(
      middle, target, SyncCoverEdgeKind::CompletionPreservingIssueOrder);
  second.scope = loop;
  passed &= check(graph.addEdge(second), "add nested completion carry");
  return LegacyCoverageOracle(graph).checkDemand(0, {}).covered;
}

bool testNestedExecutionAndExpansionLimit() {
  bool passed = true;
  passed &= check(!nestedScopeCoverage(false, passed),
                  "potentially zero-trip nested scope fails closed");
  passed &= check(nestedScopeCoverage(true, passed),
                  "proven must-execute nested scope may cover demand");

  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, std::nullopt, true), passed,
                "add expansion loop");
  const SyncCoverNodeId node =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add expansion node");
  SyncCoverDemand huge = makeDemand(node, node);
  huge.scope = loop;
  huge.distance = std::numeric_limits<unsigned>::max();
  passed &= check(graph.addDemand(huge), "add bounded-expansion probe");
  passed &= check(LegacyCoverageOracle(graph).checkDemand(0, {}).error ==
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
  LegacyCoverageOracle oracle(graph);
  const SyncCoverCoverageResult covered = oracle.checkDemand(0, {5});
  passed &= check(covered.covered && covered.witnessMechanisms.size() == 1,
                  "one mechanism ID enables an atomic multi-edge bundle");
  passed &= check(oracle.checkDemand(1, {}).error ==
                      SyncCoverCoverageError::InvalidDemand,
                  "invalid demand index fails closed");
  const std::vector<SyncCoverCoverageResult> all = oracle.checkAll({5});
  passed &= check(all.size() == 1 && all[0].covered,
                  "whole-graph coverage reuses prepared demand topology");
  const SyncCoverDemandTopologyResult topology = oracle.getDemandTopology(0);
  passed &=
      check(topology && topology.potentialMechanisms ==
                            std::vector<SyncCoverMechanismId>({5}),
            "demand topology reports selectable source-to-target mechanisms");
  const SyncCoverCoverageStatistics statistics = oracle.getStatistics();
  passed &= check(statistics.graphValidations == 1 &&
                      statistics.demandPreparations == 1 &&
                      statistics.coverageQueries == 2 &&
                      statistics.preparedVirtualNodes == 3 &&
                      statistics.preparedVirtualEdges == 2 &&
                      statistics.maximumVirtualNodes == 3 &&
                      statistics.maximumVirtualEdges == 2,
                  "one oracle epoch validates and prepares each demand once");
  passed &= check(oracle.getDemandTopology(1).error ==
                      SyncCoverCoverageError::InvalidDemand,
                  "invalid topology demand fails closed");
  passed &= check(oracle.checkDemandCanonicalSelection(0, {5, 5}).error ==
                      SyncCoverCoverageError::InvalidSelection,
                  "search fast path rejects noncanonical selections");
  return passed;
}

bool testStorageMetadataDoesNotAffectCoverage() {
  bool passed = true;
  SyncCoverGraph incomplete;
  const SyncCoverNodeId incompleteSource = takeIndex(
      incomplete.addNode(1, 1, 0, 0), passed, "add incomplete source");
  const SyncCoverNodeId incompleteTarget = takeIndex(
      incomplete.addNode(2, 1, 0, 1), passed, "add incomplete target");
  SyncCoverDemand incompleteDemand =
      makeDemand(incompleteSource, incompleteTarget);
  incompleteDemand.kind = SyncCoverDemandKind::MemoryRAW;
  incompleteDemand.storageProvenance =
      SyncCoverStorageProvenance::Incomplete;
  passed &= check(incomplete.addDemand(incompleteDemand),
                  "add incomplete storage demand");
  passed &= check(incomplete.addEdge(makeEdge(
                      incompleteSource, incompleteTarget,
                      SyncCoverEdgeKind::CompletionSupply, 9)),
                  "add incomplete storage completion");

  SyncCoverGraph exact;
  const SyncCoverNodeId exactSource =
      takeIndex(exact.addNode(1, 1, 0, 0), passed, "add exact source");
  const SyncCoverNodeId exactTarget =
      takeIndex(exact.addNode(2, 1, 0, 1), passed, "add exact target");
  const SyncCoverStorageDomainId domain = takeIndex(
      exact.addStorageDomain(), passed, "add exact storage domain");
  const SyncCoverStorageAccessId sourceAccess = takeIndex(
      exact.addStorageAccess(exactSource, domain, 1, {0, 16},
                             SyncCoverStorageAccessMode::Write, 0),
      passed, "add exact write");
  const SyncCoverStorageAccessId targetAccess = takeIndex(
      exact.addStorageAccess(exactTarget, domain, 2, {8, 24},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "add exact read");
  const SyncCoverStorageWitnessId witness = takeIndex(
      exact.addStorageWitness(sourceAccess, targetAccess), passed,
      "add exact storage witness");
  SyncCoverDemand exactDemand = makeDemand(exactSource, exactTarget);
  exactDemand.kind = SyncCoverDemandKind::MemoryRAW;
  exactDemand.storageProvenance = SyncCoverStorageProvenance::Complete;
  exactDemand.storageWitnesses = {witness};
  passed &= check(exact.addDemand(exactDemand), "add exact storage demand");
  passed &= check(exact.addEdge(makeEdge(exactSource, exactTarget,
                                         SyncCoverEdgeKind::CompletionSupply,
                                         9)),
                  "add exact storage completion");

  const SyncCoverCoverageResult incompleteMissing =
      LegacyCoverageOracle(incomplete).checkDemand(0, {});
  const SyncCoverCoverageResult exactMissing =
      LegacyCoverageOracle(exact).checkDemand(0, {});
  const SyncCoverCoverageResult incompleteSelected =
      LegacyCoverageOracle(incomplete).checkDemand(0, {9});
  const SyncCoverCoverageResult exactSelected =
      LegacyCoverageOracle(exact).checkDemand(0, {9});
  passed &= check(incompleteMissing.covered == exactMissing.covered &&
                      incompleteMissing.cutMechanisms ==
                          exactMissing.cutMechanisms &&
                      incompleteSelected.covered == exactSelected.covered &&
                      incompleteSelected.witnessMechanisms ==
                          exactSelected.witnessMechanisms,
                  "storage provenance is invisible to the coverage oracle");
  return passed;
}

} // namespace

bool sameCoverage(const SyncCoverCoverageResult &first,
                  const SyncCoverCoverageResult &second) {
  return first.error == second.error && first.covered == second.covered &&
         first.witnessMechanisms == second.witnessMechanisms &&
         first.reachableStates == second.reachableStates &&
         first.cutMechanisms == second.cutMechanisms;
}

/// The prepared-topology cache is keyed by coverage context, so demands with
/// DIFFERENT endpoints sharing one key reuse a topology first prepared for
/// another demand. Every cached answer must match a fresh oracle regardless
/// of query order.
bool testContextCacheSharesAcrossDemands() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, false, std::nullopt, true), passed, "add loop scope");
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add first source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, loop, 1), passed, "add first target");
  const SyncCoverNodeId secondSource =
      takeIndex(graph.addNode(1, 1, loop, 2), passed, "add second source");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, loop, 3), passed, "add second target");
  // Same scope, distance, and (empty) guards: one context key, four distinct
  // endpoint pairs across forward and distance-one recurrence demands.
  for (unsigned distance : {0U, 1U}) {
    SyncCoverDemand first = makeDemand(firstSource, firstTarget);
    first.scope = loop;
    first.distance = distance;
    SyncCoverDemand second = makeDemand(secondSource, secondTarget);
    second.scope = loop;
    second.distance = distance;
    passed &= check(graph.addDemand(first), "add first demand");
    passed &= check(graph.addDemand(second), "add second demand");
  }
  SyncCoverEdge covering = makeEdge(
      firstSource, firstTarget, SyncCoverEdgeKind::CompletionSupply, 3);
  covering.scope = loop;
  passed &= check(graph.addEdge(covering), "add first covering edge");
  SyncCoverEdge carried = makeEdge(
      secondSource, secondTarget, SyncCoverEdgeKind::CompletionSupply, 4);
  carried.scope = loop;
  carried.distance = 1;
  passed &= check(graph.addEdge(carried), "add carried covering edge");

  const std::vector<SyncCoverMechanismId> selected{3, 4};
  const std::vector<std::vector<SyncCoverDemandId>> orders = {
      {0, 1, 2, 3}, {3, 2, 1, 0}, {1, 3, 0, 2}};
  for (const std::vector<SyncCoverDemandId> &order : orders) {
    LegacyCoverageOracle shared(graph);
    for (SyncCoverDemandId demand : order) {
      const SyncCoverCoverageResult cached =
          shared.checkDemand(demand, selected);
      const SyncCoverCoverageResult fresh =
          LegacyCoverageOracle(graph).checkDemand(demand, selected);
      passed &= check(sameCoverage(cached, fresh),
                      "context-cached coverage matches a fresh oracle for "
                      "every demand order");
      const SyncCoverSingletonWitnessResult cachedWitnesses =
          shared.getSingletonMechanismWitnesses(demand);
      const SyncCoverSingletonWitnessResult freshWitnesses =
          LegacyCoverageOracle(graph).getSingletonMechanismWitnesses(
              demand);
      passed &= check(cachedWitnesses.error == freshWitnesses.error &&
                          cachedWitnesses.mechanisms ==
                              freshWitnesses.mechanisms,
                      "context-cached singleton witnesses match a fresh "
                      "oracle for every demand order");
    }
  }
  return passed;
}

int main() {
  bool passed = true;
  passed &= testSharedExpansionRequiresFrozenStructure();
  passed &= testSelectedCompletionAndCut();
  passed &= testContextCacheSharesAcrossDemands();
  passed &= testCompletionTransitionKinds();
  passed &= testRecurrenceCopiesAndGuards();
  passed &= testIntermediateConditionalFailsClosed();
  passed &= testEndpointImpliedOptionalScopes();
  passed &= testEndpointImpliedScopeDoesNotCrossRecurrenceCopies();
  passed &= testComposedRecurrencePath();
  passed &= testExactRecurrenceScopeAndOuterControl();
  passed &= testNestedExecutionAndExpansionLimit();
  passed &= testAtomicBundleAndErrors();
  passed &= testStorageMetadataDoesNotAffectCoverage();
  return passed ? 0 : 1;
}
