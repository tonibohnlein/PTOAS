// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverProtocol.h"

#include <algorithm>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

constexpr SyncCoverOrderingRequirementMask kCompletion =
    syncCoverOrderingRequirementBit(
        SyncCoverOrderingRequirement::PipelineCompletionBeforeAccess);

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverProtocolTest failure: " << message << '\n';
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

SyncCoverProtocolTargetContract target() {
  return {{{1, 2, kCompletion}, {2, 1, kCompletion}}, {0, 1, 2, 3, 4, 5}, {}};
}

SyncCoverCutPoint point(SyncCoverCutPointKind kind, std::uint32_t resource,
                        SyncCoverAnchorKind anchor, SyncCoverNodeId node,
                        std::size_t ordinal = 0) {
  return {kind, resource, {anchor, node, 0, 0}, {}, ordinal};
}

struct ProtocolGraph {
  SyncCoverGraph graph;
  SyncCoverRegionId root = 0;
  SyncCoverRegionId body = 0;
  SyncCoverScopeId loop = 0;
  SyncCoverNodeId producer = 0;
  SyncCoverNodeId consumer = 0;
  std::optional<SyncCoverControlId> phaseControl;
  std::vector<SyncCoverScopeId> phaseScopes;
  std::vector<SyncCoverRegionId> phaseSequences;
};

ProtocolGraph makeProtocolGraph(bool loop, bool &passed) {
  ProtocolGraph result;
  result.root =
      takeIndex(result.graph.addRegion(0, SyncCoverRegionKind::Sequence,
                                       SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add protocol root sequence");
  if (loop) {
    result.loop = takeIndex(
        result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 16}, true,
                              {}, result.root),
        passed, "add protocol loop");
    const SyncCoverRegionId loopRegion =
        takeIndex(result.graph.addRegion(result.root, SyncCoverRegionKind::Loop,
                                         SyncCoverRegionCardinality::ZeroOrMore,
                                         result.loop),
                  passed, "add protocol loop region");
    passed &= check(result.graph.setScopeRegion(result.loop, loopRegion),
                    "bind protocol loop region");
    result.body =
        takeIndex(result.graph.addRegion(
                      loopRegion, SyncCoverRegionKind::Sequence,
                      SyncCoverRegionCardinality::ExactlyOnce, result.loop),
                  passed, "add protocol loop body");
  } else {
    result.body = result.root;
  }
  result.producer = takeIndex(
      result.graph.addNode(1, 1, result.loop, 0, {}, {}, std::nullopt, false,
                           std::numeric_limits<std::size_t>::max(), -1, {},
                           result.body),
      passed, "add protocol producer");
  result.consumer = takeIndex(
      result.graph.addNode(2, 1, result.loop, 1, {}, {}, std::nullopt, false,
                           std::numeric_limits<std::size_t>::max(), -1, {},
                           result.body),
      passed, "add protocol consumer");
  return result;
}

void addModuloPhase(ProtocolGraph &fixture, unsigned phaseCount, bool &passed,
                    bool bindRelation = true) {
  const SyncCoverControlId control =
      takeIndex(fixture.graph.addControl(phaseCount, fixture.loop), passed,
                "add authoritative protocol phase control");
  const SyncCoverRegionId choice = takeIndex(
      fixture.graph.addRegion(fixture.body, SyncCoverRegionKind::Choice,
                              SyncCoverRegionCardinality::ExactlyOnce,
                              fixture.loop, {}, control),
      passed, "add protocol phase choice");
  passed &= check(fixture.graph.setControlRegion(control, choice),
                  "bind protocol phase choice");
  for (unsigned alternative = 0; alternative < phaseCount; ++alternative) {
    SyncCoverGuard guard{{{control, alternative}}};
    const SyncCoverScopeId scope =
        takeIndex(fixture.graph.addScope(fixture.loop, true, std::nullopt,
                                         false, guard, choice),
                  passed, "add protocol phase alternative scope");
    const SyncCoverRegionId region = takeIndex(
        fixture.graph.addRegion(choice, SyncCoverRegionKind::Alternative,
                                SyncCoverRegionCardinality::ZeroOrOne, scope,
                                guard, control, alternative),
        passed, "add protocol phase alternative");
    passed &= check(fixture.graph.setScopeRegion(scope, region),
                    "bind protocol phase alternative");
    const SyncCoverRegionId sequence =
        takeIndex(fixture.graph.addRegion(
                      region, SyncCoverRegionKind::Sequence,
                      SyncCoverRegionCardinality::ExactlyOnce, scope, guard),
                  passed, "add protocol phase alternative sequence");
    fixture.phaseScopes.push_back(scope);
    fixture.phaseSequences.push_back(sequence);
  }
  if (bindRelation) {
    std::vector<std::size_t> nextPhase(phaseCount);
    std::vector<unsigned> activeAlternative(phaseCount);
    for (unsigned phase = 0; phase < phaseCount; ++phase) {
      nextPhase[phase] = (phase + 1) % phaseCount;
      activeAlternative[phase] = phase;
    }
    passed &= check(fixture.graph.setControlPhaseRelation(
                        control, {fixture.loop, 0, std::move(nextPhase),
                                  std::move(activeAlternative)}),
                    "bind authoritative protocol phase relation");
  }
  fixture.phaseControl = control;
}

void addAlternatingPhase(ProtocolGraph &fixture, bool &passed,
                         bool bindRelation = true) {
  addModuloPhase(fixture, 2, passed, bindRelation);
}

SyncCoverEventChannel readyChannel(const ProtocolGraph &fixture,
                                   std::size_t width = 1) {
  return {0,
          fixture.loop ? SyncCoverEventChannelFlow::SameIteration
                       : SyncCoverEventChannelFlow::SingleShot,
          point(SyncCoverCutPointKind::EventSet, 1,
                SyncCoverAnchorKind::AfterNode, fixture.producer),
          point(SyncCoverCutPointKind::EventWait, 2,
                SyncCoverAnchorKind::BeforeNode, fixture.consumer),
          width,
          0,
          kCompletion,
          {},
          false};
}

SyncCoverEventChannel releaseChannel(const ProtocolGraph &fixture,
                                     std::size_t width = 1) {
  return {1,
          SyncCoverEventChannelFlow::LoopCarry,
          point(SyncCoverCutPointKind::EventSet, 2,
                SyncCoverAnchorKind::AfterNode, fixture.consumer),
          point(SyncCoverCutPointKind::EventWait, 1,
                SyncCoverAnchorKind::BeforeNode, fixture.producer),
          width,
          static_cast<unsigned>(width),
          kCompletion,
          {},
          true};
}

SyncCoverEventProtocol roundTrip(const ProtocolGraph &fixture,
                                 std::size_t width = 1) {
  SyncCoverProtocolLoopSchedule schedule;
  schedule.scope = fixture.loop;
  schedule.mayExecuteZeroTimes = true;
  schedule.laneByPhase.resize(width);
  if (width > 1) {
    schedule.phaseControl = fixture.phaseControl;
  }
  for (std::size_t phase = 0; phase < width; ++phase) {
    schedule.laneByPhase[phase] = phase;
  }
  return {7,
          width == 1 ? SyncCoverEventProtocolKind::RoundTrip
                     : SyncCoverEventProtocolKind::RotatingLanes,
          schedule,
          {readyChannel(fixture, width), releaseChannel(fixture, width)},
          {}};
}

bool testSingleShotAndTargetContract() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(false, passed);
  passed &= check(fixture.graph.freezeStructure(), "freeze single-shot graph");
  SyncCoverEventProtocol protocol;
  protocol.mechanism = 3;
  protocol.kind = SyncCoverEventProtocolKind::SingleShot;
  protocol.channels = {readyChannel(fixture)};
  passed &= check(static_cast<bool>(verifySyncCoverEventProtocol(
                      fixture.graph, target(), protocol)),
                  "legal single-shot protocol is accepted");

  SyncCoverProtocolTargetContract invalidIds = target();
  invalidIds.compilerUsableEventIds.push_back(6);
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, invalidIds, protocol).error ==
          SyncCoverProtocolError::InvalidTargetContract,
      "event ID 6 is rejected by the compiler-owned contract");
  SyncCoverProtocolTargetContract wrongPair{
      {{2, 1, kCompletion}}, {0, 1, 2, 3, 4, 5}, {}};
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, wrongPair, protocol).error ==
          SyncCoverProtocolError::InvalidProtocol,
      "unsupported directed HardEvent is rejected");
  return passed;
}

bool testRoundTripAndRotatingLifecycle() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(true, passed);
  passed &= check(fixture.graph.freezeStructure(), "freeze round-trip graph");
  const SyncCoverProtocolVerificationResult single =
      verifySyncCoverEventProtocol(fixture.graph, target(), roundTrip(fixture));
  passed &= check(single && single.statistics.tripCountsChecked > 2,
                  "round trip verifies zero, one, and multiple trips");
  passed &= check(single.exitExports.size() == 1 &&
                      !single.exitExports.front().availableOnZeroTrip &&
                      single.exitExports.front().availableOnNonzeroTrip,
                  "loop exit export distinguishes zero and nonzero trips");
  SyncCoverProtocolLimits exactLaneInitialization;
  exactLaneInitialization.maximumLaneInitializationWork =
      single.statistics.laneInitializationWork;
  passed &= check(static_cast<bool>(verifySyncCoverEventProtocol(
                      fixture.graph, target(), roundTrip(fixture),
                      exactLaneInitialization)),
                  "an exact lane-initialization work limit succeeds");
  exactLaneInitialization.maximumLaneInitializationWork =
      single.statistics.laneInitializationWork - 1;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, target(),
                                               roundTrip(fixture),
                                               exactLaneInitialization)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "a one-less lane-initialization work limit fails closed");
  ProtocolGraph rotatingFixture = makeProtocolGraph(true, passed);
  addAlternatingPhase(rotatingFixture, passed);
  const SyncCoverNodeId rotatingAfterLoop = takeIndex(
      rotatingFixture.graph.addNode(1, 1, 0, 9, {}, {}, std::nullopt, false,
                                    std::numeric_limits<std::size_t>::max(), -1,
                                    {}, rotatingFixture.root),
      passed, "add rotating post-loop target");
  SyncCoverDemand rotatingExport;
  rotatingExport.source = rotatingFixture.consumer;
  rotatingExport.target = rotatingAfterLoop;
  passed &= check(rotatingFixture.graph.addDemand(rotatingExport),
                  "add rotating exit-export demand");
  passed &=
      check(rotatingFixture.graph.freezeStructure(), "freeze rotating graph");
  const SyncCoverEventProtocol rotating = roundTrip(rotatingFixture, 2);
  const SyncCoverProtocolVerificationResult rotatingVerification =
      verifySyncCoverEventProtocol(rotatingFixture.graph, target(), rotating);
  passed &= check(rotatingVerification &&
                      rotatingVerification.exitExports.size() == 1,
                  "two-lane rotating lifecycle exports its drained lanes");
  const SyncCoverProtocolCoverageResult rotatingCoverage =
      computeSyncCoverProtocolExactWorlds(rotatingFixture.graph, target(),
                                          {rotating}, {{{7}}});
  passed &= check(rotatingCoverage &&
                      rotatingCoverage.coveredByWorld.front().contains(0),
                  "multi-lane draining exports completion to the parent");
  SyncCoverProtocolLimits exactExitLimits;
  exactExitLimits.maximumExitExports = 1;
  passed &=
      check(static_cast<bool>(verifySyncCoverEventProtocol(
                rotatingFixture.graph, target(), rotating, exactExitLimits)),
            "an exact exit-export limit succeeds");
  exactExitLimits.maximumExitExports = 0;
  passed &= check(verifySyncCoverEventProtocol(rotatingFixture.graph, target(),
                                               rotating, exactExitLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "a one-less exit-export limit fails closed");

  SyncCoverEventProtocol unsafe = roundTrip(fixture);
  unsafe.kind = SyncCoverEventProtocolKind::ProvenNoOverlap;
  unsafe.channels.resize(1);
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, target(), unsafe).error ==
          SyncCoverProtocolError::InvalidProtocol,
      "one-way loop event without rearm evidence is rejected");

  SyncCoverEventProtocol certified;
  certified.mechanism = 9;
  certified.kind = SyncCoverEventProtocolKind::ProvenNoOverlap;
  certified.loop = roundTrip(fixture).loop;
  certified.channels = {readyChannel(fixture)};
  certified.rearmProofs = {{0, 0, 1, 41}};
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, target(), certified).error ==
          SyncCoverProtocolError::InvalidProtocol,
      "unregistered rearm evidence is rejected");
  SyncCoverProtocolTargetContract certifiedTarget = target();
  certifiedTarget.certifiedRearmFacts = {
      {41, certified.channels[0].wait.resource,
       certified.channels[0].wait.anchor, certified.channels[0].wait.guard,
       certified.channels[0].set.resource, certified.channels[0].set.anchor,
       certified.channels[0].set.guard, fixture.loop, 1, 1}};
  passed &= check(static_cast<bool>(verifySyncCoverEventProtocol(
                      fixture.graph, certifiedTarget, certified)),
                  "versioned target evidence can certify no overlap");
  const SyncCoverProtocolVerificationResult certifiedResult =
      verifySyncCoverEventProtocol(fixture.graph, certifiedTarget, certified);
  SyncCoverProtocolLimits exactProofLimits;
  exactProofLimits.maximumTargetRearmFacts = 1;
  exactProofLimits.maximumRearmProofLaneIncidences =
      certifiedResult.statistics.rearmProofLaneIncidences;
  exactProofLimits.maximumRearmLookupWork =
      certifiedResult.statistics.rearmLookupWork;
  passed &=
      check(static_cast<bool>(verifySyncCoverEventProtocol(
                fixture.graph, certifiedTarget, certified, exactProofLimits)),
            "exact rearm-table and lane-incidence limits succeed");
  SyncCoverProtocolLimits shortProofLimits = exactProofLimits;
  shortProofLimits.maximumRearmProofLaneIncidences =
      certifiedResult.statistics.rearmProofLaneIncidences - 1;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, certifiedTarget,
                                               certified, shortProofLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "one-less rearm lane-incidence limit fails closed");
  shortProofLimits = exactProofLimits;
  shortProofLimits.maximumRearmLookupWork =
      certifiedResult.statistics.rearmLookupWork - 1;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, certifiedTarget,
                                               certified, shortProofLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "one-less rearm lookup-work limit fails closed");
  shortProofLimits = exactProofLimits;
  shortProofLimits.maximumTargetRearmFacts = 0;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, certifiedTarget,
                                               certified, shortProofLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "one-less target rearm-table limit fails closed");
  SyncCoverEventProtocol wrongDistance = certified;
  wrongDistance.rearmProofs[0].iterationDistance = 2;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, certifiedTarget,
                                               wrongDistance)
                          .error == SyncCoverProtocolError::InvalidProtocol,
                  "rearm evidence cannot certify another iteration distance");
  SyncCoverProtocolTargetContract wrongGuard = certifiedTarget;
  wrongGuard.certifiedRearmFacts[0].fromWaitGuard.literals = {{0, 0}};
  passed &=
      check(verifySyncCoverEventProtocol(fixture.graph, wrongGuard, certified)
                    .error == SyncCoverProtocolError::InvalidProtocol,
            "rearm evidence for another guard cannot be reused");
  return passed;
}

bool testGuardAndPhaseQualification() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(true, passed);
  addAlternatingPhase(fixture, passed);
  const SyncCoverControlId control = *fixture.phaseControl;
  SyncCoverDemand first;
  first.source = fixture.producer;
  first.target = fixture.consumer;
  first.scope = fixture.loop;
  first.sourceGuard.literals = {{control, 0}};
  first.targetGuard.literals = {{control, 0}};
  passed &= check(fixture.graph.addDemand(first), "add phase-zero demand");
  SyncCoverDemand second = first;
  second.sourceGuard.literals = {{control, 1}};
  second.targetGuard.literals = {{control, 1}};
  passed &= check(fixture.graph.addDemand(second), "add phase-one demand");
  SyncCoverDemand alternating;
  alternating.source = fixture.consumer;
  alternating.target = fixture.producer;
  alternating.scope = fixture.loop;
  alternating.distance = 1;
  alternating.sourceGuard.literals = {{control, 0}};
  alternating.targetGuard.literals = {{control, 1}};
  passed &= check(fixture.graph.addDemand(alternating),
                  "add alternating recurrence demand");
  passed &= check(fixture.graph.freezeStructure(), "freeze phase graph");

  SyncCoverEventProtocol phased = roundTrip(fixture);
  const SyncCoverProtocolCoverageResult coverage =
      computeSyncCoverProtocolExactWorlds(fixture.graph, target(), {phased},
                                          {{{7}}});
  passed &= check(coverage && coverage.coveredByWorld.front().count() == 3,
                  "copy-qualified endpoint guards preserve alternating rows");

  SyncCoverEventProtocol invented = phased;
  invented.channels[0].activePhases = {0};
  invented.channels[0].set.guard.literals = {{control, 0}};
  invented.channels[0].wait.guard.literals = {{control, 0}};
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, target(), invented).error ==
          SyncCoverProtocolError::InvalidProtocol,
      "a caller cannot invent a phase guard at an unguarded anchor");
  return passed;
}

bool testDirectProtocolRectangles() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(true, passed);
  SyncCoverDemand ready;
  ready.source = fixture.producer;
  ready.target = fixture.consumer;
  ready.scope = fixture.loop;
  passed &= check(fixture.graph.addDemand(ready), "add ready demand");
  SyncCoverDemand release;
  release.source = fixture.consumer;
  release.target = fixture.producer;
  release.scope = fixture.loop;
  release.distance = 1;
  passed &= check(fixture.graph.addDemand(release), "add release demand");
  passed &= check(fixture.graph.freezeStructure(), "freeze rectangle graph");
  const SyncCoverProtocolCoverageResult coverage =
      computeSyncCoverProtocolExactWorlds(fixture.graph, target(),
                                          {roundTrip(fixture)}, {{{7}}});
  passed &= check(coverage && coverage.coveredByWorld.size() == 1 &&
                      coverage.coveredByWorld.front().count() == 2,
                  "one verified round trip covers ready and release rows");
  SyncCoverProtocolLimits exactTransitionLimit;
  exactTransitionLimit.maximumCoverageTransitions =
      coverage.statistics.coverageTransitions;
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      fixture.graph, target(), {roundTrip(fixture)}, {{{7}}},
                      exactTransitionLimit)),
                  "an exact coverage-transition limit succeeds");
  exactTransitionLimit.maximumCoverageTransitions =
      coverage.statistics.coverageTransitions - 1;
  passed &= check(computeSyncCoverProtocolExactWorlds(
                      fixture.graph, target(), {roundTrip(fixture)}, {{{7}}},
                      exactTransitionLimit)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "a one-less coverage-transition limit fails closed");
  SyncCoverCoverageWorkBudget measuredBudget;
  const SyncCoverProtocolCoverageResult measuredCoverage =
      computeSyncCoverProtocolExactWorlds(fixture.graph, target(),
                                          {roundTrip(fixture)}, {{{7}}}, {},
                                          &measuredBudget);
  passed &=
      check(static_cast<bool>(measuredCoverage) && measuredBudget.workUnits > 0,
            "coverage reports nonzero bounded work");
  SyncCoverCoverageWorkBudget exactBudget(measuredBudget.workUnits);
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      fixture.graph, target(), {roundTrip(fixture)}, {{{7}}},
                      {}, &exactBudget)),
                  "an exact end-to-end coverage work budget succeeds");
  SyncCoverCoverageWorkBudget shortBudget(measuredBudget.workUnits - 1);
  passed &= check(computeSyncCoverProtocolExactWorlds(fixture.graph, target(),
                                                      {roundTrip(fixture)},
                                                      {{{7}}}, {}, &shortBudget)
                          .error == SyncCoverProtocolError::WorkLimitExceeded,
                  "a one-less coverage work budget fails closed");
  return passed;
}

bool testWholeWorldCompositionAndExitExport() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add composed source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add composed middle");
  const SyncCoverNodeId sink =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add composed sink");
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = sink;
  passed &= check(graph.addDemand(demand), "add composed protocol demand");
  passed &= check(graph.freezeStructure(), "freeze composed graph");

  SyncCoverEventProtocol first;
  first.mechanism = 10;
  first.kind = SyncCoverEventProtocolKind::SingleShot;
  first.channels = {{0,
                     SyncCoverEventChannelFlow::SingleShot,
                     point(SyncCoverCutPointKind::EventSet, 1,
                           SyncCoverAnchorKind::AfterNode, source),
                     point(SyncCoverCutPointKind::EventWait, 2,
                           SyncCoverAnchorKind::BeforeNode, middle),
                     1,
                     0,
                     kCompletion,
                     {},
                     false}};
  SyncCoverEventProtocol second;
  second.mechanism = 11;
  second.kind = SyncCoverEventProtocolKind::SingleShot;
  second.channels = {{0,
                      SyncCoverEventChannelFlow::SingleShot,
                      point(SyncCoverCutPointKind::EventSet, 2,
                            SyncCoverAnchorKind::AfterNode, middle),
                      point(SyncCoverCutPointKind::EventWait, 3,
                            SyncCoverAnchorKind::BeforeNode, sink),
                      1,
                      0,
                      kCompletion,
                      {},
                      false}};
  SyncCoverProtocolTargetContract composedTarget = target();
  composedTarget.eventCapabilities.push_back({2, 3, kCompletion});
  std::sort(composedTarget.eventCapabilities.begin(),
            composedTarget.eventCapabilities.end());
  const SyncCoverProtocolCoverageResult composed =
      computeSyncCoverProtocolExactWorlds(
          graph, composedTarget, {first, second}, {{{10}}, {{11}}, {{10, 11}}});
  passed &= check(composed && !composed.coveredByWorld[0].contains(0) &&
                      !composed.coveredByWorld[1].contains(0) &&
                      composed.coveredByWorld[2].contains(0),
                  "whole-world closure composes two protocol cuts");
  SyncCoverCoverageWorkBudget worldMeasurement;
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      graph, composedTarget, {first, second},
                      {{{10}}, {{11}}, {{10, 11}}}, {}, &worldMeasurement)),
                  "multi-world catalog reports bounded work");
  SyncCoverCoverageWorkBudget exactWorldBudget(worldMeasurement.workUnits);
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      graph, composedTarget, {first, second},
                      {{{10}}, {{11}}, {{10, 11}}}, {}, &exactWorldBudget)),
                  "an exact multi-world catalog budget succeeds");
  SyncCoverCoverageWorkBudget shortWorldBudget(worldMeasurement.workUnits - 1);
  passed &= check(computeSyncCoverProtocolExactWorlds(
                      graph, composedTarget, {first, second},
                      {{{10}}, {{11}}, {{10, 11}}}, {}, &shortWorldBudget)
                          .error == SyncCoverProtocolError::WorkLimitExceeded,
                  "a one-less multi-world catalog budget fails closed");
  SyncCoverProtocolLimits exactAutomatonLimits;
  exactAutomatonLimits.maximumAutomatonEdges =
      composed.statistics.automatonEdges;
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      graph, composedTarget, {first, second}, {{{10, 11}}},
                      exactAutomatonLimits)),
                  "an exact catalog-wide automaton-edge limit succeeds");
  exactAutomatonLimits.maximumAutomatonEdges =
      composed.statistics.automatonEdges - 1;
  passed &= check(computeSyncCoverProtocolExactWorlds(
                      graph, composedTarget, {first, second}, {{{10, 11}}},
                      exactAutomatonLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "a one-less catalog automaton-edge limit fails closed");
  const SyncCoverProtocolVerificationResult firstVerification =
      verifySyncCoverEventProtocol(graph, composedTarget, first);
  const SyncCoverProtocolVerificationResult secondVerification =
      verifySyncCoverEventProtocol(graph, composedTarget, second);
  SyncCoverProtocolLimits exactAllocationEdges;
  exactAllocationEdges.maximumAutomatonEdges =
      firstVerification.statistics.automatonEdges +
      secondVerification.statistics.automatonEdges;
  passed &= check(
      static_cast<bool>(allocateSyncCoverProtocolEventIds(
          graph, composedTarget, {first, second}, {}, exactAllocationEdges)),
      "allocation honors an exact aggregate automaton-edge limit");
  exactAllocationEdges.maximumAutomatonEdges -= 1;
  passed &= check(allocateSyncCoverProtocolEventIds(graph, composedTarget,
                                                    {first, second}, {},
                                                    exactAllocationEdges)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "allocation rejects a one-less aggregate edge limit");

  ProtocolGraph loopFixture = makeProtocolGraph(true, passed);
  const SyncCoverNodeId afterLoop = takeIndex(
      loopFixture.graph.addNode(1, 1, 0, 9, {}, {}, std::nullopt, false,
                                std::numeric_limits<std::size_t>::max(), -1, {},
                                loopFixture.root),
      passed, "add post-loop export target");
  SyncCoverDemand exported;
  exported.source = loopFixture.consumer;
  exported.target = afterLoop;
  passed &= check(loopFixture.graph.addDemand(exported),
                  "add child-to-parent export demand");
  passed &=
      check(loopFixture.graph.freezeStructure(), "freeze exit-export graph");
  const SyncCoverProtocolCoverageResult exitCoverage =
      computeSyncCoverProtocolExactWorlds(loopFixture.graph, target(),
                                          {roundTrip(loopFixture)}, {{{7}}});
  return passed &&
         check(exitCoverage && exitCoverage.coveredByWorld[0].contains(0),
               "verified loop drain exports completion to its parent");
}

bool testStructuredMustAndFixedSupplyCoverage() {
  bool passed = true;
  SyncCoverGraph fixed;
  const SyncCoverNodeId fixedSource =
      takeIndex(fixed.addNode(1, 1, 0, 0), passed, "add fixed source");
  const SyncCoverNodeId fixedTarget =
      takeIndex(fixed.addNode(2, 1, 0, 1), passed, "add fixed target");
  passed &= check(fixed.addEdge({fixedSource, fixedTarget,
                                 SyncCoverEdgeKind::CompletionSupply}),
                  "add baseline fixed completion supply");
  passed &= check(fixed.addDemand({fixedSource, fixedTarget}),
                  "add fixed-supply demand");
  passed &= check(fixed.freezeStructure(), "freeze fixed-supply graph");
  const SyncCoverProtocolCoverageResult fixedCoverage =
      computeSyncCoverProtocolExactWorlds(fixed, target(), {}, {{{}}});
  passed &=
      check(fixedCoverage && fixedCoverage.coveredByWorld.front().contains(0),
            "an empty protocol world retains fixed completion supplies");

  SyncCoverGraph optional;
  const SyncCoverRegionId root =
      takeIndex(optional.addRegion(0, SyncCoverRegionKind::Sequence,
                                   SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add optional root");
  const SyncCoverRegionId child =
      takeIndex(optional.addRegion(root, SyncCoverRegionKind::Transparent,
                                   SyncCoverRegionCardinality::ZeroOrOne),
                passed, "add unguarded optional child");
  const SyncCoverNodeId source = takeIndex(
      optional.addNode(1, 1, 0, 0, {}, {}, std::nullopt, false,
                       std::numeric_limits<std::size_t>::max(), -1, {}, root),
      passed, "add optional source");
  const SyncCoverNodeId middle = takeIndex(
      optional.addNode(2, 1, 0, 1, {}, {}, std::nullopt, false,
                       std::numeric_limits<std::size_t>::max(), -1, {}, child),
      passed, "add optional middle");
  const SyncCoverNodeId sink = takeIndex(
      optional.addNode(2, 1, 0, 2, {}, {}, std::nullopt, false,
                       std::numeric_limits<std::size_t>::max(), -1, {}, root),
      passed, "add optional sink");
  passed &= check(
      optional.addEdge({source, middle, SyncCoverEdgeKind::CompletionSupply}),
      "add optional completion supply");
  passed &= check(
      optional.addEdge(
          {middle, sink, SyncCoverEdgeKind::NonCompletionPreservingIssueOrder}),
      "add optional issue edge");
  passed &=
      check(optional.addDemand({source, sink}), "add optional-path demand");
  passed &= check(optional.freezeStructure(), "freeze optional graph");
  const SyncCoverProtocolCoverageResult optionalCoverage =
      computeSyncCoverProtocolExactWorlds(optional, target(), {}, {{{}}});
  passed &= check(optionalCoverage &&
                      !optionalCoverage.coveredByWorld.front().contains(0),
                  "an unguarded optional child contributes no must fact");

  SyncCoverGraph choice;
  const SyncCoverRegionId choiceRoot =
      takeIndex(choice.addRegion(0, SyncCoverRegionKind::Sequence,
                                 SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add choice root");
  const SyncCoverNodeId choiceSource =
      takeIndex(choice.addNode(1, 1, 0, 0, {}, {}, std::nullopt, false,
                               std::numeric_limits<std::size_t>::max(), -1, {},
                               choiceRoot),
                passed, "add choice source");
  const SyncCoverControlId control =
      takeIndex(choice.addControl(2, 0), passed, "add choice control");
  const SyncCoverRegionId choiceRegion = takeIndex(
      choice.addRegion(choiceRoot, SyncCoverRegionKind::Choice,
                       SyncCoverRegionCardinality::ExactlyOnce, 0, {}, control),
      passed, "add choice region");
  passed &= check(choice.setControlRegion(control, choiceRegion),
                  "bind choice region");
  SyncCoverNodeId choiceMiddle = 0;
  for (unsigned alternative = 0; alternative < 2; ++alternative) {
    SyncCoverGuard guard{{{control, alternative}}};
    const SyncCoverScopeId scope = takeIndex(
        choice.addScope(0, false, std::nullopt, false, guard, choiceRegion),
        passed, "add choice alternative scope");
    const SyncCoverRegionId alternativeRegion = takeIndex(
        choice.addRegion(choiceRegion, SyncCoverRegionKind::Alternative,
                         SyncCoverRegionCardinality::ZeroOrOne, scope, guard,
                         control, alternative),
        passed, "add choice alternative");
    passed &= check(choice.setScopeRegion(scope, alternativeRegion),
                    "bind choice alternative");
    const SyncCoverRegionId sequence = takeIndex(
        choice.addRegion(alternativeRegion, SyncCoverRegionKind::Sequence,
                         SyncCoverRegionCardinality::ExactlyOnce, scope, guard),
        passed, "add choice alternative sequence");
    if (alternative == 0) {
      choiceMiddle = takeIndex(
          choice.addNode(2, 1, scope, 1, guard, {}, std::nullopt, false,
                         std::numeric_limits<std::size_t>::max(), -1, {},
                         sequence),
          passed, "add choice middle");
    }
  }
  const SyncCoverNodeId choiceSink =
      takeIndex(choice.addNode(2, 1, 0, 2, {}, {}, std::nullopt, false,
                               std::numeric_limits<std::size_t>::max(), -1, {},
                               choiceRoot),
                passed, "add choice sink");
  passed &= check(choice.addEdge({choiceSource, choiceMiddle,
                                  SyncCoverEdgeKind::CompletionSupply}),
                  "add choice completion supply");
  passed &= check(
      choice.addEdge({choiceMiddle, choiceSink,
                      SyncCoverEdgeKind::NonCompletionPreservingIssueOrder}),
      "add choice issue edge");
  passed &= check(choice.addDemand({choiceSource, choiceSink}),
                  "add unspecialized choice demand");
  SyncCoverDemand specializedChoice;
  specializedChoice.source = choiceSource;
  specializedChoice.target = choiceSink;
  specializedChoice.sourceGuard.literals = {{control, 0}};
  specializedChoice.targetGuard.literals = {{control, 0}};
  passed &= check(choice.addDemand(specializedChoice),
                  "add alternative-specialized choice demand");
  passed &= check(choice.freezeStructure(), "freeze choice graph");
  const SyncCoverProtocolCoverageResult choiceCoverage =
      computeSyncCoverProtocolExactWorlds(choice, target(), {}, {{{}}});
  passed &= check(choiceCoverage &&
                      !choiceCoverage.coveredByWorld.front().contains(0) &&
                      choiceCoverage.coveredByWorld.front().contains(1),
                  "choice facts are must only for a proven alternative");

  SyncCoverGraph zeroTrip;
  const SyncCoverRegionId zeroRoot =
      takeIndex(zeroTrip.addRegion(0, SyncCoverRegionKind::Sequence,
                                   SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add zero-trip root");
  const SyncCoverNodeId beforeLoop =
      takeIndex(zeroTrip.addNode(1, 1, 0, 0, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, zeroRoot),
                passed, "add zero-trip source");
  const SyncCoverScopeId zeroLoop =
      takeIndex(zeroTrip.addScope(0, true, SyncCoverTimelineInterval{2, 3},
                                  true, {}, zeroRoot),
                passed, "add zero-trip loop");
  const SyncCoverRegionId zeroLoopRegion = takeIndex(
      zeroTrip.addRegion(zeroRoot, SyncCoverRegionKind::Loop,
                         SyncCoverRegionCardinality::ZeroOrMore, zeroLoop),
      passed, "add zero-trip loop region");
  passed &= check(zeroTrip.setScopeRegion(zeroLoop, zeroLoopRegion),
                  "bind zero-trip loop region");
  const SyncCoverRegionId zeroBody = takeIndex(
      zeroTrip.addRegion(zeroLoopRegion, SyncCoverRegionKind::Sequence,
                         SyncCoverRegionCardinality::ExactlyOnce, zeroLoop),
      passed, "add zero-trip body");
  const SyncCoverNodeId inLoop =
      takeIndex(zeroTrip.addNode(2, 1, zeroLoop, 1, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, zeroBody),
                passed, "add zero-trip body node");
  const SyncCoverNodeId afterLoop =
      takeIndex(zeroTrip.addNode(2, 1, 0, 2, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, zeroRoot),
                passed, "add zero-trip target");
  passed &= check(zeroTrip.addEdge({beforeLoop, inLoop,
                                    SyncCoverEdgeKind::CompletionSupply}),
                  "add zero-trip completion supply");
  passed &= check(
      zeroTrip.addEdge({inLoop, afterLoop,
                        SyncCoverEdgeKind::NonCompletionPreservingIssueOrder}),
      "add zero-trip issue edge");
  passed &= check(zeroTrip.addDemand({beforeLoop, afterLoop}),
                  "add zero-trip demand");
  passed &= check(zeroTrip.freezeStructure(), "freeze zero-trip graph");
  const SyncCoverProtocolCoverageResult zeroTripCoverage =
      computeSyncCoverProtocolExactWorlds(zeroTrip, target(), {}, {{{}}});
  return passed &&
         check(zeroTripCoverage &&
                   !zeroTripCoverage.coveredByWorld.front().contains(0),
               "a zero-trip loop contributes no nonzero-only completion");
}

bool testNestedLoopDistancesAreScopeTyped() {
  bool passed = true;
  ProtocolGraph fixture;
  fixture.root = takeIndex(
      fixture.graph.addRegion(0, SyncCoverRegionKind::Sequence,
                              SyncCoverRegionCardinality::ExactlyOnce),
      passed, "add nested root");
  const SyncCoverScopeId outer = takeIndex(
      fixture.graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true,
                             {}, fixture.root),
      passed, "add outer loop");
  const SyncCoverRegionId outerLoop = takeIndex(
      fixture.graph.addRegion(fixture.root, SyncCoverRegionKind::Loop,
                              SyncCoverRegionCardinality::OneOrMore, outer),
      passed, "add outer loop region");
  passed &= check(fixture.graph.setScopeRegion(outer, outerLoop),
                  "bind outer loop region");
  const SyncCoverRegionId outerBody = takeIndex(
      fixture.graph.addRegion(outerLoop, SyncCoverRegionKind::Sequence,
                              SyncCoverRegionCardinality::ExactlyOnce, outer),
      passed, "add outer body");
  fixture.loop = takeIndex(
      fixture.graph.addScope(outer, true, SyncCoverTimelineInterval{2, 16},
                             true, {}, outerBody),
      passed, "add inner loop");
  const SyncCoverRegionId innerLoop =
      takeIndex(fixture.graph.addRegion(outerBody, SyncCoverRegionKind::Loop,
                                        SyncCoverRegionCardinality::ZeroOrMore,
                                        fixture.loop),
                passed, "add inner loop region");
  passed &= check(fixture.graph.setScopeRegion(fixture.loop, innerLoop),
                  "bind inner loop region");
  fixture.body =
      takeIndex(fixture.graph.addRegion(
                    innerLoop, SyncCoverRegionKind::Sequence,
                    SyncCoverRegionCardinality::ExactlyOnce, fixture.loop),
                passed, "add inner body");
  fixture.producer = takeIndex(
      fixture.graph.addNode(1, 1, fixture.loop, 2, {}, {}, std::nullopt, false,
                            std::numeric_limits<std::size_t>::max(), -1, {},
                            fixture.body),
      passed, "add inner producer");
  fixture.consumer = takeIndex(
      fixture.graph.addNode(2, 1, fixture.loop, 3, {}, {}, std::nullopt, false,
                            std::numeric_limits<std::size_t>::max(), -1, {},
                            fixture.body),
      passed, "add inner consumer");
  SyncCoverDemand outerCarry;
  outerCarry.source = fixture.consumer;
  outerCarry.target = fixture.producer;
  outerCarry.scope = outer;
  outerCarry.distance = 1;
  passed &= check(fixture.graph.addDemand(outerCarry),
                  "add outer recurrence over inner endpoints");
  passed &= check(fixture.graph.freezeStructure(), "freeze nested graph");
  const SyncCoverProtocolCoverageResult coverage =
      computeSyncCoverProtocolExactWorlds(fixture.graph, target(),
                                          {roundTrip(fixture)}, {{{7}}});
  return passed &&
         check(coverage && !coverage.coveredByWorld.front().contains(0),
               "an inner-loop carry cannot satisfy an outer-loop distance");
}

bool testAuthoritativePhaseAndCardinality() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(true, passed);
  addAlternatingPhase(fixture, passed, false);
  const SyncCoverControlId control = *fixture.phaseControl;
  passed &= check(fixture.graph.freezeStructure(),
                  "freeze nondeterministic phase graph");
  SyncCoverEventProtocol invented = roundTrip(fixture);
  invented.loop->phaseControl = control;
  invented.loop->laneByPhase = {0, 0};
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, target(), invented).error ==
          SyncCoverProtocolError::InvalidProtocol,
      "caller-invented phase schedule is rejected");

  SyncCoverEventProtocol single;
  single.mechanism = 13;
  single.kind = SyncCoverEventProtocolKind::SingleShot;
  single.channels = {readyChannel(fixture)};
  single.channels[0].flow = SyncCoverEventChannelFlow::SingleShot;
  return passed &&
         check(verifySyncCoverEventProtocol(fixture.graph, target(), single)
                       .error == SyncCoverProtocolError::InvalidProtocol,
               "single-shot cuts cannot execute repeatedly in a loop");
}

bool testSingleShotRequiresExactNonLoopCardinality() {
  bool passed = true;
  SyncCoverGraph optional;
  const SyncCoverRegionId optionalRoot =
      takeIndex(optional.addRegion(0, SyncCoverRegionKind::Sequence,
                                   SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add single-shot optional root");
  const SyncCoverRegionId setRegion = takeIndex(
      optional.addRegion(optionalRoot, SyncCoverRegionKind::Transparent,
                         SyncCoverRegionCardinality::ZeroOrOne),
      passed, "add optional set region");
  const SyncCoverRegionId waitRegion = takeIndex(
      optional.addRegion(optionalRoot, SyncCoverRegionKind::Transparent,
                         SyncCoverRegionCardinality::ZeroOrOne),
      passed, "add optional wait region");
  const SyncCoverNodeId optionalProducer =
      takeIndex(optional.addNode(1, 1, 0, 0, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, setRegion),
                passed, "add independently optional producer");
  const SyncCoverNodeId optionalConsumer =
      takeIndex(optional.addNode(2, 1, 0, 1, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, waitRegion),
                passed, "add independently optional consumer");
  passed &= check(optional.freezeStructure(), "freeze optional protocol graph");
  SyncCoverEventProtocol optionalProtocol;
  optionalProtocol.mechanism = 20;
  optionalProtocol.kind = SyncCoverEventProtocolKind::SingleShot;
  optionalProtocol.channels = {{
      0,
      SyncCoverEventChannelFlow::SingleShot,
      point(SyncCoverCutPointKind::EventSet, 1, SyncCoverAnchorKind::AfterNode,
            optionalProducer),
      point(SyncCoverCutPointKind::EventWait, 2,
            SyncCoverAnchorKind::BeforeNode, optionalConsumer),
      1,
      0,
      kCompletion,
      {},
      false,
  }};
  passed &=
      check(verifySyncCoverEventProtocol(optional, target(), optionalProtocol)
                    .error == SyncCoverProtocolError::InvalidProtocol,
            "single-shot rejects independently optional actions");

  SyncCoverGraph repeated;
  const SyncCoverRegionId repeatedRoot =
      takeIndex(repeated.addRegion(0, SyncCoverRegionKind::Sequence,
                                   SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add repeated protocol root");
  const SyncCoverRegionId repeatedRegion = takeIndex(
      repeated.addRegion(repeatedRoot, SyncCoverRegionKind::Transparent,
                         SyncCoverRegionCardinality::OneOrMore),
      passed, "add non-loop repeated region");
  const SyncCoverNodeId repeatedProducer =
      takeIndex(repeated.addNode(1, 1, 0, 0, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, repeatedRegion),
                passed, "add repeated producer");
  const SyncCoverNodeId repeatedConsumer =
      takeIndex(repeated.addNode(2, 1, 0, 1, {}, {}, std::nullopt, false,
                                 std::numeric_limits<std::size_t>::max(), -1,
                                 {}, repeatedRegion),
                passed, "add repeated consumer");
  passed &= check(repeated.freezeStructure(), "freeze repeated protocol graph");
  SyncCoverEventProtocol repeatedProtocol;
  repeatedProtocol.mechanism = 21;
  repeatedProtocol.kind = SyncCoverEventProtocolKind::SingleShot;
  repeatedProtocol.channels = {{
      0,
      SyncCoverEventChannelFlow::SingleShot,
      point(SyncCoverCutPointKind::EventSet, 1, SyncCoverAnchorKind::AfterNode,
            repeatedProducer),
      point(SyncCoverCutPointKind::EventWait, 2,
            SyncCoverAnchorKind::BeforeNode, repeatedConsumer),
      1,
      0,
      kCompletion,
      {},
      false,
  }};
  return passed &&
         check(
             verifySyncCoverEventProtocol(repeated, target(), repeatedProtocol)
                     .error == SyncCoverProtocolError::InvalidProtocol,
             "single-shot rejects non-loop repeated actions");
}

bool testAllocationScarcity() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(false, passed);
  passed &= check(fixture.graph.freezeStructure(), "freeze allocation graph");
  std::vector<SyncCoverEventProtocol> protocols;
  for (std::size_t index = 0; index < 7; ++index) {
    SyncCoverEventProtocol protocol;
    protocol.mechanism = index;
    protocol.kind = SyncCoverEventProtocolKind::SingleShot;
    protocol.channels = {readyChannel(fixture)};
    protocols.push_back(std::move(protocol));
  }
  passed &= check(
      allocateSyncCoverProtocolEventIds(fixture.graph, target(), protocols)
              .error == SyncCoverProtocolError::ResourceInfeasible,
      "seven independent protocols fail with only IDs 0 through 5");
  protocols.pop_back();
  const SyncCoverProtocolAllocationResult allocation =
      allocateSyncCoverProtocolEventIds(fixture.graph, target(), protocols);
  passed &=
      check(allocation && allocation.channels.size() == 6 &&
                allocation.channels.back().eventIds == std::vector<unsigned>{5},
            "six protocols receive deterministic IDs 0 through 5");
  std::reverse(protocols.begin(), protocols.end());
  const SyncCoverProtocolAllocationResult permuted =
      allocateSyncCoverProtocolEventIds(fixture.graph, target(), protocols);
  passed &= check(permuted &&
                      permuted.channels.size() == allocation.channels.size() &&
                      permuted.channels.front().mechanism ==
                          allocation.channels.front().mechanism &&
                      permuted.channels.front().eventIds ==
                          allocation.channels.front().eventIds,
                  "allocation is invariant under protocol input order");
  const std::vector<SyncCoverProtocolEventReservation> reservations = {
      {1, 2, {0, 2, 4}}, {2, 1, {1, 3, 5}}};
  SyncCoverCoverageWorkBudget reservationMeasurement;
  passed &= check(static_cast<bool>(allocateSyncCoverProtocolEventIds(
                      fixture.graph, target(), {}, reservations, {},
                      &reservationMeasurement)),
                  "multi-reservation validation reports bounded work");
  SyncCoverCoverageWorkBudget exactReservationBudget(
      reservationMeasurement.workUnits);
  passed &= check(static_cast<bool>(allocateSyncCoverProtocolEventIds(
                      fixture.graph, target(), {}, reservations, {},
                      &exactReservationBudget)),
                  "an exact multi-reservation budget succeeds");
  SyncCoverCoverageWorkBudget shortReservationBudget(
      reservationMeasurement.workUnits - 1);
  passed &= check(allocateSyncCoverProtocolEventIds(fixture.graph, target(), {},
                                                    reservations, {},
                                                    &shortReservationBudget)
                          .error == SyncCoverProtocolError::WorkLimitExceeded,
                  "a one-less multi-reservation budget fails closed");
  return passed;
}

bool testProtocolBounds() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(false, passed);
  SyncCoverDemand demand;
  demand.source = fixture.producer;
  demand.target = fixture.consumer;
  passed &= check(fixture.graph.addDemand(demand), "add bounded demand");
  passed &= check(fixture.graph.freezeStructure(), "freeze bounded graph");
  SyncCoverEventProtocol protocol;
  protocol.mechanism = 3;
  protocol.kind = SyncCoverEventProtocolKind::SingleShot;
  protocol.channels = {readyChannel(fixture)};
  SyncCoverProtocolLimits limits;
  limits.maximumDynamicActions = 1;
  passed &= check(
      verifySyncCoverEventProtocol(fixture.graph, target(), protocol, limits)
              .error == SyncCoverProtocolError::LimitExceeded,
      "dynamic-action bound fails before oversubscription");
  SyncCoverCoverageWorkBudget budget(1);
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, target(),
                                               protocol, {}, &budget)
                          .error == SyncCoverProtocolError::WorkLimitExceeded,
                  "protocol verification honors its shared work budget");

  SyncCoverProtocolLimits laneLimits;
  laneLimits.maximumChannelLaneIncidences = 0;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, target(),
                                               protocol, laneLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "channel-lane storage is rejected before allocation");

  SyncCoverProtocolLimits graphLimits;
  graphLimits.maximumGraphRegions = fixture.graph.getRegions().size();
  passed &= check(static_cast<bool>(verifySyncCoverEventProtocol(
                      fixture.graph, target(), protocol, graphLimits)),
                  "an exact graph-region limit succeeds");
  graphLimits.maximumGraphRegions = fixture.graph.getRegions().size() - 1;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, target(),
                                               protocol, graphLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "a one-less graph-region limit fails before validation");

  SyncCoverProtocolLimits targetLimits;
  targetLimits.maximumTargetCapabilities = target().eventCapabilities.size();
  targetLimits.maximumTargetEventIds = target().compilerUsableEventIds.size();
  passed &= check(static_cast<bool>(verifySyncCoverEventProtocol(
                      fixture.graph, target(), protocol, targetLimits)),
                  "exact target-contract table limits succeed");
  targetLimits.maximumTargetCapabilities =
      target().eventCapabilities.size() - 1;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, target(),
                                               protocol, targetLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "one-less target-capability limit fails closed");
  targetLimits.maximumTargetCapabilities = target().eventCapabilities.size();
  targetLimits.maximumTargetEventIds =
      target().compilerUsableEventIds.size() - 1;
  passed &= check(verifySyncCoverEventProtocol(fixture.graph, target(),
                                               protocol, targetLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "one-less target-ID-table limit fails closed");

  ProtocolGraph phasedFixture = makeProtocolGraph(true, passed);
  addAlternatingPhase(phasedFixture, passed);
  passed &= check(phasedFixture.graph.freezeStructure(),
                  "freeze bounded phase graph");
  SyncCoverProtocolLimits phaseLimits;
  phaseLimits.maximumReachablePhases = 2;
  phaseLimits.maximumPhaseIncidences = 2;
  passed &= check(static_cast<bool>(verifySyncCoverEventProtocol(
                      phasedFixture.graph, target(),
                      roundTrip(phasedFixture, 2), phaseLimits)),
                  "an exact authoritative phase-table limit succeeds");
  phaseLimits.maximumReachablePhases = 1;
  passed &= check(
      verifySyncCoverEventProtocol(phasedFixture.graph, target(),
                                   roundTrip(phasedFixture, 2), phaseLimits)
              .error == SyncCoverProtocolError::LimitExceeded,
      "one-less authoritative phase-table limit fails before copying");

  SyncCoverProtocolLimits worldLimits;
  worldLimits.maximumWorlds = 1;
  passed &= check(
      computeSyncCoverProtocolExactWorlds(fixture.graph, target(), {protocol},
                                          {{{3}}, {{3}}}, worldLimits)
              .error == SyncCoverProtocolError::LimitExceeded,
      "world rows are bounded even for a tiny demand universe");
  worldLimits = {};
  worldLimits.maximumResultWords = 0;
  passed &= check(computeSyncCoverProtocolExactWorlds(
                      fixture.graph, target(), {protocol}, {{{3}}}, worldLimits)
                          .error == SyncCoverProtocolError::LimitExceeded,
                  "coverage result words are preflighted");
  return passed;
}

bool testDeepHierarchyAndLongPhaseCatalogWorkBounds() {
  bool passed = true;
  SyncCoverGraph deep;
  SyncCoverRegionId parentRegion =
      takeIndex(deep.addRegion(0, SyncCoverRegionKind::Sequence,
                               SyncCoverRegionCardinality::ExactlyOnce),
                passed, "add deep catalog root");
  SyncCoverScopeId parentScope = 0;
  for (unsigned depth = 0; depth < 12; ++depth) {
    const SyncCoverScopeId scope = takeIndex(
        deep.addScope(parentScope, true, std::nullopt, false, {}, parentRegion),
        passed, "add deep catalog scope");
    const SyncCoverRegionId region = takeIndex(
        deep.addRegion(parentRegion, SyncCoverRegionKind::Transparent,
                       SyncCoverRegionCardinality::ExactlyOnce, scope),
        passed, "add deep catalog region");
    passed &=
        check(deep.setScopeRegion(scope, region), "bind deep catalog region");
    const SyncCoverRegionId sequence = takeIndex(
        deep.addRegion(region, SyncCoverRegionKind::Sequence,
                       SyncCoverRegionCardinality::ExactlyOnce, scope),
        passed, "add deep catalog sequence");
    parentScope = scope;
    parentRegion = sequence;
  }
  const SyncCoverNodeId source =
      takeIndex(deep.addNode(1, 1, parentScope, 0, {}, {}, std::nullopt, false,
                             std::numeric_limits<std::size_t>::max(), -1, {},
                             parentRegion),
                passed, "add deep catalog source");
  const SyncCoverNodeId middle =
      takeIndex(deep.addNode(2, 1, parentScope, 1, {}, {}, std::nullopt, false,
                             std::numeric_limits<std::size_t>::max(), -1, {},
                             parentRegion),
                passed, "add deep catalog middle");
  const SyncCoverNodeId sink =
      takeIndex(deep.addNode(3, 1, parentScope, 2, {}, {}, std::nullopt, false,
                             std::numeric_limits<std::size_t>::max(), -1, {},
                             parentRegion),
                passed, "add deep catalog sink");
  passed &= check(deep.addDemand({source, sink}), "add deep catalog demand");
  passed &= check(deep.freezeStructure(), "freeze deep catalog graph");

  SyncCoverEventProtocol first;
  first.mechanism = 30;
  first.kind = SyncCoverEventProtocolKind::SingleShot;
  first.channels = {{0,
                     SyncCoverEventChannelFlow::SingleShot,
                     point(SyncCoverCutPointKind::EventSet, 1,
                           SyncCoverAnchorKind::AfterNode, source),
                     point(SyncCoverCutPointKind::EventWait, 2,
                           SyncCoverAnchorKind::BeforeNode, middle),
                     1,
                     0,
                     kCompletion,
                     {},
                     false}};
  SyncCoverEventProtocol second;
  second.mechanism = 31;
  second.kind = SyncCoverEventProtocolKind::SingleShot;
  second.channels = {{0,
                      SyncCoverEventChannelFlow::SingleShot,
                      point(SyncCoverCutPointKind::EventSet, 2,
                            SyncCoverAnchorKind::AfterNode, middle),
                      point(SyncCoverCutPointKind::EventWait, 3,
                            SyncCoverAnchorKind::BeforeNode, sink),
                      1,
                      0,
                      kCompletion,
                      {},
                      false}};
  SyncCoverProtocolTargetContract deepTarget = target();
  deepTarget.eventCapabilities.push_back({2, 3, kCompletion});
  std::sort(deepTarget.eventCapabilities.begin(),
            deepTarget.eventCapabilities.end());
  SyncCoverCoverageWorkBudget deepMeasurement;
  const SyncCoverProtocolCoverageResult deepCoverage =
      computeSyncCoverProtocolExactWorlds(deep, deepTarget, {first, second},
                                          {{{30, 31}}}, {}, &deepMeasurement);
  passed &=
      check(deepCoverage && deepCoverage.coveredByWorld.front().contains(0),
            "deep hierarchy composes two verified cuts");
  SyncCoverCoverageWorkBudget exactDeepBudget(deepMeasurement.workUnits);
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      deep, deepTarget, {first, second}, {{{30, 31}}}, {},
                      &exactDeepBudget)),
                  "an exact deep-hierarchy catalog budget succeeds");
  SyncCoverCoverageWorkBudget shortDeepBudget(deepMeasurement.workUnits - 1);
  passed &= check(
      computeSyncCoverProtocolExactWorlds(deep, deepTarget, {first, second},
                                          {{{30, 31}}}, {}, &shortDeepBudget)
              .error == SyncCoverProtocolError::WorkLimitExceeded,
      "a one-less deep-hierarchy catalog budget fails closed");

  ProtocolGraph phased = makeProtocolGraph(true, passed);
  addModuloPhase(phased, 8, passed);
  const SyncCoverControlId phaseControl = *phased.phaseControl;
  const SyncCoverGuard phaseZero{{{phaseControl, 0}}};
  const SyncCoverScopeId sparseScope =
      takeIndex(phased.graph.addScope(phased.loop, true, std::nullopt, false,
                                      phaseZero, phased.body),
                passed, "add sparse-phase scope");
  const SyncCoverRegionId sparseRegion = takeIndex(
      phased.graph.addRegion(phased.body, SyncCoverRegionKind::Transparent,
                             SyncCoverRegionCardinality::ExactlyOnce,
                             sparseScope, phaseZero),
      passed, "add sparse-phase region");
  passed &= check(phased.graph.setScopeRegion(sparseScope, sparseRegion),
                  "bind sparse-phase region");
  const SyncCoverRegionId sparseSequence = takeIndex(
      phased.graph.addRegion(sparseRegion, SyncCoverRegionKind::Sequence,
                             SyncCoverRegionCardinality::ExactlyOnce,
                             sparseScope, phaseZero),
      passed, "add sparse-phase sequence");
  const SyncCoverNodeId sparseProducer = takeIndex(
      phased.graph.addNode(1, 1, sparseScope, 4, phaseZero, {}, std::nullopt,
                           false, std::numeric_limits<std::size_t>::max(), -1,
                           {}, sparseSequence),
      passed, "add sparse-phase producer");
  const SyncCoverNodeId sparseConsumer = takeIndex(
      phased.graph.addNode(2, 1, sparseScope, 5, phaseZero, {}, std::nullopt,
                           false, std::numeric_limits<std::size_t>::max(), -1,
                           {}, sparseSequence),
      passed, "add sparse-phase consumer");
  SyncCoverDemand ready;
  ready.source = phased.producer;
  ready.target = phased.consumer;
  ready.scope = phased.loop;
  passed &= check(phased.graph.addDemand(ready), "add long-phase ready row");
  SyncCoverDemand release;
  release.source = phased.consumer;
  release.target = phased.producer;
  release.scope = phased.loop;
  release.distance = 8;
  passed &=
      check(phased.graph.addDemand(release), "add long-phase release row");
  passed &=
      check(phased.graph.freezeStructure(), "freeze long-phase catalog graph");
  SyncCoverEventProtocol phasedFirst = roundTrip(phased, 8);
  SyncCoverEventProtocol phasedSecond = phasedFirst;
  phasedFirst.mechanism = 40;
  phasedSecond.mechanism = 41;
  SyncCoverCoverageWorkBudget phaseMeasurement;
  const SyncCoverProtocolCoverageResult phaseCoverage =
      computeSyncCoverProtocolExactWorlds(phased.graph, target(),
                                          {phasedFirst, phasedSecond},
                                          {{{40, 41}}}, {}, &phaseMeasurement);
  passed &=
      check(phaseCoverage && phaseCoverage.coveredByWorld.front().count() == 2,
            "long authoritative phase catalog covers ready and release");
  SyncCoverCoverageWorkBudget exactPhaseBudget(phaseMeasurement.workUnits);
  passed &= check(static_cast<bool>(computeSyncCoverProtocolExactWorlds(
                      phased.graph, target(), {phasedFirst, phasedSecond},
                      {{{40, 41}}}, {}, &exactPhaseBudget)),
                  "an exact long-phase catalog budget succeeds");
  SyncCoverCoverageWorkBudget shortPhaseBudget(phaseMeasurement.workUnits - 1);
  passed &= check(computeSyncCoverProtocolExactWorlds(
                      phased.graph, target(), {phasedFirst, phasedSecond},
                      {{{40, 41}}}, {}, &shortPhaseBudget)
                          .error == SyncCoverProtocolError::WorkLimitExceeded,
                  "a one-less long-phase catalog budget fails closed");

  SyncCoverProtocolLoopSchedule sparseSchedule;
  sparseSchedule.scope = phased.loop;
  sparseSchedule.mayExecuteZeroTimes = true;
  sparseSchedule.phaseControl = phaseControl;
  sparseSchedule.laneByPhase.assign(8, 0);
  SyncCoverEventProtocol sparse;
  sparse.mechanism = 42;
  sparse.kind = SyncCoverEventProtocolKind::ProvenNoOverlap;
  sparse.loop = std::move(sparseSchedule);
  sparse.channels = {{
      0,
      SyncCoverEventChannelFlow::SameIteration,
      point(SyncCoverCutPointKind::EventSet, 1, SyncCoverAnchorKind::AfterNode,
            sparseProducer),
      point(SyncCoverCutPointKind::EventWait, 2,
            SyncCoverAnchorKind::BeforeNode, sparseConsumer),
      1,
      0,
      kCompletion,
      {0},
      false,
  }};
  sparse.rearmProofs = {{0, 0, 8, 77}};
  SyncCoverProtocolTargetContract sparseTarget = target();
  sparseTarget.certifiedRearmFacts = {{
      77,
      sparse.channels[0].wait.resource,
      sparse.channels[0].wait.anchor,
      sparse.channels[0].wait.guard,
      sparse.channels[0].set.resource,
      sparse.channels[0].set.anchor,
      sparse.channels[0].set.guard,
      phased.loop,
      8,
      1,
  }};
  SyncCoverCoverageWorkBudget sparseMeasurement;
  passed &=
      check(static_cast<bool>(verifySyncCoverEventProtocol(
                phased.graph, sparseTarget, sparse, {}, &sparseMeasurement)),
            "inactive phase-channel inspections report bounded work");
  SyncCoverCoverageWorkBudget exactSparseBudget(sparseMeasurement.workUnits);
  passed &=
      check(static_cast<bool>(verifySyncCoverEventProtocol(
                phased.graph, sparseTarget, sparse, {}, &exactSparseBudget)),
            "an exact sparse-phase protocol budget succeeds");
  SyncCoverCoverageWorkBudget shortSparseBudget(sparseMeasurement.workUnits -
                                                1);
  passed &= check(verifySyncCoverEventProtocol(phased.graph, sparseTarget,
                                               sparse, {}, &shortSparseBudget)
                          .error == SyncCoverProtocolError::WorkLimitExceeded,
                  "a one-less sparse-phase protocol budget fails closed");
  return passed;
}

bool testLifecycleSccDiscovery() {
  bool passed = true;
  ProtocolGraph fixture = makeProtocolGraph(true, passed);
  const SyncCoverStorageDomainId domain = takeIndex(
      fixture.graph.addStorageDomain(SyncCoverStorageDomainRole::Other, 3),
      passed, "add exact lifecycle domain");
  const SyncCoverStorageAccessId write =
      takeIndex(fixture.graph.addStorageAccess(
                    fixture.producer, domain, 1, {0, 64},
                    SyncCoverStorageAccessMode::Write, 0, true,
                    SyncCoverStorageAccessPath::PhysicalPipeline),
                passed, "add exact lifecycle write");
  const SyncCoverStorageAccessId read =
      takeIndex(fixture.graph.addStorageAccess(
                    fixture.consumer, domain, 1, {0, 64},
                    SyncCoverStorageAccessMode::Read, 0, true,
                    SyncCoverStorageAccessPath::PhysicalPipeline),
                passed, "add exact lifecycle read");
  const SyncCoverStorageWitnessId readyWitness =
      takeIndex(fixture.graph.addStorageWitness(write, read), passed,
                "add ready storage witness");
  const SyncCoverStorageWitnessId releaseWitness =
      takeIndex(fixture.graph.addStorageWitness(read, write), passed,
                "add release storage witness");
  SyncCoverDemand ready;
  ready.source = fixture.producer;
  ready.target = fixture.consumer;
  ready.scope = fixture.loop;
  ready.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  ready.storageWitnesses = {readyWitness};
  passed &= check(fixture.graph.addDemand(ready), "add lifecycle RAW");
  SyncCoverDemand release;
  release.source = fixture.consumer;
  release.target = fixture.producer;
  release.scope = fixture.loop;
  release.distance = 1;
  release.provenanceKinds = {SyncCoverDemandKind::MemoryWAR};
  release.storageWitnesses = {releaseWitness};
  passed &= check(fixture.graph.addDemand(release), "add lifecycle WAR");
  passed &= check(fixture.graph.freezeStructure(), "freeze lifecycle graph");
  const SyncCoverLifecycleSccResult result =
      discoverSyncCoverLifecycleSccs(fixture.graph);
  passed &= check(result && result.components.size() == 1 &&
                      result.components.front().demands.size() == 2 &&
                      result.components.front().maximumDistance == 1 &&
                      result.components.front().storageDomains ==
                          std::vector<SyncCoverStorageDomainId>{domain},
                  "RAW plus loop-carried WAR forms one exact-slot SCC");
  return passed;
}

} // namespace

int main() {
  const bool passed = testSingleShotAndTargetContract() &&
                      testRoundTripAndRotatingLifecycle() &&
                      testGuardAndPhaseQualification() &&
                      testDirectProtocolRectangles() &&
                      testWholeWorldCompositionAndExitExport() &&
                      testStructuredMustAndFixedSupplyCoverage() &&
                      testNestedLoopDistancesAreScopeTyped() &&
                      testAuthoritativePhaseAndCardinality() &&
                      testSingleShotRequiresExactNonLoopCardinality() &&
                      testAllocationScarcity() && testProtocolBounds() &&
                      testDeepHierarchyAndLongPhaseCatalogWorkBounds() &&
                      testLifecycleSccDiscovery();
  return passed ? 0 : 1;
}
