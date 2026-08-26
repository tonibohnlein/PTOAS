// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncCoveringSelection.h"
#include "CanonicalSyncCoveringSlotRecipe.h"

#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include <array>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

namespace {

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_covering;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncCoveringTranslationTest failure: " << message
              << '\n';
  }
  return condition;
}

std::size_t takeGraphIndex(const SyncCoverGraphResult &result, bool &passed,
                           std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

std::size_t takeMechanismIndex(const SyncCoverMechanismResult &result,
                               bool &passed, std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

bool testDescriptorAttestation() {
  bool passed = true;
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const Location location = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "translation_attestation",
      FunctionType::get(&context, TypeRange{}, TypeRange{}));
  module.push_back(function);
  Block *body = function.addEntryBlock();
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(body);
  Operation *sourceOperation =
      builder.create<arith::ConstantIndexOp>(location, 0).getOperation();
  Operation *targetOperation =
      builder.create<arith::ConstantIndexOp>(location, 1).getOperation();
  builder.create<func::ReturnOp>(location);

  const std::uint32_t sourceResource =
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE2);
  const std::uint32_t targetResource =
      static_cast<std::uint32_t>(PipelineType::PIPE_V);
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(sourceResource, 1, 0, 0, {}, {targetResource}), passed,
      "add attested source");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(targetResource, 1, 0, 1), passed,
      "add attested target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId,
                                 sourceResource, targetResource, 8),
      passed, "add attested event domain");
  const SyncCoverResourceDomainId wrongDomain = takeMechanismIndex(
      universe.addResourceDomain(
          SyncCoverResourceKind::EventId,
          static_cast<std::uint32_t>(PipelineType::PIPE_M),
          static_cast<std::uint32_t>(PipelineType::PIPE_FIX), 8),
      passed, "add unrelated event domain");
  DomainMap domains;
  domains.emplace(CanonicalEventDomainKey{PipelineType::PIPE_MTE2,
                                           PipelineType::PIPE_V},
                  domain);
  std::map<Region *, SyncCoverScopeId, std::less<Region *>> regionScopes;
  regionScopes.emplace(&function.getBody(), 0);
  DenseMap<Operation *, SyncCoverScopeId> loopScopes;
  const auto getAnchorPosition = [&](const CanonicalAnchor &anchor) {
    if (anchor.operation == sourceOperation) {
      return anchor.before ? std::size_t{0} : std::size_t{1};
    }
    return anchor.before ? std::size_t{2} : std::size_t{3};
  };

  CanonicalEvent event;
  event.source = source;
  event.target = target;
  event.sourcePipe = PipelineType::PIPE_MTE2;
  event.targetPipe = PipelineType::PIPE_V;
  event.setAnchor = {sourceOperation, false};
  event.waitAnchor = {targetOperation, true};
  event.intervalBegin = 1;
  event.intervalEnd = 2;
  CanonicalEventAction set;
  set.kind = CanonicalEventActionKind::Set;
  set.anchor = event.setAnchor;
  CanonicalEventAction wait;
  wait.kind = CanonicalEventActionKind::Wait;
  wait.anchor = event.waitAnchor;
  event.actions = {set, wait};
  event.completions.push_back({source, target, 0, nullptr, 0, 1});
  CanonicalEventBundleCandidate bundle;
  bundle.kind = CanonicalEventBundleKind::Standalone;
  bundle.events.push_back(event);

  const std::optional<TranslatedEventBundleMechanism> translated =
      translateVerifiedEventBundle(bundle, 1, domains, universe, regionScopes,
                                   loopScopes, getAnchorPosition);
  passed &= check(translated.has_value(), "translate canonical event bundle");
  if (!translated) {
    return false;
  }
  const auto verifies = [&](const CanonicalEventBundleCandidate &candidate,
                            const TranslatedEventBundleMechanism &translation,
                            const SyncCoverMechanismDescriptor &descriptor,
                            const DomainMap &domainMap) {
    return verifyTranslatedEventBundleCorrespondence(
        candidate, translation, descriptor, universe, domainMap, regionScopes,
        loopScopes, getAnchorPosition);
  };
  passed &= check(verifies(bundle, *translated, translated->descriptor,
                           domains),
                  "exact translated descriptor is attested");

  TranslatedEventBundleMechanism wrongUse = *translated;
  wrongUse.descriptor.resourceUses.front().distance = 1;
  passed &= check(!verifies(bundle, wrongUse, wrongUse.descriptor, domains),
                  "wrong recurrence distance is rejected");
  TranslatedEventBundleMechanism wrongAction = *translated;
  wrongAction.descriptor.actions.front().kind =
      SyncCoverResourceActionKind::Consume;
  passed &= check(
      !verifies(bundle, wrongAction, wrongAction.descriptor, domains),
                  "wrong physical action is rejected");
  TranslatedEventBundleMechanism wrongAnchor = *translated;
  ++wrongAnchor.descriptor.actions.front().anchor.position;
  passed &= check(
      !verifies(bundle, wrongAnchor, wrongAnchor.descriptor, domains),
                  "wrong physical anchor is rejected");
  TranslatedEventBundleMechanism wrongEdge = *translated;
  wrongEdge.descriptor.supplyEdges.front().target = source;
  passed &= check(!verifies(bundle, wrongEdge, wrongEdge.descriptor, domains),
                  "wrong completion endpoint is rejected");
  TranslatedEventBundleMechanism wrongBinding = *translated;
  wrongBinding.descriptor.supplyBindings.front().consumeAction =
      wrongBinding.descriptor.supplyBindings.front().produceAction;
  passed &= check(
      !verifies(bundle, wrongBinding, wrongBinding.descriptor, domains),
                  "wrong set/wait binding is rejected");
  CanonicalEventBundleCandidate wrongLifetime = bundle;
  ++wrongLifetime.events.front().intervalEnd;
  passed &= check(!verifies(wrongLifetime, *translated,
                            translated->descriptor, domains),
                  "wrong canonical lifetime is rejected");
  DomainMap wrongDomains = domains;
  wrongDomains[CanonicalEventDomainKey{PipelineType::PIPE_MTE2,
                                        PipelineType::PIPE_V}] = wrongDomain;
  TranslatedEventBundleMechanism wrongDomainTranslation = *translated;
  SyncCoverResourceUse &wrongDomainUse =
      wrongDomainTranslation.descriptor.resourceUses.front();
  wrongDomainUse.domain = wrongDomain;
  wrongDomainTranslation.descriptor.actions[0].resource =
      static_cast<std::uint32_t>(PipelineType::PIPE_M);
  wrongDomainTranslation.descriptor.actions[1].resource =
      static_cast<std::uint32_t>(PipelineType::PIPE_FIX);
  passed &= check(!verifies(bundle, wrongDomainTranslation,
                            wrongDomainTranslation.descriptor, wrongDomains),
                  "event-domain resources must match canonical pipes");
  return passed;
}

bool testExactSlotOrdinalPairing() {
  bool passed = true;
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect>();
  const Location location = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "slot_pairing",
      FunctionType::get(&context, TypeRange{IndexType::get(&context)},
                        TypeRange{}));
  module.push_back(function);
  Block *body = function.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(body);
  Value two = builder.create<arith::ConstantIndexOp>(location, 2);
  Value slot = builder.create<arith::RemUIOp>(location, body->getArgument(0),
                                              two);
  builder.create<func::ReturnOp>(location);

  const auto distanceOne = enumerateSlotSSAOrdinalPairs(
      slot, slot, 2, body->getArgument(0), 1);
  const auto distanceTwo = enumerateSlotSSAOrdinalPairs(
      slot, slot, 2, body->getArgument(0), 2);
  passed &= check(
      distanceOne &&
          *distanceOne ==
              SmallVector<SlotOrdinalPair, 4>{{0, 1}, {1, 0}},
      "distance-one ping-pong accesses select disjoint ordinals");
  passed &= check(
      distanceTwo &&
          *distanceTwo ==
              SmallVector<SlotOrdinalPair, 4>{{0, 0}, {1, 1}},
      "distance-two ping-pong accesses return to matching ordinals");
  Value bare = body->getArgument(0);
  passed &= check(!enumerateSlotSSAOrdinalPairs(bare, bare, 2),
                  "unnormalized selectors fail exact pairing closed");
  return passed;
}

bool testSlotProtocolRecipeAttestation() {
  bool passed = true;
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
  const Location location = UnknownLoc::get(&context);
  ModuleOp module = ModuleOp::create(location);
  auto function = func::FuncOp::create(
      location, "slot_recipe_attestation",
      FunctionType::get(&context, TypeRange{}, TypeRange{}));
  module.push_back(function);
  Block *body = function.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(body);
  Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
  Value one = builder.create<arith::ConstantIndexOp>(location, 1);
  Value two = builder.create<arith::ConstantIndexOp>(location, 2);
  auto outer = builder.create<scf::ForOp>(location, zero, two, one);
  builder.setInsertionPointToStart(outer.getBody());
  Operation *unitTarget =
      builder.create<arith::ConstantIndexOp>(location, 3).getOperation();
  auto inner = builder.create<scf::ForOp>(location, zero, two, one);
  builder.setInsertionPointToStart(inner.getBody());
  Operation *firstNestedTarget =
      builder.create<arith::ConstantIndexOp>(location, 4).getOperation();
  Operation *secondNestedTarget =
      builder.create<arith::ConstantIndexOp>(location, 5).getOperation();
  builder.setInsertionPointAfter(inner);
  Operation *source =
      builder.create<arith::ConstantIndexOp>(location, 6).getOperation();
  builder.setInsertionPointAfter(outer);
  builder.create<func::ReturnOp>(location);

  const std::uint32_t sourceResource =
      static_cast<std::uint32_t>(PipelineType::PIPE_FIX);
  const std::uint32_t targetResource =
      static_cast<std::uint32_t>(PipelineType::PIPE_M);
  SyncCoverGraph graph;
  const SyncCoverScopeId outerScope = takeGraphIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 100}, true),
      passed, "add recipe outer loop");
  const SyncCoverScopeId innerScope = takeGraphIndex(
      graph.addScope(outerScope, false, SyncCoverTimelineInterval{10, 70},
                     true),
      passed, "add recipe inner loop");
  const SyncCoverNodeId unitTargetNode = takeGraphIndex(
      graph.addNode(targetResource, 1, outerScope, 10, {}, {sourceResource}),
      passed, "add recipe unit target");
  const SyncCoverNodeId firstNestedNode = takeGraphIndex(
      graph.addNode(targetResource, 1, innerScope, 15, {}, {sourceResource}),
      passed, "add first recipe nested target");
  const SyncCoverNodeId secondNestedNode = takeGraphIndex(
      graph.addNode(targetResource, 1, innerScope, 20, {}, {sourceResource}),
      passed, "add second recipe nested target");
  const SyncCoverNodeId sourceNode = takeGraphIndex(
      graph.addNode(sourceResource, 1, outerScope, 45), passed,
      "add recipe source");
  passed &= check(static_cast<bool>(graph.freezeStructure()),
                  "freeze recipe graph");

  std::vector<CanonicalSyncNode> nodes(4);
  const std::array<Operation *, 4> operations = {
      unitTarget, firstNestedTarget, secondNestedTarget, source};
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    nodes[index].id = index;
    nodes[index].operation = operations[index];
  }
  DenseMap<Operation *, SyncCoverScopeId> loopScopes;
  loopScopes[outer.getOperation()] = outerScope;
  loopScopes[inner.getOperation()] = innerScope;
  const auto getAnchorPosition = [&](const CanonicalAnchor &anchor) {
    if (anchor.operation == outer.getOperation()) {
      return anchor.before ? std::size_t{2} : std::size_t{100};
    }
    if (anchor.operation == inner.getOperation()) {
      return anchor.before ? std::size_t{10} : std::size_t{70};
    }
    if (anchor.operation == unitTarget) {
      return anchor.before ? std::size_t{20} : std::size_t{21};
    }
    if (anchor.operation == firstNestedTarget) {
      return anchor.before ? std::size_t{30} : std::size_t{31};
    }
    if (anchor.operation == secondNestedTarget) {
      return anchor.before ? std::size_t{40} : std::size_t{41};
    }
    return anchor.before ? std::size_t{90} : std::size_t{91};
  };

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId,
                                 sourceResource, targetResource, 8),
      passed, "add recipe event domain");
  const auto addProtocol = [&](const SyncCoverSlotProtocolCandidate &candidate,
                               std::uint64_t provider) {
    const auto descriptor = makeSyncCoverSlotProtocolDescriptor(
        universe.getResourceDomains()[domain], candidate, provider);
    passed &= check(descriptor.has_value(), "build recipe descriptor");
    if (!descriptor) {
      return std::optional<SyncCoverMechanismId>{};
    }
    const SyncCoverMechanismResult result = universe.addVerifiedProtocol(
        *descriptor, [](const auto &) { return true; });
    passed &= check(result && result.index.has_value(),
                    "admit recipe descriptor");
    return result.index;
  };

  SyncCoverSlotProtocolCandidate unit;
  unit.kind = SyncCoverSlotProtocolKind::UnitRelease;
  unit.releases = {0};
  unit.completionEdges = {{sourceNode, unitTargetNode}};
  unit.sources = {sourceNode};
  unit.sourceLanes = {0};
  unit.source = sourceNode;
  unit.targets = {unitTargetNode};
  unit.targetWaits = {unitTargetNode};
  unit.paths = {{{sourceNode}, unitTargetNode}};
  unit.sourceResource = sourceResource;
  unit.targetResource = targetResource;
  unit.recurrenceScope = outerScope;
  unit.distance = 1;
  const std::optional<SyncCoverMechanismId> unitMechanismId =
      addProtocol(unit, 501);
  if (!unitMechanismId) {
    return false;
  }
  const SyncCoverMechanism &unitMechanism =
      universe.getMechanisms()[*unitMechanismId];
  const auto unitRecipe = buildSlotProtocolRecipe(
      nodes, universe, unit, unitMechanism, loopScopes, getAnchorPosition);
  passed &= check(
      unitRecipe && verifySlotProtocolRecipeCorrespondence(
                        nodes, universe, unit, unitMechanism, loopScopes,
                        getAnchorPosition, *unitRecipe),
      "attest unit slot protocol recipe independently");

  SyncCoverSlotProtocolCandidate hierarchical = unit;
  hierarchical.id = 1;
  hierarchical.kind = SyncCoverSlotProtocolKind::HierarchicalRelease;
  hierarchical.releases = {1, 2};
  hierarchical.completionEdges = {
      {sourceNode, firstNestedNode}, {sourceNode, secondNestedNode}};
  hierarchical.targets = {firstNestedNode, secondNestedNode};
  hierarchical.waitScope = innerScope;
  hierarchical.targetWaits = {firstNestedNode, firstNestedNode};
  hierarchical.paths = {{{sourceNode}, firstNestedNode}};
  const std::optional<SyncCoverMechanismId> hierarchicalMechanismId =
      addProtocol(hierarchical, 502);
  if (!hierarchicalMechanismId) {
    return false;
  }
  const SyncCoverMechanism &hierarchicalMechanism =
      universe.getMechanisms()[*hierarchicalMechanismId];
  const auto hierarchicalRecipe = buildSlotProtocolRecipe(
      nodes, universe, hierarchical, hierarchicalMechanism, loopScopes,
      getAnchorPosition);
  passed &= check(
          hierarchicalRecipe &&
          hierarchicalRecipe->event.actions[1].anchor.operation ==
              inner.getOperation() &&
          hierarchicalRecipe->event.actions[1].anchor.before &&
          hierarchicalRecipe->event.completions.size() == 2 &&
          verifySlotProtocolRecipeCorrespondence(
              nodes, universe, hierarchical, hierarchicalMechanism,
              loopScopes, getAnchorPosition, *hierarchicalRecipe),
      "attest hierarchical slot protocol recipe independently");
  if (!hierarchicalRecipe) {
    return false;
  }

  const CanonicalSelectionMechanismRef hierarchicalProvider{
      CanonicalSelectionMechanismKind::SlotProtocol, hierarchical.id};
  const CanonicalSyncCoveringSelectedSlotProtocol selectedRecipe{
      *hierarchicalMechanismId, hierarchicalProvider, 0,
      hierarchicalRecipe->event};
  CanonicalSyncCoveringSelectedResourceUse selectedUse;
  selectedUse.mechanism = *hierarchicalMechanismId;
  selectedUse.provider = hierarchicalProvider;
  selectedUse.domain = domain;
  selectedUse.kind = SyncCoverResourceKind::EventId;
  selectedUse.sourceResource = sourceResource;
  selectedUse.targetResource = targetResource;
  selectedUse.scope = outerScope;
  selectedUse.distance = 1;
  selectedUse.width = 1;
  selectedUse.lifetime = {2, 100};
  CanonicalSyncCoveringResourceAllocation selectedAllocation;
  selectedAllocation.mechanism = *hierarchicalMechanismId;
  selectedAllocation.provider = hierarchicalProvider;
  selectedAllocation.domain = domain;
  selectedAllocation.kind = SyncCoverResourceKind::EventId;
  selectedAllocation.sourceResource = sourceResource;
  selectedAllocation.targetResource = targetResource;
  selectedAllocation.ids = {6};
  const auto materialized = materializeSlotProtocolBundle(
      selectedRecipe, selectedUse, selectedAllocation, 900);
  passed &= check(
      materialized && materialized->id == 900 &&
          materialized->kind == CanonicalEventBundleKind::Standalone &&
          materialized->events.size() == 1 &&
          materialized->events.front().eventIds ==
              SmallVector<unsigned, 2>{6} &&
          materialized->events.front().actions.size() == 4 &&
          materialized->events.front().actions[0].anchor.operation ==
              outer.getOperation() &&
          materialized->events.front().actions[0].anchor.before &&
          materialized->events.front().actions[1].anchor.operation ==
              inner.getOperation() &&
          materialized->events.front().actions[1].anchor.before &&
          materialized->events.front().actions[2].anchor.operation == source &&
          !materialized->events.front().actions[2].anchor.before &&
          materialized->events.front().actions[3].anchor.operation ==
              outer.getOperation() &&
          !materialized->events.front().actions[3].anchor.before &&
          materialized->events.front().completions.size() == 2,
      "materialize a forced hierarchical selection with its exact event ID");
  CanonicalSyncCoveringResourceAllocation wideAllocation =
      selectedAllocation;
  wideAllocation.ids.push_back(7);
  passed &= check(!materializeSlotProtocolBundle(
                      selectedRecipe, selectedUse, wideAllocation, 901),
                  "reject a hierarchical allocation with the wrong width");

  SlotProtocolRecipe wrongAnchor = *hierarchicalRecipe;
  wrongAnchor.event.actions[1].anchor = {source, true};
  passed &= check(
      !verifySlotProtocolRecipeCorrespondence(
          nodes, universe, hierarchical, hierarchicalMechanism, loopScopes,
          getAnchorPosition, wrongAnchor),
      "reject a recipe with a moved hierarchical wait");
  SlotProtocolRecipe wrongCompletion = *hierarchicalRecipe;
  wrongCompletion.event.completions.front().target = sourceNode;
  passed &= check(
      !verifySlotProtocolRecipeCorrespondence(
          nodes, universe, hierarchical, hierarchicalMechanism, loopScopes,
          getAnchorPosition, wrongCompletion),
      "reject a recipe with a wrong completion endpoint");
  SlotProtocolRecipe wrongAction = *hierarchicalRecipe;
  wrongAction.event.actions[2].kind = CanonicalEventActionKind::Wait;
  passed &= check(
      !verifySlotProtocolRecipeCorrespondence(
          nodes, universe, hierarchical, hierarchicalMechanism, loopScopes,
          getAnchorPosition, wrongAction),
      "reject a recipe with a wrong physical action");
  SyncCoverMechanism wrongBinding = hierarchicalMechanism;
  wrongBinding.supplyBindings.front().consumeAction = 3;
  passed &= check(
      !verifySlotProtocolRecipeCorrespondence(
          nodes, universe, hierarchical, wrongBinding, loopScopes,
          getAnchorPosition, *hierarchicalRecipe),
      "reject a recipe backed by a wrong supply binding");
  SyncCoverMechanism wrongMechanismAction = hierarchicalMechanism;
  wrongMechanismAction.actions[1].anchor.scope = outerScope;
  passed &= check(
      !verifySlotProtocolRecipeCorrespondence(
          nodes, universe, hierarchical, wrongMechanismAction, loopScopes,
          getAnchorPosition, *hierarchicalRecipe),
      "reject a recipe backed by a moved mechanism action");
  DenseMap<Operation *, SyncCoverScopeId> missingLoopScopes = loopScopes;
  missingLoopScopes.erase(outer.getOperation());
  passed &= check(!canBuildSlotProtocolRecipe(hierarchical,
                                               missingLoopScopes),
                  "skip an unmaterializable hierarchical loop");
  return passed;
}

bool testProviderIdentityNamespaces() {
  constexpr std::uint64_t kSlotProtocolTag = std::uint64_t{1} << 63;
  const auto barrierZero = encodeProviderIdentity(
      CanonicalSelectionMechanismKind::Barrier, 0);
  const auto eventZero = encodeProviderIdentity(
      CanonicalSelectionMechanismKind::EventBundle, 0);
  const auto slotZero = encodeProviderIdentity(
      CanonicalSelectionMechanismKind::SlotProtocol, 0);
  bool passed = check(barrierZero == 1 && eventZero == 2 &&
                          slotZero == kSlotProtocolTag + 1,
                      "provider namespaces preserve legacy identities");
  passed &= check(
      !encodeProviderIdentity(
          static_cast<CanonicalSelectionMechanismKind>(255), 0),
      "unknown provider kinds are rejected");
  const bool hasWideSize =
      std::numeric_limits<std::size_t>::max() >= kSlotProtocolTag - 1;
  const std::size_t slotOverflowId =
      static_cast<std::size_t>(kSlotProtocolTag - 1);
  const std::size_t maximumEventId =
      static_cast<std::size_t>((kSlotProtocolTag - 3) / 2);
  passed &= check(
      !hasWideSize ||
          !encodeProviderIdentity(
              CanonicalSelectionMechanismKind::SlotProtocol, slotOverflowId),
      "slot protocol provider overflow is rejected");
  passed &= check(
      !hasWideSize ||
          (encodeProviderIdentity(
               CanonicalSelectionMechanismKind::EventBundle, maximumEventId)
               .has_value() &&
           !encodeProviderIdentity(
               CanonicalSelectionMechanismKind::EventBundle,
               maximumEventId + 1)),
      "legacy provider identities cannot enter the slot namespace");
  return passed;
}

} // namespace

int main() {
  return testDescriptorAttestation() && testExactSlotOrdinalPairing() &&
                 testSlotProtocolRecipeAttestation() &&
                 testProviderIdentityNamespaces()
             ? 0
             : 1;
}
