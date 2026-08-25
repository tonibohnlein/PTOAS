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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include <iostream>
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

bool testBarrierFreeOwnershipProviderClassification() {
  bool passed = true;
  CanonicalOwnershipCycle compositeCycle;
  compositeCycle.id = 1;
  CanonicalOwnershipCycle nativeCycle;
  nativeCycle.id = 2;
  nativeCycle.protocol = CanonicalOwnershipProtocolKind::RoundTrip;

  CanonicalEvent compositeEvent;
  compositeEvent.ownershipProtocol = true;
  compositeEvent.ownershipCycle = compositeCycle.id;
  CanonicalEventBundleCandidate composite;
  composite.id = 7;
  composite.kind = CanonicalEventBundleKind::CompositeOwnership;
  composite.events.push_back(compositeEvent);

  CanonicalEventBundleCandidate native;
  native.id = 9;
  native.kind = CanonicalEventBundleKind::Ownership;
  native.protocolIdentity = nativeCycle.id;
  native.ownershipProtocol = nativeCycle.protocol;

  const SmallVector<CanonicalOwnershipCycle, 2> cycles = {compositeCycle,
                                                          nativeCycle};
  const BarrierFreeOwnershipProviderSet complete =
      buildBarrierFreeOwnershipProviderSet(cycles, {composite, native});
  passed &= check(
      complete.applicable && complete.complete &&
          complete.providers ==
              std::vector<CanonicalSelectionMechanismRef>(
                  {{CanonicalSelectionMechanismKind::EventBundle, 7},
                   {CanonicalSelectionMechanismKind::EventBundle, 9}}),
      "composite and remaining native ownership form a complete seed");

  const BarrierFreeOwnershipProviderSet missing =
      buildBarrierFreeOwnershipProviderSet(cycles, {composite});
  passed &= check(missing.applicable && !missing.complete &&
                      missing.providers.empty(),
                  "missing native ownership is classified as incomplete");

  CanonicalEventBundleCandidate secondComposite = composite;
  secondComposite.id = 8;
  const BarrierFreeOwnershipProviderSet ambiguous =
      buildBarrierFreeOwnershipProviderSet(
          cycles, {composite, secondComposite, native});
  passed &= check(ambiguous.applicable && !ambiguous.complete &&
                      ambiguous.providers.empty(),
                  "ambiguous composite ownership is classified as incomplete");

  const BarrierFreeOwnershipProviderSet notApplicable =
      buildBarrierFreeOwnershipProviderSet(cycles, {native});
  passed &= check(!notApplicable.applicable && !notApplicable.complete,
                  "membership is not applicable without a composite");
  return passed;
}

} // namespace

int main() {
  return testDescriptorAttestation() &&
                 testBarrierFreeOwnershipProviderClassification()
             ? 0
             : 1;
}
