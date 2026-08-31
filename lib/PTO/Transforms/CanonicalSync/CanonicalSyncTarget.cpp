// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED
// ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
// FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
// repository for the full text of the License.

#include "CanonicalSyncTarget.h"

#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <initializer_list>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {

std::uint32_t resourceId(PipelineType resource) {
  return static_cast<std::uint32_t>(resource);
}

CanonicalSyncResourceCapability
makeResourceCapability(std::initializer_list<PipelineType> resources) {
  CanonicalSyncResourceCapability capability;
  capability.version = 1;
  capability.resources.reserve(resources.size());
  for (PipelineType resource : resources) {
    capability.resources.push_back(resourceId(resource));
  }
  llvm::sort(capability.resources);
  capability.resources.erase(
      std::unique(capability.resources.begin(), capability.resources.end()),
      capability.resources.end());
  return capability;
}

CanonicalSyncDirectedResourceCapability makeDirectedCapability(
    std::initializer_list<std::pair<PipelineType, PipelineType>> pairs) {
  CanonicalSyncDirectedResourceCapability capability;
  capability.version = 1;
  capability.resourcePairs.reserve(pairs.size());
  for (const auto &[source, target] : pairs) {
    capability.resourcePairs.emplace_back(resourceId(source),
                                          resourceId(target));
  }
  llvm::sort(capability.resourcePairs);
  capability.resourcePairs.erase(std::unique(capability.resourcePairs.begin(),
                                             capability.resourcePairs.end()),
                                 capability.resourcePairs.end());
  return capability;
}

CanonicalSyncDirectedResourceCapability makeAiv2201HardwareEvents() {
  using P = PipelineType;
  return makeDirectedCapability({
      {P::PIPE_S, P::PIPE_V},
      {P::PIPE_S, P::PIPE_MTE2},
      {P::PIPE_S, P::PIPE_MTE3},
      {P::PIPE_V, P::PIPE_S},
      {P::PIPE_V, P::PIPE_MTE2},
      {P::PIPE_V, P::PIPE_MTE3},
      {P::PIPE_MTE2, P::PIPE_S},
      {P::PIPE_MTE2, P::PIPE_V},
      {P::PIPE_MTE2, P::PIPE_MTE3},
      {P::PIPE_MTE3, P::PIPE_S},
      {P::PIPE_MTE3, P::PIPE_V},
      {P::PIPE_MTE3, P::PIPE_MTE2},
  });
}

CanonicalSyncDirectedResourceCapability makeAic2201HardwareEvents() {
  using P = PipelineType;
  return makeDirectedCapability({
      {P::PIPE_M, P::PIPE_MTE1},
      {P::PIPE_M, P::PIPE_MTE2},
      {P::PIPE_M, P::PIPE_FIX},
      {P::PIPE_MTE1, P::PIPE_M},
      {P::PIPE_MTE1, P::PIPE_MTE2},
      {P::PIPE_MTE1, P::PIPE_MTE3},
      {P::PIPE_MTE1, P::PIPE_FIX},
      {P::PIPE_MTE2, P::PIPE_M},
      {P::PIPE_MTE2, P::PIPE_MTE1},
      {P::PIPE_MTE2, P::PIPE_MTE3},
      {P::PIPE_MTE2, P::PIPE_FIX},
      {P::PIPE_MTE3, P::PIPE_MTE1},
      {P::PIPE_MTE3, P::PIPE_MTE2},
      {P::PIPE_MTE3, P::PIPE_FIX},
      {P::PIPE_FIX, P::PIPE_M},
      {P::PIPE_FIX, P::PIPE_MTE1},
      {P::PIPE_FIX, P::PIPE_MTE2},
      {P::PIPE_FIX, P::PIPE_MTE3},
  });
}

CanonicalSyncDirectedResourceCapability makeAic2201ExposedEvents() {
  using P = PipelineType;
  // The four omitted hardware pairs are documented as having no current
  // programming scenario. They are not compiler-authorized merely because
  // the hardware table acknowledges them.
  return makeDirectedCapability({
      {P::PIPE_M, P::PIPE_MTE1},
      {P::PIPE_M, P::PIPE_MTE2},
      {P::PIPE_M, P::PIPE_FIX},
      {P::PIPE_MTE1, P::PIPE_M},
      {P::PIPE_MTE1, P::PIPE_MTE2},
      {P::PIPE_MTE1, P::PIPE_MTE3},
      {P::PIPE_MTE1, P::PIPE_FIX},
      {P::PIPE_MTE2, P::PIPE_M},
      {P::PIPE_MTE2, P::PIPE_MTE1},
      {P::PIPE_MTE2, P::PIPE_MTE3},
      {P::PIPE_MTE3, P::PIPE_MTE1},
      {P::PIPE_MTE3, P::PIPE_MTE2},
      {P::PIPE_FIX, P::PIPE_M},
      {P::PIPE_FIX, P::PIPE_MTE1},
  });
}

CanonicalSyncCoreDomain resolveCoreDomain(func::FuncOp function) {
  std::optional<CanonicalSyncCoreDomain> resolved;
  const auto merge = [&](CanonicalSyncCoreDomain candidate) {
    if (resolved && *resolved != candidate) {
      return false;
    }
    resolved = candidate;
    return true;
  };
  for (Operation *owner = function; owner != nullptr;
       owner = owner->getParentOp()) {
    auto kind = owner->getAttrOfType<FunctionKernelKindAttr>(
        FunctionKernelKindAttr::name);
    if (!kind) {
      continue;
    }
    const CanonicalSyncCoreDomain candidate =
        kind.getKernelKind() == FunctionKernelKind::Cube
            ? CanonicalSyncCoreDomain::AIC
            : CanonicalSyncCoreDomain::AIV;
    if (!merge(candidate)) {
      return CanonicalSyncCoreDomain::Conflict;
    }
  }
  const WalkResult sectionResult = function.walk([&](Operation *operation) {
    std::optional<CanonicalSyncCoreDomain> candidate;
    if (isa<SectionCubeOp>(operation)) {
      candidate = CanonicalSyncCoreDomain::AIC;
    } else if (isa<SectionVectorOp>(operation)) {
      candidate = CanonicalSyncCoreDomain::AIV;
    }
    return !candidate || merge(*candidate) ? WalkResult::advance()
                                           : WalkResult::interrupt();
  });
  if (sectionResult.wasInterrupted()) {
    return CanonicalSyncCoreDomain::Conflict;
  }
  return resolved.value_or(CanonicalSyncCoreDomain::Unresolved);
}

void addCommonEvidence(CanonicalSyncTargetCapabilities &capabilities) {
  capabilities.evidence = {
      CanonicalSyncTargetEvidence::AscendIntraCoreSync7008190b,
      CanonicalSyncTargetEvidence::AscendSetWait7008190b,
      CanonicalSyncTargetEvidence::AscendKeyFeatures7008190b,
      CanonicalSyncTargetEvidence::AscendPipeBarrier850,
  };
  capabilities.compilerUsableEventIds = {0, 1, 2, 3, 4, 5};
}

void add2201EventCapabilities(CanonicalSyncTargetCapabilities &capabilities) {
  switch (capabilities.coreDomain) {
  case CanonicalSyncCoreDomain::AIC:
    capabilities.hardwareEventCompletion = makeAic2201HardwareEvents();
    capabilities.directEventCompletion = makeAic2201ExposedEvents();
    return;
  case CanonicalSyncCoreDomain::AIV:
    capabilities.hardwareEventCompletion = makeAiv2201HardwareEvents();
    capabilities.directEventCompletion = capabilities.hardwareEventCompletion;
    return;
  case CanonicalSyncCoreDomain::Unresolved:
  case CanonicalSyncCoreDomain::Conflict:
    return;
  }
}

void add2201HardwareHazards(CanonicalSyncTargetCapabilities &capabilities) {
  capabilities.crossPipeAccumulatorReadReadHazard.version = 1;
  capabilities.evidence.push_back(
      CanonicalSyncTargetEvidence::PTOASInsertSyncAccRarFfe46c09);
}

CanonicalSyncTargetCapabilities makeCoreTargetCapabilities(
    CanonicalSyncTargetProfile profile, CanonicalSyncCoreDomain coreDomain,
    bool vectorCompletionOrdered, bool vectorTargetedBarrierSupported) {
  CanonicalSyncTargetCapabilities capabilities;
  capabilities.profile = profile;
  capabilities.coreDomain = coreDomain;
  addCommonEvidence(capabilities);
  capabilities.sameResourceCompletionOrdering = makeResourceCapability(
      vectorCompletionOrdered
          ? std::initializer_list<PipelineType>{PipelineType::PIPE_S,
                                                PipelineType::PIPE_V}
          : std::initializer_list<PipelineType>{PipelineType::PIPE_S});
  capabilities.targetedBarrierDrainsSourcePrefix = makeResourceCapability(
      vectorTargetedBarrierSupported
          ? std::initializer_list<PipelineType>{PipelineType::PIPE_M,
                                                PipelineType::PIPE_MTE1,
                                                PipelineType::PIPE_MTE2,
                                                PipelineType::PIPE_MTE3,
                                                PipelineType::PIPE_FIX,
                                                PipelineType::PIPE_V}
          : std::initializer_list<PipelineType>{
                PipelineType::PIPE_M, PipelineType::PIPE_MTE1,
                PipelineType::PIPE_MTE2, PipelineType::PIPE_MTE3,
                PipelineType::PIPE_FIX});
  capabilities.legalPipeBarriers =
      capabilities.targetedBarrierDrainsSourcePrefix;
  // PTO's targeted barrier contract is intra-pipeline. No supported target
  // currently certifies that a naked source-pipe barrier publishes completion
  // to an independently issued operation on another pipe.
  capabilities.crossResourceTargetedBarrierCompletion = {};
  return capabilities;
}

} // namespace

CanonicalSyncTargetCapabilities
mlir::pto::canonical_sync_detail::getCanonicalSyncTargetCapabilities(
    func::FuncOp function) {
  const CanonicalSyncCoreDomain coreDomain = resolveCoreDomain(function);
  switch (resolvePTOInheritedTarget(function)) {
  case PTOTargetKind::A2: {
    CanonicalSyncTargetCapabilities capabilities =
        makeCoreTargetCapabilities(CanonicalSyncTargetProfile::A2V1, coreDomain,
                                   /*vectorCompletionOrdered=*/false,
                                   /*vectorTargetedBarrierSupported=*/true);
    capabilities.syncSpecVersion =
        CanonicalSyncTargetSyncSpecVersion::Ascend2201V1;
    add2201EventCapabilities(capabilities);
    add2201HardwareHazards(capabilities);
    return capabilities;
  }
  case PTOTargetKind::A2A3: {
    CanonicalSyncTargetCapabilities capabilities = makeCoreTargetCapabilities(
        CanonicalSyncTargetProfile::A2A3IntersectionV1, coreDomain,
        /*vectorCompletionOrdered=*/false,
        /*vectorTargetedBarrierSupported=*/true);
    capabilities.syncSpecVersion =
        CanonicalSyncTargetSyncSpecVersion::Ascend2201V1;
    add2201EventCapabilities(capabilities);
    add2201HardwareHazards(capabilities);
    return capabilities;
  }
  case PTOTargetKind::A3: {
    CanonicalSyncTargetCapabilities capabilities =
        makeCoreTargetCapabilities(CanonicalSyncTargetProfile::A3V1, coreDomain,
                                   /*vectorCompletionOrdered=*/false,
                                   /*vectorTargetedBarrierSupported=*/true);
    capabilities.syncSpecVersion =
        CanonicalSyncTargetSyncSpecVersion::Ascend2201V1;
    add2201EventCapabilities(capabilities);
    add2201HardwareHazards(capabilities);
    capabilities.targetCompletionResources = SyncCoverTargetCompletionResources{
        resourceId(PipelineType::PIPE_MTE1), resourceId(PipelineType::PIPE_M),
        resourceId(PipelineType::PIPE_FIX)};
    capabilities.mte1L0ReadySetCompletesPrefix.version = 1;
    capabilities.mL0AlternativeJoinSetCompletes.version = 1;
    capabilities.mte1ScopeExitSetCompletesPrefix.version = 1;
    capabilities.mToFixAccumulatorBoundaryCompletes.version = 1;
    capabilities.intrinsicMmadAccumulatorOrdering.version = 1;
    return capabilities;
  }
  case PTOTargetKind::A5: {
    CanonicalSyncTargetCapabilities capabilities =
        makeCoreTargetCapabilities(CanonicalSyncTargetProfile::A5V1, coreDomain,
                                   /*vectorCompletionOrdered=*/true,
                                   /*vectorTargetedBarrierSupported=*/false);
    capabilities.syncSpecVersion =
        CanonicalSyncTargetSyncSpecVersion::Ascend3510PartialV1;
    return capabilities;
  }
  case PTOTargetKind::Unspecified:
  case PTOTargetKind::Unsupported:
  case PTOTargetKind::Conflict:
    return {};
  }
  return {};
}
