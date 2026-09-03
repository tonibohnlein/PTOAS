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
#include <iterator>

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
  capability.resourcePairs.erase(
      std::unique(capability.resourcePairs.begin(),
                  capability.resourcePairs.end()),
      capability.resourcePairs.end());
  return capability;
}

CanonicalSyncDirectedResourceCapability makeAiv2201Events() {
  using P = PipelineType;
  return makeDirectedCapability({
      {P::PIPE_S, P::PIPE_V},       {P::PIPE_S, P::PIPE_MTE2},
      {P::PIPE_S, P::PIPE_MTE3},    {P::PIPE_V, P::PIPE_S},
      {P::PIPE_V, P::PIPE_MTE2},    {P::PIPE_V, P::PIPE_MTE3},
      {P::PIPE_MTE2, P::PIPE_S},    {P::PIPE_MTE2, P::PIPE_V},
      {P::PIPE_MTE2, P::PIPE_MTE3}, {P::PIPE_MTE3, P::PIPE_S},
      {P::PIPE_MTE3, P::PIPE_V},    {P::PIPE_MTE3, P::PIPE_MTE2},
  });
}

CanonicalSyncDirectedResourceCapability makeAic2201Events() {
  using P = PipelineType;
  // Include combinations documented as implemented but having no current
  // programming scenario. Exclude cells explicitly marked not involved.
  return makeDirectedCapability({
      {P::PIPE_M, P::PIPE_MTE1},    {P::PIPE_M, P::PIPE_MTE2},
      {P::PIPE_M, P::PIPE_FIX},     {P::PIPE_MTE1, P::PIPE_M},
      {P::PIPE_MTE1, P::PIPE_MTE2}, {P::PIPE_MTE1, P::PIPE_MTE3},
      {P::PIPE_MTE1, P::PIPE_FIX},  {P::PIPE_MTE2, P::PIPE_M},
      {P::PIPE_MTE2, P::PIPE_MTE1}, {P::PIPE_MTE2, P::PIPE_MTE3},
      {P::PIPE_MTE2, P::PIPE_FIX},  {P::PIPE_MTE3, P::PIPE_MTE1},
      {P::PIPE_MTE3, P::PIPE_MTE2}, {P::PIPE_MTE3, P::PIPE_FIX},
      {P::PIPE_FIX, P::PIPE_M},     {P::PIPE_FIX, P::PIPE_MTE1},
      {P::PIPE_FIX, P::PIPE_MTE2},  {P::PIPE_FIX, P::PIPE_MTE3},
  });
}

enum class CoreDomain : std::uint8_t { Aic, Aiv, SharedOnly, Conflict };

CoreDomain resolveCoreDomain(func::FuncOp function,
                             ArrayRef<std::uint32_t> resources) {
  if (auto kind = function->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name)) {
    return kind.getKernelKind() == FunctionKernelKind::Cube ? CoreDomain::Aic
                                                            : CoreDomain::Aiv;
  }
  const auto has = [&](PipelineType resource) {
    return llvm::is_contained(resources, resourceId(resource));
  };
  const bool aic = has(PipelineType::PIPE_M) ||
                   has(PipelineType::PIPE_MTE1) ||
                   has(PipelineType::PIPE_FIX);
  const bool aiv = has(PipelineType::PIPE_V) || has(PipelineType::PIPE_V2);
  if (aic && aiv) {
    return CoreDomain::Conflict;
  }
  if (aic) {
    return CoreDomain::Aic;
  }
  if (aiv) {
    return CoreDomain::Aiv;
  }
  return CoreDomain::SharedOnly;
}

CanonicalSyncDirectedResourceCapability intersectCapabilities(
    const CanonicalSyncDirectedResourceCapability &first,
    const CanonicalSyncDirectedResourceCapability &second) {
  CanonicalSyncDirectedResourceCapability result;
  result.version = 1;
  std::set_intersection(first.resourcePairs.begin(), first.resourcePairs.end(),
                        second.resourcePairs.begin(),
                        second.resourcePairs.end(),
                        std::back_inserter(result.resourcePairs));
  return result;
}

CanonicalSyncTargetCapabilities
makeCoreTargetCapabilities(CanonicalSyncTargetProfile profile,
                           bool vectorCompletionOrdered,
                           bool vectorTargetedBarrierSupported) {
  CanonicalSyncTargetCapabilities capabilities;
  capabilities.profile = profile;
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
  switch (resolvePTOInheritedTarget(function)) {
  case PTOTargetKind::A2:
    return makeCoreTargetCapabilities(CanonicalSyncTargetProfile::A2V1,
                                      /*vectorCompletionOrdered=*/false,
                                      /*vectorTargetedBarrierSupported=*/true);
  case PTOTargetKind::A2A3:
    return makeCoreTargetCapabilities(
        CanonicalSyncTargetProfile::A2A3IntersectionV1,
        /*vectorCompletionOrdered=*/false,
        /*vectorTargetedBarrierSupported=*/true);
  case PTOTargetKind::A3: {
    CanonicalSyncTargetCapabilities capabilities =
        makeCoreTargetCapabilities(CanonicalSyncTargetProfile::A3V1,
                                   /*vectorCompletionOrdered=*/false,
                                   /*vectorTargetedBarrierSupported=*/true);
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
  case PTOTargetKind::A5:
    return makeCoreTargetCapabilities(CanonicalSyncTargetProfile::A5V1,
                                      /*vectorCompletionOrdered=*/true,
                                      /*vectorTargetedBarrierSupported=*/false);
  case PTOTargetKind::Unspecified:
  case PTOTargetKind::Unsupported:
  case PTOTargetKind::Conflict:
    return {};
  }
  return {};
}

void mlir::pto::canonical_sync_detail::
    configureCanonicalSyncDirectEventCompletion(
        func::FuncOp function, ArrayRef<std::uint32_t> resources,
        CanonicalSyncTargetCapabilities &capabilities) {
  switch (capabilities.profile) {
  case CanonicalSyncTargetProfile::A2V1:
  case CanonicalSyncTargetProfile::A2A3IntersectionV1:
  case CanonicalSyncTargetProfile::A3V1:
    break;
  case CanonicalSyncTargetProfile::A5V1:
  case CanonicalSyncTargetProfile::Unsupported:
    // The 2201 table is not evidence for 3510. Leave version zero until an
    // authoritative 3510 directed event contract is encoded.
    capabilities.directEventCompletion = {};
    return;
  }
  const CanonicalSyncDirectedResourceCapability aic = makeAic2201Events();
  const CanonicalSyncDirectedResourceCapability aiv = makeAiv2201Events();
  switch (resolveCoreDomain(function, resources)) {
  case CoreDomain::Aic:
    capabilities.directEventCompletion = aic;
    return;
  case CoreDomain::Aiv:
    capabilities.directEventCompletion = aiv;
    return;
  case CoreDomain::SharedOnly:
    capabilities.directEventCompletion = intersectCapabilities(aic, aiv);
    return;
  case CoreDomain::Conflict:
    capabilities.directEventCompletion.version = 1;
    capabilities.directEventCompletion.resourcePairs.clear();
    return;
  }
}
