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
