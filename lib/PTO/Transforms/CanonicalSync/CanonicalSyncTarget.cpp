// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncTarget.h"

#include "PTO/IR/PTOTypeUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

CanonicalPhysicalResource resource(CanonicalCore core, PIPE pipe) {
  return CanonicalPhysicalResource{core, pipe};
}

bool containsResource(ArrayRef<CanonicalPhysicalResource> resources,
                      CanonicalPhysicalResource resource) {
  return llvm::is_contained(resources, resource);
}

} // namespace

CanonicalSyncTarget mlir::pto::makeNpu2201CanonicalSyncTarget() {
  CanonicalSyncTarget target;
  const auto addEvent = [&target](CanonicalCore core, PIPE source,
                                  PIPE destination) {
    target.eventPairs.push_back(
        {resource(core, source), resource(core, destination)});
  };
  target.name = "npu2201-a2a3-v1";
  for (PIPE pipe : {PIPE::PIPE_M, PIPE::PIPE_MTE1, PIPE::PIPE_MTE2,
                    PIPE::PIPE_MTE3, PIPE::PIPE_FIX}) {
    target.resources.push_back(resource(CanonicalCore::AIC, pipe));
    target.barrierResources.push_back(resource(CanonicalCore::AIC, pipe));
  }
  for (PIPE pipe :
       {PIPE::PIPE_S, PIPE::PIPE_V, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3}) {
    target.resources.push_back(resource(CanonicalCore::AIV, pipe));
  }
  target.intrinsicCompletion.push_back(
      resource(CanonicalCore::AIV, PIPE::PIPE_S));
  for (PIPE pipe : {PIPE::PIPE_V, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3}) {
    target.barrierResources.push_back(resource(CanonicalCore::AIV, pipe));
  }

  addEvent(CanonicalCore::AIC, PIPE::PIPE_M, PIPE::PIPE_MTE1);
  addEvent(CanonicalCore::AIC, PIPE::PIPE_M, PIPE::PIPE_MTE2);
  addEvent(CanonicalCore::AIC, PIPE::PIPE_M, PIPE::PIPE_FIX);
  for (PIPE destination :
       {PIPE::PIPE_M, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3, PIPE::PIPE_FIX}) {
    addEvent(CanonicalCore::AIC, PIPE::PIPE_MTE1, destination);
  }
  for (PIPE destination :
       {PIPE::PIPE_M, PIPE::PIPE_MTE1, PIPE::PIPE_MTE3, PIPE::PIPE_FIX}) {
    addEvent(CanonicalCore::AIC, PIPE::PIPE_MTE2, destination);
  }
  for (PIPE destination : {PIPE::PIPE_MTE1, PIPE::PIPE_MTE2, PIPE::PIPE_FIX}) {
    addEvent(CanonicalCore::AIC, PIPE::PIPE_MTE3, destination);
  }
  for (PIPE destination :
       {PIPE::PIPE_M, PIPE::PIPE_MTE1, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3}) {
    addEvent(CanonicalCore::AIC, PIPE::PIPE_FIX, destination);
  }

  constexpr PIPE aivPipes[] = {PIPE::PIPE_S, PIPE::PIPE_V, PIPE::PIPE_MTE2,
                               PIPE::PIPE_MTE3};
  for (PIPE source : aivPipes) {
    for (PIPE destination : aivPipes) {
      if (source != destination) {
        addEvent(CanonicalCore::AIV, source, destination);
      }
    }
  }
  target.compilerEventIds = {0, 1, 2, 3, 4, 5};
  target.compilerCrossCoreEventIds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  return target;
}

FailureOr<CanonicalSyncTarget>
CanonicalSyncTarget::resolve(func::FuncOp function) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (module) {
    if (auto attribute =
            module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName)) {
      const StringRef name = attribute.getValue();
      const bool supported = name.equals_insensitive("a2") ||
                             name.equals_insensitive("a3") ||
                             name.equals_insensitive("a2a3");
      const bool knownA5 = name.equals_insensitive("a5");
      if (!supported && !knownA5) {
        function.emitError("canonical synchronization rejects unknown target "
                           "profile '")
            << name << "'";
        return failure();
      }
    }
    if (auto attribute = module->getAttrOfType<StringAttr>("pto.device-spec")) {
      const StringRef name = attribute.getValue();
      const bool knownA5 =
          name.starts_with("Ascend950") || name.starts_with("Ascend910_95");
      const bool supported = name.starts_with("Ascend910") && !knownA5;
      if (!supported && !knownA5) {
        function.emitError("canonical synchronization rejects unknown device "
                           "profile '")
            << name << "'";
        return failure();
      }
    }
  }
  const PTOArch architecture = getTargetArch(function.getOperation());
  switch (architecture) {
  case PTOArch::A3:
    return makeNpu2201CanonicalSyncTarget();
  case PTOArch::A5:
    function.emitError("canonical synchronization supports only the verified "
                       "A2/A3 NPU 2201 target model; A5 is unsupported");
    return failure();
  default:
    function.emitError(
        "canonical synchronization rejects an unsupported target profile");
    return failure();
  }
}

bool CanonicalSyncTarget::supportsResource(
    CanonicalPhysicalResource resource) const {
  return containsResource(resources, resource);
}

bool CanonicalSyncTarget::hasIntrinsicCompletion(
    CanonicalPhysicalResource resource) const {
  return containsResource(intrinsicCompletion, resource);
}

bool CanonicalSyncTarget::supportsPipeBarrier(
    CanonicalPhysicalResource resource) const {
  return containsResource(barrierResources, resource);
}

bool CanonicalSyncTarget::supportsEvent(
    CanonicalPhysicalResource source, CanonicalPhysicalResource target) const {
  return source.core == target.core &&
         llvm::is_contained(eventPairs, std::make_pair(source, target));
}

bool CanonicalSyncTarget::supportsCrossCoreEvent(
    CanonicalPhysicalResource source, CanonicalPhysicalResource target) const {
  return source.core != target.core && source.pipe != PIPE::PIPE_S &&
         target.pipe != PIPE::PIPE_S && supportsResource(source) &&
         supportsResource(target);
}

FailureOr<SmallVector<CanonicalPhysicalResource, 8>>
CanonicalSyncTarget::getFenceDrainedResources(Operation *fence) const {
  const std::optional<bool> vectorKernel = resolvePTOExecutionVector(fence);
  if (!vectorKernel) {
    return failure();
  }
  const CanonicalCore executionCore =
      *vectorKernel ? CanonicalCore::AIV : CanonicalCore::AIC;
  SmallVector<CanonicalPhysicalResource, 8> drained;
  for (CanonicalPhysicalResource resource : resources) {
    if (resource.core != executionCore) {
      continue;
    }
    const bool supportedPipe =
        *vectorKernel || resource.pipe == PIPE::PIPE_MTE2 ||
        resource.pipe == PIPE::PIPE_MTE3 || resource.pipe == PIPE::PIPE_FIX;
    if (supportedPipe) {
      drained.push_back(resource);
    }
  }
  return drained;
}
