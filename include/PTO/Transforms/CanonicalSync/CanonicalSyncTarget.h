// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCTARGET_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCTARGET_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncModel.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace pto {

struct CanonicalEventDomain {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  llvm::SmallVector<unsigned, 6> reservedIds;
};

class CanonicalSyncTarget {
public:
  static FailureOr<CanonicalSyncTarget> resolve(func::FuncOp function);

  bool supportsResource(CanonicalPhysicalResource resource) const;
  bool hasIntrinsicCompletion(CanonicalPhysicalResource resource) const;
  bool supportsPipeBarrier(CanonicalPhysicalResource resource) const;
  bool supportsEvent(CanonicalPhysicalResource source,
                     CanonicalPhysicalResource target) const;
  FailureOr<llvm::SmallVector<CanonicalPhysicalResource, 8>>
  getFenceDrainedResources(Operation *fence) const;
  llvm::ArrayRef<unsigned> getCompilerEventIds() const {
    return compilerEventIds;
  }
  llvm::StringRef getName() const { return name; }

private:
  std::string name;
  llvm::SmallVector<CanonicalPhysicalResource, 10> resources;
  llvm::SmallVector<CanonicalPhysicalResource, 8> intrinsicCompletion;
  llvm::SmallVector<CanonicalPhysicalResource, 8> barrierResources;
  llvm::SmallVector<
      std::pair<CanonicalPhysicalResource, CanonicalPhysicalResource>, 24>
      eventPairs;
  llvm::SmallVector<unsigned, 6> compilerEventIds;

  friend CanonicalSyncTarget makeNpu2201CanonicalSyncTarget();
};

CanonicalSyncTarget makeNpu2201CanonicalSyncTarget();

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCTARGET_H
