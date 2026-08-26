// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverTargetCapabilities.h - Sync target model -------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNC_COVER_TARGET_CAPABILITIES_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNC_COVER_TARGET_CAPABILITIES_H

#include <cstdint>
#include <set>
#include <string>

namespace mlir {
namespace pto {

enum class SyncCoverEvidenceLevel : std::uint8_t {
  None,
  VendorKernels,
  Measured,
  Documented,
};

/// Target facts that may change which synchronization mechanisms are sound.
/// Resource identifiers deliberately remain independent of PipelineType so
/// the covering core and its unit tests stay MLIR-free.
struct SyncCoverTargetCapabilities {
  std::string name;
  std::set<std::uint32_t> prefixSetResources;
  SyncCoverEvidenceLevel prefixEvidence = SyncCoverEvidenceLevel::None;
  std::set<std::uint32_t> hardwareCompletionResources;
  unsigned eventIdBudget = 8;

  bool hasPrefixSetSemantics(std::uint32_t resource) const {
    return prefixEvidence != SyncCoverEvidenceLevel::None &&
           prefixSetResources.count(resource) != 0;
  }

  bool hasHardwareCompletion(std::uint32_t resource) const {
    return hardwareCompletionResources.count(resource) != 0;
  }
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNC_COVER_TARGET_CAPABILITIES_H
