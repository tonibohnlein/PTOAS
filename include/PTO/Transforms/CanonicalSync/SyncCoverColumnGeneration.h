// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverColumnGeneration.h - Cover candidate factories -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNC_COVER_COLUMN_GENERATION_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNC_COVER_COLUMN_GENERATION_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverTargetCapabilities.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

struct SyncCoverColumnGenerationOptions {
  std::size_t maximumCandidates = 4096;
  std::size_t maximumInspections = 65536;
  std::size_t maximumSupplyEdges = 262144;
};

struct SyncCoverColumnGenerationContext {
  const SyncCoverTargetCapabilities &target;
  std::vector<SyncCoverDemandId> activeDemands;
  SyncCoverColumnGenerationOptions options;
};

struct SyncCoverColumnGeneratorReport {
  std::string generator;
  std::size_t candidates = 0;
  std::size_t admitted = 0;
  std::size_t rejectedByVerifier = 0;
  std::size_t skippedByCapability = 0;
  std::size_t inspections = 0;
  std::size_t supplyEdges = 0;
  bool truncated = false;
};

class SyncCoverColumnGenerator {
public:
  virtual ~SyncCoverColumnGenerator() = default;
  virtual const char *name() const = 0;
  virtual SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const = 0;
};

struct SyncCoverColumnGenerationResult {
  enum class Error : std::uint8_t {
    None,
    InvalidUniverse,
    InvalidOptions,
    InvalidTarget,
  };

  Error error = Error::None;
  std::vector<SyncCoverColumnGeneratorReport> reports;
  std::size_t totalAdmitted = 0;
  bool truncated = false;

  explicit operator bool() const { return error == Error::None; }
};

SyncCoverColumnGenerationResult runSyncCoverColumnGenerators(
    const SyncCoverColumnGenerationContext &context,
    SyncCoverMechanismUniverse &universe,
    const std::vector<std::unique_ptr<SyncCoverColumnGenerator>> &generators);

std::unique_ptr<SyncCoverColumnGenerator>
makeSyncCoverCanonicalEventGenerator();
std::unique_ptr<SyncCoverColumnGenerator>
makeSyncCoverMergedPrefixEventGenerator();
std::unique_ptr<SyncCoverColumnGenerator>
makeSyncCoverPiercedBarrierGenerator();

bool verifySyncCoverMergedPrefixEvent(
    const SyncCoverGraph &graph, const SyncCoverResourceDomain &domain,
    const SyncCoverMechanismDescriptor &descriptor,
    const SyncCoverTargetCapabilities &target);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNC_COVER_COLUMN_GENERATION_H
