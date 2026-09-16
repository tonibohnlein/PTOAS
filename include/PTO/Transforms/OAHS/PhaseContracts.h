// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_PHASE_CONTRACTS_H
#define PTO_TRANSFORMS_OAHS_PHASE_CONTRACTS_H
#include "PTO/IR/SyncTargetProfile.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace mlir::pto::oahs {
  // A SUPPLIED semantic profile. These declarations do not select UnitFlag modes
  // or establish that any native lowering implements the modeled service order.
  struct FinalBlockGroup {
    std::string name;
    ::mlir::pto::SyncPipe producer = ::mlir::pto::SyncPipe::M;
    ::mlir::pto::SyncPipe consumer = ::mlir::pto::SyncPipe::FIX;
    std::vector<unsigned> blocks;
  };
  struct FinalBlockProfile {
    std::string name = "final-block-pair-v1";
    std::string qualification = "reference-contract-only";
    std::vector<FinalBlockGroup> groups;
    // All premises default false. The supplied model must state them explicitly.
    bool entryWritable = false;
    bool orderedPerBlockService = false;
    bool enabledPhaseProgress = false;
    bool commonForwardControl = false;
  };
  struct FinalBlockOperation {
    enum class Role : unsigned {
      Producer, Consumer
    }
    role = Role::Producer;
    std::size_t group = 0;
    std::vector<unsigned> blocks;
    // Producer: exactly LEFT, RIGHT in order. Consumer: one GM output per block.
    std::vector<unsigned> operands, outputs;
    std::string mode = "final";
    std::string layout = "exact-block-map-v1";
  };
  struct PhaseResourceObligation {
    enum Kind {
      WrongRole, UnconsumedBlock, UnestablishedPermission
    }
    kind = WrongRole;
    std::size_t cut = 0, operation = 0;
    unsigned cell = 0;
    std::string reason;
  };
  struct PhaseResourceFacts {
    unsigned cell = 0;
    // may mask: 1 writable, 2 readable; 3 means an invalid/unproved permission.
    unsigned possiblePermission = 1;
    // Whole-operation and latest-event-consumption facts at the CURRENT anchor.
    // The permission mask itself implies none of these causal facts.
    std::vector<uint8_t> completedOperations, carriedConsumptions;
  };
}
// namespace mlir::pto::oahs
#endif
