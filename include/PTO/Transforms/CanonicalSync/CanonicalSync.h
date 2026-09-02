// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncModel.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace mlir {
namespace pto {

struct CanonicalSyncOptions {
  bool analysisOnly = false;
  bool dump = false;
  bool statistics = false;
  CanonicalGmAliasPolicy gmAliasPolicy = CanonicalGmAliasPolicy::Conservative;
  CanonicalStructuralCoverFamilies structuralCoverFamilies = 0U;
};

std::optional<CanonicalGmAliasPolicy>
parseCanonicalGmAliasPolicy(llvm::StringRef value);
std::optional<CanonicalStructuralCoverFamilies>
parseCanonicalStructuralCoverFamilies(llvm::StringRef value);
std::string stringifyCanonicalStructuralCoverFamilies(
    CanonicalStructuralCoverFamilies families);

FailureOr<std::unique_ptr<CanonicalSyncProgram>> buildCanonicalSyncProgram(
    func::FuncOp function,
    CanonicalGmAliasPolicy gmAliasPolicy = CanonicalGmAliasPolicy::Conservative,
    CanonicalSyncStatistics *statistics = nullptr);
LogicalResult buildCanonicalDirectMechanisms(CanonicalSyncProgram &program);
LogicalResult evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program);
LogicalResult buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program);
LogicalResult solveCanonicalSyncSetCover(CanonicalSyncProgram &program);
LogicalResult allocateCanonicalSyncEvents(CanonicalSyncProgram &program);
LogicalResult materializeAndVerifyCanonicalSync(CanonicalSyncProgram &program);
LogicalResult runCanonicalSync(func::FuncOp function,
                               const CanonicalSyncOptions &options = {});

std::unique_ptr<Pass>
createPTOCanonicalSyncPass(const CanonicalSyncOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
