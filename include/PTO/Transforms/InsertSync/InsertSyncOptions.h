// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_INSERTSYNCOPTIONS_H
#define PTO_TRANSFORMS_INSERTSYNC_INSERTSYNCOPTIONS_H
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <memory>
#include <string>
namespace mlir::pto {
inline constexpr uint64_t kDefaultLogicalSyncWorkBudget = 384000000;
struct InsertSyncOptions {
  std::string planner = "existing";
  uint64_t logicalWorkBudget = kDefaultLogicalSyncWorkBudget;
  std::string gmAlias;
  // Explicit source-qualified S7 contract; selecting an architecture alone
  // does not assert a release/toolchain's optional instruction guarantees.
  std::string hardwareContract = "conservative";
};
std::unique_ptr<Pass> createPTOInsertSyncPass(const InsertSyncOptions &options);
} // namespace mlir::pto
#endif
