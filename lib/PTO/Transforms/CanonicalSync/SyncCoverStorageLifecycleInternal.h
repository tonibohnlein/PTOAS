// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGELIFECYCLEINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGELIFECYCLEINTERNAL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageLifecycle.h"

namespace mlir {
namespace pto {

class SyncCoverStorageLifecycleWorkBudget {
public:
  SyncCoverStorageLifecycleWorkBudget(std::size_t limit, std::size_t &used)
      : limit_(limit), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (amount > limit_ - used_) {
      failed_ = true;
      return false;
    }
    used_ += amount;
    return true;
  }

  bool failed() const { return failed_; }

private:
  std::size_t limit_;
  std::size_t &used_;
  bool failed_ = false;
};

SyncCoverStorageLifecycleError buildSyncCoverStorageLifecycleSccs(
    SyncCoverStorageLifecycleComponent &component,
    const SyncCoverStorageLifecycleLimits &limits,
    SyncCoverStorageLifecycleStatistics &statistics,
    SyncCoverStorageLifecycleWorkBudget &workBudget);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSTORAGELIFECYCLEINTERNAL_H
