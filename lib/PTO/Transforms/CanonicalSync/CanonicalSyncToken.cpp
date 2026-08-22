// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"

using namespace mlir;
using namespace mlir::pto;

bool mlir::pto::verifySyncTokenTrace(
    unsigned laneCount, const std::vector<unsigned> &initiallyFullLanes,
    const std::vector<SyncTokenAction> &actions,
    const std::vector<unsigned> &expectedFullLanes) {
  if (laneCount == 0) {
    return false;
  }
  std::vector<bool> full(laneCount, false);
  std::vector<bool> seen(laneCount, false);
  for (unsigned lane : initiallyFullLanes) {
    if (lane >= laneCount || seen[lane]) {
      return false;
    }
    seen[lane] = true;
    full[lane] = true;
  }
  for (const SyncTokenAction &action : actions) {
    if (action.lane >= laneCount) {
      return false;
    }
    const bool isSet = action.kind == SyncTokenActionKind::Set;
    if (full[action.lane] == isSet) {
      return false;
    }
    full[action.lane] = isSet;
  }

  std::vector<bool> expected(laneCount, false);
  seen.assign(laneCount, false);
  for (unsigned lane : expectedFullLanes) {
    if (lane >= laneCount || seen[lane]) {
      return false;
    }
    seen[lane] = true;
    expected[lane] = true;
  }
  return full == expected;
}
