// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ROTATIONPATTERN_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ROTATIONPATTERN_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
namespace mlir::pto::frontiersynch::detail {
struct SlotPattern {
    uint64_t stride = 0;
    uint64_t offset = 0;
    bool arithmeticProven = true;
};
std::optional<SlotPattern> matchSlot(Value slot, Value induction, uint64_t count);
std::optional<std::pair<uint64_t, uint64_t>> withinSlotRange(const SyncStorageEffect& effect,
                                                           const SyncInput& input, uint64_t bytes);
} // namespace mlir::pto::frontiersynch::detail
#endif
