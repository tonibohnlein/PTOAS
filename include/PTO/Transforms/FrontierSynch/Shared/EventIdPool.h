// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Physical encoding limits are independent of the caller-selected pool size.
#ifndef PTO_FRONTIERSYNCH_EVENTIDPOOL_H
#define PTO_FRONTIERSYNCH_EVENTIDPOOL_H
#include "llvm/ADT/ArrayRef.h"
#include <array>
#include <cstdint>
namespace mlir::pto::frontiersynch {
inline constexpr uint64_t NativeEventIdCount = 8;
// IDs 6/7 are framework-reserved in the documented programming contract.
// Explicitly listing them opts into an experimental deployment policy.
inline constexpr std::array<int64_t, 6> DefaultEventIds{0, 1, 2, 3, 4, 5};
inline llvm::ArrayRef<int64_t> defaultEventIds() { return DefaultEventIds; }
inline bool isNativeEventId(int64_t id)
{
    return id >= 0 && static_cast<uint64_t>(id) < NativeEventIdCount;
}
} // namespace mlir::pto::frontiersynch
#endif
