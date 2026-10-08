// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINDISTANCES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINDISTANCES_H
#include <cstdint>
#include <optional>
namespace mlir::pto::frontiersynch {
// Bounds on possible iteration distances, not a claim that every distance in
// the interval occurs. Unreachable is distinct from reachable with no finite
// upper bound. minimum/maximum have meaning only when reachable is true.
struct OriginDistanceInterval {
    bool reachable = false;
    uint64_t minimum = 0;
    std::optional<uint64_t> maximum;
};
} // namespace mlir::pto::frontiersynch
#endif
