// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Allocation construction policy is immutable for one analysis session/export owner.
#ifndef PTO_FRONTIERSYNCH_ARITHMETICALLOCATIONPOLICY_H
#define PTO_FRONTIERSYNCH_ARITHMETICALLOCATIONPOLICY_H
namespace mlir::pto::frontiersynch {
// SingleId preserves the historical regional proof. CoordinateCandidate is
// an explicit construction investigation, not a capacity or fallback policy.
enum class ArithmeticRegionalAllocationPolicy { SingleId, CoordinateCandidate };
}
#endif
