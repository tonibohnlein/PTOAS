// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Invocation-wide publication consumes certified incoming ordinals; no demand analysis.
// Shared protocol adapter after occurrence/availability interpretation.
#ifndef PTO_FRONTIERSYNCH_PREFIX_DEMAND_H
#define PTO_FRONTIERSYNCH_PREFIX_DEMAND_H
#include "InvocationPrefixCollection.h"
namespace mlir::pto::frontiersynch {
// active is exact at the original consumer. sourceRank is the actual captured
// source identity, needed only across pipes. This helper never reconstructs
// occurrence coordinates, mathematical guards or synchronization ownership.
// The producer-owned sourceRank callback reads captured slots and emits pure
// scalar selection only. It is qualified once during construction, including
// for a dynamic inactive guard; its emitted code may execute inside that guard.
LogicalResult appendPrefixIncoming(
    InvocationPrefixCollection& collection, Operation* consumer, PrefixDirection direction,
    PrefixIncomingSelector active, PrefixIncomingSelector sourceRank, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
