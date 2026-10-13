// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Executable expression recipes with invocation-owned historical evaluations.
#ifndef PTO_FRONTIERSYNCH_PREFIXEXPRESSION_H
#define PTO_FRONTIERSYNCH_PREFIXEXPRESSION_H
#include "InvocationPrefixCollection.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
namespace mlir::pto::frontiersynch {
// Layout snapshots before state materialization. Emission/proofs remain lazy.
// Unknown availability refuses the recipe; no opaque producer is replayed.
PrefixIncomingSelector preparePrefixExpression(std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id root, Operation* cut, InvocationPrefixCollection& collection);
} // namespace mlir::pto::frontiersynch
#endif
