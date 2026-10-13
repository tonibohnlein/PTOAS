// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_PREPAREDCONTROLLIVENESS_H
#define PTO_FRONTIERSYNCH_PREPAREDCONTROLLIVENESS_H
#include "llvm/ADT/DenseSet.h"
namespace mlir {
class Operation;
namespace pto::frontiersynch {
struct PreparedLogicalPlan;
// Generated SSA plus non-SSA evidence roots. Pending allocation callbacks keep
// the entire preparation alive because their captured Values are opaque.
// The endpoint-only view is an existing diagnostic, never a deletion criterion.
llvm::DenseSet<Operation*> livePreparedControl(const PreparedLogicalPlan& plan, bool includeEvidence = true);
} // namespace pto::frontiersynch
} // namespace mlir
#endif
