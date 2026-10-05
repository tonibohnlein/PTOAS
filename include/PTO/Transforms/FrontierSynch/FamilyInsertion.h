// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Prepare structured logical endpoint families directly from analysis coordinates.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYINSERTION_H
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
namespace mlir::pto::frontiersynch {
// The caller preflights cuts, bounds, 64-bit index arithmetic, and correspondence
// of prepared.families to plan. Arithmetic is staged in detached blocks; failure
// leaves the original function unchanged. Source and destination pieces select
// the same original record identity, independently of their piece partitions.
LogicalResult prepareFamilyEndpointCode(const NumericTemplateEndpoints& plan, PreparedLogicalPlan& prepared);
} // namespace mlir::pto::frontiersynch
#endif
