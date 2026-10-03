// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite physical assignment of an already prepared explicit logical plan.
#ifndef PTO_FRONTIERSYNCH_EXPLICIT_PHYSICAL_EMISSION_H
#define PTO_FRONTIERSYNCH_EXPLICIT_PHYSICAL_EMISSION_H
#include "mlir/Support/LogicalResult.h"
namespace mlir::pto {
class SyncInput;
namespace frontiersynch {
class TraceDemandAnalysis;
class CostLedger;
struct DirectEmissionResult;
LogicalResult emitExplicitPhysical(const SyncInput&, const TraceDemandAnalysis&,
                                  DirectEmissionResult&, CostLedger&, bool finiteOneWayFamily = false);
} // namespace frontiersynch
} // namespace mlir::pto
#endif
