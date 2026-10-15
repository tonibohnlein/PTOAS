// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Live matching coordinates of an already certified complete physical map.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALMAPINPUTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALMAPINPUTS_H
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
namespace mlir::pto::frontiersynch {
// Unused tuple positions preserve the schema but require no emitted SSA.
// Coordinate-selected base/phase recipes use original coordinates directly.
bool physicalMapNeedsCoordinate(const PhysicalEndpointMap& mapping, uint64_t coordinate);
} // namespace mlir::pto::frontiersynch
#endif
