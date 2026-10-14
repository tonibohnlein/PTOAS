// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical and guarded points share one coactive-order inventory.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_MIXEDENDPOINTPIECES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_MIXEDENDPOINTPIECES_H
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
namespace mlir::pto::frontiersynch {
// Only qualified numerical points can merge. The producer establishes exact
// numerical activity separately; coordinates here prove exclusion only.
EndpointPieces buildMixedEndpointPieces(ArrayRef<EndpointPiece> points, ArrayRef<uint8_t> numerical);
} // namespace mlir::pto::frontiersynch
#endif
