// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Build exact endpoint-family selectors directly from original loop coordinates.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYEXPRESSIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYEXPRESSIONS_H
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
#include "mlir/IR/Builders.h"
#include <string>
namespace mlir::pto::frontiersynch {
struct FamilyExpressions {
    std::string error;
    Value present;
    Value member; // Selected label (default: list ordinal); meaningful only when present.
};
// The caller establishes a 64-bit index layout and availability of every loop
// induction value at the insertion point. Each tuple uses the same ordered loop
// schema. Literal lower/step bounds permit lattice normalization; otherwise
// exact raw induction coordinates preserve environment-evaluated nested bounds.
// Invalid inputs return an error without creating IR. Empty input denotes no
// members; a singleton empty tuple denotes one unconditional member. Emitted
// arithmetic is total even outside the exact member domain. Optional labels map
// each coordinate tuple to its original record identity without requiring paired
// endpoints to have the same piece partition. An empty label list means ordinals.
// A nonzero modulus (at most six) selects labels modulo that modulus and permits
// verified modular affine expressions; zero retains the exact-label mapping.
FamilyExpressions emitFamilyExpressions(OpBuilder& builder, Location location,
    ArrayRef<SmallVector<TemplateCoordinate>> members, ArrayRef<int64_t> labels = {}, uint64_t modulus = 0);
} // namespace mlir::pto::frontiersynch
#endif
