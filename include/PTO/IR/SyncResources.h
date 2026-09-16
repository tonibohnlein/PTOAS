// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_IR_SYNCRESOURCES_H
#define PTO_IR_SYNCRESOURCES_H
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::pto {
// An effect annotation supplied by the operation implementation, not an IR
// attribute trusted from callers. Descriptor accesses are synchronous in the
// original issue order. Retain DefaultResource so generic compiler passes keep
// their existing conservative aliasing with tile command operands.
inline StringAttr tileDescriptorEffect(MLIRContext *context) {
  return StringAttr::get(context, "pto.issue_ordered_tile_descriptor");
}
inline bool isTileDescriptorEffect(const MemoryEffects::EffectInstance &effect) {
  auto tag = dyn_cast_or_null<StringAttr>(effect.getParameters());
  return effect.getResource() == SideEffects::DefaultResource::get() && tag &&
         tag.getValue() == "pto.issue_ordered_tile_descriptor";
}
} // namespace mlir::pto
#endif
