// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
namespace mlir::pto {
SyncInput::~SyncInput() = default;
LogicalResult SyncInput::build(func::FuncOp function) {
  phases.clear();
  nodes.clear();
  storage.clear();
  if (function.isDeclaration()) {
    return success();
  }
  PTOIRTranslator translator(nodes, storage, function, SyncAnalysisMode::NORMALSYNC);
  if (failed(translator.Build())) {
    return failure();
  }
  for (const auto &node : nodes) {
    if (const auto *phase = dyn_cast<CompoundInstanceElement>(node.get())) {
      phases.push_back(phase);
    }
  }
  return success();
}
} // namespace mlir::pto
