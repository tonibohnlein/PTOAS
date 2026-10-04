// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_RECOGNITIONINTERNAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_RECOGNITIONINTERNAL_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
namespace mlir::pto::frontiersynch::detail {
bool entryExpression(Value value, Operation* entry, const PhaseIndex& index,
                     SmallVectorImpl<Operation*>& recipe);
void normalizeFragments(RecognitionResult& result, const SyncStorageEffects& effects);
void inspectLeaf(Operation& op, const PhaseIndex& index, RecognitionResult& result);
bool checkRotatingDomain(scf::ForOp loop, RecognitionResult& result, bool canonical = false);
void inspectRotatingPhases(scf::ForOp loop, ArrayRef<const CompoundInstanceElement*> phases,
                          const SyncInput& input, const SyncStorageEffects& effects, RecognitionResult& result,
                          const PhaseIndex& index, bool allowParameters = false);
} // namespace mlir::pto::frontiersynch::detail
#endif
