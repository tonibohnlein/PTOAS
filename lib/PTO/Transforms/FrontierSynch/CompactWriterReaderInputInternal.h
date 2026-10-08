// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTWRITERREADERINPUTINTERNAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTWRITERREADERINPUTINTERNAL_H
#include "PTO/Transforms/FrontierSynch/CompactWriterReaderInput.h"
namespace mlir::pto::frontiersynch::detail {
// Caller supplies the structurally proved fixed slot sequence: each slot has
// exactly one executed alternative, all with its common pipe. Every original
// phase in the loop belongs to exactly one slot. This is a shared extraction
// core, not an independent structural recognizer or geometry recovery path.
CompactWriterReaderInput buildCompactWriterReaderSlotInput(scf::ForOp loop,
    const SyncInput& input, const PhaseIndex& index,
    llvm::ArrayRef<std::vector<const CompoundInstanceElement*>> slots,
    const CompactWriterReaderBindings& bindings);
} // namespace mlir::pto::frontiersynch::detail
#endif
