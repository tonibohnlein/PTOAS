// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_HANDOFFFACTS_H
#define PTO_TRANSFORMS_INSERTSYNC_HANDOFFFACTS_H

#include "PTO/Transforms/InsertSync/LifecycleSynthesis.h"
#include "llvm/Support/JSON.h"

namespace mlir::pto {
// Artifact identities refer to the immutable pass-entry function. Abstract
// guard-product nodes are never serialized as concrete loop occurrences.
struct HandoffFacts {
    llvm::json::Object record;
    bool supported = false;
};
HandoffFacts collectInsertSyncHandoffFacts(func::FuncOp function, const SyncIRs& ir,
                                         const InsertSyncLifecycleStructure& structure);
// Diagnostics only: inspect a clone, and leave payload, descriptors and pass
// behavior unchanged. An unsupported projection still produces a status record.
LogicalResult exportInsertSyncHandoffFacts(func::FuncOp function, StringRef directory,
                                          uint64_t workLimit = 8000000, StringRef bypassReason = "");
} // namespace mlir::pto
#endif
