// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared intra-core topology/mechanism facts. Documentary support is not a
// native-completion, intrinsic-binding, allocation or device certificate.
#ifndef PTO_IR_PTOSYNCCAPABILITIES_H
#define PTO_IR_PTOSYNCCAPABILITIES_H
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
namespace mlir {
class Operation;
namespace pto {
enum class PIPE : uint32_t;
enum class SyncPhysicalCore { Unknown, AIC, AIV, Conflict };
enum class SyncMechanismAvailability {
    Documented, Nonexistent, RecipeUnmet, Unqualified
};
struct SyncMechanismFact {
    SyncMechanismAvailability availability = SyncMechanismAvailability::Unqualified;
    llvm::StringRef source;
    llvm::StringRef obligation;
};
SyncPhysicalCore recoverSyncPhysicalCore(Operation* anchor);
std::optional<bool> getSyncPipelinePresence(llvm::StringRef architecture, SyncPhysicalCore core, PIPE pipe);
SyncMechanismFact getSyncEventFact(llvm::StringRef architecture, SyncPhysicalCore core, PIPE source, PIPE target);
SyncMechanismFact getSyncBarrierFact(llvm::StringRef architecture, SyncPhysicalCore core, PIPE pipe);
} // namespace pto
} // namespace mlir
#endif
