// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- GMAliasPolicy.h - Explicit GM argument alias contracts ---*- C++ -*-===//
//
// This policy supplies only disjointness promised by the caller. It never
// supplies completion or publication and does not change local alias rules.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_GMALIASPOLICY_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_GMALIASPOLICY_H

#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::pto::protocol_sync {

inline constexpr llvm::StringLiteral kSyncGMAliasContract = "pto.gm_alias";

std::optional<SyncGMAliasMode> parseSyncGMAliasMode(llvm::StringRef value);
llvm::StringRef stringifySyncGMAliasMode(SyncGMAliasMode mode);
FailureOr<SyncGMAliasMode> resolveSyncGMAliasMode(
    func::FuncOp function, std::optional<SyncGMAliasMode> overrideMode = std::nullopt);

struct SyncGMArgumentRoots {
    llvm::SmallVector<Value, 2> roots;
    bool complete = false;
};

/// Retain all possible roots through supported pointer/view and SCF forwarding.
/// Unknown definitions and trace limits forbid using the disjointness promise.
SyncGMArgumentRoots traceSyncGMArgumentRoots(func::FuncOp function, Value value);
bool haveDisjointGMArgumentRoots(const StructuredSyncIR& schedule, const SyncAccess& first, const SyncAccess& second);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_GMALIASPOLICY_H
