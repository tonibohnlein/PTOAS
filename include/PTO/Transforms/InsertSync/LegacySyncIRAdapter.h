// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LegacySyncIRAdapter.h - ProtocolSync shadow comparison ---*- C++ -*-===//

#ifndef PTO_TRANSFORMS_INSERTSYNC_LEGACYSYNCIRADAPTER_H
#define PTO_TRANSFORMS_INSERTSYNC_LEGACYSYNCIRADAPTER_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace mlir::pto::protocol_sync {

struct LegacySyncSnapshot {
    SyncIRs syncIR;
    Buffer2MemInfoMap storage;
};

struct LegacySyncParityMismatch {
    std::string category;
    std::uint32_t index = 0;
    std::string detail;
};

struct LegacySyncParityResult {
    llvm::SmallVector<LegacySyncParityMismatch, 4> mismatches;
    llvm::SmallVector<LegacySyncParityMismatch, 4> internalConsistencyIssues;

    bool matches() const { return mismatches.empty(); }
    bool isInternallyConsistent() const { return internalConsistencyIssues.empty(); }
};

class LegacySyncIRAdapter {
public:
    LogicalResult buildSnapshot(func::FuncOp function, LegacySyncSnapshot& snapshot) const;
    SyncSemanticContext buildSemanticContext(const LegacySyncSnapshot& snapshot) const;
    LegacySyncParityResult compare(const LegacySyncSnapshot& legacy, const StructuredSyncIR& schedule) const;
};

void printLegacySyncParity(func::FuncOp function, const LegacySyncParityResult& result, llvm::raw_ostream& output);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_INSERTSYNC_LEGACYSYNCIRADAPTER_H
