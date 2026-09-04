// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- ProtocolSyncTarget.h - ProtocolSync target contract -------*- C++ -*-===//
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_PROTOCOLSYNCTARGET_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_PROTOCOLSYNCTARGET_H

#include "PTO/Transforms/ProtocolSync/SyncSemantics.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <utility>

namespace mlir::pto::protocol_sync {

enum class SyncOneShotTargetKind : std::uint8_t { Npu2201A3, Unsupported };

struct SyncOneShotResource {
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    bool operator==(const SyncOneShotResource& other) const { return core == other.core && pipe == other.pipe; }
};

class ProtocolSyncTarget {
public:
    static ProtocolSyncTarget resolve(func::FuncOp function);
    bool isSupported() const { return kind == SyncOneShotTargetKind::Npu2201A3; }
    SyncOneShotTargetKind getKind() const { return kind; }
    llvm::StringRef getName() const { return name; }
    llvm::StringRef getUnsupportedReason() const { return unsupportedReason; }
    bool supportsPipeBarrier(SyncOneShotResource resource) const;
    bool supportsEvent(SyncOneShotResource source, SyncOneShotResource target) const;
    llvm::ArrayRef<unsigned> getCompilerEventIds() const { return compilerEventIds; }

private:
    SyncOneShotTargetKind kind = SyncOneShotTargetKind::Unsupported;
    std::string name = "unsupported";
    std::string unsupportedReason = "unsupported target";
    llvm::SmallVector<SyncOneShotResource, 8> barrierResources;
    llvm::SmallVector<std::pair<SyncOneShotResource, SyncOneShotResource>, 24> eventPairs;
    llvm::SmallVector<unsigned, 6> compilerEventIds;
};

llvm::StringRef stringifySyncOneShotTargetKind(SyncOneShotTargetKind kind);

} // namespace mlir::pto::protocol_sync
#endif // PTO_TRANSFORMS_PROTOCOLSYNC_PROTOCOLSYNCTARGET_H
