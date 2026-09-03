// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTimelineInternal.h - Timeline access grouping ---*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETIMELINEINTERNAL_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETIMELINEINTERNAL_H

#include "PTO/Transforms/ProtocolSync/StorageTimeline.h"

namespace mlir::pto::protocol_sync::detail {

struct StorageAccessClass {
    SyncStorageFamilyId family = kInvalidSyncId;
    unsigned addressSpace = 0;
    llvm::SmallVector<SyncByteInterval, 2> slice;
    std::optional<SyncSlotExpression> slot;
    llvm::SmallVector<SyncAccessId, 8> accesses;
    bool unknownRange = false;
    bool aliasesUnknownRange = false;
    bool partialOverlap = false;
    bool conflictingPhysicalRange = false;
    bool inPlace = false;
    bool multipleGenerations = false;
};

bool sameTimelineIntervals(llvm::ArrayRef<SyncByteInterval> first, llvm::ArrayRef<SyncByteInterval> second);
bool sameTimelineSlotClass(
    const std::optional<SyncSlotExpression>& first, const std::optional<SyncSlotExpression>& second);
llvm::SmallVector<StorageAccessClass, 16> buildGenerationAccessClasses(
    const StructuredSyncIR& schedule, ProtocolSyncStatistics* statistics);

} // namespace mlir::pto::protocol_sync::detail

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETIMELINEINTERNAL_H
