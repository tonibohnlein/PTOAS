// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Cell partition of shared translated memory bounds. This never extracts or
// qualifies instruction effects; access incidences come from shared SyncIR.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSTORAGEBOUNDS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSTORAGEBOUNDS_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <cstdint>
#include <optional>
namespace mlir::pto {
class CompoundInstanceElement;
struct BaseMemInfo;
// Bounds describe may-access cells, never definite full-cell overwrites.
enum class SyncBoundStatus { ConstantPhysical, UnknownCoordinates, AmbiguousAddresses, WrappedInterval };
struct SyncBoundCell {
    unsigned addressSpace = 0;
    uint64_t begin = 0;
    uint64_t end = 0;
};
struct SyncCapturedMemory {
    std::size_t id = 0;
    unsigned addressSpace = 0;
    llvm::SmallVector<uint64_t> boundingAddresses;
    uint64_t boundingSize = 0;
    bool knownPhysicalAddresses = false;
    bool aliasesUnknownRange = false;
    SyncBoundStatus bounds = SyncBoundStatus::UnknownCoordinates;
    llvm::SmallVector<std::size_t> boundCells;
};
class SyncStorageBounds {
public:
    void clear();
    void capture(llvm::ArrayRef<const CompoundInstanceElement*> phases);
    llvm::ArrayRef<SyncCapturedMemory> memory() const { return memoryRecords; }
    llvm::ArrayRef<SyncBoundCell> cells() const { return boundCells; }
    std::optional<std::size_t> memoryIdentity(const BaseMemInfo* memory) const;
private:
    llvm::SmallVector<SyncCapturedMemory> memoryRecords;
    llvm::SmallVector<SyncBoundCell> boundCells;
    llvm::DenseMap<const BaseMemInfo*, std::size_t> memoryIdentities;
    void normalizeBounds();
};
} // namespace mlir::pto
#endif
