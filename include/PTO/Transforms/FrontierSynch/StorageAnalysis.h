// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Storage facts only. All phases and memory records are borrowed from SyncInput.
// The input and its unchanged source operations must outlive these results.
// Program/control structure remains in MLIR; this analysis does not reconstruct it.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_STORAGEANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_STORAGEANALYSIS_H

#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"

namespace mlir::pto::frontiersynch {
struct StorageAccess {
    const CompoundInstanceElement* phase = nullptr;
    bool read = false;
    bool write = false;
};
struct StorageFootprint {
    const BaseMemInfo* memory = nullptr;
    SmallVector<StorageAccess> accesses;
    std::optional<std::size_t> sharedMemory = std::nullopt;
};
struct StorageAlias {
    std::size_t first = 0;
    std::size_t second = 0;
};
class StorageAnalysis {
public:
    explicit StorageAnalysis(const SyncInput& input);
    StorageAnalysis(const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases);
    ArrayRef<StorageFootprint> footprints() const { return footprintFacts; }
    ArrayRef<StorageAlias> aliases() const { return aliasFacts; }
    std::size_t aliasQueries() const { return queryCount; }
    // Cells partition shared may-access bounds; they never authorize overwrite kills.
    const SyncStorageBounds& bounds() const { return *storageBoundsInfo; }

private:
    SmallVector<StorageFootprint> footprintFacts;
    SmallVector<StorageAlias> aliasFacts;
    std::size_t queryCount = 0;
    const SyncStorageBounds* storageBoundsInfo = nullptr;
};
} // namespace mlir::pto::frontiersynch
#endif
