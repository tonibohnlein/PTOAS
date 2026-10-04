// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Precision of the supplied phase effects, independent of dependence analysis.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSTORAGEEFFECTS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSTORAGEEFFECTS_H

#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncAccessRegion.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::pto {
enum class SyncAccessMode { Read, Write };
enum class SyncAccessPrecision { Exact, UpperBound, Unknown };

// Half-open physical byte interval. Space is part of identity; SSA roots are not.
struct SyncStorageCell {
    AddressSpace space;
    uint64_t begin = 0;
    uint64_t end = 0;
};

struct SyncStorageEffect {
    const CompoundInstanceElement* phase = nullptr;
    const BaseMemInfo* memory = nullptr;
    SyncAccessMode mode = SyncAccessMode::Read;
    SyncAccessPrecision precision = SyncAccessPrecision::Unknown;
    // Buffer geometry is retained independently from actual access precision.
    std::optional<SyncAccessRegion> descriptorRegion;
    std::optional<SyncAccessRegion> region;
    // Empty only when the shared contract and physical mapping are exact.
    std::string precisionReason;
    // Whether ranges enumerate the exact set, rather than its capacity bound.
    bool exactRanges = false;
    SmallVector<SyncStorageCell> ranges;
    SmallVector<std::size_t> cells;

    // Exact writes overwrite every cell in this effect. Upper bounds never kill
    // old writers/readers, even when their bounding cells match exactly.
    bool hasDefiniteWrites() const
    {
        return mode == SyncAccessMode::Write && precision == SyncAccessPrecision::Exact;
    }
};

class SyncStorageEffects {
public:
    // Borrows the unchanged SyncInput and source MLIR. Does not unfold control,
    // enumerate conflicting pairs, insert commands, or establish completeness of
    // the upstream operation-effect registry. Failure publishes no partial state.
    LogicalResult build(const SyncInput& input);
    ArrayRef<SyncStorageEffect> effects() const { return records; }
    ArrayRef<SyncStorageCell> cells() const { return partition; }
    ArrayRef<std::size_t> effectsFor(const CompoundInstanceElement* phase) const;
    // All supplied effects have exact byte sets; this is not a control/alias
    // certificate for effects omitted by the input producer.
    bool allAccessesExact() const;
    // Unknown local coordinates may alias every access in the same memory space.
    // GM uses the shared root-alias policy, without needing absolute addresses.
    // A positive answer is not proof of conflict. Invalid IDs are conservative.
    bool mayOverlap(std::size_t first, std::size_t second) const;
    // Read/read pairs need no ordering. Pipe and space come from shared phases.
    bool mayConflict(std::size_t first, std::size_t second) const;

private:
    GMAliasPolicy gmAliasPolicy = GMAliasPolicy::MayNotAlias;
    SmallVector<SyncStorageEffect, 0> records;
    SmallVector<SyncStorageCell> partition;
    DenseMap<const CompoundInstanceElement*, SmallVector<std::size_t>> phaseEffects;
    void partitionRanges();
};
} // namespace mlir::pto
#endif
