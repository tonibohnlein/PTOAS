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

// Half-open physical byte interval. Local addresses are absolute (base is null).
// GM intervals may be relative to a canonical entry pointer. Local allocation
// SSA roots never distinguish physical reuse; a GM base is part of identity.
struct SyncStorageCell {
    AddressSpace space;
    uint64_t begin = 0;
    uint64_t end = 0;
    Value base = {};
};

inline bool sameStorageDomain(const SyncStorageCell& a, const SyncStorageCell& b)
{
    return a.space == b.space && a.base == b.base;
}

// One canonical GM base is comparable under either policy. Different bases
// require MayNotAlias; absolute/based GM mixtures have unknown displacement.
// This checks identity comparability, not access precision or byte overlap.
bool storageBasesAreComparable(ArrayRef<SyncStorageCell> ranges, GMAliasPolicy policy);

struct SyncSlotSelection {
    Value family;
    Value selector;
    SmallVector<uint64_t> addresses;
};

struct SyncStorageEffect {
    const CompoundInstanceElement* phase = nullptr;
    const BaseMemInfo* memory = nullptr;
    // Legacy roots/ranges do not include arbitrary structured-control backedges.
    bool sharedProvenanceComplete = false;
    SyncAccessMode mode = SyncAccessMode::Read;
    SyncAccessPrecision precision = SyncAccessPrecision::Unknown;
    // Buffer geometry is retained independently from actual access precision.
    std::optional<SyncAccessRegion> descriptorRegion;
    // Preserve address-table selection even when it has no affine byte map.
    std::optional<SyncSlotSelection> selection;
    std::optional<SyncAccessRegion> region;
    // Union of exact declarations; region is populated for a single piece.
    SmallVector<SyncAccessRegion> regions;
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
    // Every selected effect must have an exact finite byte set with complete
    // base relationships. Omitted effects need a separate discharge proof.
    bool hasExactCellPartition(ArrayRef<std::size_t> effectIds) const;
    // Reuse InsertSync's buffer-range/alias checks, then refine with supplied
    // access regions. Unknown local addresses remain conservative. GM uses the
    // shared root-alias policy, without needing absolute addresses.
    // A positive answer is not proof of conflict. Invalid IDs are conservative.
    bool mayOverlap(std::size_t first, std::size_t second) const;
    // Read/read pairs need no ordering. Pipe and space come from shared phases.
    bool mayConflict(std::size_t first, std::size_t second) const;
    // A GM effect has no possible conflict with another static phase in this
    // unchanged input. Same-phase accesses are excluded; callers must establish
    // that they do not need ordering between repeated occurrences of that phase.
    // Invalid/non-GM IDs return false. Each queried GM effect scans the effects
    // at most once per build; rebuilding invalidates the lazy result cache.
    bool independentOfOtherPhases(std::size_t effect) const;
    // Preserve legacy buffer identities for insertion/allocation. The optional
    // filter selects one operand of the second phase (broadcast hazards).
    bool dependencies(const CompoundInstanceElement* first, SyncAccessMode firstMode,
                      const CompoundInstanceElement* second, SyncAccessMode secondMode,
                      DepBaseMemInfoPairVec& result, Value secondOperand = {}) const;

private:
    MemoryDependentAnalyzer memory;
    SmallVector<SyncStorageEffect, 0> records;
    SmallVector<SyncStorageCell> partition;
    // Translator indices are unique within this input and survive the copies
    // used by InsertSync's loop-backedge scan, including multi-phase operations.
    DenseMap<unsigned, SmallVector<std::size_t>> phaseEffects;
    mutable DenseMap<std::size_t, bool> independentEffects;
    void partitionRanges();
};
} // namespace mlir::pto
#endif
