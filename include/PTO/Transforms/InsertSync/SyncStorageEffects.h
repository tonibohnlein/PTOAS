// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared modeled accesses: available geometry and alias relationships.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSTORAGEEFFECTS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSTORAGEEFFECTS_H

#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncAccessRegion.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::pto {
enum class SyncAccessMode { Read, Write };

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
    // Buffer geometry is retained independently from access selections.
    std::optional<SyncAccessRegion> descriptorRegion;
    // Preserve address-table selection even when it has no affine byte map.
    std::optional<SyncSlotSelection> selection;
    std::optional<SyncAccessRegion> region;
    // Modeled access union; region is populated for a single piece.
    SmallVector<SyncAccessRegion> regions;
    // Whether ranges completely materialize the modeled set, including empty sets.
    bool rangesMaterialized = false;
    SmallVector<SyncStorageCell> ranges;
    SmallVector<std::size_t> cells;
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
    // Every modeled access is materialized, including known-empty accesses.
    bool allAccessesMaterialized() const;
    // Every selected effect has a finite modeled byte set with comparable
    // base relationships. Omitted effects need a separate discharge proof.
    bool hasMaterializedCellPartition(ArrayRef<std::size_t> effectIds) const;
    // Reuse InsertSync's buffer-range/alias checks, then refine with supplied
    // access regions. Unknown local addresses remain conservative. GM uses the
    // shared root-alias policy, without needing absolute addresses.
    // A positive answer is not proof of conflict. Invalid IDs are conservative.
    bool mayOverlap(std::size_t first, std::size_t second) const;
    // Read/read pairs need no ordering. Pipe and space come from shared phases.
    bool mayConflict(std::size_t first, std::size_t second) const;
    // A modeled conflict not represented by the common materialized cells.
    // This preserves unresolved relationships without merging disjoint domains.
    bool residualConflict(std::size_t first, std::size_t second) const;
    // A conflict whose unresolved address relationship applies to every pair
    // of occurrences. Known iteration-dependent maps are handled by adapters.
    bool uniformConflict(std::size_t first, std::size_t second) const;
    // Linear precheck keeps uniform-pair enumeration off fully comparable inputs.
    bool hasUniformRelationships(ArrayRef<const CompoundInstanceElement*> phases) const;
    bool needsOverlapQueries(const CompoundInstanceElement* phase, bool unresolvedBases = true) const;
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
