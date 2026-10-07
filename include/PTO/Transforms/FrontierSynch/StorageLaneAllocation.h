// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_STORAGELANEALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_STORAGELANEALLOCATION_H
#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto::frontiersynch {
// A normalized disjoint byte atom in a physical slot family. The local window
// cell denotes slot (stride*sourceOrdinal+offset)%slots, not slot offset on
// every invocation. Family/atom identities and this map persist across visits.
struct StorageLaneCell {
    uint32_t cell = 0, family = 0;
    uint64_t begin = 0, end = 0, slots = 1, stride = 0, offset = 0;
};
struct StorageLaneAllocation {
    std::string error;
    uint64_t budget = 0;
    // One expression per original sourceDemands record, including local ones.
    // This is a derived source selector, NOT an enclosing visit coordinate.
    // Both cuts must evaluate the same selector at the canonical source ordinal.
    std::vector<RegionExpressions::Id> lanes;
    std::vector<uint32_t> records; // Original records wholly covered by storage witnesses.
};
// Uses only supplied sparse lifetime provenance, never window query rows as
// global reachability. Every retained cross-pipe guard must have a storage
// witness unless allowPartial is requested. Then uncovered records are omitted
// from records and require a separate certificate for their complete guards.
// Work/representation depends on descriptors and witnesses, not visit count or
// guard valuations. Physical labels are bounded by the six-ID hardware pool.
StorageLaneAllocation buildStorageLaneAllocation(RegionExpressions& expressions,
    const LifetimeWindowInput& window, const LifetimeWindowAnalysis& analysis,
    llvm::ArrayRef<StorageLaneCell> cells, RegionExpressions::Id sourceOrdinal, bool allowPartial = false);
// Whole-function certificate only. Endpoint tuples have canonical source
// ordinal at coordinate0 and the paired derived lane selector at coordinate1.
// This coordinate must not be exported as a RegionalAnalysis enclosing visit.
DictionaryAttr storageLaneAllocationCertificate(func::FuncOp function,
    const LifetimeWindowInput& window, llvm::ArrayRef<GuardedRankEdge> demands,
    const StorageLaneAllocation& allocation, int64_t plan);
// Keep all palettes disjoint across both proofs. Empty means the complete
// composite is unavailable or exceeds six labels, never a minimum-width claim.
DictionaryAttr combineStorageLaneCertificates(DictionaryAttr storage, DictionaryAttr residual);
} // namespace mlir::pto::frontiersynch
#endif
