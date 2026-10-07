// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sufficient allocation for one certified affine varying-length invocation.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_VARYINGROTATINGALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_VARYINGROTATINGALLOCATION_H
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/RotatingBoundary.h"
#include <functional>
namespace mlir::pto::frontiersynch {
enum class VaryingAllocationRecordKind { Internal, SingletonCrossing, SuffixCrossing };
struct VaryingAllocationRecord {
    uint32_t record = 0, source = 0, target = 0;
    VaryingAllocationRecordKind kind = VaryingAllocationRecordKind::Internal;
    uint64_t displacement = 0;
    BoundaryOccurrence publication, consumption;
    // Absolute target visit for singleton records, suffix phase otherwise.
    uint64_t targetVisit = 0;
    uint32_t childRecord = UINT32_MAX;
};
// Includes local barrier records, preserving CircuitEndpoints family numbering.
std::optional<std::vector<VaryingAllocationRecord>> enumerateVaryingAllocationRecords(
    const AffineRotatingVisits& visits, std::string& error);
// A short sourceVisit identifies one exact visit. A sourceVisit >= startup
// identifies its entire certified suffix residue class, not one sampled visit.
// The callback must prove the fixed-gap relation for that whole class. Missing
// means the numerical proof is unavailable; false means this relation fails.
using VaryingBoundaryReuseQuery = std::function<std::optional<bool>(
    BoundaryOccurrence, PeriodicEventKind, BoundaryOccurrence, PeriodicEventKind, uint64_t, uint64_t)>;
// Separate palettes have independently certified internal reuse. Global M3
// lane matching assigns the one shared physical pool; directions are metadata.
// Failure is an unavailable sufficient certificate, never hardware scarcity.
std::shared_ptr<RegionalAllocationSummary> buildVaryingRotatingAllocation(
    const RegionalAnalysis& region, const AffineRotatingVisits& visits, RegionExpressions::Id trips,
    ArrayRef<VaryingAllocationRecord> records, const VaryingBoundaryReuseQuery& query, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
