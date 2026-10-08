// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Qualified conservative class repetition with original nested occurrence tuples.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTCLASSREPETITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTCLASSREPETITION_H
#include "PTO/Transforms/FrontierSynch/BoundingRepetition.h"
#include "PTO/Transforms/FrontierSynch/CompactStorageBoundary.h"
namespace mlir::pto::frontiersynch {
struct CompactClassRepetition {
    std::string error;
    std::vector<CompactClasses> captured; // Static intermediate owners, also retained on failed exports.
    CompactClasses original; // Relative one-body invocation, retained on qualification failure.
    std::shared_ptr<const BoundingRepetitionResult> mathematical;
    CompactClasses boundary;
    uint64_t scalarValues = 0, accessPairs = 0, effectPairs = 0;
};
// Concrete q=1 producer: body must be a privately captured whole invocation of
// loop's block on the unchanged shared input. Proves inner bounds/control are
// deterministic and independent of this loop's IV/carried state. The upper
// storage specification is its conservative class union, not invariant original
// byte addresses: all sites survive, no uncertain writer kills another site.
// Every possible class pair gets a next-visit last->first requirement (apart
// from universally proved shared protection/separation). Native target chains
// then cover ALL later visits. Lower wrapping contains native facts only.
// Uses the shared counted-loop domain and weighted periodic query machinery;
// no trips/distances are unfolded. Returned selectors prefix the original loop
// coordinate, retain inner coordinates, and mask zero counts independently.
// Unavailable qualification/export retains original; physical storage and actual
// endpoint/excess claims are never fabricated. Existing frontend/manual-sync
// contracts still apply to the borrowed unchanged input.
CompactClassRepetition repeatCompactClassBoundary(
    func::FuncOp function, scf::ForOp loop, CompactClasses body);
// Automatic scoped producer for an executed sequence of explicit spans and
// counted children. Fixed/balanced leaf words use the shared compact adapter;
// nested children recurse through this same qualified repetition contract.
// Unbalanced optional bodies and varying inner counts need their own adapter;
// a failed repetition retains any completed relative-body mathematics.
// Extraction/indexing is charged per static body/span, plus existing compact
// analysis and O(A^2 + effect-pair work) boundary construction per repeat.
CompactClassRepetition analyzeCompactClassRepetition(func::FuncOp function, scf::ForOp loop,
    const SyncInput& input, std::shared_ptr<RegionExpressions> arena);
} // namespace mlir::pto::frontiersynch
#endif
