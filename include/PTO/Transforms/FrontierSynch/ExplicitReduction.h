// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite forward-demand reduction with native start and completion chains.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_EXPLICITREDUCTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_EXPLICITREDUCTION_H
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
namespace mlir::pto::frontiersynch {
struct ExplicitReduction {
    std::string error;
    std::vector<StorageGenerator> retained;
    // Rows follow occurrence order; columns follow first appearance of each pipe.
    std::vector<uint32_t> payloads;
    std::vector<uint32_t> pipeLabels;
    std::vector<uint32_t> pipeColumns;
    std::vector<uint32_t> localRanks;
    // startRanks[b][p] is the last completion on p reaching I_b.
    // completionRanks[b][p] also includes C_b itself when p is b's pipe.
    std::vector<std::vector<uint32_t>> startRanks;
    std::vector<std::vector<uint32_t>> completionRanks;
};
// Only payload IDs and pipe labels are consumed from occurrences. IDs must be
// unique and generators must point forward in occurrence order. Duplicate edges
// are allowed. Failure returns only an error, never a partial reduction.
// Expected O(n + g + nk + k|F*|) time and O(nk + n + g) storage. Pipe labels
// are compacted; their numerical values do not determine allocation sizes.
ExplicitReduction reduceExplicitDemands(llvm::ArrayRef<ExplicitEffects> occurrences,
                                        llvm::ArrayRef<StorageGenerator> generators);
} // namespace mlir::pto::frontiersynch
#endif
