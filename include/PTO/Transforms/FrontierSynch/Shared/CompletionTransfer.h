// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Section 6 transfer over a compact immediate-child CFG. This is mathematical
// completion placement; callers separately certify child code, closed matching,
// forward entry prerequisites and executable common completion protocols.
#ifndef PTO_FRONTIERSYNCH_COMPLETIONTRANSFER_H
#define PTO_FRONTIERSYNCH_COMPLETIONTRANSFER_H
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
// One byte per hierarchy-wide storage class: bit 0 reads, bit 1 writes.
using CompletionFacts = std::vector<uint8_t>;
struct CompletionSummary {
    CompletionFacts all, exposed, suffix;
    bool transparent = true;
};
struct CompletionVertex {
    CompletionSummary summary;
    std::vector<std::size_t> successors;
    // The caller has certified that every source precedes this child entry
    // and that a common completion establishes the outstanding prerequisite.
    bool requiresEntryCompletion = false;
};
struct CompletionTransferWork {
    uint64_t propagatedFacts = 0;
    uint64_t edgeFactVisits = 0;
};
struct CompletionTransferResult {
    CompletionSummary summary;
    std::vector<bool> reachable, protectEntry;
    CompletionTransferWork work;
    std::string error;
};
// Input summaries use exactly classCount entries, with E and S contained in A.
// The designated exit has no successors. Unreachable nodes do not contribute.
// Each solve sends each class/mode over each reachable edge at most once.
// No visit unfolding, symbolic selectors or target synchronization is involved.
CompletionTransferResult buildCompletionTransfer(const std::vector<CompletionVertex>& graph,
    std::size_t entry, std::size_t exit, std::size_t classCount);
} // namespace mlir::pto::frontiersynch
#endif
