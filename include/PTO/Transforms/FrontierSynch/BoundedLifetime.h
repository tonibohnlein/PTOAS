// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded-window demand circuits. Predicate recovery and the uniform span
// proof are separate inputs; endpoint availability is deliberately not inferred.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDEDLIFETIME_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDEDLIFETIME_H
#include "PTO/Transforms/FrontierSynch/GuardedRanks.h"
#include "PTO/Transforms/FrontierSynch/RotatingExtraction.h"
namespace mlir::pto::frontiersynch {
struct LifetimeAccess {
    uint32_t payload, cell;
    RegionExpressions::Id read, write;
    uint64_t protectionGroup = 0;
};
struct LifetimeWindowInput {
    uint32_t sites = 0;
    uint64_t span = 0;
    StorageProtectionPolicy storageProtection;
    // Exactly sites*(span+1) potential occurrences in reference order.
    // The supplier pads iterations after the actual trip count with false.
    std::vector<GuardedRankPayload> payloads;
    // Optional dynamic operation identities. Equal nonzero identities denote
    // simultaneous pipe envelopes of one operation, not ordered payloads.
    std::vector<uint64_t> operations;
    std::vector<LifetimeAccess> accesses;
    std::vector<GuardedRankEdge> prerequisites, nativePrerequisites;
};
struct LifetimeWindowAnalysis {
    std::string error;
    GuardedRanks window;
    std::vector<GuardedRankEdge> sourceDemands;
    uint64_t accessPairs = 0;
};
// Precondition: a uniform certificate bounds every sparse lifetime generator
// and supplied prerequisite by input.span (including entry and exit). Merely
// examining this window cannot establish that certificate. The returned local
// query rows are NOT global profiles. sourceDemands originate in iteration 0.
LifetimeWindowAnalysis analyzeLifetimeWindow(RegionExpressions& expressions, const LifetimeWindowInput& input);
struct RefreshCertificate {
    std::string error;
    uint64_t span = 0;
};
// A sufficient certificate from normalized rotating accesses: unconditional
// writer fragments must cover each writable slot orbit. Other participants
// may vary on every iteration. Constant moduli/strides and disjoint atoms are
// the same contract as rotating extraction; no bank enumeration is needed.
RefreshCertificate certifyRotatingRefresh(
    llvm::ArrayRef<PeriodicPayload> payloads, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<uint8_t> unconditional, uint64_t prerequisiteSpan = 0);
} // namespace mlir::pto::frontiersynch
#endif
