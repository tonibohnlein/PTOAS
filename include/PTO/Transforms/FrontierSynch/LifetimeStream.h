// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LIFETIMESTREAM_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LIFETIMESTREAM_H
#include "PTO/Transforms/FrontierSynch/GuardedRanks.h"
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
namespace mlir::pto::frontiersynch {
struct LifetimeStreamAccess {
    uint32_t site = 0, cell = 0;
    uint64_t protectionGroup = 0;
    bool invariantProtection = false;
};
struct LifetimeStreamEdge {
    uint32_t age = 0, source = 0, target = 0;
    bool native = false;
};
struct LifetimeStreamTemplate {
    uint32_t span = 0;
    uint64_t maximumIterations = 0;
    std::vector<uint32_t> pipes;
    // Equal nonzero identities denote simultaneous envelopes within one visit.
    std::vector<uint64_t> operations;
    // Unique (site,cell) descriptions; repeated descriptors must be normalized.
    std::vector<LifetimeStreamAccess> accesses;
    std::vector<LifetimeStreamEdge> prerequisites;
    StorageProtectionPolicy storageProtection;
};
struct LifetimeStreamAccessState {
    RegionExpressions::Id read, write, noLaterWriter;
};
struct LifetimeStreamVisit {
    std::vector<RegionExpressions::Id> present, ranks;
    std::vector<std::vector<RegionExpressions::Id>> completions;
    std::vector<LifetimeStreamAccessState> accesses;
};
struct LifetimeStreamRegisters {
    GuardedRankFrontier frontier;
    // Most recent completed iteration first. Exactly span entries.
    std::vector<LifetimeStreamVisit> history;
};
struct LifetimeStreamInputs {
    std::vector<RegionExpressions::Id> present, read, write, prerequisites;
};
struct LifetimeStreamDemand {
    LifetimeStreamEdge edge;
    RegionExpressions::Id guard;
};
struct LifetimeStreamTransition {
    std::string error;
    LifetimeStreamRegisters next;
    LifetimeStreamVisit current;
    std::vector<std::vector<RegionExpressions::Id>> starts;
    std::vector<LifetimeStreamDemand> retained;
    uint64_t accessPairs = 0;
};
// Caller certifies uniform generator span <=span and invariant physical cells.
// For cells with simultaneous envelopes, short-span raw pairs replace the plain
// scan: the caller additionally certifies this candidate set generates the order.
// Registers must be reachable from initialization; at most maximumIterations
// transitions may execute. Rank words use the existing uint64 expression domain.
// Each call constructs ONE transition. Supply independent register inputs (or
// resolved constants in a fresh arena); chaining expressions retains history.
// The gate/state bound applies to the constructed transition, not the host
// descriptor copying/validation cost. This is not a resolved sparse-generator
// executor with the output-sensitive word bound.
// Decisions become known at consumers. Existing stateless windows remain the
// endpoint-availability interface; no source publication recipe is implied.
LifetimeStreamTransition advanceLifetimeStream(RegionExpressions&, const LifetimeStreamTemplate&,
    const LifetimeStreamRegisters&, const LifetimeStreamInputs&);
LifetimeStreamRegisters initializeLifetimeStream(RegionExpressions&, const LifetimeStreamTemplate&);
} // namespace mlir::pto::frontiersynch
#endif
