// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCCOMPOSITION_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCCOMPOSITION_H

#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include <array>
#include <vector>

namespace mlir::pto::structured_sync::composition {
constexpr unsigned LaneCount = 7;
constexpr unsigned MaxCells = 256;
// Bits denote MAY outstanding accesses, never definite initialization. Each
// observer has its own history: a wait on V does not stop MTE2 or the host.
struct History {
    uint8_t readers = 0, writers = 0;
};
using Effects = std::vector<History>;
struct State {
    std::array<Effects, LaneCount> pending;
    // Visibility is not completion. No current rendezvous/barrier discharges
    // same-address GM publication, so this history survives those mechanisms.
    std::vector<uint8_t> written;
    explicit State(unsigned cells = 0);
    void join(const State& other);
    void seed(const Effects& effects);
    void barrier(unsigned lane);
    void rendezvous(unsigned first, unsigned second);
    uint8_t demands(unsigned observer, const Effects& effects) const;
};
struct Node {
    enum Kind { Operation, Sequence, Choice, For, While } kind = Sequence;
    unsigned lane = 0;
    Effects effects;
    // Postorder, strictly smaller child IDs. For has one body, While has before
    // and after, Choice has both arms (an omitted else is an empty Sequence).
    std::vector<unsigned> children;
};
struct Program {
    Core core = Core::AIV;
    Target target;
    unsigned cells = 0;
    std::vector<bool> globalMemory;
    std::vector<Node> nodes;
};
struct Mechanism {
    enum Kind { Barrier, Rendezvous, Publish, Acquire } kind = Barrier;
    unsigned first = 0, second = 0;
    unsigned forwardKey = 0, reverseKey = 0;
    bool operator==(const Mechanism& other) const;
};
struct Result {
    bool success = false;
    std::string reason;
    std::vector<std::vector<Mechanism>> before;
    uint64_t nodeVisits = 0, cellVisits = 0, acquisitions = 0;
    uint64_t cutCycles = 0, allocationRetries = 0;
};
// One summary pass and one structural transfer. No trip-count enumeration,
// symbolic arithmetic, dense closure, or iterative loop invariant discovery.
Result construct(const Program& program);
// Actual mechanisms, not selected coverage receipts. Rebuilds requirements
// from Program effects and checks every complete region/backedge transfer.
Result verify(const Program& program, const std::vector<std::vector<Mechanism>>& actual);
// Bounded structural cuts and recurring physical-storage handoffs. Unsupported
// precision leaves the general transfer in place; it is never an admission rule.
Result constructCuts(const Program& program);
Result verifyCuts(const Program& program, const std::vector<std::vector<Mechanism>>& actual);
} // namespace mlir::pto::structured_sync::composition
#endif
