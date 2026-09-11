// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCCORE_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCCORE_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::pto::structured_sync {

// This header intentionally has no MLIR, Presburger, solver or work-budget
// dependency. IDs index immutable native facts, not guessed operation names.
enum class Core : uint8_t { AIC, AIV };
enum class Pipe : uint8_t { S, V, M, MTE1, MTE2, MTE3, FIX };
struct Lane {
    Core core = Core::AIV;
    Pipe pipe = Pipe::V;
    bool operator==(const Lane &b) const { return core == b.core && pipe == b.pipe; }
    bool operator!=(const Lane &b) const { return !(*this == b); }
    bool operator<(const Lane &b) const {
        return core != b.core ? core < b.core : pipe < b.pipe;
    }
};
struct Reservation {
    Lane source, target;
    unsigned key = 0;
};
struct Target {
    // NPU2201, static-tensor library-safe pool. IDs 6/7 are NOT claimed absent
    // in hardware: they are withheld by this selected lowering contract.
    std::vector<unsigned> compilerKeys{0, 1, 2, 3, 4, 5};
    std::vector<Reservation> reservations;
    bool supports(Lane lane) const;
    bool event(Lane source, Lane target) const;
    bool barrier(Lane lane) const;
    bool synchronous(Lane lane) const { return lane.pipe == Pipe::S && supports(lane); }
    bool available(Lane source, Lane target, unsigned key) const;
};

// A Body atom occurs once at t = period*k + residue. Prelude/Epilogue
// atoms execute exactly once before/after the loop, even for zero trips.
// Boundary atoms have residue zero. Original payload order is unchanged.
// 'order' is the order of
// original payloads within t, not an arbitrary phase ID. All occurrences are
// clipped by 0 <= t < tripCount. Thus a finite execution is a PREFIX of the
// admitted periodic schedule. No enumeration depends on tripCount.
enum class Segment : uint8_t { Prelude, Body, Epilogue };
struct Atom {
    uint64_t residue = 0;
    uint64_t order = 0;
    Lane lane;
    Segment segment = Segment::Body; // one-shot boundaries of ONE loop invocation
};
enum class Property : uint8_t { Completion, AccResource, Visibility };
struct Requirement {
    std::size_t source = 0, target = 0;
    uint64_t distance = 0; // Body/Body: target epoch - source epoch; boundary: zero
    Property property = Property::Completion;
};
struct Model {
    uint64_t period = 1;
    bool recurring = true;
    std::vector<Atom> atoms;
    // Conservative immutable RAW/WAR/WAW and typed resource witnesses.
    // Multiple readers are not collapsed during discovery.
    std::vector<Requirement> requirements;
    Target target;
};
struct Handoff {
    std::size_t source = 0, target = 0;
    uint64_t distance = 0;
    unsigned key = 0;
};
struct Plan {
    std::vector<Handoff> handoffs;
    std::vector<std::size_t> barriers; // before these original payloads
    std::vector<std::size_t> firstBarriers; // once, before the first occurrence of a body atom
};
enum class Status : uint8_t { Applied, Unsupported, AllocationFailure, InvalidPlan };
struct Result {
    Status status = Status::Unsupported;
    std::string reason;
    Plan plan;
    uint64_t completionRelaxations = 0;
    uint64_t eventRelaxations = 0;
};

// Actual commands recovered from emission. Order at a common boundary is
// explicit so two sets on the same key cannot be merged by projection.
struct Action {
    enum Kind : uint8_t { Set, Wait, Barrier } kind = Set;
    std::size_t anchor = 0;
    bool after = false;
    uint64_t order = 0;
    Lane source, target;
    unsigned key = 0;
    uint64_t distanceInIterations = 0; // Every only; boundary forms keep zero
    enum Participation : uint8_t { Every, First, Last, IfBody } participation = Every;
    uint64_t guardResidue = 0; // IfBody only: execute iff tripCount > guardResidue
    // First/Last select the first/last occurrence of the anchored body atom.
    // IfBody pairs with one of those and tests N > guardResidue outside the loop.
    // Every Set: execute iff matching target t+distance exists.
    // Every Wait: execute iff matching source t-distance exists.
    // Barrier: distance must be zero. No event token is consumed.
};

// The latest ordinary occurrence of source before target. A boundary
// requirement uses distance zero; its endpoint segments specify Once->First,
// Last->Once, or Once->Once. It is NOT an invented periodic epoch.
// A recurring self access is in the preceding epoch. No must-alias or definite-write claim.
std::optional<uint64_t> priorDistance(const Model &, std::size_t source, std::size_t target);
std::optional<uint64_t> iterationDistance(const Model &, const Handoff &);

// Deterministic staircase selection, staged same-lane repair, and finite
// target-key assignment. There is no backtracking or quota. Allocation may
// decline a feasible plan that requires a different sharing policy.
Result construct(const Model &);

// Fresh checking uses the ORIGINAL requirements and RECOVERED actual actions.
// It does not consume the planner's coverage receipts or an assumed pairing.
Result verify(const Model &, const std::vector<Action> &);
std::vector<Action> actionsForPlan(const Model &, const Plan &);

// Testing/query client for exact symbolic epoch cuts. A path using only issue
// order is NEVER completion. True means all represented instances are covered.
bool supplies(const Model &, const Plan &, const Requirement &);

} // namespace mlir::pto::structured_sync
#endif
