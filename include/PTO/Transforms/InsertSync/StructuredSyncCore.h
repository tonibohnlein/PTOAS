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
// A hardware contract is explicitly selected by the compiler invocation. The
// development documentation is NOT a release/device qualification. Conservative
// is the default; A2A3MmadAccV1 is the source-qualified S7 experimental contract.
enum class HardwareContract : uint8_t { Conservative, A2A3MmadAccV1 };
struct Target {
    // NPU2201, static-tensor library-safe pool. IDs 6/7 are NOT claimed absent
    // in hardware: they are withheld by this selected lowering contract.
    std::vector<unsigned> compilerKeys{0, 1, 2, 3, 4, 5};
    std::vector<Reservation> reservations;
    HardwareContract hardware = HardwareContract::Conservative;
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
// Immutable lowering facts, not a completion receipt. Unknown means no rule.
// The native adapter accepts only plain, default-phase, in-place f16/bf16->f32
// TMATMUL with exact effective dimensions and an exact full L0C footprint.
struct MmadInfo {
    enum Kind : uint8_t { Unknown, Initialize, Accumulate } kind = Unknown;
    uint64_t accumulatorBase = 0, accumulatorBytes = 0;
    uint64_t m = 0, n = 0, k = 0;
    enum Input : uint8_t { Unsupported, F16, BF16 } input = Unsupported;
    bool operator==(const MmadInfo &b) const {
        return kind == b.kind && accumulatorBase == b.accumulatorBase &&
            accumulatorBytes == b.accumulatorBytes && m == b.m && n == b.n &&
            k == b.k && input == b.input;
    }
    bool operator!=(const MmadInfo &b) const { return !(*this == b); }
};
struct Atom {
    uint64_t residue = 0;
    uint64_t order = 0;
    Lane lane;
    Segment segment = Segment::Body; // one-shot boundaries of ONE loop invocation
    MmadInfo matrix{};
};
// AccumulatorUpdate orders accesses to this same accumulator. It never means
// source-operation completion, L0A/L0B reclamation, FIX ownership or visibility.
enum class Property : uint8_t { Completion, AccResource, Visibility, AccumulatorUpdate };
struct Requirement {
    std::size_t source = 0, target = 0;
    uint64_t distance = 0; // Body/Body: target epoch - source epoch; boundary: zero
    Property property = Property::Completion;
    // Exact physical storage witness for a storage-specific target proof.
    // Ordinary completion/resource/visibility requirements leave this absent.
    bool hasStorageWitness = false;
    uint64_t storageBase = 0, storageBytes = 0;
};
struct Model {
    uint64_t period = 1;
    bool recurring = true;
    std::vector<Atom> atoms;
    // Conservative immutable RAW/WAR/WAW and typed resource witnesses.
    // Multiple readers are not collapsed during discovery.
    std::vector<Requirement> requirements;
    Target target;
    // S4's virtual startup prefix may continue a recurring notification on
    // the same key. Enable only with the complete mixed-boundary causal check;
    // this is not a request to serialize or change a handoff boundary.
    bool allowBoundaryKeyReuse = false;
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
    uint64_t refinementAttempts = 0, removedHandoffs = 0;
    uint64_t rekeyAttempts = 0, rekeyedHandoffs = 0, coalescedSites = 0;
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
    // S3: an invocation is one complete S2 region, not one inner iteration.
    // These predicates name the lexicographic successor/predecessor of the
    // enclosing rectangular loop nest. They never reset a live event key.
    enum Invocation : uint8_t { Local, ToNextInvocation, FromPreviousInvocation } invocation = Local;
    bool invocationBodyGuard = false;
    uint64_t invocationGuardResidue = 0; // conjunction N > r, when required
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

// One bounded reverse-deletion sweep over an already allocated periodic plan.
// Every accepted deletion rechecks original requirements AND consumption-before-
// rearm with the remaining numeric keys unchanged. No endpoint motion, barrier
// deletion, recoloring, or repeated fixed-point optimization is performed.
// Only a single local periodic body is admitted: callers must NOT refine the
// local component of a re-entrant invocation independently of its interface.
Result refinePeriodicHandoffs(const Model &, const Plan &);

// S6: one reverse sweep over a COMPLETE local region, including its startup,
// exit and empty-body paths. Every trial preserves the remaining keys, cuts
// and command order and rechecks the full protocol. Never apply to a component
// of InvocationPlan without rechecking that enclosing interface.
Result refineRegionHandoffs(const Model &, const Plan &);

// Original payload identity is supplied by the native import, not guessed from
// equal footprints or operation names. Initial/Steady name disjoint executions
// of that SAME original operation. S6 coalesces only period-one startup units.
struct SiteOrigin {
    enum Phase : uint8_t { Other, Initial, Steady } phase = Other;
    std::size_t payload = 0;
};
struct EmissionSite {
    // One original action, or one Initial plus one Steady action, at the same
    // original cut. A pair is emitted once without a phase predicate; it is NOT
    // two consecutive publications. Indices refer to the provided actions.
    std::vector<std::size_t> members;
};
// Preserves EACH phase's same-cut command order using a stable common
// subsequence. Only unconditional local commands with identical direction/key
// can match. Nullopt means invalid origins/actions; no supplied proof is lost.
std::optional<std::vector<EmissionSite>> startupEmissionSites(
    const Model &, const std::vector<SiteOrigin> &, const std::vector<Action> &);
// Try boundary-key continuation only when it reduces static emission sites at
// an IDENTICAL original payload cut. Whole-region verification is mandatory for
// each trial. No new keys, endpoint motion, guard broadening, or hardware elision.
Result coalesceStartupHandoffs(const Model &, const std::vector<SiteOrigin> &, const Plan &);

struct HandoffAudit {
    std::size_t handoff = 0;
    std::vector<std::size_t> completionLost; // indices into ORIGINAL requirements
    bool protocolWithout = false;
    std::string protocolReason;
};
// Opt-in diagnostic: check each deletion against completion and, separately,
// against event participation/rearm. "Lost" means unproved by this model, not
// hardware necessity. Do not run this O(H) checker campaign in normal compile.
std::optional<std::vector<HandoffAudit>> auditRegionHandoffs(const Model &, const Plan &);



// Fresh checking uses the ORIGINAL requirements and RECOVERED actual actions.
// It does not consume the planner's coverage receipts or an assumed pairing.
Result verify(const Model &, const std::vector<Action> &);
std::vector<Action> actionsForPlan(const Model &, const Plan &);

// Narrow property query. No completion edge is inserted by an intrinsic proof.
// Every intervening M occurrence must continue the same qualified accumulator
// chain. Different segments/invocations retain ordinary synchronization in S7.
bool intrinsicAccumulatorOrder(const Model &, const Requirement &);

// Testing/query client for exact symbolic epoch cuts. A path using only issue
// order is NEVER completion. True means all represented instances are covered.
bool supplies(const Model &, const Plan &, const Requirement &);

// S3 composes ONE immutable S1/S2 region through arbitrarily many enclosing
// rectangular invocations. Region shape, inner bound and selected branch are
// invariant over those invocations. The native adapter qualifies that contract.
// 'carried' contains every conservative cross-invocation conflict, independently
// of the local priorDistance relation. Source/target may be the SAME atom.
struct InvocationRequirement {
    std::size_t source = 0, target = 0;
    Property property = Property::Completion;
};
struct InvocationModel {
    Model region;
    std::vector<InvocationRequirement> carried;
};
struct InvocationHandoff {
    std::size_t source = 0, target = 0; // last source[j] -> first target[j+1]
    unsigned key = 0;
};
struct InvocationBarrier {
    std::size_t target = 0; // only at first target, and only after invocation 0
    bool bodyGuard = false;
    uint64_t guardResidue = 0;
};
struct InvocationPlan {
    Plan local;
    std::vector<InvocationHandoff> handoffs;
    std::vector<InvocationBarrier> barriers;
};
enum class InvocationFailure : uint8_t { None, Completion, Progress, EventReuse };
struct InvocationResult {
    Status status = Status::Unsupported;
    std::string reason;
    InvocationPlan plan;
    uint64_t proofViews = 0, graphVisits = 0;
    InvocationFailure failure = InvocationFailure::None;
    // Only failure to prove concrete recycling may be classified as allocation.
    // An invalid construction/coverage/progress result remains an internal error
    // at the native boundary; never turn it into permission to try another policy.
    Status constructionStatus() const {
        return status == Status::InvalidPlan && failure == InvocationFailure::EventReuse
            ? Status::AllocationFailure : status;
    }
};
InvocationResult constructInvocations(const InvocationModel &);
InvocationResult verifyInvocations(const InvocationModel &, const std::vector<Action> &);
std::vector<Action> actionsForInvocations(const InvocationModel &, const InvocationPlan &);
// Test/diagnostic query: independently rebuild the finite symbolic interface;
// do not reuse a planner receipt or assume an event pair from its numeric key.
bool suppliesInvocation(const InvocationModel &, const InvocationPlan &, const InvocationRequirement &);

} // namespace mlir::pto::structured_sync
#endif
