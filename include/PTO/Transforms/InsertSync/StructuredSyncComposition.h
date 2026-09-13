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
    // Only Operation nodes carry physical effects. Structural nodes must have
    // a cells-sized zero vector; their MAY summaries are derived from children.
    Effects effects;
    // Postorder, strictly smaller child IDs. For has one body, While has before
    // and after, Choice has both arms (an omitted else is an empty Sequence).
    std::vector<unsigned> children;
    // Optional lowering contract for a unit-positive counted For: earliest
    // parent Sequence cut at which its original trip predicate is available.
    // No contract means conservative composition, not unsupported control flow.
    unsigned entryGuardStart = ~0u;
    // Optional lowering-owned periodic execution contract, relative to a
    // constant nonnegative lower bound and unit-step counted owner. For a
    // Choice, residues describes its true arm; for a Sequence it describes
    // execution of that complete word. Native reconstruction rederives this
    // from original scalar IR. Absence never changes semantic admission.
    unsigned periodicOwner = ~0u, periodicPeriod = 0;
    uint32_t periodicResidues = 0;
    int64_t periodicLower = 0;
    std::optional<int64_t> firstActive() const
    {
        if (!periodicPeriod || periodicPeriod > 32 || !periodicResidues || periodicLower < 0)
            return {};
        for (unsigned ordinal = 0; ordinal < periodicPeriod; ++ordinal)
            if (periodicResidues & (uint32_t(1) << ordinal)) {
                if (periodicLower > INT64_MAX - int64_t(ordinal))
                    return {};
                return periodicLower + int64_t(ordinal);
            }
        return {};
    }
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
    // Previous executes before every visit except the first. LoopExit executes
    // once after a nonempty counted loop. They are reserved for the bounded
    // deferred-wrap certificate; neither is a general predicate vocabulary.
    enum Participation { Every, NonEmpty, First, Previous, LoopExit } participation = Every;
    unsigned loop = ~0u;
    // A qualified periodic word uses its first active ordinal, not ordinal
    // zero, for Previous/LoopExit. Other mechanisms must leave this unset.
    unsigned word = ~0u;
    bool operator==(const Mechanism& other) const;
};
// Unnumbered completion obligation at original structural cuts. Cell witnesses
// explain why the producer prefix is needed; they do not own event resources.
struct CompletionDemand {
    unsigned scope = 0, publication = 0, acquisition = 0;
    unsigned source = 0, observer = 0;
    std::vector<unsigned> cells;
};
struct Result {
    bool success = false;
    std::string reason;
    std::vector<std::vector<Mechanism>> before;
    uint64_t nodeVisits = 0, cellVisits = 0, acquisitions = 0;
    uint64_t cutCycles = 0, allocationRetries = 0;
    uint64_t directHandoffs = 0, sharedAcknowledgments = 0, demandFallbacks = 0;
    uint64_t reusedAcknowledgments = 0;
    uint64_t completionRefinements = 0, rejectedRefinements = 0;
    uint64_t ownedRefinements = 0;
    uint64_t protocolKeys = 0, sharedProtocolKeys = 0, allocationFallbackScopes = 0;
    uint64_t allocationFallbackKeys = 0;
    uint64_t allocationReplays = 0, rejectedAllocationReplays = 0, replayCommandsRemoved = 0;
    uint64_t replayedFallbackDemands = 0;
    uint64_t entryEpisodes = 0, entryReplyFamilies = 0, rejectedEntryProposals = 0;
    // Selected-analysis work for bounded first-consumer entry proposals. Scan
    // units count examined summary/effect cells; storage units are conservative
    // representation cells, not bytes. Witness cells count Effects storage
    // allocated only for accepted, shared (acquisition, observer) witnesses.
    uint64_t entrySummarySlots = 0, entrySummaryScans = 0, entryStorageUnits = 0;
    uint64_t entryCandidatePairs = 0;
    uint64_t entryWitnessCells = 0, entryWitnesses = 0, entrySourceOverlapRejections = 0;
    uint64_t entrySummarySkipped = 0;
    uint64_t lateEntryCandidates = 0, lateEntryFamilies = 0, lateEntrySites = 0;
    uint64_t rejectedLateEntryFamilies = 0;
    // Optional ordinary-Choice incoming-prefix placement. Work bounds proposal
    // construction; accepted families remain ordinary Every protocols and are
    // checked/allocated with the complete demand word.
    uint64_t choiceDemandCandidates = 0, choiceDemandFamilies = 0;
    uint64_t rejectedChoiceDemands = 0, choiceDemandWork = 0;
    uint64_t choiceDemandReservedWork = 0, choiceDemandAnalysisWork = 0;
    uint64_t choiceDemandAnalysisPasses = 0, choiceDemandBudgetPass = 0;
    // Bounded child-return causal summaries only remove ordinary family ACKs;
    // they never provide physical completion or alter event capabilities.
    uint64_t childReturnCandidates = 0, childReturnAcksRemoved = 0;
    uint64_t rejectedChildReturns = 0, childReturnWork = 0;
    uint64_t childReturnChecks = 0, childReturnBudgetCheck = 0;
    bool childReturnBudgetExhausted = false;
    // Parent-owned ordinary Choice families: one source publication and one
    // exclusive acquisition/return path. Prefix steps count bypassed source
    // operations, not time or a hardware-latency estimate.
    uint64_t alternativeChoiceCandidates = 0, alternativeChoiceFamilies = 0;
    uint64_t alternativeChoiceSites = 0, alternativeChoiceSetsRemoved = 0;
    uint64_t alternativeChoiceSourceScopes = 0;
    uint64_t alternativeChoicePrefixSteps = 0, rejectedAlternativeChoices = 0;
    uint64_t alternativeChoiceWork = 0;
    bool alternativeChoiceBudgetExhausted = false;
    uint64_t ringCandidates = 0, rejectedRings = 0, ringCandidateCommandsRemoved = 0;
    uint64_t deferredRingCandidates = 0, deferredRings = 0, rejectedDeferredRings = 0;
    uint64_t deferredProtocolSteps = 0;
    uint64_t periodicDeferredRings = 0, periodicWriteOverlapRejections = 0;
    uint64_t deferredEligibilityWork = 0, deferredReceiptCells = 0, deferredSkippedFamilies = 0;
    // Reserved bounded representation work for repeated discovery, separate
    // from actual protocol steps and selected-family eligibility scans.
    uint64_t deferredDiscoveryWork = 0, deferredDiscoveryRefusals = 0;
    // Optional candidate diagnostics never replace the accepted baseline's
    // ordinary reason. They identify why deferred-wrap rollback occurred.
    std::string deferredRejectionStage, deferredRejectionReason;
    std::vector<CompletionDemand> demands;
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
// Demand-driven migration candidate. Direct placement/sharing are independent
// of cycle recognition; unmatched structural domains retain conservative transfer.
Result constructDemands(const Program& program);
Result verifyDemands(const Program& program, const std::vector<std::vector<Mechanism>>& actual);
constexpr uint64_t DeferredDiscoveryLimit = 1u << 27;
namespace testing {
// Same arithmetic preflight used by production, exposed for boundary tests.
std::optional<uint64_t> deferredDiscoveryReservation(
    uint64_t nodes, uint64_t cells, uint64_t keys, uint64_t commands, uint64_t limit = DeferredDiscoveryLimit);
Result constructDemandsWithoutRings(const Program& program);
Result constructDemandsRejectingRings(const Program& program);
// Fault injection after a deferred-wrap candidate is formed. The independently
// verified closed-ring plan must be returned unchanged.
Result constructDemandsRejectingDeferredRings(const Program& program);
Result constructDemandsWithoutDeferredDiscovery(const Program& program);
Result constructDemandsWithoutLateEntry(const Program& program);
Result constructDemandsWithLateEntryWorkLimit(const Program& program, uint64_t limit);
// Forms one bounded late-entry candidate, corrupts its actual First-wait
// population, and must return the exact verified pre-candidate plan.
Result constructDemandsRejectingLateEntry(const Program& program);
// Exact pre-refinement baseline and bounded/fault-injected ordinary-Choice
// incoming-prefix candidates. Failed optional candidates return that baseline.
Result constructDemandsWithoutChoiceDemands(const Program& program);
Result constructDemandsWithChoiceWorkLimit(const Program& program, uint64_t limit);
Result constructDemandsRejectingChoiceDemands(const Program& program);
Result constructDemandsWithoutChildReturns(const Program& program);
Result constructDemandsWithChildReturnWorkLimit(const Program& program, uint64_t limit);
Result constructDemandsRejectingChildReturns(const Program& program);
Result constructDemandsWithoutAlternativeChoices(const Program& program);
Result constructDemandsWithAlternativeChoiceWorkLimit(const Program& program, uint64_t limit);
Result constructDemandsRejectingAlternativeChoices(const Program& program);
// Fault injection before refinement checking/emission, never on the initial
// plan: discard the optional candidate's mechanisms to exercise exact rollback.
Result constructDemandsRejectingRefinement(const Program& program);
Result constructDemandsRejectingEntryProposal(const Program& program);
Result constructDemandsWithoutAllocationReplay(const Program& program);
Result constructDemandsRejectingAllocationReplay(const Program& program);
} // namespace testing
} // namespace mlir::pto::structured_sync::composition
#endif
