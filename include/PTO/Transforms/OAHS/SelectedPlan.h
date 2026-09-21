// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_SELECTEDPLAN_H
#define PTO_TRANSFORMS_OAHS_SELECTEDPLAN_H

#include "PTO/Transforms/OAHS/CausalFrontier.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"

namespace mlir::pto::oahs {

enum class SelectedFailure {
    None, InvalidInput, UnsupportedContract, UnqualifiedControl,
    MissingParticipation, EventResource, SelectedUpdate, LoopInvariant, FinalValidation
};
enum class EndpointPurpose { Fixed, Completion, ConsumptionAcknowledgment, LocalFence, Retirement, RecurringCompletion };
enum class RequirementStage { Known, Overlap };

// IDs survive insertion into an earlier word. cut/word positions are reconstructed
// from this ledger, not cached in an event-generation certificate.
struct SelectedEndpoint {
    std::size_t id = NoAnalysisId;
    Cut cut = NoAnalysisId;
    Command command;
    EndpointPurpose purpose = EndpointPurpose::Completion;
    // RecurringCompletion indexes channels; ordinary completion indexes decisions.
    std::size_t request = NoAnalysisId;
    std::size_t acknowledges = NoAnalysisId;
};
struct SelectedSource {
    Pipe pipe = Pipe::S;
    std::size_t origin = NoAnalysisId;
    Cut cut = NoAnalysisId;
    uint64_t version = 0;
    FrontierState snapshot;
    // Stable gap: after the original payload, before every endpoint in cut's
    // command word. This sentinel boundary has no shifting numeric offset.
    FrontierState postOrigin;
};
struct SelectedDecision {
    Cut consumer = NoAnalysisId, publication = NoAnalysisId;
    RequirementStage stage = RequirementStage::Overlap;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<FrontierRequirement> required;
    std::vector<std::size_t> endpoints;
    bool commonCut = false, enlargedPrefix = false;
    bool publicationAtWordStart = false;
    // Present only when F7 repaired consumption-before-republication. These
    // are physical key numbers and actual ledger endpoint IDs, not a claim of
    // storage completion by the helper.
    std::size_t repairedAcquisition = NoAnalysisId;
    unsigned repairedForwardKey = 0, repairReverseKey = 0;
    uint64_t repairInputVersion = 0, repairOutputVersion = 0;
    // Nonempty for one key published at alternative original source cuts and
    // acquired once at the common consumer. These are actual emitted cuts.
    std::vector<Cut> publicationFrontier = {};
};
struct SelectedUpdate {
    uint64_t version = 0, siteEvaluations = 0;
    std::size_t finalizedQueries = 0;
    // Components whose previous least solution this update actually reused, and
    // whether it had to solve the whole original graph instead.
    std::size_t reusedComponents = 0;
    bool contextual = false;
    std::vector<Cut> changedCuts;
};
// Opt-in contextual replay attribution. These records never supply causal
// facts or affect admission, placement, or the worklist order.
struct SelectedReplayTrace {
    uint64_t version = 0, microseconds = 0, evaluations = 0, uniqueSites = 0;
    uint64_t successorJoins = 0, changedJoins = 0, finalizedQueries = 0;
    std::size_t current = NoAnalysisId, activeComponent = 0;
    std::size_t changedBoundary = 0, fixedBoundary = 0, resume = 0;
    std::size_t reusedSites = 0, sharedWordLowerings = 0;
    std::size_t siblingComponents = 0;
    uint64_t invalidationSites = 0, invalidationEdges = 0, sharedWordOccurrences = 0;
    bool success = false;
    std::vector<Cut> changedCuts;
    std::vector<std::size_t> changedComponents;
    struct ComponentWork {
        std::size_t sites = 0, uniqueSites = 0;
        uint64_t evaluations = 0;
        bool cyclic = false;
        std::vector<std::size_t> successors;
    };
    std::vector<ComponentWork> components;
};
struct SelectedChannel {
    unsigned cell = 0, key = 0;
    // One recurring prefix can serve several compatible cells. `cell` remains
    // the stable first cell for source compatibility and diagnostics.
    std::vector<unsigned> cells;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<Cut> publications, acquisitions;
    std::size_t owner = NoAnalysisId;
    // The actual original modulo period this role was qualified with. Native
    // period begins at 1; the key pool limits admission, it does not select it.
    uint64_t period = 0;
};
struct SelectedFence {
    Cut cut = NoAnalysisId;
    Pipe observer = Pipe::S;
    uint64_t version = 0;
    // Exact same-engine residual present immediately before construction
    // inserted this fence. These are diagnostic obligations, not assumptions
    // supplied to final validation.
    std::vector<FrontierRequirement> residuals;
};
// A clause is conditional on the original observation at its exact cut. Its
// snapshot is established by the selected ledger at version, never assumed by
// the checker. Fixed physical cell/key names are not dynamic generation ranks.
struct SelectedRoleClause {
    Cut cut = NoAnalysisId;
    std::size_t observation = NoAnalysisId;
    uint64_t version = 0;
    FrontierState incoming, beforeIssue, outgoing;
};
struct SelectedLoopInterface {
    std::size_t owner = NoAnalysisId;
    Cut entry = NoAnalysisId, exit = NoAnalysisId;
    uint64_t version = 0;
    FrontierState incoming, outgoing;
    std::vector<SelectedRoleClause> clauses;
};
struct SelectedWork {
    uint64_t frontierVisits = 0, selectedUpdates = 0, replaySiteEvaluations = 0;
    uint64_t forwardSiteEvaluations = 0;
    uint64_t keyQueries = 0, invariantSiteEvaluations = 0;
    // Total portable construction includes model/control/storage preparation and
    // final validation. It excludes native import/emission and test references.
    uint64_t elapsedMicroseconds = 0, preparationMicroseconds = 0;
    std::size_t sourceHandles = 0, acknowledgments = 0, commonCutTransfers = 0;
    std::size_t recurringChannels = 0;
    std::size_t recurringTrials = 0, redundantRecurringChannels = 0;
    uint64_t recurringAnalysisSites = 0;
    // Qualification runs once in this constructor. Native admission queries
    // are outside this timer, as is selected replay. Final helper trials are
    // separate from recurring omission trials and from the final certificate.
    uint64_t recurringQualificationMicroseconds = 0;
    uint64_t proposalCheckSites = 0, proposalCheckMicroseconds = 0;
    std::size_t rejectedProtocolProposals = 0, rejectedResourceProposals = 0;
    std::size_t rejectedSupportProposals = 0;
    std::size_t gapPublications = 0, deferredAcknowledgments = 0;
    std::size_t equalCoveragePairs = 0, bindingProbes = 0, bindingChoices = 0;
    std::size_t helperCompositionTrials = 0;
    uint64_t helperCompositionSiteEvaluations = 0, helperCompositionMicroseconds = 0;
    uint64_t finalCertificateSiteEvaluations = 0, finalCertificateMicroseconds = 0;
    std::size_t loopEntryTransfers = 0;
    std::size_t rearmingDischarged = 0;
    std::size_t rearmingComposed = 0;
    std::size_t rearmingRestored = 0;
    uint64_t rearmingPairVisits = 0, rearmingQuerySites = 0;
    uint64_t loopEntryAnalysisSites = 0;
    uint64_t loopEntryPreparationSites = 0;
    // Dimensions of the graph actually constructed over, recorded once. They
    // separate refinement expansion from repeated visits and state-copy cost;
    // they are not work allowances and never affect a decision.
    std::size_t constructedSites = 0, commandWords = 0, cells = 0, eligibleKeys = 0;
    std::size_t components = 0, cyclicComponents = 0;
    // Immutable original-program requirements. A source boundary is counted
    // only when its acyclic occurrence is qualified; the remainder needs
    // recurring, guarded, or alternative-source correspondence.
    std::size_t requirementFrontiers = 0, qualifiedSourceFrontiers = 0;
    std::size_t unqualifiedSourceFrontiers = 0;
    std::size_t frontierAcyclic = 0, frontierSameVisit = 0, frontierPreviousUse = 0;
    std::size_t frontierRegionEntry = 0, frontierRegionContinuation = 0;
    std::size_t frontierGuarded = 0, frontierUnknown = 0;
    // Whole-original-graph contextual solves, and updates that reused nothing.
    std::size_t contextualReplays = 0, unreusedUpdates = 0;
    uint64_t replayInvalidationSites = 0, replayInvalidationEdges = 0, replaySharedWordOccurrences = 0;
    std::size_t siblingReusedComponents = 0;
};
// These switches never disable mandatory checking or native reconstruction.
struct SelectedOptions {
    bool recurring = true;
    bool recurringOmissionTrials = true;
    bool finalHelperTrials = true;
    bool movingFrontiers = true;
    bool sourceGaps = false;
    bool deferredAcyclicAcknowledgments = false;
    bool classInvariantInputs = false;
    bool equalCoverageBinding = false;
    bool traceReplay = false;
    // Diagnostic comparison with the former prefix-only contextual cache.
    bool siblingReplayReuse = true;
};
struct SelectedPlan {
    bool success = false;
    SelectedFailure failure = SelectedFailure::None;
    std::string reason;
    Cut cut = NoAnalysisId;
    // Populated only after final validation. A partial diagnostic ledger is not
    // an emitted synchronization program.
    Commands commands;
    std::vector<SelectedEndpoint> ledger;
    std::vector<SelectedSource> sources;
    std::vector<SelectedDecision> decisions;
    std::vector<SelectedChannel> channels;
    std::vector<SelectedFence> fences;
    std::vector<SelectedUpdate> updates;
    std::vector<SelectedReplayTrace> replayTraces;
    // Populated only from the final, entry-containing original-graph proof.
    std::vector<SelectedLoopInterface> loops;
    FrontierCheck certificate;
    SelectedWork work;
};

// F1--F8 construction service used by the live handoff pass.
// Fixed words are preserved in order. Unsupported typed effects are refused,
// never erased or delegated to another constructor on failure.
// Read-only physical-role qualification. This grants neither completion nor
// event credit and does not imply that construction or allocation will succeed.
bool hasQualifiedRecurringAccesses(const Program&);

SelectedPlan constructSelectedPlan(const Program&, const Commands& fixed = {},
                                   SelectedOptions options = {});

} // namespace mlir::pto::oahs
#endif
