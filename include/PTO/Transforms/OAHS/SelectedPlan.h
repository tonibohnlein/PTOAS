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
#include <optional>

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
    // Original dormant fallback word, retained when its inactive WAIT is placed
    // at a proved later reuse deadline. No duplicate ownership is introduced.
    Cut originalCut = NoAnalysisId;
};
struct SelectedRestoration {
    std::size_t consumption = NoAnalysisId, publication = NoAnalysisId, acquisition = NoAnalysisId;
    std::size_t deadlinePublication = NoAnalysisId;
    Cut fallbackCut = NoAnalysisId, placedCut = NoAnalysisId;
    uint64_t version = 0;
    std::string fallbackReason;
};
// A sufficient structural ordering certificate, separate from causal legality.
// Unknown includes a changed symbolic dependency and unsupported recurrence;
// it is not a claim that the payload order necessarily grew.
struct SelectedPublicationSupport {
    std::size_t publication = NoAnalysisId;
    Cut occurrence = NoAnalysisId;
    uint64_t admittedVersion = 0, checkedVersion = 0;
    std::size_t admittedRoot = NoAnalysisId;
    bool preserved = false;
};
struct SelectedSource {
    Pipe pipe = Pipe::S;
    std::size_t origin = NoAnalysisId;
    Cut cut = NoAnalysisId;
    uint64_t version = 0;
    FrontierState snapshot;
};
// Original physical relationship and endpoint bounds retained by a binding.
// Return deadlines are possible support; they are never acquired event credit.
struct SelectedLifecycleDemand {
    StorageRelationship relationship;
    Cut release = NoAnalysisId, deadline = NoAnalysisId;
    std::vector<Cut> returnDeadlines;
};
struct SelectedDecision {
    Cut consumer = NoAnalysisId, publication = NoAnalysisId;
    // Original source milestone and exact selected word gap. A later selected
    // return can intentionally broaden this new publication's prefix.
    Cut sourceMilestone = NoAnalysisId;
    std::size_t publicationGapLeft = NoAnalysisId, publicationGapRight = NoAnalysisId;
    std::size_t supportingReceipt = NoAnalysisId;
    RequirementStage stage = RequirementStage::Overlap;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<FrontierRequirement> required;
    // Extra actual source coverage used to select this provider. Exact-gap
    // placement must preserve it; only the executed receipt grants credit.
    std::vector<FrontierRequirement> supporting;
    std::vector<SelectedLifecycleDemand> lifecycles;
    std::vector<std::size_t> endpoints;
    bool commonCut = false, enlargedPrefix = false;
    bool existingPublicationsPreserved = true;
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
struct SelectedRealizationChoice {
    Cut deadline = NoAnalysisId;
    unsigned placementClass = 0;
    bool recurring = false, known = false;
    std::size_t ordinaryCandidates = 0, recurringCandidates = 0, covered = 0;
    uint64_t inputVersion = 0, outputVersion = 0;
};
struct SelectedRecurringActivation {
    Cut deadline = NoAnalysisId;
    std::vector<std::size_t> families;
    std::vector<FrontierRequirement> before, after;
    uint64_t version = 0;
};
struct SelectedRecurringRefusal {
    Cut deadline = NoAnalysisId;
    std::size_t family = NoAnalysisId;
    uint64_t version = 0;
    std::string reason;
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
    uint64_t occurrenceAnalysisSites = 0;
    uint64_t boundaryAnalysisSites = 0;
    uint64_t physicalUseQuerySites = 0;
    uint64_t requirementClassifications = 0, classificationSites = 0, classificationOrigins = 0;
    uint64_t witnessQueries = 0, witnessSites = 0;
    uint64_t producerSupportWork = 0;
    uint64_t publicationSupportSites = 0, publicationSupportCommands = 0;
    uint64_t publicationSupportChecks = 0, publicationSupportNodes = 0;
    uint64_t restorationDeadlineQueries = 0, restorationUseChecks = 0, deadlineRestorations = 0;
    uint64_t restorationPositionEntries = 0, restorationDeadlineFallbacks = 0;
    uint64_t acknowledgmentPrefixReplays = 0, acknowledgmentPrefixReplaySites = 0;
    uint64_t sourceGapQueries = 0, sourceGapCommands = 0, earlyPublications = 0;
    // Native original-program discovery, before selected construction/retries.
    uint64_t nativeEndpointDiscoveryWork = 0;
    uint64_t unsummarizedBackedges = 0, finiteOccurrenceTransitions = 0;
    uint64_t transitionClassificationWork = 0;
    uint64_t frontierVisits = 0, selectedUpdates = 0, replaySiteEvaluations = 0;
    uint64_t forwardSiteEvaluations = 0;
    uint64_t keyQueries = 0, invariantSiteEvaluations = 0;
    uint64_t ownershipQueries = 0, ownershipChecks = 0, ownershipCheckSites = 0, ownershipBindings = 0;
    uint64_t acknowledgmentChecks = 0, acknowledgmentCheckSites = 0, joinedAcknowledgments = 0;
    // Total portable construction includes model/control/storage preparation and
    // final validation. It excludes native import/emission and test references.
    uint64_t elapsedMicroseconds = 0, preparationMicroseconds = 0;
    std::size_t sourceHandles = 0, acknowledgments = 0, commonCutTransfers = 0;
    std::size_t recurringChannels = 0, recurringProposals = 0;
    std::size_t recurringTrials = 0, redundantRecurringChannels = 0;
    uint64_t normalCandidates = 0, normalSelected = 0, normalRecurringSelected = 0;
    uint64_t repairCandidates = 0, repairSelected = 0, repairSourceCommands = 0, repairNeighborUses = 0;
    uint64_t corridorReceiptScans = 0, corridorWordEndpoints = 0;
    uint64_t commonCutContinuationQueries = 0, commonCutContinuationSites = 0;
    uint64_t commonCutContinuationWords = 0;
    uint64_t normalPublicationSites = 0, normalizedDue = 0, normalizationIncidences = 0, normalKeySites = 0;
    uint64_t recurringInterfaceQueries = 0, recurringInterfaceSites = 0;
    uint64_t recurringInterfaceAccesses = 0, recurringLocalPackets = 0;
    uint64_t recurringInterfaceEmbedding = 0, recurringLocalDeclines = 0;
    uint64_t recurringAnalysisSites = 0, recurringReplaySites = 0, recurringSupportQueries = 0;
    std::size_t recurringFamilies = 0, recurringIndexEntries = 0;
    std::size_t recurringCandidates = 0, recurringAttempts = 0;
    std::size_t recurringActivations = 0, recurringDeclines = 0;
    std::size_t loopEntryTransfers = 0;
    std::size_t rearmingDischarged = 0;
    std::size_t rearmingRestored = 0;
    std::size_t rearmingDeferred = 0, deferredMaterialized = 0, latentSupportChecks = 0, latentSupportRetained = 0;
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
};
// Diagnostics and work from one discarded optional attempt. Its
// endpoints and causal state never participate in the returned construction.
struct DeclinedRecurringAttempt {
    SelectedFailure failure = SelectedFailure::None;
    std::string reason;
    Cut cut = NoAnalysisId;
    SelectedWork work;
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
    std::vector<SelectedRealizationChoice> realizationChoices;
    std::vector<SelectedChannel> channels;
    std::vector<SelectedRecurringActivation> activations;
    std::vector<SelectedRecurringRefusal> recurringRefusals;
    std::vector<SelectedFence> fences;
    std::vector<SelectedRestoration> restorations;
    std::vector<SelectedUpdate> updates;
    std::vector<SelectedPublicationSupport> publicationSupport;
    // Populated only from the final, entry-containing original-graph proof.
    std::vector<SelectedLoopInterface> loops;
    FrontierCheck certificate;
    SelectedWork work;
    std::optional<DeclinedRecurringAttempt> declinedRecurring;
    // The last optional first-use observation layer was declined while
    // retaining previously qualified storage/occurrence interfaces.
    std::optional<DeclinedRecurringAttempt> declinedFirstUse;
    // Native local-only admission retry; discarded work remains visible.
    std::optional<DeclinedRecurringAttempt> declinedObservation;
};

// F1--F8 construction service used by the live handoff pass.
// Fixed words are preserved in order. Unsupported typed effects are refused,
// never erased or delegated to another constructor on failure.
SelectedPlan constructSelectedPlan(const Program&, const Commands& fixed = {});

} // namespace mlir::pto::oahs
#endif
