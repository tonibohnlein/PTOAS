// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- PTOProtocolSync.cpp - Protocol-first synchronization pass --------===//
#include "PTO/Transforms/Passes.h"

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/DirectRepair.h"
#include "PTO/Transforms/ProtocolSync/LaneFrontierAnalysis.h"
#include "PTO/Transforms/ProtocolSync/LanePatternAnalysis.h"
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"
#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"
#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "PTO/Transforms/ProtocolSync/StorageTrackAnalysis.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <chrono>

namespace mlir::pto {
#define GEN_PASS_DEF_PTOPROTOCOLSYNC
#include "PTO/Transforms/Passes.h.inc"

namespace {

using ProtocolSyncClock = std::chrono::steady_clock;
using namespace protocol_sync;

std::uint64_t elapsedMicroseconds(ProtocolSyncClock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(ProtocolSyncClock::now() - start).count();
}

void collectScheduleStatistics(const StructuredSyncIR& schedule, ProtocolSyncStatistics& statistics)
{
    statistics.structuredRegions = schedule.getRegions().size();
    statistics.semanticActions = schedule.getSemanticActions().size();
    statistics.phases = schedule.getPhases().size();
    statistics.accesses = schedule.getAccesses().size();
    statistics.storageFamilies = schedule.getStorageFamilies().size();
    for (const SyncAccess& access : schedule.getAccesses()) {
        if (access.storage.unknownRange) {
            ++statistics.unknownRanges;
        } else {
            ++statistics.knownRanges;
        }
        const SyncStorageFamily* family = schedule.findStorageFamily(access.family);
        if (!family || !family->slotCount) {
            ++statistics.unknownCapacityAccesses;
        } else if (*family->slotCount == 1) {
            ++statistics.depthOneAccesses;
        } else if (*family->slotCount == 2) {
            ++statistics.depthTwoAccesses;
        }
        if (access.slot && access.slot->kind == SyncSlotExpressionKind::Unknown) {
            ++statistics.slotExpressionsUnknown;
        } else if (access.slot) {
            ++statistics.slotExpressionsKnown;
        }
    }
}

void printStatistics(
    StringRef function, const ProtocolSyncStatistics& statistics, StringRef status, StringRef failureStage,
    ProtocolSyncProducer producer = ProtocolSyncProducer::AnalysisOnly, StringRef plannerResult = "analysis-only",
    StringRef fallback = "none")
{
    llvm::json::Object record;
    record["kind"] = "protocol-sync-statistics";
    record["function"] = function.str();
    record["status"] = status.str();
    record["producer"] = stringifyProtocolSyncProducer(producer).str();
    if (!failureStage.empty()) {
        record["failure_stage"] = failureStage.str();
    }
    llvm::json::Object counts;
    counts["operations_visited"] = static_cast<std::int64_t>(statistics.operationsVisited);
    counts["operations_summarized"] = static_cast<std::int64_t>(statistics.operationsSummarized);
    counts["provider_lookup_attempts"] = static_cast<std::int64_t>(statistics.providerLookupAttempts);
    counts["provider_hits"] = static_cast<std::int64_t>(statistics.providerHits);
    counts["fixed_supplied_protocols"] = static_cast<std::int64_t>(statistics.fixedSuppliedProtocols);
    counts["hidden_event_reservations"] = static_cast<std::int64_t>(statistics.hiddenEventReservations);
    counts["regions"] = static_cast<std::int64_t>(statistics.structuredRegions);
    counts["semantic_actions"] = static_cast<std::int64_t>(statistics.semanticActions);
    counts["phases"] = static_cast<std::int64_t>(statistics.phases);
    counts["accesses"] = static_cast<std::int64_t>(statistics.accesses);
    counts["storage_families"] = static_cast<std::int64_t>(statistics.storageFamilies);
    counts["pipeline_stages"] = static_cast<std::int64_t>(statistics.pipelineStages);
    counts["known_ranges"] = static_cast<std::int64_t>(statistics.knownRanges);
    counts["unknown_ranges"] = static_cast<std::int64_t>(statistics.unknownRanges);
    counts["interval_index_queries"] = static_cast<std::int64_t>(statistics.intervalIndexQueries);
    counts["slot_expressions_known"] = static_cast<std::int64_t>(statistics.slotExpressionsKnown);
    counts["slot_expressions_unknown"] = static_cast<std::int64_t>(statistics.slotExpressionsUnknown);
    counts["depth_one_accesses"] = static_cast<std::int64_t>(statistics.depthOneAccesses);
    counts["depth_two_accesses"] = static_cast<std::int64_t>(statistics.depthTwoAccesses);
    counts["unknown_capacity_accesses"] = static_cast<std::int64_t>(statistics.unknownCapacityAccesses);
    counts["rejected_operations"] = static_cast<std::int64_t>(statistics.rejectedOperations);
    counts["legacy_parity_mismatches"] = static_cast<std::int64_t>(statistics.legacyParityMismatches);
    counts["generation_timelines_attempted"] = static_cast<std::int64_t>(statistics.generationTimelinesAttempted);
    counts["generation_timelines_admitted"] = static_cast<std::int64_t>(statistics.generationTimelinesAdmitted);
    counts["generation_timelines_rejected"] = static_cast<std::int64_t>(statistics.generationTimelinesRejected);
    counts["raw_access_endpoints_retained"] = static_cast<std::int64_t>(statistics.rawAccessEndpointsRetained);
    counts["raw_access_pairs"] = static_cast<std::int64_t>(statistics.rawAccessPairs);
    counts["raw_access_pair_completion_intrinsic"] =
        static_cast<std::int64_t>(statistics.rawAccessPairCompletionIntrinsic);
    counts["raw_access_pair_completion_pipe_barrier"] =
        static_cast<std::int64_t>(statistics.rawAccessPairCompletionPipeBarrier);
    counts["raw_access_pair_completion_unsupported_target"] =
        static_cast<std::int64_t>(statistics.rawAccessPairCompletionUnsupportedTarget);
    counts["raw_access_pair_completion_unsupported_mechanism"] =
        static_cast<std::int64_t>(statistics.rawAccessPairCompletionUnsupportedMechanism);
    counts["raw_access_pair_completion_not_applicable"] =
        static_cast<std::int64_t>(statistics.rawAccessPairCompletionNotApplicable);
    counts["channel_candidates_attempted"] = static_cast<std::int64_t>(statistics.channelCandidatesAttempted);
    counts["channel_candidates_admitted"] = static_cast<std::int64_t>(statistics.channelCandidatesAdmitted);
    counts["channel_candidates_rejected"] = static_cast<std::int64_t>(statistics.channelCandidatesRejected);
    counts["execution_lanes"] = static_cast<std::int64_t>(statistics.executionLanes);
    counts["lane_occurrences"] = static_cast<std::int64_t>(statistics.laneOccurrences);
    counts["lane_frontier_experiments"] = static_cast<std::int64_t>(statistics.laneFrontierExperiments);
    counts["lane_ready_experiments"] = static_cast<std::int64_t>(statistics.laneReadyExperiments);
    counts["lane_release_experiments"] = static_cast<std::int64_t>(statistics.laneReleaseExperiments);
    counts["lane_residual_experiments"] = static_cast<std::int64_t>(statistics.laneResidualExperiments);
    counts["lane_frontiers_found"] = static_cast<std::int64_t>(statistics.laneFrontiersFound);
    counts["same_lane_frontiers_found"] = static_cast<std::int64_t>(statistics.sameLaneFrontiersFound);
    counts["cross_lane_frontiers_found"] = static_cast<std::int64_t>(statistics.crossLaneFrontiersFound);
    counts["linear_frontiers_coalesced"] = static_cast<std::int64_t>(statistics.linearFrontiersCoalesced);
    counts["choice_boundary_frontiers_found"] = static_cast<std::int64_t>(statistics.choiceBoundaryFrontiersFound);
    counts["frontiers_found_for_rejected_channels"] =
        static_cast<std::int64_t>(statistics.frontiersFoundForRejectedChannels);
    counts["lane_pattern_candidates"] = static_cast<std::int64_t>(statistics.lanePatternCandidates);
    counts["shared_one_shot_frontiers"] = static_cast<std::int64_t>(statistics.sharedOneShotFrontiers);
    counts["same_lane_completion_cuts"] = static_cast<std::int64_t>(statistics.sameLaneCompletionCuts);
    counts["choice_balanced_round_trips"] = static_cast<std::int64_t>(statistics.choiceBalancedRoundTrips);
    counts["lane_pattern_target_supported"] = static_cast<std::int64_t>(statistics.lanePatternTargetSupported);
    counts["lane_pattern_target_rejected"] = static_cast<std::int64_t>(statistics.lanePatternTargetRejected);
    counts["lane_pattern_checkpoint_e_admitted"] = static_cast<std::int64_t>(statistics.lanePatternCheckpointEAdmitted);
    counts["lane_pattern_checkpoint_e_rejected"] = static_cast<std::int64_t>(statistics.lanePatternCheckpointERejected);
    counts["lane_pattern_checkpoint_e_not_applicable"] =
        static_cast<std::int64_t>(statistics.lanePatternCheckpointENotApplicable);
    counts["lane_pattern_logical_cost"] = static_cast<std::int64_t>(statistics.lanePatternLogicalCost);
    counts["lane_pattern_steady_state_actions"] = static_cast<std::int64_t>(statistics.lanePatternSteadyStateActions);
    counts["storage_track_accesses_attempted"] = static_cast<std::int64_t>(statistics.storageTrackAccessesAttempted);
    counts["storage_track_accesses_projected"] = static_cast<std::int64_t>(statistics.storageTrackAccessesProjected);
    counts["storage_track_accesses_unprojected"] =
        static_cast<std::int64_t>(statistics.storageTrackAccessesUnprojected);
    counts["storage_tracks"] = static_cast<std::int64_t>(statistics.storageTracks);
    counts["storage_track_occurrences"] = static_cast<std::int64_t>(statistics.storageTrackOccurrences);
    counts["storage_tracks_multiple_families"] = static_cast<std::int64_t>(statistics.storageTracksMultipleFamilies);
    counts["storage_tracks_uncertain_alias"] = static_cast<std::int64_t>(statistics.storageTracksUncertainAlias);
    counts["storage_tracks_multiple_physical_cores"] =
        static_cast<std::int64_t>(statistics.storageTracksMultiplePhysicalCores);
    counts["storage_projection_exact_accesses"] = static_cast<std::int64_t>(statistics.storageProjectionExactAccesses);
    counts["storage_projection_access_mask_mismatches"] =
        static_cast<std::int64_t>(statistics.storageProjectionAccessMaskMismatches);
    counts["storage_projection_pair_relations"] = static_cast<std::int64_t>(statistics.storageProjectionPairRelations);
    counts["storage_projection_overlap_pairs"] = static_cast<std::int64_t>(statistics.storageProjectionOverlapPairs);
    counts["storage_projection_disjoint_pairs"] = static_cast<std::int64_t>(statistics.storageProjectionDisjointPairs);
    counts["storage_projection_overlap_pairs_missing_track"] =
        static_cast<std::int64_t>(statistics.storageProjectionOverlapPairsMissingTrack);
    counts["storage_projection_disjoint_pairs_sharing_track"] =
        static_cast<std::int64_t>(statistics.storageProjectionDisjointPairsSharingTrack);
    counts["storage_projection_read_read_overlap_pairs"] =
        static_cast<std::int64_t>(statistics.storageProjectionReadReadOverlapPairs);
    counts["storage_projection_accumulator_read_read_overlap_pairs"] =
        static_cast<std::int64_t>(statistics.storageProjectionAccumulatorReadReadOverlapPairs);
    counts["storage_projection_cross_lane_read_read_overlap_pairs"] =
        static_cast<std::int64_t>(statistics.storageProjectionCrossLaneReadReadOverlapPairs);
    counts["storage_projection_overlap_components"] =
        static_cast<std::int64_t>(statistics.storageProjectionOverlapComponents);
    counts["storage_projection_maximum_atoms_per_component"] =
        static_cast<std::int64_t>(statistics.storageProjectionMaximumAtomsPerComponent);
    counts["storage_projection_maximum_atoms_per_access"] =
        static_cast<std::int64_t>(statistics.storageProjectionMaximumAtomsPerAccess);
    counts["storage_transition_frontiers"] = static_cast<std::int64_t>(statistics.storageTransitionFrontiers);
    counts["storage_lifecycle_transitions"] = static_cast<std::int64_t>(statistics.storageLifecycleTransitions);
    counts["storage_completion_transitions"] = static_cast<std::int64_t>(statistics.storageCompletionTransitions);
    counts["storage_residual_transitions"] = static_cast<std::int64_t>(statistics.storageResidualTransitions);
    counts["storage_raw_pairs_covered"] = static_cast<std::int64_t>(statistics.storageRawPairsCovered);
    counts["storage_raw_pairs_uncovered"] = static_cast<std::int64_t>(statistics.storageRawPairsUncovered);
    counts["storage_transition_target_supported"] =
        static_cast<std::int64_t>(statistics.storageTransitionTargetSupported);
    counts["storage_transition_target_rejected"] =
        static_cast<std::int64_t>(statistics.storageTransitionTargetRejected);
    counts["storage_transition_pair_memberships"] =
        static_cast<std::int64_t>(statistics.storageTransitionPairMemberships);
    counts["storage_transition_pairs_covered_once"] =
        static_cast<std::int64_t>(statistics.storageTransitionPairsCoveredOnce);
    counts["storage_transition_pairs_multiply_covered"] =
        static_cast<std::int64_t>(statistics.storageTransitionPairsMultiplyCovered);
    counts["storage_transition_invalid_pair_memberships"] =
        static_cast<std::int64_t>(statistics.storageTransitionInvalidPairMemberships);
    counts["storage_transition_track_mask_mismatches"] =
        static_cast<std::int64_t>(statistics.storageTransitionTrackMaskMismatches);
    counts["storage_transition_linear_frontier_memberships"] =
        static_cast<std::int64_t>(statistics.storageTransitionLinearFrontierMemberships);
    counts["storage_transition_linear_frontier_mismatches"] =
        static_cast<std::int64_t>(statistics.storageTransitionLinearFrontierMismatches);
    counts["storage_transition_frontier_memberships_not_linear"] =
        static_cast<std::int64_t>(statistics.storageTransitionFrontierMembershipsNotLinear);
    counts["storage_lifecycle_components_attempted"] =
        static_cast<std::int64_t>(statistics.storageLifecycleComponentsAttempted);
    counts["storage_lifecycles_reconstructed"] = static_cast<std::int64_t>(statistics.storageLifecyclesReconstructed);
    counts["storage_lifecycle_e_matches"] = static_cast<std::int64_t>(statistics.storageLifecycleEMatches);
    counts["storage_lifecycle_e_mismatches"] = static_cast<std::int64_t>(statistics.storageLifecycleEMismatches);
    counts["storage_lifecycle_independent_only"] =
        static_cast<std::int64_t>(statistics.storageLifecycleIndependentOnly);
    counts["storage_lifecycle_e_only"] = static_cast<std::int64_t>(statistics.storageLifecycleEOnly);
    counts["concrete_verifier_runs"] = static_cast<std::int64_t>(statistics.concreteVerifierRuns);
    counts["concrete_verifier_accepted"] = static_cast<std::int64_t>(statistics.concreteVerifierAccepted);
    counts["concrete_verifier_rejected"] = static_cast<std::int64_t>(statistics.concreteVerifierRejected);
    counts["interpreter_transitions"] = static_cast<std::int64_t>(statistics.interpreterTransitions);
    counts["interpreter_peak_states"] = static_cast<std::int64_t>(statistics.interpreterPeakStates);
    counts["memory_pair_tests"] = static_cast<std::int64_t>(statistics.memoryPairTests);
    counts["no_alias_results"] = static_cast<std::int64_t>(statistics.noAliasResults);
    counts["may_alias_results"] = static_cast<std::int64_t>(statistics.mayAliasResults);
    counts["unknown_alias_results"] = static_cast<std::int64_t>(statistics.unknownAliasResults);
    counts["token_certificate_transitions"] = static_cast<std::int64_t>(statistics.tokenCertificateTransitions);
    counts["protocol_candidates"] = static_cast<std::int64_t>(statistics.protocolCandidates);
    counts["protocol_plans_attempted"] = static_cast<std::int64_t>(statistics.protocolPlansAttempted);
    counts["protocol_plans_admitted"] = static_cast<std::int64_t>(statistics.protocolPlansAdmitted);
    counts["protocol_plans_rejected"] = static_cast<std::int64_t>(statistics.protocolPlansRejected);
    counts["selected_one_shot_protocols"] = static_cast<std::int64_t>(statistics.selectedOneShotProtocols);
    counts["selected_directed_event_pairs"] = static_cast<std::int64_t>(statistics.selectedDirectedEventPairs);
    counts["selected_same_pipe_barriers"] = static_cast<std::int64_t>(statistics.selectedSamePipeBarriers);
    counts["selected_tail_drains"] = static_cast<std::int64_t>(statistics.selectedTailDrains);
    counts["selected_ready_release_protocols"] = static_cast<std::int64_t>(statistics.selectedReadyReleaseProtocols);
    counts["selected_ready_release_lanes"] = static_cast<std::int64_t>(statistics.selectedReadyReleaseLanes);
    counts["direct_repair_candidates"] = static_cast<std::int64_t>(statistics.directRepairCandidates);
    counts["direct_repair_shared_candidates"] = static_cast<std::int64_t>(statistics.directRepairSharedCandidates);
    counts["selected_direct_repairs"] = static_cast<std::int64_t>(statistics.selectedDirectRepairs);
    counts["direct_repair_uncovered"] = static_cast<std::int64_t>(statistics.directRepairUncovered);
    counts["mixed_selection_candidates"] = static_cast<std::int64_t>(statistics.mixedSelectionCandidates);
    counts["reverse_deletion_attempts"] = static_cast<std::int64_t>(statistics.reverseDeletionAttempts);
    counts["reverse_deletion_removed"] = static_cast<std::int64_t>(statistics.reverseDeletionRemoved);
    counts["complete_worlds_attempted"] = static_cast<std::int64_t>(statistics.completeWorldsAttempted);
    counts["complete_worlds_feasible"] = static_cast<std::int64_t>(statistics.completeWorldsFeasible);
    counts["selected_world_event_pairs"] = static_cast<std::int64_t>(statistics.selectedWorldEventPairs);
    counts["selected_world_targeted_barriers"] = static_cast<std::int64_t>(statistics.selectedWorldTargetedBarriers);
    counts["selected_world_fixed_exit_drains"] = static_cast<std::int64_t>(statistics.selectedWorldFixedExitDrains);
    counts["allocation_retries"] = static_cast<std::int64_t>(statistics.allocationRetries);
    counts["event_domains"] = static_cast<std::int64_t>(statistics.eventDomains);
    counts["max_event_domain_pressure"] = static_cast<std::int64_t>(statistics.maxEventDomainPressure);
    counts["maximum_event_id"] =
        statistics.maximumEventIdPlusOne == 0 ? -1 : static_cast<std::int64_t>(statistics.maximumEventIdPlusOne - 1);
    counts["logical_actions"] = static_cast<std::int64_t>(statistics.logicalActions);
    counts["residual_obligations"] = static_cast<std::int64_t>(statistics.residualObligations);
    counts["allocation_graph_vertices"] = static_cast<std::int64_t>(statistics.allocationGraphVertices);
    counts["allocation_graph_edges"] = static_cast<std::int64_t>(statistics.allocationGraphEdges);
    counts["allocation_backtracking_nodes"] = static_cast<std::int64_t>(statistics.allocationBacktrackingNodes);
    counts["allocation_search_limit_hits"] = static_cast<std::int64_t>(statistics.allocationSearchLimitHits);
    counts["materialization_transitions"] = static_cast<std::int64_t>(statistics.materializationTransitions);
    counts["verifier_transitions"] = static_cast<std::int64_t>(statistics.verifierTransitions);
    record["counts"] = std::move(counts);
    llvm::json::Object generationRejections;
    for (const auto& [reason, count] : statistics.generationRejections) {
        generationRejections[reason] = static_cast<std::int64_t>(count);
    }
    record["generation_rejections"] = std::move(generationRejections);
    llvm::json::Object channelRejections;
    for (const auto& [reason, count] : statistics.channelRejections) {
        channelRejections[reason] = static_cast<std::int64_t>(count);
    }
    record["channel_rejections"] = std::move(channelRejections);
    llvm::json::Object residualObligations;
    for (const auto& [kind, count] : statistics.residualObligationsByKind) {
        residualObligations[kind] = static_cast<std::int64_t>(count);
    }
    record["residual_obligations_by_kind"] = std::move(residualObligations);
    llvm::json::Object laneFrontierRejections;
    for (const auto& [reason, count] : statistics.laneFrontierRejections) {
        laneFrontierRejections[reason] = static_cast<std::int64_t>(count);
    }
    record["lane_frontier_rejections"] = std::move(laneFrontierRejections);
    llvm::json::Object lanePatternKinds;
    for (const auto& [kind, count] : statistics.lanePatternKinds) {
        lanePatternKinds[kind] = static_cast<std::int64_t>(count);
    }
    record["lane_pattern_kinds"] = std::move(lanePatternKinds);
    llvm::json::Object storageProjectionRejections;
    for (const auto& [reason, count] : statistics.storageProjectionRejections) {
        storageProjectionRejections[reason] = static_cast<std::int64_t>(count);
    }
    record["storage_projection_rejections"] = std::move(storageProjectionRejections);
    llvm::json::Object storageTransitionKinds;
    for (const auto& [kind, count] : statistics.storageTransitionKinds) {
        storageTransitionKinds[kind] = static_cast<std::int64_t>(count);
    }
    record["storage_transition_kinds"] = std::move(storageTransitionKinds);
    llvm::json::Object storageLifecycleRejections;
    for (const auto& [reason, count] : statistics.storageLifecycleRejections) {
        storageLifecycleRejections[reason] = static_cast<std::int64_t>(count);
    }
    record["storage_lifecycle_rejections"] = std::move(storageLifecycleRejections);
    llvm::json::Object storageLifecycleIndependentERejections;
    for (const auto& [reason, count] : statistics.storageLifecycleIndependentERejections) {
        storageLifecycleIndependentERejections[reason] = static_cast<std::int64_t>(count);
    }
    record["storage_lifecycle_independent_e_rejections"] = std::move(storageLifecycleIndependentERejections);
    record["planner_result"] = plannerResult.str();
    record["fallback"] = fallback.str();
    llvm::json::Object timing;
    timing["semantic_extraction"] = static_cast<std::int64_t>(statistics.semanticExtractionUs);
    timing["schedule_construction"] = static_cast<std::int64_t>(statistics.scheduleConstructionUs);
    timing["legacy_snapshot"] = static_cast<std::int64_t>(statistics.legacySnapshotUs);
    timing["semantic_context"] = static_cast<std::int64_t>(statistics.semanticContextUs);
    timing["legacy_comparison"] = static_cast<std::int64_t>(statistics.legacyComparisonUs);
    timing["storage_analysis"] = static_cast<std::int64_t>(statistics.storageAnalysisUs);
    timing["generation_analysis"] = static_cast<std::int64_t>(statistics.generationAnalysisUs);
    timing["channel_analysis"] = static_cast<std::int64_t>(statistics.channelAnalysisUs);
    timing["interpretation"] = static_cast<std::int64_t>(statistics.interpretationUs);
    timing["lane_frontier_analysis"] = static_cast<std::int64_t>(statistics.laneFrontierAnalysisUs);
    timing["lane_pattern_analysis"] = static_cast<std::int64_t>(statistics.lanePatternAnalysisUs);
    timing["storage_track_analysis"] = static_cast<std::int64_t>(statistics.storageTrackAnalysisUs);
    timing["planning"] = static_cast<std::int64_t>(statistics.planningUs);
    timing["allocation"] = static_cast<std::int64_t>(statistics.allocationUs);
    timing["materialization"] = static_cast<std::int64_t>(statistics.materializationUs);
    timing["verification"] = static_cast<std::int64_t>(statistics.verificationUs);
    timing["total"] = static_cast<std::int64_t>(statistics.totalUs);
    record["time_us"] = std::move(timing);
    llvm::errs() << llvm::json::Value(std::move(record)) << '\n';
}

struct PendingProtocolSyncStatistics {
    std::string function;
    ProtocolSyncStatistics statistics;
    std::string status;
    std::string failureStage;
    ProtocolSyncProducer producer = ProtocolSyncProducer::AnalysisOnly;
    std::string plannerResult;
    std::string fallback;
    bool stagedMutation = false;
};

struct PTOProtocolSyncPass : public impl::PTOProtocolSyncBase<PTOProtocolSyncPass> {
    PTOProtocolSyncPass() = default;

    explicit PTOProtocolSyncPass(const PTOProtocolSyncOptions& options)
    {
        dumpMode = options.dumpMode;
        executionMode = options.executionMode;
        fallbackMode = options.fallbackMode;
        statistics = options.statistics;
    }

    void runOnOperation() final
    {
        pendingStatistics.clear();
        const bool analysisOnly = executionMode == "analysis";
        const bool emitOneShot = executionMode == "one-shot";
        const bool emitReadyRelease = executionMode == "ready-release";
        const bool emitDirectRepair = executionMode == "direct-repair";
        const bool emitMixed = executionMode == "mixed";
        if (!analysisOnly && !emitOneShot && !emitReadyRelease && !emitDirectRepair && !emitMixed) {
            getOperation().emitError("unknown ProtocolSync execution mode '")
                << executionMode << "'; expected 'analysis', 'one-shot', 'ready-release', 'direct-repair', or 'mixed'";
            signalPassFailure();
            return;
        }
        if (fallbackMode != "legacy" && fallbackMode != "fail") {
            getOperation().emitError("unknown ProtocolSync fallback mode '")
                << fallbackMode << "'; expected 'legacy' or 'fail'";
            signalPassFailure();
            return;
        }
        const bool validDumpMode = dumpMode == "none" || dumpMode == "schedule" || dumpMode == "channels" ||
                                   dumpMode == "lane-frontiers" || dumpMode == "storage-tracks" ||
                                   dumpMode == "concrete-verification" || dumpMode == "residuals" ||
                                   dumpMode == "plan";
        if (!validDumpMode) {
            getOperation().emitError("unknown ProtocolSync dump mode '")
                << dumpMode
                << "'; expected 'none', 'schedule', 'channels', 'lane-frontiers', 'storage-tracks', "
                   "'concrete-verification', 'residuals', or 'plan'";
            signalPassFailure();
            return;
        }
        if (dumpMode == "plan" && analysisOnly) {
            getOperation().emitError("ProtocolSync plan dumps require an emission execution mode");
            signalPassFailure();
            return;
        }
        if (dumpMode == "concrete-verification" && !analysisOnly) {
            getOperation().emitError("ProtocolSync concrete-verification dumps require analysis execution mode");
            signalPassFailure();
            return;
        }
        if (analysisOnly) {
            SmallVector<func::FuncOp, 8> functions;
            getOperation().walk([&](func::FuncOp function) { functions.push_back(function); });
            for (func::FuncOp function : functions) {
                if (function.isDeclaration()) {
                    continue;
                }
                if (failed(analyzeFunction(function, false, false, false, false))) {
                    flushStatistics(false);
                    signalPassFailure();
                    return;
                }
                flushStatistics(true);
            }
            markAllAnalysesPreserved();
            return;
        }

        OwningOpRef<ModuleOp> stagingModule = cast<ModuleOp>(getOperation()->clone());
        SmallVector<func::FuncOp, 8> stagedFunctions;
        stagingModule->walk([&](func::FuncOp function) { stagedFunctions.push_back(function); });
        for (func::FuncOp function : stagedFunctions) {
            if (function.isDeclaration()) {
                continue;
            }
            if (failed(analyzeFunction(function, emitOneShot, emitReadyRelease, emitDirectRepair, emitMixed))) {
                flushStatistics(false);
                signalPassFailure();
                return;
            }
        }
        if (failed(mlir::verify(*stagingModule))) {
            getOperation().emitError("ProtocolSync rejected its staged module; original IR is unchanged");
            flushStatistics(false);
            signalPassFailure();
            return;
        }

        SmallVector<func::FuncOp, 8> originalFunctions;
        getOperation().walk([&](func::FuncOp function) { originalFunctions.push_back(function); });
        if (originalFunctions.size() != stagedFunctions.size()) {
            getOperation().emitError("ProtocolSync staged-module function correspondence changed");
            flushStatistics(false);
            signalPassFailure();
            return;
        }
        for (auto [original, staged] : llvm::zip_equal(originalFunctions, stagedFunctions)) {
            if (original.getSymName() != staged.getSymName() || original.isDeclaration() != staged.isDeclaration() ||
                original.getFunctionType() != staged.getFunctionType()) {
                getOperation().emitError("ProtocolSync staged-module function correspondence is invalid");
                flushStatistics(false);
                signalPassFailure();
                return;
            }
        }
        for (auto [original, staged] : llvm::zip_equal(originalFunctions, stagedFunctions)) {
            if (!original.isDeclaration()) {
                original.getBody().takeBody(staged.getBody());
            }
        }
        flushStatistics(true);
    }

private:
    void recordStatistics(
        func::FuncOp function, const ProtocolSyncStatistics& result, StringRef status, StringRef failureStage,
        ProtocolSyncProducer producer = ProtocolSyncProducer::AnalysisOnly, StringRef plannerResult = "analysis-only",
        StringRef fallback = "none", bool stagedMutation = false)
    {
        if (!statistics) {
            return;
        }
        pendingStatistics.push_back(
            {function.getSymName().str(), result, status.str(), failureStage.str(), producer, plannerResult.str(),
             fallback.str(), stagedMutation});
    }

    void flushStatistics(bool transactionCommitted)
    {
        for (const PendingProtocolSyncStatistics& report : pendingStatistics) {
            if (!transactionCommitted && report.stagedMutation) {
                printStatistics(
                    report.function, report.statistics, "rolled-back", "module-transaction", report.producer,
                    "rolled-back", "module-rollback");
                continue;
            }
            printStatistics(
                report.function, report.statistics, report.status, report.failureStage, report.producer,
                report.plannerResult, report.fallback);
        }
        pendingStatistics.clear();
    }

    LogicalResult handleUnsupported(
        func::FuncOp function, ProtocolSyncStatistics& result, StringRef failureStage, StringRef fallbackReason,
        bool allowFallback, ProtocolSyncClock::time_point totalStart)
    {
        if (allowFallback && fallbackMode == "legacy") {
            if (failed(runPTOInsertSync(function))) {
                result.totalUs = elapsedMicroseconds(totalStart);
                recordStatistics(
                    function, result, "internal-error", "legacy-fallback", ProtocolSyncProducer::InternalError,
                    "internal-error", fallbackReason);
                function.emitError("ProtocolSync legacy fallback failed internally");
                return failure();
            }
            result.totalUs = elapsedMicroseconds(totalStart);
            const std::string plannerResult = (Twine("legacy-fallback-") + fallbackReason).str();
            const ProtocolSyncProducer producer = fallbackReason == "resource-infeasible" ?
                                                      ProtocolSyncProducer::LegacyFallbackResourceInfeasible :
                                                      ProtocolSyncProducer::LegacyFallbackUnsupported;
            recordStatistics(
                function, result, "ok", "", producer, plannerResult, fallbackReason,
                /*stagedMutation=*/true);
            return success();
        }
        result.totalUs = elapsedMicroseconds(totalStart);
        recordStatistics(
            function, result, "unsupported", failureStage, ProtocolSyncProducer::FailClosedPolicy, "unsupported",
            allowFallback ? "disabled" : "not-permitted");
        function.emitError("ProtocolSync ") << executionMode << " mode found no complete supported plan";
        return failure();
    }

    FailureOr<SyncInterpretationResult> evaluateWorld(
        func::FuncOp function, const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        const SyncSelectedWorld& world, ProtocolSyncStatistics& result, bool allowDump = true)
    {
        const ProtocolSyncClock::time_point start = ProtocolSyncClock::now();
        FailureOr<SyncInterpretationResult> interpretation =
            interpretSelectedWorld(schedule, stages, timelines, channels, world, &result);
        result.interpretationUs += elapsedMicroseconds(start);
        const bool shouldDump = allowDump && succeeded(interpretation) && dumpMode == "residuals";
        if (shouldDump) {
            printResidualObligations(function, world, *interpretation, llvm::errs());
        }
        return interpretation;
    }

    LogicalResult emitDirectRepairPlan(
        func::FuncOp function, const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        ProtocolSyncStatistics& result, ProtocolSyncClock::time_point totalStart)
    {
        SyncSelectedWorld world;
        FailureOr<SyncInterpretationResult> initial =
            evaluateWorld(function, schedule, stages, timelines, channels, world, result, false);
        if (failed(initial)) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "selected-world", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync direct-repair interpretation failed internally");
            return failure();
        }

        ProtocolSyncClock::time_point start = ProtocolSyncClock::now();
        ++result.protocolPlansAttempted;
        FailureOr<SyncDirectRepairPlan> plan = buildDirectRepairPlan(schedule, stages, initial->obligations, &result);
        result.planningUs = elapsedMicroseconds(start);
        const bool validPlan = succeeded(plan) && succeeded(verifyDirectRepairPlan(
                                                      schedule, stages, initial->obligations, *plan, &result));
        if (!validPlan) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "direct-repair-planning", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync direct-repair planning failed internally");
            return failure();
        }
        if (dumpMode == "plan" && plan->status != SyncDirectRepairPlanStatus::Ready) {
            printDirectRepairPlan(function, *plan, llvm::errs());
        }
        const bool internalRejection = llvm::any_of(plan->rejections, [](const auto& rejection) {
            return rejection.reason == SyncDirectRepairRejection::InternalInvariant;
        });
        if (internalRejection) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "direct-repair-planning", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync direct-repair planner violated an internal invariant");
            return failure();
        }
        if (!plan->isComplete()) {
            if (dumpMode == "residuals") {
                printResidualObligations(function, world, *initial, llvm::errs());
            }
            ++result.protocolPlansRejected;
            const bool targetRejection = llvm::any_of(plan->rejections, [](const auto& rejection) {
                return rejection.reason == SyncDirectRepairRejection::UnsupportedTarget;
            });
            result.totalUs = elapsedMicroseconds(totalStart);
            return handleUnsupported(
                function, result, "direct-repair-planning", "unsupported", !targetRejection, totalStart);
        }
        if (plan->status == SyncDirectRepairPlanStatus::Empty) {
            ++result.protocolPlansAdmitted;
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "ok", "", ProtocolSyncProducer::ProtocolPlusDirectResiduals, "no-op", "none");
            return success();
        }

        start = ProtocolSyncClock::now();
        const bool validAllocation =
            succeeded(allocateDirectRepairEvents(schedule, *plan, &result)) &&
            succeeded(verifyDirectRepairPlan(schedule, stages, initial->obligations, *plan, &result));
        if (!validAllocation) {
            result.allocationUs = elapsedMicroseconds(start);
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "direct-repair-allocation", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync direct-repair allocation failed internally");
            return failure();
        }
        result.allocationUs = elapsedMicroseconds(start);
        if (dumpMode == "plan") {
            printDirectRepairPlan(function, *plan, llvm::errs());
        }
        if (plan->status == SyncDirectRepairPlanStatus::ResourceInfeasible) {
            ++result.protocolPlansRejected;
            result.totalUs = elapsedMicroseconds(totalStart);
            return handleUnsupported(
                function, result, "direct-repair-allocation", "resource-infeasible", true, totalStart);
        }

        SmallVector<SyncDirectCandidateId, 8> selected;
        for (const SyncDirectRepairCandidate& candidate : plan->candidates) {
            selected.push_back(candidate.id);
            ++result.selectedDirectRepairs;
            result.logicalActions += candidate.kind == SyncDirectRepairKind::DirectedEvent ? 2 : 1;
        }
        if (failed(applyDirectRepairCandidates(*plan, initial->obligations, selected, world))) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "direct-repair-selected-world", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync direct-repair selected-world construction failed internally");
            return failure();
        }
        FailureOr<SyncInterpretationResult> final =
            evaluateWorld(function, schedule, stages, timelines, channels, world, result);
        const bool completeWorld = succeeded(final) && final->isComplete();
        if (!completeWorld) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "direct-repair-selected-world", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync direct-repair selected world is not obligation-complete");
            return failure();
        }
        ++result.protocolPlansAdmitted;
        if (failed(materializeAndVerifyDirectRepairPlanInDisposableModule(
                schedule, stages, initial->obligations, *plan, &result))) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "direct-repair-materialization-verification",
                ProtocolSyncProducer::InternalError, "rejected");
            return failure();
        }
        result.totalUs = elapsedMicroseconds(totalStart);
        recordStatistics(
            function, result, "ok", "", ProtocolSyncProducer::ProtocolPlusDirectResiduals, "materialized-direct-repair",
            "none", /*stagedMutation=*/true);
        return success();
    }

    LogicalResult emitMixedPlan(
        func::FuncOp function, const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        const SyncSemanticContext& context, ProtocolSyncStatistics& result, ProtocolSyncClock::time_point totalStart)
    {
        ProtocolSyncClock::time_point start = ProtocolSyncClock::now();
        ++result.protocolPlansAttempted;
        FailureOr<SyncMixedProtocolPlan> plan =
            buildMixedProtocolPlan(schedule, stages, timelines, channels, true, &result);
        result.planningUs = elapsedMicroseconds(start);
        const LogicalResult verified =
            failed(plan) ? failure() : verifyMixedProtocolPlan(schedule, stages, timelines, channels, *plan, &result);
        if (failed(verified)) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "mixed-selection", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync mixed selection failed internally");
            return failure();
        }
        if (dumpMode == "plan" && !plan->isComplete()) {
            printMixedProtocolPlan(function, *plan, llvm::errs());
        }
        if (plan->status == SyncMixedPlanStatus::Unsupported) {
            ++result.protocolPlansRejected;
            const bool targetRejection = llvm::any_of(plan->failures, [](const SyncMixedPlanFailure& failure) {
                return failure.reason == SyncMixedPlanRejection::UnsupportedTarget;
            });
            result.totalUs = elapsedMicroseconds(totalStart);
            return handleUnsupported(function, result, "mixed-selection", "unsupported", !targetRejection, totalStart);
        }
        if (plan->status == SyncMixedPlanStatus::Empty) {
            ++result.protocolPlansAdmitted;
            result.totalUs = elapsedMicroseconds(totalStart);
            if (dumpMode == "plan") {
                printMixedProtocolPlan(function, *plan, llvm::errs());
            }
            recordStatistics(
                function, result, "ok", "", ProtocolSyncProducer::ProtocolPlusDirectResiduals, "no-op", "none");
            return success();
        }

        start = ProtocolSyncClock::now();
        const bool selectionProvedResourceInfeasible = plan->status == SyncMixedPlanStatus::ResourceInfeasible;
        if (!selectionProvedResourceInfeasible && failed(allocateMixedProtocolEvents(schedule, *plan, &result))) {
            result.allocationUs = elapsedMicroseconds(start);
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "mixed-allocation", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync mixed allocation failed internally");
            return failure();
        }
        const bool retryAfterResourceFailure = !selectionProvedResourceInfeasible &&
                                               plan->status == SyncMixedPlanStatus::ResourceInfeasible &&
                                               plan->hasProtocol();
        if (retryAfterResourceFailure) {
            ++result.allocationRetries;
            ++result.protocolPlansRejected;
            ++result.protocolPlansAttempted;
            FailureOr<SyncMixedProtocolPlan> retry =
                buildMixedProtocolPlan(schedule, stages, timelines, channels, false, &result);
            const LogicalResult retryVerified =
                failed(retry) ? failure() :
                                verifyMixedProtocolPlan(schedule, stages, timelines, channels, *retry, &result);
            if (failed(retryVerified)) {
                result.allocationUs = elapsedMicroseconds(start);
                result.totalUs = elapsedMicroseconds(totalStart);
                ++result.protocolPlansRejected;
                recordStatistics(
                    function, result, "internal-error", "mixed-retry", ProtocolSyncProducer::InternalError,
                    "internal-error");
                function.emitError("ProtocolSync mixed allocation retry failed internally");
                return failure();
            }
            const LogicalResult retryAllocated =
                retry->isComplete() ? allocateMixedProtocolEvents(schedule, *retry, &result) : success();
            if (failed(retryAllocated)) {
                result.allocationUs = elapsedMicroseconds(start);
                result.totalUs = elapsedMicroseconds(totalStart);
                ++result.protocolPlansRejected;
                recordStatistics(
                    function, result, "internal-error", "mixed-retry-allocation", ProtocolSyncProducer::InternalError,
                    "internal-error");
                function.emitError("ProtocolSync mixed retry allocation failed internally");
                return failure();
            }
            plan = std::move(*retry);
        }
        result.allocationUs = elapsedMicroseconds(start);
        if (!plan->isComplete()) {
            if (dumpMode == "plan") {
                printMixedProtocolPlan(function, *plan, llvm::errs());
            }
            ++result.protocolPlansRejected;
            result.totalUs = elapsedMicroseconds(totalStart);
            const bool resourceInfeasible =
                retryAfterResourceFailure || plan->status == SyncMixedPlanStatus::ResourceInfeasible;
            const StringRef reason = resourceInfeasible ? "resource-infeasible" : "unsupported";
            return handleUnsupported(function, result, "mixed-allocation", reason, true, totalStart);
        }
        if (failed(verifyMixedProtocolPlan(schedule, stages, timelines, channels, *plan, &result))) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "mixed-post-allocation", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync mixed allocated plan failed independent verification");
            return failure();
        }
        FailureOr<SyncInterpretationResult> final =
            evaluateWorld(function, schedule, stages, timelines, channels, plan->selectedWorld, result);
        const bool finalWorldComplete = succeeded(final) && final->isComplete();
        if (!finalWorldComplete) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "mixed-selected-world", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync mixed selected world is not obligation-complete");
            return failure();
        }
        if (dumpMode == "plan") {
            printMixedProtocolPlan(function, *plan, llvm::errs());
        }
        ++result.protocolPlansAdmitted;
        if (failed(materializeAndVerifyMixedProtocolPlanInDisposableModule(
                schedule, stages, timelines, channels, *plan, context, &result))) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "mixed-materialization-verification",
                ProtocolSyncProducer::InternalError, "rejected");
            return failure();
        }
        result.totalUs = elapsedMicroseconds(totalStart);
        recordStatistics(
            function, result, "ok", "", ProtocolSyncProducer::ProtocolPlusDirectResiduals, "materialized-mixed", "none",
            /*stagedMutation=*/true);
        return success();
    }

    LogicalResult emitReadyReleasePlan(
        func::FuncOp function, const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        const LaneFrontierAnalysisResult& laneFrontiers, const StorageTrackAnalysisResult& storageTracks,
        ProtocolSyncStatistics& result, ProtocolSyncClock::time_point totalStart)
    {
        ProtocolSyncClock::time_point start = ProtocolSyncClock::now();
        ++result.protocolPlansAttempted;
        FailureOr<SyncReadyReleasePlan> plan =
            buildReadyReleaseProtocolPlan(schedule, stages, timelines, channels, &result);
        result.planningUs = elapsedMicroseconds(start);
        if (failed(plan)) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "planning", ProtocolSyncProducer::InternalError, "internal-error");
            function.emitError("ProtocolSync ReadyRelease planning failed internally");
            return failure();
        }

        SyncSelectedWorld world;
        if (plan->status == SyncReadyReleasePlanStatus::Ready) {
            FailureOr<SyncSelectedWorld> selected = buildSelectedWorld(*plan);
            if (failed(selected)) {
                result.totalUs = elapsedMicroseconds(totalStart);
                ++result.protocolPlansRejected;
                recordStatistics(
                    function, result, "internal-error", "selected-world", ProtocolSyncProducer::InternalError,
                    "internal-error");
                function.emitError("ProtocolSync ReadyRelease selected-world construction failed internally");
                return failure();
            }
            world = std::move(*selected);
        }
        if (plan->status != SyncReadyReleasePlanStatus::Unsupported) {
            FailureOr<SyncInterpretationResult> interpretation =
                evaluateWorld(function, schedule, stages, timelines, channels, world, result);
            const bool incomplete = failed(interpretation) || !interpretation->isComplete();
            if (incomplete) {
                result.totalUs = elapsedMicroseconds(totalStart);
                ++result.protocolPlansRejected;
                recordStatistics(
                    function, result, "internal-error", "selected-world", ProtocolSyncProducer::InternalError,
                    "internal-error");
                function.emitError("ProtocolSync ReadyRelease selected world is not obligation-complete");
                return failure();
            }
        }

        start = ProtocolSyncClock::now();
        if (failed(allocateReadyReleaseProtocolEvents(schedule, *plan, &result))) {
            result.allocationUs = elapsedMicroseconds(start);
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "allocation", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync ReadyRelease event allocation failed internally");
            return failure();
        }
        result.allocationUs = elapsedMicroseconds(start);
        if (dumpMode == "plan") {
            printReadyReleaseProtocolPlan(function, *plan, llvm::errs());
        } else if (dumpMode == "lane-frontiers" && plan->status == SyncReadyReleasePlanStatus::Ready) {
            printReadyReleaseFrontierComparison(schedule, laneFrontiers, *plan, llvm::errs());
        } else if (dumpMode == "storage-tracks" && plan->status == SyncReadyReleasePlanStatus::Ready) {
            printReadyReleaseStorageTrackComparison(schedule, storageTracks, *plan, llvm::errs());
        }
        if (plan->status == SyncReadyReleasePlanStatus::Unsupported) {
            ++result.protocolPlansRejected;
            const bool allocationFailure =
                llvm::any_of(plan->rejections, [](const SyncReadyReleasePlanRejection& rejection) {
                    return rejection.reason == SyncReadyReleaseRejection::EventCapacity;
                });
            const bool internalRejection =
                llvm::any_of(plan->rejections, [](const SyncReadyReleasePlanRejection& rejection) {
                    return rejection.reason == SyncReadyReleaseRejection::InternalInvariant;
                });
            const bool targetRejection =
                llvm::any_of(plan->rejections, [](const SyncReadyReleasePlanRejection& rejection) {
                    return rejection.reason == SyncReadyReleaseRejection::UnsupportedTarget;
                });
            result.totalUs = elapsedMicroseconds(totalStart);
            if (internalRejection) {
                recordStatistics(
                    function, result, "internal-error", "planning", ProtocolSyncProducer::InternalError,
                    "internal-error", "not-permitted");
                function.emitError("ProtocolSync ReadyRelease planning rejected an internal invariant");
                return failure();
            }
            return handleUnsupported(
                function, result, allocationFailure ? "allocation" : "planning",
                allocationFailure ? "resource-infeasible" : "unsupported", !targetRejection, totalStart);
        }
        if (plan->status == SyncReadyReleasePlanStatus::Empty) {
            ++result.protocolPlansAdmitted;
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(function, result, "ok", "", ProtocolSyncProducer::ProtocolPlan, "no-op", "none");
            return success();
        }
        ++result.protocolPlansAdmitted;
        if (failed(materializeAndVerifyReadyReleaseProtocolPlanInPlace(schedule, stages, *plan, &result))) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "materialization-verification", ProtocolSyncProducer::InternalError,
                "rejected");
            return failure();
        }
        result.totalUs = elapsedMicroseconds(totalStart);
        recordStatistics(
            function, result, "ok", "", ProtocolSyncProducer::ProtocolPlan, "materialized-ready-release", "none",
            /*stagedMutation=*/true);
        return success();
    }

    LogicalResult analyzeFunction(
        func::FuncOp function, bool emitOneShot, bool emitReadyRelease, bool emitDirectRepair, bool emitMixed)
    {
        ProtocolSyncStatistics result;
        LegacySyncIRAdapter adapter;
        LegacySyncSnapshot legacy;
        const ProtocolSyncClock::time_point totalStart = ProtocolSyncClock::now();
        ProtocolSyncClock::time_point start = ProtocolSyncClock::now();
        if (failed(adapter.buildSnapshot(function, legacy))) {
            result.legacySnapshotUs = elapsedMicroseconds(start);
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(function, result, "internal-error", "legacy-shadow", ProtocolSyncProducer::InternalError);
            function.emitError("ProtocolSync legacy shadow construction failed");
            return failure();
        }
        result.legacySnapshotUs = elapsedMicroseconds(start);
        start = ProtocolSyncClock::now();
        SyncSemanticContext context = adapter.buildSemanticContext(legacy);
        result.semanticContextUs = elapsedMicroseconds(start);
        StructuredSyncIR schedule(function);
        start = ProtocolSyncClock::now();
        StructuredSyncIRBuilder builder(context, &result);
        if (failed(builder.build(function, schedule))) {
            result.scheduleConstructionUs = elapsedMicroseconds(start);
            collectScheduleStatistics(schedule, result);
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(function, result, "internal-error", "schedule", ProtocolSyncProducer::InternalError);
            function.emitError("ProtocolSync structured schedule failed");
            return failure();
        }
        result.scheduleConstructionUs = elapsedMicroseconds(start);
        collectScheduleStatistics(schedule, result);
        start = ProtocolSyncClock::now();
        FailureOr<PipelineStageAnalysisResult> stages = analyzePipelineStages(schedule);
        result.storageAnalysisUs = elapsedMicroseconds(start);
        if (failed(stages)) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "pipeline-stages", ProtocolSyncProducer::InternalError);
            function.emitError("ProtocolSync pipeline-stage analysis failed");
            return failure();
        }
        result.pipelineStages = stages->getStages().size();
        start = ProtocolSyncClock::now();
        StorageTimelineAnalysisResult timelines = analyzeStorageTimelines(schedule, *stages, &result);
        result.generationAnalysisUs = elapsedMicroseconds(start);

        start = ProtocolSyncClock::now();
        ChannelAnalysisResult channels = analyzeChannels(schedule, *stages, timelines, &result);
        compareWithLegacyDemandOracle(legacy, schedule, *stages, timelines, channels);
        result.channelAnalysisUs = elapsedMicroseconds(start);
        start = ProtocolSyncClock::now();
        LaneFrontierAnalysisResult laneFrontiers =
            analyzeLaneFrontiers(schedule, *stages, timelines, channels, &result);
        result.laneFrontierAnalysisUs = elapsedMicroseconds(start);
        start = ProtocolSyncClock::now();
        LanePatternAnalysisResult lanePatterns =
            analyzeLanePatterns(schedule, *stages, timelines, channels, laneFrontiers, &result);
        result.lanePatternAnalysisUs = elapsedMicroseconds(start);
        start = ProtocolSyncClock::now();
        StorageTrackAnalysisResult storageTracks =
            analyzeStorageTracks(schedule, *stages, timelines, channels, laneFrontiers, lanePatterns, &result);
        result.storageTrackAnalysisUs = elapsedMicroseconds(start);

        start = ProtocolSyncClock::now();
        LegacySyncParityResult parity = adapter.compare(legacy, schedule);
        result.legacyComparisonUs = elapsedMicroseconds(start);
        result.legacyParityMismatches = parity.mismatches.size();
        printLegacySyncParity(function, parity, llvm::errs());
        const bool hasInternalScheduleFailure = llvm::any_of(schedule.getFailures(), [](const SyncFailure& failure) {
            return failure.reason == SyncFailureReason::InternalInvariant;
        });
        if (!parity.isInternallyConsistent() || hasInternalScheduleFailure) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "semantic-consistency", ProtocolSyncProducer::InternalError);
            function.emitError("ProtocolSync internal semantic consistency check failed");
            return failure();
        }

        if (dumpMode == "schedule") {
            printStructuredSyncIR(schedule, llvm::errs());
        } else if (dumpMode == "channels") {
            printProtocolSyncChannels(schedule, *stages, timelines, channels, llvm::errs());
        } else if (dumpMode == "lane-frontiers") {
            printLaneFrontierAnalysis(schedule, laneFrontiers, llvm::errs());
            printLanePatternAnalysis(schedule, timelines, lanePatterns, llvm::errs());
        } else if (dumpMode == "storage-tracks") {
            printStructuredSyncIR(schedule, llvm::errs());
            FailureOr<SyncReadyReleasePlan> diagnosticPlan =
                buildReadyReleaseProtocolPlan(schedule, *stages, timelines, channels);
            if (succeeded(diagnosticPlan)) {
                printReadyReleaseProtocolPlan(function, *diagnosticPlan, llvm::errs());
            }
            printStorageTrackAnalysis(schedule, laneFrontiers, storageTracks, llvm::errs());
        } else if (dumpMode == "concrete-verification") {
            printStructuredSyncIR(schedule, llvm::errs());
            const ProtocolSyncClock::time_point verificationStart = ProtocolSyncClock::now();
            ++result.concreteVerifierRuns;
            StringRef firstFailedStage;
            const bool verified = succeeded(verifyConcreteSyncSemantics(context, function, &result, &firstFailedStage));
            result.verificationUs += elapsedMicroseconds(verificationStart);
            result.concreteVerifierAccepted += verified ? 1 : 0;
            result.concreteVerifierRejected += verified ? 0 : 1;
            llvm::errs() << "PROTOCOL-SYNC concrete-verification function=@" << function.getSymName()
                         << " status=" << (verified ? "accepted" : "rejected")
                         << " first-failed-stage=" << firstFailedStage << '\n';
        }
        const bool oracleMismatch = llvm::any_of(channels.getChannels(), [](const SyncChannel& channel) {
            return channel.readyOracle == SyncDemandOracleStatus::Mismatch ||
                   channel.releaseOracle == SyncDemandOracleStatus::Mismatch;
        });
        const bool diagnosticRejected = !parity.matches() || !schedule.getFailures().empty() || oracleMismatch;
        if (!emitOneShot && !emitReadyRelease && !emitDirectRepair && !emitMixed) {
            SyncSelectedWorld world;
            FailureOr<SyncInterpretationResult> interpretation =
                evaluateWorld(function, schedule, *stages, timelines, channels, world, result);
            if (failed(interpretation)) {
                result.totalUs = elapsedMicroseconds(totalStart);
                recordStatistics(
                    function, result, "internal-error", "selected-world", ProtocolSyncProducer::InternalError);
                function.emitError("ProtocolSync selected-world interpretation failed internally");
                return failure();
            }
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(function, result, diagnosticRejected ? "diagnostic-rejection" : "ok", "");
            return success();
        }
        const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(function);
        const bool targetSupportsEmission =
            (emitOneShot && target.supportsOneShotEmission()) ||
            (emitReadyRelease && target.supportsReadyReleaseEmission()) ||
            (emitDirectRepair && target.supportsDirectRepairEmission()) ||
            (emitMixed && target.supportsMixedEmission());
        if (diagnosticRejected && targetSupportsEmission) {
            return handleUnsupported(function, result, "semantic-diagnostics", "unsupported", true, totalStart);
        }

        if (emitReadyRelease) {
            return emitReadyReleasePlan(
                function, schedule, *stages, timelines, channels, laneFrontiers, storageTracks, result, totalStart);
        }
        if (emitMixed) {
            return emitMixedPlan(function, schedule, *stages, timelines, channels, context, result, totalStart);
        }
        if (emitDirectRepair) {
            return emitDirectRepairPlan(function, schedule, *stages, timelines, channels, result, totalStart);
        }

        start = ProtocolSyncClock::now();
        ++result.protocolPlansAttempted;
        FailureOr<SyncOneShotPlan> plan = buildOneShotProtocolPlan(schedule, *stages, timelines, channels, &result);
        result.planningUs = elapsedMicroseconds(start);
        if (failed(plan)) {
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "planning", ProtocolSyncProducer::InternalError, "internal-error");
            function.emitError("ProtocolSync one-shot planning failed internally");
            return failure();
        }

        SyncSelectedWorld world;
        if (plan->status == SyncOneShotPlanStatus::Ready) {
            FailureOr<SyncSelectedWorld> selected = buildSelectedWorld(*plan, channels);
            if (failed(selected)) {
                result.totalUs = elapsedMicroseconds(totalStart);
                ++result.protocolPlansRejected;
                recordStatistics(
                    function, result, "internal-error", "selected-world", ProtocolSyncProducer::InternalError,
                    "internal-error");
                function.emitError("ProtocolSync one-shot selected-world construction failed internally");
                return failure();
            }
            world = std::move(*selected);
        }
        if (plan->status != SyncOneShotPlanStatus::Unsupported) {
            FailureOr<SyncInterpretationResult> interpretation =
                evaluateWorld(function, schedule, *stages, timelines, channels, world, result);
            const bool incomplete = failed(interpretation) || !interpretation->isComplete();
            if (incomplete) {
                result.totalUs = elapsedMicroseconds(totalStart);
                ++result.protocolPlansRejected;
                recordStatistics(
                    function, result, "internal-error", "selected-world", ProtocolSyncProducer::InternalError,
                    "internal-error");
                function.emitError("ProtocolSync one-shot selected world is not obligation-complete");
                return failure();
            }
        }

        start = ProtocolSyncClock::now();
        if (failed(allocateOneShotProtocolEvents(schedule, *plan, &result))) {
            result.allocationUs = elapsedMicroseconds(start);
            result.totalUs = elapsedMicroseconds(totalStart);
            ++result.protocolPlansRejected;
            recordStatistics(
                function, result, "internal-error", "allocation", ProtocolSyncProducer::InternalError,
                "internal-error");
            function.emitError("ProtocolSync one-shot event allocation failed internally");
            return failure();
        }
        result.allocationUs = elapsedMicroseconds(start);
        if (dumpMode == "plan") {
            printOneShotProtocolPlan(function, *plan, llvm::errs());
        }
        if (plan->status == SyncOneShotPlanStatus::Unsupported) {
            ++result.protocolPlansRejected;
            const bool allocationFailure =
                llvm::any_of(plan->rejections, [](const SyncOneShotPlanRejection& rejection) {
                    return rejection.reason == SyncOneShotRejection::EventCapacity;
                });
            const bool internalRejection =
                llvm::any_of(plan->rejections, [](const SyncOneShotPlanRejection& rejection) {
                    return rejection.reason == SyncOneShotRejection::InternalInvariant;
                });
            const bool targetRejection = llvm::any_of(plan->rejections, [](const SyncOneShotPlanRejection& rejection) {
                return rejection.reason == SyncOneShotRejection::UnsupportedTarget;
            });
            result.totalUs = elapsedMicroseconds(totalStart);
            if (internalRejection) {
                recordStatistics(
                    function, result, "internal-error", "planning", ProtocolSyncProducer::InternalError,
                    "internal-error", "not-permitted");
                function.emitError("ProtocolSync one-shot planning rejected an internal invariant");
                return failure();
            }
            return handleUnsupported(
                function, result, allocationFailure ? "allocation" : "planning",
                allocationFailure ? "resource-infeasible" : "unsupported", !targetRejection, totalStart);
        }
        if (plan->status == SyncOneShotPlanStatus::Empty) {
            ++result.protocolPlansAdmitted;
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(function, result, "ok", "", ProtocolSyncProducer::ProtocolPlan, "no-op", "none");
            return success();
        }
        ++result.protocolPlansAdmitted;
        if (failed(materializeAndVerifyOneShotProtocolPlanInPlace(schedule, *stages, *plan, &result))) {
            result.totalUs = elapsedMicroseconds(totalStart);
            recordStatistics(
                function, result, "internal-error", "materialization-verification", ProtocolSyncProducer::InternalError,
                "rejected");
            return failure();
        }
        result.totalUs = elapsedMicroseconds(totalStart);
        recordStatistics(
            function, result, "ok", "", ProtocolSyncProducer::ProtocolPlan, "materialized", "none",
            /*stagedMutation=*/true);
        return success();
    }

    SmallVector<PendingProtocolSyncStatistics, 8> pendingStatistics;
};

} // namespace

std::unique_ptr<Pass> createPTOProtocolSyncPass(const PTOProtocolSyncOptions& options)
{
    return std::make_unique<PTOProtocolSyncPass>(options);
}

} // namespace mlir::pto
