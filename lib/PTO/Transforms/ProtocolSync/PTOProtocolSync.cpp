// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- PTOProtocolSync.cpp - Analysis-only ProtocolSync pass ------------===//

#include "PTO/Transforms/Passes.h"

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    statistics.phases = schedule.getPhases().size();
    statistics.accesses = schedule.getAccesses().size();
    statistics.storageFamilies = schedule.getStorageFamilies().size();
    for (const SyncAccess& access : schedule.getAccesses()) {
        if (access.storage.unknownRange) {
            ++statistics.unknownRanges;
        } else {
            ++statistics.knownRanges;
        }
        if (!access.slot) {
            ++statistics.depthOneAccesses;
        } else if (access.slot->depth == 0) {
            ++statistics.unknownCapacityAccesses;
        } else if (access.slot->depth == 1) {
            ++statistics.depthOneAccesses;
        } else if (access.slot->depth == 2) {
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
    func::FuncOp function, const ProtocolSyncStatistics& statistics, StringRef status, StringRef failureStage)
{
    llvm::json::Object record;
    record["kind"] = "protocol-sync-statistics";
    record["function"] = function.getSymName().str();
    record["status"] = status.str();
    record["producer"] = "analysis-only";
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
    counts["channel_candidates_attempted"] = static_cast<std::int64_t>(statistics.channelCandidatesAttempted);
    counts["channel_candidates_admitted"] = static_cast<std::int64_t>(statistics.channelCandidatesAdmitted);
    counts["channel_candidates_rejected"] = static_cast<std::int64_t>(statistics.channelCandidatesRejected);
    counts["interpreter_transitions"] = static_cast<std::int64_t>(statistics.interpreterTransitions);
    counts["interpreter_peak_states"] = static_cast<std::int64_t>(statistics.interpreterPeakStates);
    counts["protocol_candidates"] = static_cast<std::int64_t>(statistics.protocolCandidates);
    counts["logical_actions"] = static_cast<std::int64_t>(statistics.logicalActions);
    counts["residual_obligations"] = static_cast<std::int64_t>(statistics.residualObligations);
    counts["allocation_graph_vertices"] = static_cast<std::int64_t>(statistics.allocationGraphVertices);
    counts["allocation_graph_edges"] = static_cast<std::int64_t>(statistics.allocationGraphEdges);
    counts["allocation_backtracking_nodes"] = static_cast<std::int64_t>(statistics.allocationBacktrackingNodes);
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
    record["planner_result"] = "analysis-only";
    record["fallback"] = "none";
    llvm::json::Object timing;
    timing["semantic_extraction"] = static_cast<std::int64_t>(statistics.semanticExtractionUs);
    timing["schedule_construction"] = static_cast<std::int64_t>(statistics.scheduleConstructionUs);
    timing["legacy_snapshot"] = static_cast<std::int64_t>(statistics.legacySnapshotUs);
    timing["semantic_context"] = static_cast<std::int64_t>(statistics.semanticContextUs);
    timing["legacy_comparison"] = static_cast<std::int64_t>(statistics.legacyComparisonUs);
    timing["storage_analysis"] = static_cast<std::int64_t>(statistics.storageAnalysisUs);
    timing["generation_analysis"] = static_cast<std::int64_t>(statistics.generationAnalysisUs);
    timing["channel_analysis"] = static_cast<std::int64_t>(statistics.channelAnalysisUs);
    timing["planning"] = static_cast<std::int64_t>(statistics.planningUs);
    timing["allocation"] = static_cast<std::int64_t>(statistics.allocationUs);
    timing["materialization"] = static_cast<std::int64_t>(statistics.materializationUs);
    timing["verification"] = static_cast<std::int64_t>(statistics.verificationUs);
    timing["total"] = static_cast<std::int64_t>(statistics.totalUs);
    record["time_us"] = std::move(timing);
    llvm::errs() << llvm::json::Value(std::move(record)) << '\n';
}

struct PTOProtocolSyncPass : public impl::PTOProtocolSyncBase<PTOProtocolSyncPass> {
    PTOProtocolSyncPass() = default;

    explicit PTOProtocolSyncPass(const PTOProtocolSyncOptions& options)
    {
        dumpMode = options.dumpMode;
        statistics = options.statistics;
    }

    void runOnOperation() final
    {
        if (dumpMode != "none" && dumpMode != "schedule" && dumpMode != "channels") {
            getOperation().emitError("unknown ProtocolSync dump mode '")
                << dumpMode << "'; expected 'none', 'schedule', or 'channels'";
            signalPassFailure();
            return;
        }

        SmallVector<func::FuncOp, 8> functions;
        getOperation().walk([&](func::FuncOp function) { functions.push_back(function); });
        for (func::FuncOp function : functions) {
            const bool isDeclaration = function.isDeclaration();
            if (isDeclaration) {
                continue;
            }
            if (failed(analyzeFunction(function))) {
                signalPassFailure();
                return;
            }
        }
        markAllAnalysesPreserved();
    }

private:
    LogicalResult analyzeFunction(func::FuncOp function)
    {
        ProtocolSyncStatistics result;
        LegacySyncIRAdapter adapter;
        LegacySyncSnapshot legacy;
        const ProtocolSyncClock::time_point totalStart = ProtocolSyncClock::now();

        ProtocolSyncClock::time_point start = ProtocolSyncClock::now();
        if (failed(adapter.buildSnapshot(function, legacy))) {
            result.legacySnapshotUs = elapsedMicroseconds(start);
            result.totalUs = elapsedMicroseconds(totalStart);
            if (statistics) {
                printStatistics(function, result, "internal-error", "legacy-shadow");
            }
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
            if (statistics) {
                printStatistics(function, result, "internal-error", "schedule");
            }
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
            if (statistics) {
                printStatistics(function, result, "internal-error", "pipeline-stages");
            }
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
        LegacySyncParityResult parity = adapter.compare(legacy, schedule);
        result.legacyComparisonUs = elapsedMicroseconds(start);
        result.legacyParityMismatches = parity.mismatches.size();
        printLegacySyncParity(function, parity, llvm::errs());
        if (dumpMode == "schedule") {
            printStructuredSyncIR(schedule, llvm::errs());
        } else if (dumpMode == "channels") {
            printProtocolSyncChannels(schedule, *stages, timelines, channels, llvm::errs());
        }
        if (statistics) {
            const bool oracleMismatch = llvm::any_of(channels.getChannels(), [](const SyncChannel& channel) {
                return channel.readyOracle == SyncDemandOracleStatus::Mismatch ||
                       channel.releaseOracle == SyncDemandOracleStatus::Mismatch;
            });
            StringRef status =
                parity.matches() && schedule.getFailures().empty() && !oracleMismatch ? "ok" : "diagnostic-rejection";
            result.totalUs = elapsedMicroseconds(totalStart);
            printStatistics(function, result, status, "");
        }
        return success();
    }
};

} // namespace

std::unique_ptr<Pass> createPTOProtocolSyncPass(const PTOProtocolSyncOptions& options)
{
    return std::make_unique<PTOProtocolSyncPass>(options);
}

} // namespace mlir::pto
