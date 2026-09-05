// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncSemantics.h - Shared ProtocolSync operation facts ----*- C++ -*-===//
//
// This file defines the target-independent, operation-local facts shared by
// ProtocolSync and the legacy shadow adapter. It deliberately contains no
// synchronization candidate or allocation state.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_SYNCSEMANTICS_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_SYNCSEMANTICS_H

#include "PTO/IR/PTO.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

using SyncRegionId = std::uint32_t;

inline constexpr std::uint32_t kInvalidSyncId = std::numeric_limits<std::uint32_t>::max();

enum class SyncPhysicalCore : std::uint8_t { Cube, Vector, Unknown };

enum class SyncAccessMode : std::uint8_t {
    Read,
    Write,
    ReadWrite,
    Ordered,
};

enum class SyncVisibilityClass : std::uint8_t { Local, Global, Unknown };

enum class SyncCompletionKind : std::uint8_t {
    PhaseEnd,
    MacroInternal,
    Unknown,
};

enum class SyncQueueRole : std::uint8_t {
    Initialize,
    ProducerAcquire,
    ProducerPublish,
    ConsumerAcquire,
    ConsumerRelease,
};

enum class SyncSummaryProvider : std::uint8_t {
    FixedSynchronization,
    Queue,
    Macro,
    Helper,
    Pipeline,
    Structural,
    None,
};

enum class SyncFailureReason : std::uint8_t {
    None,
    MissingPipeline,
    MissingPhysicalCore,
    MissingStorageProvenance,
    UnscopedMemoryEffect,
    UnsupportedMemoryEffectKind,
    UnsupportedEffectfulOperation,
    UnsupportedRegion,
    UnsupportedCFG,
    LegacyStructureMismatch,
    LegacyPhaseMismatch,
    InternalInvariant,
};

enum class ProtocolSyncProducer : std::uint8_t {
    AnalysisOnly,
    ProtocolPlan,
    ProtocolPlusDirectResiduals,
    LegacyFallbackUnsupported,
    LegacyFallbackResourceInfeasible,
    FailClosedPolicy,
    InternalError,
};

struct SyncByteInterval {
    std::uint64_t begin = 0;
    std::uint64_t size = 0;
};

enum class SyncSlotExpressionKind : std::uint8_t {
    Unknown,
    Constant,
    AffineModulo,
};

enum class SyncSlotRelation : std::uint8_t {
    Same,
    Different,
    Unknown,
};

struct SyncSlotExpression {
    // AffineModulo is expressed in the loop's logical iteration coordinate:
    // slot(t) = (coefficient * t + offset) mod modulus.
    Value selector;
    std::uint32_t depth = 0;
    SyncSlotExpressionKind kind = SyncSlotExpressionKind::Unknown;
    Value induction;
    SyncRegionId loop = kInvalidSyncId;
    std::int64_t coefficient = 0;
    std::int64_t offset = 0;
    std::uint32_t modulus = 0;
};

struct SyncStorageProvenance {
    Value value;
    Value root;
    AddressSpace space = AddressSpace::Zero;
    llvm::SmallVector<SyncByteInterval, 2> intervals;
    bool physical = false;
    bool unknownRange = true;
    bool aliasesUnknownRange = false;
};

struct SyncMemoryEffect {
    Value value;
    SyncAccessMode mode = SyncAccessMode::ReadWrite;
    SyncStorageProvenance provenance;
    SyncVisibilityClass visibility = SyncVisibilityClass::Unknown;
    std::optional<SyncSlotExpression> slot;
};

struct SyncPhaseCompletion {
    SyncCompletionKind kind = SyncCompletionKind::PhaseEnd;
};

struct SyncPhysicalPhase {
    unsigned phaseIndex = 0;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    llvm::SmallVector<SyncMemoryEffect, 4> effects;
    SyncPhaseCompletion completion;
};

struct SyncInternalProtocol {
    std::string kind;
};

struct SyncEventReservation {
    PIPE source = PIPE::PIPE_UNASSIGNED;
    PIPE target = PIPE::PIPE_UNASSIGNED;
    llvm::SmallVector<unsigned, 2> eventIds;
};

struct SyncCompletionContract {
    SyncCompletionKind kind = SyncCompletionKind::PhaseEnd;
};

struct SyncQueueSemantics {
    SyncQueueRole role = SyncQueueRole::Initialize;
    Value handle;
    std::optional<std::uint32_t> depth;
    std::uint8_t directionMask = 0;
    std::optional<std::uint32_t> localSlotCount;
    std::optional<std::uint32_t> flagBase;
    Value endpoint;
    Value peerEndpoint;
};

struct SyncOpSummary {
    Operation* operation = nullptr;
    SyncSummaryProvider provider = SyncSummaryProvider::None;
    llvm::SmallVector<SyncPhysicalPhase, 2> phases;
    llvm::SmallVector<SyncInternalProtocol, 1> suppliedProtocols;
    llvm::SmallVector<SyncEventReservation, 2> eventReservations;
    SyncCompletionContract completion;
    std::optional<SyncQueueSemantics> queue;
    SyncFailureReason failure = SyncFailureReason::None;
    std::string failureDetail;

    bool isSupported() const { return failure == SyncFailureReason::None; }
};

struct ProtocolSyncStatistics {
    std::uint64_t operationsVisited = 0;
    std::uint64_t operationsSummarized = 0;
    std::uint64_t providerLookupAttempts = 0;
    std::uint64_t providerHits = 0;
    std::uint64_t fixedSuppliedProtocols = 0;
    std::uint64_t hiddenEventReservations = 0;
    std::uint64_t structuredRegions = 0;
    std::uint64_t semanticActions = 0;
    std::uint64_t phases = 0;
    std::uint64_t accesses = 0;
    std::uint64_t storageFamilies = 0;
    std::uint64_t pipelineStages = 0;
    std::uint64_t knownRanges = 0;
    std::uint64_t unknownRanges = 0;
    std::uint64_t intervalIndexQueries = 0;
    std::uint64_t slotExpressionsKnown = 0;
    std::uint64_t slotExpressionsUnknown = 0;
    std::uint64_t depthOneAccesses = 0;
    std::uint64_t depthTwoAccesses = 0;
    std::uint64_t unknownCapacityAccesses = 0;
    std::uint64_t rejectedOperations = 0;
    std::uint64_t legacyParityMismatches = 0;
    std::uint64_t generationTimelinesAttempted = 0;
    std::uint64_t generationTimelinesAdmitted = 0;
    std::uint64_t generationTimelinesRejected = 0;
    std::uint64_t channelCandidatesAttempted = 0;
    std::uint64_t channelCandidatesAdmitted = 0;
    std::uint64_t channelCandidatesRejected = 0;
    std::uint64_t interpreterTransitions = 0;
    std::uint64_t interpreterPeakStates = 0;
    std::uint64_t memoryPairTests = 0;
    std::uint64_t noAliasResults = 0;
    std::uint64_t mayAliasResults = 0;
    std::uint64_t unknownAliasResults = 0;
    std::uint64_t tokenCertificateTransitions = 0;
    std::uint64_t protocolCandidates = 0;
    std::uint64_t protocolPlansAttempted = 0;
    std::uint64_t protocolPlansAdmitted = 0;
    std::uint64_t protocolPlansRejected = 0;
    std::uint64_t selectedOneShotProtocols = 0;
    std::uint64_t selectedDirectedEventPairs = 0;
    std::uint64_t selectedSamePipeBarriers = 0;
    std::uint64_t selectedTailDrains = 0;
    std::uint64_t selectedReadyReleaseProtocols = 0;
    std::uint64_t selectedReadyReleaseLanes = 0;
    std::uint64_t directRepairCandidates = 0;
    std::uint64_t directRepairSharedCandidates = 0;
    std::uint64_t selectedDirectRepairs = 0;
    std::uint64_t directRepairUncovered = 0;
    std::uint64_t mixedSelectionCandidates = 0;
    std::uint64_t reverseDeletionAttempts = 0;
    std::uint64_t reverseDeletionRemoved = 0;
    std::uint64_t completeWorldsAttempted = 0;
    std::uint64_t completeWorldsFeasible = 0;
    std::uint64_t selectedWorldEventPairs = 0;
    std::uint64_t selectedWorldTargetedBarriers = 0;
    std::uint64_t selectedWorldFixedExitDrains = 0;
    std::uint64_t allocationRetries = 0;
    std::uint64_t eventDomains = 0;
    std::uint64_t maxEventDomainPressure = 0;
    std::uint64_t maximumEventIdPlusOne = 0;
    std::uint64_t logicalActions = 0;
    std::uint64_t residualObligations = 0;
    std::uint64_t allocationGraphVertices = 0;
    std::uint64_t allocationGraphEdges = 0;
    std::uint64_t allocationBacktrackingNodes = 0;
    std::uint64_t allocationSearchLimitHits = 0;
    std::uint64_t materializationTransitions = 0;
    std::uint64_t verifierTransitions = 0;
    std::uint64_t semanticExtractionUs = 0;
    std::uint64_t scheduleConstructionUs = 0;
    std::uint64_t legacySnapshotUs = 0;
    std::uint64_t semanticContextUs = 0;
    std::uint64_t legacyComparisonUs = 0;
    std::uint64_t storageAnalysisUs = 0;
    std::uint64_t generationAnalysisUs = 0;
    std::uint64_t channelAnalysisUs = 0;
    std::uint64_t interpretationUs = 0;
    std::uint64_t planningUs = 0;
    std::uint64_t allocationUs = 0;
    std::uint64_t materializationUs = 0;
    std::uint64_t verificationUs = 0;
    std::uint64_t totalUs = 0;
    std::map<std::string, std::uint64_t> generationRejections;
    std::map<std::string, std::uint64_t> channelRejections;
    std::map<std::string, std::uint64_t> residualObligationsByKind;
};

class SyncSemanticContext {
public:
    void addStorage(Value value, SyncStorageProvenance provenance);
    llvm::ArrayRef<SyncStorageProvenance> lookupStorage(Value value) const;

private:
    llvm::DenseMap<Value, llvm::SmallVector<SyncStorageProvenance, 2>> storage;
};

class SyncSemanticExtractor {
public:
    SyncSemanticExtractor(const SyncSemanticContext& context, ProtocolSyncStatistics* statistics = nullptr)
        : context(context), statistics(statistics)
    {}

    SyncOpSummary summarize(Operation* operation) const;

private:
    const SyncSemanticContext& context;
    ProtocolSyncStatistics* statistics;
};

std::optional<std::uint32_t> getSyncSlotDepth(Value value);
/// Compare first(t) with second(t + distance). Unknown inputs remain unknown.
SyncSlotRelation compareSlotsAtDistance(
    const SyncSlotExpression& first, const SyncSlotExpression& second, unsigned distance);
/// Return the proven period when it does not exceed searchLimit.
FailureOr<unsigned> findFirstPositiveReuseDistance(const SyncSlotExpression& slot, unsigned searchLimit);
std::optional<SyncQueueSemantics> getSyncQueueSemantics(Operation* operation);
bool isFixedSyncOperation(Operation* operation);
llvm::StringRef stringifySyncSlotExpressionKind(SyncSlotExpressionKind kind);
llvm::StringRef stringifySyncSlotRelation(SyncSlotRelation relation);
llvm::StringRef stringifySyncPhysicalCore(SyncPhysicalCore core);
llvm::StringRef stringifySyncAccessMode(SyncAccessMode mode);
llvm::StringRef stringifySyncVisibility(SyncVisibilityClass visibility);
llvm::StringRef stringifySyncCompletionKind(SyncCompletionKind completion);
llvm::StringRef stringifySyncQueueRole(SyncQueueRole role);
llvm::StringRef stringifySyncSummaryProvider(SyncSummaryProvider provider);
llvm::StringRef stringifySyncFailureReason(SyncFailureReason reason);
llvm::StringRef stringifyProtocolSyncProducer(ProtocolSyncProducer producer);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_SYNCSEMANTICS_H
