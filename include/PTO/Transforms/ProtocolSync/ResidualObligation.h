// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ResidualObligation.h - Selected-world interpretation ---*- C++ -*-===//
//
// A selected world contains logical protocol effects, never concrete event
// IDs. The interpreter validates those effects against the frozen generation
// model and returns only the synchronization obligations that remain.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_RESIDUALOBLIGATION_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_RESIDUALOBLIGATION_H

#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

struct SyncOneShotPlan;
struct SyncReadyReleasePlan;

using SyncObligationId = std::uint32_t;

enum class SyncObligationKind : std::uint8_t {
    Completion,
    Reclamation,
    SSACompletion,
    OrderedMemory,
    AccConflict,
    Visibility,
    ExitCompletion,
    UnknownAlias,
};

enum class SyncControlRelation : std::uint8_t {
    MustExecute,
    SameGuard,
    Unknown,
};

enum class SyncIterationRelationKind : std::uint8_t {
    SameIteration,
    LoopCarried,
    /// Outside-before -> body(i), for every executing iteration i.
    LoopEntry,
    /// Body(i) -> outside-after, for every executing iteration i.
    LoopExit,
    /// Outside-before -> outside-after on the zero-trip path only.
    LoopBypass,
    Unknown,
};

struct SyncIterationRelation {
    SyncIterationRelationKind kind = SyncIterationRelationKind::Unknown;
    unsigned distance = 0;
    SyncRegionId carrier = kInvalidSyncId;
};

struct SyncResidualObligation {
    SyncObligationId id = kInvalidSyncId;
    SyncObligationKind kind = SyncObligationKind::Completion;
    SyncPhaseId source = kInvalidSyncId;
    SyncPhaseId target = kInvalidSyncId;
    std::optional<SyncGenerationId> generation;
    std::optional<SyncChannelId> channel;
    SyncControlRelation control = SyncControlRelation::Unknown;
    SyncIterationRelation iteration;
    std::string detail;
    /// Stable provenance in the canonical local store, independent of residual
    /// numbering and optional protocol selection. Empty for legacy obligations.
    std::optional<std::uint32_t> localRequirement;
    llvm::SmallVector<std::uint32_t, 2> atoms;
    SyncAccessId sourceAccess = kInvalidSyncId;
    SyncAccessId targetAccess = kInvalidSyncId;
    SyncRegionPrecision precision = SyncRegionPrecision::Unknown;
};

enum class SyncSelectedProtocolKind : std::uint8_t {
    OneShotPublish,
    ReadyRelease,
};

struct SyncSelectedProtocol {
    SyncSelectedProtocolKind kind = SyncSelectedProtocolKind::OneShotPublish;
    SyncChannelId channel = kInvalidSyncId;
    SyncGenerationId generation = kInvalidSyncId;
    unsigned capacity = 0;
};

struct SyncSelectedCompletion {
    SyncPhaseId source = kInvalidSyncId;
    SyncPhaseId target = kInvalidSyncId;
    SyncControlRelation control = SyncControlRelation::MustExecute;
    SyncIterationRelation iteration{SyncIterationRelationKind::SameIteration, 0};
};

struct SyncSelectedVisibility {
    SyncPhaseId source = kInvalidSyncId;
    SyncPhaseId target = kInvalidSyncId;
    SyncControlRelation control = SyncControlRelation::MustExecute;
    SyncIterationRelation iteration{SyncIterationRelationKind::SameIteration, 0};
};

struct SyncSelectedWorld {
    /// Atomic fully serialized ordinary-loop completion certificate. Never
    /// implies visibility. Concrete verification reconstructs it without tags.
    std::optional<SyncRegionId> orderedLoop;
    llvm::SmallVector<SyncSelectedProtocol, 4> protocols;
    llvm::SmallVector<SyncSelectedCompletion, 8> completions;
    llvm::SmallVector<SyncSelectedVisibility, 2> visibility;
    llvm::SmallVector<SyncPhaseId, 8> exitCompletedPhases;
};

struct SyncInterpretationResult {
    llvm::SmallVector<SyncResidualObligation, 16> obligations;
    std::uint64_t transitions = 0;
    std::uint64_t peakStates = 0;
    std::uint64_t memoryPairTests = 0;
    std::uint64_t noAliasResults = 0;
    std::uint64_t mayAliasResults = 0;
    std::uint64_t unknownAliasResults = 0;
    std::uint64_t localAtoms = 0;
    std::uint64_t localRequirements = 0;
    std::uint64_t localRequirementsCovered = 0;
    std::string localAnalysisBoundary;

    bool isComplete() const { return obligations.empty(); }
};

struct SyncInterpretationOptions {
    /// Concrete verification reconstructs fixed synchronization as selected
    /// world effects. Those operations must then not also appear as opaque
    /// semantic actions. Planning keeps the default fail-closed behavior.
    bool fixedSynchronizationIsModeled = false;
};

FailureOr<SyncSelectedWorld> buildSelectedWorld(const SyncOneShotPlan& plan, const ChannelAnalysisResult& channels);
FailureOr<SyncSelectedWorld> buildSelectedWorld(const SyncReadyReleasePlan& plan);
FailureOr<SyncInterpretationResult> interpretSelectedWorld(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncSelectedWorld& world, ProtocolSyncStatistics* statistics = nullptr,
    const SyncInterpretationOptions& options = {});
void printResidualObligations(
    func::FuncOp function, const SyncSelectedWorld& world, const SyncInterpretationResult& result,
    llvm::raw_ostream& output);

llvm::StringRef stringifySyncObligationKind(SyncObligationKind kind);
llvm::StringRef stringifySyncControlRelation(SyncControlRelation relation);
llvm::StringRef stringifySyncIterationRelationKind(SyncIterationRelationKind kind);
llvm::StringRef stringifySyncSelectedProtocolKind(SyncSelectedProtocolKind kind);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_RESIDUALOBLIGATION_H
