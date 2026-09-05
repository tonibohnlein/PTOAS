// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackStrings.cpp - Storage diagnostic names ------------===//

#include "StorageTrackInternal.h"

using namespace mlir;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

SyncStorageSlotBinding mlir::pto::protocol_sync::detail::classifyStorageSlotBinding(const SyncAccess& access)
{
    if (!access.slot) {
        return SyncStorageSlotBinding::Unslotted;
    }
    switch (access.slot->kind) {
        case SyncSlotExpressionKind::Constant:
            return SyncStorageSlotBinding::Constant;
        case SyncSlotExpressionKind::AffineModulo:
            return SyncStorageSlotBinding::AffineConditional;
        case SyncSlotExpressionKind::Unknown:
            return SyncStorageSlotBinding::UnknownConditional;
    }
    return SyncStorageSlotBinding::UnknownConditional;
}

SyncStorageTargetResult mlir::pto::protocol_sync::detail::mapSameLaneCompletionToStorageTarget(
    ProtocolSyncSameLaneCompletion completion)
{
    switch (completion) {
        case ProtocolSyncSameLaneCompletion::Intrinsic:
            return SyncStorageTargetResult::Intrinsic;
        case ProtocolSyncSameLaneCompletion::PipeBarrier:
            return SyncStorageTargetResult::PipeBarrier;
        case ProtocolSyncSameLaneCompletion::UnsupportedTarget:
            return SyncStorageTargetResult::UnsupportedTarget;
        case ProtocolSyncSameLaneCompletion::UnsupportedMechanism:
            return SyncStorageTargetResult::UnsupportedMechanism;
        case ProtocolSyncSameLaneCompletion::NotApplicable:
            return SyncStorageTargetResult::NotApplicable;
    }
    return SyncStorageTargetResult::UnsupportedMechanism;
}

SyncStorageTargetResult mlir::pto::protocol_sync::detail::queryStorageLaneTransfer(
    const ProtocolSyncTarget& target, const LaneFrontierAnalysisResult& laneFrontiers, SyncExecutionLaneId sourceId,
    SyncExecutionLaneId targetId)
{
    if (!target.isSupported()) {
        return SyncStorageTargetResult::UnsupportedTarget;
    }
    const SyncExecutionLane* source = laneFrontiers.findLane(sourceId);
    const SyncExecutionLane* destination = laneFrontiers.findLane(targetId);
    if (!source || !destination) {
        return SyncStorageTargetResult::UnsupportedMechanism;
    }
    if (sourceId == targetId) {
        const ProtocolSyncResource resource{source->core, source->pipe};
        return target.supportsPipeBarrier(resource) ? SyncStorageTargetResult::PipeBarrier :
                                                      SyncStorageTargetResult::UnsupportedMechanism;
    }
    const bool eventSupported =
        target.supportsEvent({source->core, source->pipe}, {destination->core, destination->pipe});
    return eventSupported ? SyncStorageTargetResult::Event : SyncStorageTargetResult::UnsupportedMechanism;
}

void mlir::pto::protocol_sync::detail::refineSameLaneStorageTarget(
    const LanePatternAnalysisResult& lanePatterns, SyncStorageTransitionFrontier& transition)
{
    if (transition.sourceLane != transition.targetLane || transition.rawPairMembers.empty()) {
        return;
    }
    bool sawIntrinsic = false;
    bool sawBarrier = false;
    for (SyncLaneRawAccessPairId pairId : transition.rawPairMembers) {
        if (pairId >= lanePatterns.getRawAccessPairs().size()) {
            transition.targetResult = SyncStorageTargetResult::UnsupportedMechanism;
            return;
        }
        switch (lanePatterns.getRawAccessPairs()[pairId].completion) {
            case ProtocolSyncSameLaneCompletion::Intrinsic:
                sawIntrinsic = true;
                break;
            case ProtocolSyncSameLaneCompletion::PipeBarrier:
                sawBarrier = true;
                break;
            case ProtocolSyncSameLaneCompletion::UnsupportedTarget:
                transition.targetResult = SyncStorageTargetResult::UnsupportedTarget;
                return;
            case ProtocolSyncSameLaneCompletion::UnsupportedMechanism:
            case ProtocolSyncSameLaneCompletion::NotApplicable:
                transition.targetResult = SyncStorageTargetResult::UnsupportedMechanism;
                return;
        }
    }
    if (sawBarrier) {
        transition.targetResult = SyncStorageTargetResult::PipeBarrier;
    } else if (sawIntrinsic) {
        transition.targetResult = SyncStorageTargetResult::Intrinsic;
    }
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageSlotBinding(SyncStorageSlotBinding binding)
{
    switch (binding) {
        case SyncStorageSlotBinding::Unslotted:
            return "unslotted";
        case SyncStorageSlotBinding::Constant:
            return "constant";
        case SyncStorageSlotBinding::AffineConditional:
            return "affine-conditional";
        case SyncStorageSlotBinding::UnknownConditional:
            return "unknown-conditional";
    }
    return "unknown-conditional";
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageProjectionRejection(SyncStorageProjectionRejection rejection)
{
    switch (rejection) {
        case SyncStorageProjectionRejection::MissingFamily:
            return "missing-family";
        case SyncStorageProjectionRejection::NonPhysical:
            return "non-physical";
        case SyncStorageProjectionRejection::UnknownRange:
            return "unknown-range";
        case SyncStorageProjectionRejection::InvalidInterval:
            return "invalid-interval";
    }
    return "unknown-range";
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageTransitionKind(SyncStorageTransitionKind kind)
{
    switch (kind) {
        case SyncStorageTransitionKind::Ready:
            return "ready";
        case SyncStorageTransitionKind::Release:
            return "release";
        case SyncStorageTransitionKind::Completion:
            return "completion";
        case SyncStorageTransitionKind::Residual:
            return "residual";
    }
    return "residual";
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageTransitionOrigin(SyncStorageTransitionOrigin origin)
{
    switch (origin) {
        case SyncStorageTransitionOrigin::LaneFrontier:
            return "lane-frontier";
        case SyncStorageTransitionOrigin::CompletionCut:
            return "completion-cut";
        case SyncStorageTransitionOrigin::RawAccessPair:
            return "raw-access-pair";
    }
    return "raw-access-pair";
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageTargetResult(SyncStorageTargetResult result)
{
    switch (result) {
        case SyncStorageTargetResult::Intrinsic:
            return "intrinsic";
        case SyncStorageTargetResult::PipeBarrier:
            return "pipe-barrier";
        case SyncStorageTargetResult::Event:
            return "event";
        case SyncStorageTargetResult::UnsupportedTarget:
            return "unsupported-target";
        case SyncStorageTargetResult::UnsupportedMechanism:
            return "unsupported-mechanism";
        case SyncStorageTargetResult::NotApplicable:
            return "not-applicable";
    }
    return "not-applicable";
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageLifecycleRejection(SyncStorageLifecycleRejection rejection)
{
    switch (rejection) {
        case SyncStorageLifecycleRejection::None:
            return "none";
        case SyncStorageLifecycleRejection::NotRecurring:
            return "not-recurring";
        case SyncStorageLifecycleRejection::UnsupportedControl:
            return "unsupported-control";
        case SyncStorageLifecycleRejection::MultipleLoops:
            return "multiple-loops";
        case SyncStorageLifecycleRejection::UnsupportedAccessShape:
            return "unsupported-access-shape";
        case SyncStorageLifecycleRejection::ProducerAfterConsumer:
            return "producer-after-consumer";
        case SyncStorageLifecycleRejection::IncompleteTrackSet:
            return "incomplete-track-set";
        case SyncStorageLifecycleRejection::UnknownCapacity:
            return "unknown-capacity";
        case SyncStorageLifecycleRejection::UnsupportedCapacity:
            return "unsupported-capacity";
        case SyncStorageLifecycleRejection::UnknownSlotRelation:
            return "unknown-slot-relation";
        case SyncStorageLifecycleRejection::InvalidReuseDistance:
            return "invalid-reuse-distance";
        case SyncStorageLifecycleRejection::UnresolvedLane:
            return "unresolved-lane";
    }
    return "unsupported-access-shape";
}

StringRef mlir::pto::protocol_sync::stringifySyncStorageEDifferentialStatus(SyncStorageEDifferentialStatus status)
{
    switch (status) {
        case SyncStorageEDifferentialStatus::Match:
            return "match";
        case SyncStorageEDifferentialStatus::Mismatch:
            return "mismatch";
        case SyncStorageEDifferentialStatus::IndependentOnly:
            return "independent-only";
        case SyncStorageEDifferentialStatus::EOnly:
            return "e-only";
        case SyncStorageEDifferentialStatus::Neither:
            return "neither";
        case SyncStorageEDifferentialStatus::Unavailable:
            return "unavailable";
    }
    return "unavailable";
}
