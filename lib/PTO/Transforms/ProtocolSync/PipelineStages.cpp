// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- PipelineStages.cpp - Conservative discovery stages --------------===//

#include "PTO/Transforms/ProtocolSync/PipelineStages.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool accessReads(SyncAccessMode mode)
{
    return mode == SyncAccessMode::Read || mode == SyncAccessMode::ReadWrite || mode == SyncAccessMode::Ordered;
}

bool accessWrites(SyncAccessMode mode)
{
    return mode == SyncAccessMode::Write || mode == SyncAccessMode::ReadWrite || mode == SyncAccessMode::Ordered;
}

SyncStageRole classifyStage(const StructuredSyncIR& schedule, const SyncStage& stage)
{
    bool readsGlobal = false;
    bool writesGlobal = false;
    bool readsLocal = false;
    bool writesLocal = false;
    for (SyncAccessId id : stage.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        if (!access) {
            return SyncStageRole::Unknown;
        }
        const bool local = access->visibility == SyncVisibilityClass::Local;
        const bool global = access->visibility == SyncVisibilityClass::Global;
        readsLocal = readsLocal || (local && accessReads(access->mode));
        writesLocal = writesLocal || (local && accessWrites(access->mode));
        readsGlobal = readsGlobal || (global && accessReads(access->mode));
        writesGlobal = writesGlobal || (global && accessWrites(access->mode));
    }
    if (readsGlobal && writesLocal) {
        return SyncStageRole::CopyIn;
    }
    if (readsLocal && writesGlobal) {
        return SyncStageRole::CopyOut;
    }
    if (readsLocal && writesLocal) {
        return SyncStageRole::Compute;
    }
    if (writesLocal) {
        return SyncStageRole::Publish;
    }
    if (readsLocal) {
        return SyncStageRole::Acquire;
    }
    return SyncStageRole::Unknown;
}

} // namespace

const SyncStage* PipelineStageAnalysisResult::findStage(SyncStageId id) const
{
    return id < stages.size() ? &stages[id] : nullptr;
}

const SyncStage* PipelineStageAnalysisResult::findStageForPhase(SyncPhaseId phase) const
{
    if (phase >= phaseToStage.size()) {
        return nullptr;
    }
    return findStage(phaseToStage[phase]);
}

FailureOr<PipelineStageAnalysisResult> mlir::pto::protocol_sync::analyzePipelineStages(const StructuredSyncIR& schedule)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    PipelineStageAnalysisResult result;
    result.phaseToStage.assign(schedule.getPhases().size(), kInvalidSyncId);
    for (const SyncPhase& phase : schedule.getPhases()) {
        SyncStage stage;
        stage.id = result.stages.size();
        stage.region = phase.region;
        stage.core = phase.core;
        stage.pipe = phase.pipe;
        stage.guard = phase.guard;
        stage.iterationDomain = phase.iterationDomain;
        stage.phases.push_back(phase.id);
        stage.accesses.append(phase.accesses.begin(), phase.accesses.end());
        stage.role = classifyStage(schedule, stage);
        result.phaseToStage[phase.id] = stage.id;
        result.stages.push_back(std::move(stage));
    }
    return result;
}

StringRef mlir::pto::protocol_sync::stringifySyncStageRole(SyncStageRole role)
{
    switch (role) {
        case SyncStageRole::CopyIn:
            return "copy-in";
        case SyncStageRole::Compute:
            return "compute";
        case SyncStageRole::CopyOut:
            return "copy-out";
        case SyncStageRole::Publish:
            return "publish";
        case SyncStageRole::Acquire:
            return "acquire";
        case SyncStageRole::Unknown:
            return "unknown";
    }
    return "unknown";
}
