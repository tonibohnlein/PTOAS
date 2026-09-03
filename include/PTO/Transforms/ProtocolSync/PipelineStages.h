// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- PipelineStages.h - Conservative protocol discovery stages -*- C++ -*-===//
//
// Stages are immutable discovery evidence. Exact SyncPhase records remain the
// correctness identity; this first analysis deliberately creates one stage per
// phase and therefore never merges across a semantic boundary.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_PIPELINESTAGES_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_PIPELINESTAGES_H

#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::pto::protocol_sync {

using SyncStageId = std::uint32_t;

enum class SyncStageRole : std::uint8_t {
    CopyIn,
    Compute,
    CopyOut,
    Publish,
    Acquire,
    Unknown,
};

struct SyncStage {
    SyncStageId id = kInvalidSyncId;
    SyncRegionId region = kInvalidSyncId;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncIterationDomain iterationDomain;
    llvm::SmallVector<SyncPhaseId, 1> phases;
    llvm::SmallVector<SyncAccessId, 8> accesses;
    SyncStageRole role = SyncStageRole::Unknown;
};

class PipelineStageAnalysisResult {
public:
    llvm::ArrayRef<SyncStage> getStages() const { return stages; }
    const SyncStage* findStage(SyncStageId id) const;
    const SyncStage* findStageForPhase(SyncPhaseId phase) const;

private:
    friend FailureOr<PipelineStageAnalysisResult> analyzePipelineStages(const StructuredSyncIR& schedule);
    llvm::SmallVector<SyncStage, 32> stages;
    llvm::SmallVector<SyncStageId, 32> phaseToStage;
};

FailureOr<PipelineStageAnalysisResult> analyzePipelineStages(const StructuredSyncIR& schedule);
llvm::StringRef stringifySyncStageRole(SyncStageRole role);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_PIPELINESTAGES_H
