// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LanePatternInternal.h - Lane pattern experiment helpers ---------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_LANEPATTERNINTERNAL_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_LANEPATTERNINTERNAL_H

#include "PTO/Transforms/ProtocolSync/LanePatternAnalysis.h"

namespace mlir::pto::protocol_sync::detail {

llvm::SmallVector<SyncLaneRawAccessPair, 32> buildLaneRawAccessPairs(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LaneFrontierAnalysisResult& frontiers);
void appendSameLaneCompletionCuts(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    llvm::ArrayRef<SyncLaneRawAccessPair> rawPairs, llvm::SmallVectorImpl<SyncLanePatternCandidate>& candidates);
void appendSharedOneShotFrontiers(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LaneFrontierAnalysisResult& frontiers, llvm::SmallVectorImpl<SyncLanePatternCandidate>& candidates);
void appendChoiceBalancedRoundTrips(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const LaneFrontierAnalysisResult& frontiers, llvm::SmallVectorImpl<SyncLanePatternCandidate>& candidates);

} // namespace mlir::pto::protocol_sync::detail

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LANEPATTERNINTERNAL_H
