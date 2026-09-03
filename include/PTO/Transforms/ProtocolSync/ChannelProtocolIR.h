// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ChannelProtocolIR.h - Diagnostic lifecycle channels ----*- C++ -*-===//
//
// Channels summarize availability and reclamation relations. Checkpoint C
// admits semantic channel shapes only; it does not select or lower protocols.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_CHANNELPROTOCOLIR_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_CHANNELPROTOCOLIR_H

#include "PTO/Transforms/ProtocolSync/StorageTimeline.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir::pto::protocol_sync {

struct LegacySyncSnapshot;

using SyncChannelId = std::uint32_t;

enum class SyncChannelKind : std::uint8_t {
    OneShot,
    ReadyRelease,
};

enum class SyncChannelRejection : std::uint8_t {
    None,
    TimelineRejected,
    UnresolvedScheduleFailure,
    MultipleProducers,
    MultipleConsumers,
    UnsupportedControlFlow,
    NestedLoop,
    CrossCore,
    UnknownStageRole,
    StaticOverwrite,
    ReuseCapacityMismatch,
};

enum class SyncDemandOracleStatus : std::uint8_t {
    Match,
    Mismatch,
    NotApplicable,
    Unavailable,
};

struct SyncChannel {
    SyncChannelId id = kInvalidSyncId;
    SyncGenerationId generation = kInvalidSyncId;
    SyncChannelKind kind = SyncChannelKind::OneShot;
    unsigned capacity = 0;
    SyncChannelRejection rejection = SyncChannelRejection::None;
    SyncDemandOracleStatus readyOracle = SyncDemandOracleStatus::Unavailable;
    SyncDemandOracleStatus releaseOracle = SyncDemandOracleStatus::Unavailable;

    bool isAdmitted() const { return rejection == SyncChannelRejection::None; }
};

class ChannelAnalysisResult {
public:
    llvm::ArrayRef<SyncChannel> getChannels() const { return channels; }
    llvm::MutableArrayRef<SyncChannel> getMutableChannels() { return channels; }

private:
    friend ChannelAnalysisResult analyzeChannels(
        const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, ProtocolSyncStatistics* statistics);
    llvm::SmallVector<SyncChannel, 16> channels;
};

ChannelAnalysisResult analyzeChannels(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, ProtocolSyncStatistics* statistics = nullptr);
void compareWithLegacyDemandOracle(
    const LegacySyncSnapshot& legacy, const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, ChannelAnalysisResult& channels);
void printProtocolSyncChannels(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, llvm::raw_ostream& output);
llvm::StringRef stringifySyncChannelKind(SyncChannelKind kind);
llvm::StringRef stringifySyncChannelRejection(SyncChannelRejection reason);
llvm::StringRef stringifySyncDemandOracleStatus(SyncDemandOracleStatus status);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_CHANNELPROTOCOLIR_H
