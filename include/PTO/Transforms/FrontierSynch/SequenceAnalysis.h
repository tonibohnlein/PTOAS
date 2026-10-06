// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_SEQUENCEANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_SEQUENCEANALYSIS_H
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include <map>
namespace mlir::pto::frontiersynch {
using SequenceCost = RegionalCost;
struct SequenceSelectedEvent {
    uint32_t port = 0;
    RegionExpressions::Id present = 0;
};
struct SequenceStorageBoundary {
    SyncStorageCell cell;
    std::vector<SequenceSelectedEvent> firstWriters, lastWriters;
    std::map<uint32_t, std::vector<SequenceSelectedEvent>> firstReaders, lastReaders;
};
struct SequenceEvent {
    uint32_t child = 0, type = 0;
    RegionExpressions::Id ordinal = 0;
    PeriodicEventKind kind = PeriodicEventKind::Start;
};
struct SequenceOccurrence {
    uint32_t child = 0, type = 0;
    TemplateEndpointAnchor anchor;
    scf::ForOp loop;
    RegionExpressions::Id ordinal = 0;
};
struct SequenceAnalysisState;
struct SequenceAnalysis {
    std::string error;
    std::string insertionError;
    SequenceCost cost;
    std::vector<SequenceOccurrence> occurrences;
    std::vector<SequenceStorageBoundary> storageBoundary;
    std::map<uint32_t, std::vector<SequenceSelectedEvent>> firstPayloads, lastPayloads;
    std::shared_ptr<SequenceAnalysisState> state;
};
// The state owns expression/query DAGs and internal demand recipes. Physical
// effects, recognition input and original IR remain borrowed; invalidate after any IR change.
SequenceAnalysis analyzeSequence(func::FuncOp function, const SyncInput& input,
                                 const ProgramRecognition& program);
SequenceAnalysis composeRegionalSequence(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children);
RegionalAnalysis sequenceRegionalResult(const SequenceAnalysis& analysis);
RegionExpressions* sequenceExpressions(SequenceAnalysis& analysis);
std::optional<RegionExpressions::Id> sequenceEventReachability(const SequenceAnalysis& analysis,
    uint32_t sourcePort, PeriodicEventKind sourceKind, uint32_t targetPort, PeriodicEventKind targetKind);
std::optional<RegionExpressions::Id> sequenceEventReachability(SequenceAnalysis& analysis,
    SequenceEvent source, SequenceEvent target);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(SequenceAnalysis& analysis);
// Analyze children independently, reconcile their physical ranges, then reduce
// crossings in one shared all-event port graph. Child internals stay intact.
// Preparation is detached; allocation lifetimes are composed through the same reachability interface.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program,
    std::string& error, SequenceCost* cost = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
