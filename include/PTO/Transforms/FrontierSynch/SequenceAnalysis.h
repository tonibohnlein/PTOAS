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
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
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
    std::vector<RegionExpressions::Id> visits;
};
struct SequenceOccurrence {
    uint32_t child = 0, type = 0;
    TemplateEndpointAnchor anchor;
    scf::ForOp loop;
    RegionExpressions::Id ordinal = 0;
    std::vector<RegionExpressions::Id> visits;
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
// Record immediately after original whole-function analysis, before preparing
// endpoints. Only a successful state bound to these exact original inputs can
// establish the sequence contract. Failed/unrelated results remain unproved;
// no demand analysis is rerun and no endpoint or allocation proof is inferred.
void recordSequenceContractAttempt(ProgramRecognition& program, const SyncInput& input,
                                   const SequenceAnalysis& analysis);
SequenceAnalysis composeRegionalSequence(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children,
    bool reconstructPrerequisites = true, bool requireEndpoints = true);
// Enclosing loops are fixed during this composition, not repeated by it.
SequenceAnalysis composeRegionalSequenceWithin(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children,
    bool reconstructPrerequisites, bool requireEndpoints, ArrayRef<scf::ForOp> enclosing);
// Analyze original children of a selected sequence, or one explicit/loop node.
// The caller supplies a common arena; original cuts and access records remain borrowed.
SequenceAnalysis analyzeSequenceRegion(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<PhaseIndex> index = {}, bool requireEndpoints = true);
// The resolver is construction-only and receives proper original descendants.
// Specialized phase views retain their own arena/context adapters.
struct NormalizedControlDescription;
struct NumericBodyMathematics;
struct GuardedRotatingSpecialization;
struct GuardedRotatingMathematics;
struct SequenceRegionResolver {
    // Cached guarded mathematics stays private; each request imports a fresh
    // view into the supplied speculative arena before exposing any IDs.
    std::function<std::shared_ptr<GuardedRotatingMathematics>(const GuardedRotatingSpecialization&,
        std::shared_ptr<RegionExpressions>, std::string&)> specializedGuarded;
    std::function<std::shared_ptr<const NumericBodyMathematics>(scf::ForOp,
        const TemplateGeometryConstant&, const TemplateControlConstant&,
        std::shared_ptr<const NormalizedControlDescription>, std::string&)> specializedNumeric;
    std::function<FailureOr<RegionalAnalysis>(std::size_t, std::string&)> finiteArithmetic;
    // Current-form input only; requesting it never selects another alternative.
    std::function<std::shared_ptr<const NormalizedControlDescription>(std::size_t)> normalized;
    std::function<FailureOr<RegionalAnalysis>(std::size_t, bool, std::string&)> region;
    std::function<std::shared_ptr<const MathematicalResult>(std::size_t, AnalysisBackend)> demands;
    std::function<FailureOr<RegionalAnalysis>(std::size_t, AnalysisBackend, std::string&)> exports;
    // Only expression-free mathematics is shared across specialized contexts.
    std::function<std::shared_ptr<const ArithmeticRegionalRelations>(ArithmeticRegionContext,
        const ArithmeticEntryConstant&, std::string&)> specializedDemands;
    explicit operator bool() const { return static_cast<bool>(region); }
    FailureOr<RegionalAnalysis> operator()(std::size_t node, bool endpoints, std::string& error) const
    { return region(node, endpoints, error); }
};
SequenceAnalysis analyzeSequenceRegionWithResolver(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<PhaseIndex> index, bool requireEndpoints, SequenceRegionResolver resolver);
RegionalAnalysis sequenceRegionalResult(const SequenceAnalysis& analysis);
RegionExpressions* sequenceExpressions(SequenceAnalysis& analysis);
NumericalChainQueryCost sequenceNumericalQueryCounts(const SequenceAnalysis& analysis);
// Counts from the latest preparation, before shared emission CSE: imported
// child code and newly prepared crossings (including identity adapters).
std::pair<uint64_t, uint64_t> sequencePreparationCounts(const SequenceAnalysis& analysis);
std::optional<RegionExpressions::Id> sequenceEventReachability(const SequenceAnalysis& analysis,
    uint32_t sourcePort, PeriodicEventKind sourceKind, uint32_t targetPort, PeriodicEventKind targetKind);
std::optional<RegionExpressions::Id> sequenceEventReachability(SequenceAnalysis& analysis,
    SequenceEvent source, SequenceEvent target);
// Preparation-only resolver. It must preserve the supplied exact selected
// order and original occurrence coordinates; it is never retained by a result.
using SequenceEndpointResolver = std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(
    std::size_t, const RegionalAnalysis&, ArrayRef<scf::ForOp>)>;
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceLogicalInsertion(SequenceAnalysis& analysis,
    ArrayRef<scf::ForOp> enclosing = {}, const SequenceEndpointResolver& resolver = {});
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(SequenceAnalysis& analysis);
// Analyze children independently, reconcile their physical ranges, then reduce
// crossings in one shared all-event port graph. Child internals stay intact.
// Preparation is detached; allocation lifetimes are composed through the same reachability interface.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program,
    std::string& error, SequenceCost* cost = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
