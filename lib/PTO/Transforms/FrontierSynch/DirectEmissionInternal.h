// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private endpoint compilation helpers. Source structure and effects remain in SyncInput/MLIR.
#ifndef PTO_FRONTIERSYNCH_DIRECT_EMISSION_INTERNAL_H
#define PTO_FRONTIERSYNCH_DIRECT_EMISSION_INTERNAL_H
#include "PTO/Transforms/FrontierSynch/AnalysisContract.h"
#include "PTO/Transforms/FrontierSynch/GuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/SignedDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/StructuredInputAdapter.h"
#include "PTO/Transforms/FrontierSynch/TraceDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CostLedger.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
enum class ArithmeticClass { Differences, Octagons };
// Precision configuration, independent of quality and requested interfaces.
enum class RegionalRoutePolicy { Economical, GeneralExtension };
// One fixed-point qualification run and schema per unchanged source invocation.
class EndpointRelationImporter {
public:
    EndpointRelationImporter(func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
                             std::shared_ptr<const PhaseIndex> phases = {});
    ~EndpointRelationImporter();
    FailureOr<StructuredInputHandle> build(Operation* scope, std::string& reason);
private:
    struct Impl;
    std::unique_ptr<Impl> implementation;
};
FailureOr<SignedInputs> specializePrimitives(
    StructuredInputHandle input, ArithmeticClass arithmetic, std::string& reason, SignedSpaceHandle space = {});
struct AnalysisAttempt {
    Region* region = nullptr;
    std::string route, outcome, obligation;
    int64_t inclusiveNanoseconds = 0;
};
struct RegionalContext;
class RegionalRequests;
class StationaryCellInput;
class FixedBodyUpper;
struct SingleStreamLoop;
struct SingleStreamPending;
class PendingPlanProvenance;
struct GeneralQueries;
struct GeneralEndpointPlan;
struct SelectedAnalysis {
    enum class Kind { Explicit, Periodic, Signed, Guarded, General, BoundaryLoop };
    Kind kind = Kind::Explicit;
    std::string route;
    AnalysisContract contract;
    std::shared_ptr<const RegionalContext> regionalContext, requestContext;
    SmallVector<std::size_t> sites;
    SmallVector<Demand> generators;
    RankReduction explicitReduction;
    PeriodicDemandReduction periodic;
    std::shared_ptr<const StationaryCellInput> stationary;
    std::shared_ptr<const FixedBodyUpper> upper;
    std::shared_ptr<const SingleStreamLoop> boundaryLoop;
    scf::ForOp loop;
    GuardedDemandAnalysis guarded;
    StructuredInputHandle structured;
    SignedAnalysisHandle signedAnalysis;
    std::shared_ptr<const GeneralQueries> general;
    // Invocation-local memo of the immutable signed-to-general interchange.
    mutable std::shared_ptr<const GeneralQueries> generalInterchange;
    std::shared_ptr<const GeneralEndpointPlan> generalEndpoints;
    SignedRelationHandle context, minimum, native, reachability;
    // Matching maps are qualified before accepting a logical route and reused
    // during emission. This is the executable endpoint analysis interface.
    std::map<std::tuple<PipelineType, PipelineType, bool>, SignedSelectorHandle> endpoints;
    // Finalized immutable results, not a reconstructed control tree.
    SmallVector<AnalysisAttempt> attempts;
    SmallVector<std::shared_ptr<const SelectedAnalysis>> regionalChildren;
    std::size_t cacheHits = 0, inspectedRegions = 0;
};
using SelectedAnalysisHandle = std::shared_ptr<const SelectedAnalysis>;
struct DirectEmissionResult {
    OwningOpRef<func::FuncOp> pending;
    bool emitted = false;
    bool privateSelectors = false;
    std::string route, reason;
    std::size_t sets = 0, waits = 0, barriers = 0;
    SelectedAnalysisHandle selected;
    AnalysisContract logicalContract;
    std::shared_ptr<const SingleStreamPending> singleStreamPending;
    std::shared_ptr<const PendingPlanProvenance> pendingProvenance;
    SmallVector<AnalysisAttempt> attempts;
};
// A privately captured builder seal binds the original source and whole clone,
// including block arguments, successors, empty regions and payload operands.
// The original owning root must outlive realization; erased descendants are rejected.
bool pendingPlanUnchanged(const DirectEmissionResult& result);
llvm::StringRef realizedOrderStatement(const DirectEmissionResult& result);
SelectedAnalysisHandle selectAnalysis(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    SmallVectorImpl<AnalysisAttempt>& attempts, std::string& reason, CostLedger& costs,
    const AnalysisNeeds& needs = AnalysisNeeds::defaultPolicy());
llvm::json::Object selectedAnalysisReport(
    const DirectEmissionResult& result, Attribute retained);
llvm::json::Object costReport(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& source,
    const DirectEmissionResult& result, bool dumpedDemands);
LogicalResult printFrontierReport(func::FuncOp function, llvm::json::Value report);
void finishCostReport(llvm::json::Object& report, const CostLedger& costs, const DirectEmissionResult& result);
Attribute retainSelectedAnalysis(
    func::FuncOp function, const TraceDemandAnalysis& source, const DirectEmissionResult& result);
LogicalResult emitExplicitDemands(
    const TraceDemandAnalysis& trace, const SelectedAnalysis& selected, IRMapping& mapping,
    DirectEmissionResult& result);
LogicalResult liftPeriodicQueries(SelectedAnalysis& selected, const SignedInputs& inputs, std::string& reason);
unsigned periodicEndpointWidth(scf::ForOp loop);
LogicalResult emitPeriodicDemands(
    const TraceDemandAnalysis& trace, const SelectedAnalysis& selected, IRMapping& mapping,
    DirectEmissionResult& result);
LogicalResult emitGuardedDemands(
    const TraceDemandAnalysis& trace, const SelectedAnalysis& selected, IRMapping& mapping,
    DirectEmissionResult& result);
FailureOr<SelectedAnalysisHandle> composeEndpointRegions(
    func::FuncOp function, StructuredInputHandle input, const SignedInputs& primitives,
    SmallVectorImpl<AnalysisAttempt>& attempts, std::string& reason, CostLedger& costs,
    RegionalRequests* regional = nullptr, const AnalysisNeeds& childNeeds = AnalysisNeeds::modeledCovers(),
    bool childMatching = false, RegionalRoutePolicy policy = RegionalRoutePolicy::Economical);
// Emit on a detached clone and publish atomically. Logical keys have no hardware IDs.
DirectEmissionResult emitDirectDemands(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace, CostLedger& costs);
LogicalResult prepareEndpoints(SelectedAnalysis& selected, std::string& reason);
LogicalResult emitPreparedEndpoints(IRMapping& mapping, const SelectedAnalysis& selected, DirectEmissionResult& result);
} // namespace mlir::pto::frontiersynch
#endif
