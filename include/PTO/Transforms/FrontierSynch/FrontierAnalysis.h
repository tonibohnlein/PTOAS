// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owning MLIR analysis for the frontier synchronization pass. Step 0 storage
// outlives all borrowed structure/recognition links. Default MLIR invalidation
// discards this state after an IR-changing pass.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERANALYSIS_H
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
namespace mlir::pto::frontiersynch {
// Detached whole-function producer shared by module coordination and tests.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFunctionSynchronization(
    func::FuncOp function, GMAliasPolicy policy);
struct ArithmeticPeriodicExports {
    bool endpointsAvailable = false, allocationAvailable = false;
    std::string endpointError, allocationError;
};
struct FiniteVisitAnalysis;
struct BoundedLifetimeDemandResult;
struct AnalysisSessionState;
struct NumericTemplatePlan;
class FrontierAnalysis {
public:
    explicit FrontierAnalysis(Operation* operation) : function(dyn_cast<func::FuncOp>(operation)) {}
    // Callers needing structural diagnostics retain the default. The logical
    // pass may defer recognition when the whole-function native proof succeeds.
    LogicalResult initialize(GMAliasPolicy policy = GMAliasPolicy::MayNotAlias, bool requireStructure = true);
    // Request adapters are migrated by backend. A failed export retains the
    // owned mathematical result; stateful evaluation is a separate capability.
    AnalysisOutcome minimumDemands(std::size_t region = 0, AnalysisNeeds exports = {});
    AnalysisOutcome analyze(const AnalysisRequest& request);
    // Explicit finite-visit adapter for callers selecting this representation.
    // Missing arbitrary-word interfaces retain the exact demand templates.
    AnalysisOutcome analyzeFiniteVisit(const AnalysisRequest& request);
    AnalysisOutcome analyzeVaryingBoundary(const AnalysisRequest& request);
    AnalysisOutcome analyzeFiniteExpansion(const AnalysisRequest& request);
    AnalysisOutcome analyzeArithmeticRegional(const AnalysisRequest& request);
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareLogical(const AnalysisOutcome& result);
    LogicalResult attachAllocation(const AnalysisOutcome& result, PreparedLogicalPlan& plan);
    const AnalysisConstructionCounts& constructionCounts() const { return construction; }
    uint64_t arithmeticGeneratorConstructions() const;
    uint64_t arithmeticRegionConstructions() const;
    uint64_t specializedArithmeticConstructions() const;
    // Constants must be deterministic within this request. Reuse compares every
    // observed lookup, including unavailable values, in the original context.
    std::shared_ptr<const ArithmeticRegionalRelations> specializedArithmeticDemands(
        ArithmeticRegionContext context, const ArithmeticEntryConstant& constants, std::string& error,
        ArrayRef<ArithmeticLimits> profiles = {});
    uint64_t finiteExpansionPreflights() const;
    uint64_t numericTemplatePreflights() const;
    uint64_t numericalRegionConstructions() const;
    uint64_t normalizationConstructions() const;
    AnalysisOutcome analyzeNumericalRegion(const AnalysisRequest& request);
    // Rank supported arithmetic reducers without constructing generators.
    std::vector<AnalysisBackend> arithmeticMethods(const AnalysisRequest& request);
    // Rank descriptions once per request without producing demands or exports.
    std::vector<AnalysisForm> exactForms(const AnalysisRequest& request);
    std::vector<AnalysisCostRecord> costRecords() const;
    // Call before reusing this object after original IR mutation. Previously
    // returned handles must no longer be queried against that IR.
    void invalidate();
    // Build and cache the whole-function arithmetic candidate only on request.
    // Requires successful initialization; uses fixed class limits, not input-derived limits.
    LogicalResult recognizeArithmetic();
    // Profiles are immutable once any arithmetic form is requested. Defaults
    // preserve the separately declared whole-function and regional classes.
    LogicalResult configureArithmeticProfiles(ArrayRef<ArithmeticLimits> whole,
                                              ArrayRef<ArithmeticLimits> regional);
    LogicalResult recognizeRegionalArithmetic();
    // Audit every original region, including children of a compact match.
    // Results own their exact mathematics; no endpoint or allocation is built.
    std::vector<RegionCertification> certifyRegions();
    // Demand reduction, endpoint recipes and allocation are backend work.
    // Cache both successful and failed numeric attempts without changing recognition.
    LogicalResult analyzeNumericCandidates();
    LogicalResult prepareNumericCandidateExports();
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareRotatingFunction(bool guarded);
    bool hasOnlyNativeScalarRequirements() const { return nativeScalarOnly; }
    LogicalResult analyzeExplicitFunction();
    SequenceAnalysis* analyzeSequenceFunction();
    // Demand-only regional libraries; their success never implies a whole-loop
    // query, executable type-selection recipe or physical allocation export.
    LogicalResult analyzeFiniteVisitCandidates();
    const std::map<std::size_t, std::shared_ptr<const FiniteVisitAnalysis>>& finiteVisitDemands() const {
        return finiteVisitAnalyses;
    }
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeFunction(std::string& error);
    const BoundedLifetimeDemandResult* boundedLifetimeDemands() const { return boundedAnalysis.get(); }
    LogicalResult analyzeArithmeticFunction();
    // Pause at exact generators before attempting the distance-interval adapter.
    // Failed conversion/endpoint export can resume this stage without rebuilding.
    LogicalResult analyzeArithmeticPeriodicFunction();
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticPeriodicFunction();
    const ArithmeticPeriodicExports& arithmeticPeriodicExports() const { return periodicExports; }
    const ArithmeticPeriodicProgram* arithmeticPeriodicDemands() const {
        return arithmeticPeriodicAnalysis ? &*arithmeticPeriodicAnalysis : nullptr;
    }
    const ArithmeticDemandAnalysis* arithmeticDemands() const {
        return arithmeticAnalysis ? &*arithmeticAnalysis : nullptr;
    }
    const GeneralArithmeticDemandAnalysis* generalArithmeticDemands() const {
        return generalArithmeticAnalysis ? &*generalArithmeticAnalysis : nullptr;
    }
    // Whole-invocation evidence only. Endpoint preparation cannot revoke it;
    // success for a proper child does not establish this property.
    bool hasWholeFunctionMinimumDemands() const;
    void noteSequenceEndpointOutcome(StringRef error);
    const ExplicitAnalysis* explicitResult() const { return explicitAnalysis ? &*explicitAnalysis : nullptr; }
    const SyncInput* input() const { return storage.get(); }
    std::shared_ptr<const SyncInput> sharedInput() const { return storage; }
    const ProgramRecognition* result() const { return program ? &*program : nullptr; }
private:
    LogicalResult recognizeStructure();
    std::shared_ptr<const NormalizedControlDescription> normalizedRegion(std::size_t region);
    const NumericTemplatePlan* numericalPreflight(std::size_t region);
    std::shared_ptr<const MathematicalResult> produceExpandedFinite(std::size_t region, std::string& error);
    LogicalResult ensureArithmeticGenerators();
    const NumericTemplate& materializeNumericRegion(std::size_t region);
    const ArithmeticProgram* recognizeArithmeticRegion(std::size_t region);
    SequenceRegionResolver regionalResolver(
        std::shared_ptr<const MathematicalResult> retainedNumerical = {});
    std::shared_ptr<const MathematicalResult> adaptNumericalSequence(std::size_t region, std::string& error);
    std::shared_ptr<const RegionalAnalysis> arithmeticExports(
        const MathematicalResult& demands, bool selectors, std::string& error);
    std::shared_ptr<const RegionalAnalysis> varyingExports(
        const MathematicalResult& demands, bool selectors, std::string& error);
    void recordWholeRegion(AnalysisBackend backend);
    AnalysisOutcome requestBackend(AnalysisBackend backend, const AnalysisRequest& request);
    std::shared_ptr<const MathematicalResult> produceBackend(AnalysisBackend backend, std::string& error);
    std::shared_ptr<const MathematicalResult> produceRegionBackend(
        AnalysisBackend backend, std::size_t region, std::string& error);
    std::shared_ptr<const MathematicalResult> produceArithmeticRegion(std::size_t region, std::string& error);
    std::shared_ptr<const MathematicalResult> produceLoopBackend(
        AnalysisBackend backend, std::string& error, std::size_t region = 0);
    std::shared_ptr<const MathematicalResult> constructLoopBackend(
        AnalysisBackend backend, std::size_t region, std::string& error);
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareRetained(const MathematicalResult& demands);
    std::shared_ptr<AnalysisSessionState> sessionState;
    func::FuncOp function;
    SmallVector<ArithmeticLimits> wholeArithmeticProfiles{{8, 8, 1, 8}, {8, 8, 2, 8}};
    SmallVector<ArithmeticLimits> regionalArithmeticProfiles{{8, 8, 1, 4096}, {8, 8, 2, 4096}};
    std::map<std::size_t, std::shared_ptr<const ArithmeticProgram>> arithmeticForms;
    bool initialized = false;
    bool nativeScalarOnly = false;
    GMAliasPolicy policy = GMAliasPolicy::MayNotAlias;
    std::shared_ptr<SyncInput> storage;
    std::shared_ptr<ProgramRecognition> program;
    std::shared_ptr<PhaseIndex> structuralIndex;
    std::shared_ptr<ExplicitAnalysis> explicitAnalysis;
    AnalysisConstructionCounts construction;
    std::shared_ptr<BoundedLifetimeDemandResult> boundedAnalysis;
    std::shared_ptr<SequenceAnalysis> sequenceAnalysis;
    std::shared_ptr<ArithmeticDemandAnalysis> arithmeticAnalysis;
    std::shared_ptr<GeneralArithmeticDemandAnalysis> generalArithmeticAnalysis;
    std::optional<GeneralArithmeticGeneratorStage> arithmeticGeneratorStage;
    std::shared_ptr<ArithmeticPeriodicProgram> arithmeticPeriodicAnalysis;
    ArithmeticPeriodicExports periodicExports;
    std::map<std::size_t, std::shared_ptr<const FiniteVisitAnalysis>> finiteVisitAnalyses;
};
} // namespace mlir::pto::frontiersynch
#endif
