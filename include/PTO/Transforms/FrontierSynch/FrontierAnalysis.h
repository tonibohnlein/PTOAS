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
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticPeriodicConversion.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include <functional>
#include <utility>
namespace mlir::pto::frontiersynch {
// Detached whole-function producer shared by module coordination and tests.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFunctionSynchronization(
    func::FuncOp function, GMAliasPolicy policy, bool completeInvocation = false);
struct FiniteVisitAnalysis;
struct AnalysisSessionState;
struct BackendAttempt;
struct NumericTemplatePlan;
class FrontierAnalysis {
public:
    // Allocation policy is immutable for this owner and all of its cached requests.
    // invalidate() discards answers without changing the policy.
    explicit FrontierAnalysis(Operation* operation,
        ArithmeticRegionalAllocationPolicy allocationPolicy = ArithmeticRegionalAllocationPolicy::SingleId)
        : function(dyn_cast<func::FuncOp>(operation)), regionalAllocationPolicy(allocationPolicy) {}
    ArithmeticRegionalAllocationPolicy arithmeticAllocationPolicy() const { return regionalAllocationPolicy; }
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
    // A nonempty pool enables parent-aware regional candidate selection before
    // scalar materialization. Other providers retain their existing certificate
    // export; the physical allocator still validates fit and command bindings.
    LogicalResult attachAllocation(const AnalysisOutcome& result, PreparedLogicalPlan& plan,
        ArrayRef<int64_t> eligibleIds = {});
    const AnalysisConstructionCounts& constructionCounts() const { return construction; }
    uint64_t arithmeticGeneratorConstructions() const;
    uint64_t arithmeticRegionConstructions() const;
    uint64_t finiteArithmeticStorageProbes() const;
    // Gate the finite-only repetition adapter before constructing mathematics;
    // a successful gate reuses canonical query and selector exports.
    FailureOr<RegionalAnalysis> finiteArithmeticRegion(std::size_t region, std::string& error);
    uint64_t specializedGuardedConstructions() const;
    uint64_t guardedExpressionImportWork() const;
    std::shared_ptr<const GuardedRotatingMathematics> specializedGuardedDemands(
        const GuardedRotatingSpecialization& request, std::string& error);
    std::shared_ptr<GuardedRotatingMathematics> instantiateGuardedDemands(
        const GuardedRotatingSpecialization& request, std::shared_ptr<RegionExpressions> target, std::string& error);
    uint64_t specializedNumericConstructions() const;
    std::shared_ptr<const NumericBodyMathematics> specializedNumericDemands(scf::ForOp loop,
        const TemplateGeometryConstant& geometry, const TemplateControlConstant& control, std::string& error,
        NumericTemplateLimits limits = {}, std::shared_ptr<const NormalizedControlDescription> normalized = {},
        TemplateGeometryPolicy policy = TemplateGeometryPolicy::AllCertifiedMaps);
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
    // Original syntax planning is shared; regional views do not add visits.
    uint64_t normalizationSourceVisits() const;
    // Explicit audit workload; production requests materialize forms on demand.
    LogicalResult recognizeAllForms();
    AnalysisOutcome analyzeNumericalRegion(const AnalysisRequest& request);
    // Rank supported arithmetic reducers without constructing generators.
    std::vector<AnalysisBackend> arithmeticMethods(const AnalysisRequest& request);
    // Rank descriptions once per request without producing demands or exports.
    std::vector<AnalysisForm> exactForms(const AnalysisRequest& request);
    std::vector<AnalysisCostRecord> costRecords() const;
    // Enable before initialize. Profiling reports are separate from generated IR.
    void enableStageProfiling(bool enabled = true);
    std::shared_ptr<const AnalysisLedger> diagnosticLedger() const;
    void writeStageProfile(llvm::raw_ostream& stream, StringRef workload) const;
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
    // Exhaustive diagnostic/test workloads, separate from production request
    // selection. Populate the legacy recognition snapshots inspected by audits.
    LogicalResult auditNumericCandidates();
    LogicalResult auditNumericCandidateExports();
    bool hasOnlyNativeScalarRequirements() const { return nativeScalarOnly; }
    // Named backend requests; these never select a different exact reducer.
    LogicalResult analyzeExplicitFunction();
    // Concrete composition oracle, including where production can use an
    // identity wrapper instead. Ordinary callers use analyze/minimumDemands.
    SequenceAnalysis* auditSequenceFunction();
    // Exhaustive diagnostic/test workload and its retained demand inventory.
    // Success implies no whole-loop query, endpoint or allocation export.
    LogicalResult auditFiniteVisitCandidates();
    const std::map<std::size_t, std::shared_ptr<const FiniteVisitAnalysis>>& auditedFiniteVisitDemands() const {
        return finiteVisitAnalyses;
    }
    LogicalResult analyzeArithmeticFunction();
    // Pause at exact generators before attempting the distance-interval adapter.
    // Failed conversion/endpoint export can resume this stage without rebuilding.
    LogicalResult analyzeArithmeticPeriodicFunction();
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
    const ExplicitAnalysis* explicitResult() const { return concreteFinite ? concreteFinite->concrete() : nullptr; }
    const SyncInput* input() const { return storage.get(); }
    std::shared_ptr<const SyncInput> sharedInput() const { return storage; }
    // Diagnostic snapshots are current at this boundary. Reacquire after a
    // mutating session request; retained pointers are not live audit views.
    const ProgramRecognition* result() const {
        if (program) { materializeProgramContractAudit(*program); }
        return program ? &*program : nullptr;
    }
private:
    LogicalResult produceExplicitFunction();
    SequenceAnalysis* produceSequenceFunction();
    LogicalResult produceArithmeticFunction();
    LogicalResult produceArithmeticPeriodicFunction();
    AnalysisOutcome requestWholeFunctionBackend(AnalysisBackend backend);
    LogicalResult recognizeStructure();
    void recognizeForm(std::size_t region, ProgramForm form);
    std::shared_ptr<const NormalizedControlDescription> normalizedRegion(std::size_t region);
    const NumericTemplatePlan* numericalPreflight(std::size_t region);
    std::shared_ptr<const MathematicalResult> produceExpandedFinite(std::size_t region, std::string& error);
    LogicalResult ensureArithmeticGenerators();
    const NumericTemplate& materializeNumericRegion(std::size_t region);
    const ArithmeticProgram* recognizeArithmeticRegion(std::size_t region);
    SequenceRegionResolver regionalResolver(
        std::shared_ptr<const MathematicalResult> retainedNumerical = {});
    std::shared_ptr<const MathematicalResult> adaptNumericalSequence(std::size_t region, std::string& error);
    std::shared_ptr<const RegionalAnalysis> boundedExports(
        const MathematicalResult& demands, bool selectors, std::string& error);
    std::shared_ptr<const RegionalAnalysis> arithmeticExports(
        const MathematicalResult& demands, bool selectors, std::string& error);
    std::shared_ptr<const RegionalAnalysis> varyingExports(
        const MathematicalResult& demands, bool selectors, std::string& error);
    void recordWholeRegion(AnalysisBackend backend);
    AnalysisOutcome requestBackend(AnalysisBackend backend, const AnalysisRequest& request,
        bool sameNodeContinuation = false);
    std::vector<AnalysisBackend> exactAttemptFamilies(
        const AnalysisRequest& request, AnalysisForm form, bool alternative);
    std::vector<AnalysisBackend> rankExactCandidates(const AnalysisRequest& request,
        const std::vector<AnalysisBackend>& candidates);
    std::pair<AnalysisOutcome, AnalysisBackend> selectIdentityChild(
        const AnalysisRequest& request, bool continueSameRegion);
    std::pair<AnalysisOutcome, AnalysisBackend> continueIdentityChild(const AnalysisRequest& request);
    std::shared_ptr<const MathematicalResult> produceIdentitySequence(
        const AnalysisRequest& request, bool endpoints, std::string& error);
    AnalysisOutcome requestIdentity(const AnalysisRequest& request, BackendAttempt& attempt);
    DictionaryAttr allocateIdentity(const MathematicalResult& demands, PreparedLogicalPlan& plan,
        class AllocationProfile* profile = nullptr);
    std::shared_ptr<const MathematicalResult> produceBackend(
        AnalysisBackend backend, const AnalysisRequest& request, std::string& error);
    std::shared_ptr<const MathematicalResult> produceRegionBackend(
        AnalysisBackend backend, const AnalysisRequest& request, std::string& error);
    std::shared_ptr<const MathematicalResult> produceArithmeticRegion(std::size_t region, std::string& error);
    std::shared_ptr<const MathematicalResult> produceLoopBackend(
        AnalysisBackend backend, std::string& error, std::size_t region = 0);
    std::shared_ptr<const MathematicalResult> constructLoopBackend(
        AnalysisBackend backend, std::size_t region, std::string& error);
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareRetained(const MathematicalResult& demands);
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareRetainedPrefix(
        std::shared_ptr<const MathematicalResult> demands, std::string& error);
    std::shared_ptr<AnalysisSessionState> sessionState;
    func::FuncOp function;
    const ArithmeticRegionalAllocationPolicy regionalAllocationPolicy;
    SmallVector<ArithmeticLimits> wholeArithmeticProfiles{{8, 8, 1, 8}, {8, 8, 2, 8}};
    SmallVector<ArithmeticLimits> regionalArithmeticProfiles{{8, 8, 1, 4096}, {8, 8, 2, 4096}};
    std::map<std::size_t, std::shared_ptr<const ArithmeticProgram>> arithmeticForms;
    bool initialized = false;
    bool profileStages = false;
    bool nativeScalarOnly = false;
    GMAliasPolicy policy = GMAliasPolicy::MayNotAlias;
    std::shared_ptr<SyncInput> storage;
    std::shared_ptr<ProgramRecognition> program;
    std::shared_ptr<PhaseIndex> structuralIndex;
    std::shared_ptr<const FiniteOccurrenceInput> finiteInput;
    std::shared_ptr<FiniteAnalysis> concreteFinite;
    AnalysisConstructionCounts construction;
    std::shared_ptr<SequenceAnalysis> sequenceAnalysis;
    std::shared_ptr<ArithmeticDemandAnalysis> arithmeticAnalysis;
    std::shared_ptr<GeneralArithmeticDemandAnalysis> generalArithmeticAnalysis;
    std::shared_ptr<ArithmeticPeriodicProgram> arithmeticPeriodicAnalysis;
    std::map<std::size_t, std::shared_ptr<const FiniteVisitAnalysis>> finiteVisitAnalyses;
};
} // namespace mlir::pto::frontiersynch
#endif
