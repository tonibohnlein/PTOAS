// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Session cache entries retain demands independently of capability failures.
// An instance belongs to one unchanged SyncInput, alias/hardware policy and
// original occurrence tree. initialize/invalidate destroys every entry when
// that context changes. Specialized parameter/phase views do not use these
// original-region keys. Their expression-free mathematical cache revalidates
// every observed constant binding. Guarded circuits have a separate private
// canonical arena; exports import transaction-local views and never lend it.
#ifndef PTO_FRONTIERSYNCH_ANALYSISSESSIONINTERNAL_H
#define PTO_FRONTIERSYNCH_ANALYSISSESSIONINTERNAL_H
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/AnalysisProfile.h"
#include "PTO/Transforms/FrontierSynch/Structured/GuardedRotatingAnalysis.h"
#include <set>
#include <tuple>
#include <chrono>
#include <array>
namespace mlir::pto::frontiersynch {
// Full request capability/mode key, shared by selection and cost recipes.
inline uint8_t analysisRequestKey(const AnalysisRequest& request)
{
    return unsigned(request.needs.queries) | (unsigned(request.needs.selectors) << 1) |
        (unsigned(request.needs.synchronization) << 2) | (unsigned(request.mode == AnalysisMode::Fallback) << 3) |
        (unsigned(request.needs.evaluation == AnalysisEvaluation::Stateful) << 4);
}
// Structural description only; capability-dependent estimates are computed
// separately. No relation, generated SSA or request-specific answer is cached.
struct ArithmeticPrimitiveShape {
    EstimatedCount pieces = 0, native = 0, rows = 0, domains = 0;
    uint64_t dimensions = 0;
};
struct NormalizedControlTree;
StringRef analysisBackendName(AnalysisBackend backend);
// Pure classification shared by execution and retained-work screening.
AnalysisExports mathematicalExports(const MathematicalResult& demands);
struct VaryingRegionalExports;
struct FiniteExpansionPlan;
struct NumericTemplatePlan;
struct NormalizedControlDescription;
struct ProducerFailure {
    AnalysisStatus status = AnalysisStatus::UnmetObligation;
    AnalysisStage stage = AnalysisStage::Demands;
    SmallVector<RecognitionDiagnostic> diagnostics;
    SmallVector<ArithmeticDiagnostic> arithmeticDiagnostics;
};
struct ArithmeticRegionalPreparation;
struct LogicalFamilyTemplate {
    int64_t planId = 0;
    bool completeInvocation = false, nestedIdentities = false;
    bool independentPieces = false, groupedFamilies = false;
    std::shared_ptr<const FamilyEndpointRecipe> recipe;
    std::vector<EndpointFamily> families;
};
struct BackendAttempt {
    ProducerFailure failure;
    bool produced = false;
    std::optional<bool> finiteArithmeticStorage;
    std::string finiteArithmeticStorageError;
    std::shared_ptr<const MathematicalResult> mathematical;
    std::string demandError;
    bool varyingQueriesAttempted = false, varyingSelectorsAttempted = false;
    std::shared_ptr<VaryingRegionalExports> varyingExports;
    std::shared_ptr<const RegionalAnalysis> varyingQueries, varyingSelectors;
    std::string varyingQueryError;
    std::shared_ptr<const RegionalAnalysis> boundedQueries, boundedSelectors;
    bool boundedQueriesAttempted = false, boundedSelectorsAttempted = false;
    std::string boundedQueryError, boundedSelectorError;
    std::shared_ptr<const RegionalAnalysis> arithmeticQueries, arithmeticSelectors;
    bool arithmeticQueriesAttempted = false, arithmeticSelectorsAttempted = false;
    std::string arithmeticQueryError, arithmeticSelectorError;
    std::shared_ptr<const RegionalAnalysis> expandedQueries, expandedSelectors;
    bool expandedSelectorsAttempted = false;
    std::string expandedSelectorError;
    std::optional<bool> endpoints;
    std::string prefixEmissionError;
    // The first successful synchronization request owns its detached fragment
    // until PrepareLogical consumes it. Later preparations instantiate afresh.
    std::unique_ptr<PreparedLogicalPlan> pendingLogical;
    // Successful scalar-free construction only. Fresh preparations share its
    // immutable recipes/evidence and never share mutable plans or generated SSA.
    std::shared_ptr<const LogicalFamilyTemplate> logicalFamilyTemplate;
    std::shared_ptr<const ArithmeticRegionalPreparation> arithmeticPreparation;
    std::shared_ptr<const NumericTemplateEndpoints> periodicEndpoints;
    // Only the allocation capability is memoized here, including unavailable exports.
    // Deferred exports bind evidence to the complete frozen protocol key.
    // Concrete compatibility plans retain the legacy empty-key contract.
    struct CachedAllocation {
        std::shared_ptr<const FamilyEndpointRecipe> protocolOwner;
        DictionaryAttr certificate;
    };
    std::map<std::pair<int64_t, std::vector<uint64_t>>, CachedAllocation> allocation;
    std::map<std::pair<std::size_t, std::size_t>, int64_t> arithmeticRecords;
};
struct SpecializedArithmeticAttempt {
    SmallVector<Operation*> roots;
    std::vector<std::tuple<unsigned, unsigned, uint64_t, uint64_t>> profiles;
    // Include failed constant queries: a previously unknown value becoming a
    // constant can change recognition even if it was absent from parameters.
    std::map<const void*, std::pair<Value, std::optional<int64_t>>> bindings;
    std::shared_ptr<const ArithmeticRegionalRelations> mathematics;
    std::string error;
};
struct SpecializedNumericAttempt {
    TemplateGeometryPolicy policy = TemplateGeometryPolicy::AllCertifiedMaps;
    std::shared_ptr<const NormalizedControlDescription> normalized;
    std::tuple<uint64_t, uint64_t, uint64_t, unsigned> limits;
    std::map<const void*, std::pair<Value, std::optional<int64_t>>> geometry;
    std::map<const void*, std::pair<Value, std::optional<bool>>> control;
    std::shared_ptr<const NumericBodyMathematics> mathematics;
    std::string error;
};
struct SpecializedGuardedAttempt {
    bool sliced = false;
    DenseMap<Value, bool> choices;
    SmallVector<Value> guards;
    DenseMap<Value, RegionExpressions::Id> bindings;
    std::shared_ptr<const GuardedRotatingMathematics> mathematics;
    std::string error;
};
template <typename Choice> struct RetainedSelectionOrder {
    std::vector<Choice> order;
    // Identity-only references into immutable session-owned qualifications.
    // A newly retained qualification invalidates a cold unknown estimate.
    std::vector<const void*> evidence;
};
enum class RetainedStage { Missing, Ready, Refused };
struct RetainedBackendWork {
    // Qualification/demands, queries, selectors, synchronization. A capability
    // not requested has zero work but its retained status remains visible.
    std::array<RetainedStage, 4> stages{};
    std::array<bool, 4> required{};
    std::array<EstimatedCount, 4> work;
    // Root arithmetic reducers share a separately retained generator stage.
    // Its work is additive, never inferred by subtracting a whole-route cost.
    std::optional<RetainedStage> generators;
    EstimatedCount generatorWork = 0;
    std::vector<const void*> evidence;
    EstimatedCount total;
    bool operator==(const RetainedBackendWork& other) const
    {
        return stages == other.stages && required == other.required && work == other.work &&
            generators == other.generators && generatorWork == other.generatorWork &&
            evidence == other.evidence && total == other.total;
    }
};
struct BackendSelectionRecipe {
    std::vector<AnalysisBackend> candidates, order;
    std::vector<RetainedBackendWork> evidence;
};
using ExhaustedRequestKey = std::tuple<std::size_t, uint8_t, bool, bool, std::vector<std::size_t>>;
struct ExhaustedRequestRecipe {
    uint64_t evidenceGeneration = 0;
    AnalysisOutcome outcome;
    AnalysisBackend backend = AnalysisBackend::Explicit;
};
struct AnalysisSessionState {
    // Context belongs to this unchanged input/profile/policy session. The final
    // active stack and continuation namespace prevent a refusal under one
    // recursive caller from becoming an unrestricted negative answer.
    uint64_t retainedEvidenceGeneration = 0;
    std::map<ExhaustedRequestKey, ExhaustedRequestRecipe> exhaustedRequests;
    void retainedEvidencePublished()
    {
        if (retainedEvidenceGeneration == UINT64_MAX) {
            exhaustedRequests.clear();
            retainedEvidenceGeneration = 0;
        } else { ++retainedEvidenceGeneration; }
    }
    ExhaustedRequestKey exhaustedKey(const AnalysisRequest& request, bool continuation = false) const
    {
        return {request.region, analysisRequestKey(request), certifying, continuation, activeRegions};
    }
    const ExhaustedRequestRecipe* exhausted(const ExhaustedRequestKey& key) const
    {
        auto found = exhaustedRequests.find(key);
        return found != exhaustedRequests.end() &&
            found->second.evidenceGeneration == retainedEvidenceGeneration ? &found->second : nullptr;
    }
    void retainExhausted(const ExhaustedRequestKey& key, const AnalysisOutcome& outcome,
        AnalysisBackend backend = AnalysisBackend::Explicit)
    {
        exhaustedRequests[key] = {retainedEvidenceGeneration, outcome, backend};
    }
    bool completionAttempted = false;
    std::shared_ptr<const CompletionRegion> completion;
    std::string completionError;
    // The diagnostic owner exists during initialization. Only an actual analysis
    // request freezes configurable profiles, matching the prior lazy session.
    bool requestsStarted = false;
    // Never passed to speculative exporters; canonical imported key IDs stay
    // in the committed prefix even when a producer attempt rolls back.
    std::shared_ptr<RegionExpressions> guardedExpressions = std::make_shared<RegionExpressions>();
    std::map<Operation*, std::vector<SpecializedGuardedAttempt>> specializedGuarded;
    uint64_t specializedGuardedBuilds = 0, guardedImportWork = 0;
    std::shared_ptr<void> guardedInputOwner;
    std::map<Operation*, std::vector<SpecializedNumericAttempt>> specializedNumeric;
    uint64_t specializedNumericBuilds = 0;
    std::map<Operation*, std::vector<SpecializedArithmeticAttempt>> specializedArithmetic;
    uint64_t specializedArithmeticBuilds = 0;
    std::optional<DifferenceArithmeticGeneratorStage> differenceGenerators;
    std::optional<GeneralArithmeticGeneratorStage> generalGenerators;
    uint64_t arithmeticGeneratorBuilds = 0;
    // Completed exact answers only. Audit selection has different native-scalar
    // admission rules. Mutable preparation/allocation plans are never cached here.
    std::map<std::tuple<std::size_t, uint8_t, bool>, AnalysisOutcome> selectedAnswers;
    // Same-node identity continuation has a restricted eligible suffix. Its
    // selections must never replace unrestricted answers for the same request.
    std::map<std::tuple<std::size_t, uint8_t, bool>,
        std::pair<AnalysisOutcome, AnalysisBackend>> identityAnswers;
    std::map<std::tuple<std::size_t, uint8_t, bool, uint64_t>, BackendSelectionRecipe> backendOrders;
    std::map<std::pair<std::size_t, uint8_t>, RetainedSelectionOrder<AnalysisBackend>> arithmeticOrders;
    // Certified root/regional forms are immutable and retained for this entire
    // session. The session owns the full input/profile/alias-policy context;
    // initialize/invalidate destroys these pointer keys before any reuse.
    mutable std::map<const ArithmeticProgram*, ArithmeticPrimitiveShape> primitiveShapes;
    mutable uint64_t primitiveShapeBuilds = 0, primitiveShapeRelationVisits = 0, primitiveShapePieceVisits = 0;
    std::shared_ptr<AnalysisLedger> ledger = std::make_shared<AnalysisLedger>();
    std::map<std::pair<std::size_t, uint8_t>, RetainedSelectionOrder<AnalysisForm>> formOrders;
    // Request-keyed estimates are execution state, independent of reports.
    std::map<std::pair<std::size_t, uint8_t>, AnalysisCostEstimate> expandedFormEstimates;
    std::set<std::pair<std::size_t, uint8_t>> numericalSequenceEstimates;
    std::map<std::pair<std::size_t, AnalysisForm>, uint64_t> formAttempts;
    std::map<std::size_t, std::shared_ptr<const NormalizedControlDescription>> normalizedInputs;
    std::shared_ptr<const NormalizedControlTree> originalControl;
    std::map<const NormalizedControlDescription*, std::shared_ptr<const FiniteExpansionPlan>> finiteExpansionPlans;
    std::map<const NormalizedControlDescription*, BackendAttempt> expandedAttempts;
    std::map<std::pair<std::size_t, bool>, std::shared_ptr<const NumericTemplatePlan>> numericTemplatePlans;
    uint64_t numericalRegionBuilds = 0;
    std::map<std::size_t, std::map<AnalysisBackend, BackendAttempt>> attempts;
    // Canonical original-loop construction is independent of root/regional exports.
    std::map<std::pair<std::size_t, AnalysisBackend>, BackendAttempt> loopAttempts;
    // Some regional producers return fresh exact mathematical views. Keep each
    // tagged view alive until consumers lock their weak identity; no rebuilding.
    std::map<const MathematicalResult*, std::shared_ptr<const MathematicalResult>> regionalMathematics;
    // Nonrecursive arithmetic leaf construction is shared by a sequence's
    // local adapter and subsequent standalone requests for the same region.
    std::map<std::size_t, BackendAttempt> arithmeticRegionAttempts;
    uint64_t arithmeticRegionBuilds = 0;
    std::shared_ptr<RegionExpressions> expressions = std::make_shared<RegionExpressions>();
    std::set<AnalysisBackend> wholeRegionEvidence;
    std::vector<std::size_t> activeRegions;
    std::optional<std::vector<RegionCertification>> certifications;
    bool certifying = false;
};
// Observe only immutable retained outputs, never a mutable detached plan. This
// scope also covers exports requested directly, outside ordinary selection.
class RetainedAttemptPublication {
public:
    RetainedAttemptPublication(AnalysisSessionState& session, const BackendAttempt& attempt)
        : session(session), attempt(attempt), before(outputs(attempt)), endpoints(attempt.endpoints.value_or(false)),
          finiteStorage(attempt.finiteArithmeticStorage.value_or(false)) {}
    ~RetainedAttemptPublication()
    {
        const auto after = outputs(attempt);
        bool changed = (!endpoints && attempt.endpoints.value_or(false)) ||
            (!finiteStorage && attempt.finiteArithmeticStorage.value_or(false));
        for (std::size_t i = 0; i < after.size(); ++i) {
            changed |= after[i] && after[i] != before[i];
        }
        if (changed) { session.retainedEvidencePublished(); }
    }
    RetainedAttemptPublication(const RetainedAttemptPublication&) = delete;
    RetainedAttemptPublication& operator=(const RetainedAttemptPublication&) = delete;
private:
    static std::array<const void*, 13> outputs(const BackendAttempt& attempt)
    {
        return {attempt.mathematical.get(), attempt.varyingExports.get(), attempt.varyingQueries.get(),
            attempt.varyingSelectors.get(), attempt.boundedQueries.get(), attempt.boundedSelectors.get(),
            attempt.arithmeticQueries.get(), attempt.arithmeticSelectors.get(), attempt.expandedQueries.get(),
            attempt.expandedSelectors.get(), attempt.logicalFamilyTemplate.get(),
            attempt.arithmeticPreparation.get(), attempt.periodicEndpoints.get()};
    }
    AnalysisSessionState& session;
    const BackendAttempt& attempt;
    std::array<const void*, 13> before;
    bool endpoints, finiteStorage;
};
RetainedBackendWork previewRetainedBackendWork(const AnalysisSessionState& session,
    const ProgramRecognition& program, std::size_t region, std::optional<std::size_t> canonical,
    AnalysisBackend backend, const AnalysisNeeds& needs, const ArithmeticProgram* arithmetic = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
