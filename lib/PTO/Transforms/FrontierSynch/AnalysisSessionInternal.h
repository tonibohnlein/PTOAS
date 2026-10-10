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
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include <set>
namespace mlir::pto::frontiersynch {
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
struct BackendAttempt {
    ProducerFailure failure;
    bool produced = false;
    std::optional<bool> finiteArithmeticStorage;
    std::string finiteArithmeticStorageError;
    std::shared_ptr<const MathematicalResult> mathematical;
    std::string demandError;
    bool varyingQueriesAttempted = false, varyingSelectorsAttempted = false;
    std::shared_ptr<VaryingRegionalExports> varyingExports;
    std::string varyingQueryError;
    std::shared_ptr<const RegionalAnalysis> arithmeticQueries, arithmeticSelectors;
    bool arithmeticQueriesAttempted = false, arithmeticSelectorsAttempted = false;
    std::string arithmeticQueryError, arithmeticSelectorError;
    std::shared_ptr<const RegionalAnalysis> expandedQueries, expandedSelectors;
    bool expandedSelectorsAttempted = false;
    std::string expandedSelectorError;
    std::optional<bool> endpoints;
    // The first successful synchronization request owns its detached fragment
    // until PrepareLogical consumes it. Later preparations instantiate afresh.
    std::unique_ptr<PreparedLogicalPlan> pendingLogical;
    std::shared_ptr<const NumericTemplateEndpoints> periodicEndpoints;
    // Only the allocation capability is memoized here, including unavailable exports.
    std::map<int64_t, DictionaryAttr> allocation;
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
struct AnalysisSessionState {
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
    uint64_t arithmeticGeneratorBuilds = 0;
    std::map<std::pair<std::size_t, uint8_t>, std::vector<AnalysisBackend>> arithmeticOrders;
    std::vector<AnalysisCostRecord> costs;
    std::map<std::pair<std::size_t, uint8_t>, std::vector<AnalysisForm>> formOrders;
    std::map<std::pair<std::size_t, AnalysisForm>, uint64_t> formAttempts;
    std::map<std::size_t, std::shared_ptr<const NormalizedControlDescription>> normalizedInputs;
    std::map<const NormalizedControlDescription*, std::shared_ptr<const FiniteExpansionPlan>> finiteExpansionPlans;
    std::map<const NormalizedControlDescription*, BackendAttempt> expandedAttempts;
    std::map<std::pair<std::size_t, bool>, std::shared_ptr<const NumericTemplatePlan>> numericTemplatePlans;
    uint64_t numericalRegionBuilds = 0;
    std::map<std::size_t, std::map<AnalysisBackend, BackendAttempt>> attempts;
    // Canonical original-loop construction is independent of root/regional exports.
    std::map<std::pair<std::size_t, AnalysisBackend>, BackendAttempt> loopAttempts;
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
} // namespace mlir::pto::frontiersynch
#endif
