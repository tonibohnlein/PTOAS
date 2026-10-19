// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Per-function analysis and detached synchronization preparation. The public
// logical pass uses the transactional module coordinator; physical event IDs
// are assigned by the following allocation pass. Analysis-only mode retains
// the mathematical certification checkpoint without changing the original IR.
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
#include "AnalysisSessionInternal.h"
#include "PTO/Transforms/FrontierSynch/ResultGuarantees.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/Shared/ClosedCallees.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/Structured/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Structured/GuardedPeriodicInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompactAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Shared/FiniteVisitRecognition.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Shared/RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
namespace mlir::pto::frontiersynch {
namespace {
// Whole-invocation shortcut only: every payload uses the protected scalar
// pipe, and every other effect is accounted for by the shared leaf contract.
// No storage geometry, visit enumeration or regional selector is required.
bool nativeScalarRequirements(func::FuncOp function, const SyncInput& input, const PhaseIndex& index)
{
    auto scalar = ptoStorageProtection().scalarPipe;
    if (!scalar || function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) ||
        llvm::any_of(input.instructions(), [&](const auto* phase) {
            return static_cast<uint32_t>(phase->kPipeValue) != *scalar ||
                llvm::any_of(phase->elementOp->getResults(), [&](Value value) {
                    return resultAvailability(*phase, value) == SyncResultAvailability::RequiresCompletion;
                });
        })) { return false; }
    RecognitionResult checked;
    function.walk([&](Operation* op) {
        if (op == function.getOperation()) { return; }
        if (llvm::any_of(index.prerequisitesFor(op), [](const auto& prerequisite) {
                return !prerequisite.native;
            })) {
            checked.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (op->getNumRegions()) {
            if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
                checked.note(RecognitionIssue::UnsupportedControl, op, true);
            }
            return;
        }
        detail::inspectLeaf(*op, index, checked);
    });
    return checked.state == RecognitionState::Applicable;
}
} // namespace
LogicalResult FrontierAnalysis::initialize(GMAliasPolicy requestedPolicy, bool requireStructure) {
    if (initialized && policy == requestedPolicy) {
        if (!storage) { return failure(); }
        return nativeScalarOnly && !requireStructure ? success() : recognizeStructure();
    }
    invalidate();
    construction = {};
    sessionState = std::make_shared<AnalysisSessionState>();
    sessionState->ledger->profiling = profileStages;
    AnalysisProfileScope profile(sessionState->ledger, "initialize", 0, {}, true);
    initialized = true;
    policy = requestedPolicy;
    if (!function) {
        return failure();
    }
    auto pending = std::make_shared<SyncInput>(policy);
    {
        AnalysisProfileScope inputProfile(sessionState->ledger, "shared-input");
        inputProfile.constructed();
        if (failed(pending->build(function, SyncInstructionView::PipeEnvelopes))) { return failure(); }
    }
    structuralIndex = std::make_shared<PhaseIndex>();
    ++construction.structuralIndices;
    {
        AnalysisProfileScope indexProfile(sessionState->ledger, "structural-index");
        indexProfile.constructed();
        if (failed(structuralIndex->build(function, *pending))) {
            structuralIndex.reset();
            return failure();
        }
    }
    nativeScalarOnly = nativeScalarRequirements(function, *pending, *structuralIndex);
    storage = std::move(pending);
    return nativeScalarOnly && !requireStructure ? success() : recognizeStructure();
}
LogicalResult FrontierAnalysis::recognizeStructure() {
    AnalysisProfileScope profile(sessionState ? sessionState->ledger : nullptr, "structural-forms");
    if (program) { profile.cacheHit(); return success(); }
    if (!storage) { return failure(); }
    profile.constructed();
    auto recognized = recognizeProgram(function, *storage, *structuralIndex, ProgramRecognitionMode::Inventory);
    if (failed(recognized)) { return failure(); }
    program = std::make_shared<ProgramRecognition>(std::move(*recognized));
    program->allocationPolicy = arithmeticAllocationPolicy();
    program->deferContractAudit = true;
    program->regionalArithmeticProfiles = regionalArithmeticProfiles;
    return success();
}
void FrontierAnalysis::recognizeForm(std::size_t region, ProgramForm form)
{
    if (!program || region >= program->nodes.size() ||
        static_cast<unsigned>(form) > static_cast<unsigned>(ProgramForm::FiniteVisit)) { return; }
    AnalysisProfileScope profile(sessionState ? sessionState->ledger : nullptr, "form-check", region);
    const auto bit = static_cast<uint8_t>(1U << static_cast<unsigned>(form));
    if (program->nodes[region].attemptedForms & bit) { profile.cacheHit(); return; }
    profile.constructed();
    materializeProgramForm(*program, region, form, *structuralIndex, *storage);
    if (sessionState) { sessionState->retainedEvidencePublished(); }
}
LogicalResult FrontierAnalysis::recognizeAllForms()
{
    if (failed(recognizeStructure())) { return failure(); }
    AnalysisProfileScope profile(sessionState ? sessionState->ledger : nullptr, "form-audit");
    std::vector<uint8_t> attempted;
    for (const auto& node : program->nodes) { attempted.push_back(node.attemptedForms); }
    materializeAllProgramForms(*program, *structuralIndex, *storage);
    if (sessionState && llvm::any_of(llvm::enumerate(program->nodes), [&](auto node) {
            return attempted[node.index()] != node.value().attemptedForms;
        })) { sessionState->retainedEvidencePublished(); }
    return success();
}
LogicalResult FrontierAnalysis::auditNumericCandidates() {
    if (failed(recognizeStructure())) { return failure(); }
    const auto& index = *structuralIndex;
    for (auto [id, node] : llvm::enumerate(program->nodes)) {
        if (node.kind == StructureKind::Loop) { recognizeForm(id, ProgramForm::Varying); }
        if (!node.unsupportedContext && node.kind == StructureKind::Loop && node.payloadCount) {
            (void)materializeNumericRegion(id);
        }
        if (node.varyingRotating && !node.varyingDemands &&
            node.varyingRotating->result.state == RecognitionState::Applicable) {
            if (construction.varyingBoundaryReductions != UINT64_MAX) { ++construction.varyingBoundaryReductions; }
            node.varyingDemands = analyzeVaryingRotating(*node.varyingRotating, index, *storage);
            sessionState->retainedEvidencePublished();
        }
        if (!node.numericTemplate || node.periodicAnalysis ||
            node.numericTemplate->result.state != RecognitionState::Applicable) { continue; }
        node.periodicAnalysis = analyzeNumericTemplate(*node.numericTemplate);
        sessionState->retainedEvidencePublished();
        const auto& numeric = *node.numericTemplate;
        const bool whole = !node.unsupportedContext && !numeric.specializedBody && numeric.outer &&
            numeric.outer->getParentOp() == function.operator->() &&
            llvm::all_of(program->payloads, [&](const auto& payload) {
                return numeric.outer->isProperAncestor(payload.phase->elementOp);
            });
        if (whole && node.periodicAnalysis->error.empty()) { recordWholeRegion(AnalysisBackend::NumericalPeriodic); }
    }
    refreshProgramContractAudit(*program);
    return success();
}
LogicalResult FrontierAnalysis::auditNumericCandidateExports() {
    if (failed(auditNumericCandidates())) { return failure(); }
    for (auto& node : program->nodes) {
        if (!node.numericTemplate || !node.periodicAnalysis || !node.periodicAnalysis->error.empty()) { continue; }
        if (!node.logicalEndpoints) {
            node.logicalEndpoints = buildNumericTemplateEndpoints(*node.numericTemplate, *node.periodicAnalysis);
            sessionState->retainedEvidencePublished();
        }
        const bool endpointsAvailable = node.logicalEndpoints->logical.error.empty();
        if (!endpointsAvailable || node.periodicAllocation) { continue; }
        node.periodicAllocation = buildPeriodicAllocation(*node.periodicAnalysis);
        sessionState->retainedEvidencePublished();
    }
    refreshProgramContractAudit(*program);
    return success();
}
LogicalResult FrontierAnalysis::produceExplicitFunction() {
    if (!storage || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        return failure();
    }
    if (!finiteInput) { finiteInput = finiteFunctionInput(function, storage, structuralIndex); }
    if (!concreteFinite) {
        ++construction.explicitReductions;
        concreteFinite = analyzeFinite(finiteInput, FiniteStrategy::Concrete);
    }
    if (!concreteFinite) { return failure(); }
    if (concreteFinite->error().empty()) { recordWholeRegion(AnalysisBackend::Explicit); }
    return success(concreteFinite->error().empty());
}
SequenceAnalysis* FrontierAnalysis::produceSequenceFunction() {
    if (failed(recognizeStructure())) { return nullptr; }
    if (!sequenceAnalysis) {
        if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
        sessionState->requestsStarted = true;
        sequenceAnalysis = std::make_shared<SequenceAnalysis>(analyzeSequenceRegionWithResolver(
            function, *storage, *program, 0, sessionState->expressions, structuralIndex, false, regionalResolver()));
        if (sequenceAnalysis->error.empty()) { sessionState->retainedEvidencePublished(); }
        recordSequenceContractAttempt(*program, *storage, *sequenceAnalysis);
        if (program->sequenceContract && program->sequenceContract->membership == ContractStatus::Established &&
            program->sequenceContract->demands == ContractImplementation::Available) {
            recordWholeRegion(AnalysisBackend::Sequence);
        }
        refreshProgramContractAudit(*program);
    }
    return &*sequenceAnalysis;
}
LogicalResult FrontierAnalysis::auditFiniteVisitCandidates()
{
    if (failed(recognizeStructure())) { return failure(); }
    for (std::size_t id = 0; id < program->nodes.size(); ++id) {
        if (program->nodes[id].kind == StructureKind::Loop) { recognizeForm(id, ProgramForm::FiniteVisit); }
    }
    for (const auto& candidate : program->finiteVisitContracts) {
        if (!candidate.node || finiteVisitAnalyses.count(*candidate.node)) { continue; }
        const auto eligible = llvm::find_if(candidate.obligations, [](const auto& item) {
            return item.name == "exhaustive-original-type-selection" && item.status == ContractStatus::Established;
        });
        if (eligible == candidate.obligations.end()) { continue; }
        auto result = std::make_shared<FiniteVisitAnalysis>(
            analyzeFiniteVisitLoop(function, *storage, *program, *candidate.node, structuralIndex,
                [this](std::size_t id, ProgramForm form) { recognizeForm(id, form); }));
        finiteVisitAnalyses.emplace(*candidate.node, std::move(result));
        sessionState->retainedEvidencePublished();
    }
    for (auto& candidate : program->finiteVisitContracts) {
        if (!candidate.node) { continue; }
        auto found = finiteVisitAnalyses.find(*candidate.node);
        if (found != finiteVisitAnalyses.end()) { candidate = found->second->recognition.contract; }
    }
    refreshProgramContractAudit(*program);
    return success();
}
uint64_t FrontierAnalysis::arithmeticRegionConstructions() const
{
    return sessionState ? sessionState->arithmeticRegionBuilds : 0;
}
uint64_t FrontierAnalysis::arithmeticGeneratorConstructions() const
{
    return sessionState ? sessionState->arithmeticGeneratorBuilds : 0;
}
LogicalResult FrontierAnalysis::ensureArithmeticGenerators()
{
    const bool recognized = succeeded(recognizeArithmetic()) && program->arithmetic.has_value();
    if (!recognized) { return failure(); }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    sessionState->requestsStarted = true;
    const auto& arithmetic = *program->arithmetic;
    if (arithmetic.recognition.arithmeticClass == ArithmeticClass::Differences) {
        if (!sessionState->differenceGenerators) {
            const auto protection = structuredProtection(storage->accesses());
            sessionState->differenceGenerators.emplace(analyzeDifferenceArithmeticGenerators(arithmetic, &protection));
            sessionState->retainedEvidencePublished();
            if (sessionState->arithmeticGeneratorBuilds != UINT64_MAX) { ++sessionState->arithmeticGeneratorBuilds; }
        }
        return success();
    }
    if (!sessionState->generalGenerators) {
        const auto protection = structuredProtection(storage->accesses());
        sessionState->generalGenerators.emplace(analyzeGeneralArithmeticGenerators(arithmetic, &protection));
        sessionState->retainedEvidencePublished();
        if (sessionState->arithmeticGeneratorBuilds != UINT64_MAX) { ++sessionState->arithmeticGeneratorBuilds; }
    }
    return success();
}
LogicalResult FrontierAnalysis::produceArithmeticPeriodicFunction()
{
    if (failed(recognizeArithmetic()) || !program->arithmetic) { return failure(); }
    if (!arithmeticPeriodicAnalysis) {
        std::string diagnostic;
        if (!checkArithmeticPeriodicSkeleton(*program->arithmetic, diagnostic)) {
            arithmeticPeriodicAnalysis = std::make_shared<ArithmeticPeriodicProgram>();
            arithmeticPeriodicAnalysis->conversion.status = ArithmeticPeriodicStatus::AdapterUnavailable;
            arithmeticPeriodicAnalysis->conversion.diagnostic = std::move(diagnostic);
            return failure();
        }
        if (failed(ensureArithmeticGenerators())) { return failure(); }
        auto converted = sessionState->differenceGenerators ?
            convertArithmeticPeriodicProgram(*program->arithmetic, *sessionState->differenceGenerators, {}, false) :
            convertArithmeticPeriodicProgram(*program->arithmetic, *sessionState->generalGenerators, {}, false);
        arithmeticPeriodicAnalysis = std::make_shared<ArithmeticPeriodicProgram>(std::move(converted));
    }
    const auto& conversion = arithmeticPeriodicAnalysis->conversion;
    const bool exact = conversion.hasExactDemands();
    if (exact) { recordWholeRegion(AnalysisBackend::ArithmeticPeriodic); }
    return success(exact);
}
LogicalResult FrontierAnalysis::produceArithmeticFunction()
{
    if (failed(ensureArithmeticGenerators())) { return failure(); }
    if (sessionState->differenceGenerators) {
        arithmeticAnalysis = reduceDifferenceArithmeticDemands(*sessionState->differenceGenerators);
        const bool exact = arithmeticAnalysis && arithmeticAnalysis->error.empty() && arithmeticAnalysis->exactMinimum;
        if (exact) { recordWholeRegion(AnalysisBackend::Arithmetic); }
        return success(exact);
    }
    generalArithmeticAnalysis = reduceGeneralArithmeticDemands(*sessionState->generalGenerators);
    const bool exact = generalArithmeticAnalysis && generalArithmeticAnalysis->error.empty() &&
                       generalArithmeticAnalysis->exactMinimum;
    if (exact) { recordWholeRegion(AnalysisBackend::Arithmetic); }
    return success(exact);
}
bool FrontierAnalysis::hasWholeFunctionMinimumDemands() const
{
    return sessionState && !sessionState->wholeRegionEvidence.empty();
}
void FrontierAnalysis::noteSequenceEndpointOutcome(StringRef error) {
    if (!program) { return; }
    recordSequenceEndpointAttempt(*program, error);
    refreshProgramContractAudit(*program);
}
LogicalResult FrontierAnalysis::recognizeArithmetic() {
    if (failed(recognizeStructure())) { return failure(); }
    AnalysisProfileScope profile(sessionState ? sessionState->ledger : nullptr, "arithmetic-form");
    if (program->arithmetic || function.isDeclaration()) {
        profile.cacheHit();
        return success();
    }
    const auto& index = *structuralIndex;
    // Profiles are alternative representations of the same arithmetic family.
    // Later profiles remain unevaluated once an exact form is available.
    std::optional<ArithmeticProgram> selected;
    for (const auto& limits : wholeArithmeticProfiles) {
        profile.constructed();
        auto candidate = recognizeArithmeticProgram(function, index, *storage, storage->accesses(), limits);
        recordArithmeticContractAttempt(*program, limits, candidate);
        selected = std::move(candidate);
        if (selected->extraction.state == RecognitionState::Applicable &&
            selected->recognition.state == RecognitionState::Applicable) { break; }
    }
    program->arithmetic = std::move(selected);
    if (sessionState && program->arithmetic) { sessionState->retainedEvidencePublished(); }
    refreshProgramContractAudit(*program);
    return success();
}
LogicalResult FrontierAnalysis::configureArithmeticProfiles(ArrayRef<ArithmeticLimits> whole,
                                                           ArrayRef<ArithmeticLimits> regional)
{
    const bool requested = (program && program->arithmetic) || !arithmeticForms.empty() ||
        (sessionState && sessionState->requestsStarted) || sequenceAnalysis || !finiteVisitAnalyses.empty();
    const auto valid = [](const ArithmeticLimits& limits) {
        return limits.pipes && limits.dimensions && limits.period && limits.coefficient &&
            limits.period <= static_cast<uint64_t>(INT64_MAX);
    };
    const bool invalid = requested || whole.empty() || regional.empty() ||
        !llvm::all_of(whole, valid) || !llvm::all_of(regional, valid);
    if (invalid) { return failure(); }
    wholeArithmeticProfiles.assign(whole.begin(), whole.end());
    regionalArithmeticProfiles.assign(regional.begin(), regional.end());
    if (program) { program->regionalArithmeticProfiles = regionalArithmeticProfiles; }
    return success();
}
const ArithmeticProgram* FrontierAnalysis::recognizeArithmeticRegion(std::size_t region)
{
    if (!program || region >= program->nodes.size()) { return nullptr; }
    AnalysisProfileScope profile(sessionState ? sessionState->ledger : nullptr, "arithmetic-form", region);
    auto found = arithmeticForms.find(region);
    if (found != arithmeticForms.end()) { profile.cacheHit(); return found->second.get(); }
    const auto& node = program->nodes[region];
    const bool supported = node.kind == StructureKind::Loop || node.kind == StructureKind::Conditional;
    if (!supported || !node.anchor) { return nullptr; }
    std::shared_ptr<const ArithmeticProgram> selected;
    for (const auto& limits : regionalArithmeticProfiles) {
        profile.constructed();
        auto candidate = std::make_shared<ArithmeticProgram>(recognizeArithmeticProgram(
            {function, node.anchor}, *structuralIndex, *storage, storage->accesses(), limits));
        recordArithmeticContractAttempt(*program, limits, *candidate, region);
        selected = std::move(candidate);
        if (selected->extraction.state == RecognitionState::Applicable &&
            selected->recognition.state == RecognitionState::Applicable) { break; }
    }
    auto stored = arithmeticForms.emplace(region, std::move(selected));
    if (sessionState) { sessionState->retainedEvidencePublished(); }
    refreshProgramContractAudit(*program);
    return stored.first->second.get();
}
LogicalResult FrontierAnalysis::recognizeRegionalArithmetic()
{
    if (failed(recognizeStructure())) { return failure(); }
    for (std::size_t id = 1; id < program->nodes.size(); ++id) {
        (void)recognizeArithmeticRegion(id);
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
namespace mlir::pto {
#define GEN_PASS_DEF_PTOFRONTIERANALYSIS
#include "PTO/Transforms/Passes.h.inc"
namespace {
// Persist recognition independently of which logical backend succeeds. The
// report contains no borrowed operations or values and survives insertion.
DictionaryAttr contractReport(const frontiersynch::ProgramRecognition& program, StringRef logicalBackend,
                              uint64_t arithmeticGeneratorConstructions, uint64_t arithmeticRegionConstructions,
                              uint64_t specializedArithmeticConstructions,
                              ArrayRef<frontiersynch::AnalysisCostRecord> costs,
                              const frontiersynch::AnalysisConstructionCounts& counts)
{
    MLIRContext* context = program.nodes.front().anchor->getContext();
    Builder b(context);
    SmallVector<Attribute> candidates;
    for (const auto& candidate : program.contractAudit) {
        NamedAttrList entry;
        entry.set("class", b.getStringAttr(frontiersynch::contractName(candidate.kind)));
        entry.set("membership", b.getStringAttr(frontiersynch::contractName(candidate.membership)));
        entry.set("node", b.getI64IntegerAttr(candidate.node ? static_cast<int64_t>(*candidate.node) : -1));
        entry.set("demands", b.getStringAttr(frontiersynch::contractName(candidate.demands)));
        entry.set("endpoint_recipes", b.getStringAttr(frontiersynch::contractName(candidate.endpointRecipes)));
        entry.set("allocation_analysis", b.getStringAttr(frontiersynch::contractName(candidate.allocation)));
        entry.set("implementation_error", b.getStringAttr(candidate.implementationError));
        SmallVector<Attribute> reasons;
        for (const auto& diagnostic : candidate.diagnostics) {
            NamedAttrList reason;
            reason.set("criterion", b.getStringAttr(frontiersynch::recognitionName(diagnostic.issue)));
            reason.set("category", b.getStringAttr(frontiersynch::contractName(
                frontiersynch::contractDiagnosticKind(diagnostic))));
            if (diagnostic.anchor) {
                reason.set("operation", b.getStringAttr(diagnostic.anchor->getName().getStringRef()));
                reason.set("location", diagnostic.anchor->getLoc());
            }
            reasons.push_back(reason.getDictionary(context));
        }
        for (const auto& diagnostic : candidate.arithmeticDiagnostics) {
            NamedAttrList reason;
            reason.set("criterion", b.getStringAttr(frontiersynch::recognitionName(diagnostic.issue)));
            const auto category = diagnostic.issue == frontiersynch::ArithmeticIssue::InvalidConfiguration ?
                "producer-limit" : (diagnostic.outsideClass ? "candidate-criterion-violation" : "unmet-obligation");
            reason.set("category", b.getStringAttr(category));
            reason.set("relation", b.getI64IntegerAttr(diagnostic.relation));
            reason.set("piece", b.getI64IntegerAttr(diagnostic.piece));
            reason.set("count", b.getI64IntegerAttr(diagnostic.count));
            reason.set("witness", b.getStringAttr("first"));
            reasons.push_back(reason.getDictionary(context));
        }
        for (const auto& obligation : candidate.obligations) {
            NamedAttrList reason;
            reason.set("criterion", b.getStringAttr(obligation.name));
            reason.set("status", b.getStringAttr(frontiersynch::contractName(obligation.status)));
            reasons.push_back(reason.getDictionary(context));
        }
        entry.set("criteria", b.getArrayAttr(reasons));
        if (candidate.arithmeticProfile) {
            const auto& limits = *candidate.arithmeticProfile;
            entry.set("period", b.getI64IntegerAttr(limits.period));
            entry.set("pipes", b.getI64IntegerAttr(limits.pipes));
            entry.set("dimensions", b.getI64IntegerAttr(limits.dimensions));
            entry.set("coefficient", b.getI64IntegerAttr(limits.coefficient));
        }
        candidates.push_back(entry.getDictionary(context));
    }
    NamedAttrList report;
    report.set("version", b.getI64IntegerAttr(1));
    report.set("stage", b.getStringAttr("recognition-before-insertion"));
    report.set("contracts", b.getArrayAttr(candidates));
    report.set("logical_plan", b.getStringAttr("ready"));
    report.set("selected_logical_backend", b.getStringAttr(logicalBackend));
    report.set("arithmetic_generator_constructions", b.getI64IntegerAttr(arithmeticGeneratorConstructions));
    report.set("arithmetic_region_constructions", b.getI64IntegerAttr(arithmeticRegionConstructions));
    report.set("specialized_arithmetic_constructions", b.getI64IntegerAttr(specializedArithmeticConstructions));
    SmallVector<Attribute> estimates;
    for (const auto& record : costs) {
        NamedAttrList entry;
        entry.set("method", b.getStringAttr(record.method));
        entry.set("region", b.getI64IntegerAttr(record.region));
        entry.set("request", b.getI64IntegerAttr(record.request));
        entry.set("attempt_constructions", b.getI64IntegerAttr(record.attemptConstructions));
        auto count = [&](StringRef name, frontiersynch::EstimatedCount value) {
            Attribute encoded = value ? Attribute(IntegerAttr::get(
                IntegerType::get(context, 64, IntegerType::Unsigned), APInt(64, *value))) :
                Attribute(b.getStringAttr("unknown"));
            entry.set(name, encoded);
        };
        count("work", record.estimate.work); count("representation", record.estimate.representation);
        count("generator_pieces", record.estimate.generatorPieces); count("ports", record.estimate.ports);
        count("numerical_window", record.estimate.numericalWindow);
        count("circuit_nodes", record.estimate.circuitNodes);
        count("relation_conversion", record.estimate.relationConversion);
        estimates.push_back(entry.getDictionary(context));
    }
    report.set("cost_estimates", b.getArrayAttr(estimates));
    report.set("mathematical_attempts", b.getI64IntegerAttr(counts.mathematicalAttempts));
    report.set("request_walks", b.getI64IntegerAttr(counts.requestWalks));
    report.set("exhausted_walk_hits", b.getI64IntegerAttr(counts.exhaustedWalkHits));
    report.set("logical_preparations", b.getI64IntegerAttr(counts.logicalPreparations));
    report.set("allocation_exports", b.getI64IntegerAttr(counts.allocationExports));
    report.set("physical_allocation", b.getStringAttr("not-requested"));
    return report.getDictionary(context);
}
// Reports on every exit, including a failed preparation. It never attaches IR.
struct StageProfileReport {
    frontiersynch::FrontierAnalysis& analysis;
    StringRef workload;
    ~StageProfileReport() { analysis.writeStageProfile(llvm::errs(), workload); }
};
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepareFunction(
    func::FuncOp function, GMAliasPolicy policy,
    const std::optional<frontiersynch::ArithmeticLimits>& profile = std::nullopt, bool profileStages = false,
    bool prepareAllocation = true, ArrayRef<int64_t> allocationIds = {}, bool completeInvocation = false,
    bool physicalOutput = false, bool completionOnly = false, bool deferredOutput = false,
    frontiersynch::ArithmeticRegionalAllocationPolicy allocationPolicy =
        frontiersynch::ArithmeticRegionalAllocationPolicy::SingleId)
{
    if (deferredOutput && completeInvocation) {
        return function.emitError("deferred-output does not export closed child contracts"), failure();
    }
    if (completionOnly) {
        std::string error;
        auto completion = frontiersynch::CompletionRegion::buildCoarse(function, policy, error);
        if (!completion) {
            function.emitError(error);
            return failure();
        }
        auto plan = std::make_unique<frontiersynch::PreparedLogicalPlan>(0);
        plan->completeInvocation = true;
        plan->completionOwner = std::move(completion);
        frontiersynch::attachResultGuarantees(*plan, function.getContext());
        return plan;
    }
    frontiersynch::FrontierAnalysis analysis(function, allocationPolicy);
    analysis.enableStageProfiling(profileStages);
    StageProfileReport profileReport{analysis, "production-root"};
    if (failed(analysis.initialize(policy))) { return failure(); }
    if (profile && failed(analysis.configureArithmeticProfiles(
            ArrayRef<frontiersynch::ArithmeticLimits>(&*profile, 1),
            ArrayRef<frontiersynch::ArithmeticLimits>(&*profile, 1)))) {
        function.emitError("frontier arithmetic profile context could not be established");
        return failure();
    }
    frontiersynch::AnalysisRequest request;
    request.mode = frontiersynch::AnalysisMode::Fallback;
    request.needs.synchronization = true;
    auto result = analysis.analyze(request);
    FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepared = failure();
    StringRef logicalBackend;
    if (result.status == frontiersynch::AnalysisStatus::Ready) {
        prepared = analysis.prepareLogical(result);
        if (succeeded(prepared)) {
            (*prepared)->completeInvocation |= completeInvocation;
            logicalBackend = result.completion ? StringRef("completion-transfer") :
                StringRef(result.mathematical->backend);
            // Missing allocation support leaves the selected logical graph available.
            if (prepareAllocation) { (void)analysis.attachAllocation(result, **prepared, allocationIds); }
            (*prepared)->deferredOutput = deferredOutput;
            if (physicalOutput &&
                failed(frontiersynch::planPhysicalFamilyOutput(function, **prepared, allocationIds))) {
                return function.emitError(
                           "combined physical output requires a supported retained family and allocation export"),
                       failure();
            }
        }
    }
    std::string routeError;
    for (const auto& obligation : result.obligations) {
        if (!obligation.diagnostic.empty()) { routeError += "; " + obligation.diagnostic; }
    }
    if (failed(prepared) && analysis.hasWholeFunctionMinimumDemands()) {
        routeError = "unmet-exports: exact whole-function demands retained; " + routeError;
    }
    if (succeeded(prepared)) {
        (*prepared)->profileControl = profileStages;
        profileLogicalControl(function, **prepared, "prepared");
        (*prepared)->recognitionReport = contractReport(*analysis.result(), logicalBackend,
            analysis.arithmeticGeneratorConstructions(), analysis.arithmeticRegionConstructions(),
            analysis.specializedArithmeticConstructions(),
            analysis.costRecords(), analysis.constructionCounts());
        NamedAttrList allocationReport((*prepared)->recognitionReport);
        allocationReport.set("logical_realization", StringAttr::get(function.getContext(),
            (*prepared)->completionOwner ? "conservative-completion" :
                ((*prepared)->privateState ? "invocation-prefix" : "direct-endpoints")));
        allocationReport.set("allocation_support", StringAttr::get(function.getContext(),
            !prepareAllocation ? "not-requested" :
                ((*prepared)->completionOwner ?
                    ((*prepared)->completionOwner->eventfulInlineChildren() ? "closed-child-palettes" : "not-needed") :
                    ((*prepared)->allocationCertificate ? "available" : "unavailable"))));
        (*prepared)->recognitionReport = allocationReport.getDictionary(function.getContext());
        frontiersynch::attachResultGuarantees(**prepared, function.getContext());
        if (result.mathematical && result.mathematical->arithmeticPeriodicDemands) {
            const auto& conversion = result.mathematical->arithmeticPeriodicDemands->conversion;
            NamedAttrList report((*prepared)->recognitionReport);
            report.set("periodic_reducer", StringAttr::get(function.getContext(),
                conversion.numerical ? "numerical" : "guarded"));
            (*prepared)->recognitionReport = report.getDictionary(function.getContext());
        }
    }
    if (failed(prepared)) {
        auto diagnostic = function.emitError("logical plan unavailable; Section 5 contract outcomes:");
        for (const auto& candidate : analysis.result()->contractAudit) {
            diagnostic << " " << frontiersynch::contractName(candidate.kind) << "[";
            if (candidate.node) { diagnostic << *candidate.node; }
            else {
                diagnostic << "function";
                if (candidate.arithmeticProfile) { diagnostic << ",period=" << candidate.arithmeticProfile->period; }
            }
            diagnostic << "]=" << frontiersynch::contractName(candidate.membership);
            for (const auto& reason : candidate.diagnostics) {
                diagnostic << "(" << frontiersynch::recognitionName(reason.issue) << ":"
                           << frontiersynch::contractName(frontiersynch::contractDiagnosticKind(reason)) << ")";
            }
            for (const auto& reason : candidate.arithmeticDiagnostics) {
                const auto category = reason.issue == frontiersynch::ArithmeticIssue::InvalidConfiguration ?
                    "producer-limit" : (reason.outsideClass ? "candidate-criterion-violation" : "unmet-obligation");
                diagnostic << "(" << frontiersynch::recognitionName(reason.issue) << ":" << category
                           << ",count=" << reason.count << ")";
            }
            for (const auto& obligation : candidate.obligations) {
                if (obligation.status != frontiersynch::ContractStatus::Established) {
                    diagnostic << "(" << obligation.name << ":"
                               << frontiersynch::contractName(obligation.status) << ")";
                }
            }
        }
        diagnostic << "; analysis/export/endpoint failure: " << routeError;
    }
    return prepared;
}
class PTOFrontierAnalysisPass : public impl::PTOFrontierAnalysisBase<PTOFrontierAnalysisPass> {
public:
    using Base = impl::PTOFrontierAnalysisBase<PTOFrontierAnalysisPass>;
    PTOFrontierAnalysisPass() = default;
    explicit PTOFrontierAnalysisPass(const PTOFrontierAnalysisOptions& options) : Base(options) {}
    void runOnOperation() override {
        if (gmAlias != "may-alias" && gmAlias != "may-not-alias") {
            getOperation().emitError("gm-alias must be may-alias or may-not-alias");
            signalPassFailure();
            return;
        }
        llvm::SmallDenseSet<int64_t, 8> uniqueIds;
        for (auto id : allocationIds) {
            if (!frontiersynch::isNativeEventId(id) || !uniqueIds.insert(id).second) {
                getOperation().emitError("allocation-ids must be distinct native IDs from 0..7");
                signalPassFailure();
                return;
            }
        }
        if (deferredOutput && (analysisOnly || physicalOutput || completionOnly || !prepareAllocation)) {
            getOperation().emitError(
                "deferred-output requires logical allocation preparation and excludes analysis-only, "
                "physical-output and completion-only");
            signalPassFailure();
            return;
        }
        if (completionOnly && (analysisOnly || physicalOutput || !arithmeticProfile.empty())) {
            getOperation().emitError(
                "completion-only is an explicit conservative logical request without exact audit or physical-output");
            signalPassFailure();
            return;
        }
        if (physicalOutput && (analysisOnly || !prepareAllocation)) {
            getOperation().emitError("physical-output requires synchronization and prepare-allocation=true");
            signalPassFailure();
            return;
        }
        if (physicalOutput && allocationIds.empty()) {
            allocationIds = frontiersynch::defaultEventIds();
        }
        if (!allocationIds.empty() && (analysisOnly || !prepareAllocation)) {
            getOperation().emitError("allocation-ids requires logical preparation with prepare-allocation=true");
            signalPassFailure();
            return;
        }
        auto policy = gmAlias == "may-alias" ? GMAliasPolicy::MayAlias : GMAliasPolicy::MayNotAlias;
        std::optional<frontiersynch::ArithmeticLimits> profile;
        if (!arithmeticProfile.empty()) {
            auto parsed = frontiersynch::parseArithmeticProfile(arithmeticProfile);
            if (failed(parsed)) {
                getOperation().emitError("arithmetic-profile must be positive k:D:P:C with P<=INT64_MAX");
                signalPassFailure();
                return;
            }
            profile = *parsed;
        }
        if (!analysisOnly) {
            if (failed(
                    frontiersynch::insertModuleSynchronization(
                        getOperation(), policy,
                        [&](func::FuncOp function, GMAliasPolicy contextPolicy, bool closed) {
                            return prepareFunction(
                                function, contextPolicy, profile, profileStages, prepareAllocation, allocationIds,
                                closed, physicalOutput, completionOnly, deferredOutput,
                                experimentalCoordinateAllocation ?
                                    frontiersynch::ArithmeticRegionalAllocationPolicy::CoordinateCandidate :
                                    frontiersynch::ArithmeticRegionalAllocationPolicy::SingleId);
                        },
                        completionOnly, normalizeTwoTripLoops))) {
                signalPassFailure();
            }
            return;
        }
        // Explicit mathematical checkpoint: retain the detailed original-region
        // audit without preparing or attaching any synchronization commands.
        bool unfinished = false;
        getOperation().walk([&](ModuleOp module) {
            auto delegation = frontiersynch::recognizeClosedCallees(module);
            for (auto function : module.getOps<func::FuncOp>()) {
                const bool skip = function.isDeclaration() || hasManualOnCoreSynchronization(function);
                if (skip) { continue; }
                unfinished = true;
                frontiersynch::FrontierAnalysis analysis(function, experimentalCoordinateAllocation ?
                    frontiersynch::ArithmeticRegionalAllocationPolicy::CoordinateCandidate :
                    frontiersynch::ArithmeticRegionalAllocationPolicy::SingleId);
                analysis.enableStageProfiling(profileStages);
                StageProfileReport profileReport{analysis, "exhaustive-certification"};
                const auto contextPolicy = delegation.calledFunctions.contains(function) ?
                    GMAliasPolicy::MayAlias : policy;
                if (failed(analysis.initialize(contextPolicy, true))) {
                    function.emitError("frontier shared-input or structural recognition failed; "
                                       "analysis-only checkpoint unavailable");
                    continue;
                }
                if (profile && failed(analysis.configureArithmeticProfiles(
                    ArrayRef<frontiersynch::ArithmeticLimits>(&*profile, 1),
                    ArrayRef<frontiersynch::ArithmeticLimits>(&*profile, 1)))) {
                    function.emitError("frontier arithmetic profile context could not be established");
                    continue;
                }
                if (failed(analysis.recognizeAllForms())) { continue; }
                const auto certifications = analysis.certifyRegions();
                const auto& candidates = analysis.result()->contractAudit;
                const auto wrapper = delegation.wrappers.find(function);
                auto diagnostic = function.emitError(
                    "frontier analysis-only checkpoint: stopped after mathematical analysis; "
                    "synchronization emission and allocation were not requested");
                for (const auto& result : certifications) {
                    auto& note = diagnostic.attachNote(analysis.result()->nodes[result.region].anchor->getLoc());
                    note << "region=" << result.region << " ";
                    if (result.status == frontiersynch::CertificationStatus::Recognized) {
                        note << "recognized tractable class " << result.selectedClass
                             << " representation=" << result.representation << "; exact demands retained";
                    } else {
                        note << "unresolved tractable-class obligations; compilation support not evaluated";
                        for (const auto& obligation : result.analysis.obligations) {
                            if (!obligation.diagnostic.empty()) { note << "; " << obligation.diagnostic; }
                        }
                    }
                }
                if (wrapper != delegation.wrappers.end()) {
                    diagnostic.attachNote(function.getLoc()) << "class=closed-callee region=function membership="
                        << frontiersynch::recognitionName(wrapper->second.result.state)
                        << "; emission not requested";
                }
                for (const auto& candidate : candidates) {
                    auto& note = diagnostic.attachNote(function.getLoc());
                    note << "class=" << frontiersynch::contractName(candidate.kind) << " region=";
                    if (candidate.node) { note << *candidate.node; }
                    else { note << "function"; }
                    note << " membership=" << frontiersynch::contractName(candidate.membership)
                         << "; emission not requested";
                    for (const auto& reason : candidate.diagnostics) {
                        note << "; " << frontiersynch::recognitionName(reason.issue);
                    }
                    for (const auto& reason : candidate.arithmeticDiagnostics) {
                        note << "; " << frontiersynch::recognitionName(reason.issue);
                    }
                }
                diagnostic.attachNote(function.getLoc()) <<
                    "mathematical analysis ran; synchronization insertion and allocation were not run; IR unchanged";
            }
        });
        if (unfinished) { signalPassFailure(); }
    }
};
} // namespace
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> frontiersynch::prepareFunctionSynchronization(
    func::FuncOp function, GMAliasPolicy policy, bool completeInvocation)
{
    return prepareFunction(function, policy, std::nullopt, false, true, {}, completeInvocation);
}
std::unique_ptr<Pass> createPTOFrontierAnalysisPass() {
    return std::make_unique<PTOFrontierAnalysisPass>();
}
std::unique_ptr<Pass> createPTOFrontierAnalysisPass(const PTOFrontierAnalysisOptions& options) {
    return std::make_unique<PTOFrontierAnalysisPass>(options);
}
} // namespace mlir::pto
