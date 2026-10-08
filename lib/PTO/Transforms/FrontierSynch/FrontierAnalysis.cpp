// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Per-function recognition and detached demand production. The public pass
// coordinates module closure before committing logical synchronization.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/ClosedCallees.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicInsertion.h"
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingInsertion.h"
namespace mlir::pto::frontiersynch {
namespace {
// Whole-invocation shortcut only: every payload uses the protected scalar
// pipe, and every other effect is accounted for by the shared leaf contract.
// No storage geometry, visit enumeration or regional selector is required.
bool nativeScalarRequirements(func::FuncOp function, const SyncInput& input)
{
    auto scalar = ptoStorageProtection().scalarPipe;
    if (!scalar || function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) ||
        llvm::any_of(input.instructions(), [&](const auto* phase) {
            return static_cast<uint32_t>(phase->kPipeValue) != *scalar ||
                llvm::any_of(phase->elementOp->getResults(), [&](Value value) {
                    return resultAvailability(*phase, value) == SyncResultAvailability::RequiresCompletion;
                });
        })) { return false; }
    PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
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
    nativeScalarOnly = false;
    arithmeticAnalysis.reset();
    generalArithmeticAnalysis.reset();
    arithmeticGeneratorStage.reset();
    arithmeticPeriodicAnalysis.reset();
    periodicExports = {};
    explicitAnalysis.reset();
    boundedAnalysis.reset();
    sequenceAnalysis.reset();
    finiteVisitAnalyses.clear();
    program.reset();
    storage.reset();
    initialized = true;
    policy = requestedPolicy;
    if (!function) {
        return failure();
    }
    auto pending = std::make_shared<SyncInput>(policy);
    if (failed(pending->build(function, SyncInstructionView::PipeEnvelopes))) {
        return failure();
    }
    nativeScalarOnly = nativeScalarRequirements(function, *pending);
    storage = std::move(pending);
    return nativeScalarOnly && !requireStructure ? success() : recognizeStructure();
}
LogicalResult FrontierAnalysis::recognizeStructure() {
    if (program) { return success(); }
    if (!storage) { return failure(); }
    auto recognized = recognizeProgram(function, *storage);
    if (failed(recognized)) { return failure(); }
    program = std::move(*recognized);
    return success();
}
LogicalResult FrontierAnalysis::analyzeNumericCandidates() {
    if (failed(recognizeStructure())) { return failure(); }
    PhaseIndex index;
    if (failed(index.build(function, *storage))) { return failure(); }
    for (auto& node : program->nodes) {
        if (node.varyingRotating && !node.varyingDemands &&
            node.varyingRotating->result.state == RecognitionState::Applicable) {
            node.varyingDemands = analyzeVaryingRotating(*node.varyingRotating, index, *storage);
        }
        if (!node.numericTemplate || node.periodicAnalysis ||
            node.numericTemplate->result.state != RecognitionState::Applicable) { continue; }
        node.periodicAnalysis = analyzeNumericTemplate(*node.numericTemplate);
        if (!node.periodicAnalysis->error.empty()) { continue; }
        node.logicalEndpoints = buildNumericTemplateEndpoints(*node.numericTemplate, *node.periodicAnalysis);
        if (!node.logicalEndpoints->logical.error.empty()) { continue; }
        node.periodicAllocation = buildPeriodicAllocation(*node.periodicAnalysis);
    }
    refreshProgramContractAudit(*program);
    return success();
}
LogicalResult FrontierAnalysis::analyzeExplicitFunction() {
    if (!storage || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        return failure();
    }
    if (!explicitAnalysis) {
        PhaseIndex index;
        if (failed(index.build(function, *storage))) {
            return failure();
        }
        explicitAnalysis = analyzeExplicit(function.front(), index, *storage);
    }
    return success(explicitAnalysis->error.empty());
}
SequenceAnalysis* FrontierAnalysis::analyzeSequenceFunction() {
    if (failed(recognizeStructure())) { return nullptr; }
    if (!sequenceAnalysis) {
        sequenceAnalysis = analyzeSequence(function, *storage, *program);
        recordSequenceContractAttempt(*program, *storage, *sequenceAnalysis);
        refreshProgramContractAudit(*program);
    }
    if (!sequenceAnalysis->error.empty()) { (void)analyzeFiniteVisitCandidates(); }
    return &*sequenceAnalysis;
}
LogicalResult FrontierAnalysis::analyzeFiniteVisitCandidates()
{
    if (failed(recognizeStructure())) { return failure(); }
    for (const auto& candidate : program->finiteVisitContracts) {
        if (!candidate.node || finiteVisitAnalyses.count(*candidate.node)) { continue; }
        const auto eligible = llvm::find_if(candidate.obligations, [](const auto& item) {
            return item.name == "exhaustive-original-type-selection" && item.status == ContractStatus::Established;
        });
        if (eligible == candidate.obligations.end()) { continue; }
        auto result = std::make_shared<FiniteVisitAnalysis>(
            analyzeFiniteVisitLoop(function, *storage, *program, *candidate.node));
        finiteVisitAnalyses.emplace(*candidate.node, std::move(result));
    }
    for (auto& candidate : program->finiteVisitContracts) {
        if (!candidate.node) { continue; }
        auto found = finiteVisitAnalyses.find(*candidate.node);
        if (found != finiteVisitAnalyses.end()) { candidate = found->second->recognition.contract; }
    }
    refreshProgramContractAudit(*program);
    return success();
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareBoundedLifetimeFunction(std::string& error)
{
    if (failed(recognizeStructure())) { return failure(); }
    return prepareBoundedLifetimeInsertion(function, *storage, *program, error, &boundedAnalysis);
}
LogicalResult FrontierAnalysis::analyzeArithmeticPeriodicFunction()
{
    if (failed(recognizeArithmetic()) || !program->arithmetic) { return failure(); }
    if (!arithmeticPeriodicAnalysis) {
        std::string diagnostic;
        if (!checkArithmeticPeriodicSkeleton(*program->arithmetic, diagnostic)) {
            arithmeticPeriodicAnalysis.emplace();
            arithmeticPeriodicAnalysis->conversion.status = ArithmeticPeriodicStatus::AdapterUnavailable;
            arithmeticPeriodicAnalysis->conversion.diagnostic = std::move(diagnostic);
            return failure();
        }
        if (!arithmeticGeneratorStage) {
            const auto protection = structuredProtection(storage->accesses());
            arithmeticGeneratorStage.emplace(analyzeGeneralArithmeticGenerators(*program->arithmetic, &protection));
        }
        arithmeticPeriodicAnalysis = convertArithmeticPeriodicProgram(*program->arithmetic, *arithmeticGeneratorStage);
    }
    const auto& conversion = arithmeticPeriodicAnalysis->conversion;
    return success(conversion.status == ArithmeticPeriodicStatus::Applicable && conversion.guarded &&
                   conversion.guarded->error.empty());
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareArithmeticPeriodicFunction()
{
    if (failed(analyzeArithmeticPeriodicFunction())) { return failure(); }
    const auto& converted = *arithmeticPeriodicAnalysis;
    const auto& conversion = converted.conversion;
    std::vector<uint64_t> residues;
    for (const auto& site : converted.sites) { residues.push_back(site.residue); }
    GuardedPeriodicEndpointInput input{converted.loop, conversion.expressions, converted.phases,
        conversion.payloads, conversion.generators, &*conversion.guarded, converted.period, residues};
    periodicExports = {};
    auto prepared = prepareGuardedPeriodicEndpoints(function, input, periodicExports.endpointError);
    if (failed(prepared)) { return failure(); }
    periodicExports.endpointsAvailable = true;
    (*prepared)->allocationCertificate = guardedPeriodicAllocationCertificate(*conversion.expressions,
        conversion.payloads, conversion.generators, *conversion.guarded, (*prepared)->planId, function.getContext());
    periodicExports.allocationAvailable = static_cast<bool>((*prepared)->allocationCertificate);
    if (!periodicExports.allocationAvailable) {
        periodicExports.allocationError = "periodic source-identity allocation certificate unavailable";
    }
    return prepared;
}
LogicalResult FrontierAnalysis::analyzeArithmeticFunction()
{
    if (failed(recognizeArithmetic()) || !program->arithmetic) { return failure(); }
    const auto& arithmetic = *program->arithmetic;
    // A failed periodic conversion must not switch a difference-bound input
    // to the general integer reducer merely because that adapter used it.
    if (arithmetic.recognition.arithmeticClass == ArithmeticClass::Differences) {
        arithmeticGeneratorStage.reset();
        if (!arithmeticAnalysis) {
            const auto protection = structuredProtection(storage->accesses());
            arithmeticAnalysis = analyzeArithmeticDemandsWithProtection(arithmetic, protection);
        }
        return success(arithmeticAnalysis->error.empty() && arithmeticAnalysis->exactMinimum);
    }
    if (arithmeticGeneratorStage) {
        if (!generalArithmeticAnalysis) {
            generalArithmeticAnalysis = completeGeneralArithmeticDemands(std::move(*arithmeticGeneratorStage));
        }
        arithmeticGeneratorStage.reset();
        return success(generalArithmeticAnalysis->error.empty() && generalArithmeticAnalysis->exactMinimum);
    }
    if (generalArithmeticAnalysis) {
        return success(generalArithmeticAnalysis->error.empty() && generalArithmeticAnalysis->exactMinimum);
    }
    const auto protection = structuredProtection(storage->accesses());
    if (!generalArithmeticAnalysis) {
        generalArithmeticAnalysis = analyzeGeneralArithmeticDemandsWithProtection(arithmetic, protection);
    }
    return success(generalArithmeticAnalysis->error.empty() && generalArithmeticAnalysis->exactMinimum);
}
bool FrontierAnalysis::hasWholeFunctionMinimumDemands() const
{
    if (boundedAnalysis) { return true; }
    if (explicitAnalysis && explicitAnalysis->error.empty()) { return true; }
    if (arithmeticPeriodicAnalysis) {
        const auto& converted = arithmeticPeriodicAnalysis->conversion;
        if (converted.status == ArithmeticPeriodicStatus::Applicable &&
            ((converted.guarded && converted.guarded->error.empty()) ||
             (converted.numerical && converted.numerical->error.empty()))) { return true; }
    }
    if ((arithmeticAnalysis && arithmeticAnalysis->error.empty() && arithmeticAnalysis->exactMinimum) ||
        (generalArithmeticAnalysis && generalArithmeticAnalysis->error.empty() &&
         generalArithmeticAnalysis->exactMinimum)) { return true; }
    if (!program) { return false; }
    // This snapshot is recorded before preparation and verifies original input,
    // root coverage and prerequisites. Preparation may set the state's error.
    if (program->sequenceContract && program->sequenceContract->membership == ContractStatus::Established &&
        program->sequenceContract->demands == ContractImplementation::Available) { return true; }
    for (const auto& node : program->nodes) {
        if (node.unsupportedContext || !node.numericTemplate || !node.periodicAnalysis ||
            !node.periodicAnalysis->error.empty()) { continue; }
        const auto& numeric = *node.numericTemplate;
        if (numeric.result.state != RecognitionState::Applicable || numeric.specializedBody ||
            !numeric.outer || numeric.outer->getParentOp() != function.operator->()) { continue; }
        const bool complete = llvm::all_of(program->payloads, [&](const auto& payload) {
            return numeric.outer->isProperAncestor(payload.phase->elementOp);
        });
        if (complete) { return true; }
    }
    return false;
}
void FrontierAnalysis::noteSequenceEndpointOutcome(StringRef error) {
    if (!program) { return; }
    recordSequenceEndpointAttempt(*program, error);
    refreshProgramContractAudit(*program);
}
LogicalResult FrontierAnalysis::recognizeArithmetic() {
    if (failed(recognizeStructure())) { return failure(); }
    if (program->arithmetic || function.isDeclaration()) {
        return success();
    }
    PhaseIndex index;
    if (failed(index.build(function, *storage))) {
        return failure();
    }
    // Record every configured arithmetic contract before selecting a backend.
    // A later profile never erases membership established by an earlier one.
    std::optional<ArithmeticProgram> selected;
    for (uint64_t period : {uint64_t{1}, uint64_t{2}}) {
        const ArithmeticLimits limits{8, 8, period, 8};
        auto candidate = recognizeArithmeticProgram(function, index, *storage, storage->accesses(), limits);
        recordArithmeticContractAttempt(*program, limits, candidate);
        const bool alreadyAccepted = selected && selected->extraction.state == RecognitionState::Applicable &&
                                     selected->recognition.state == RecognitionState::Applicable;
        if (!alreadyAccepted) { selected = std::move(candidate); }
    }
    program->arithmetic = std::move(selected);
    refreshProgramContractAudit(*program);
    return success();
}
} // namespace mlir::pto::frontiersynch
namespace mlir::pto {
#define GEN_PASS_DEF_PTOFRONTIERANALYSIS
#include "PTO/Transforms/Passes.h.inc"
namespace {
// Persist recognition independently of which logical backend succeeds. The
// report contains no borrowed operations or values and survives insertion.
DictionaryAttr contractReport(const frontiersynch::ProgramRecognition& program, StringRef logicalBackend)
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
    report.set("physical_allocation", b.getStringAttr("not-requested"));
    return report.getDictionary(context);
}
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepareWholeFunctionArithmetic(
    func::FuncOp function, frontiersynch::FrontierAnalysis& analysis, std::string& error, StringRef& backend)
{
    auto periodic = analysis.prepareArithmeticPeriodicFunction();
    const auto& exports = analysis.arithmeticPeriodicExports();
    if (succeeded(periodic)) {
        // Allocation is a separate request. Preserve a valid logical plan
        // even when this query representation has not certified ID reuse.
        backend = "arithmetic-periodic"; return periodic;
    }
    if (exports.endpointsAvailable) { error += "; arithmetic periodic allocation: " + exports.allocationError; }
    else if (!exports.endpointError.empty()) { error += "; arithmetic periodic endpoints: " + exports.endpointError; }
    if (const auto* converted = analysis.arithmeticPeriodicDemands()) {
        const auto& conversion = converted->conversion;
        if (!conversion.diagnostic.empty()) { error += "; arithmetic periodic adapter: " + conversion.diagnostic; }
        if (!conversion.exportError.empty()) { error += "; arithmetic periodic export: " + conversion.exportError; }
    }
    const bool analyzed = succeeded(analysis.analyzeArithmeticFunction());
    backend = "arithmetic";
    if (!analysis.result() || !analysis.result()->arithmetic) { return failure(); }
    const auto& arithmetic = *analysis.result()->arithmetic;
    if (const auto* demands = analysis.arithmeticDemands()) {
        if (!analyzed) { error += "; arithmetic: " + demands->error; return failure(); }
        return frontiersynch::prepareArithmeticInsertion(function, arithmetic, *demands, error);
    }
    if (const auto* demands = analysis.generalArithmeticDemands()) {
        if (!analyzed) { error += "; arithmetic: " + demands->error; return failure(); }
        return frontiersynch::prepareGeneralArithmeticInsertion(function, arithmetic, *demands, error);
    }
    return failure();
}
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepareFunction(
    func::FuncOp function, GMAliasPolicy policy)
{
    frontiersynch::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize(policy))) {
        return failure();
    }
    // Full arithmetic relation production is lazy. Cheap routes need only
    // structural observations; exhaustive diagnostic clients request arithmetic
    // explicitly through recognizeArithmetic().
    FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepared = failure();
    std::string routeError;
    StringRef logicalBackend;
    const bool straight = !function.isDeclaration() && llvm::hasSingleElement(function.getBody()) &&
        llvm::all_of(function.front(), [](Operation& op) { return op.getNumRegions() == 0; });
    if (analysis.hasOnlyNativeScalarRequirements()) {
        auto scalar = std::make_unique<frontiersynch::PreparedLogicalPlan>(0);
        // Protected scalar accesses need no internal demands. Keep the
        // invocation drain: protection is not completion at issue.
        scalar->completeInvocation = !analysis.input()->instructions().empty();
        frontiersynch::ExplicitAnalysis empty;
        scalar->allocationCertificate = frontiersynch::explicitAllocationCertificate(
            empty, scalar->planId, function.getContext());
        prepared = std::move(scalar);
        logicalBackend = "native-scalar";
    } else if (straight) {
        if (succeeded(analysis.analyzeExplicitFunction())) {
            prepared = frontiersynch::prepareExplicitInsertion(function, *analysis.explicitResult());
            if (succeeded(prepared)) { logicalBackend = "explicit"; }
        } else {
            routeError = analysis.explicitResult() ? analysis.explicitResult()->error :
                         "explicit analysis unavailable";
        }
    } else {
        // Preserve the existing certified numerical route (including its
        // allocation export) when available. Direct rotating extraction
        // covers symbolic rotations which have no fixed local effect word.
        if (succeeded(analysis.analyzeNumericCandidates())) {
            const bool numerical = llvm::any_of(analysis.result()->nodes, [](const auto& node) {
                return node.numericTemplate &&
                    node.numericTemplate->result.state == frontiersynch::RecognitionState::Applicable &&
                    node.logicalEndpoints && node.logicalEndpoints->logical.error.empty();
            });
            if (!numerical) {
                prepared = frontiersynch::prepareRotatingInsertion(function, *analysis.input(), *analysis.result());
                if (succeeded(prepared)) { logicalBackend = "rotating"; }
            }
            if (failed(prepared) && numerical) {
                prepared = frontiersynch::prepareNumericTemplateInsertion(function, *analysis.result());
                if (succeeded(prepared)) { logicalBackend = "numerical-periodic"; }
            }
            if (failed(prepared)) {
                std::string mixedError;
                prepared = frontiersynch::prepareMixedStrideInsertion(
                    function, *analysis.input(), *analysis.result(), mixedError);
                if (succeeded(prepared)) { logicalBackend = "mixed-stride"; }
                if (failed(prepared) && !mixedError.empty()) { routeError += "; " + mixedError; }
            }
            if (failed(prepared)) {
                prepared = frontiersynch::prepareGuardedRotatingInsertion(
                    function, *analysis.input(), *analysis.result());
                if (succeeded(prepared)) { logicalBackend = "guarded-rotating"; }
            }
            if (failed(prepared)) {
                prepared = analysis.prepareBoundedLifetimeFunction(routeError);
                if (succeeded(prepared)) { logicalBackend = "bounded-lifetime"; }
            }
            if (failed(prepared)) {
                if (auto* sequence = analysis.analyzeSequenceFunction()) {
                    prepared = frontiersynch::prepareSequenceInsertion(*sequence);
                    if (sequence->error.empty()) {
                        analysis.noteSequenceEndpointOutcome(failed(prepared) && sequence->insertionError.empty() ?
                            "sequence endpoint preparation unavailable" : sequence->insertionError);
                    }
                    routeError += sequence->error.empty() ? sequence->insertionError : sequence->error;
                    if (succeeded(prepared)) { logicalBackend = "sequence"; }
                }
            }
        } else { routeError = "exact structured recognition unavailable"; }
    }
    if (failed(prepared) && straight && analysis.result()) {
        if (auto* sequence = analysis.analyzeSequenceFunction()) {
            prepared = frontiersynch::prepareSequenceInsertion(*sequence);
            if (sequence->error.empty()) {
                analysis.noteSequenceEndpointOutcome(failed(prepared) && sequence->insertionError.empty() ?
                    "sequence endpoint preparation unavailable" : sequence->insertionError);
            }
            routeError += sequence->error.empty() ? sequence->insertionError : sequence->error;
            if (succeeded(prepared)) { logicalBackend = "sequence"; }
        }
    }
    if (failed(prepared)) {
        prepared = prepareWholeFunctionArithmetic(function, analysis, routeError, logicalBackend);
    }
    if (failed(prepared) && analysis.hasWholeFunctionMinimumDemands()) {
        routeError = "unmet-exports: exact whole-function demands retained; " + routeError;
    } else if (failed(prepared)) {
        auto compact = frontiersynch::prepareCompactBoundingInsertion(function, analysis.sharedInput());
        if (compact.prepared) { prepared = std::move(compact.prepared); logicalBackend = "compact-bounding"; }
        else {
            const auto& reason = compact.error.empty() ? compact.exportError : compact.error;
            if (!reason.empty()) { routeError += "; compact bounding: " + reason; }
        }
    }
    if (succeeded(prepared)) { (*prepared)->recognitionReport = contractReport(*analysis.result(), logicalBackend); }
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
        auto policy = gmAlias == "may-alias" ? GMAliasPolicy::MayAlias : GMAliasPolicy::MayNotAlias;
        if (failed(frontiersynch::insertModuleSynchronization(getOperation(), policy, prepareFunction))) {
            signalPassFailure();
        }
        // Inserted guards and commands invalidate structural analysis.
    }
};
} // namespace
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>>
frontiersynch::prepareFunctionSynchronization(func::FuncOp function, GMAliasPolicy policy)
{
    return prepareFunction(function, policy);
}
std::unique_ptr<Pass> createPTOFrontierAnalysisPass() {
    return std::make_unique<PTOFrontierAnalysisPass>();
}
std::unique_ptr<Pass> createPTOFrontierAnalysisPass(const PTOFrontierAnalysisOptions& options) {
    return std::make_unique<PTOFrontierAnalysisPass>(options);
}
} // namespace mlir::pto
