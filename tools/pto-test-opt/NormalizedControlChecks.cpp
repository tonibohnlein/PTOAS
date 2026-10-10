// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check shared normalized input across different mathematical adapters.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateRegional.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicInsertion.h"
#include "../../lib/PTO/Transforms/FrontierSynch/NormalizedControl.h"
#include "../../lib/PTO/Transforms/FrontierSynch/PhaseNormalization.h"
#include "../../lib/PTO/Transforms/FrontierSynch/NumericTemplatePlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FiniteExpansionPlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRelationsInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/SequenceAnalysisInternal.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
LogicalResult checkPhaseMinMax(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    pto::SyncInput input(policy);
    PhaseIndex index;
    const bool initialized = succeeded(input.build(function)) && succeeded(index.build(function, input));
    if (!initialized) { return failure(); }
    auto loops = function.getOps<scf::ForOp>();
    if (loops.empty()) { return function.emitError("phase extrema fixture requires an original loop"); }
    auto loop = *loops.begin();
    RegionExpressions expressions;
    PhaseNormalization normalizer(loop, index, expressions);
    bool valid = true;
    loop.walk([&](Operation* op) {
        if (!isa<arith::MinSIOp, arith::MaxSIOp>(op)) { return; }
        for (uint64_t phase = 0; phase < 4; ++phase) {
            auto value = normalizer.atPhase(op->getResult(0), phase, 1);
            if (function->hasAttr("test.narrow_phase")) { valid &= !value; continue; }
            const int64_t shifted = static_cast<int64_t>(phase) - 1;
            const int64_t expected = isa<arith::MinSIOp>(op) ? std::min(shifted, int64_t{0}) :
                std::max(shifted, int64_t{0});
            const auto size = expressions.size();
            auto retry = normalizer.atPhase(op->getResult(0), phase, 1);
            valid &= value && expressions.constantValue(*value) == static_cast<uint64_t>(expected) &&
                value == retry && expressions.size() == size;
        }
    });
    if (!valid) { return function.emitError("signed phase extrema or narrow-layout refusal changed"); }
    llvm::outs() << "phase-minmax: signed-extrema cached narrow-layout-refused\n";
    return success();
}
LogicalResult checkPiecewiseAccessCache(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    std::string error;
    auto first = session.specializedArithmeticDemands({function, function}, {}, error);
    const auto count = session.specializedArithmeticConstructions();
    auto again = session.specializedArithmeticDemands({function, function}, {}, error);
    const bool expected = function.getName() != "unsupported";
    const bool cached = bool(first) == expected && first == again &&
        session.specializedArithmeticConstructions() == count;
    if (!cached) {
        return function.emitError("piecewise arithmetic request did not cache its exact result or failure");
    }
    FrontierAnalysis cold(function);
    if (failed(cold.initialize(policy))) { return failure(); }
    auto independent = cold.specializedArithmeticDemands({function, function}, {}, error);
    const bool isolated = bool(independent) == expected && (!first || independent != first);
    if (!isolated) {
        return function.emitError("piecewise arithmetic cache crossed invocation contexts");
    }
    llvm::outs() << "piecewise-access: cached exact-or-unresolved isolated-context\n";
    return success();
}
LogicalResult checkSequenceEndpointRetry(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    std::string before;
    llvm::raw_string_ostream original(before); function.print(original); original.flush();
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    auto weak = session.minimumDemands();
    if (weak.status != AnalysisStatus::Ready || !weak.mathematical) {
        return function.emitError("sequence retry lacks initial exact mathematics");
    }
    AnalysisRequest request;
    request.needs.synchronization = true;
    auto strong = session.analyze(request);
    if (strong.status != AnalysisStatus::Ready || !strong.mathematical) {
        return function.emitError("sequence endpoint retry unavailable");
    }
    if (strong.mathematical->backend != "sequence-endpoints") {
        return function.emitError("strong sequence retry chose ") << strong.mathematical->backend;
    }
    bool expanded = false;
    if (weak.mathematical->sequenceDemands && weak.mathematical->sequenceDemands->state) {
        for (const auto& child : weak.mathematical->sequenceDemands->state->children) {
            if (!child.originalNode) { continue; }
            auto original = session.minimumDemands(*child.originalNode);
            expanded |= original.mathematical && original.mathematical->backend == "expanded-finite-guarded";
        }
    }
    if (!expanded || strong.mathematical == weak.mathematical) {
        return function.emitError("sequence retry did not replace expanded child coordinates");
    }
    const auto attempts = session.constructionCounts().mathematicalAttempts;
    auto again = session.analyze(request);
    auto retained = session.minimumDemands();
    const bool cached = again.status == AnalysisStatus::Ready &&
        again.mathematical == strong.mathematical && retained.mathematical == weak.mathematical &&
        session.constructionCounts().mathematicalAttempts == attempts;
    if (!cached) { return function.emitError("strong sequence retry replaced or reconstructed mathematics"); }
    auto left = session.prepareLogical(strong);
    auto right = session.prepareLogical(again);
    const bool fresh = succeeded(left) && succeeded(right) && left->get() != right->get();
    if (!fresh) { return function.emitError("sequence retry lacks fresh detached plans"); }
    llvm::DenseSet<Operation*> operations;
    for (const auto& preparation : (*left)->preparation) {
        if (preparation.code->getParent()) { return failure(); }
        preparation.code->walk([&](Operation* op) { operations.insert(op); });
    }
    for (const auto& preparation : (*right)->preparation) {
        if (preparation.code->getParent()) { return failure(); }
        bool shared = false;
        preparation.code->walk([&](Operation* op) { shared |= operations.contains(op); });
        if (shared) { return function.emitError("sequence retry plans share detached operations"); }
    }
    auto renderPlan = [](const PreparedLogicalPlan& plan) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        for (const auto& preparation : plan.preparation) {
            for (auto& op : *preparation.code) { op.print(stream); }
        }
        return text;
    };
    const auto independent = renderPlan(**right);
    left->reset();
    const bool ownerRetained = renderPlan(**right) == independent;
    if (!ownerRetained) { return function.emitError("sequence plan lost its owner"); }
    FrontierAnalysis cold(function);
    if (failed(cold.initialize(policy))) { return failure(); }
    if (cold.analyze(request).status != AnalysisStatus::Ready) {
        return function.emitError("cold sequence endpoint request failed");
    }
    std::string after;
    llvm::raw_string_ostream current(after); function.print(current); current.flush();
    if (before != after) { return function.emitError("sequence retry changed original IR"); }
    llvm::outs() << "sequence-endpoint-retry: retained-expanded-math recomposed-strong-children "
                 << "cached fresh-plans source-unchanged\n";
    return success();
}
LogicalResult checkNaturalRotatingExportRetry(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    const ArithmeticLimits profile{8, 9, 1, 4096};
    if (failed(session.configureArithmeticProfiles({profile}, {profile}))) { return failure(); }
    std::string before;
    llvm::raw_string_ostream original(before); function.print(original); original.flush();
    auto node = llvm::find_if(session.result()->nodes, [](const auto& value) {
        return value.kind == StructureKind::Loop;
    });
    if (node == session.result()->nodes.end()) { return failure(); }
    const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
    auto weak = session.minimumDemands(id);
    const bool retained = weak.mathematical &&
        (weak.mathematical->rotatingDemands || weak.mathematical->guardedRotatingDemands);
    if (!retained) { return function.emitError("natural retry requires exact rotating mathematics"); }
    std::string error;
    GuardedRotatingSpecialization specialization;
    specialization.loop = cast<scf::ForOp>(node->anchor);
    auto guarded = session.instantiateGuardedDemands(specialization, std::make_shared<RegionExpressions>(), error);
    if (!guarded || !error.empty()) { return function.emitError("natural guarded mathematics unavailable: " + error); }
    auto unavailable = guardedRotatingRegionalResult(function, *session.input(), guarded->demands, error);
    const bool naturalFailure = failed(unavailable) && error.find("supported visit limit") != std::string::npos;
    if (!naturalFailure) {
        return function.emitError("natural regional export did not reach its slot limit: " + error);
    }
    const auto reductions = session.constructionCounts().rotatingReductions;
    const auto guardedBuilds = session.specializedGuardedConstructions();
    AnalysisRequest strong{id}; strong.needs.queries = true; strong.needs.selectors = true;
    auto later = session.analyze(strong);
    auto usesArithmetic = [](const AnalysisOutcome& outcome) {
        if (!outcome.mathematical) { return false; }
        const auto& math = *outcome.mathematical;
        return math.arithmeticRegionalDemands ||
            (math.sequenceDemands && math.sequenceDemands->cost.arithmeticRegions != 0);
    };
    const bool laterReady = later.status == AnalysisStatus::Ready && later.mathematical &&
        usesArithmetic(later) && later.available.queries && later.available.selectors;
    if (!laterReady) {
        llvm::errs() << "selected " << (later.mathematical ? later.mathematical->backend : "none") << "\n";
        for (const auto& issue : later.obligations) {
            llvm::errs() << issue.diagnostic << "\n";
        }
        return function.emitError("natural rotating export failure blocked later arithmetic");
    }
    const auto arithmeticBuilds = session.arithmeticRegionConstructions();
    auto retry = session.analyze(strong);
    auto again = session.minimumDemands(id);
    auto exports = [](const AnalysisOutcome& outcome) {
        return outcome.regionalExports ? outcome.regionalExports : outcome.mathematical->regionalDemands;
    };
    const auto snapshot = exports(later);
    const bool reused = retry.status == AnalysisStatus::Ready && again.status == AnalysisStatus::Ready &&
        retry.mathematical == later.mathematical && retry.regionalExports == later.regionalExports &&
        snapshot && exports(retry) == snapshot && again.mathematical == weak.mathematical &&
        session.constructionCounts().rotatingReductions == reductions &&
        session.specializedGuardedConstructions() == guardedBuilds &&
        session.arithmeticRegionConstructions() == arithmeticBuilds &&
        !session.constructionCounts().logicalPreparations && !session.constructionCounts().allocationExports;
    if (!reused) { return function.emitError("natural retry rebuilt mathematics or constructed commands"); }
    FrontierAnalysis cold(function);
    const bool coldInitialized = succeeded(cold.initialize(policy)) &&
        succeeded(cold.configureArithmeticProfiles({profile}, {profile}));
    if (!coldInitialized) { return failure(); }
    auto coldResult = cold.analyze(strong);
    const bool coldReady = coldResult.status == AnalysisStatus::Ready && coldResult.mathematical &&
        usesArithmetic(coldResult) && coldResult.available.queries &&
        coldResult.available.selectors && !cold.constructionCounts().logicalPreparations &&
        !cold.constructionCounts().allocationExports;
    if (!coldReady) { return function.emitError("cold strong request did not reach arithmetic"); }
    auto rootRequest = strong; rootRequest.region = 0;
    auto root = session.analyze(rootRequest);
    const bool rootReady = root.status == AnalysisStatus::Ready && root.available.queries && root.available.selectors &&
        !session.constructionCounts().logicalPreparations && !session.constructionCounts().allocationExports;
    if (!rootReady) { return function.emitError("root and regional export retry outcomes disagree"); }
    std::string after;
    llvm::raw_string_ostream current(after); function.print(current); current.flush();
    if (before != after) { return function.emitError("natural export retry changed original IR"); }
    llvm::outs() << "natural-export-retry: rotating-demands retained slot-limit "
                 << "arithmetic-ready cached source-unchanged\n";
    return success();
}
LogicalResult checkNormalizedControlSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    auto node = llvm::find_if(session.result()->nodes, [&](const auto& value) {
        return value.kind == StructureKind::Loop && value.anchor->getParentOp() == function;
    });
    const bool fresh = node != session.result()->nodes.end() && !session.normalizationConstructions();
    if (!fresh) { return failure(); }
    const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
    std::string originalIR;
    llvm::raw_string_ostream originalStream(originalIR);
    function.print(originalStream);
    const auto order = session.exactForms({id});
    const auto records = session.costRecords().size();
    const bool repeatRank = session.exactForms({id}) == order && records == session.costRecords().size() &&
        session.normalizationConstructions() == 1 && !session.constructionCounts().mathematicalAttempts &&
        !session.numericalRegionConstructions();
    if (!repeatRank) { return function.emitError("description ranking constructed demands or repeated normalization"); }
    if (!function->hasAttr("test.normalization_external")) {
        session.exactForms({0});
        const bool canonical = session.normalizationConstructions() == 1;
        if (!canonical) {
            return function.emitError("whole-loop ranking lost canonical representation identity");
        }
    }
    AnalysisRequest requested{id};
    requested.needs.queries = true;
    const auto beforeExports = session.costRecords().size();
    const auto queryOrder = session.exactForms(requested);
    const auto queryRecords = session.costRecords().size();
    requested.needs.selectors = true;
    session.exactForms(requested);
    const auto selectorRecords = session.costRecords().size();
    session.exactForms(requested);
    const auto extra = order.size() > 1 ? 2 : 0;
    const bool separateRequests = queryRecords == beforeExports + extra &&
        selectorRecords == queryRecords + extra && session.costRecords().size() == selectorRecords &&
        !queryOrder.empty() && !session.constructionCounts().mathematicalAttempts &&
        session.normalizationConstructions() == 1;
    if (!separateRequests) { return function.emitError("interface ranking cache repeated or constructed mathematics"); }
    const auto numerical = session.analyzeNumericalRegion({id});
    const bool rejectsNumerical = function->hasAttr("test.normalization_runtime_guard");
    const bool expectedNumerical = rejectsNumerical ? !numerical.mathematical :
        numerical.mathematical && numerical.mathematical->normalizedInput;
    const bool oneInput = expectedNumerical && session.normalizationConstructions() == 1;
    if (!oneInput) {
        return function.emitError("numerical adapter did not retain exactly one normalized input");
    }
    const auto finite = session.analyzeFiniteExpansion({id});
    const bool available = finite.status == AnalysisStatus::Ready && finite.mathematical &&
        finite.mathematical->finiteGuardedDemands && finite.mathematical->normalizedInput;
    if (!available) { return function.emitError("finite adapter cannot consume retained normalized input"); }
    const auto owner = finite.mathematical->normalizedInput;
    const bool alternative = owner->original != nullptr;
    const bool expectedOrder = order.size() == (alternative ? 2 : 1) &&
        (alternative || order.front() == AnalysisForm::Original);
    if (!expectedOrder) {
        return function.emitError("compact fallback was advertised as another expanded description");
    }
    if (alternative) {
        const auto plan = preflightNumericTemplate(cast<scf::ForOp>(node->anchor),
            *owner->index, *owner->input, {}, true, {}, {}, owner);
        const bool eligible = plan.form.result.state == RecognitionState::Applicable;
        auto costs = session.costRecords();
        auto expanded = llvm::find_if(costs, [&](const auto& record) {
            return record.region == id && !record.request && record.method == "form-expanded";
        });
        auto original = llvm::find_if(costs, [&](const auto& record) {
            return record.region == id && !record.request && record.method == "form-original";
        });
        const bool recorded = expanded != costs.end() && original != costs.end();
        if (!recorded) { return function.emitError("candidate costs absent"); }
        EstimatedCount effects = 0;
        for (const auto& occurrence : plan.payloads) {
            effects = estimatedAdd(effects, owner->input->accesses().effectsFor(occurrence.phase).size());
        }
        const bool priced = eligible ? expanded->estimate.generatorPieces == estimatedMultiply(effects, effects) &&
            expanded->estimate.ports == estimatedMultiply(plan.form.countedPayloads, 2) :
            !expanded->estimate.work;
        const auto first = estimatedCostLess(expanded->estimate, original->estimate) ?
            AnalysisForm::SmallCountExpanded : AnalysisForm::Original;
        const bool correctRank = priced && order.front() == first;
        if (!correctRank) {
            return function.emitError("candidate order or residual occurrence cost is incorrect");
        }
        const bool unknownOriginalFirst = eligible && !original->estimate.work &&
            order.front() != AnalysisForm::SmallCountExpanded;
        if (unknownOriginalFirst) {
            return function.emitError("known expanded work did not precede unknown original work");
        }
        const bool runtimeRank = rejectsNumerical && order.front() != AnalysisForm::Original;
        if (runtimeRank) {
            return function.emitError("unknown-work candidates ignored compact representation size");
        }
        FrontierAnalysis dispatched(function);
        if (failed(dispatched.initialize(policy))) { return failure(); }
        const auto selected = dispatched.minimumDemands(id);
        auto actual = dispatched.costRecords();
        auto attempted = [&](const char* method) {
            auto record = llvm::find_if(actual, [&](const auto& value) {
                return value.region == id && !value.request && value.method == method;
            });
            return record != actual.end() && record->attemptConstructions;
        };
        const bool dispatchedFirst = attempted(first == AnalysisForm::Original ? "form-original" : "form-expanded");
        const bool selectedExpanded = first != AnalysisForm::SmallCountExpanded ||
            (selected.status == AnalysisStatus::Ready && selected.mathematical &&
             selected.mathematical->numericalDemands && !attempted("form-original"));
        const bool selectedOriginal = !rejectsNumerical ||
            (selected.status == AnalysisStatus::Ready && !attempted("form-expanded"));
        if (!dispatchedFirst || !selectedExpanded || !selectedOriginal) {
            return function.emitError("dispatcher did not preserve first successful exact form");
        }
    }
    const auto root = session.analyzeFiniteExpansion({0});
    const bool sameProducer = root.mathematical &&
        root.mathematical->finiteGuardedDemands == finite.mathematical->finiteGuardedDemands &&
        root.mathematical->normalizedInput == owner;
    const bool external = function->hasAttr("test.normalization_external");
    const bool promoted = sameProducer != external && (external || session.finiteExpansionPreflights() == 1);
    if (!promoted) {
        return function.emitError("root promotion ignored original scope or rebuilt the same finite producer");
    }

    const uint64_t expectedNormalizations = external ? 2 : 1;
    const bool shared = session.normalizationConstructions() == expectedNormalizations &&
        (!numerical.mathematical || numerical.mathematical->normalizedInput == owner) &&
        session.analyzeFiniteExpansion({id}).mathematical == finite.mathematical &&
        session.analyzeNumericalRegion({id}).mathematical == numerical.mathematical &&
        !session.constructionCounts().logicalPreparations && !session.constructionCounts().allocationExports;
    if (!shared) { return function.emitError("mathematical adapters rebuilt normalization or constructed exports"); }
    SmallVector<int64_t> coordinates;
    for (const auto& normalized : owner->nodes) {
        const bool extract = normalized.original &&
            normalized.original->getName().getStringRef() == "pto.textract";
        if (!extract) { continue; }
        for (auto fixed : normalized.fixedCoordinates) {
            const bool inner = fixed.loop.getOperation() != node->anchor;
            if (inner) { coordinates.push_back(fixed.induction); }
        }
    }
    const auto expected = function->getAttrOfType<DenseI64ArrayAttr>("test.normalization_coordinates");
    const bool mapped = expected && ArrayRef<int64_t>(coordinates) == expected.asArrayRef() &&
        (owner->expandedLoops || function->hasAttr("test.normalization_compact")) &&
        owner->result.state == RecognitionState::Applicable;
    if (!mapped) { return function.emitError("normalized visits lost zero/one/non-unit original coordinates"); }
    const auto& index = *owner->index;
    const auto& input = *owner->input;
    auto malformed = owner->context;
    malformed.roots = {malformed.root, malformed.root};
    auto duplicate = normalizeSmallCountControl(malformed, index, input);
    malformed.roots = {malformed.root->getNextNode()};
    auto displaced = normalizeSmallCountControl(malformed, index, input);
    malformed.function = {}; malformed.roots = {malformed.root, malformed.root};
    auto missingFunction = normalizeSmallCountControl(malformed, index, input);
    const bool invalidRoots = missingFunction->result.state != RecognitionState::Applicable &&
        missingFunction->nodes.empty() && duplicate->result.state != RecognitionState::Applicable &&
        displaced->result.state != RecognitionState::Applicable && duplicate->nodes.empty() && displaced->nodes.empty();
    if (!invalidRoots) { return function.emitError("normalization accepted malformed root selections"); }
    FiniteExpansionLimits small; small.visits = 1; small.payloads = 1;
    auto compact = normalizeSmallCountControl(owner->context, index, input, small);
    const bool linear = compact->result.state == RecognitionState::Applicable &&
        compact->expandedLoops == 0 && compact->planningVisits == compact->nodes.size() &&
        owner->planningVisits == compact->planningVisits;
    if (!linear) { return function.emitError("optional expansion did not retain a linearly planned compact input"); }
    uint64_t compactPayloads = 0, compactEffects = 0;
    for (const auto& source : compact->nodes) {
        for (auto* phase : index.phasesFor(source.original)) {
            ++compactPayloads;
            compactEffects += input.accesses().effectsFor(phase).size();
        }
    }
    if (compact->payloads != compactPayloads || compact->effects != compactEffects) {
        return function.emitError("compact fallback lost original payload/effect counts");
    }
    const auto direct = preflightFiniteExpansion(owner->context, index, input, {}, compact);
    if (direct.result.state != RecognitionState::Applicable) {
        return function.emitError("compact fallback lost finite enumeration or dead-arm pruning");
    }
    FiniteExpansionLimits exact;
    exact.visits = direct.visits.size();
    auto boundary = preflightFiniteExpansion(owner->context, index, input, exact, owner);
    --exact.visits;
    auto exhausted = preflightFiniteExpansion(owner->context, index, input, exact, owner);
    const bool bounded = boundary.result.state == RecognitionState::Applicable &&
        exhausted.result.state != RecognitionState::Applicable;
    if (!bounded) { return function.emitError("supplied input bypassed finite visit boundaries"); }
    if (!rejectsNumerical) {
        NumericTemplateLimits tiny; tiny.visits = 1;
        const auto rejected = preflightNumericTemplate(cast<scf::ForOp>(node->anchor), index, input,
            tiny, true, {}, {}, owner);
        if (rejected.form.result.state == RecognitionState::Applicable) {
            return function.emitError("supplied input bypassed numerical visit limits");
        }
        if (function->hasAttr("test.normalization_deep")) {
            tiny = {}; tiny.depth = 1;
            const auto shallow = preflightNumericTemplate(cast<scf::ForOp>(node->anchor), index, input,
                tiny, true, {}, {}, owner);
            if (shallow.form.result.state == RecognitionState::Applicable) {
                return function.emitError("supplied input bypassed numerical depth limits");
            }
        }
    }
    AnalysisRequest queries; queries.region = id; queries.needs.queries = true;
    const auto exported = session.analyzeFiniteExpansion(queries);
    const bool retained = exported.mathematical == finite.mathematical &&
        session.normalizationConstructions() == expectedNormalizations;
    if (!retained) { return function.emitError("stronger exports replaced the normalized alternative"); }
    const auto alternate = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    const bool reinitialized = succeeded(session.initialize(alternate)) && !session.normalizationConstructions();
    if (!reinitialized) { return failure(); }
    const auto isolated = session.analyzeFiniteExpansion({id});
    const bool independent = isolated.mathematical && isolated.mathematical->normalizedInput != owner &&
        session.normalizationConstructions() == 1 && owner->result.state == RecognitionState::Applicable;
    std::string unchangedIR;
    llvm::raw_string_ostream unchangedStream(unchangedIR);
    function.print(unchangedStream);
    if (!independent || unchangedIR != originalIR) {
        return function.emitError("normalized context isolation failed");
    }
    llvm::outs() << "normalized-control-session: shared-input numerical-to-finite "
                 << "original-coordinates isolated-context\n";
    return success();
}

LogicalResult checkExactFormFastPath(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    AnalysisCostEstimate known, unknown, tied;
    known.work = 1; known.representation = 100;
    unknown.representation = 0;
    tied = known;
    const bool ordered = estimatedCostLess(known, unknown) && !estimatedCostLess(unknown, known) &&
        !estimatedCostLess(known, tied) && !estimatedCostLess(tied, known) &&
        !estimatedAdd(UINT64_MAX, 1) && !estimatedMultiply(UINT64_MAX, 2);
    if (!ordered) { return function.emitError("unknown/overflow work or stable cost ties changed"); }
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    const auto demands = session.minimumDemands();
    AnalysisNeeds needs; needs.queries = true;
    const auto queries = session.minimumDemands(0, needs);
    const bool explicitOnly = demands.status == AnalysisStatus::Ready && demands.mathematical &&
        demands.mathematical->explicitDemands && queries.mathematical == demands.mathematical &&
        !session.normalizationConstructions() && !session.arithmeticGeneratorConstructions() &&
        !session.numericTemplatePreflights() && !session.constructionCounts().logicalPreparations;
    if (!explicitOnly) { return function.emitError("straight-line analysis left the explicit fast path"); }
    const auto certifications = session.certifyRegions();
    const bool lazyCertification = !certifications.empty() && !session.result()->arithmetic &&
        session.result()->arithmeticContracts.empty() && !session.arithmeticGeneratorConstructions();
    if (!lazyCertification) { return function.emitError("certification eagerly constructed arithmetic forms"); }
    llvm::outs() << "exact-form-fast-path: explicit no-normalization no-arithmetic\n";
    return success();
}

LogicalResult checkNumericalRequestOrder(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    if (function->hasAttr("test.numerical_existing")) { return success(); }
    FrontierAnalysis warm(function), fresh(function);
    const bool initialized = succeeded(warm.initialize(policy)) && succeeded(fresh.initialize(policy));
    if (!initialized) { return failure(); }
    auto node = llvm::find_if(warm.result()->nodes, [&](const auto& value) {
        return value.kind == StructureKind::Loop && value.anchor->getParentOp() == function;
    });
    if (node == warm.result()->nodes.end()) { return failure(); }
    const auto id = static_cast<std::size_t>(node - warm.result()->nodes.begin());
    const auto retained = warm.analyzeNumericalRegion({id});
    AnalysisRequest request{id};
    request.needs.queries = request.needs.selectors = true;
    auto first = fresh.analyze(request), second = warm.analyze(request);
    const auto firstExports = first.regionalExports ? first.regionalExports :
        (first.mathematical ? first.mathematical->regionalDemands : nullptr);
    const auto secondExports = second.regionalExports ? second.regionalExports :
        (second.mathematical ? second.mathematical->regionalDemands : nullptr);
    const bool ready = first.status == AnalysisStatus::Ready && second.status == AnalysisStatus::Ready &&
        firstExports && secondExports && first.available.queries && first.available.selectors &&
        second.available.queries && second.available.selectors && retained.mathematical &&
        warm.analyzeNumericalRegion({id}).mathematical == retained.mathematical &&
        firstExports->anchors.size() == secondExports->anchors.size() &&
        warm.numericalRegionConstructions() == 1 && fresh.numericalRegionConstructions() == 1;
    if (!ready) { return function.emitError("numerical export adapter depends on request history"); }
    auto evaluate = [&](const RegionalAnalysis& region, RegionExpressions::Id expression, uint64_t upper) {
        SmallVector<std::pair<RegionExpressions::Id, RegionExpressions::Id>> bindings;
        for (auto input : region.expressions->referencedInputs(expression)) {
            const bool supplied = function.getNumArguments() == 1 && input.second == function.getArgument(0);
            if (!supplied) {
                return std::optional<uint64_t>{};
            }
            bindings.push_back({input.first, region.expressions->constant(upper)});
        }
        RegionExpressions::Substitution substitution(bindings);
        return region.expressions->constantValue(region.expressions->substitute(expression, substitution));
    };
    const auto& a = *firstExports;
    const auto& b = *secondExports;
    auto event = [](const RegionalAnalysis& region, uint32_t type, uint64_t ordinal, PeriodicEventKind kind) {
        RegionalEvent result{type, region.expressions->constant(ordinal), kind};
        if (!region.outerLoops.empty()) {
            result.visits.assign(region.outerLoops[type].size(), region.expressions->constant(0));
        }
        return result;
    };
    auto sameSelectors = [&](const std::vector<RegionalSelector>& x, const std::vector<RegionalSelector>& y,
                             uint64_t upper) {
        const bool sameSize = x.size() == y.size();
        if (!sameSize) { return false; }
        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto xp = evaluate(a, x[i].present, upper), yp = evaluate(b, y[i].present, upper);
            const bool present = xp && yp && xp == yp;
            if (!present) { return false; }
            if (!*xp) { continue; }
            const auto xo = evaluate(a, x[i].event.ordinal, upper);
            const auto yo = evaluate(b, y[i].event.ordinal, upper);
            const bool same = xo && yo && xo == yo && x[i].event.type == y[i].event.type &&
                x[i].event.kind == y[i].event.kind &&
                x[i].event.visits.size() == y[i].event.visits.size();
            if (!same) { return false; }
            for (std::size_t j = 0; j < x[i].event.visits.size(); ++j) {
                const auto xv = evaluate(a, x[i].event.visits[j], upper);
                const auto yv = evaluate(b, y[i].event.visits[j], upper);
                const bool sameVisit = xv && yv && xv == yv;
                if (!sameVisit) {
                    return false;
                }
            }
        }
        return true;
    };
    for (uint64_t upper : {0, 1, 3, 8}) {
        const bool sameCells = a.storageBoundary.size() == b.storageBoundary.size();
        if (!sameCells) { return failure(); }
        for (std::size_t i = 0; i < a.storageBoundary.size(); ++i) {
            const auto& x = a.storageBoundary[i];
            const auto& y = b.storageBoundary[i];
            const bool sameCell = sameStorageDomain(x.cell, y.cell) && x.cell.begin == y.cell.begin &&
                x.cell.end == y.cell.end;
            const bool same = sameCell && sameSelectors(x.firstWriters, y.firstWriters, upper) &&
                sameSelectors(x.lastWriters, y.lastWriters, upper) &&
                x.firstReaders.size() == y.firstReaders.size() && x.lastReaders.size() == y.lastReaders.size();
            if (!same) { return function.emitError("numerical storage selector adapter changed cell extrema"); }
            auto readersEqual = [&](const auto& left, const auto& right) {
                for (const auto& readers : left) {
                    auto found = right.find(readers.first);
                    const bool equal = found != right.end() && sameSelectors(readers.second, found->second, upper);
                    if (!equal) { return false; }
                }
                return true;
            };
            const bool sameReaders = readersEqual(x.firstReaders, y.firstReaders) &&
                readersEqual(x.lastReaders, y.lastReaders);
            if (!sameReaders) {
                return function.emitError("numerical storage selector adapter changed pipe extrema");
            }
        }
        for (uint32_t source = 0; source < a.anchors.size(); ++source) {
            for (uint32_t target = 0; target < a.anchors.size(); ++target) {
                for (uint64_t ordinal : {0, 1, 2}) {
                    auto left = regionalReachability(a, event(a, source, 0, PeriodicEventKind::Completion),
                        event(a, target, ordinal, PeriodicEventKind::Start));
                    auto right = regionalReachability(b, event(b, source, 0, PeriodicEventKind::Completion),
                        event(b, target, ordinal, PeriodicEventKind::Start));
                    const bool equal = left && right && evaluate(a, *left, upper) == evaluate(b, *right, upper) &&
                        evaluate(a, *left, upper).has_value();
                    if (!equal) { return function.emitError("numerical query adapter changed exact event answers"); }
                }
            }
        }
    }
    request.needs.synchronization = true;
    const auto logicalFirst = fresh.analyze(request), logicalSecond = warm.analyze(request);
    const bool logical = logicalFirst.status == AnalysisStatus::Ready &&
        logicalSecond.status == AnalysisStatus::Ready &&
        logicalFirst.mathematical == first.mathematical && logicalSecond.mathematical == second.mathematical &&
        warm.numericalRegionConstructions() == 1 && fresh.numericalRegionConstructions() == 1;
    if (!logical) { return function.emitError("numerical logical adapter depends on request history"); }
    if (!function->hasAttr("test.numerical_external")) {
        request.region = 0;
        const auto rootFirst = fresh.analyze(request), rootSecond = warm.analyze(request);
        const bool rootReady = rootFirst.status == AnalysisStatus::Ready &&
            rootSecond.status == AnalysisStatus::Ready &&
            warm.numericalRegionConstructions() == 1 && fresh.numericalRegionConstructions() == 1;
        if (!rootReady) { return function.emitError("numerical root/child export adapters disagree"); }
    }
    llvm::outs() << "numerical-request-order: fresh-and-warm exact-queries selectors logical-root-and-child\n";
    return success();
}

LogicalResult checkFixedCoordinateRelations(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    AnalysisRequest request{0}; request.needs.queries = true; request.needs.selectors = true;
    const auto result = session.analyzeFiniteExpansion(request);
    if (!result.mathematical || !result.regionalExports || !result.available.selectors) { return failure(); }
    const auto& child = *result.regionalExports;
    const auto owner = result.mathematical->normalizedInput;
    const bool validOwner = owner && child.anchors.size() == 4;
    if (!validOwner) { return failure(); }
    std::string error;
    auto adapter = requestCallbackRegionalRelations(child, function, *owner->input, *owner->index, error);
    if (failed(adapter)) { return function.emitError(error); }
    auto data = (*adapter)->lower((*adapter)->parameters(), error);
    const bool validData = succeeded(data) && data->sites.size() == child.anchors.size();
    if (!validData) { return function.emitError(error); }
    for (auto [i, site] : llvm::enumerate(data->sites)) {
        const bool validCoordinates = site.loops.empty() && site.fixedCoordinates.size() == 1 &&
            site.fixedCoordinates.front().induction == child.anchors[i].coordinates.front().induction;
        if (!validCoordinates) {
            return function.emitError("fixed coordinate became a free relation dimension");
        }
    }
    const ArithmeticRelationKey same{{0, ArithmeticEvent::Completion, {}}, {1, ArithmeticEvent::Start, {}}, {}};
    const ArithmeticRelationKey backward{{2, ArithmeticEvent::Completion, {}}, {1, ArithmeticEvent::Start, {}}, {}};
    auto has = [&](const auto& key) {
        auto found = data->analysis.nativeOrder.find(key);
        return found != data->analysis.nativeOrder.end() && !found->second.empty();
    };
    const bool validNative = has(same) && !has(backward);
    if (!validNative) { return function.emitError("fixed-visit native dependency mapping differs"); }
    auto composed = composeSymbolicRegionalSequence({child}, function, *owner->input, *owner->index, error);
    const bool validComposition = succeeded(composed) && composed->anchors.size() == child.anchors.size();
    if (!validComposition) { return function.emitError(error); }
    for (unsigned i = 0; i < child.anchors.size(); ++i) {
        if (composed->anchors[i].phase != child.anchors[i].phase ||
            composed->anchors[i].coordinates.front().induction != child.anchors[i].coordinates.front().induction) {
            return function.emitError("relation round trip lost fixed occurrence identity");
        }
    }
    auto duplicate = child; duplicate.anchors[2].coordinates = duplicate.anchors[0].coordinates;
    const bool duplicateAccepted = succeeded(requestCallbackRegionalRelations(
        duplicate, function, *owner->input, *owner->index, error));
    const bool interleavedAccepted = succeeded(composeSymbolicRegionalSequence(
        {child, child}, function, *owner->input, *owner->index, error));
    if (duplicateAccepted || interleavedAccepted) {
        return function.emitError("ambiguous or interleaved original occurrence partition was accepted");
    }
    std::string before;
    llvm::raw_string_ostream beforeStream(before); function.print(beforeStream);
    const bool emitted = composed->prepare && succeeded(composed->prepare());
    const bool emittedWithVisits = composed->prepareWithVisits && succeeded(composed->prepareWithVisits({}));
    std::string after;
    llvm::raw_string_ostream afterStream(after); function.print(afterStream);
    if (emitted || emittedWithVisits || before != after || composed->capabilities.endpointRecipes) {
        return function.emitError("fixed-occurrence adapter emitted unavailable endpoint recipes or changed IR");
    }
    const auto constructions = session.constructionCounts().mathematicalAttempts;
    const auto repeated = session.analyzeFiniteExpansion(request);
    if (repeated.mathematical != result.mathematical ||
        session.constructionCounts().mathematicalAttempts != constructions) { return failure(); }
    llvm::outs() << "fixed-coordinate-relations: distinct-visits scalar-mapping retained-identities no-reanalysis\n";
    return success();
}

LogicalResult checkUniformRelationCrossings(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    const auto& input = *session.input();
    const auto phases = input.instructions();
    const bool expectedPhases = phases.size() == 3;
    if (!expectedPhases) { return failure(); }
    auto present = IntegerSystem::create(0, {});
    if (failed(present)) { return failure(); }
    auto leaf = [&](const pto::CompoundInstanceElement* phase) {
        RegionalRelationData data;
        data.input = &input; data.context = {function, function}; data.sites.push_back({phase, {}, {}, {}});
        data.analysis.period = data.selectors.period = 1; data.analysis.pipeCount = 1;
        data.analysis.exactMinimum = data.completeRequiredOrder = true;
        data.occurrences.push_back({0, {}, {}, *present});
        for (auto a : {ArithmeticEvent::Start, ArithmeticEvent::Completion}) {
            for (auto b : {ArithmeticEvent::Start, ArithmeticEvent::Completion}) {
                if (a == ArithmeticEvent::Completion && b == ArithmeticEvent::Start) { continue; }
                ArithmeticRelationKey key{{0, a, {}}, {0, b, {}}, {}};
                data.analysis.nativeOrder[key] = {*present};
                if (a != b) { data.analysis.requiredOrder[key] = {*present}; }
            }
        }
        return data;
    };
    auto first = leaf(phases[0]), middle = leaf(phases[1]), right = leaf(phases[2]);
    auto byteDomain = IntegerSystem::create(1,
        {{{BoundInteger(-1)}, BoundInteger(0)}, {{BoundInteger(1)}, BoundInteger(31)}});
    if (failed(byteDomain)) { return failure(); }
    auto addByteBoundary = [&](RegionalRelationData& data, ArithmeticBoundaryKind kind) {
        ArithmeticBoundarySelector boundary;
        boundary.kind = kind; boundary.storageSpace = pto::AddressSpace::VEC;
        if (kind == ArithmeticBoundaryKind::FirstReaderBeforeWrite) {
            boundary.pipe = static_cast<uint32_t>(phases[2]->kPipeValue);
        }
        boundary.selector.inputDimensions = 1;
        boundary.selector.pieces.push_back({*byteDomain, {0}, {}, 0, {}});
        data.selectors.boundaries.push_back(std::move(boundary));
        data.selectors.support.push_back({pto::AddressSpace::VEC, {}, 0, {}, *byteDomain});
    };
    addByteBoundary(first, ArithmeticBoundaryKind::LastWriter);
    addByteBoundary(right, ArithmeticBoundaryKind::FirstReaderBeforeWrite);
    std::string error;
    auto left = composeRegionalRelationData(first, middle, {function, function}, error);
    const bool readOnlyMerge = succeeded(left) && left->analysis.minimumDemands.empty();
    if (!readOnlyMerge) { return function.emitError(error); }
    auto combined = composeRegionalRelationData(*left, right, {function, function}, error);
    if (failed(combined)) { return function.emitError(error); }
    const bool alias = policy == pto::GMAliasPolicy::MayAlias;
    const ArithmeticRelationKey expected{{alias ? 1u : 0u, ArithmeticEvent::Completion, {}},
        {2, ArithmeticEvent::Start, {}}, {}};
    const auto found = combined->analysis.minimumDemands.find(expected);
    const bool crossing = found != combined->analysis.minimumDemands.end() && !found->second.empty();
    const bool expectedDemands = crossing && combined->analysis.minimumDemands.size() == 1;
    if (!expectedDemands) {
        return function.emitError("uniform modeled GM crossing differs from the independent alias-policy expectation");
    }
    auto reversed = composeRegionalRelationData(right, *left, {function, function}, error);
    if (succeeded(reversed)) { return function.emitError("uniform adapter accepted reversed invocation order"); }
    llvm::outs() << "uniform-relation-crossings: shared-alias-policy exact-crossing complete-reduction\n";
    return success();
}

namespace {
mlir::LogicalResult checkDeferredSlices(mlir::func::FuncOp function,
    mlir::pto::frontiersynch::FrontierAnalysis& session)
{
    using namespace mlir::pto::frontiersynch;
    PhaseIndex index;
    if (failed(index.build(function, *session.input()))) { return failure(); }
    auto loop = *function.getOps<scf::ForOp>().begin();
    auto recognized = recognizeGuardedRotating(loop, index, *session.input(), session.input()->accesses());
    auto arena = std::make_shared<RegionExpressions>();
    auto analysis = analyzeGuardedRotating(loop, *session.input(), recognized, index, arena);
    std::string error;
    auto first = guardedRotatingRegionalResult(function, *session.input(), analysis, error,
        PeriodicSlice{arena->constant(0), arena->constant(2)});
    auto last = guardedRotatingRegionalResult(function, *session.input(), analysis, error,
        PeriodicSlice{arena->constant(2), arena->constant(4)});
    const bool exported = succeeded(first) && succeeded(last) && !first->deferredAccessBoundary.empty();
    if (!exported) { return function.emitError("disjoint writer slice export failed: " + error); }
    auto accepted = composeRegionalSequence(function, arena, {*first, *last}, true, false);
    if (!accepted.error.empty()) { return function.emitError(accepted.error); }
    auto overlap = composeRegionalSequence(function, arena, {*first, *first}, true, false);
    const bool refused = overlap.error.find("deferred sibling access") != std::string::npos;
    if (!refused) { return function.emitError("overlapping discharged writer visits were accepted"); }
    llvm::outs() << "deferred-writer-slices: disjoint-visits accepted overlapping-visits refused\n";
    return success();
}
mlir::LogicalResult checkUniformCircuit(mlir::func::FuncOp function,
    mlir::pto::frontiersynch::FrontierAnalysis& session)
{
    using namespace mlir::pto::frontiersynch;
    std::vector<RegionalAnalysis> children;
    for (std::size_t id = 0; id < session.result()->nodes.size(); ++id) {
        const auto& node = session.result()->nodes[id];
        const bool directLoop = node.kind == StructureKind::Loop && node.anchor->getParentOp() == function;
        if (!directLoop) { continue; }
        AnalysisRequest request{id}; request.needs.queries = request.needs.selectors = true;
        auto result = session.analyze(request);
        auto view = result.regionalExports ? result.regionalExports :
            (result.mathematical ? result.mathematical->regionalDemands : nullptr);
        if (result.status != AnalysisStatus::Ready || !view || view->storageBoundary.empty()) {
            return function.emitError("uniform circuit fixture lacks a finite source interface");
        }
        auto symbolic = *view;
        const auto finite = *view;
        symbolic.storageBoundary.clear(); symbolic.symbolicStorageEffects.clear();
        symbolic.relations.reset(); symbolic.arithmeticRelations.reset(); symbolic.numerical.reset();
        auto certificate = std::make_shared<RegionalSymbolicStorageCertificate>();
        certificate->expressions = view->expressions; certificate->accessModel = view->accessModel;
        certificate->gmAliasPolicy = view->gmAliasPolicy;
        for (auto& access : symbolic.accessBoundary) {
            access.representedByCells = false;
            symbolic.symbolicStorageEffects.push_back(access.effect);
        }
        for (const auto& boundary : finite.storageBoundary) {
            RegionalStorageFamily family; family.space = boundary.cell.space; family.base = boundary.cell.base;
            for (auto effect : symbolic.symbolicStorageEffects) {
                const auto& modeled = finite.accessModel->effects()[effect];
                if (modeled.memory && modeled.memory->scope == family.space) { family.effects.push_back(effect); }
            }
            family.membership = [arena = finite.expressions, cell = boundary.cell](RegionalByteAddress address)
                -> std::optional<Expr> {
                if (address.space != cell.space || address.base != cell.base) { return arena->boolean(false); }
                return arena->land(arena->le(arena->constant(cell.begin), address.offset),
                    arena->lt(address.offset, arena->constant(cell.end)));
            };
            certificate->families.push_back(std::move(family));
        }
        symbolic.symbolicStorage = std::move(certificate);
        symbolic.storageSelectors = [finite](RegionalByteAddress address) -> std::optional<RegionalStorageSelectors> {
            if (finite.storageSelectors) { return finite.storageSelectors(address); }
            auto& e = *finite.expressions;
            RegionalStorageSelectors result;
            auto append = [&](auto& out, const auto& values, Expr guard) {
                for (auto value : values) { value.present = e.land(value.present, guard); out.push_back(value); }
            };
            for (const auto& boundary : finite.storageBoundary) {
                const auto& cell = boundary.cell;
                if (address.space != cell.space || address.base != cell.base) { continue; }
                const auto guard = e.land(e.le(e.constant(cell.begin), address.offset),
                    e.lt(address.offset, e.constant(cell.end)));
                append(result.firstWriters, boundary.firstWriters, guard);
                append(result.lastWriters, boundary.lastWriters, guard);
                for (const auto& [pipe, values] : boundary.firstReaders) {
                    append(result.firstReaders[pipe], values, guard);
                }
                for (const auto& [pipe, values] : boundary.lastReaders) {
                    append(result.lastReaders[pipe], values, guard);
                }
            }
            return result;
        };
        children.push_back(std::move(symbolic));
    }
    const bool compatible = children.size() == 2 && children[0].expressions == children[1].expressions;
    if (!compatible) { return failure(); }
    auto arena = children[0].expressions;
    const bool checkMissing = !function->hasAttr("test.nonuniform_deferred") &&
        session.input()->memory().gmPolicy() == mlir::pto::GMAliasPolicy::MayAlias;
    if (checkMissing) {
        auto missing = children;
        missing.front().deferredAccessBoundary.clear();
        auto unavailable = composeRegionalSequence(function, arena, missing, true, false);
        if (unavailable.error.empty()) {
            return function.emitError("uniform crossing accepted missing deferred extrema");
        }
    }
    auto composed = composeRegionalSequence(function, arena, children, true, false);
    if (function->hasAttr("test.nonuniform_deferred")) {
        if (composed.error.empty()) { return function.emitError("nonuniform deferred overlap lost its obligation"); }
        return success();
    }
    const bool circuits = composed.error.empty() && composed.state && !composed.state->relationalResult &&
        composed.state->finiteCrossingPairs.empty();
    if (!circuits) {
        return function.emitError("uniform circuit composition used byte projection or relational lowering: " +
            composed.error);
    }
    auto grouped = composeRegionalSequence(function, arena, {children.front()}, true, false);
    if (!grouped.error.empty()) { return function.emitError(grouped.error); }
    auto nested = composeRegionalSequence(function, arena,
        {sequenceRegionalResult(grouped), children.back()}, true, false);
    const bool nestedCircuit = nested.error.empty() && nested.state && !nested.state->relationalResult &&
        nested.state->finiteCrossingPairs.empty();
    if (!nestedCircuit) { return function.emitError("nested uniform crossing lost its circuit interface"); }
    composed = std::move(nested);
    auto region = sequenceRegionalResult(composed);
    for (const auto& deferred : children.front().deferredAccessBoundary) {
        const bool retained = llvm::any_of(region.deferredAccessBoundary,
            [&](const auto& value) { return value.effect == deferred.effect; });
        if (!retained) { return function.emitError("uniform parent lost a deferred effect identity"); }
    }
    auto evaluate = [&](Expr expression, int64_t n, int64_t m) {
        SmallVector<std::pair<Expr, Expr>> bindings;
        for (auto input : arena->referencedInputs(expression)) {
            if (input.second == function.getArgument(0)) { bindings.emplace_back(input.first, arena->constant(n)); }
            else if (input.second == function.getArgument(1)) {
                bindings.emplace_back(input.first, arena->constant(m));
            }
            else { return std::optional<uint64_t>{}; }
        }
        RegionExpressions::Substitution substitution(bindings);
        return arena->constantValue(arena->substitute(expression, substitution));
    };
    for (int64_t n : {0, 1, 2, 3, 5, 8}) {
        for (int64_t m : {0, 3, 4, 5, 7, 9}) {
            const int64_t nx = n > 1 ? (n - 2) / 2 + 1 : 0, ny = m > 3 ? (m - 4) / 3 + 1 : 0;
            const bool active = session.input()->memory().gmPolicy() == mlir::pto::GMAliasPolicy::MayAlias && nx && ny;
            for (int64_t x = 0; x < 5; ++x) {
                for (int64_t y = 0; y < 5; ++y) {
                    auto required = region.reachability({0, arena->constant(x), PeriodicEventKind::Completion},
                        {1, arena->constant(y), PeriodicEventKind::Start});
                    const bool correctOrder = required &&
                        evaluate(*required, n, m) == std::optional<uint64_t>(active && x < nx && y < ny);
                    if (!correctOrder) {
                        return function.emitError("uniform circuit order differs from visit enumeration");
                    }
                    bool minimum = false;
                    for (const auto& edge : composed.state->crossings) {
                        const auto& source = composed.state->ports[edge.source];
                        const auto& target = composed.state->ports[edge.target];
                        const bool selected = evaluate(edge.guard, n, m) == 1 &&
                            evaluate(source.ordinal, n, m) == uint64_t(x) &&
                            evaluate(target.ordinal, n, m) == uint64_t(y);
                        if (selected) { minimum = true; }
                    }
                    if (minimum != (active && x == nx - 1 && y == 0)) {
                        return function.emitError("uniform circuit minimum differs from extremal bridge");
                    }
                }
            }
        }
    }
    return success();
}
} // namespace
LogicalResult checkDynamicUniformCrossings(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    using I = BoundInteger;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    if (function->hasAttr("test.disjoint_deferred")) { return checkDeferredSlices(function, session); }
    if (failed(checkUniformCircuit(function, session))) { return failure(); }
    if (function->hasAttr("test.nonuniform_deferred")) {
        llvm::outs() << "deferred-nonuniform-crossing: exact-adapter-required\n";
        return success();
    }
    const auto& input = *session.input();
    const auto phases = input.instructions();
    const bool twoPhases = phases.size() == 2;
    if (!twoPhases) { return failure(); }
    auto arena = std::make_shared<RegionExpressions>();
    auto make = [&](unsigned id) {
        RegionalRelationData data;
        auto loop = cast<scf::ForOp>(phases[id]->elementOp->getParentOp());
        data.input = &input; data.context = {function, loop}; data.sites.push_back({phases[id], {loop}, {}, {}});
        data.parameterValues = {function.getArgument(0), function.getArgument(1)};
        for (auto value : data.parameterValues) { data.parameters.push_back(arena->input(value)); }
        data.analysis.period = data.selectors.period = 1;
        data.analysis.parameterCount = data.selectors.parameterCount = 2;
        data.analysis.pipeCount = 1; data.analysis.exactMinimum = data.completeRequiredOrder = true;
        const int lower = id ? 3 : 1, step = id ? 3 : 2;
        std::vector<I> bound(3); bound[0] = I(1); bound[id + 1] = I(-1);
        auto domain = IntegerSystem::create(3, {{{I(-1), I(0), I(0)}, I(-lower)}, {bound, I(-1)}},
            {{{I(1), I(0), I(0)}, I(lower % step), I(step)}});
        data.occurrences.push_back({0, {0}, {0, 0}, *domain});
        auto pair = [&](int distance, bool equal) {
            auto a = domain->remap(4, {0, 2, 3}), b = domain->remap(4, {1, 2, 3});
            std::vector<IntegerConstraint> rows{{{I(1), I(-1), I(0), I(0)}, I(-distance)}};
            if (equal) { rows.push_back({{I(-1), I(1), I(0), I(0)}, I(distance)}); }
            auto order = IntegerSystem::create(4, rows);
            return *a->intersect(*b)->intersect(*order);
        };
        for (auto a : {ArithmeticEvent::Start, ArithmeticEvent::Completion}) {
            for (auto b : {ArithmeticEvent::Start, ArithmeticEvent::Completion}) {
                ArithmeticRelationKey key{{0, a, {0}}, {0, b, {0}}, {0, 0}};
                const bool completionToStart = a == ArithmeticEvent::Completion && b == ArithmeticEvent::Start;
                if (!completionToStart) { data.analysis.nativeOrder[key] = {pair(0, false)}; }
                data.analysis.requiredOrder[key] = {pair(a == b || completionToStart ? step : 0, false)};
                if (completionToStart) { data.analysis.minimumDemands[key] = {pair(step, true)}; }
            }
        }
        ArithmeticBoundarySelector extremum;
        extremum.kind = id ? ArithmeticBoundaryKind::FirstSite : ArithmeticBoundaryKind::LastSite;
        extremum.site = 0; extremum.selector.parameterCount = 2;
        if (!id) {
            for (int parity : {0, 1}) {
                auto present = IntegerSystem::create(2, {{{I(-1), I(0)}, I(-2-parity)}},
                    {{{I(1), I(0)}, I(parity), I(2)}});
                extremum.selector.pieces.push_back({*present, {}, {0, 0}, 0,
                    {{{{I(1), I(0)}, I(-1-parity)}, I(1), 0}}});
            }
        } else {
            auto present = IntegerSystem::create(2, {{{I(0), I(-1)}, I(-4)}});
            extremum.selector.pieces.push_back({*present, {}, {0, 0}, 0,
                {{{{I(0), I(0)}, I(3)}, I(1), 0}}});
        }
        data.selectors.boundaries.push_back(std::move(extremum));
        return data;
    };
    auto left = make(0), right = make(1);
    std::string error;
    auto composed = composeRegionalRelationData(left, right, {function, function}, error);
    if (failed(composed)) { return function.emitError(error); }
    auto fractionalLeft = left, fractionalRight = right;
    for (auto* child : {&fractionalLeft, &fractionalRight}) {
        for (auto& boundary : child->selectors.boundaries) {
            for (auto& piece : boundary.selector.pieces) {
                for (auto& output : piece.outputs) {
                    for (auto& coefficient : output.numerator.coefficients) { coefficient *= I(2); }
                    output.numerator.constant *= I(2); output.denominator *= I(2);
                }
            }
        }
    }
    auto fractional = composeRegionalRelationData(fractionalLeft, fractionalRight, {function, function}, error);
    if (failed(fractional)) { return function.emitError(error); }
    const ArithmeticRelationKey crossing{{0, ArithmeticEvent::Completion, {0}},
        {1, ArithmeticEvent::Start, {0}}, {0, 0}};
    auto contains = [](const IntegerSystem& system, ArrayRef<int64_t> point) {
        auto dot = [&](const auto& coefficients) {
            I total(0);
            for (unsigned i = 0; i < point.size(); ++i) { total += coefficients[i] * I(point[i]); }
            return total;
        };
        for (const auto& row : system.constraints()) {
            const bool within = dot(row.coefficients) <= row.bound;
            if (!within) { return false; }
        }
        for (const auto& row : system.congruences()) {
            const bool congruent = (dot(row.coefficients) - row.residue) % row.modulus == I(0);
            if (!congruent) { return false; }
        }
        return true;
    };
    auto member = [&](const auto& relation, ArrayRef<int64_t> point) {
        auto found = relation.find(crossing);
        return found != relation.end() && llvm::any_of(found->second,
            [&](const auto& system) { return contains(system, point); });
    };
    for (int64_t n : {-1, 0, 1, 2, 3, 5, 8}) {
        for (int64_t m : {-1, 0, 3, 4, 5, 7, 9}) {
            int64_t last = -1;
            for (int64_t i = 1; i < n; i += 2) { last = i; }
            for (int64_t x = 0; x < 10; ++x) {
                for (int64_t y = 0; y < 10; ++y) {
                    const bool active = policy == pto::GMAliasPolicy::MayAlias && last >= 0 && m > 3;
                    const bool minimum = active && x == last && y == 3;
                    const bool required = active && x >= 1 && x < n && (x-1)%2 == 0 &&
                        y >= 3 && y < m && (y-3)%3 == 0;
                    for (const auto* result : {&*composed, &*fractional}) {
                        const bool expectedMinimum = member(result->analysis.minimumDemands, {x, y, n, m}) == minimum;
                        const bool expectedClosure = member(result->analysis.requiredOrder, {x, y, n, m}) == required;
                        if (!expectedMinimum || !expectedClosure) {
                            return function.emitError("uniform extrema differ from independent visit enumeration");
                        }
                    }
                }
            }
        }
    }
    left.selectors.boundaries.clear();
    auto unsupported = composeRegionalRelationData(left, right, {function, function}, error);
    const bool alias = policy == pto::GMAliasPolicy::MayAlias;
    const bool unexpectedSupport = succeeded(unsupported) == alias;
    if (unexpectedSupport) {
        return function.emitError("uniform adapter omitted a needed extremum or rejected unrelated composition");
    }
    llvm::outs() << "dynamic-uniform-crossings: non-unit-coordinates zero-trips extrema exact-required-closure\n";
    return success();
}

namespace {
LogicalResult checkSpecializedExports(func::FuncOp function,
    const pto::frontiersynch::ArithmeticRegionalRelations& mathematics, int64_t lower)
{
    using namespace pto::frontiersynch;
    auto arena = std::make_shared<RegionExpressions>();
    const auto before = arena->size();
    const auto binding = [&](Value value) -> std::optional<RegionExpressions::Id> {
        if (value == function.getArgument(0)) { return arena->constant(lower); }
        if (value == function.getArgument(1)) { return arena->constant(lower + 5); }
        return arena->input(value);
    };
    std::string error;
    {
        RegionExpressions::Transaction transaction(*arena);
        auto exported = exportSpecializedArithmeticRegion(mathematics, arena, true, error, binding);
        if (failed(exported)) { return function.emitError(error); }
        auto present = exported->presence({0, arena->constant(2), PeriodicEventKind::Start});
        auto absent = exported->presence({0, arena->constant(3), PeriodicEventKind::Start});
        auto forward = exported->reachability({0, arena->constant(0), PeriodicEventKind::Completion},
            {1, arena->constant(2), PeriodicEventKind::Start});
        auto reverse = exported->reachability({1, arena->constant(2), PeriodicEventKind::Completion},
            {0, arena->constant(0), PeriodicEventKind::Start});
        const bool coordinates = present && absent && forward && reverse &&
            arena->constantValue(*present) == 1 && arena->constantValue(*absent) == 0 &&
            arena->constantValue(*forward) == 1 && arena->constantValue(*reverse) == 0;
        if (!coordinates) { return function.emitError("specialized nonzero origin changed event coordinates"); }
        bool selected = false;
        for (const auto& boundary : exported->storageBoundary) {
            for (const auto* side : {&boundary.firstWriters, &boundary.lastWriters}) {
                for (const auto& value : *side) {
                    const bool active = arena->constantValue(value.present) == 1;
                    if (!active) { continue; }
                    auto ordinal = arena->constantValue(value.event.ordinal);
                    const uint64_t expected = side == &boundary.firstWriters ? 0 : 2;
                    if (!ordinal || *ordinal != expected) {
                        return function.emitError("specialized selector lost original origin");
                    }
                    selected = true;
                }
            }
        }
        if (!selected) { return function.emitError("specialized writer selectors were absent"); }
    }
    const bool rolledBack = arena->size() == before;
    if (!rolledBack) { return function.emitError("specialized export survived outer rollback"); }
    auto badBinding = [&](Value value) -> std::optional<RegionExpressions::Id> {
        if (value == function.getArgument(0)) { return arena->constant(lower + 1); }
        return binding(value);
    };
    auto rejected = exportSpecializedArithmeticRegion(mathematics, arena, true, error, badBinding);
    const bool rejectedCleanly = failed(rejected) && arena->size() == before;
    if (!rejectedCleanly) { return function.emitError("contradictory specialization was exported or leaked IDs"); }
    for (int64_t upper : {lower, lower - 1}) {
        RegionExpressions::Transaction transaction(*arena);
        auto emptyBinding = [&](Value value) -> std::optional<RegionExpressions::Id> {
            if (value == function.getArgument(1)) { return arena->constant(upper); }
            return binding(value);
        };
        auto empty = exportSpecializedArithmeticRegion(mathematics, arena, true, error, emptyBinding);
        if (failed(empty)) { return function.emitError(error); }
        auto present = empty->presence({0, arena->constant(0), PeriodicEventKind::Start});
        const bool absent = present && arena->constantValue(*present) == 0;
        if (!absent) {
            return function.emitError("specialized zero or reversed trip domain was present");
        }
        for (const auto& boundary : empty->storageBoundary) {
            for (const auto* side : {&boundary.firstWriters, &boundary.lastWriters}) {
                for (const auto& value : *side) {
                    const bool inactive = arena->constantValue(value.present) == 0;
                    if (!inactive) {
                        return function.emitError("empty specialized domain exported an active writer");
                    }
                }
            }
        }
    }
    auto missingBinding = [&](Value value) -> std::optional<RegionExpressions::Id> {
        if (value == function.getArgument(0)) { return std::nullopt; }
        return binding(value);
    };
    auto missing = exportSpecializedArithmeticRegion(mathematics, arena, true, error, missingBinding);
    const bool missingRefused = failed(missing) && arena->size() == before;
    if (!missingRefused) { return function.emitError("missing specialization binding was exported or leaked IDs"); }
    auto exported = exportSpecializedArithmeticRegion(mathematics, arena, true, error, binding);
    if (failed(exported)) { return function.emitError(error); }
    auto ordinary = exportArithmeticRegion(mathematics, arena, true, error);
    auto prepared = prepareArithmeticRegion(mathematics, arena, {}, error);
    const bool ordinaryRefused = failed(ordinary) && failed(prepared);
    if (!ordinaryRefused) { return function.emitError("ordinary adapter accepted specialized mathematics"); }
    return success();
}
} // namespace
LogicalResult checkSpecializedArithmeticSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    auto loop = *function.getOps<scf::ForOp>().begin();
    const ArithmeticRegionContext context{function, loop};
    auto constants = [&](int64_t lower, std::optional<int64_t> upper = std::nullopt) -> ArithmeticEntryConstant {
        return [lo = function.getArgument(0), n = function.getArgument(1), lower, upper]
            (Value value) -> std::optional<int64_t> {
            if (value == lo) { return lower; }
            if (value == n) { return upper; }
            return std::nullopt;
        };
    };
    std::string error;
    auto first = session.specializedArithmeticDemands(context, constants(1), error);
    auto retry = session.specializedArithmeticDemands(context, constants(1), error);
    const bool reused = first && first == retry && session.specializedArithmeticConstructions() == 1 &&
        first->parameters.empty() && first->occurrences.empty() && first->selectors.boundaries.empty() &&
        first->inputOwner && first->indexOwner;
    if (!reused) { return function.emitError("specialized mathematics was not retained independently: " + error); }
    auto changed = session.specializedArithmeticDemands(context, constants(3), error);
    auto known = session.specializedArithmeticDemands(context, constants(1, 4), error);
    const bool isolated = changed && known && changed != first && known != first &&
        session.specializedArithmeticConstructions() == 3;
    if (!isolated) {
        return function.emitError("specialized constant or unknown binding reused incompatible mathematics");
    }
    const SmallVector<ArithmeticLimits> invalid{{8, 0, 1, 4096}};
    auto rejected = session.specializedArithmeticDemands(context, constants(1), error, invalid);
    auto repeated = session.specializedArithmeticDemands(context, constants(1), error, invalid);
    const bool scopedFailure = !rejected && !repeated && session.specializedArithmeticConstructions() == 4 &&
        session.specializedArithmeticDemands(context, constants(1), error) == first;
    if (!scopedFailure) { return function.emitError("specialized profile failure poisoned another request"); }
    if (failed(checkSpecializedExports(function, *first, 1))) { return failure(); }
    if (failed(checkSpecializedExports(function, *changed, 3))) { return failure(); }
    const auto other = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    if (failed(session.initialize(other))) { return failure(); }
    const bool emptyCache = session.specializedArithmeticConstructions() == 0;
    auto renewed = session.specializedArithmeticDemands(context, constants(1), error);
    const bool reset = emptyCache && renewed && renewed != first && renewed->inputOwner != first->inputOwner;
    if (!reset) { return function.emitError("specialized cache crossed an alias-context reset"); }
    if (failed(checkSpecializedExports(function, *first, 1))) { return failure(); }
    llvm::outs() << "specialized-arithmetic-session: observed-misses context-isolated nonzero-origins rollback-safe\n";
    return success();
}

LogicalResult checkFiniteArithmeticSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    auto node = llvm::find_if(session.result()->nodes,
        [](const auto& value) { return value.kind == StructureKind::Loop; });
    if (node == session.result()->nodes.end()) { return function.emitError("finite arithmetic loop absent"); }
    const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
    std::string error;
    auto first = session.finiteArithmeticRegion(id, error);
    const bool unavailable = function->hasAttr("test.finite_arithmetic_unavailable");
    const bool expected = succeeded(first) != unavailable && session.finiteArithmeticStorageProbes() == 1;
    if (!expected) {
        return function.emitError("finite arithmetic eligibility differs from source expectation: " + error);
    }
    const auto builds = session.arithmeticRegionConstructions();
    const auto queries = session.constructionCounts().arithmeticQueryBuilds;
    const auto selectors = session.constructionCounts().arithmeticSelectorBuilds;
    auto retry = session.finiteArithmeticRegion(id, error);
    const bool reused = succeeded(retry) == succeeded(first) && session.finiteArithmeticStorageProbes() == 1 &&
        session.arithmeticRegionConstructions() == builds &&
        session.constructionCounts().arithmeticQueryBuilds == queries &&
        session.constructionCounts().arithmeticSelectorBuilds == selectors;
    if (!reused) { return function.emitError("finite arithmetic retry rebuilt producer or exports"); }
    const bool deferredCleanly = !unavailable || (!error.empty() && !builds && !queries && !selectors);
    if (!deferredCleanly) { return function.emitError("negative finite probe constructed mathematics or exports"); }
    AnalysisRequest requested{id}; requested.needs.queries = requested.needs.selectors = true;
    auto symbolic = session.analyzeArithmeticRegional(requested);
    const bool extended = symbolic.status == AnalysisStatus::Ready && symbolic.mathematical &&
        session.arithmeticRegionConstructions() == 1 && session.finiteArithmeticStorageProbes() == 1;
    if (!extended) { return function.emitError("finite probe prevented a later exact arithmetic request"); }
    const bool sameSnapshot = unavailable ||
        (symbolic.regionalExports && first->arithmeticRelations == symbolic.regionalExports->arithmeticRelations);
    if (!sameSnapshot) { return function.emitError("finite request did not share canonical selector snapshot"); }
    if (succeeded(session.finiteArithmeticRegion(SIZE_MAX, error))) {
        return function.emitError("invalid finite region was accepted");
    }
    auto arena = std::make_shared<RegionExpressions>();
    const bool finiteStorage = hasFiniteArithmeticStorage(symbolic.mathematical->arithmeticRegionalDemands->program,
        *session.input(), arena, error);
    const bool probeRollback = finiteStorage != unavailable && !arena->size();
    if (!probeRollback) { return function.emitError("finite support probe published expression IDs"); }
    const auto other = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    const bool reset = succeeded(session.initialize(other)) && !session.finiteArithmeticStorageProbes();
    if (!reset) {
        return function.emitError("finite eligibility cache survived alias-context reset");
    }
    llvm::outs() << "finite-arithmetic-session: early-gate cached-producer cached-exports later-exact-retry\n";
    return success();
}

LogicalResult checkSpecializedNumericSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    scf::ForOp outer;
    function.walk([&](scf::ForOp loop) { if (loop->getParentOp() == function) { outer = loop; } });
    const auto geometry = [&](std::optional<int64_t> offset) -> TemplateGeometryConstant {
        return [&, offset](Value value) { return value == function.getArgument(0) ? offset : std::nullopt; };
    };
    const auto control = [&](std::optional<bool> choose) -> TemplateControlConstant {
        return [&, choose](Value value) { return value == function.getArgument(1) ? choose : std::nullopt; };
    };
    std::string error;
    if (function->hasAttr("test.specialized_numeric_outer_scope")) {
        function.walk([&](scf::ForOp loop) { if (loop->getParentOp() != function) { outer = loop; } });
        auto first = session.specializedNumericDemands(outer, geometry(0), control(true), error);
        if (!first) { return function.emitError("single-body original mathematics unavailable: " + error); }
        auto whole = normalizeSmallCountControl({function, function}, *first->indexOwner, *first->inputOwner);
        auto expanded = recognizeSpecializedNumericBody(outer, *first->indexOwner, *first->inputOwner,
            geometry(0), control(true), {}, whole);
        auto invalid = analyzeNumericBody(expanded, error);
        const bool refusedWord = !invalid && !error.empty();
        auto wrongBody = first->body;
        for (auto& payload : wrongBody.payloads) { payload.coordinates.push_back({outer, 0}); }
        auto invalidBody = analyzeNumericBody(wrongBody, error);
        const bool refusedBody = !invalidBody && error.find("single-body") != std::string::npos;
        auto incompatible = session.specializedNumericDemands(outer, geometry(0), control(true), error, {}, whole);
        const bool refusedContext = !incompatible && session.specializedNumericConstructions() == 1 &&
            first->body.payloads.size() == 2;
        if (!refusedWord || !refusedBody || !refusedContext) {
            return function.emitError("single-body scope refusal mismatch word=" + std::to_string(refusedWord) +
                " body=" + std::to_string(refusedBody) + " context=" + std::to_string(refusedContext) +
                " payloads=" + std::to_string(first->body.payloads.size()) +
                " builds=" + std::to_string(session.specializedNumericConstructions()));
        }
        llvm::outs() << "specialized-numeric-scope: expanded-enclosing-visits-refused\n";
        return success();
    }
    auto missing = session.specializedNumericDemands(outer, geometry({}), control(true), error);
    auto missingRetry = session.specializedNumericDemands(outer, geometry({}), control(true), error);
    const bool failedOnce = !missing && !missingRetry && !error.empty() &&
        session.specializedNumericConstructions() == 1;
    if (!failedOnce) { return function.emitError("specialized numeric failed binding was not cached"); }
    auto first = session.specializedNumericDemands(outer, geometry(0), control(true), error);
    auto retry = session.specializedNumericDemands(outer, geometry(0), control(true), error);
    const bool reused = first && retry == first && session.specializedNumericConstructions() == 2 &&
        first->body.payloads.size() == 4 && first->demands->phases.size() == 4;
    if (!reused) {
        return function.emitError("specialized numeric word/reduction mismatch: " + error + " builds=" +
            std::to_string(session.specializedNumericConstructions()) + " payloads=" +
            std::to_string(first ? first->body.payloads.size() : 0) + " phases=" +
            std::to_string(first ? first->demands->phases.size() : 0));
    }
    const bool coordinates = first->body.payloads[0].coordinates.back().induction == 1 &&
        first->body.payloads[2].coordinates.back().induction == 3;
    if (!coordinates) { return function.emitError("specialized numeric lost non-unit inner coordinates"); }
    auto changed = session.specializedNumericDemands(outer, geometry(1), control(true), error);
    auto absent = session.specializedNumericDemands(outer, geometry(0), control(false), error);
    auto unknown = session.specializedNumericDemands(outer, geometry(0), control({}), error);
    const bool isolated = changed && changed != first && absent && absent != first &&
        absent->body.payloads.empty() && absent->demands->occurrences.empty() && !unknown &&
        session.specializedNumericConstructions() == 5;
    if (!isolated) { return function.emitError("specialized numeric bindings reused incompatible mathematics"); }
    const bool shifted = first->body.atoms.size() == 1 && changed->body.atoms.size() == 1 &&
        first->body.atoms[0].begin == 0 && first->body.atoms[0].end == 4 &&
        changed->body.atoms[0].begin == 4 && changed->body.atoms[0].end == 8;
    if (!shifted) { return function.emitError("specialized geometry did not shift the source scalar byte range"); }
    auto arena = std::make_shared<RegionExpressions>();
    func::FuncOp foreign;
    for (auto candidate : function->getParentOfType<ModuleOp>().getOps<func::FuncOp>()) {
        if (candidate != function) { foreign = candidate; break; }
    }
    auto wrongFunction = exportNumericBody(foreign, *first->inputOwner, *first, arena, {outer}, error);
    auto wrongFrames = exportNumericBody(function, *first->inputOwner, *first, arena, {}, error);
    const bool rejectedContexts = failed(wrongFunction) && failed(wrongFrames) && !arena->size();
    if (!rejectedContexts) { return function.emitError("specialized numeric accepted foreign original cuts"); }
    {
        RegionExpressions::Transaction transaction(*arena);
        arena->div(arena->constant(1), arena->constant(0));
        const auto before = arena->size();
        auto poisoned = exportNumericBody(function, *first->inputOwner, *first, arena, {outer}, error);
        const bool rejected = failed(poisoned) && arena->size() == before;
        if (!rejected) { return function.emitError("specialized numeric exported a poisoned arena"); }
    }
    {
        RegionExpressions::Transaction transaction(*arena);
        auto exported = exportNumericBody(function, *first->inputOwner, *first, arena, {outer}, error);
        if (failed(exported)) { return function.emitError("specialized numeric export failed: " + error); }
        auto zero = arena->constant(0);
        for (uint32_t a = 0; a < first->body.payloads.size(); ++a) {
            for (uint32_t b = 0; b < first->body.payloads.size(); ++b) {
                for (auto ak : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                    for (auto bk : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                        auto expected = explicitEventPrecedes(*first->demands, {a, ak}, {b, bk});
                        auto actual = exported->reachability({a, zero, ak}, {b, zero, bk});
                        const bool equal = expected && actual && arena->constantValue(*actual) == (*expected ? 1 : 0);
                        if (!equal) { return function.emitError("specialized numeric exported a different closure"); }
                    }
                }
            }
        }
    }
    const bool rolledBack = arena->size() == 0;
    if (!rolledBack) { return function.emitError("specialized numeric export survived rollback"); }
    NumericTemplateLimits small; small.visits = 1;
    auto limited = session.specializedNumericDemands(outer, geometry(0), control(true), error, small);
    auto limitedRetry = session.specializedNumericDemands(outer, geometry(0), control(true), error, small);
    const bool limitIsolated = !limited && !limitedRetry && session.specializedNumericConstructions() == 6 &&
        session.specializedNumericDemands(outer, geometry(0), control(true), error) == first;
    if (!limitIsolated) { return function.emitError("specialized numeric limits changed cached eligibility"); }
    auto expanded = normalizeSmallCountControl({function, outer}, *first->indexOwner, *first->inputOwner);
    auto alternative = session.specializedNumericDemands(outer, geometry(0), control(true), error, {}, expanded);
    const bool representationIsolated = expanded != first->normalized && alternative && alternative != first &&
        alternative->body.payloads.size() == 4 &&
        alternative->body.payloads[0].coordinates.back().induction == 1 &&
        alternative->body.payloads[2].coordinates.back().induction == 3 &&
        session.specializedNumericConstructions() == 7 &&
        session.specializedNumericDemands(outer, geometry(0), control(true), error) == first;
    if (!representationIsolated) {
        return function.emitError("specialized numeric normalization identity was ignored");
    }
    const auto other = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    const bool reset = succeeded(session.initialize(other)) && !session.specializedNumericConstructions();
    if (!reset) { return function.emitError("specialized numeric cache survived alias reset"); }
    auto exported = exportNumericBody(function, *first->inputOwner, *first, arena, {outer}, error);
    if (failed(exported)) { return function.emitError("retained numeric mathematics lost input ownership"); }
    auto renewed = session.specializedNumericDemands(outer, geometry(0), control(true), error);
    const bool renewedOnce = renewed && renewed != first && session.specializedNumericConstructions() == 1;
    if (!renewedOnce) { return function.emitError("specialized numeric reset reused old-context mathematics"); }
    llvm::outs() << "specialized-numeric-session: observed-bindings cached-reduction rollback-safe\n";
    return success();
}

namespace {
LogicalResult checkGuardedSliceSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    scf::ForOp loop;
    Value condition;
    function.walk([&](scf::ForOp value) { loop = value; });
    loop.walk([&](scf::IfOp branch) {
        const bool directChild = branch->getParentOp() == loop;
        if (directChild) { condition = branch.getCondition(); }
    });
    GuardedRotatingSpecialization request; request.loop = loop;
    std::string error;
    auto missing = session.specializedGuardedDemands(request, error);
    auto retry = session.specializedGuardedDemands(request, error);
    const bool failedFormCached = missing && retry == missing && !error.empty() &&
        !session.specializedGuardedConstructions();
    if (!failedFormCached) { return function.emitError("unavailable guarded form was not retained"); }
    request.sliced = true; request.guards.push_back(condition);
    request.expressions = std::make_shared<RegionExpressions>();
    request.bindings[condition] = request.expressions->boolean(true);
    auto selected = session.specializedGuardedDemands(request, error);
    const bool specialized = selected && error.empty() && selected->demands.phases.size() == 3 &&
        session.specializedGuardedConstructions() == 1;
    if (!specialized) {
        if (selected) {
            for (const auto& diagnostic : selected->recognition.result.diagnostics) {
                error += " / " + recognitionName(diagnostic.issue).str();
            }
        }
        return function.emitError("constant-specialized guarded slice unavailable: " + error);
    }
    request.bindings[condition] = request.expressions->boolean(false);
    request.choices[condition] = false;
    auto empty = session.specializedGuardedDemands(request, error);
    const bool selectedEmpty = empty && error.empty() && empty->demands.phases.empty() &&
        session.specializedGuardedConstructions() == 2;
    if (!selectedEmpty) { return function.emitError("constant-specialized empty slice unavailable: " + error); }
    llvm::outs() << "specialized-guarded-slice: original-form-unavailable constant-specialization exact-empty-slice\n";
    return success();
}
} // namespace

LogicalResult checkSpecializedGuardedSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    if (function->hasAttr("test.specialized_guarded_slice")) {
        return checkGuardedSliceSession(function, policy);
    }
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    std::string before;
    llvm::raw_string_ostream rendered(before); function.print(rendered); rendered.flush();
    scf::ForOp loop;
    function.walk([&](scf::ForOp candidate) { loop = candidate; });
    auto g = function.getArgument(0), h = function.getArgument(1);
    std::string error;
    GuardedRotatingSpecialization request; request.loop = loop;
    auto generic = session.specializedGuardedDemands(request, error);
    const bool genericReady = generic && error.empty() && generic->demands.phases.size() == 3;
    if (!genericReady) { return function.emitError("guarded generic mathematics unavailable: " + error); }
    auto a = std::make_shared<RegionExpressions>(), b = std::make_shared<RegionExpressions>();
    const auto aRoot = a->boolean(true), bRoot = b->boolean(false);
    if (aRoot != bRoot) { return function.emitError("test requires equal IDs in unrelated arenas"); }
    request.bindings[g] = aRoot; request.expressions = a;
    auto yes = session.specializedGuardedDemands(request, error);
    request.bindings[g] = bRoot; request.expressions = b;
    auto no = session.specializedGuardedDemands(request, error);
    const bool isolated = yes && no && yes != no && error.empty() &&
        session.specializedGuardedConstructions() == 3 &&
        yes->demands.expressions->constantValue(yes->demands.payloads.front().presence) != 0 &&
        llvm::all_of(no->demands.payloads, [&](const auto& payload) {
            return no->demands.expressions->constantValue(payload.presence) == 0;
        });
    if (!isolated) { return function.emitError("guarded source arena IDs contaminated specialization"); }
    b->constant(123); request.bindings[g] = b->boolean(true);
    auto same = session.specializedGuardedDemands(request, error);
    if (same != yes) { return function.emitError("equal guarded circuits did not reuse mathematics"); }
    request.bindings[g] = b->input(h); request.bindings[h] = b->lnot(b->input(h));
    auto correlated = session.specializedGuardedDemands(request, error);
    if (!correlated || !error.empty()) { return function.emitError("correlated guards unavailable"); }
    request.sliced = true; request.choices[g] = true;
    const auto count = session.specializedGuardedConstructions();
    auto contradictory = session.specializedGuardedDemands(request, error);
    const bool refused = !contradictory && !error.empty() && session.specializedGuardedConstructions() == count;
    if (!refused) { return function.emitError("unproved guarded choice was accepted"); }
    request.choices[g] = false; request.bindings.clear();
    auto empty = session.specializedGuardedDemands(request, error);
    const bool emptyReady = empty && error.empty() && empty->demands.phases.empty();
    if (!emptyReady) { return function.emitError("empty guarded slice unavailable: " + error); }
    request = GuardedRotatingSpecialization(); request.loop = loop;
    auto target = std::make_shared<RegionExpressions>(); target->constant(999);
    const auto targetPrefix = target->size();
    const auto constructions = session.specializedGuardedConstructions();
    {
        RegionExpressions::Transaction rollback(*target);
        auto imported = session.instantiateGuardedDemands(request, target, error);
        const bool importedReady = imported && error.empty() && imported->demands.expressions == target &&
            imported->demands.periodic.expressions == target;
        if (!importedReady) { return function.emitError("guarded import unavailable: " + error); }
        auto equivalent = [&](const GuardedRotatingAnalysis& source, const GuardedRotatingAnalysis& copy) {
            for (uint32_t first = 0; first < source.phases.size(); ++first) {
                for (uint32_t second = 0; second < source.phases.size(); ++second) {
                    for (auto start : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                        for (auto end : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                            auto expected = source.periodic.eventThreshold({first, start}, {second, end});
                            auto actual = copy.periodic.eventThreshold({first, start}, {second, end});
                            if (!expected || !actual) { return false; }
                            for (bool gv : {false, true}) {
                                for (bool hv : {false, true}) {
                                    for (uint64_t offset : {0, 1, 2, 3}) {
                                        auto evaluate = [&](auto& arena, auto id) {
                                            RegionExpressions::Substitution values({
                                                {arena.input(g), arena.boolean(gv)},
                                                {arena.input(h), arena.boolean(hv)},
                                                {arena.input(function.getArgument(3)), arena.constant(offset)}});
                                            return arena.constantValue(arena.substitute(id, values));
                                        };
                                        auto left = evaluate(*source.expressions, expected->reachable);
                                        auto right = evaluate(*copy.expressions, actual->reachable);
                                        if (!left || right != left) { return false; }
                                        const bool sameDistance = !*left ||
                                            evaluate(*source.expressions, expected->distance) ==
                                            evaluate(*copy.expressions, actual->distance);
                                        if (!sameDistance) { return false; }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return true;
        };
        if (!equivalent(generic->demands, imported->demands)) {
            return function.emitError("guarded import changed all-event thresholds");
        }
    }
    const bool rolledBack = target->size() == targetPrefix;
    if (!rolledBack) { return function.emitError("guarded export transaction leaked nodes"); }
    target->constant(321); // Reuse rejected IDs before importing again.
    auto renewed = session.instantiateGuardedDemands(request, target, error);
    const bool retryReady = renewed && error.empty();
    if (!retryReady) { return function.emitError("guarded rollback retry failed"); }
    const auto privateSize = generic->demands.expressions->size();
    auto poisoned = std::make_shared<RegionExpressions>();
    auto disposable = importGuardedRotating(generic->demands, poisoned);
    if (failed(disposable)) { return function.emitError("guarded disposable import failed"); }
    poisoned->add(poisoned->boolean(true), poisoned->constant(1));
    auto& broken = *disposable;
    GuardedPeriodicEndpointInput endpoint{broken.loop, broken.expressions, broken.phases,
        broken.payloads, broken.generators, &broken.periodic};
    auto failedPlan = prepareGuardedPeriodicEndpoints(function, endpoint, error);
    const bool privateUnchanged = failed(failedPlan) && generic->demands.expressions->size() == privateSize &&
        generic->demands.expressions->error().empty() &&
        session.specializedGuardedDemands(request, error) == generic &&
        session.specializedGuardedConstructions() == constructions;
    if (!privateUnchanged) { return function.emitError("failed endpoint preparation contaminated guarded cache"); }
    auto outcome = session.minimumDemands();
    if (outcome.mathematical && outcome.mathematical->guardedRotatingDemands) {
        const auto size = generic->demands.expressions->size();
        auto plan = session.prepareLogical(outcome);
        if (failed(plan)) { return function.emitError("guarded root preparation did not reach allocation check"); }
        (void)session.attachAllocation(outcome, **plan);
        const bool isolatedExports = generic->demands.expressions->size() == size &&
            generic->demands.expressions->error().empty();
        if (!isolatedExports) { return function.emitError("guarded root exports modified private mathematics"); }
    } else { return function.emitError("guarded fixture did not select guarded root mathematics"); }
    const auto work = session.guardedExpressionImportWork();
    if (!work) { return function.emitError("guarded import work was not recorded"); }
    const auto other = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    const bool reset = succeeded(session.initialize(other)) && !session.specializedGuardedConstructions() &&
        !session.guardedExpressionImportWork();
    if (!reset) { return function.emitError("guarded cache survived context reset"); }
    {
        auto oldExport = importGuardedRotating(generic->demands, std::make_shared<RegionExpressions>());
        if (failed(oldExport)) { return function.emitError("retained guarded circuit import failed"); }
    }
    auto fresh = session.specializedGuardedDemands(request, error);
    const bool renewedContext = fresh && fresh != generic && error.empty() &&
        session.specializedGuardedConstructions() == 1;
    if (!renewedContext) { return function.emitError("guarded mathematical owner/reset failed"); }
    std::weak_ptr<const pto::SyncInput> weakInput = generic->inputOwner;
    std::weak_ptr<const PhaseIndex> weakIndex = generic->indexOwner;
    auto retainedExport = guardedRotatingRegionalResult(function, *generic->inputOwner, renewed->demands, error);
    if (failed(retainedExport)) { return function.emitError("retained guarded regional export failed: " + error); }
    generic.reset(); yes.reset(); no.reset(); same.reset(); correlated.reset(); empty.reset();
    renewed.reset(); target.reset(); outcome = AnalysisOutcome();
    auto presence = regionalPresence(*retainedExport, {0, retainedExport->expressions->constant(0),
        PeriodicEventKind::Start});
    const bool independentlyOwned = !weakInput.expired() && !weakIndex.expired() && presence.has_value();
    if (!independentlyOwned) {
        return function.emitError("guarded export did not independently retain borrowed input");
    }
    std::string after; llvm::raw_string_ostream finalStream(after); function.print(finalStream); finalStream.flush();
    if (before != after) { return function.emitError("guarded specialization mutated original IR"); }
    llvm::outs() << "specialized-guarded-session: private-circuits context-isolated rollback-safe retained-owner\n";
    return success();
}
