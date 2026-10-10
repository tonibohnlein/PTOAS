// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check shared normalized input across different mathematical adapters.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/NormalizedControl.h"
#include "../../lib/PTO/Transforms/FrontierSynch/NumericTemplatePlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FiniteExpansionPlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRelationsInternal.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
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
