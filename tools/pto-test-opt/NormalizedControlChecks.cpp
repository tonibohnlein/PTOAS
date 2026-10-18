// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check shared normalized input across different mathematical adapters.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateRegional.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Structured/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/Structured/GuardedPeriodicInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/NormalizedControl.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PhaseNormalization.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/NumericTemplatePlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/FiniteExpansionPlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalRelationsInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/BoundarySlices.h"
#include "../../lib/PTO/Transforms/FrontierSynch/WholeInvocationIdentity.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Diagnostics.h"
#include "NumericalRequestOracle.h"
#include "FiniteRegionalChecks.h"
#include "NumericalLogicalTraceChecks.h"
using namespace mlir;
LogicalResult checkCrossingCostSelection(func::FuncOp function, pto::GMAliasPolicy)
{
    using namespace pto::frontiersynch;
    auto loops = function.getOps<scf::ForOp>();
    if (loops.empty()) { return function.emitError("crossing fixture requires an occurrence loop"); }
    EstimatedCount callbackIndexEstimate;
    for (unsigned variant = 0; variant < 9; ++variant) {
        const bool poison = variant == 8;
        const bool existing = variant == 6;
        const bool bothUnavailable = variant == 7;
        const bool large = variant == 1 || variant == 3 || existing;
        const bool refuse = variant == 2 || variant == 3;
        const unsigned count = large ? 4 : 2;
        const unsigned visits = large ? 16 : ((variant == 2 || bothUnavailable || poison) ? 2 : 1);
        auto arena = std::make_shared<RegionExpressions>();
        auto state = std::make_shared<SequenceAnalysisState>(function, arena);
        std::vector<pto::CompoundInstanceElement> phases;
        phases.reserve(count);
        auto refused = std::make_shared<bool>(!refuse);
        auto memo = std::make_shared<std::optional<Expr>>();
        auto callbackCalls = std::make_shared<uint64_t>(0);
        for (unsigned child = 0; child < count; ++child) {
            phases.emplace_back(child, SmallVector<const pto::BaseMemInfo*>{},
                SmallVector<const pto::BaseMemInfo*>{}, pto::PipelineType::PIPE_V, function->getName());
            phases.back().elementOp = function;
            Child local;
            local.anchors.push_back({&phases.back(), {}, {}, {}});
            local.regional.anchors = local.anchors;
            local.regional.occurrenceLoops.push_back(*loops.begin());
            local.regional.expressions = arena;
            local.regional.capabilities = {true, true, true, false};
            local.regional.cost.expressionNodes = variant == 4 ? 0 : (variant == 5 ? UINT64_MAX : 1);
            local.regional.presence = [arena, visits](RegionalEvent event) -> std::optional<Expr> {
                return arena->lt(event.ordinal, arena->constant(visits));
            };
            local.regional.reachability = [arena, memo, refused, callbackCalls, bothUnavailable](
                RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
                ++*callbackCalls;
                if (!*memo) { *memo = arena->constant(1234567); }
                if (bothUnavailable) { return std::nullopt; }
                if (arena->constantValue(**memo) != 1234567) { return std::nullopt; }
                if (!*refused) { *refused = true; return std::nullopt; }
                auto av = arena->constantValue(a.ordinal), bv = arena->constantValue(b.ordinal);
                if (!av || !bv) { return std::nullopt; }
                const bool forward = *av <= *bv &&
                    (a.kind == PeriodicEventKind::Start || b.kind == PeriodicEventKind::Completion);
                return arena->boolean(forward);
            };
            if (existing) {
                auto numerical = std::make_shared<RegionalNumericalInterface>();
                const auto pipe = static_cast<uint32_t>(pto::PipelineType::PIPE_V);
                std::vector<std::vector<uint32_t>> chains(2);
                for (unsigned visit = 0; visit < visits; ++visit) {
                    for (unsigned kind = 0; kind < 2; ++kind) {
                        numerical->events.push_back({0, arena->constant(visit),
                            kind ? PeriodicEventKind::Completion : PeriodicEventKind::Start, {}});
                        chains[kind].push_back(2 * visit + kind);
                    }
                }
                numerical->chainKeys = {{pipe, PeriodicEventKind::Start}, {pipe, PeriodicEventKind::Completion}};
                auto index = std::make_shared<NumericalChainInterface>(buildNumericalChainInterface(
                    std::move(chains), [](uint32_t a, uint32_t b) -> std::optional<bool> {
                        return a / 2 <= b / 2 && (!(a % 2) || b % 2);
                    }));
                if (!index->error.empty()) { return function.emitError("prebuilt child index failed"); }
                numerical->index = index;
                auto eventId = [arena, visits](RegionalEvent event) -> std::optional<uint32_t> {
                    auto ordinal = arena->constantValue(event.ordinal);
                    if (!ordinal || *ordinal >= visits || event.type || !event.visits.empty()) { return {}; }
                    return 2 * uint32_t(*ordinal) + (event.kind == PeriodicEventKind::Completion);
                };
                numerical->query = [index, eventId](RegionalEvent a, RegionalEvent b,
                    NumericalChainQueryCost&) -> std::optional<bool> {
                    auto x = eventId(a), y = eventId(b);
                    return x && y ? std::optional<bool>(index->reaches(*x, *y)) : std::nullopt;
                };
                numerical->thresholds = [index, eventId](RegionalEvent event, bool reverse,
                    NumericalChainQueryCost&) -> std::optional<std::vector<uint32_t>> {
                    auto id = eventId(event);
                    if (!id) { return {}; }
                    return reverse ? index->reverse[*id] : index->forward[*id];
                };
                local.regional.numerical = std::move(numerical);
            }
            state->children.push_back(std::move(local));
            for (unsigned visit = 0; visit < visits; ++visit) { state->port(child, 0, arena->constant(visit)); }
            const auto pipe = static_cast<uint32_t>(pto::PipelineType::PIPE_V);
            state->nativeFirst.push_back({{pipe, {{child * visits, arena->boolean(true)}}}});
            state->nativeLast.push_back({{pipe, {{(child + 1) * visits - 1, arena->boolean(true)}}}});
        }
        if (variant != 5) { state->crossings.push_back({0, (count - 1) * visits, arena->boolean(true)}); }
        if (variant == 2 || bothUnavailable) { state->crossings.push_back({1, visits + 1, arena->boolean(true)}); }
        if (poison) {
            std::vector<RegionalAnalysis> children;
            const auto pipe = static_cast<uint32_t>(pto::PipelineType::PIPE_V);
            for (auto& child : state->children) {
                auto regional = child.regional;
                for (unsigned visit = 0; visit < visits; ++visit) {
                    RegionalSelector selector{{0, arena->constant(visit), PeriodicEventKind::Start, {}},
                        arena->boolean(true)};
                    RegionalStorageBoundary boundary;
                    boundary.cell = {pto::AddressSpace::VEC, 4 * visit, 4 * (visit + 1)};
                    boundary.firstWriters.push_back(selector); boundary.lastWriters.push_back(selector);
                    regional.storageBoundary.push_back(std::move(boundary));
                }
                regional.firstPayloads[pipe] = {{{0, arena->constant(0), PeriodicEventKind::Start, {}},
                    arena->boolean(true)}};
                regional.lastPayloads[pipe] = {{{0, arena->constant(visits - 1), PeriodicEventKind::Start, {}},
                    arena->boolean(true)}};
                regional.referenceBefore = [arena](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
                    (void)arena->div(arena->constant(1), arena->constant(0));
                    return arena->lt(a.ordinal, b.ordinal);
                };
                children.push_back(std::move(regional));
            }
            auto composed = composeRegionalSequence(function, arena, std::move(children), false, false);
            auto exported = sequenceRegionalResult(composed);
            if (arena->constructionError().empty() || composed.error.empty() || !composed.state ||
                composed.state->crossingMethods[0].method.empty() || exported.capabilities.exactQueries ||
                succeeded(prepareSequenceLogicalInsertion(composed)) ||
                sequenceEventReachability(composed, {0, 0, arena->constant(0), PeriodicEventKind::Start},
                    {1, 0, arena->constant(0), PeriodicEventKind::Completion})) {
                return function.emitError("poisoned crossing construction published a successful result");
            }
            continue;
        }
        const auto originalCrossings = state->crossings;
        const unsigned events = 2 * count * visits;
        std::vector<std::vector<bool>> expected(events, std::vector<bool>(events));
        for (unsigned payload = 0; payload < count * visits; ++payload) {
            expected[2 * payload][2 * payload] = expected[2 * payload + 1][2 * payload + 1] = true;
            expected[2 * payload][2 * payload + 1] = true;
            if (payload + 1 < count * visits) {
                expected[2 * payload][2 * payload + 2] = true;
                expected[2 * payload + 1][2 * payload + 3] = true;
            }
        }
        for (const auto& edge : originalCrossings) { expected[2 * edge.source + 1][2 * edge.target] = true; }
        for (unsigned k = 0; k < events; ++k) {
            for (unsigned a = 0; a < events; ++a) {
                for (unsigned b = 0; b < events; ++b) { expected[a][b] = expected[a][b] ||
                    (expected[a][k] && expected[k][b]); }
            }
        }
        const auto portCount = state->ports.size();
        const bool accepted = state->closure();
        if (bothUnavailable) {
            bool unchanged = state->crossings.size() == originalCrossings.size();
            for (std::size_t i = 0; unchanged && i < originalCrossings.size(); ++i) {
                const auto& a = state->crossings[i];
                const auto& b = originalCrossings[i];
                unchanged = a.source == b.source && a.target == b.target && a.guard == b.guard;
            }
            if (accepted || state->error.empty() || state->crossingObligations.size() != 2 ||
                state->crossingMethods[0].attemptConstructions != 1 ||
                state->crossingMethods[1].attemptConstructions != 1 || !unchanged ||
                state->ports.size() != portCount || state->numerical || state->numericalTree ||
                !*memo || arena->constantValue(**memo) != 1234567 || !arena->constructionError().empty()) {
                return function.emitError("failed crossing attempts lost graph or callback ownership");
            }
            continue;
        }
        if (!accepted) {
            return function.emitError("estimated crossing retry failed: ") << variant << ": " << state->error;
        }
        if (variant == 1) { callbackIndexEstimate = state->crossingMethods[0].estimate.work; }
        const auto reuseEstimate = state->crossingMethods[0].estimate.work;
        if (existing && (!reuseEstimate || !callbackIndexEstimate || *reuseEstimate >= *callbackIndexEstimate ||
            *callbackCalls || state->costs.numericalLeafQueries ||
            state->costs.numericalReusedChildren != count)) {
            return function.emitError("preexisting numerical indices rebuilt callback leaves");
        }
        const auto numerical = state->crossingMethods[0].attemptConstructions;
        const auto symbolic = state->crossingMethods[1].attemptConstructions;
        const bool selections = refuse ? numerical == 1 && symbolic == 1 :
            ((large || variant == 4) ? numerical == 1 && !symbolic : symbolic == 1 && !numerical);
        if (variant == 4 && (state->crossingMethods[0].estimate.work || state->crossingMethods[1].estimate.work)) {
            return function.emitError("missing callback costs did not remain unknown");
        }
        if (variant == 5 && (state->crossingMethods[0].estimate.work || !state->crossingMethods[1].estimate.work)) {
            return function.emitError("overflow estimate did not sort after known symbolic work");
        }
        if (!selections) {
            return function.emitError("crossing estimates did not select the expected reducer: ")
                << variant << ": " << numerical << "," << symbolic;
        }
        for (unsigned a = 0; a < events; ++a) {
            for (unsigned b = 0; b < events; ++b) {
                auto actual = state->eventReachability(a, b);
                if (!actual || arena->constantValue(*actual) != uint64_t(expected[a][b])) {
                    return function.emitError("crossing reducer changed the selected graph");
                }
            }
        }
        if (*memo && arena->constantValue(**memo) != 1234567) {
            return function.emitError("crossing retry invalidated a child callback cache");
        }
    }
    llvm::outs() << "crossing-costs: both preferences both retries exact-graphs unknown overflow "
                    "cached-ids reused-indices failed-retention poisoned-export\n";
    return success();
}
LogicalResult checkProducerFailureStages(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    session.enableStageProfiling();
    if (failed(session.initialize(policy))) { return failure(); }
    const bool dynamic = function->hasAttr("test.dynamic_producer_failure");
    auto first = dynamic ? session.analyzeFiniteExpansion({}) : session.analyzeNumericalRegion({});
    const auto attempts = session.constructionCounts().mathematicalAttempts;
    auto retry = dynamic ? session.analyzeFiniteExpansion({}) : session.analyzeNumericalRegion({});
    const bool unresolved = dynamic || function->hasAttr("test.unproved_outer");
    const auto expected = unresolved ? AnalysisStatus::UnmetObligation : AnalysisStatus::NotApplicable;
    const bool retained = first.status == expected && first.stage == AnalysisStage::Form &&
        !first.mathematical && !first.obligations.empty() && retry.status == first.status &&
        retry.stage == first.stage && session.constructionCounts().mathematicalAttempts == attempts;
    if (!retained) { return function.emitError("producer failure lost its cached form status or stage"); }
    auto invalid = session.analyze({SIZE_MAX});
    if (invalid.status != AnalysisStatus::UnmetObligation || invalid.stage != AnalysisStage::Form) {
        return function.emitError("invalid request was reported as a proved class exclusion");
    }
    if (dynamic) {
        auto aggregate = session.minimumDemands();
        if (!aggregate.mathematical && aggregate.status != AnalysisStatus::UnmetObligation) {
            return function.emitError("exhausted attempts were reported as a proved class exclusion");
        }
    }
    if (function->hasAttr("test.unproved_outer")) {
        auto aggregate = session.minimumDemands();
        if (aggregate.mathematical || aggregate.status != AnalysisStatus::UnmetObligation ||
            aggregate.stage != AnalysisStage::Demands) {
            return function.emitError("incomplete producer coverage lost its demand-stage obligation");
        }
        const auto counts = session.constructionCounts();
        const auto stages = session.diagnosticLedger()->stages.size();
        const auto selections = session.diagnosticLedger()->selections.size();
        const auto repeated = session.minimumDemands();
        if (repeated.status != aggregate.status || repeated.stage != aggregate.stage ||
            repeated.mathematical != aggregate.mathematical ||
            repeated.obligations.size() != aggregate.obligations.size() ||
            repeated.requestId == aggregate.requestId ||
            session.constructionCounts().requestWalks != counts.requestWalks ||
            session.constructionCounts().mathematicalAttempts != counts.mathematicalAttempts ||
            session.constructionCounts().exhaustedWalkHits != counts.exhaustedWalkHits + 1 ||
            session.diagnosticLedger()->stages.size() != stages + 1 ||
            session.diagnosticLedger()->selections.size() != selections) {
            return function.emitError("identical exhausted request walked cached candidate attempts");
        }
        for (std::size_t i = 0; i < aggregate.obligations.size(); ++i) {
            if (repeated.obligations[i].stage != aggregate.obligations[i].stage ||
                repeated.obligations[i].diagnostic != aggregate.obligations[i].diagnostic) {
                return function.emitError("exhausted request changed its retained diagnostic recipe");
            }
        }
    }
    if (!dynamic && !function->hasAttr("test.unproved_outer")) {
        AnalysisRequest stateful;
        stateful.needs.evaluation = AnalysisEvaluation::Stateful;
        auto refused = session.analyze(stateful);
        const auto walks = session.constructionCounts().requestWalks;
        auto repeated = session.analyze(stateful);
        if (refused.status != AnalysisStatus::UnmetObligation || repeated.status != refused.status ||
            session.constructionCounts().requestWalks != walks) {
            return function.emitError("stateful exhausted recipe was not retained independently");
        }
        // The weak native answer did not request arithmetic recognition. This
        // public audit publishes a reusable original form even though the
        // stateful capability itself remains unavailable.
        if (failed(session.recognizeArithmetic())) { return failure(); }
        auto afterEvidence = session.analyze(stateful);
        if (afterEvidence.status != refused.status ||
            session.constructionCounts().requestWalks != walks + 1 ||
            afterEvidence.requestId == repeated.requestId) {
            return function.emitError("new retained form failed to invalidate an exhausted walk");
        }
        const auto retried = session.constructionCounts().requestWalks;
        if (failed(session.recognizeArithmetic()) ||
            session.analyze(stateful).status != refused.status ||
            session.constructionCounts().requestWalks != retried) {
            return function.emitError("unchanged retained form invalidated an exhausted walk");
        }
        auto fallback = stateful;
        fallback.mode = AnalysisMode::Fallback;
        if (session.analyze(fallback).status != refused.status ||
            session.constructionCounts().requestWalks == retried) {
            return function.emitError("exhausted request crossed its mode context");
        }
    }
    {
        // Exact cache key controls, including contexts that ordinary public
        // calls cannot forge: audit, active parent and same-node continuation.
        AnalysisSessionState cache;
        AnalysisRequest request;
        AnalysisOutcome refusal;
        refusal.status = AnalysisStatus::UnmetObligation;
        const auto key = cache.exhaustedKey(request);
        cache.retainExhausted(key, refusal);
        if (!cache.exhausted(key)) { return failure(); }
        request.needs.queries = true;
        if (cache.exhausted(cache.exhaustedKey(request))) { return failure(); }
        request.needs.queries = false;
        cache.certifying = true;
        if (cache.exhausted(cache.exhaustedKey(request))) { return failure(); }
        cache.certifying = false;
        cache.activeRegions.push_back(1);
        if (cache.exhausted(cache.exhaustedKey(request))) { return failure(); }
        cache.activeRegions.clear();
        if (cache.exhausted(cache.exhaustedKey(request, true))) { return failure(); }
        auto& completion = cache.completions[0];
        completion.attempted = true;
        completion.evidenceGeneration = cache.retainedEvidenceGeneration;
        completion.error = "missing child capability";
        if (!completion.current(cache.retainedEvidenceGeneration)) { return failure(); }
        cache.retainedEvidencePublished();
        if (cache.exhausted(key) || completion.current(cache.retainedEvidenceGeneration)) { return failure(); }
        cache.retainExhausted(key, refusal);
        cache.retainedEvidenceGeneration = UINT64_MAX;
        completion.evidenceGeneration = 0; // A pre-wrap attempt must never revive.
        cache.retainExhausted(key, refusal);
        cache.retainedEvidencePublished();
        if (cache.exhausted(key) || !cache.exhaustedRequests.empty() ||
            completion.current(cache.retainedEvidenceGeneration)) { return failure(); }
        AnalysisSessionState fresh;
        if (fresh.exhausted(key)) { return failure(); }
    }
    llvm::outs() << "producer-failure: typed form-status cached invalid-request-unresolved\n";
    return success();
}
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
LogicalResult checkConditionalEndpointUpgrade(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    auto render = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text); function.print(stream); stream.flush(); return text;
    };
    const auto original = render();
    pto::SyncInput input(policy);
    auto index = std::make_shared<PhaseIndex>();
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index->build(function, input))) { return failure(); }
    auto program = recognizeProgram(function, input, *index, ProgramRecognitionMode::Exhaustive);
    if (failed(program)) { return failure(); }
    std::optional<std::size_t> conditional;
    for (auto [id, node] : llvm::enumerate(program->nodes)) {
        if (node.kind != StructureKind::Conditional) { continue; }
        if (conditional) { return function.emitError("conditional upgrade fixture has multiple branches"); }
        conditional = id;
    }
    if (!conditional) { return function.emitError("conditional upgrade fixture lacks a branch"); }
    auto arena = std::make_shared<RegionExpressions>();
    std::map<std::size_t, RegionalAnalysis> arms;
    unsigned builds = 0, prepared = 0;
    SequenceRegionResolver resolver;
    resolver.region = [&](std::size_t id, bool endpoints, std::string& error) -> FailureOr<RegionalAnalysis> {
        auto found = arms.find(id);
        if (found == arms.end()) {
            ++builds;
            auto child = analyzeSequenceRegion(function, input, *program, id, arena, index, false);
            if (!child.error.empty()) { error = child.error; return failure(); }
            found = arms.emplace(id, sequenceRegionalResult(child)).first;
        }
        auto out = found->second;
        if (endpoints) {
            const auto owner = std::make_shared<const RegionalAnalysis>(out);
            if (owner->prepare) {
                out.prepare = [owner, &prepared]() { ++prepared; return owner->prepare(); };
            }
            if (owner->prepareWithVisits) {
                out.prepareWithVisits = [owner, &prepared](ArrayRef<scf::ForOp> loops) {
                    ++prepared; return owner->prepareWithVisits(loops);
                };
            }
        }
        return out;
    };
    auto weak = analyzeSequenceRegionWithResolver(
        function, input, *program, *conditional, arena, index, false, resolver);
    if (!weak.error.empty() || !weak.state || weak.state->relationalResult || weak.state->children.size() != 2 ||
        llvm::any_of(weak.state->children, [](const Child& child) {
            return !child.originalNode || !child.originalArmGuard;
        })) { return function.emitError("conditional upgrade fixture lacks two original guarded arm owners"); }
    const auto old = weak.state;
    const auto nodes = arena->size();
    const auto childBuilds = builds;
    auto upgraded = upgradeSequenceEndpoints(weak, *conditional, resolver);
    if (!upgraded || !upgraded->error.empty() || !upgraded->state || upgraded->state->upgradeSource != old ||
        upgraded->upgradeWork.retainedChildren != 2 || upgraded->upgradeWork.replacedChildren ||
        upgraded->upgradeWork.recomposedCrossings || upgraded->upgradeWork.reusedCrossings != old->crossings.size() ||
        builds != childBuilds || arena->size() != nodes) {
        return function.emitError("conditional endpoint upgrade rebuilt unchanged mathematics");
    }
    auto branch = cast<scf::IfOp>(program->nodes[*conditional].anchor);
    const auto condition = arena->input(branch.getCondition());
    for (unsigned arm = 0; arm < 2; ++arm) {
        const auto& before = old->children[arm];
        const auto& after = upgraded->state->children[arm];
        if (before.originalArmGuard != after.originalArmGuard ||
            !sameOriginalCoordinates(before.regional, after.regional)) { return failure(); }
        for (unsigned flag : {0U, 1U}) {
            for (unsigned type = 0; type < after.anchors.size(); ++type) {
                auto present = regionalPresence(after.regional,
                    {type, arena->constant(0), PeriodicEventKind::Start, {}});
                if (!present) { return failure(); }
                RegionExpressions::Transaction evaluation(*arena);
                RegionExpressions::Substitution binding({{condition, arena->boolean(flag)}});
                if (arena->constantValue(arena->substitute(*present, binding)) != (flag == unsigned(arm == 0))) {
                    return function.emitError("retained conditional arm lost exact branch presence");
                }
            }
        }
    }
    // Both public preparation paths must invoke each raw arm once. The default
    // path uses its owned guarded callback; the explicit resolver receives the
    // preserved semantic view and returns a fresh raw plan for the existing mask.
    auto direct = prepareSequenceLogicalInsertion(*upgraded);
    if (failed(direct) || prepared != 2) { return function.emitError("owned conditional preparation failed"); }
    unsigned resolved = 0;
    SequenceEndpointResolver preparation = [&](std::size_t id, const RegionalAnalysis& retained,
        ArrayRef<scf::ForOp> enclosing) -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        ++resolved;
        const auto found = arms.find(id);
        if (found == arms.end() || !sameOriginalCoordinates(found->second, retained)) { return failure(); }
        return found->second.prepareWithVisits ? found->second.prepareWithVisits(enclosing) : found->second.prepare();
    };
    auto resolvedPlan = prepareSequenceLogicalInsertion(*upgraded, {}, preparation);
    auto cold = analyzeSequenceRegionWithResolver(
        function, input, *program, *conditional, arena, index, true, resolver);
    auto coldPlan = cold.error.empty() ? prepareSequenceLogicalInsertion(cold) :
        FailureOr<std::unique_ptr<PreparedLogicalPlan>>(failure());
    if (failed(resolvedPlan) || failed(coldPlan) || resolved != 2 || prepared != 4 ||
        (*direct)->completeInvocation || (*resolvedPlan)->completeInvocation || (*coldPlan)->completeInvocation) {
        return function.emitError("conditional preparation changed invocation or callback ownership");
    }
    auto signature = [&](PreparedLogicalPlan& plan) -> std::optional<std::string> {
        if (failed(materializeLogicalFamilies(plan)) || plan.endpoints.empty()) { return std::nullopt; }
        std::string text;
        llvm::raw_string_ostream stream(text);
        for (const auto& stage : plan.preparation) {
            for (auto& operation : *stage.code) { operation.print(stream); }
        }
        for (const auto& endpoint : plan.endpoints) {
            stream << unsigned(endpoint.kind) << ":" << endpoint.record << ":" << endpoint.sourcePipe << ":"
                << endpoint.targetPipe << ":" << endpoint.memberCoordinates.size() << "\n";
            endpoint.before->print(stream);
        }
        stream.flush();
        return text;
    };
    auto expected = signature(**coldPlan);
    if (!expected || signature(**direct) != expected || signature(**resolvedPlan) != expected) {
        return function.emitError("conditional retained/default/resolved/cold command recipes differ");
    }
    auto changed = resolver;
    changed.region = [&](std::size_t id, bool, std::string&) -> FailureOr<RegionalAnalysis> {
        auto out = arms.at(id);
        out.firstOrdinal = arena->constant(1);
        return out;
    };
    if (upgradeSequenceEndpoints(weak, *conditional, changed)) {
        return function.emitError("changed conditional coordinates incorrectly reused old closure");
    }
    auto foreign = resolver;
    foreign.region = [&](std::size_t id, bool, std::string&) -> FailureOr<RegionalAnalysis> {
        auto out = arms.at(id);
        out.expressions = std::make_shared<RegionExpressions>();
        return out;
    };
    if (upgradeSequenceEndpoints(weak, *conditional, foreign)) {
        return function.emitError("conditional upgrade reinterpreted an original guard in another arena");
    }
    auto mismatched = weak;
    mismatched.state = std::make_shared<SequenceAnalysisState>(*weak.state);
    mismatched.state->children.front().originalArmGuard = arena->boolean(true);
    if (upgradeSequenceEndpoints(mismatched, *conditional, resolver)) {
        return function.emitError("conditional upgrade accepted a foreign arm guard");
    }
    weak.state.reset();
    if (upgraded->state->upgradeSource != old || render() != original) {
        return function.emitError("conditional upgrade lost ownership or mutated original source");
    }
    llvm::outs() << "conditional-endpoint-upgrade: guarded-closure-reused "
        "both-preparation-routes cold-equivalent residual\n";
    return success();
}
LogicalResult checkPhaseCapabilityUpgrade(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    for (bool warm : {true, false}) {
        FrontierAnalysis session(function);
        if (failed(session.initialize(policy))) { return failure(); }
        AnalysisOutcome weak;
        if (warm) {
            weak = session.minimumDemands();
            if (!weak.mathematical || !weak.mathematical->sequenceDemands ||
                !sequenceNeedsEndpointUpgrade(*weak.mathematical->sequenceDemands)) {
                return function.emitError("weak phase request did not retain compact mathematics");
            }
        }
        AnalysisRequest request; request.needs.synchronization = true;
        auto strong = session.analyze(request);
        if (strong.status != AnalysisStatus::Ready || !strong.mathematical ||
            !strong.mathematical->sequenceDemands ||
            sequenceNeedsEndpointUpgrade(*strong.mathematical->sequenceDemands)) {
            return function.emitError("strong phase request did not upgrade the retained compact representation");
        }
        const auto attempts = session.constructionCounts().mathematicalAttempts;
        auto again = session.analyze(request);
        if (again.mathematical != strong.mathematical ||
            session.constructionCounts().mathematicalAttempts != attempts ||
            (warm && session.minimumDemands().mathematical != weak.mathematical)) {
            return function.emitError("phase capability upgrade replaced or rebuilt retained mathematics");
        }
        auto plan = session.prepareLogical(strong);
        if (failed(plan) || failed(session.attachAllocation(strong, **plan, {0, 1, 2, 3, 4, 5}))) {
            return function.emitError("phase capability lost its own physical allocation evidence");
        }
    }
    llvm::outs() << "phase capability: compact mathematics retained, warm/cold upgrade and six-ID evidence passed\n";
    return success();
}
LogicalResult checkSequenceEndpointRetry(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    std::string before;
    llvm::raw_string_ostream original(before); function.print(original); original.flush();
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    // Establish the representation whose later endpoint upgrade this fixture
    // checks. Candidate ranking is free to prefer an original-coordinate form
    // in a cold request; it need not select expansion to exercise this test.
    const auto* structure = session.result();
    const auto outer = llvm::find_if(structure->nodes, [&](const auto& node) {
        return node.kind == StructureKind::Loop && node.anchor->getParentOp() == function;
    });
    if (outer == structure->nodes.end()) { return function.emitError("sequence retry lacks an outer loop"); }
    const auto outerId = static_cast<std::size_t>(outer - structure->nodes.begin());
    AnalysisRequest expandedRequest;
    expandedRequest.region = outerId;
    expandedRequest.needs.queries = expandedRequest.needs.selectors = true;
    const auto seeded = session.analyzeFiniteExpansion(expandedRequest);
    if (seeded.status != AnalysisStatus::Ready || !seeded.mathematical ||
        seeded.mathematical->backend != "expanded-finite-guarded" || !seeded.regionalExports ||
        !seeded.regionalExports->capabilities.exactQueries || !seeded.regionalExports->capabilities.exactSelectors) {
        return function.emitError("sequence retry could not establish expanded child queries and selectors");
    }
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
            if (original.mathematical && original.mathematical->backend == "expanded-finite-guarded") {
                FrontierAnalysis cold(function);
                if (failed(cold.initialize(policy))) { return failure(); }
                const auto id = *child.originalNode;
                const auto order = cold.exactForms({id});
                if (order != std::vector<AnalysisForm>{AnalysisForm::Original, AnalysisForm::SmallCountExpanded} ||
                    cold.normalizationConstructions() || cold.numericTemplatePreflights() ||
                    cold.result()->formConstructions) {
                    return function.emitError("cold expanded fallback eagerly qualified a candidate");
                }
                const auto selected = cold.minimumDemands(id);
                const auto costs = cold.costRecords();
                auto attempted = [&](StringRef method) {
                    return llvm::any_of(costs, [&](const auto& row) {
                        return row.region == id && !row.request && row.method == method && row.attemptConstructions;
                    });
                };
                if (selected.status != AnalysisStatus::Ready || !selected.mathematical ||
                    !attempted("form-original")) {
                    return function.emitError("cold screening failed to produce exact original-region mathematics");
                }
                const bool choseExpanded = selected.mathematical->backend == "expanded-finite-guarded";
                if (attempted("form-expanded") != choseExpanded) {
                    return function.emitError("cold screening qualified an unused expanded form or lost its fallback");
                }
            }
        }
    }
    if (!expanded || strong.mathematical == weak.mathematical) {
        return function.emitError("sequence retry did not replace expanded child coordinates");
    }
    const auto& upgradedSequence = *strong.mathematical->sequenceDemands;
    const auto& oldSequence = *weak.mathematical->sequenceDemands;
    if (!upgradedSequence.state->upgradeSource || !upgradedSequence.upgradeWork.replacedChildren ||
        !upgradedSequence.upgradeWork.retainedChildren || upgradedSequence.upgradeWork.reusedCrossings ||
        upgradedSequence.upgradeWork.recomposedCrossings != upgradedSequence.state->crossings.size() ||
        upgradedSequence.upgradeWork.retainedChildren + upgradedSequence.upgradeWork.replacedChildren !=
            oldSequence.state->children.size()) {
        return function.emitError("sequence upgrade recollected unchanged children or reused stale closure");
    }
    SequenceRegionResolver sameCoordinates;
    sameCoordinates.region = [source = upgradedSequence.state](std::size_t region, bool endpoints,
        std::string&) -> FailureOr<RegionalAnalysis> {
        if (!endpoints) { return failure(); }
        const auto found = llvm::find_if(source->children, [&](const Child& child) {
            return child.originalNode == region;
        });
        return found == source->children.end() ? FailureOr<RegionalAnalysis>(failure()) : found->regional;
    };
    const auto expressionCount = upgradedSequence.state->expressions.size();
    auto unchanged = upgradeSequenceEndpoints(upgradedSequence, 0, std::move(sameCoordinates));
    if (!unchanged || !unchanged->error.empty() || unchanged->state == upgradedSequence.state ||
        unchanged->state->upgradeSource != upgradedSequence.state || unchanged->upgradeWork.replacedChildren ||
        unchanged->upgradeWork.recomposedCrossings ||
        unchanged->upgradeWork.reusedCrossings != upgradedSequence.state->crossings.size() ||
        unchanged->upgradeWork.retainedChildren != upgradedSequence.state->children.size() ||
        unchanged->state->expressions.size() != expressionCount) {
        return function.emitError("same-coordinate endpoint upgrade reconstructed mathematical work");
    }
    auto unchangedExports = sequenceRegionalResult(*unchanged);
    if (!unchangedExports.capabilities.endpointRecipes || !unchangedExports.prepareWithVisits) {
        return function.emitError("same-coordinate upgrade lost its owned endpoint export");
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
    if (failed(materializeLogicalFamilies(**left)) || failed(materializeLogicalFamilies(**right))) {
        return function.emitError("sequence retry family materialization failed");
    }
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
    auto unchangedPlan = prepareSequenceLogicalInsertion(*unchanged);
    if (failed(unchangedPlan) || failed(materializeLogicalFamilies(**unchangedPlan)) ||
        renderPlan(**unchangedPlan) != independent) {
        return function.emitError("retained closure preparation changed emitted control");
    }
    left->reset();
    const bool ownerRetained = renderPlan(**right) == independent;
    if (!ownerRetained) { return function.emitError("sequence plan lost its owner"); }
    FrontierAnalysis cold(function), audited(function);
    if (failed(cold.initialize(policy)) || failed(audited.initialize(policy)) ||
        failed(audited.recognizeAllForms())) { return failure(); }
    auto coldResult = cold.analyze(request), auditedResult = audited.analyze(request);
    if (coldResult.status != AnalysisStatus::Ready || auditedResult.status != AnalysisStatus::Ready ||
        !coldResult.mathematical || !auditedResult.mathematical ||
        coldResult.mathematical->backend != auditedResult.mathematical->backend) {
        return function.emitError("cold sequence dispatch differs from exhaustive-audit dispatch");
    }
    auto coldPlan = cold.prepareLogical(coldResult), auditedPlan = audited.prepareLogical(auditedResult);
    if (failed(coldPlan) || failed(auditedPlan) || failed(materializeLogicalFamilies(**coldPlan)) ||
        failed(materializeLogicalFamilies(**auditedPlan)) || renderPlan(**coldPlan) != renderPlan(**auditedPlan)) {
        return function.emitError("exhaustive audit changed selected sequence command graph");
    }
    const auto coldForms = cold.result()->formConstructions;
    const auto auditedForms = audited.result()->formConstructions;
    (void)cold.analyze(request); (void)audited.analyze(request);
    if (cold.result()->formConstructions != coldForms || audited.result()->formConstructions != auditedForms ||
        coldForms > auditedForms) {
        return function.emitError("repeated sequence dispatch reconstructed class forms");
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
        const auto* math = outcome.mathematical.get();
        // Identity composition transports the retained child's mathematics.
        // Its wrapper need not duplicate the child's producer-specific fields.
        while (math && math->identityComposition) { math = math->identityComposition->child.get(); }
        return math && (math->arithmeticRegionalDemands ||
            (math->sequenceDemands && math->sequenceDemands->cost.arithmeticRegions != 0));
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
        root.mathematical && root.mathematical->identityComposition &&
        root.mathematical->identityComposition->child == later.mathematical && root.regionalExports == snapshot &&
        usesArithmetic(root) && session.arithmeticRegionConstructions() == arithmeticBuilds &&
        !session.constructionCounts().logicalPreparations && !session.constructionCounts().allocationExports;
    if (!rootReady) {
        llvm::errs() << "natural root: backend=" << (root.mathematical ? root.mathematical->backend : "none")
            << " status=" << unsigned(root.status) << " stage=" << unsigned(root.stage)
            << " queries=" << root.available.queries << " selectors=" << root.available.selectors
            << " arithmetic-builds=" << session.arithmeticRegionConstructions()
            << " prior-builds=" << arithmeticBuilds << "\n";
        for (const auto& issue : root.obligations) { llvm::errs() << issue.diagnostic << "\n"; }
        return function.emitError("root and regional export retry outcomes disagree");
    }
    auto rootRetry = session.analyze(rootRequest);
    if (rootRetry.status != AnalysisStatus::Ready || rootRetry.mathematical != root.mathematical ||
        rootRetry.regionalExports != root.regionalExports ||
        session.arithmeticRegionConstructions() != arithmeticBuilds) {
        return function.emitError("strong identity root retry replaced its retained child or exports");
    }
    FrontierAnalysis coldRoot(function);
    if (failed(coldRoot.initialize(policy)) || failed(coldRoot.configureArithmeticProfiles({profile}, {profile}))) {
        return failure();
    }
    auto coldRootResult = coldRoot.analyze(rootRequest);
    if (coldRootResult.status != AnalysisStatus::Ready || !coldRootResult.mathematical ||
        !coldRootResult.mathematical->identityComposition || !usesArithmetic(coldRootResult) ||
        !coldRootResult.available.queries || !coldRootResult.available.selectors ||
        coldRoot.constructionCounts().logicalPreparations || coldRoot.constructionCounts().allocationExports) {
        llvm::errs() << "natural cold root: backend="
            << (coldRootResult.mathematical ? coldRootResult.mathematical->backend : "none")
            << " status=" << unsigned(coldRootResult.status) << " stage=" << unsigned(coldRootResult.stage) << "\n";
        for (const auto& issue : coldRootResult.obligations) { llvm::errs() << issue.diagnostic << "\n"; }
        return function.emitError("cold strong identity root lost requested arithmetic exports");
    }
    std::string after;
    llvm::raw_string_ostream current(after); function.print(current); current.flush();
    if (before != after) { return function.emitError("natural export retry changed original IR"); }
    llvm::outs() << "natural-export-retry: rotating-demands retained slot-limit "
                 << "arithmetic-ready cached source-unchanged\n";
    return success();
}
namespace {
LogicalResult checkSharedControlViews(
    func::FuncOp function, const pto::frontiersynch::PhaseIndex& index, const pto::SyncInput& input)
{
    using namespace pto::frontiersynch;
    auto backing = buildNormalizedControlTree(function, index, input);
    auto whole = normalizeSmallCountControl({function, function}, index, input, {}, backing);
    auto compact = whole->original ? whole->original : whole;
    const auto visits = normalizedControlSourceVisits(*backing);
    SmallVector<std::shared_ptr<const NormalizedControlDescription>> views;
    bool valid = visits >= 2 * compact->nodes.size() && visits <= 3 * compact->nodes.size();
    function.walk([&](Operation* operation) {
        auto selected = normalizeSmallCountControl({function, operation}, index, input, {}, backing);
        auto local = selected->original ? selected->original : selected;
        valid &= local->result.state == RecognitionState::Applicable && local->backing == backing &&
                 local->nodes.data() == compact->nodes.data() && local->originalNode(operation).has_value();
        if (operation == function.getOperation()) {
            views.push_back(std::move(selected));
            return;
        }
        for (auto& sibling : *operation->getBlock()) {
            if (&sibling != operation && operation != function.getOperation()) {
                valid &= !local->originalNode(&sibling).has_value();
            }
        }
        views.push_back(std::move(selected));
    });
    valid &= normalizedControlSourceVisits(*backing) == visits;
    // A global operation index must not admit a sibling's regional view.
    auto loops = function.getOps<scf::ForOp>();
    if (!loops.empty()) {
        auto loop = *loops.begin();
        auto* sibling = loop->getPrevNode();
        if (sibling) {
            auto wrong = normalizeSmallCountControl({function, sibling}, index, input, {}, backing);
            auto rejected = preflightNumericTemplate(loop, index, input, {}, true, {}, {}, wrong);
            valid &= rejected.form.result.state != RecognitionState::Applicable;
        }
    }
    std::weak_ptr<const NormalizedControlTree> lifetime = backing;
    backing.reset();
    whole.reset();
    compact.reset();
    valid &= !lifetime.expired() && !views.empty() && !views.front()->nodes.empty();
    views.clear();
    valid &= lifetime.expired();
    if (!valid) {
        return function.emitError("shared control views lost membership, ownership or bounded source visits");
    }
    return success();
}
LogicalResult checkLazyProgramForms(
    func::FuncOp function, const pto::frontiersynch::PhaseIndex& index, const pto::SyncInput& input)
{
    using namespace pto::frontiersynch;
    auto lazy = recognizeProgram(function, input, index, ProgramRecognitionMode::Inventory);
    if (failed(lazy) || lazy->formConstructions ||
        llvm::any_of(lazy->nodes, [](const auto& node) { return node.attemptedForms != 0; })) {
        return function.emitError("structural inventory eagerly constructed class forms");
    }
    auto* identities = lazy->nodes.data();
    auto loop = llvm::find_if(lazy->nodes, [](const auto& node) { return node.kind == StructureKind::Loop; });
    if (loop != lazy->nodes.end()) {
        auto id = static_cast<std::size_t>(loop - identities);
        materializeProgramForm(*lazy, id, ProgramForm::Rotating, index, input);
        const auto constructions = lazy->formConstructions;
        materializeProgramForm(*lazy, id, ProgramForm::Rotating, index, input);
        if (constructions != 1 || lazy->formConstructions != constructions || loop->guardedRotatingResult ||
            loop->boundedLifetime || loop->varyingRotating) {
            return function.emitError("lazy form check repeated or constructed unrelated families");
        }
    }
    materializeAllProgramForms(*lazy, index, input);
    const auto count = lazy->formConstructions;
    materializeAllProgramForms(*lazy, index, input);
    auto eager = recognizeProgram(function, input, index);
    if (failed(eager) || lazy->nodes.data() != identities || lazy->formConstructions != count ||
        count != eager->formConstructions || lazy->contractAudit.size() != eager->contractAudit.size()) {
        return function.emitError("exhaustive form audit changed stable inventory or repeated checks");
    }
    for (std::size_t id = 0; id < lazy->contractAudit.size(); ++id) {
        const auto& a = lazy->contractAudit[id];
        const auto& b = eager->contractAudit[id];
        if (a.kind != b.kind || a.node != b.node || a.membership != b.membership ||
            a.diagnostics.size() != b.diagnostics.size()) {
            return function.emitError("lazy and exhaustive form audit disagree");
        }
    }
    return success();
}
} // namespace
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
        order == std::vector<AnalysisForm>{AnalysisForm::Original, AnalysisForm::SmallCountExpanded} &&
        !session.normalizationConstructions() && !session.normalizationSourceVisits() &&
        !session.numericTemplatePreflights() && !session.result()->formConstructions &&
        !session.constructionCounts().mathematicalAttempts && !session.numericalRegionConstructions();
    if (!repeatRank) { return function.emitError("cold screening constructed an unused qualification"); }
    if (!function->hasAttr("test.normalization_external")) {
        session.exactForms({0});
        const bool canonical = !session.normalizationConstructions();
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
        !session.normalizationConstructions() && !session.numericTemplatePreflights();
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
        finite.mathematical->expandedFiniteDemands && finite.mathematical->normalizedInput;
    if (!available) { return function.emitError("finite adapter cannot consume retained normalized input"); }
    const auto owner = finite.mathematical->normalizedInput;
    const auto sourceVisits = session.normalizationSourceVisits();
    const auto preflights = session.numericTemplatePreflights();
    const auto formChecks = session.result()->formConstructions;
    const auto warmOrder = session.exactForms({id});
    const auto warmRecords = session.costRecords().size();
    const bool retainedOnly = session.exactForms({id}) == warmOrder &&
        session.costRecords().size() == warmRecords && session.normalizationConstructions() == 1 &&
        session.numericTemplatePreflights() == preflights && session.result()->formConstructions == formChecks &&
        session.normalizationSourceVisits() == sourceVisits;
    if (!retainedOnly) { return function.emitError("warm screening reconstructed retained qualification"); }
    const bool alternative = owner->original != nullptr;
    const bool expectedOrder = warmOrder.size() == (alternative ? 2 : 1) &&
        (alternative || warmOrder.front() == AnalysisForm::Original);
    if (!expectedOrder) {
        return function.emitError("compact fallback was advertised as another expanded description");
    }
    if (alternative) {
        const auto plan = preflightNumericTemplate(cast<scf::ForOp>(node->anchor),
            *owner->index, *owner->input, {}, true, {}, {}, owner);
        const bool eligible = plan.form.result.state == RecognitionState::Applicable;
        auto costs = session.costRecords();
        // Refreshed evidence is append-only; inspect its latest estimate pair.
        std::reverse(costs.begin(), costs.end());
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
        const auto first = estimatedSelectionWorkLess(expanded->estimate, original->estimate) ?
            AnalysisForm::SmallCountExpanded : AnalysisForm::Original;
        const bool correctRank = priced && warmOrder.front() == first;
        if (!correctRank) {
            return function.emitError("candidate order or residual occurrence cost is incorrect");
        }
        const bool unknownOriginalFirst = eligible && !original->estimate.work &&
            warmOrder.front() != AnalysisForm::SmallCountExpanded;
        if (unknownOriginalFirst) {
            return function.emitError("known expanded work did not precede unknown original work");
        }
        const bool runtimeRank = rejectsNumerical && warmOrder.front() != AnalysisForm::Original;
        if (runtimeRank) {
            return function.emitError("unknown-work candidates lost stable ordering");
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
        const bool dispatchedFirst = attempted("form-original");
        const bool selectedOriginal = selected.status == AnalysisStatus::Ready &&
            selected.mathematical && !attempted("form-expanded");
        if (!dispatchedFirst || !selectedOriginal) {
            return function.emitError("dispatcher did not preserve first successful exact form");
        }
    }
    const auto root = session.analyzeFiniteExpansion({0});
    const bool sameProducer = root.mathematical &&
        root.mathematical->expandedFiniteDemands == finite.mathematical->expandedFiniteDemands &&
        root.mathematical->normalizedInput == owner;
    const bool external = function->hasAttr("test.normalization_external");
    const bool promoted = sameProducer != external && (external || session.finiteExpansionPreflights() == 1);
    if (!promoted) {
        return function.emitError("root promotion ignored original scope or rebuilt the same finite producer");
    }

    const uint64_t expectedNormalizations = external ? 2 : 1;
    const bool shared = session.normalizationConstructions() == expectedNormalizations && sourceVisits &&
        session.normalizationSourceVisits() == sourceVisits &&
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
    if (failed(checkSharedControlViews(function, index, input)) ||
        failed(checkLazyProgramForms(function, index, input))) { return failure(); }
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
    SmallVector<std::size_t> compactPending(compact->roots);
    DenseSet<std::size_t> compactVisited;
    while (!compactPending.empty()) {
        const auto id = compactPending.pop_back_val();
        if (id >= compact->nodes.size() || !compactVisited.insert(id).second) {
            return function.emitError("compact regional view has invalid or repeated node identities");
        }
        const auto& source = compact->nodes[id];
        llvm::append_range(compactPending, source.children);
        for (auto* phase : index.phasesFor(source.original)) {
            ++compactPayloads;
            compactEffects += input.accesses().effectsFor(phase).size();
        }
    }
    if (compact->payloads != compactPayloads || compact->effects != compactEffects ||
        compact->representedNodes != compactVisited.size()) {
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
        demands.mathematical->concreteDemands() && queries.mathematical == demands.mathematical &&
        !session.normalizationConstructions() && !session.normalizationSourceVisits() &&
        !session.result()->formConstructions && !session.arithmeticGeneratorConstructions() &&
        !session.numericTemplatePreflights() && !session.constructionCounts().logicalPreparations;
    if (!explicitOnly) { return function.emitError("straight-line analysis left the explicit fast path"); }
    const auto owner = demands.mathematical->finiteDemands;
    if (!owner || owner->guarded() || !owner->originalInput() ||
        &owner->originalInput()->storage() != session.input() ||
        owner != queries.mathematical->finiteDemands || session.constructionCounts().explicitReductions != 1) {
        return function.emitError("finite concrete ownership or capability reuse changed");
    }
    if (!pto::test::checkFiniteRegionalPartitions(owner)) {
        return function.emitError("finite partition queries, boundaries or ownership disagree");
    }
    auto prepared = session.prepareLogical(demands);
    const auto& explicitAnalysis = *demands.mathematical->concreteDemands();
    if (failed(prepared)) { return function.emitError("explicit endpoint recipe preparation failed"); }
    if (!(*prepared)->preparation.empty() || !(*prepared)->endpoints.empty() ||
        (!explicitAnalysis.reduction.retained.empty() && !(*prepared)->familyRecipe)) {
        return function.emitError("explicit endpoint preparation did not retain an unmaterialized owned recipe");
    }
    if (session.normalizationConstructions() || session.result()->formConstructions ||
        session.arithmeticGeneratorConstructions() || session.numericTemplatePreflights()) {
        return function.emitError("explicit endpoint preparation constructed later forms: normalization=")
            << session.normalizationConstructions() << ", forms=" << session.result()->formConstructions
            << ", arithmetic=" << session.arithmeticGeneratorConstructions()
            << ", preflights=" << session.numericTemplatePreflights();
    }
    if (failed(materializeLogicalFamilies(**prepared))) { return failure(); }
    std::size_t endpointIndex = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> namespaces;
    bool repeatedNamespace = false;
    for (auto [record, edge] : llvm::enumerate(explicitAnalysis.reduction.retained)) {
        const auto sourcePipe = explicitAnalysis.occurrences[edge.source].pipe;
        const auto targetPipe = explicitAnalysis.occurrences[edge.target].pipe;
        // Explicit records have zero displacement. The first retained record
        // for a direction owns its namespace; member labels remain distinct.
        auto space = namespaces.emplace(std::make_pair(sourcePipe, targetPipe), static_cast<uint32_t>(record));
        repeatedNamespace |= !space.second;
        auto* source = explicitAnalysis.phases[edge.source]->elementOp;
        auto* target = explicitAnalysis.phases[edge.target]->elementOp;
        const bool local = sourcePipe == targetPipe;
        for (unsigned side = 0; side < (local ? 1U : 2U); ++side) {
            if (endpointIndex >= (*prepared)->endpoints.size()) { return failure(); }
            const auto& endpoint = (*prepared)->endpoints[endpointIndex++];
            const auto kind = local ? LogicalCommandKind::Barrier :
                side ? LogicalCommandKind::Wait : LogicalCommandKind::Set;
            auto guard = endpoint.guard.getDefiningOp<arith::ConstantOp>();
            if (endpoint.kind != kind || endpoint.before != (local || side ? target : source->getNextNode()) ||
                endpoint.record != static_cast<int64_t>(space.first->second) ||
                endpoint.records != SmallVector<uint32_t>{static_cast<uint32_t>(record)} ||
                endpoint.sourcePipe != sourcePipe || endpoint.targetPipe != targetPipe || !guard ||
                !cast<IntegerAttr>(guard.getValue()).getValue().isOne()) { return failure(); }
            if (!local) {
                auto ordinal = endpoint.identity.getDefiningOp<arith::ConstantOp>();
                auto member = endpoint.memberCoordinates.size() == 1 ?
                    endpoint.memberCoordinates.front().getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
                if (!ordinal || !cast<IntegerAttr>(ordinal.getValue()).getValue().isZero() || !member ||
                    cast<IntegerAttr>(member.getValue()).getInt() != static_cast<int64_t>(record)) { return failure(); }
            }
        }
    }
    if (endpointIndex != (*prepared)->endpoints.size()) { return failure(); }
    if (function->hasAttr("test.repeated_explicit_namespace") && !repeatedNamespace) {
        return function.emitError("explicit fixture did not retain repeated-direction original records");
    }
    if (function->hasAttr("test.repeated_explicit_namespace")) {
        auto incomplete = prepareExplicitInsertion(function, explicitAnalysis, false);
        if (failed(incomplete) || (*incomplete)->families.size() < 2) { return failure(); }
        (*incomplete)->families.pop_back();
        const auto recipe = (*incomplete)->familyRecipe;
        if (succeeded(normalizeLogicalFamilies(**incomplete)) || (*incomplete)->familyRecipe != recipe ||
            (*incomplete)->familyProtocol || !(*incomplete)->preparation.empty() ||
            !(*incomplete)->endpoints.empty()) {
            return function.emitError("incomplete explicit original-record directory was accepted or mutated");
        }
        auto& endpoint = (*prepared)->endpoints.front();
        const auto canonical = endpoint.record;
        endpoint.record = INT64_MAX;
        std::string before, after;
        llvm::raw_string_ostream original(before); function.print(original); original.flush();
        LogicalResult refused = success();
        {
            ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
            refused = insertLogicalSynchronization(function, **prepared);
        }
        llvm::raw_string_ostream unchanged(after); function.print(unchanged); unchanged.flush();
        endpoint.record = canonical;
        if (succeeded(refused) || before != after) {
            return function.emitError("malformed explicit namespace escaped strict insertion or changed original IR");
        }
    }
    // Auditing every child is separate from constructing this selected root.
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
    const auto freshBuilds = fresh.numericalRegionConstructions();
    auto diagnose = [&](StringRef stage, const AnalysisOutcome& a, const AnalysisOutcome& b) {
        llvm::errs() << "numerical " << stage << ": fresh="
            << (a.mathematical ? a.mathematical->backend : "none") << " status=" << unsigned(a.status)
            << " queries=" << a.available.queries << " selectors=" << a.available.selectors
            << " sync=" << a.available.synchronization << " builds=" << fresh.numericalRegionConstructions()
            << "; warm=" << (b.mathematical ? b.mathematical->backend : "none")
            << " status=" << unsigned(b.status) << " queries=" << b.available.queries
            << " selectors=" << b.available.selectors << " sync=" << b.available.synchronization
            << " builds=" << warm.numericalRegionConstructions() << "\n";
    };
    const bool ready = first.status == AnalysisStatus::Ready && second.status == AnalysisStatus::Ready &&
        firstExports && secondExports && first.available.queries && first.available.selectors &&
        second.available.queries && second.available.selectors && retained.mathematical &&
        warm.analyzeNumericalRegion({id}).mathematical == retained.mathematical &&
        warm.numericalRegionConstructions() == 1 && freshBuilds <= 1;
    if (!ready) {
        diagnose("exports", first, second);
        llvm::errs() << "anchors: fresh=" << (firstExports ? firstExports->anchors.size() : 0)
            << " warm=" << (secondExports ? secondExports->anchors.size() : 0)
            << " retained=" << bool(retained.mathematical) << "\n";
        return function.emitError("numerical export adapter depends on request history");
    }
    // A cold request may select an exact cheaper producer without building the
    // numerical form. Subsequent identical requests must retain that answer.
    const auto repeatFirst = fresh.analyze(request), repeatSecond = warm.analyze(request);
    if (repeatFirst.mathematical != first.mathematical || repeatSecond.mathematical != second.mathematical ||
        repeatFirst.regionalExports != first.regionalExports ||
        repeatSecond.regionalExports != second.regionalExports ||
        fresh.numericalRegionConstructions() != freshBuilds || warm.numericalRegionConstructions() != 1) {
        return function.emitError("numerical export retry rebuilt or replaced the retained answer");
    }
    numerical_request_oracle::Oracle oracle(function);
    if (!oracle.initialize(*warm.input(), cast<scf::ForOp>(node->anchor))) {
        return function.emitError("numerical oracle requires unique original payloads in unconditional counted loops");
    }
    for (uint64_t upper : {0, 1, 3, 8}) {
        if (!oracle.compare(*firstExports, *secondExports, upper)) {
            diagnose("original-occurrence equivalence", first, second);
            llvm::errs() << "numerical oracle upper=" << upper << " fresh-anchors=" << firstExports->anchors.size()
                << " warm-anchors=" << secondExports->anchors.size() << "\n";
            return function.emitError(
                "numerical adapters disagree on original occurrence presence, extrema or queries");
        }
    }
    request.needs.synchronization = true;
    const auto logicalFirst = fresh.analyze(request), logicalSecond = warm.analyze(request);
    const bool logical = logicalFirst.status == AnalysisStatus::Ready &&
        logicalSecond.status == AnalysisStatus::Ready && logicalFirst.mathematical && logicalSecond.mathematical &&
        logicalFirst.available.queries && logicalFirst.available.selectors && logicalFirst.available.synchronization &&
        logicalSecond.available.queries && logicalSecond.available.selectors &&
        logicalSecond.available.synchronization &&
        warm.numericalRegionConstructions() == 1 && fresh.numericalRegionConstructions() == freshBuilds;
    if (!logical) {
        diagnose("logical", logicalFirst, logicalSecond);
        return function.emitError("numerical logical adapter depends on request history");
    }
    auto weakRequest = request; weakRequest.needs.synchronization = false;
    const auto stillFirst = fresh.analyze(weakRequest), stillSecond = warm.analyze(weakRequest);
    if (stillFirst.mathematical != first.mathematical || stillSecond.mathematical != second.mathematical ||
        stillFirst.regionalExports != first.regionalExports || stillSecond.regionalExports != second.regionalExports) {
        return function.emitError("strong logical request replaced retained weaker mathematics or exports");
    }
    const auto strongFirst = logicalFirst.regionalExports ? logicalFirst.regionalExports :
        logicalFirst.mathematical->regionalDemands;
    const auto strongSecond = logicalSecond.regionalExports ? logicalSecond.regionalExports :
        logicalSecond.mathematical->regionalDemands;
    if (!strongFirst || !strongSecond) { return failure(); }
    for (uint64_t bound : {0, 1, 3, 8}) {
        if (!oracle.compare(*firstExports, *strongFirst, bound) ||
            !oracle.compare(*secondExports, *strongSecond, bound)) {
            return function.emitError("strong logical export changed retained original-occurrence mathematics");
        }
    }
    const auto coldTrace = numerical_request_oracle::numericalLogicalTraces(
        function, policy, id, false, logicalFirst.mathematical->backend);
    const auto warmTrace = numerical_request_oracle::numericalLogicalTraces(
        function, policy, id, true, logicalSecond.mathematical->backend);
    if (!coldTrace || !warmTrace || *coldTrace != *warmTrace) {
        return function.emitError("numerical cold/warm logical output changed payloads, matching or command causality");
    }
    if (!function->hasAttr("test.numerical_external")) {
        request.region = 0;
        const auto rootFirst = fresh.analyze(request), rootSecond = warm.analyze(request);
        const bool rootReady = rootFirst.status == AnalysisStatus::Ready &&
            rootSecond.status == AnalysisStatus::Ready &&
            warm.numericalRegionConstructions() == 1 && fresh.numericalRegionConstructions() == freshBuilds;
        if (!rootReady) {
            diagnose("root", rootFirst, rootSecond);
            return function.emitError("numerical root/child export adapters disagree");
        }
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
mlir::LogicalResult checkSliceAllocation(mlir::func::FuncOp function,
    mlir::pto::frontiersynch::FrontierAnalysis& session)
{
    using namespace mlir::pto::frontiersynch;
    PhaseIndex index;
    if (failed(index.build(function, *session.input()))) { return failure(); }
    auto loop = *function.getOps<scf::ForOp>().begin();
    auto recognized = recognizeGuardedRotating(loop, index, *session.input(), session.input()->accesses());
    auto arena = std::make_shared<RegionExpressions>();
    auto analysis = analyzeGuardedRotating(loop, *session.input(), recognized, index, arena);
    if (!analysis.error.empty()) { return function.emitError(analysis.error); }
    unsigned checked = 0;
    for (uint64_t begin = 0; begin <= 5; ++begin) {
        for (uint64_t end = begin; end <= 10; ++end) {
            std::string error;
            auto region = guardedRotatingRegionalResult(function, *session.input(), analysis, error,
                PeriodicSlice{arena->constant(begin), arena->constant(end)});
            if (failed(region)) { return function.emitError(error); }
            auto plan = region->prepare();
            if (failed(plan)) { return failure(); }
            auto allocation = requestAllocationSummary(**plan, AllocationStrength::Shared);
            if (!allocation || allocation->groups.empty()) { return failure(); }
            using Point = std::pair<uint32_t, uint64_t>;
            for (const auto& group : allocation->groups) {
                if (group.lanes.size() != group.budget) { return failure(); }
                for (uint64_t lane = 0; lane < group.budget; ++lane) {
                    std::multiset<Point> first, last;
                    // Independent finite enumeration uses the certified ID
                    // formula, not the congruence/span extraction helper.
                    for (const auto& member : group.members) {
                        const auto& edge = analysis.generators[member.record];
                        auto distance = arena->constantValue(edge.displacement);
                        if (!distance) { return failure(); }
                        std::vector<uint64_t> visits;
                        for (uint64_t visit = begin; visit < end; ++visit) {
                            if (*distance < end - visit &&
                                (member.stride * visit + member.phase) % group.budget == lane) {
                                visits.push_back(visit);
                            }
                        }
                        if (!visits.empty()) {
                            first.insert({edge.source, visits.front()});
                            last.insert({edge.target, visits.back() + *distance});
                        }
                    }
                    auto matches = [&](const auto& selectors, const auto& expected, PeriodicEventKind kind) {
                        std::multiset<Point> actual;
                        for (const auto& selector : selectors) {
                            auto active = arena->constantValue(selector.present);
                            if (!active) { return false; }
                            if (!*active) { continue; }
                            auto ordinal = arena->constantValue(selector.event.ordinal);
                            if (!ordinal || selector.event.kind != kind) { return false; }
                            actual.insert({selector.event.type, *ordinal});
                        }
                        return actual == expected;
                    };
                    if (!matches(group.lanes[lane].firstSources, first, PeriodicEventKind::Start) ||
                        !matches(group.lanes[lane].lastTargets, last, PeriodicEventKind::Completion)) {
                        return function.emitError("periodic slice lane extrema disagree with finite enumeration");
                    }
                }
            }
            ++checked;
        }
    }
    llvm::outs() << "periodic-slice-allocation: " << checked << " finite intervals checked\n";
    return success();
}
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
    // A retained partition must not authorize a child whose endpoints changed.
    auto certificate = std::make_shared<BoundaryPartition>();
    certificate->loop = loop;
    certificate->expressions = arena;
    certificate->intervals = {{arena->constant(0), arena->constant(2)},
                              {arena->constant(2), arena->constant(4)}};
    for (bool stale : {false, true}) {
        auto state = std::make_shared<SequenceAnalysisState>(function, arena);
        if (failed(state->index.build(function, *session.input()))) { return failure(); }
        state->children.resize(2);
        for (std::size_t i = 0; i < 2; ++i) {
            auto& child = state->children[i];
            child.regional = i == 0 || stale ? *first : *last;
            child.anchors = child.regional.anchors;
            child.boundaryPartition = certificate;
            child.boundarySlice = i;
        }
        if (state->importSummaries(false) == stale) {
            return function.emitError("partition evidence was ignored or survived changed slice endpoints");
        }
    }
    llvm::outs() << "deferred-writer-slices: disjoint-visits accepted overlapping-visits refused\n";
    llvm::outs() << "deferred-writer-slices: retained partition endpoints revalidated\n";
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
        auto finite = *view;
        // This fixture has one payload type per child. Its GM accesses cannot
        // need an internal cross-pipe handoff. Present their already exact
        // extrema as deferred effects, independently of which producer won.
        // The parent must still establish every cross-child relationship.
        if (symbolic.anchors.size() != 1) {
            return function.emitError("uniform deferred fixture requires one payload type per child");
        }
        llvm::erase_if(symbolic.accessBoundary, [&](const auto& access) {
            const auto& effect = symbolic.accessModel->effects()[access.effect];
            if (!effect.memory || effect.memory->scope != pto::AddressSpace::GM) { return false; }
            auto deferred = access;
            deferred.representedByCells = false;
            symbolic.deferredAccessBoundary.push_back(std::move(deferred));
            return true;
        });
        llvm::erase_if(finite.storageBoundary, [](const auto& boundary) {
            return boundary.cell.space == pto::AddressSpace::GM;
        });
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
            if (address.space == pto::AddressSpace::GM) { return RegionalStorageSelectors{}; }
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
        if (children.front().deferredAccessBoundary.empty()) {
            return function.emitError("uniform circuit fixture lacks the deferred extrema under test");
        }
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
    if (function->hasAttr("test.slice_allocation")) { return checkSliceAllocation(function, session); }
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

namespace {
LogicalResult checkNumericGeometryPolicy(func::FuncOp function, pto::GMAliasPolicy alias)
{
    using namespace pto::frontiersynch;
    scf::ForOp outer;
    for (auto loop : function.getOps<scf::ForOp>()) { outer = loop; }
    if (!outer) { return function.emitError("numeric policy fixture requires an outer loop"); }
    TemplateGeometryConstant geometry = [&](Value value) -> std::optional<int64_t> {
        return value == outer.getInductionVar() ? std::optional<int64_t>(0) : std::nullopt;
    };
    const TemplateControlConstant control;
    const auto preserved = TemplateGeometryPolicy::PreserveGlobalCoordinates;
    const auto legacy = TemplateGeometryPolicy::AllCertifiedMaps;
    for (bool reverse : {false, true}) {
        FrontierAnalysis session(function);
        if (failed(session.initialize(alias))) { return failure(); }
        std::shared_ptr<const NumericBodyMathematics> owned;
        for (auto policy : {reverse ? preserved : legacy, reverse ? legacy : preserved}) {
            std::string error;
            const auto before = session.specializedNumericConstructions();
            auto result = session.specializedNumericDemands(outer, geometry, control, error, {}, {}, policy);
            const auto diagnostic = error;
            auto retry = session.specializedNumericDemands(outer, geometry, control, error, {}, {}, policy);
            const bool cached = retry == result && error == diagnostic &&
                session.specializedNumericConstructions() == before + 1;
            if (!cached) { return function.emitError("numeric geometry policy attempt was not cached independently"); }
            if (policy == preserved && !result) {
                return function.emitError("original GM coordinate policy failed: " + error);
            }
            if (!result) {
                if (error.empty()) { return function.emitError("legacy numeric refusal lost its obligation"); }
                continue;
            }
            if (result->body.geometryPolicy != policy) {
                return function.emitError("numeric cache reused another geometry policy");
            }
            bool gmWrite = false;
            for (const auto& payload : result->body.payloads) {
                for (const auto& effect : payload.effects) {
                    const auto& original = result->inputOwner->accesses().effects()[effect.sourceEffect];
                    if (original.memory->scope != pto::AddressSpace::GM ||
                        original.mode != pto::SyncAccessMode::Write) { continue; }
                    gmWrite = true;
                    if (effect.regions.empty()) {
                        return function.emitError("numeric policy discarded original GM map");
                    }
                    for (const auto& region : effect.regions) {
                        if (region.byteOffset.isFunctionOfSymbol(0) != (policy == preserved)) {
                            return function.emitError("numeric policy bound or retained the wrong GM coordinate");
                        }
                    }
                    if (policy == preserved &&
                        (effect.discharge != TemplateDischarge::IterationPrivateBase || effect.outerStride != 32768)) {
                        return function.emitError("numeric GM discharge lost its exact visit stride");
                    }
                }
            }
            if (!gmWrite) { return function.emitError("numeric geometry policy fixture had no GM write"); }
            if (policy == preserved) { owned = result; }
        }
        if (session.specializedNumericConstructions() != 2 || !owned) {
            return function.emitError("numeric policies did not build exactly one result each");
        }
        const auto other = alias == pto::GMAliasPolicy::MayAlias ?
            pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
        if (failed(session.initialize(other)) || session.specializedNumericConstructions()) {
            return function.emitError("numeric policy cache survived alias reset");
        }
        std::string error;
        for (auto policy : {legacy, preserved}) {
            auto result = session.specializedNumericDemands(outer, geometry, control, error, {}, {}, policy);
            if (policy == preserved && (!result || result == owned || result->inputOwner == owned->inputOwner)) {
                return function.emitError("numeric policy reused old alias-context mathematics");
            }
        }
        if (session.specializedNumericConstructions() != 2) {
            return function.emitError("alias reset did not clear both numeric policy attempts");
        }
    }
    llvm::outs() << "numeric-geometry-policy: both-orders separate-attempts physical-GM-coordinate alias-reset\n";
    return success();
}
} // namespace

LogicalResult checkSpecializedNumericSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    if (function->hasAttr("test.specialized_numeric_geometry_policy")) {
        return checkNumericGeometryPolicy(function, policy);
    }
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
        absent->body.payloads.empty() && absent->demands->reduction.payloads.empty() && !unknown &&
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
                        auto expected = explicitEventPrecedes(first->demands->reduction, {a, ak}, {b, bk});
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
        if (!(*plan)->preparation.empty() || !(*plan)->endpoints.empty() ||
            (!(*plan)->families.empty() && !(*plan)->familyRecipe)) {
            return function.emitError("guarded periodic preparation materialized scalar endpoints");
        }
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
