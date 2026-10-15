// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "AnalysisSessionInternal.h"
#include "WholeInvocationIdentity.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInsertion.h"
#include "Shared/NormalizedControl.h"
#include "Structured/NumericTemplatePlan.h"
#include "PTO/Transforms/FrontierSynch/Structured/GuardedPeriodicWork.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <tuple>
#include <numeric>
namespace mlir::pto::frontiersynch {
AnalysisProfileScope::AnalysisProfileScope(std::shared_ptr<AnalysisLedger> owner, StringRef stage,
    std::size_t region, StringRef method, bool newRequest, uint64_t originatingRequest)
{
    if (!owner) { return; }
    ledger = std::move(owner);
    previousRequest = ledger->currentRequest;
    ownsRequest = newRequest || !previousRequest;
    if (ownsRequest && ledger->nextRequest != UINT64_MAX) { request = ++ledger->nextRequest; }
    else if (!ownsRequest) { request = previousRequest; }
    if (ownsRequest) { ledger->currentRequest = request; }
    if (!ledger->profiling) { return; }
    timed = true;
    AnalysisStageRecord record;
    record.id = ledger->stages.size() + 1;
    record.region = region;
    record.stage = stage.str();
    record.method = method.str();
    record.request = request;
    record.parentRequest = originatingRequest;
    if (!ledger->activeStages.empty()) {
        const auto& parent = ledger->stages[ledger->activeStages.back()];
        record.parent = parent.id;
        record.parentRequest = parent.request;
    }
    index = ledger->stages.size();
    ledger->stages.push_back(std::move(record));
    ledger->activeStages.push_back(index);
    started = std::chrono::steady_clock::now();
}
AnalysisProfileScope::~AnalysisProfileScope()
{
    if (!ledger) { return; }
    if (ownsRequest) { ledger->currentRequest = previousRequest; }
    if (!timed) { return; }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    auto& record = ledger->stages[index];
    record.elapsedNanoseconds = static_cast<uint64_t>(elapsed);
    // While active, exclusiveNanoseconds accumulates immediate child durations.
    record.exclusiveNanoseconds = record.elapsedNanoseconds - record.exclusiveNanoseconds;
    ledger->activeStages.pop_back();
    if (!ledger->activeStages.empty()) {
        accumulateCost(ledger->stages[ledger->activeStages.back()].exclusiveNanoseconds,
            record.elapsedNanoseconds);
    }
}
uint64_t AnalysisProfileScope::requestId() const
{
    return request;
}
void AnalysisProfileScope::constructed()
{
    if (timed) {
        auto& count = ledger->stages[index].constructions;
        count = estimatedAdd(count.value_or(0), 1);
    }
}
void AnalysisProfileScope::dispatched()
{
    if (timed) { accumulateCost(ledger->stages[index].firstDispatches, 1); }
}
void AnalysisProfileScope::cacheHit()
{
    if (timed) { accumulateCost(ledger->stages[index].cacheHits, 1); }
}
void AnalysisProfileScope::recordWork(std::vector<std::pair<std::string, uint64_t>> work)
{
    if (timed) { ledger->stages[index].work = std::move(work); }
}

StringRef analysisBackendName(AnalysisBackend backend)
{
    switch (backend) {
    case AnalysisBackend::Explicit: return "explicit";
    case AnalysisBackend::NumericalPeriodic: return "numerical-periodic";
    case AnalysisBackend::Rotating: return "rotating";
    case AnalysisBackend::MixedStride: return "mixed-stride";
    case AnalysisBackend::GuardedRotating: return "guarded-rotating";
    case AnalysisBackend::BoundedLifetime: return "bounded-lifetime";
    case AnalysisBackend::Sequence: return "sequence";
    case AnalysisBackend::SequenceEndpoints: return "sequence-endpoints";
    case AnalysisBackend::NumericalSequence: return "numerical-sequence";
    case AnalysisBackend::VaryingBoundary: return "varying-boundary";
    case AnalysisBackend::FiniteVisit: return "finite-visit";
    case AnalysisBackend::ExpandedFinite: return "expanded-finite";
    case AnalysisBackend::ArithmeticPeriodic: return "arithmetic-periodic";
    case AnalysisBackend::Arithmetic: return "arithmetic";
    case AnalysisBackend::FiniteGuarded: return "finite-guarded";
    case AnalysisBackend::CompactBounding: return "compact-bounding";
    }
    llvm_unreachable("invalid analysis backend");
}
void FrontierAnalysis::enableStageProfiling(bool enabled)
{
    profileStages = enabled;
    if (sessionState) { sessionState->ledger->profiling = enabled; }
}
std::shared_ptr<const AnalysisLedger> FrontierAnalysis::diagnosticLedger() const
{
    return sessionState ? sessionState->ledger : nullptr;
}
void FrontierAnalysis::writeStageProfile(llvm::raw_ostream& stream, StringRef workload) const
{
    if (!sessionState || !sessionState->ledger->profiling) { return; }
    llvm::json::Array stages;
    for (const auto& record : sessionState->ledger->stages) {
        llvm::json::Object entry{{"id", record.id}, {"request", record.request},
            {"parent", record.parent}, {"parent_request", record.parentRequest},
            {"region", record.region}, {"stage", record.stage}, {"method", record.method},
            {"elapsed_ns", record.elapsedNanoseconds}, {"exclusive_ns", record.exclusiveNanoseconds},
            {"cache_hits", record.cacheHits}, {"first_dispatches", record.firstDispatches}};
        entry["constructions"] = record.constructions ? llvm::json::Value(*record.constructions) :
            llvm::json::Value(nullptr);
        if (record.aggregateCalls) {
            entry["aggregate_calls"] = record.aggregateCalls;
            entry["aggregate_cache_hits"] = record.aggregateCacheHits;
            entry["aggregate_cache_misses"] = record.aggregateCacheMisses;
        }
        if (!record.work.empty()) {
            llvm::json::Object work;
            for (const auto& counter : record.work) { work[counter.first] = counter.second; }
            entry["work"] = std::move(work);
        }
        stages.push_back(std::move(entry));
    }
    const auto* recognition = result();
    llvm::json::Array selections;
    for (const auto& selection : sessionState->ledger->selections) {
        auto strings = [](const std::vector<std::string>& values) {
            llvm::json::Array array;
            for (const auto& value : values) { array.push_back(value); }
            return array;
        };
        llvm::json::Object entry{{"request", selection.request}, {"region", selection.region},
            {"method", selection.method}, {"reason", selection.reason},
            {"reused", strings(selection.reused)}, {"missing", strings(selection.missing)},
            {"retained_unrequested", strings(selection.retainedUnrequested)},
            {"refused", strings(selection.refused)}, {"attempted", selection.attempted},
            {"failure_stage", selection.failureStage}};
        entry["unbuilt_work"] = selection.unbuiltWork ? llvm::json::Value(*selection.unbuiltWork) :
            llvm::json::Value(nullptr);
        selections.push_back(std::move(entry));
    }
    llvm::json::Object audit;
    if (recognition) {
        const auto& work = recognition->contractAuditWork;
        audit["refresh_requests"] = work.refreshRequests;
        audit["materializations"] = work.materializations;
        audit["node_visits"] = work.nodeVisits;
        audit["candidates_published"] = work.candidatesPublished;
    }
    auto profiledFunction = function;
    stream << "frontier-profile " << llvm::json::Value(llvm::json::Object{
        {"function", profiledFunction.getSymName()}, {"workload", workload},
        {"cost_snapshot_records_copied", sessionState->ledger->snapshotRecordsCopied},
        {"contract_audit", std::move(audit)},
        {"canonical_arithmetic_generators", arithmeticGeneratorConstructions()},
        {"canonical_arithmetic_regions", arithmeticRegionConstructions()},
        {"canonical_numerical_regions", numericalRegionConstructions()},
        {"normalization_source_visits", normalizationSourceVisits()},
        {"regional_normalization_views", normalizationConstructions()},
        {"form_constructions", recognition ? recognition->formConstructions : 0},
        {"selections", std::move(selections)}, {"stages", std::move(stages)}}) << "\n";
}
namespace {
// Estimate joins from certified primitive descriptions before projection or
// generator construction. These are selection estimates, not output bounds.
const ArithmeticPrimitiveShape& primitiveShape(const AnalysisSessionState& session, const ArithmeticProgram& program)
{
    auto found = session.primitiveShapes.find(&program);
    if (found != session.primitiveShapes.end()) { return found->second; }
    ArithmeticPrimitiveShape shape;
    accumulateCost(session.primitiveShapeBuilds, 1);
    shape.domains = program.sites.size();
    shape.dimensions = program.recognition.observedDimensions;
    EstimatedCount reads = 0, writes = 0, prerequisite = 0, rows = 0, pieces = 0;
    for (const auto& relation : program.primitives.relations) {
        accumulateCost(session.primitiveShapeRelationVisits, 1);
        const auto count = EstimatedCount(relation.pieces.size());
        if (relation.kind == PrimitiveKind::Reads) { reads = estimatedAdd(reads, count); }
        if (relation.kind == PrimitiveKind::Writes) { writes = estimatedAdd(writes, count); }
        if (relation.kind == PrimitiveKind::Prerequisites) { prerequisite = estimatedAdd(prerequisite, count); }
        if (relation.kind == PrimitiveKind::Native) { shape.native = estimatedAdd(shape.native, count); }
        pieces = estimatedAdd(pieces, count);
        for (const auto& piece : relation.pieces) {
            accumulateCost(session.primitiveShapePieceVisits, 1);
            EstimatedCount constraints;
            if (piece.machineSystem) {
                constraints = estimatedAdd(piece.machineSystem->constraints().size(),
                                          piece.machineSystem->congruences().size());
            } else if (piece.system != IntegerSet{}) {
                constraints = piece.system.getNumConstraints();
            }
            rows = estimatedAdd(rows, constraints);
        }
    }
    shape.pieces = estimatedAdd(prerequisite, estimatedMultiply(writes, estimatedAdd(reads, writes)));
    // Joining accesses includes their domain/order descriptions. Projection can
    // increase the normalized description; use its class dimensions in costs.
    const auto averageRows = rows && pieces && *pieces ?
        EstimatedCount(*rows / *pieces + uint64_t(*rows % *pieces != 0)) : EstimatedCount{};
    shape.rows = estimatedMultiply(estimatedAdd(shape.pieces, shape.native),
        estimatedMultiply(averageRows, 3));
    return session.primitiveShapes.emplace(&program, std::move(shape)).first->second;
}
EstimatedCount generatorConstructionWork(const ArithmeticPrimitiveShape& shape)
{
    const auto dimension = estimatedAdd(shape.dimensions, 1);
    // Price primitive imports and projected conflict/prerequisite joins as a
    // separate stage. These are structural work estimates, not relation bounds.
    return estimatedAdd(estimatedMultiply(shape.rows, dimension),
        estimatedMultiply(shape.pieces, estimatedPower(dimension, 3)));
}
std::pair<AnalysisCostEstimate, AnalysisCostEstimate> estimates(
    const ArithmeticProgram& program, const ArithmeticPrimitiveShape& shape, const AnalysisNeeds& needs,
    bool includeGenerators = true)
{
    AnalysisCostEstimate periodic, arithmetic;
    const auto types = estimatedMultiply(program.sites.size(), program.primitives.period);
    const auto vertices = estimatedMultiply(2, types);
    const auto pipes = EstimatedCount(program.primitives.pipeCount);
    const auto generators = estimatedAdd(shape.pieces, shape.native);
    const auto phaseCopies = estimatedMultiply(program.primitives.period, program.primitives.period);
    const auto liftedDemands = estimatedMultiply(shape.pieces, phaseCopies);
    const auto liftedNative = estimatedMultiply(shape.native, phaseCopies);
    const auto dimension = estimatedAdd(shape.dimensions, 1);
    const auto quotient = guardedPeriodicWork(types, pipes, liftedDemands, liftedNative);
    periodic.generatorPieces = arithmetic.generatorPieces = shape.pieces;
    periodic.ports = arithmetic.ports = vertices;
    periodic.numericalWindow = arithmetic.numericalWindow = 0;
    periodic.circuitNodes = estimatedAdd(liftedDemands,
        estimatedAdd(quotient.relaxations, estimatedAdd(quotient.nativeScanVisits, quotient.exclusions)));
    // Grouping/winner expressions are linear in raw records. Charge ordered
    // map comparisons separately without building or deduplicating candidates.
    EstimatedCount groupLevels;
    if (liftedDemands) {
        groupLevels = 0;
        for (auto count = *liftedDemands; count; count >>= 1) { ++*groupLevels; }
    }
    periodic.relationConversion = estimatedAdd(estimatedMultiply(shape.rows, dimension),
        estimatedMultiply(estimatedMultiply(generators, shape.domains), estimatedMultiply(dimension, dimension)));
    // DBM adaptation validates the sparse row description with one closure.
    if (program.recognition.arithmeticClass == ArithmeticClass::Differences) {
        periodic.relationConversion = estimatedAdd(periodic.relationConversion,
            estimatedMultiply(generators, estimatedPower(dimension, 3)));
    }
    periodic.relationConversion = estimatedMultiply(periodic.relationConversion, phaseCopies);
    periodic.work = estimatedAdd(estimatedMultiply(liftedDemands, groupLevels),
        estimatedAdd(periodic.relationConversion, estimatedMultiply(periodic.circuitNodes, 2)));
    periodic.representation = estimatedAdd(periodic.circuitNodes, estimatedMultiply(pipes, vertices));
    const auto arrangement = estimatedPower(estimatedAdd(shape.pieces, 1),
        estimatedMultiply(shape.dimensions, dimension));
    arithmetic.work = estimatedAdd(estimatedMultiply(estimatedMultiply(generators, generators),
        estimatedMultiply(estimatedAdd(pipes, 1), estimatedPower(dimension, 3))), arrangement);
    arithmetic.representation = estimatedAdd(generators, arrangement);
    arithmetic.relationConversion = arithmetic.circuitNodes = 0;
    const uint64_t interfaces = uint64_t(needs.queries) + uint64_t(needs.selectors) +
                                uint64_t(needs.synchronization);
    const auto exports = estimatedMultiply(interfaces, estimatedMultiply(generators, vertices));
    periodic.work = estimatedAdd(periodic.work, exports);
    arithmetic.work = estimatedAdd(arithmetic.work, estimatedMultiply(exports, dimension));
    if (includeGenerators) {
        const auto generatorsWork = generatorConstructionWork(shape);
        periodic.work = estimatedAdd(periodic.work, generatorsWork);
        arithmetic.work = estimatedAdd(arithmetic.work, generatorsWork);
    }
    periodic.representation = estimatedAdd(periodic.representation, exports);
    arithmetic.representation = estimatedAdd(arithmetic.representation, exports);
    return {periodic, arithmetic};
}
const BackendAttempt* namedAttempt(const AnalysisSessionState& session, std::size_t region,
    AnalysisBackend backend)
{
    const auto scope = session.attempts.find(region);
    if (scope == session.attempts.end()) { return nullptr; }
    const auto found = scope->second.find(backend);
    return found == scope->second.end() ? nullptr : &found->second;
}
const BackendAttempt* loopAttempt(const AnalysisSessionState& session, std::size_t region,
    AnalysisBackend backend)
{
    const auto found = session.loopAttempts.find({region, backend});
    return found == session.loopAttempts.end() ? nullptr : &found->second;
}
RetainedStage retainedStage(bool attempted, bool available)
{
    return available ? RetainedStage::Ready : (attempted ? RetainedStage::Refused : RetainedStage::Missing);
}
RetainedBackendWork retainedBackendWork(const AnalysisSessionState& session, const ProgramRecognition& program,
    std::size_t region, std::optional<std::size_t> canonical, AnalysisBackend backend,
    const AnalysisNeeds& needs, const ArithmeticProgram* arithmetic, std::size_t depth = 0)
{
    RetainedBackendWork result;
    const auto* named = namedAttempt(session, region, backend);
    const BackendAttempt* owner = named;
    const auto normalized = session.normalizedInputs.find(canonical.value_or(region));
    const auto* representation = normalized == session.normalizedInputs.end() ? nullptr : normalized->second.get();
    if (!owner || !owner->produced) {
        const BackendAttempt* retained = nullptr;
        if (canonical) { retained = loopAttempt(session, *canonical, backend); }
        if (backend == AnalysisBackend::Arithmetic && region) {
            const auto found = session.arithmeticRegionAttempts.find(region);
            if (found != session.arithmeticRegionAttempts.end()) { retained = &found->second; }
        }
        if (backend == AnalysisBackend::ExpandedFinite && representation) {
            const auto found = session.expandedAttempts.find(representation);
            if (found != session.expandedAttempts.end()) { retained = &found->second; }
        }
        if (retained && retained->produced) { owner = retained; }
    }
    result.evidence = {named, owner, representation, arithmetic};
    const auto* math = owner ? owner->mathematical.get() : nullptr;
    bool identityNeedsQueries = false;
    result.evidence.push_back(math);
    result.stages[0] = retainedStage(owner && owner->produced, math != nullptr);
    if (math && math->identityComposition && depth < program.nodes.size()) {
        const auto& identity = *math->identityComposition;
        auto childNeeds = needs;
        childNeeds.synchronization = false;
        auto child = retainedBackendWork(session, program, identity.childRegion, identity.childRegion,
            identity.childBackend, childNeeds, nullptr, depth + 1);
        result.stages[1] = child.stages[1]; result.stages[2] = child.stages[2];
        result.work[1] = child.work[1]; result.work[2] = child.work[2];
        identityNeedsQueries = child.required[1];
        result.evidence.insert(result.evidence.end(), child.evidence.begin(), child.evidence.end());
    } else if (math && !math->identityComposition) {
        const auto available = mathematicalExports(*math);
        result.stages[1] = retainedStage(true, available.queries);
        result.stages[2] = retainedStage(true, available.selectors);
        const BackendAttempt* exports = nullptr;
        if (math->boundedDemands) {
            const auto found = llvm::find_if(program.nodes, [&](const auto& node) {
                return node.anchor == math->boundedDemands->loop.getOperation();
            });
            if (found != program.nodes.end()) {
                exports = loopAttempt(session, std::distance(program.nodes.begin(), found),
                    AnalysisBackend::BoundedLifetime);
            }
            result.stages[1] = retainedStage(exports && exports->boundedQueriesAttempted,
                exports && bool(exports->boundedQueries));
            result.stages[2] = retainedStage(exports && exports->boundedSelectorsAttempted,
                exports && bool(exports->boundedSelectors));
        } else if (math->arithmeticRegionalDemands) {
            const auto found = session.arithmeticRegionAttempts.find(region);
            if (found != session.arithmeticRegionAttempts.end()) { exports = &found->second; }
            result.stages[1] = retainedStage(exports && exports->arithmeticQueriesAttempted,
                exports && bool(exports->arithmeticQueries));
            result.stages[2] = retainedStage(exports && exports->arithmeticSelectorsAttempted,
                exports && bool(exports->arithmeticSelectors));
        } else if (math->varyingNode) {
            exports = loopAttempt(session, *math->varyingNode, AnalysisBackend::VaryingBoundary);
            result.stages[1] = retainedStage(exports && exports->varyingQueriesAttempted,
                exports && bool(exports->varyingQueries));
            result.stages[2] = retainedStage(exports && exports->varyingSelectorsAttempted,
                exports && bool(exports->varyingSelectors));
        } else if (math->backend == "expanded-finite-guarded") {
            exports = named;
            result.stages[1] = retainedStage(exports && bool(exports->expandedQueries),
                exports && exports->expandedQueries && exports->expandedQueries->capabilities.exactQueries);
            result.stages[2] = retainedStage(exports && exports->expandedSelectorsAttempted,
                exports && exports->expandedSelectors && exports->expandedSelectors->capabilities.exactSelectors);
        }
        result.evidence.push_back(exports);
    }
    result.stages[3] = retainedStage(named && named->endpoints.has_value(),
        named && named->endpoints.value_or(false));
    // Identity recursion already filtered child work by its effective needs;
    // retain a selector's hidden query prerequisite at the wrapper as well.
    const bool queryDependency = math && (identityNeedsQueries || (needs.selectors &&
        (math->boundedDemands || math->arithmeticRegionalDemands || math->backend == "expanded-finite-guarded")) ||
        ((needs.selectors || needs.synchronization) && math->varyingBoundaryDemands));
    const std::array<bool, 4> required{true, needs.queries || queryDependency, needs.selectors, needs.synchronization};
    result.required = required;
    const bool arithmeticMethod =
        backend == AnalysisBackend::Arithmetic || backend == AnalysisBackend::ArithmeticPeriodic;
    const bool certified = arithmetic && arithmetic->extraction.state == RecognitionState::Applicable &&
        arithmetic->recognition.state == RecognitionState::Applicable;
    if (arithmeticMethod && certified) {
        const auto& shape = primitiveShape(session, *arithmetic);
        const bool sharedGenerators = region == 0;
        const auto cold = estimates(*arithmetic, shape, {}, !sharedGenerators);
        const auto& estimate = backend == AnalysisBackend::Arithmetic ? cold.second : cold.first;
        if (result.stages[0] == RetainedStage::Missing) { result.work[0] = estimate.work; }
        if (sharedGenerators) {
            result.generators = RetainedStage::Missing;
            auto retained = [&](const auto& stage) {
                if (!stage || !stage->belongsTo(*arithmetic)) { return; }
                result.evidence.push_back(&*stage);
                result.generators = stage->analysis().error.empty() ? RetainedStage::Ready : RetainedStage::Refused;
            };
            retained(session.differenceGenerators);
            retained(session.generalGenerators);
            if (result.stages[0] == RetainedStage::Missing) {
                result.generatorWork = *result.generators == RetainedStage::Missing ?
                    generatorConstructionWork(shape) : EstimatedCount{0};
                // A failed owned generator stage cannot enter either reducer.
                // The ordinary request still reports its precise refusal.
                if (*result.generators == RetainedStage::Refused) { result.work[0] = 0; }
            }
        }
        auto exportWork = estimatedMultiply(estimatedAdd(shape.pieces, shape.native), estimate.ports);
        if (backend == AnalysisBackend::Arithmetic) {
            exportWork = estimatedMultiply(exportWork, estimatedAdd(shape.dimensions, 1));
        }
        for (unsigned i = 1; i < 4; ++i) {
            if (result.stages[i] == RetainedStage::Missing) { result.work[i] = exportWork; }
        }
    }
    result.total = result.generatorWork;
    for (unsigned i = 0; i < required.size(); ++i) {
        if (!required[i] || result.stages[i] != RetainedStage::Missing) { result.work[i] = 0; }
        result.total = estimatedAdd(result.total, result.work[i]);
    }
    return result;
}
} // namespace
RetainedBackendWork previewRetainedBackendWork(const AnalysisSessionState& session,
    const ProgramRecognition& program, std::size_t region, std::optional<std::size_t> canonical,
    AnalysisBackend backend, const AnalysisNeeds& needs, const ArithmeticProgram* arithmetic)
{
    return retainedBackendWork(session, program, region, canonical, backend, needs, arithmetic);
}
std::vector<AnalysisForm> FrontierAnalysis::exactForms(const AnalysisRequest& request)
{
    const bool valid = storage && succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) { return {}; }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    sessionState->requestsStarted = true;
    AnalysisProfileScope profile(sessionState->ledger, "selection", request.region, "exactForms");
    const auto identity = std::make_pair(request.region, analysisRequestKey(request));
    const StructureNode* loop = request.region ? &program->nodes[request.region] :
        wholeInvocationLoop(function, *storage, *program, *structuralIndex);
    const auto canonical = loop ? static_cast<std::size_t>(loop - program->nodes.data()) : request.region;
    const auto retained = sessionState->normalizedInputs.find(canonical);
    const auto* normalized = retained == sessionState->normalizedInputs.end() ? nullptr : retained->second.get();
    const auto preflight = sessionState->numericTemplatePlans.find({canonical, true});
    const auto* numerical = preflight == sessionState->numericTemplatePlans.end() ? nullptr : preflight->second.get();
    const RecognitionResult* rotating = loop && loop->rotatingResult ? &*loop->rotatingResult : nullptr;
    const std::vector<const void*> evidence{normalized, numerical, rotating};
    auto cached = sessionState->formOrders.find(identity);
    if (cached != sessionState->formOrders.end() && cached->second.evidence == evidence) {
        profile.cacheHit(); return cached->second.order;
    }
    std::vector<AnalysisForm> order{AnalysisForm::Original};
    if (normalized && !normalized->original) {
        sessionState->expandedFormEstimates[identity] = {};
        sessionState->formOrders[identity] = {order, evidence}; return order;
    }
    AnalysisCostEstimate original, expanded;
    // Absence of retained qualification leaves both candidates eligible. In
    // particular, the expanded bucket also owns the numerical-sequence adapter.
    if (normalized) {
        const auto& source = *normalized->original;
        const uint64_t interfaces = uint64_t(request.needs.queries) + uint64_t(request.needs.selectors) +
                                    uint64_t(request.needs.synchronization);
        // Count backend occurrences, not just normalized syntax: retained dependent
        // finite domains still enumerate visits during numerical materialization.
        const bool numericalEligible = numerical && numerical->form.result.state == RecognitionState::Applicable;
        EstimatedCount payloads, effects, visits;
        if (numericalEligible) {
            payloads = numerical->form.countedPayloads;
            visits = numerical->form.countedVisits;
            effects = 0;
            for (const auto& occurrence : numerical->payloads) {
                effects = estimatedAdd(effects, storage->accesses().effectsFor(occurrence.phase).size());
            }
        }
        const auto joins = estimatedMultiply(effects, effects);
        expanded.generatorPieces = joins;
        expanded.ports = estimatedMultiply(payloads, 2);
        expanded.numericalWindow = estimatedMultiply(payloads, estimatedAdd(effects, 1));
        expanded.circuitNodes = expanded.relationConversion = 0;
        // Keep the existing conservative per-form estimate for stable selection.
        // Actual shared-tree source visits are recorded separately by the session.
        const auto normalization = estimatedAdd(estimatedMultiply(source.representedNodes, 2),
            estimatedAdd(normalized->representedNodes, source.representedNodes));
        expanded.work = estimatedAdd(normalization, estimatedAdd(visits,
            estimatedAdd(joins, estimatedMultiply(expanded.numericalWindow, estimatedAdd(interfaces, 1)))));
        original.representation = estimatedAdd(source.representedNodes, source.effects);
        expanded.representation = numericalEligible ? estimatedAdd(normalized->representedNodes,
            estimatedAdd(effects, estimatedMultiply(interfaces, expanded.ports))) :
            estimatedAdd(normalized->representedNodes, normalized->effects);
        // Only a complete known original form receives a known estimate. An
        // unavailable signature leaves work unknown, never removes that attempt.
        if (rotating && rotating->state == RecognitionState::Applicable) {
            const EstimatedCount fragments = rotating->accesses.size();
            const EstimatedCount sites = loop->payloadCount;
            original.generatorPieces = estimatedMultiply(fragments, fragments);
            original.ports = estimatedMultiply(sites, 2);
            original.numericalWindow = original.circuitNodes = original.relationConversion = 0;
            original.work = estimatedAdd(source.representedNodes, estimatedMultiply(original.generatorPieces,
                estimatedAdd(sites, estimatedAdd(interfaces, 1))));
            original.representation = estimatedAdd(original.generatorPieces,
                estimatedMultiply(original.ports, estimatedAdd(interfaces, 1)));
        }
    }
    order.push_back(AnalysisForm::SmallCountExpanded);
    if (estimatedSelectionWorkLess(expanded, original)) { std::reverse(order.begin(), order.end()); }
    sessionState->ledger->recordEstimate({request.region, identity.second, "form-original", original});
    sessionState->ledger->recordEstimate({request.region, identity.second, "form-expanded", expanded});
    // Records are append-only evidence snapshots; refreshed qualifications get
    // a new record, while identical evidence produces only a selection cache hit.
    sessionState->expandedFormEstimates[identity] = expanded;
    sessionState->formOrders[identity] = {order, evidence};
    return order;
}
std::vector<AnalysisBackend> FrontierAnalysis::arithmeticMethods(const AnalysisRequest& request)
{
    const bool valid = storage && succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) { return {}; }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    sessionState->requestsStarted = true;
    AnalysisProfileScope profile(sessionState->ledger, "selection", request.region, "arithmeticMethods");
    const auto key = analysisRequestKey(request);
    const auto identity = std::make_pair(request.region, key);
    // The regional periodic adapter is not implemented. Do not advertise it as
    // a reducer choice; the available regional arithmetic path stays eligible.
    std::vector<AnalysisBackend> order = request.region ? std::vector<AnalysisBackend>{AnalysisBackend::Arithmetic} :
        std::vector<AnalysisBackend>{AnalysisBackend::ArithmeticPeriodic, AnalysisBackend::Arithmetic};
    const ArithmeticProgram* form = nullptr;
    if (request.region) {
        auto found = arithmeticForms.find(request.region);
        if (found != arithmeticForms.end()) { form = found->second.get(); }
    } else if (program->arithmetic) { form = &*program->arithmetic; }
    const std::vector<const void*> evidence{form};
    if (auto found = sessionState->arithmeticOrders.find(identity);
        found != sessionState->arithmeticOrders.end() && found->second.evidence == evidence) {
        profile.cacheHit(); return found->second.order;
    }
    AnalysisCostEstimate periodic, arithmetic;
    const bool certified = form && form->extraction.state == RecognitionState::Applicable &&
        form->recognition.state == RecognitionState::Applicable;
    if (certified) {
        std::tie(periodic, arithmetic) = estimates(*form, primitiveShape(*sessionState, *form), request.needs);
    }
    if (!request.region && estimatedSelectionWorkLess(arithmetic, periodic)) {
        std::reverse(order.begin(), order.end());
    }
    for (auto method : order) {
        const bool periodicMethod = method == AnalysisBackend::ArithmeticPeriodic;
        sessionState->ledger->recordEstimate({request.region, key,
                                      periodicMethod ? "arithmetic-periodic" : "arithmetic",
                                      periodicMethod ? periodic : arithmetic});
    }
    sessionState->arithmeticOrders[identity] = {order, evidence};
    return order;
}
std::vector<AnalysisBackend> FrontierAnalysis::rankExactCandidates(const AnalysisRequest& request,
    const std::vector<AnalysisBackend>& candidates)
{
    AnalysisProfileScope profile(sessionState->ledger, "selection", request.region, "retained-work");
    uint64_t candidateMask = 0;
    for (const auto backend : candidates) { candidateMask |= uint64_t{1} << unsigned(backend); }
    const auto identity = std::make_tuple(request.region, analysisRequestKey(request), sessionState->certifying,
        candidateMask);
    const StructureNode* loop = request.region ? &program->nodes[request.region] :
        wholeInvocationLoop(function, *storage, *program, *structuralIndex);
    const auto canonical = loop ? std::optional<std::size_t>(loop - program->nodes.data()) : std::nullopt;
    const ArithmeticProgram* arithmetic = nullptr;
    if (!request.region && program->arithmetic) { arithmetic = &*program->arithmetic; }
    if (request.region) {
        const auto found = arithmeticForms.find(request.region);
        if (found != arithmeticForms.end()) { arithmetic = found->second.get(); }
    }
    std::vector<RetainedBackendWork> previews;
    for (const auto backend : candidates) {
        previews.push_back(previewRetainedBackendWork(*sessionState, *program, request.region, canonical,
            backend, request.needs, arithmetic));
    }
    auto found = sessionState->backendOrders.find(identity);
    const bool retained = found != sessionState->backendOrders.end() &&
        found->second.candidates == candidates && found->second.evidence == previews;
    std::vector<AnalysisBackend> order;
    if (retained) { profile.cacheHit(); order = found->second.order; }
    else {
        std::vector<std::size_t> positions(candidates.size());
        std::iota(positions.begin(), positions.end(), 0);
        std::stable_sort(positions.begin(), positions.end(), [&](auto a, auto b) {
            AnalysisCostEstimate left, right;
            left.work = previews[a].total; right.work = previews[b].total;
            return estimatedSelectionWorkLess(left, right);
        });
        // Named adapters must follow their dependency even when its remaining
        // work is unknown. They cannot memoize a premature prerequisite failure.
        while (!positions.empty()) {
            auto eligible = llvm::find_if(positions, [&](auto position) {
                const auto backend = candidates[position];
                std::optional<AnalysisBackend> dependency;
                if (backend == AnalysisBackend::NumericalSequence) { dependency = AnalysisBackend::NumericalPeriodic; }
                if (backend == AnalysisBackend::SequenceEndpoints) { dependency = AnalysisBackend::Sequence; }
                return !dependency || !llvm::any_of(positions, [&](auto other) {
                    return candidates[other] == *dependency;
                });
            });
            assert(eligible != positions.end() && "exact candidate dependencies must be acyclic");
            order.push_back(candidates[*eligible]);
            positions.erase(eligible);
        }
        sessionState->backendOrders[identity] = {candidates, order, previews};
    }
    if (sessionState->ledger->profiling) {
        constexpr const char* stages[] = {"demands", "queries", "selectors", "synchronization"};
        for (const auto backend : order) {
            const auto position = std::distance(candidates.begin(), llvm::find(candidates, backend));
            const auto& work = previews[position];
            AnalysisLedger::Selection record;
            record.request = profile.requestId(); record.region = request.region;
            record.method = analysisBackendName(backend).str(); record.unbuiltWork = work.total;
            record.reason = retained ? "retained-selection" :
                (!work.total ? "unknown-unbuilt-work" : (*work.total ? "estimated-unbuilt-work" : "no-unbuilt-work"));
            if (work.generators) {
                auto& list = *work.generators == RetainedStage::Ready ? record.reused :
                    (*work.generators == RetainedStage::Refused ? record.refused : record.missing);
                list.push_back("arithmetic-generators");
            }
            for (unsigned stage = 0; stage < work.stages.size(); ++stage) {
                if (!work.required[stage]) {
                    if (work.stages[stage] == RetainedStage::Ready) {
                        record.retainedUnrequested.push_back(stages[stage]);
                    }
                    continue;
                }
                auto& list = work.stages[stage] == RetainedStage::Ready ? record.reused :
                    (work.stages[stage] == RetainedStage::Refused ? record.refused : record.missing);
                list.push_back(stages[stage]);
            }
            sessionState->ledger->selections.push_back(std::move(record));
        }
    }
    return order;
}
std::vector<AnalysisCostRecord> FrontierAnalysis::costRecords() const
{
    if (!sessionState) { return {}; }
    auto records = sessionState->ledger->estimates;
    accumulateCost(sessionState->ledger->snapshotRecordsCopied, records.size());
    for (auto& record : records) {
        if (StringRef(record.method).starts_with("sequence-crossings-")) { continue; }
        if (record.method == "form-original" || record.method == "form-expanded") {
            const auto form = record.method == "form-original" ?
                AnalysisForm::Original : AnalysisForm::SmallCountExpanded;
            auto attempt = sessionState->formAttempts.find({record.region, form});
            record.attemptConstructions = attempt == sessionState->formAttempts.end() ? 0 : attempt->second;
            continue;
        }
        const auto backend = record.method == "numerical-sequence" ? AnalysisBackend::NumericalSequence :
            (record.method == "arithmetic-periodic" ?
                AnalysisBackend::ArithmeticPeriodic : AnalysisBackend::Arithmetic);
        if (record.region && backend == AnalysisBackend::Arithmetic) {
            const auto attempt = sessionState->arithmeticRegionAttempts.find(record.region);
            record.attemptConstructions = attempt != sessionState->arithmeticRegionAttempts.end() &&
                attempt->second.produced;
            continue;
        }
        const auto region = sessionState->attempts.find(record.region);
        if (region == sessionState->attempts.end()) { continue; }
        const auto attempt = region->second.find(backend);
        record.attemptConstructions = attempt != region->second.end() && attempt->second.produced;
    }
    return records;
}
} // namespace mlir::pto::frontiersynch
