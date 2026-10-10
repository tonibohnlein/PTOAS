// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "AnalysisSessionInternal.h"
#include "NormalizedControl.h"
#include "NumericTemplatePlan.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include <algorithm>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
uint8_t requestKey(const AnalysisRequest& request)
{
    return unsigned(request.needs.queries) | (unsigned(request.needs.selectors) << 1) |
        (unsigned(request.needs.synchronization) << 2) | (unsigned(request.mode == AnalysisMode::Fallback) << 3);
}
struct GeneratorShape {
    EstimatedCount pieces = 0, native = 0, rows = 0, domains = 0;
    uint64_t dimensions = 0;
};
// Estimate joins from certified primitive descriptions before projection or
// generator construction. These are selection estimates, not output bounds.
GeneratorShape primitiveShape(const ArithmeticProgram& program)
{
    GeneratorShape shape;
    shape.domains = program.sites.size();
    shape.dimensions = program.recognition.observedDimensions;
    EstimatedCount reads = 0, writes = 0, prerequisite = 0, rows = 0, pieces = 0;
    for (const auto& relation : program.primitives.relations) {
        const auto count = EstimatedCount(relation.pieces.size());
        if (relation.kind == PrimitiveKind::Reads) { reads = estimatedAdd(reads, count); }
        if (relation.kind == PrimitiveKind::Writes) { writes = estimatedAdd(writes, count); }
        if (relation.kind == PrimitiveKind::Prerequisites) { prerequisite = estimatedAdd(prerequisite, count); }
        if (relation.kind == PrimitiveKind::Native) { shape.native = estimatedAdd(shape.native, count); }
        pieces = estimatedAdd(pieces, count);
        for (const auto& piece : relation.pieces) {
            rows = estimatedAdd(rows, piece.system.getNumConstraints());
        }
    }
    shape.pieces = estimatedAdd(prerequisite, estimatedMultiply(writes, estimatedAdd(reads, writes)));
    // Joining accesses includes their domain/order descriptions. Projection can
    // increase the normalized description; use its class dimensions in costs.
    const auto averageRows = rows && pieces && *pieces ?
        EstimatedCount(*rows / *pieces + uint64_t(*rows % *pieces != 0)) : EstimatedCount{};
    shape.rows = estimatedMultiply(estimatedAdd(shape.pieces, shape.native),
        estimatedMultiply(averageRows, 3));
    return shape;
}
std::pair<AnalysisCostEstimate, AnalysisCostEstimate> estimates(
    const ArithmeticProgram& program, const GeneratorShape& shape, const AnalysisNeeds& needs)
{
    AnalysisCostEstimate periodic, arithmetic;
    const auto types = estimatedMultiply(program.sites.size(), program.primitives.period);
    const auto vertices = estimatedMultiply(2, types);
    const auto pipes = EstimatedCount(program.primitives.pipeCount);
    const auto generators = estimatedAdd(shape.pieces, shape.native);
    const auto phaseCopies = estimatedMultiply(program.primitives.period, program.primitives.period);
    const auto liftedGenerators = estimatedMultiply(generators, phaseCopies);
    const auto dimension = estimatedAdd(shape.dimensions, 1);
    // Include every potential guarded native skip edge, not just active edges.
    const auto edges = estimatedAdd(liftedGenerators, estimatedAdd(vertices, estimatedMultiply(types, types)));
    periodic.generatorPieces = arithmetic.generatorPieces = shape.pieces;
    periodic.ports = arithmetic.ports = vertices;
    periodic.numericalWindow = arithmetic.numericalWindow = 0;
    periodic.circuitNodes = estimatedAdd(estimatedMultiply(shape.pieces, shape.pieces),
        estimatedMultiply(estimatedMultiply(pipes, vertices), edges));
    periodic.relationConversion = estimatedAdd(estimatedMultiply(shape.rows, dimension),
        estimatedMultiply(estimatedMultiply(generators, shape.domains), estimatedMultiply(dimension, dimension)));
    // DBM adaptation validates the sparse row description with one closure.
    if (program.recognition.arithmeticClass == ArithmeticClass::Differences) {
        periodic.relationConversion = estimatedAdd(periodic.relationConversion,
            estimatedMultiply(generators, estimatedPower(dimension, 3)));
    }
    periodic.relationConversion = estimatedMultiply(periodic.relationConversion, phaseCopies);
    periodic.work = estimatedAdd(periodic.relationConversion, estimatedMultiply(periodic.circuitNodes, 2));
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
    periodic.representation = estimatedAdd(periodic.representation, exports);
    arithmetic.representation = estimatedAdd(arithmetic.representation, exports);
    return {periodic, arithmetic};
}
} // namespace
std::vector<AnalysisForm> FrontierAnalysis::exactForms(const AnalysisRequest& request)
{
    const bool valid = storage && succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) { return {}; }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    const auto identity = std::make_pair(request.region, requestKey(request));
    auto cached = sessionState->formOrders.find(identity);
    if (cached != sessionState->formOrders.end()) { return cached->second; }
    auto normalized = normalizedRegion(request.region);
    std::vector<AnalysisForm> order{AnalysisForm::Original};
    if (!normalized->original) {
        sessionState->formOrders.emplace(identity, order); return order;
    }
    AnalysisCostEstimate original, expanded;
    const auto* numerical = numericalPreflight(request.region);
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
    const auto normalization = estimatedAdd(estimatedMultiply(source.nodes.size(), 2),
        estimatedAdd(normalized->nodes.size(), normalized->planningVisits));
    expanded.work = estimatedAdd(normalization, estimatedAdd(visits,
        estimatedAdd(joins, estimatedMultiply(expanded.numericalWindow, estimatedAdd(interfaces, 1)))));
    original.representation = estimatedAdd(source.nodes.size(), source.effects);
    expanded.representation = numericalEligible ? estimatedAdd(normalized->nodes.size(),
        estimatedAdd(effects, estimatedMultiply(interfaces, expanded.ports))) :
        estimatedAdd(normalized->nodes.size(), normalized->effects);
    // Only a complete known original form receives a known estimate. An
    // unavailable signature leaves work unknown, never removes that attempt.
    const StructureNode* loop = request.region ? &program->nodes[request.region] : nullptr;
    if (!request.region) {
        for (const auto& node : program->nodes) {
            const bool topLevelLoop = node.kind == StructureKind::Loop && node.anchor->getParentOp() == function;
            if (!topLevelLoop) { continue; }
            const bool complete = llvm::all_of(storage->instructions(), [&](auto* phase) {
                return node.anchor->isProperAncestor(phase->elementOp);
            });
            if (complete) { loop = &node; break; }
        }
    }
    const RecognitionResult* rotating = loop && loop->rotatingResult ? &*loop->rotatingResult : nullptr;
    if (rotating && rotating->state == RecognitionState::Applicable) {
        const EstimatedCount fragments = rotating->accesses.size();
        const EstimatedCount sites = loop->payloadCount;
        original.generatorPieces = estimatedMultiply(fragments, fragments);
        original.ports = estimatedMultiply(sites, 2);
        original.numericalWindow = original.circuitNodes = original.relationConversion = 0;
        original.work = estimatedAdd(source.nodes.size(), estimatedMultiply(original.generatorPieces,
            estimatedAdd(sites, estimatedAdd(interfaces, 1))));
        original.representation = estimatedAdd(original.generatorPieces,
            estimatedMultiply(original.ports, estimatedAdd(interfaces, 1)));
    }
    order.push_back(AnalysisForm::SmallCountExpanded);
    if (estimatedCostLess(expanded, original)) { std::reverse(order.begin(), order.end()); }
    sessionState->costs.push_back({request.region, identity.second, "form-original", original});
    sessionState->costs.push_back({request.region, identity.second, "form-expanded", expanded});
    sessionState->formOrders.emplace(identity, order);
    return order;
}
std::vector<AnalysisBackend> FrontierAnalysis::arithmeticMethods(const AnalysisRequest& request)
{
    const bool valid = storage && succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) { return {}; }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    const auto key = requestKey(request);
    const auto identity = std::make_pair(request.region, key);
    if (auto found = sessionState->arithmeticOrders.find(identity); found != sessionState->arithmeticOrders.end()) {
        return found->second;
    }
    // The regional periodic adapter is not implemented. Do not advertise it as
    // a reducer choice; the available regional arithmetic path stays eligible.
    std::vector<AnalysisBackend> order = request.region ? std::vector<AnalysisBackend>{AnalysisBackend::Arithmetic} :
        std::vector<AnalysisBackend>{AnalysisBackend::ArithmeticPeriodic, AnalysisBackend::Arithmetic};
    const ArithmeticProgram* form = nullptr;
    if (request.region) { form = recognizeArithmeticRegion(request.region); }
    else if (succeeded(recognizeArithmetic()) && program->arithmetic) { form = &*program->arithmetic; }
    AnalysisCostEstimate periodic, arithmetic;
    const bool certified = form && form->extraction.state == RecognitionState::Applicable &&
        form->recognition.state == RecognitionState::Applicable;
    if (certified) {
        std::tie(periodic, arithmetic) = estimates(*form, primitiveShape(*form), request.needs);
        std::string diagnostic;
        if (!request.region && !checkArithmeticPeriodicSkeleton(*form, diagnostic)) {
            periodic.work.reset(); periodic.representation.reset();
        }
    }
    if (!request.region && estimatedCostLess(arithmetic, periodic)) { std::reverse(order.begin(), order.end()); }
    for (auto method : order) {
        const bool periodicMethod = method == AnalysisBackend::ArithmeticPeriodic;
        sessionState->costs.push_back({request.region, key, periodicMethod ? "arithmetic-periodic" : "arithmetic",
                                      periodicMethod ? periodic : arithmetic});
    }
    sessionState->arithmeticOrders.emplace(identity, order);
    return order;
}
std::vector<AnalysisCostRecord> FrontierAnalysis::costRecords() const
{
    if (!sessionState) { return {}; }
    auto records = sessionState->costs;
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
