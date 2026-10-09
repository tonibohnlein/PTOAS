// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "AnalysisSessionInternal.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include <algorithm>
#include <type_traits>
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
template<class Stage>
GeneratorShape generatorShape(const Stage& stage)
{
    GeneratorShape shape;
    if (!stage.analysis().error.empty()) { shape.pieces.reset(); return shape; }
    shape.domains = stage.occurrences().size();
    for (const auto* relation : {&stage.analysis().generators, &stage.analysis().nativeOrder}) {
        for (const auto& [key, pieces] : *relation) {
            (void)key;
            auto& count = relation == &stage.analysis().generators ? shape.pieces : shape.native;
            count = estimatedAdd(count, pieces.size());
            for (const auto& piece : pieces) {
                shape.dimensions = std::max(shape.dimensions, uint64_t(piece.dimensions()));
                EstimatedCount rows;
                if constexpr (std::is_same_v<typename Stage::System, DifferenceBoundSystem>) {
                    auto dimensions = estimatedAdd(piece.dimensions(), 1);
                    rows = estimatedMultiply(dimensions, dimensions);
                } else {
                    rows = estimatedAdd(piece.constraints().size(), piece.congruences().size());
                }
                shape.rows = estimatedAdd(shape.rows, rows);
            }
        }
    }
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
std::vector<AnalysisBackend> FrontierAnalysis::arithmeticMethods(const AnalysisRequest& request)
{
    std::vector<AnalysisBackend> order{AnalysisBackend::ArithmeticPeriodic, AnalysisBackend::Arithmetic};
    if (request.region || failed(ensureArithmeticGenerators())) { return order; }
    const auto key = requestKey(request);
    if (auto found = sessionState->arithmeticOrders.find(key); found != sessionState->arithmeticOrders.end()) {
        return found->second;
    }
    const auto shape = sessionState->differenceGenerators ? generatorShape(*sessionState->differenceGenerators) :
                                                          generatorShape(*arithmeticGeneratorStage);
    auto [periodic, arithmetic] = estimates(*program->arithmetic, shape, request.needs);
    std::string diagnostic;
    if (!checkArithmeticPeriodicSkeleton(*program->arithmetic, diagnostic)) {
        periodic.work.reset(); periodic.representation.reset();
    }
    if (estimatedCostLess(arithmetic, periodic)) { std::reverse(order.begin(), order.end()); }
    for (auto method : order) {
        const bool periodicMethod = method == AnalysisBackend::ArithmeticPeriodic;
        sessionState->costs.push_back({0, key, periodicMethod ? "arithmetic-periodic" : "arithmetic",
                                      periodicMethod ? periodic : arithmetic});
    }
    sessionState->arithmeticOrders.emplace(key, order);
    return order;
}
std::vector<AnalysisCostRecord> FrontierAnalysis::costRecords() const
{
    if (!sessionState) { return {}; }
    auto records = sessionState->costs;
    for (auto& record : records) {
        const auto backend = record.method == "arithmetic-periodic" ? AnalysisBackend::ArithmeticPeriodic :
                                                                     AnalysisBackend::Arithmetic;
        const auto region = sessionState->attempts.find(record.region);
        if (region == sessionState->attempts.end()) { continue; }
        const auto attempt = region->second.find(backend);
        record.attemptConstructions = attempt != region->second.end() && attempt->second.produced;
    }
    return records;
}
} // namespace mlir::pto::frontiersynch
