// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_ARITHMETIC_REGIONAL_CONSTRUCTION_H
#define PTO_FRONTIER_ARITHMETIC_REGIONAL_CONSTRUCTION_H
#include "DifferenceIntegerConversion.h"
#include "PTO/Transforms/FrontierSynch/AnalysisProfile.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
namespace mlir::pto::frontiersynch::detail {
inline bool supportsDifferenceConstruction(const ArithmeticProgram& program)
{
    return program.recognition.state == RecognitionState::Applicable &&
        program.recognition.arithmeticClass == ArithmeticClass::Differences &&
        llvm::none_of(program.recognition.pieces, [](const auto& piece) {
            return !piece.empty && bool(piece.machineSystem);
        });
}
// Retain the qualified mathematical representation. Regional consumers currently
// require integer relations; convert once, and share both views on export retries.
inline void constructRegionalArithmetic(ArithmeticRegionalRelations& result,
    const StructuredProtection& protection, ArithmeticDemandProfile profile = {})
{
    if (!supportsDifferenceConstruction(result.program)) {
        result.analysis = analyzeGeneralArithmeticDemandsWithProtection(result.program, protection, std::move(profile));
        return;
    }
    result.differenceDemands = std::make_shared<const ArithmeticDemandAnalysis>(
        analyzeArithmeticDemandsWithProtection(result.program, protection, profile));
    const auto& source = *result.differenceDemands;
    auto& target = result.analysis;
    target.error = source.error; target.period = source.period;
    target.parameterCount = source.parameterCount; target.pipeCount = source.pipeCount;
    target.exactMinimum = false; target.adjacentLocalDemands = source.adjacentLocalDemands;
    target.cost = source.cost;
    if (!source.error.empty() || !source.exactMinimum) { return; }
    AnalysisProfileScope adapter(profile.ledger, "arithmetic-difference-export", profile.region);
    adapter.constructed();
    std::vector<std::pair<GeneralArithmeticRelation*, GeneralArithmeticRelation>> staged;
    for (auto pair : {std::make_pair(&source.generators, &target.generators),
                     std::make_pair(&source.nativeOrder, &target.nativeOrder),
                     std::make_pair(&source.requiredOrder, &target.requiredOrder),
                     std::make_pair(&source.minimumDemands, &target.minimumDemands)}) {
        auto converted = exportDifferenceRelation(*pair.first, &target.cost);
        if (failed(converted)) {
            target.error = "retained difference mathematics requires an unavailable integer export";
            return;
        }
        staged.emplace_back(pair.second, std::move(*converted));
    }
    for (auto& entry : staged) { *entry.first = std::move(entry.second); }
    target.exactMinimum = true;
    adapter.recordWork({{"systems", target.cost.differenceExportSystems},
        {"atoms", target.cost.differenceExportAtoms}, {"coefficients", target.cost.differenceExportCoefficients}});
}
} // namespace mlir::pto::frontiersynch::detail
#endif
