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
// Mathematical construction owns its specialized result; no integer adapter or
// last reachability round is requested until an actual consumer needs that view.
void initializeArithmeticExportCache(ArithmeticRegionalRelations& result, ArithmeticDemandProfile profile);
inline void constructRegionalArithmetic(ArithmeticRegionalRelations& result,
    const StructuredProtection& protection, ArithmeticDemandProfile profile = {})
{
    if (!supportsDifferenceConstruction(result.program)) {
        result.analysis = analyzeGeneralArithmeticDemandsWithProtection(result.program, protection, profile,
            ArithmeticDemandOutputs::Demands);
    } else {
        result.differenceDemands = std::make_shared<const ArithmeticDemandAnalysis>(
            analyzeArithmeticDemandsWithProtection(result.program, protection, profile,
                ArithmeticDemandOutputs::Demands));
        const auto& source = *result.differenceDemands;
        auto& target = result.analysis;
        target.error = source.error; target.period = source.period;
        target.parameterCount = source.parameterCount; target.pipeCount = source.pipeCount;
        target.exactMinimum = source.exactMinimum; target.adjacentLocalDemands = source.adjacentLocalDemands;
        target.requiredOrderAvailable = false; target.availableRelations = 0;
        target.cost = source.cost;
    }
    initializeArithmeticExportCache(result, std::move(profile));
}
} // namespace mlir::pto::frontiersynch::detail
#endif
