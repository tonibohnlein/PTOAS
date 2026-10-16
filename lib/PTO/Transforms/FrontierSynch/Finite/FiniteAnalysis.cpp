// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
namespace mlir::pto::frontiersynch {
const std::string& FiniteAnalysis::error() const
{
    if (const auto* value = concrete()) { return value->error; }
    return guarded()->error;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FiniteAnalysis::prepare()
{
    preparationError.clear();
    if (!input || !error().empty()) { return failure(); }
    if (const auto* value = concrete()) {
        return prepareExplicitInsertion(input->function(), *value, false);
    }
    auto& value = std::get<FiniteGuardedAnalysis>(representation);
    auto plan = prepareFiniteGuardedLogicalInsertion(value);
    preparationError = value.insertionError;
    return plan;
}
std::shared_ptr<const FiniteOccurrenceInput> finiteFunctionInput(func::FuncOp function,
    std::shared_ptr<const SyncInput> storage, std::shared_ptr<const PhaseIndex> index)
{
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !storage || !index) {
        return {};
    }
    if (!index->sourceFactsFor(function, storage->instructions())) { return {}; }
    for (const auto* phase : storage->instructions()) {
        if (!phase || !phase->elementOp || !function->isProperAncestor(phase->elementOp) ||
            !llvm::is_contained(index->phasesFor(phase->elementOp), phase)) { return {}; }
    }
    // Private construction prevents an arbitrary subset from masquerading as a
    // complete function. Regional input factories are separate qualifications.
    auto input = std::shared_ptr<FiniteOccurrenceInput>(
        new FiniteOccurrenceInput(function, std::move(storage), std::move(index)));
    for (auto& operation : function.front()) {
        if (!operation.hasTrait<OpTrait::IsTerminator>()) { input->originalRoots.push_back(&operation); }
    }
    return input;
}
std::shared_ptr<FiniteAnalysis> analyzeFinite(std::shared_ptr<const FiniteOccurrenceInput> input,
    FiniteStrategy strategy, std::shared_ptr<RegionExpressions> expressions)
{
    if (!input) { return {}; }
    if (strategy == FiniteStrategy::Concrete) {
        auto result = analyzeExplicit(input->function().front(), input->index(), input->storage());
        return std::make_shared<FiniteAnalysis>(std::move(input), std::move(result));
    }
    auto result = analyzeFiniteGuarded(input->function(), input->roots(), input->index(), input->storage(),
                                      std::move(expressions));
    return std::make_shared<FiniteAnalysis>(std::move(input), std::move(result));
}
} // namespace mlir::pto::frontiersynch
