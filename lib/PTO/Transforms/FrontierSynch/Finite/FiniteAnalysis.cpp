// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/Recognition.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalFamilyFragment.h"
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
        if (recipe) { accumulateCost(work.recipeHits, 1); return recipe->instantiate(); }
        accumulateCost(work.recipeBuilds, 1);
        auto prepared = prepareExplicitInsertion(input->function(), *value, false);
        if (failed(prepared)) { return failure(); }
        if (!input->wholeFunction()) { (*prepared)->completeInvocation = false; }
        recipe = LogicalFamilyFragment::capture(**prepared);
        return prepared;
    }
    auto& value = std::get<FiniteGuardedAnalysis>(representation);
    auto plan = prepareFiniteGuardedLogicalInsertion(value);
    preparationError = value.insertionError;
    return plan;
}
std::shared_ptr<const FiniteOccurrenceInput> finiteBorrowedFunctionInput(func::FuncOp function,
    const SyncInput& storage, std::shared_ptr<const PhaseIndex> index)
{
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !index ||
        !index->sourceFactsFor(function, storage.instructions())) { return {}; }
    for (const auto* phase : storage.instructions()) {
        if (!phase || !phase->elementOp || !function->isProperAncestor(phase->elementOp) ||
            !llvm::is_contained(index->phasesFor(phase->elementOp), phase)) { return {}; }
    }
    auto input = std::shared_ptr<FiniteOccurrenceInput>(new FiniteOccurrenceInput(function, storage, std::move(index)));
    for (auto& operation : function.front()) {
        if (!operation.hasTrait<OpTrait::IsTerminator>()) { input->originalRoots.push_back(&operation); }
    }
    accumulateCost(*input->rootCopyWork, input->originalRoots.size());
    return input;
}
std::shared_ptr<const FiniteOccurrenceInput> finiteFunctionInput(func::FuncOp function,
    std::shared_ptr<const SyncInput> storage, std::shared_ptr<const PhaseIndex> index)
{
    if (!storage) { return {}; }
    auto borrowed = finiteBorrowedFunctionInput(function, *storage, std::move(index));
    if (!borrowed) { return {}; }
    auto owned = std::make_shared<FiniteOccurrenceInput>(*borrowed);
    accumulateCost(*owned->rootCopyWork, owned->originalRoots.size());
    owned->inputOwner = std::move(storage);
    return owned;
}
std::shared_ptr<const FiniteOccurrenceInput> finiteRegionInput(
    std::shared_ptr<const FiniteOccurrenceInput> source, ArrayRef<Operation*> roots)
{
    if (!source) { return {}; }
    Operation* previous = nullptr;
    for (auto* root : roots) {
        if (!root || !source->function()->isProperAncestor(root) || root->hasTrait<OpTrait::IsTerminator>() ||
            (previous && previous->getNextNode() != root)) { return {}; }
        previous = root;
    }
    auto input = std::shared_ptr<FiniteOccurrenceInput>(
        new FiniteOccurrenceInput(source->function(), source->storage(), source->indexOwner, source->rootCopyWork));
    input->inputOwner = source->inputOwner;
    input->originalRoots.assign(roots.begin(), roots.end());
    accumulateCost(*input->rootCopyWork, roots.size());
    input->whole = false;
    return input;
}
std::shared_ptr<FiniteAnalysis> analyzeFinite(std::shared_ptr<const FiniteOccurrenceInput> input,
    FiniteStrategy strategy, std::shared_ptr<RegionExpressions> expressions)
{
    if (!input) { return {}; }
    if (strategy == FiniteStrategy::Concrete) {
        ExplicitAnalysis result;
        if (input->wholeFunction()) {
            result = analyzeExplicit(input->function().front(), input->index(), input->storage());
        } else if (recognizeExplicitRun(input->roots(), input->index(), input->storage().accesses()).state !=
                   RecognitionState::Applicable) {
            result.error = "finite concrete region requires resolved control, accesses and prerequisites";
        } else {
            SmallVector<const CompoundInstanceElement*> phases;
            for (auto* root : input->roots()) { llvm::append_range(phases, input->index().phasesFor(root)); }
            result = analyzeExplicit(phases, input->index(), input->storage(), true);
        }
        return std::make_shared<FiniteAnalysis>(std::move(input), std::move(result));
    }
    auto result = analyzeFiniteGuarded(input->function(), input->roots(), input->index(), input->storage(),
                                      std::move(expressions));
    return std::make_shared<FiniteAnalysis>(std::move(input), std::move(result));
}
} // namespace mlir::pto::frontiersynch
