// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One original finite input and one selected representation. Numerical results
// remain numerical; endpoint preparation is a capability, not another analysis.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEANALYSIS_H
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteGuardedAnalysis.h"
#include <variant>
namespace mlir::pto::frontiersynch {
enum class FiniteStrategy { Concrete, Guarded };
// Shared original roots, effects and occurrence index. Original MLIR is borrowed
// and must remain unchanged; retaining this object retains all analysis facts.
class FiniteOccurrenceInput {
public:
    func::FuncOp function() const { return originalFunction; }
    ArrayRef<Operation*> roots() const { return originalRoots; }
    const SyncInput& storage() const { return *inputView; }
    std::shared_ptr<const SyncInput> storageOwner() const { return inputOwner; }
    bool wholeFunction() const { return whole; }
    uint64_t rootCopies() const { return *rootCopyWork; }
    const PhaseIndex& index() const { return *indexOwner; }
private:
    friend std::shared_ptr<const FiniteOccurrenceInput> finiteFunctionInput(func::FuncOp,
        std::shared_ptr<const SyncInput>, std::shared_ptr<const PhaseIndex>);
    friend std::shared_ptr<const FiniteOccurrenceInput> finiteBorrowedFunctionInput(
        func::FuncOp, const SyncInput&, std::shared_ptr<const PhaseIndex>);
    friend std::shared_ptr<const FiniteOccurrenceInput> finiteRegionInput(
        std::shared_ptr<const FiniteOccurrenceInput>, ArrayRef<Operation*>);
    FiniteOccurrenceInput(func::FuncOp function, const SyncInput& storage,
                          std::shared_ptr<const PhaseIndex> index, std::shared_ptr<uint64_t> work = {})
        : originalFunction(function), inputView(&storage),
          rootCopyWork(work ? std::move(work) : std::make_shared<uint64_t>(0)), indexOwner(std::move(index)) {}
    func::FuncOp originalFunction;
    SmallVector<Operation*> originalRoots;
    const SyncInput* inputView;
    std::shared_ptr<const SyncInput> inputOwner;
    bool whole = true;
    std::shared_ptr<uint64_t> rootCopyWork;
    std::shared_ptr<const PhaseIndex> indexOwner;
};
class LogicalFamilyFragment;
class FiniteAnalysis {
public:
    FiniteAnalysis(std::shared_ptr<const FiniteOccurrenceInput> input, ExplicitAnalysis result)
        : input(std::move(input)), representation(std::move(result)) {}
    FiniteAnalysis(std::shared_ptr<const FiniteOccurrenceInput> input, FiniteGuardedAnalysis result)
        : input(std::move(input)), representation(std::move(result)) {}
    // Diagnostic work only; never used for acceptance or semantic decisions.
    struct Work { uint64_t numericalQueries = 0, boundaryEntries = 0, recipeBuilds = 0, recipeHits = 0; } work;
    const ExplicitAnalysis* concrete() const { return std::get_if<ExplicitAnalysis>(&representation); }
    const FiniteGuardedAnalysis* guarded() const { return std::get_if<FiniteGuardedAnalysis>(&representation); }
    const std::string& error() const;
    const std::string& insertionError() const { return preparationError; }
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepare();
    std::shared_ptr<const FiniteOccurrenceInput> originalInput() const { return input; }
private:
    std::shared_ptr<const FiniteOccurrenceInput> input;
    std::variant<ExplicitAnalysis, FiniteGuardedAnalysis> representation;
    std::string preparationError;
    std::shared_ptr<const LogicalFamilyFragment> recipe;
};
std::shared_ptr<const FiniteOccurrenceInput> finiteFunctionInput(func::FuncOp function,
    std::shared_ptr<const SyncInput> storage, std::shared_ptr<const PhaseIndex> index);
// Legacy reference-based sequence entry points borrow storage explicitly. They
// cannot advertise detached prefix ownership. Production passes use the owner factory.
std::shared_ptr<const FiniteOccurrenceInput> finiteBorrowedFunctionInput(func::FuncOp function,
    const SyncInput& storage, std::shared_ptr<const PhaseIndex> index);
// One adjacent original block slice, including indivisible structured operations.
std::shared_ptr<const FiniteOccurrenceInput> finiteRegionInput(
    std::shared_ptr<const FiniteOccurrenceInput> source, ArrayRef<Operation*> roots);
RegionalAnalysis finiteRegionalResult(std::shared_ptr<FiniteAnalysis> analysis,
    std::shared_ptr<RegionExpressions> expressions = {});
std::shared_ptr<FiniteAnalysis> analyzeFinite(std::shared_ptr<const FiniteOccurrenceInput> input,
    FiniteStrategy strategy, std::shared_ptr<RegionExpressions> expressions = {});
} // namespace mlir::pto::frontiersynch
#endif
