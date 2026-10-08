// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owning MLIR analysis for the frontier synchronization pass. Step 0 storage
// outlives all borrowed structure/recognition links. Default MLIR invalidation
// discards this state after an IR-changing pass.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERANALYSIS_H
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
namespace mlir::pto::frontiersynch {
// Detached whole-function producer shared by module coordination and tests.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFunctionSynchronization(
    func::FuncOp function, GMAliasPolicy policy);
class FrontierAnalysis {
public:
    explicit FrontierAnalysis(Operation* operation) : function(dyn_cast<func::FuncOp>(operation)) {}
    // Callers needing structural diagnostics retain the default. The logical
    // pass may defer recognition when the whole-function native proof succeeds.
    LogicalResult initialize(GMAliasPolicy policy = GMAliasPolicy::MayNotAlias, bool requireStructure = true);
    // Build and cache the whole-function arithmetic candidate only on request.
    // Requires successful initialization; uses fixed class limits, not input-derived limits.
    LogicalResult recognizeArithmetic();
    // Demand reduction, endpoint recipes and allocation are backend work.
    // Cache both successful and failed numeric attempts without changing recognition.
    LogicalResult analyzeNumericCandidates();
    bool hasOnlyNativeScalarRequirements() const { return nativeScalarOnly; }
    LogicalResult analyzeExplicitFunction();
    SequenceAnalysis* analyzeSequenceFunction();
    LogicalResult analyzeArithmeticFunction();
    const ArithmeticDemandAnalysis* arithmeticDemands() const {
        return arithmeticAnalysis ? &*arithmeticAnalysis : nullptr;
    }
    const GeneralArithmeticDemandAnalysis* generalArithmeticDemands() const {
        return generalArithmeticAnalysis ? &*generalArithmeticAnalysis : nullptr;
    }
    // Whole-invocation evidence only. Endpoint preparation cannot revoke it;
    // success for a proper child does not establish this property.
    bool hasWholeFunctionMinimumDemands() const;
    void noteSequenceEndpointOutcome(StringRef error);
    const ExplicitAnalysis* explicitResult() const { return explicitAnalysis ? &*explicitAnalysis : nullptr; }
    const SyncInput* input() const { return storage.get(); }
    std::shared_ptr<const SyncInput> sharedInput() const { return storage; }
    const ProgramRecognition* result() const { return program ? &*program : nullptr; }
private:
    LogicalResult recognizeStructure();
    func::FuncOp function;
    bool initialized = false;
    bool nativeScalarOnly = false;
    GMAliasPolicy policy = GMAliasPolicy::MayNotAlias;
    std::shared_ptr<SyncInput> storage;
    std::optional<ProgramRecognition> program;
    std::optional<ExplicitAnalysis> explicitAnalysis;
    std::optional<SequenceAnalysis> sequenceAnalysis;
    std::optional<ArithmeticDemandAnalysis> arithmeticAnalysis;
    std::optional<GeneralArithmeticDemandAnalysis> generalArithmeticAnalysis;
};
} // namespace mlir::pto::frontiersynch
#endif
