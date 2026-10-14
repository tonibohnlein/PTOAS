// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Root-only owned private-state publication, separate from direct endpoint plans.
#ifndef PTO_FRONTIERSYNCH_PRIVATESTATEPLAN_H
#define PTO_FRONTIERSYNCH_PRIVATESTATEPLAN_H
#include "InvocationPrefix.h"
#include "llvm/Support/JSON.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
namespace mlir::pto::frontiersynch {
enum class PrivateStateTarget { Unavailable, ScalarControl64 };
class PrivateStatePlan {
public:
    ~PrivateStatePlan();
    LogicalResult insert(func::FuncOp function, PreparedLogicalPlan& plan);
    func::FuncOp preparedFunction() const;
    llvm::json::Object expressionProfile() const;
    // Trusted module coordinator only: call for EVERY pending private plan
    // before changing any sibling body. Subsequent edits may only insert the
    // already prepared synchronization or materialize physical section guards.
    // Failure abandons the coordinator's entire module clone. This is not
    // permission for arbitrary callee rewrites or user-visible partial commit.
    LogicalResult preflightSiblingSynchronization(func::FuncOp function, PreparedLogicalPlan& plan);
private:
    friend FailureOr<std::unique_ptr<PreparedLogicalPlan>> preparePrivateStatePlan(
        func::FuncOp, const InvocationPrefixRecipe&, std::shared_ptr<const MathematicalResult>,
        PrivateStateTarget, PrivateStateError&);
    PrivateStatePlan() = default;
    func::FuncOp original;
    ModuleOp originalModule;
    LogicalResult validate(func::FuncOp function, PreparedLogicalPlan& plan, bool fullModule);
    std::string originalText, functionText, contextText, stagedText;
    std::vector<Operation*> originalOperations;
    std::unique_ptr<StructuredPrivateState> staged;
    std::shared_ptr<const PrefixExpressionProfile> expressions;
    std::shared_ptr<const MathematicalResult> mathematics;
    int64_t planId = 0;
    bool consumed = false;
    bool siblingSynchronization = false;
};
// Caller supplies the complete invocation's retained mathematics and certified
// selectors. Materialization and snapshot capture happen together; no arbitrary
// prebuilt clone can be attached. Original IR must outlive the prepared plan.
// The caller certifies mathematics belongs to this complete function. Successful
// insertion replaces source anchors and invalidates source-dependent analyses.
// This private target capability does not imply physical allocation support.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> preparePrivateStatePlan(
    func::FuncOp original, const InvocationPrefixRecipe& recipe,
    std::shared_ptr<const MathematicalResult> mathematics, PrivateStateTarget target, PrivateStateError& error);
} // namespace mlir::pto::frontiersynch
#endif
