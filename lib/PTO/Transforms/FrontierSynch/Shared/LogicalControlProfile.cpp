// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Aggregate preparation and emitted-control inventories. Reachability here is
// structural SSA liveness, not constant propagation or a dynamic cost model.
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PrivateStatePlan.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
namespace {
bool arithmetic(Operation* operation)
{
    return operation->getName().getDialectNamespace() == "arith";
}
llvm::json::Object preparation(const PreparedLogicalPlan& plan)
{
    llvm::DenseSet<Operation*> operations, live;
    SmallVector<Operation*> pending;
    uint64_t total = 0;
    for (const auto& stage : plan.preparation) {
        for (auto& operation : *stage.code) {
            operations.insert(&operation);
            total += arithmetic(&operation);
            if (!isMemoryEffectFree(&operation)) { pending.push_back(&operation); }
        }
    }
    auto root = [&](Value value) {
        if (auto* operation = value ? value.getDefiningOp() : nullptr) { pending.push_back(operation); }
    };
    for (const auto& endpoint : plan.endpoints) {
        root(endpoint.guard); root(endpoint.identity);
        for (auto value : endpoint.memberCoordinates) { root(value); }
    }
    uint64_t reachable = 0;
    while (!pending.empty()) {
        auto* operation = pending.pop_back_val();
        if (!operations.contains(operation) || !live.insert(operation).second) { continue; }
        reachable += arithmetic(operation);
        for (auto value : operation->getOperands()) { root(value); }
    }
    return llvm::json::Object{{"arithmetic", total}, {"endpoint_reachable_arithmetic", reachable},
                             {"blocks", plan.preparation.size()}};
}
llvm::json::Object functionControl(func::FuncOp function)
{
    uint64_t arithmeticCount = 0, selects = 0, divisions = 0, branches = 0, sets = 0, waits = 0, barriers = 0;
    function.walk([&](Operation* operation) {
        const auto name = operation->getName().getStringRef();
        arithmeticCount += arithmetic(operation);
        selects += name == "arith.select";
        divisions += name == "arith.divui" || name == "arith.divsi" ||
                     name == "arith.remui" || name == "arith.remsi";
        branches += name == "scf.if";
        sets += name == "pto.logical_set";
        waits += name == "pto.logical_wait";
        barriers += name == "pto.barrier";
    });
    llvm::json::Object result{{"arithmetic_including_payload", arithmeticCount}, {"selects", selects},
        {"div_rem", divisions}, {"ifs", branches}, {"sets", sets}, {"waits", waits}, {"barriers", barriers}};
    if (auto protocol = function->getAttrOfType<DictionaryAttr>("pto.prefix_protocol")) {
        if (auto slots = protocol.getAs<IntegerAttr>("state_slots")) { result["state_slots"] = slots.getInt(); }
    }
    return result;
}
} // namespace
void profileLogicalControl(func::FuncOp function, const PreparedLogicalPlan& plan, StringRef stage)
{
    if (!plan.profileControl) { return; }
    uint64_t records = 0;
    for (const auto& family : plan.families) { records += family.members.size(); }
    auto staged = plan.privateState ? plan.privateState->preparedFunction() : func::FuncOp{};
    llvm::errs() << "frontier-control-profile " << llvm::json::Value(llvm::json::Object{
        {"function", function.getSymName()}, {"stage", stage}, {"plan", plan.planId},
        {"producer", plan.mathematicalOwner ? plan.mathematicalOwner->backend : "unreported"},
        {"realization", plan.privateState ? "invocation-prefix" :
            (plan.completionOwner ? "conservative-completion" : "direct-endpoints")},
        {"record_members", records}, {"families", plan.families.size()},
        {"endpoint_sites", plan.endpoints.size()},
        {"recipe_sites", plan.familyRecipe ? familyEndpointPieceCount(*plan.familyRecipe) : 0},
        {"preparation", preparation(plan)},
        {"function_scope", staged ? "staged-invocation" : "original-or-inserted"},
        {"function_control", functionControl(staged ? staged : function)}}) << "\n";
}
} // namespace mlir::pto::frontiersynch
