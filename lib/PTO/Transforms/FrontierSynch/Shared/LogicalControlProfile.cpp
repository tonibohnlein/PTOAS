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
#include "MixedFamilyInsertion.h"
#include "PreparedControlLiveness.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
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
    uint64_t total = 0;
    for (const auto& stage : plan.preparation) {
        for (auto& operation : *stage.code) { total += arithmetic(&operation); }
    }
    auto count = [&](bool evidence) {
        auto live = livePreparedControl(plan, evidence);
        return llvm::count_if(live, arithmetic);
    };
    return llvm::json::Object{{"arithmetic", total}, {"endpoint_reachable_arithmetic", count(false)},
                             {"required_arithmetic", count(true)}, {"blocks", plan.preparation.size()}};
}
llvm::json::Object familyForms(const PreparedLogicalPlan& plan)
{
    if (!plan.familyRecipe) { return {}; }
    auto forms = familyEndpointForms(*plan.familyRecipe);
    if (failed(forms)) { return llvm::json::Object{{"error", "invalid family recipe inventory"}}; }
    return llvm::json::Object{{"numerical_sites", forms->numericalSites}, {"contextual_sites", forms->contextualSites},
            {"mixed_leaves", forms->mixedLeaves}, {"shared_planning_roots", forms->sharedRoots},
            {"shared_planning_lookups", forms->sharedLookups},
            {"shared_planning_candidates", forms->sharedCandidates},
            {"shared_planning_bindings", forms->sharedBindings},
            {"wrapper_discovery_nodes", forms->wrapperDiscoveryNodes},
            {"wrapper_discovery_pieces", forms->wrapperDiscoveryPieces},
            {"wrapper_index_imports", forms->wrapperIndexImports},
            {"wrapper_index_entries", forms->wrapperIndexEntries},
            {"wrapper_binding_requests", forms->wrapperBindingRequests}};
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
        sets += name == "pto.logical_set" || name == "pto.set_flag" || name == "pto.set_flag_dyn";
        waits += name == "pto.logical_wait" || name == "pto.wait_flag" || name == "pto.wait_flag_dyn";
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
    llvm::errs() << "frontier-control-profile "
                 << llvm::json::Value(
                        llvm::json::Object{
                            {"function", function.getSymName()},
                            {"stage", stage},
                            {"plan", plan.planId},
                            {"producer", plan.mathematicalOwner ? plan.mathematicalOwner->backend : "unreported"},
                            {"realization", plan.privateState ? "invocation-prefix" :
                                                                (plan.completionOwner ? "conservative-completion" :
                                                                                        "direct-endpoints")},
                            {"record_members", records},
                            {"families", plan.families.size()},
                            {"endpoint_sites", plan.endpoints.size()},
                            {"recipe_sites", plan.familyRecipe ? familyEndpointPieceCount(*plan.familyRecipe) : 0},
                            {"preparation", preparation(plan)},
                            {"recipe_forms", familyForms(plan)},
                            {"prefix_expressions",
                             plan.privateState ? plan.privateState->expressionProfile() : llvm::json::Object{}},
                            {"family_materialization",
                             llvm::json::Object{
                                 {"normalization_inputs", plan.familyWork.normalizationInputs},
                                 {"normalized_sites", plan.familyWork.normalizedSites},
                                 {"materializations", plan.familyWork.materializations},
                                 {"shared_published", plan.familyWork.sharedPublished},
                                 {"shared_requests", plan.familyWork.sharedRequests},
                                 {"shared_imports", plan.familyWork.sharedImports},
                                 {"shared_candidates", plan.familyWork.sharedCandidates},
                                 {"local_entries", plan.familyWork.localEntries},
                                 {"contexts", plan.familyWork.contexts}}},
                            {"function_scope", staged ? "staged-invocation" : "original-or-inserted"},
                            {"function_control", functionControl(staged ? staged : function)}})
                 << "\n";
}
} // namespace mlir::pto::frontiersynch
