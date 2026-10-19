// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared capture boundary for producer templates, independent of child policy.
#include "PTO/Transforms/FrontierSynch/Shared/LogicalFamilyFragment.h"
namespace mlir::pto::frontiersynch {
LogicalFamilyFragment::LogicalFamilyFragment(const PreparedLogicalPlan& plan)
    : planId(plan.planId), completeInvocation(plan.completeInvocation), nestedIdentities(plan.nestedIdentities),
      independentPieces(plan.independentPieces), groupedFamilies(plan.groupedFamilies),
      recipe(plan.familyRecipe), families(plan.families) {}
std::shared_ptr<const LogicalFamilyFragment> LogicalFamilyFragment::capture(
    const PreparedLogicalPlan& plan, Context context)
{
    // This compatibility capture is deliberately exhaustive. Producers may only
    // retain recipe state; callers re-establish allocation and request metadata
    // on each new plan. Never silently drop evidence from a general plan.
    const bool session = context == Context::Session;
    if ((session && !plan.familyRecipe) || !plan.preparation.empty() || !plan.endpoints.empty() ||
        plan.allocationPreparation || plan.allocationCertificate || plan.regionalAllocation || plan.privateState ||
        plan.familyProtocol ||
        plan.physicalOutput || plan.executedOutput || plan.allocationProofs || (!session && plan.mathematicalOwner) ||
        plan.deferredOutput || plan.completionOwner || plan.completionPhysicalIds || plan.compactBoundingOwner ||
        plan.recognitionReport ||
        plan.allocationProfile || plan.profileControl ||
        llvm::any_of(plan.families, [session](const auto& family) {
            return (family.sourceChoices && (!session || family.sourceChoices->consumerPredicate())) ||
                (family.targetChoices && (!session || family.targetChoices->consumerPredicate()));
        })) { return {}; }
    return std::shared_ptr<const LogicalFamilyFragment>(new LogicalFamilyFragment(plan));
}
std::unique_ptr<PreparedLogicalPlan> LogicalFamilyFragment::instantiate() const
{
    auto plan = std::make_unique<PreparedLogicalPlan>(planId);
    plan->completeInvocation = completeInvocation;
    plan->nestedIdentities = nestedIdentities;
    plan->independentPieces = independentPieces;
    plan->groupedFamilies = groupedFamilies;
    plan->familyRecipe = recipe;
    plan->families = families;
    return plan;
}
} // namespace mlir::pto::frontiersynch
