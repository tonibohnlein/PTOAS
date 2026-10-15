// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/AllocationPreparation.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "llvm/Support/SaveAndRestore.h"
#include <array>
namespace mlir::pto::frontiersynch {
struct AllocationPreparation::State {
    Legacy legacy;
    Builder build;
    std::array<Summary, 2> answers;
    std::array<bool, 2> complete{false, false};
    bool active = false;
    bool oneShot = false;
    DictionaryAttr certificate;
};
AllocationPreparation& AllocationPreparation::operator=(Legacy prepare)
{
    if (!prepare) { state.reset(); return *this; }
    state = std::make_shared<State>();
    state->legacy = std::move(prepare);
    state->oneShot = true;
    return *this;
}
AllocationPreparation AllocationPreparation::candidates(Builder build)
{
    AllocationPreparation result;
    result.state = std::make_shared<State>();
    result.state->build = std::move(build);
    return result;
}
AllocationPreparation::Summary AllocationPreparation::request(
    PreparedLogicalPlan& plan, AllocationStrength strength)
{
    if (!state) { return plan.regionalAllocation; }
    auto owner = state; // A compatibility callback can replace the plan's owner.
    const unsigned index = strength == AllocationStrength::Independent ? 0 : 1;
    if (owner->complete[index]) { return owner->answers[index]; }
    if (owner->complete[1]) { return owner->answers[1]; }
    if (owner->active) { return {}; }
    llvm::SaveAndRestore<Summary> summary(plan.regionalAllocation);
    llvm::SaveAndRestore<DictionaryAttr> certificate(plan.allocationCertificate);
    owner->active = true;
    if (owner->legacy) {
        auto prepare = std::move(owner->legacy);
        prepare(plan);
        owner->answers[0] = owner->answers[1] = plan.regionalAllocation;
        owner->certificate = plan.allocationCertificate;
        owner->complete = {true, true};
    } else {
        owner->answers[index] = owner->build ? owner->build(plan, strength) : Summary{};
        owner->complete[index] = true;
        if (index == 1) { owner->build = {}; }
    }
    owner->active = false;
    return owner->answers[index];
}
void AllocationPreparation::operator()(PreparedLogicalPlan& plan)
{
    auto owner = state;
    auto selected = request(plan, AllocationStrength::Shared);
    if (owner && owner->oneShot && owner->complete[1]) { plan.allocationCertificate = owner->certificate; }
    if (selected) { plan.regionalAllocation = std::move(selected); }
}
AllocationPreparation::operator bool() const
{
    return state && (state->legacy || state->build);
}
std::shared_ptr<RegionalAllocationSummary> requestAllocationSummary(
    PreparedLogicalPlan& plan, AllocationStrength strength, AllocationProfile* profile)
{
    if (plan.privateState || plan.completionOwner || failed(normalizeLogicalFamilies(plan))) {
        return {};
    }
    llvm::SaveAndRestore<AllocationProfile*> profiling(plan.allocationProfile, profile);
    return plan.allocationPreparation.request(plan, strength);
}
} // namespace mlir::pto::frontiersynch
