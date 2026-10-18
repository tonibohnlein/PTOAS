// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Scalar-free immutable producer fragment. Instantiation copies only mutable
// inventories; qualification of matching, resources and invocation stays with
// the caller. No allocation callback, generated SSA or finalized protocol lives here.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALFAMILYFRAGMENT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALFAMILYFRAGMENT_H
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
namespace mlir::pto::frontiersynch {
class LogicalFamilyFragment {
public:
    enum class Context { Producer, Session };
    // Session restores mathematical ownership on each fresh plan and may retain
    // immutable terminal choices that contain no generated consumer predicate.
    static std::shared_ptr<const LogicalFamilyFragment> capture(
        const PreparedLogicalPlan& plan, Context context = Context::Producer);
    std::unique_ptr<PreparedLogicalPlan> instantiate() const;
private:
    explicit LogicalFamilyFragment(const PreparedLogicalPlan& plan);
    int64_t planId;
    bool completeInvocation, nestedIdentities, independentPieces, groupedFamilies;
    std::shared_ptr<const FamilyEndpointRecipe> recipe;
    std::vector<EndpointFamily> families;
};
} // namespace mlir::pto::frontiersynch
#endif
