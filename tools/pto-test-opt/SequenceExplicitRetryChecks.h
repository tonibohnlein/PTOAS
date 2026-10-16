// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_SEQUENCE_EXPLICIT_RETRY_CHECKS_H
#define PTO_FRONTIER_SEQUENCE_EXPLICIT_RETRY_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::test {
inline bool checkSequenceExplicitRetry(const frontiersynch::SequenceAnalysis& analysis)
{
    namespace fs = frontiersynch;
    if (!analysis.state) { return false; }
    auto sourceIR = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text);
        analysis.state->function.print(stream);
        return text;
    };
    const auto original = sourceIR();
    unsigned checked = 0;
    for (const auto& child : analysis.state->children) {
        if (child.loop || child.explicitAnalysis.reduction.retained.empty()) { continue; }
        const auto prepare = child.regional.prepare;
        if (!prepare) { return false; }
        auto left = prepare(), right = prepare();
        if (failed(left) || failed(right) || left->get() == right->get() ||
            !(*left)->familyRecipe || (*left)->familyRecipe != (*right)->familyRecipe ||
            (*left)->families.empty() || (*right)->families.empty()) { return false; }
        auto scalarFree = [](const fs::PreparedLogicalPlan& plan) {
            return plan.endpoints.empty() && plan.preparation.empty() && !plan.completeInvocation &&
                !plan.allocationCertificate && !plan.allocationPreparation && !plan.regionalAllocation;
        };
        if (!scalarFree(**left) || !scalarFree(**right)) { return false; }
        const auto owner = (*left)->familyRecipe;
        const auto pipe = (*left)->families.front().sourcePipe;
        (*left)->families.front().sourcePipe = 7;
        auto third = prepare();
        if (failed(third) || (*third)->familyRecipe != owner || !scalarFree(**third) ||
            (*right)->families.front().sourcePipe != pipe || (*third)->families.front().sourcePipe != pipe) {
            return false;
        }
        (*left)->families.front().sourcePipe = pipe;
        if (failed(fs::normalizeLogicalFamilies(**left)) || failed(fs::normalizeLogicalFamilies(**right)) ||
            fs::normalizedFamilyProtocolKey(*(*left)->familyProtocol) !=
                fs::normalizedFamilyProtocolKey(*(*right)->familyProtocol)) { return false; }
        if (failed(fs::materializeLogicalFamilies(**left)) || (*left)->endpoints.empty() ||
            !scalarFree(**third)) { return false; }
        left->reset();
        if (failed(fs::materializeLogicalFamilies(**right)) || (*right)->endpoints.empty() ||
            (*third)->familyRecipe != owner || !scalarFree(**third)) { return false; }
        ++checked;
    }
    return checked != 0 && sourceIR() == original;
}
} // namespace mlir::pto::test
#endif
