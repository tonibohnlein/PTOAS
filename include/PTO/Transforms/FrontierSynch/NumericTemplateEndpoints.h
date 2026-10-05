// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bind generic logical recipes to original structured-program endpoint cuts.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEENDPOINTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEENDPOINTS_H
#include "PTO/Transforms/FrontierSynch/LogicalEndpoints.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
namespace mlir::pto::frontiersynch {
struct TemplateEndpointCut {
    Block* block = nullptr;
    Operation* before = nullptr; // Null denotes block end.
    // Before(op)=(block,op); After(op)=(block,op.getNextNode()). Equal pairs
    // denote an actual coincident insertion point, subject to endpoint guards.
};
struct TemplateEndpointAnchor {
    const CompoundInstanceElement* phase = nullptr;
    SmallVector<TemplateCoordinate> coordinates; // Own endpoint's complete tuple.
    TemplateEndpointCut before;
    TemplateEndpointCut after;
};
struct TemplateEndpointGroup {
    TemplateEndpointCut cut;
    SmallVector<uint32_t> recipes; // Filter by each recipe's own coordinate and ordinal guards first.
};
struct NumericTemplateEndpoints {
    LogicalEndpointPlan logical;
    scf::ForOp outer; // Bounds are evaluated at this loop entry and stay available.
    std::vector<TemplateEndpointAnchor> anchors; // Indexed by template type.
    // Static cut groups. Evaluate each separately at each dynamic visit, after
    // matching the endpoint's complete inner tuple and outer-ordinal predicate.
    std::vector<TemplateEndpointGroup> groups;
    // Original IR and SyncInput must remain alive and unchanged. The shared
    // logical identity is (record, source outer ordinal), never an EventAttr.
};
// Bind a certified periodic relation to supplied original occurrence anchors.
NumericTemplateEndpoints bindPeriodicEndpoints(scf::ForOp outer,
    ArrayRef<TemplateEndpointAnchor> anchors, const PeriodicAnalysis& analysis);
NumericTemplateEndpoints buildNumericTemplateEndpoints(const NumericTemplate& input,
                                                       const PeriodicAnalysis& analysis);
} // namespace mlir::pto::frontiersynch
#endif
