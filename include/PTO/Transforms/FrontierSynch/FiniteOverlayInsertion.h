// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEOVERLAYINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEOVERLAYINSERTION_H
#include "PTO/Transforms/FrontierSynch/FiniteOverlay.h"
namespace mlir::pto::frontiersynch {
// Uses filtered base recipes and adds the retained finite demands at their
// original cuts, binding the existing hierarchical occurrence frame. Typed base
// lifetimes survive filtering; added singleton lifetimes use augmented queries.
// Local added covers emit a barrier immediately before the consumer.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteOverlayInsertion(
    func::FuncOp function, const FiniteOverlayAnalysis& analysis,
    std::string& error);
// Export this selected graph for sequence composition when a filtered base
// producer exists. Preparation still checks cuts and executable guards. This
// wrapper does not bind additional enclosing visits or export further filtering.
RegionalAnalysis finiteOverlayRegionalResult(func::FuncOp function,
    const FiniteOverlayAnalysis& analysis);
} // namespace mlir::pto::frontiersynch
#endif
