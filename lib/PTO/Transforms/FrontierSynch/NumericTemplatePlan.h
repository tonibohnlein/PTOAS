// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_NUMERICTEMPLATEPLAN_H
#define PTO_FRONTIERSYNCH_NUMERICTEMPLATEPLAN_H
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
#include "NormalizedControl.h"
namespace mlir::pto::frontiersynch {
// Bounded control preflight only: no effects, atoms, reduction or endpoints.
// Records own their vectors; IR/input/index and callback contexts are borrowed
// from one unchanged invocation. Applicable is not an exact demand certificate.
struct NumericTemplatePlan {
    NumericTemplate form;
    const PhaseIndex* index = nullptr;
    const SyncInput* input = nullptr;
    bool regional = false;
    std::shared_ptr<const NormalizedControlDescription> normalized;
    SmallVector<TemplatePayload> payloads; // Original identities/coordinates; no effects.
    TemplateGeometryConstant geometry;
    TemplateControlConstant control;
};
NumericTemplatePlan preflightNumericTemplate(scf::ForOp outer, const PhaseIndex& index,
    const SyncInput& input, NumericTemplateLimits limits = {}, bool regional = false,
    TemplateGeometryConstant geometry = {}, TemplateControlConstant control = {},
    std::shared_ptr<const NormalizedControlDescription> normalized = {},
    TemplateGeometryPolicy policy = TemplateGeometryPolicy::AllCertifiedMaps);
NumericTemplate materializeNumericTemplate(const NumericTemplatePlan& plan,
    const PhaseIndex& index, const SyncInput& input);
} // namespace mlir::pto::frontiersynch
#endif
