// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact local-storage analysis of a certified numeric inner template.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEANALYSIS_H
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
namespace mlir::pto::frontiersynch {
// Result type IDs are indices into NumericTemplate::payloads. Original phase,
// fixed inner coordinates and cuts remain in that template, without IR cloning.
// The template's whole-function GM discharge and period-one refresh certificate
// are prerequisites; no external storage selectors are synthesized here.
PeriodicAnalysis analyzeNumericTemplate(const NumericTemplate& input);
} // namespace mlir::pto::frontiersynch
#endif
