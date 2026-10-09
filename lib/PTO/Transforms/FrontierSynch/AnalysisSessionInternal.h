// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Session cache entries retain demands independently of capability failures.
#ifndef PTO_FRONTIERSYNCH_ANALYSISSESSIONINTERNAL_H
#define PTO_FRONTIERSYNCH_ANALYSISSESSIONINTERNAL_H
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include <set>
namespace mlir::pto::frontiersynch {
struct BackendAttempt {
    bool produced = false;
    std::shared_ptr<const MathematicalResult> mathematical;
    std::string demandError;
    std::optional<bool> endpoints;
    std::shared_ptr<const NumericTemplateEndpoints> periodicEndpoints;
};
struct AnalysisSessionState {
    std::map<AnalysisBackend, BackendAttempt> rootAttempts;
    std::set<AnalysisBackend> wholeRegionEvidence;
};
} // namespace mlir::pto::frontiersynch
#endif
