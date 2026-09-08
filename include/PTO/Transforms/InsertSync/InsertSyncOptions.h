// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_INSERTSYNCOPTIONS_H
#define PTO_TRANSFORMS_INSERTSYNC_INSERTSYNCOPTIONS_H
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir::pto {
// Component-local configuration avoids coupling every PTO transform to
// InsertSync's rollout options. The zero-argument factory remains compatible.
struct InsertSyncOptions {
    bool deferSamePipe = false;
    std::string gmAlias;
    std::string audit = "off";
    std::string effectCoverage = "report";
    bool pruneCompletedBarriers = false;
    bool mmadChains = false;
    bool frontierRefinement = false;
    bool frontierPlacement = false;
    bool lifecycleSynthesis = false;
    bool bufferGenerations = false;
};
std::unique_ptr<Pass> createPTOInsertSyncPass(const InsertSyncOptions& options);
} // namespace mlir::pto
#endif
