// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared geometry validation. Entries preserve supplied fragment order so
// guaranteed-writer coverage remains associated with its producer evidence.
#ifndef PTO_FRONTIERSYNCH_ROTATINGNORMALIZATION_H
#define PTO_FRONTIERSYNCH_ROTATINGNORMALIZATION_H
#include "PTO/Transforms/FrontierSynch/Structured/RotatingExtraction.h"
namespace mlir::pto::frontiersynch {
struct NormalizedRotatingFragment {
    RotatingFragment fragment;
    uint64_t residue = 0, phase = 0, period = 0;
};
struct RotatingNormalization {
    std::string error;
    std::vector<NormalizedRotatingFragment> accesses;
    uint64_t refreshBound = 0;
    uint64_t inspectedFragments = 0, normalizedFamilies = 0;
};
// Validates the same geometry/protection contract as generator extraction,
// without constructing generators or expanding slots/iterations. O(A log A)
// work plus one GCD/inverse per family. Invalid input returns no partial data.
RotatingNormalization normalizeRotatingFragments(llvm::ArrayRef<PeriodicPayload> payloads,
    llvm::ArrayRef<RotatingFragment> fragments);
} // namespace mlir::pto::frontiersynch
#endif
