// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared modeled accesses merged at balanced structural slots.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_CONDITIONALCOMPACTINPUT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_CONDITIONALCOMPACTINPUT_H
#include "PTO/Transforms/FrontierSynch/BalancedCompactBody.h"
#include "PTO/Transforms/FrontierSynch/CompactOrderBounds.h"
namespace mlir::pto::frontiersynch {
struct ConditionalCompactInput {
    std::string error;
    std::shared_ptr<const BalancedCompactBody> body;
    CompactWriterReaderInput storage;
    CompactWriterReaderAnalysis mathematical;
};
// One structural slot occurs per visit, not one occurrence per branch arm.
// Every alternative's original effects contribute to may-read/write unions;
// source bounds include all effect pairs and protection removes a slot demand
// only when every possible contributor is covered. Same-slot pairs refer to
// different visits, never simultaneous mutually exclusive alternatives.
// Full-overwrite proofs would need intersection over every alternative; shared
// geometry alone does not establish them, so this adapter exports no kills.
// Default native prerequisites require every alternative pair. Arm-specific
// facts become conservative software upper demands. Boundary/control/carried
// obligations remain explicit in storage/body; this does not certify placement.
// The same shared extraction core serves ordinary fixed bodies. Its comparison
// costs depend on static effects/alternatives, never branch paths or trip counts.
// Successful mathematical results survive unavailable endpoint/query exports;
// original IR and SyncInput remain borrowed and unchanged.
ConditionalCompactInput buildConditionalCompactInput(scf::ForOp loop,
    const SyncInput& input, const PhaseIndex& index, const CompactWriterReaderBindings& bindings = {});
} // namespace mlir::pto::frontiersynch
#endif
