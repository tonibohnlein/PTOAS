// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Certified late expansion of numeric inner visits, retaining one outer loop.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATE_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
namespace mlir::pto::frontiersynch {
struct NumericTemplateLimits {
    uint64_t visits = 65536; // Expanded operation visits, including metadata.
    uint64_t payloads = 4096;
    uint64_t fragments = 65536; // Maps, concrete ranges, atoms and atom references.
    unsigned depth = 16;
};
struct TemplateCoordinate {
    scf::ForOp loop;
    int64_t induction = 0;
};
// Dimensions are local footprint coordinates; sole symbol s0 is the OUTER
// ITERATION ORDINAL. It is not an SSA value or a numeric pointer base.
struct TemplateRegion {
    Value base;
    AffineExpr byteOffset;
    SmallVector<AffineExpr> extents;
    unsigned elementBytes = 0;
};
enum class TemplateDischarge { None, ReadOnlyBase, IterationPrivateBase };
struct TemplateEffect {
    std::size_t sourceEffect = 0;
    SyncAccessMode mode = SyncAccessMode::Read;
    SmallVector<TemplateRegion> regions;
    SmallVector<SyncStorageCell> ranges; // Local absolute bytes, or GM at ordinal 0.
    SmallVector<std::size_t> atoms; // Indices into the template-wide physical partition.
    TemplateDischarge discharge = TemplateDischarge::None;
    int64_t outerStride = 0; // Certified translation for iteration-private GM.
};
struct TemplatePayload {
    const CompoundInstanceElement* phase = nullptr;
    // Together these identify an expanded occurrence. Original phase anchor
    // supplies its before/after cuts; no IR operation is cloned or moved.
    SmallVector<TemplateCoordinate> coordinates;
    SmallVector<TemplateEffect, 0> effects;
};
struct NumericTemplate {
    RecognitionResult result;
    scf::ForOp outer;
    // Original upper bound is outer.getUpperBound(); original IV = lower +
    // step*ordinal. A proven empty invocation needs no body effect schema.
    int64_t lower = 0;
    int64_t step = 1;
    // Valid only for the retained constant outer bounds, not a changed trip count.
    bool emptyInvocation = false;
    NumericTemplateLimits limits;
    uint64_t countedVisits = 0; // Preflight upper bound before allocating visits.
    uint64_t countedPayloads = 0; // Includes both arms before guard specialization.
    uint64_t fragments = 0;
    SmallVector<TemplatePayload, 0> payloads;
    SmallVector<SyncStorageCell> atoms;
    uint64_t period = 0; // 1 only after complete contract verification.
    uint64_t refresh = 0; // 1 for every written local atom; read-only atoms need none.
    // No selectors, reachability, allocation or synchronization are exported.
    // GM discharge is valid only for a whole-function outer-loop candidate;
    // all other payloads and prerequisites must be accounted for or rejected.
};
NumericTemplate recognizeNumericTemplate(scf::ForOp outer, const PhaseIndex& index,
                                         const SyncInput& input,
                                         NumericTemplateLimits limits = {});
} // namespace mlir::pto::frontiersynch
#endif
