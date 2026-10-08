// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGREGION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGREGION_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
namespace mlir::pto::frontiersynch {
// Numerical affine bank-map profile of the rotating-visit theorem. General
// injective bank maps require their own inverse/geometry adapter. One bank per
// visit. The caller proves that the complete child becomes
// invariant after renaming that bank. Different families are physically
// disjoint; local cells and effects retain their original shared identities.
struct RotatingRegionFamily {
    SyncStorageCell firstBank;
    uint64_t bankStride = 0, banks = 1, stride = 0, offset = 0;
    std::vector<std::size_t> effects;
};
struct RotatingRegionInput {
    RegionalAnalysis child;
    std::vector<RotatingRegionFamily> families;
    std::vector<RepeatedCrossing> prerequisites;
    // Original effects omitted from the bank profile. Their owned/read-only
    // physical maps are checked by the shared residual-storage construction.
    std::vector<std::size_t> residualEffects;
};
// Finite local selector partition, compact bank family geometry. Neither banks,
// visits nor joint phases are expanded. All residual effects must be accounted
// for by the supplied families or checked residual maps. Per-cell refresh is
// checked before constructing the crossing graph. The result
// exports original-coordinate logical endpoints; allocation remains separate.
RepeatedRegionAnalysis repeatRotatingRegion(func::FuncOp function, scf::ForOp loop,
    RotatingRegionInput input, RegionExpressions::Id trips);
RepeatedRegionAnalysis recognizeRotatingRegion(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, const PhaseIndex& index,
    std::shared_ptr<RegionExpressions> expressions);
} // namespace mlir::pto::frontiersynch
#endif
