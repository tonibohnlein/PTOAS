// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Lift invariant symbolic storage through conflict-free readers or complete finite support.
#ifndef PTO_FRONTIERSYNCH_REPEATEDREADONLYSTORAGE_H
#define PTO_FRONTIERSYNCH_REPEATEDREADONLYSTORAGE_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
namespace mlir::pto::frontiersynch {
// Materialize exact singleton boundary atoms only from complete finite support.
// A failure leaves the input unchanged. The expansion is representation-bounded.
// periodEffects, when supplied by phased composition, names effects with
// validated extrema in sibling phases of this same period/shared model. It
// satisfies only the collective coverage check; it never supplies selectors.
std::string materializeRepeatedSymbolicStorage(RegionalAnalysis& body, scf::ForOp loop, RegionExpressions::Id trips,
                                              ArrayRef<std::size_t> periodEffects = {});
// Qualify one invariant phase and construct only its crossing view. Preserve
// the original view separately for exact byte callbacks and effect exports.
// Reuses the same read-only/all-writer or finite-support proof as repetition;
// failure leaves body unchanged. No repeated graph or visit expansion occurs.
std::string prepareRepeatedSymbolicStorage(RegionalAnalysis& body, scf::ForOp loop, RegionExpressions::Id trips,
                                          ArrayRef<std::size_t> periodEffects = {});
RepeatedRegionAnalysis repeatSymbolicStorageRegion(func::FuncOp function, scf::ForOp loop,
    RegionalAnalysis body, RegionExpressions::Id trips);
} // namespace mlir::pto::frontiersynch
#endif
