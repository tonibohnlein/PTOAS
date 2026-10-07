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
std::string materializeRepeatedSymbolicStorage(RegionalAnalysis& body, scf::ForOp loop, RegionExpressions::Id trips);
RepeatedRegionAnalysis repeatSymbolicStorageRegion(func::FuncOp function, scf::ForOp loop,
    RegionalAnalysis body, RegionExpressions::Id trips);
} // namespace mlir::pto::frontiersynch
#endif
