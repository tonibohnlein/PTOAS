// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Allocation-only lowering of a certified logical plan. No scarcity repair.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTALLOCATION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include <map>
namespace mlir::pto::frontiersynch {
// Sufficient allocation strategies, not minimum-capacity claims. Empty attr
// means the available exact query representation did not prove uniform reuse.
// Arithmetic records use constant source identity zero. Every pair of retained
// instances of one record is checked, including different pieces/residues.
DictionaryAttr arithmeticAllocationCertificate(const ArithmeticDemandAnalysis& analysis,
    ArrayRef<uint32_t> pipes, const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records,
    int64_t plan, MLIRContext* context);
// Immutable retained guards select a guaranteed-active numerical witness graph.
// Distinct potential record phases receive disjoint cyclic subsequences.
DictionaryAttr guardedPeriodicAllocationCertificate(RegionExpressions& expressions,
    ArrayRef<GuardedPeriodicPayload> payloads, ArrayRef<GuardedPeriodicRecord> generators,
    const GuardedPeriodicQuotient& periodic, int64_t plan, MLIRContext* context);
DictionaryAttr guardedAllocationCertificate(const GuardedRotatingAnalysis& analysis,
    int64_t plan, MLIRContext* context);
} // namespace mlir::pto::frontiersynch
#endif
