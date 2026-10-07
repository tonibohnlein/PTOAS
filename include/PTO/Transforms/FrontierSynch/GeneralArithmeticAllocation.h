// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact reuse certification for general restricted-arithmetic demand relations.
#ifndef PTO_FRONTIERSYNCH_GENERALARITHMETICALLOCATION_H
#define PTO_FRONTIERSYNCH_GENERALARITHMETICALLOCATION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
namespace mlir::pto::frontiersynch {
// Assign one dedicated ID per nonlocal (source site, target site) family.
// Every pair of retained instances with increasing original source coordinates
// must satisfy C(earlier target) -> I(later source), in their shared parameter
// context. Exact projection/subtraction preserves guards and congruences.
// Source identity remains zero; the paired source tuple is a member coordinate.
// Empty attr means this sufficient strategy was not certified, not that no
// allocation exists. This does not optimize capacity or repair event scarcity.
DictionaryAttr generalArithmeticAllocationCertificate(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes,
    const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records, int64_t plan, MLIRContext* context);
} // namespace mlir::pto::frontiersynch
#endif
