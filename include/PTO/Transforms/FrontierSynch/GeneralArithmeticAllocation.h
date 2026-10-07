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
#include "PTO/Transforms/FrontierSynch/ArithmeticHandoffAllocation.h"
#include "mlir/IR/BuiltinAttributes.h"
namespace mlir::pto::frontiersynch {
// Width-one proofs retain the static serializer. Wider families use certified
// executed-rank counters at SET/WAIT, never original ordinals modulo E.
// Empty means unavailable sufficient allocation, never event-ID scarcity.
DictionaryAttr encodeGeneralArithmeticAllocationCertificate(
    const ArithmeticHandoffAllocation& allocation, ArrayRef<uint32_t> pipes,
    const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records, int64_t plan, MLIRContext* context);
DictionaryAttr generalArithmeticAllocationCertificate(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes,
    const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records, int64_t plan, MLIRContext* context);
} // namespace mlir::pto::frontiersynch
#endif
