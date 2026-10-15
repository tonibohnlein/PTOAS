// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Internal exact relation kernels, exposed to independent construction oracles.
// Callers supply validated endpoint schemas with bounded dimensions.
#ifndef PTO_FRONTIERSYNCH_ARITHMETICHANDOFFPROOF_H
#define PTO_FRONTIERSYNCH_ARITHMETICHANDOFFPROOF_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffAllocation.h"
namespace mlir::pto::frontiersynch::detail {
FailureOr<GeneralArithmeticRelation> materializeHandoffComposition(const GeneralArithmeticRelation& left,
    const GeneralArithmeticRelation& right, unsigned parameters, ArithmeticAnalysisCost& cost);
FailureOr<bool> handoffCompositionCovered(const GeneralArithmeticRelation& left,
    const GeneralArithmeticRelation& right, const GeneralArithmeticRelation& required,
    unsigned parameters, ArithmeticAnalysisCost& cost, AllocationProfile* profile = nullptr);
} // namespace mlir::pto::frontiersynch::detail
#endif
