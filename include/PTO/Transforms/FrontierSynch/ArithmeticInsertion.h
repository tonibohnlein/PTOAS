// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICINSERTION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticSelectors.h"
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticSelectors.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include <map>
namespace mlir::pto::frontiersynch {
using ArithmeticRecordMap = std::map<std::pair<std::size_t, std::size_t>, int64_t>;
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticLogicalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const ArithmeticDemandAnalysis& analysis, std::string& error, ArithmeticRecordMap& records);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticLogicalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error, ArithmeticRecordMap& records);
// Analysis and selectors remain available if original cuts cannot carry their
// executable predicates. Preparation is detached; no physical IDs are assigned.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const ArithmeticDemandAnalysis& analysis, std::string& error);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error);
// Same emission with composable family provenance and source matching tuples.
// Preparation preserves every original cut and leaves invocation draining to
// the parent. Enclosing repeat adapters prefix their own visit coordinates.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticRegionalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
