// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared bounded integer codec and periodic extension of the rotating driver.
#ifndef PTO_FRONTIER_PERIODIC_DRIVER_H
#define PTO_FRONTIER_PERIODIC_DRIVER_H
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "llvm/Support/JSON.h"
#include <memory>

namespace frontier_test {
struct Input {
    llvm::SmallVector<std::unique_ptr<mlir::pto::CompoundInstanceElement>> owned;
    llvm::SmallVector<const mlir::pto::CompoundInstanceElement*> sites;
    llvm::SmallVector<mlir::pto::frontiersynch::RotatingFamily> families;
    llvm::SmallVector<mlir::pto::frontiersynch::RotatingFragment> fragments;
};
mlir::LogicalResult readFamilies(const llvm::json::Array& rows, Input& input);
mlir::LogicalResult readFragments(const llvm::json::Array& rows, Input& input);
mlir::FailureOr<llvm::json::Object> analyzeSymbolic(
    const llvm::json::Object& object, llvm::ArrayRef<const mlir::pto::CompoundInstanceElement*> anchors,
    mlir::MLIRContext& context);

mlir::FailureOr<llvm::DynamicAPInt> integer(const llvm::json::Value& value);
mlir::FailureOr<std::size_t> index(const llvm::json::Value& value, std::size_t bound);
std::string decimal(const llvm::DynamicAPInt& value);
mlir::FailureOr<llvm::json::Object> analyzePeriodic(
    const llvm::json::Object& object, llvm::ArrayRef<const mlir::pto::CompoundInstanceElement*> sites,
    llvm::ArrayRef<mlir::pto::frontiersynch::RotatingFamily> families,
    llvm::ArrayRef<mlir::pto::frontiersynch::RotatingFragment> fragments, mlir::MLIRContext& context);
} // namespace frontier_test
#endif
