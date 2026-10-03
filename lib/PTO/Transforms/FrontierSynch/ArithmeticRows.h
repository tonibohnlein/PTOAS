// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICROWS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICROWS_H
#include "PTO/Transforms/FrontierSynch/ArithmeticRecognition.h"
namespace mlir::pto::frontiersynch::detail {
std::optional<ArithmeticIssue> collectRow(AffineExpr expression, unsigned dimensions, unsigned symbols, LinearRow& row);
bool normalizeRow(LinearRow& row);
uint64_t magnitude(int64_t value);
ArithmeticClass classifyRow(const LinearRow& row);
} // namespace mlir::pto::frontiersynch::detail
#endif
