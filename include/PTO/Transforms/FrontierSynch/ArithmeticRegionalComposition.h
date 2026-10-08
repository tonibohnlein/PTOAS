// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_ARITHMETICREGIONALCOMPOSITION_H
#define PTO_FRONTIERSYNCH_ARITHMETICREGIONALCOMPOSITION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
namespace mlir::pto::frontiersynch {
// Binary sequence composition of compatible, exact symbolic child interfaces.
// All relation operations are exact; costs depend on generated relation pieces,
// not enumerated byte addresses or execution counts. Original IR is borrowed.
FailureOr<ArithmeticRegionalRelations> composeArithmeticRegionalRelations(
    const ArithmeticRegionalRelations& left, const ArithmeticRegionalRelations& right,
    ArithmeticRegionContext context, std::string& error);
}
#endif
