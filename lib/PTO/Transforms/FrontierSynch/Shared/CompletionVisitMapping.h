// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Convert an already decoded visit map into the completion palette frame.
// Applicability and matching stay with the original recipe. Invalid rules and
// noncontiguous nonconstant tables remain unavailable; records are never erased.
#ifndef PTO_COMPLETION_VISIT_MAPPING_H
#define PTO_COMPLETION_VISIT_MAPPING_H
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
namespace mlir::pto::frontiersynch {
std::optional<PhysicalRecordAllocation> completionVisitMapping(
    PhysicalRecordAllocation record, uint64_t originalArity, uint64_t enclosingArity);
}
#endif
