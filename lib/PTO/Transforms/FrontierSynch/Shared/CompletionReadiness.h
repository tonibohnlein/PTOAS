// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Qualified forward scalar prerequisites, separate from storage may-effects.
#ifndef PTO_COMPLETION_READINESS_H
#define PTO_COMPLETION_READINESS_H
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
namespace mlir::pto::frontiersynch {
class CompletionReadiness {
public:
    // Protect the same-block first direct scalar use before it can observe
    // an unfinished result. Indirect branch prerequisites must be dominated
    // by that cut; carried and escaping readiness remains unsupported.
    static FailureOr<CompletionReadiness> qualify(ArrayRef<Operation*> roots, const PhaseIndex& index);
    bool protects(Operation* op) const { return cuts.contains(op); }
    std::size_t size() const { return cuts.size(); }
    uint64_t operationVisits() const { return inspectedOperations; }
    uint64_t useVisits() const { return inspectedUses; }
private:
    DenseSet<Operation*> cuts;
    uint64_t inspectedOperations = 0, inspectedUses = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
