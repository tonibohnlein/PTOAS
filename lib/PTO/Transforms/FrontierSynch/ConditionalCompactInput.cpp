// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ConditionalCompactInput.h"
#include "CompactWriterReaderInputInternal.h"
namespace mlir::pto::frontiersynch {
ConditionalCompactInput buildConditionalCompactInput(scf::ForOp loop,
    const SyncInput& input, const PhaseIndex& index, const CompactWriterReaderBindings& bindings)
{
    ConditionalCompactInput result;
    result.body = std::make_shared<const BalancedCompactBody>(recognizeBalancedCompactBody(loop, input, index));
    if (!result.body->error.empty()) { result.error = result.body->error; return result; }
    std::vector<std::vector<const CompoundInstanceElement*>> slots;
    for (const auto& slot : result.body->slots) {
        std::vector<const CompoundInstanceElement*> alternatives;
        for (const auto& alternative : slot.alternatives) { alternatives.push_back(alternative.phase); }
        slots.push_back(std::move(alternatives));
    }
    result.storage = detail::buildCompactWriterReaderSlotInput(loop, input, index, slots, bindings);
    if (!result.storage.error.empty()) { result.error = result.storage.error; return result; }
    result.mathematical = analyzeCompactWriterReader(result.storage.payloads, result.storage.accesses,
        result.storage.queries, result.storage.additional, result.storage.native);
    result.error = result.mathematical.error;
    return result;
}
} // namespace mlir::pto::frontiersynch
