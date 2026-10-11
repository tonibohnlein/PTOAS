// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned relative storage identities for a certified whole numerical word.
#ifndef PTO_FRONTIERSYNCH_NUMERICTEMPLATERECURRINGGLOBAL_H
#define PTO_FRONTIERSYNCH_NUMERICTEMPLATERECURRINGGLOBAL_H
#include "AlignedModularMap.h"
#include "NumericTemplateInternal.h"
namespace mlir::pto::frontiersynch::detail {
struct RecurringGlobalEffect {
    std::size_t payload;
    std::size_t effect;
    std::size_t sourceEffect;
    const CompoundInstanceElement* phase;
    SmallVector<TemplateCoordinate> coordinates;
    SmallVector<SyncAccessRegion> originalMaps;
    std::vector<uint32_t> atoms;
};
struct RecurringGlobalTemplate {
    const SyncInput* input = nullptr;
    const PhaseIndex* index = nullptr;
    scf::ForOp outer;
    std::shared_ptr<const SyncInput> inputOwner;
    std::shared_ptr<const PhaseIndex> indexOwner;
    std::vector<AlignedModularMapGroup> groups;
    std::vector<RecurringGlobalEffect> effects;
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> effectIndex;
    uint64_t atomCount = 0;
    std::size_t payloadCount = 0;
    int64_t lower = 0, step = 0;
    SmallVector<TemplatePayload, 0> signature;
    bool applies(const NumericTemplate& form) const;
    const RecurringGlobalEffect* find(std::size_t payload, std::size_t effect) const;
};
bool prepareRecurringGlobalBase(TemplateBuilder& builder,
    ArrayRef<std::pair<std::size_t, std::size_t>> effects);
bool sealRecurringGlobal(NumericTemplate& form);
bool retainRecurringGlobalOwners(NumericTemplate& form, std::shared_ptr<const SyncInput> input,
    std::shared_ptr<const PhaseIndex> index);
} // namespace mlir::pto::frontiersynch::detail
#endif
