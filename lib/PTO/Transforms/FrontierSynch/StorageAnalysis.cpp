// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/StorageAnalysis.h"

namespace mlir::pto::frontiersynch {
StorageAnalysis::StorageAnalysis(const SyncInput& input) : StorageAnalysis(input, input.instructions()) {}
StorageAnalysis::StorageAnalysis(const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases)
    : storageBoundsInfo(&input.storageBounds())
{
    DenseMap<const BaseMemInfo*, std::size_t> ids;
    auto observe = [&](const CompoundInstanceElement* phase, const BaseMemInfo* memory, bool write) {
        auto [found, added] = ids.try_emplace(memory, footprintFacts.size());
        if (added) {
            footprintFacts.push_back({memory, {}, storageBoundsInfo->memoryIdentity(memory)});
        }
        auto& accesses = footprintFacts[found->second].accesses;
        auto access = llvm::find_if(accesses, [&](const StorageAccess& entry) { return entry.phase == phase; });
        if (access == accesses.end()) {
            accesses.push_back({phase, !write, write});
        } else {
            access->read |= !write;
            access->write |= write;
        }
    };
    for (const auto* phase : phases) {
        for (const auto* memory : phase->useVec) {
            observe(phase, memory, false);
        }
        for (const auto* memory : phase->defVec) {
            observe(phase, memory, true);
        }
    }
    // Query original records for every pair, including coordinate-qualified ones.
    // Separate witnesses preserve nontransitive may-alias information. No private
    // cell partition, address specialization, origin closure or overwrite kill is applied.
    for (std::size_t first = 0; first < footprintFacts.size(); ++first) {
        for (std::size_t second = first + 1; second < footprintFacts.size(); ++second) {
            ++queryCount;
            if (input.memory().MemAlias(footprintFacts[first].memory, footprintFacts[second].memory)) {
                aliasFacts.push_back({first, second});
            }
        }
    }
}
} // namespace mlir::pto::frontiersynch
