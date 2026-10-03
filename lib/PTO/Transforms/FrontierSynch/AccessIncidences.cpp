// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "AccessIncidences.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
namespace mlir::pto::frontiersynch::detail {
FailureOr<Incidences> indexAccesses(
    ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<StorageFootprint> footprints)
{
    DenseMap<const CompoundInstanceElement*, std::size_t> sites;
    for (auto [id, phase] : llvm::enumerate(sequence)) {
        const bool valid = phase && sites.try_emplace(phase, id).second;
        if (!valid) {
            return failure();
        }
    }
    Incidences result(sequence.size());
    DenseMap<std::pair<std::size_t, std::size_t>, std::size_t> slots;
    for (auto [id, footprint] : llvm::enumerate(footprints)) {
        for (const auto& access : footprint.accesses) {
            const bool valid = access.phase && (access.read || access.write);
            if (!valid) {
                return failure();
            }
            auto site = sites.find(access.phase);
            if (site == sites.end()) {
                continue;
            }
            auto& entries = result[site->second];
            auto [slot, added] = slots.try_emplace(std::make_pair(site->second, id), entries.size());
            if (added) {
                entries.push_back({id, access.read, access.write});
            } else {
                auto& entry = entries[slot->second];
                entry.read |= access.read;
                entry.write |= access.write;
            }
        }
    }
    return result;
}

FailureOr<Neighbors> indexAliases(std::size_t count, ArrayRef<StorageAlias> aliases)
{
    Neighbors neighbors(count);
    for (std::size_t id = 0; id < count; ++id) {
        neighbors[id].push_back(id);
    }
    for (const auto& alias : aliases) {
        const bool valid = alias.first < alias.second && alias.second < count;
        if (!valid) {
            return failure();
        }
        neighbors[alias.first].push_back(alias.second);
        neighbors[alias.second].push_back(alias.first);
    }
    for (auto& entries : neighbors) {
        llvm::sort(entries);
        entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
    }
    return neighbors;
}

} // namespace mlir::pto::frontiersynch::detail
