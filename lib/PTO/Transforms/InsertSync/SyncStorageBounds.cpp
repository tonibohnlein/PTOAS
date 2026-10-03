// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include <limits>
#include <map>
namespace mlir::pto {
void SyncStorageBounds::clear()
{
    memoryRecords.clear();
    boundCells.clear();
    memoryIdentities.clear();
}
void SyncStorageBounds::capture(ArrayRef<const CompoundInstanceElement*> phases)
{
    clear();
    for (const auto* phase : phases) {
        auto observe = [&](const BaseMemInfo* memory) {
            auto [found, added] = memoryIdentities.try_emplace(memory, memoryRecords.size());
            if (added) {
                memoryRecords.push_back({found->second, static_cast<unsigned>(memory->scope),
                    memory->baseAddresses, memory->allocateSize, memory->hasKnownPhysicalAddresses,
                    memory->aliasesUnknownRange, SyncBoundStatus::UnknownCoordinates, {}});
            }
        };
        for (const auto* memory : phase->useVec) { observe(memory); }
        for (const auto* memory : phase->defVec) { observe(memory); }
    }
    normalizeBounds();
}
std::optional<std::size_t> SyncStorageBounds::memoryIdentity(const BaseMemInfo* memory) const
{
    auto found = memoryIdentities.find(memory);
    return found == memoryIdentities.end() ? std::nullopt : std::optional<std::size_t>(found->second);
}
void SyncStorageBounds::normalizeBounds()
{
    struct Event { std::size_t memory; bool begin; };
    std::map<unsigned, std::map<uint64_t, SmallVector<Event>>> events;
    for (auto& memory : memoryRecords) {
        // Multiple candidate addresses are alternative coordinates, not a
        // proved simultaneous footprint. Keep their original list unresolved.
        if (memory.addressSpace == static_cast<unsigned>(pto::AddressSpace::GM) ||
            memory.aliasesUnknownRange || !memory.knownPhysicalAddresses || !memory.boundingSize) { continue; }
        if (memory.boundingAddresses.size() != 1) {
            memory.bounds = SyncBoundStatus::AmbiguousAddresses; continue;
        }
        auto begin = memory.boundingAddresses.front();
        if (memory.boundingSize > std::numeric_limits<uint64_t>::max() - begin) {
            memory.bounds = SyncBoundStatus::WrappedInterval; continue;
        }
        memory.bounds = SyncBoundStatus::ConstantPhysical;
        auto& space = events[memory.addressSpace];
        space[begin].push_back({memory.id, true});
        space[begin + memory.boundingSize].push_back({memory.id, false});
    }
    // Endpoint sweep is O(N log N + emitted memory/cell incidences). Neither
    // physical byte counts nor numerical gaps determine representation size.
    for (const auto& [space, points] : events) {
        std::map<std::size_t, bool> active;
        for (auto point = points.begin(); point != points.end(); ++point) {
            for (const auto& event : point->second) {
                if (event.begin) { active.emplace(event.memory, true); }
                else { active.erase(event.memory); }
            }
            auto next = std::next(point);
            if (next == points.end() || active.empty()) { continue; }
            auto id = boundCells.size();
            boundCells.push_back({space, point->first, next->first});
            for (const auto& entry : active) { memoryRecords[entry.first].boundCells.push_back(id); }
        }
    }
}
} // namespace mlir::pto
