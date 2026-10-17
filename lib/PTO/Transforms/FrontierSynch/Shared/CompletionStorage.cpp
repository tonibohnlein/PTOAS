// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "CompletionStorage.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Interval = std::pair<uint64_t, uint64_t>;
bool knownSpace(const BaseMemInfo* memory)
{
    return memory && memory->scope != AddressSpace::Zero && symbolizeAddressSpace(
        static_cast<uint32_t>(memory->scope)).has_value();
}
bool knownReservation(const BaseMemInfo* memory)
{
    // GM offsets are not absolute reservations in this input contract.
    return knownSpace(memory) && memory->scope != AddressSpace::GM && memory->hasKnownPhysicalAddresses &&
        !memory->aliasesUnknownRange && memory->allocateSize && !memory->baseAddresses.empty() &&
        llvm::all_of(memory->baseAddresses, [&](uint64_t base) {
            return base <= std::numeric_limits<uint64_t>::max() - memory->allocateSize;
        });
}
struct Space {
    std::vector<Interval> intervals;
    std::size_t offset = 0, count = 1;
    void normalize()
    {
        llvm::sort(intervals);
        std::size_t used = 0;
        for (auto interval : intervals) {
            if (used && interval.first <= intervals[used - 1].second) {
                intervals[used - 1].second = std::max(intervals[used - 1].second, interval.second);
            } else { intervals[used++] = interval; }
        }
        intervals.resize(used);
        count = std::max<std::size_t>(1, used);
    }
};
uint8_t combined(const CompletionFacts& facts)
{
    uint8_t modes = 0;
    for (uint8_t fact : facts) { modes |= fact; }
    return modes;
}
} // namespace
CompletionStorage::CompletionStorage(ArrayRef<const CompoundInstanceElement*> phases)
{
    DenseSet<const BaseMemInfo*> memories;
    for (const auto* phase : phases) {
        memories.insert(phase->useVec.begin(), phase->useVec.end());
        memories.insert(phase->defVec.begin(), phase->defVec.end());
    }
    std::map<AddressSpace, Space> spaces;
    for (const auto* memory : memories) {
        if (!knownSpace(memory)) { continue; }
        auto& space = spaces[memory->scope];
        if (knownReservation(memory)) {
            for (uint64_t base : memory->baseAddresses) {
                space.intervals.emplace_back(base, base + memory->allocateSize);
            }
        }
    }
    classCount = 0;
    for (auto& entry : spaces) {
        auto& space = entry.second;
        space.normalize();
        space.offset = classCount;
        classCount += space.count;
    }
    // Unknown-only and call-only functions must have a nonempty fact universe.
    classCount = std::max<std::size_t>(1, classCount);
    for (const auto* memory : memories) {
        auto& membership = memberships[memory];
        auto& ids = membership.ids;
        if (!knownSpace(memory)) {
            membership.count = classCount;
            continue;
        }
        const auto& space = spaces.at(memory->scope);
        if (!knownReservation(memory)) {
            membership.first = space.offset;
            membership.count = space.count;
            continue;
        }
        for (uint64_t base : memory->baseAddresses) {
            auto found = std::upper_bound(space.intervals.begin(), space.intervals.end(), base,
                [](uint64_t address, const Interval& interval) { return address < interval.first; });
            // Every interval was inserted above; upper_bound is strictly after
            // its containing merged reservation, including at address zero.
            ids.push_back(space.offset + std::size_t(std::prev(found) - space.intervals.begin()));
        }
        llvm::sort(ids);
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }
}
CompletionFacts CompletionStorage::effects(ArrayRef<const CompoundInstanceElement*> phases) const
{
    CompletionFacts result(classCount, 0);
    auto add = [&](const BaseMemInfo* memory, uint8_t mode) {
        auto found = memberships.find(memory);
        if (found == memberships.end()) {
            for (auto& fact : result) { fact |= mode; }
        } else {
            const auto& membership = found->second;
            for (std::size_t i = 0; i < membership.count; ++i) { result[membership.first + i] |= mode; }
            for (auto id : membership.ids) { result[id] |= mode; }
        }
    };
    for (const auto* phase : phases) {
        for (const auto* memory : phase->useVec) { add(memory, 1); }
        for (const auto* memory : phase->defVec) { add(memory, 2); }
    }
    return result;
}
CompletionSummary CompletionStorage::summarize(ArrayRef<Operation*> roots, const PhaseIndex& index) const
{
    CompletionFacts all(classCount, 0);
    for (auto* root : roots) {
        root->walk([&](Operation* op) {
            auto facts = effects(index.phasesFor(op));
            for (std::size_t i = 0; i < classCount; ++i) { all[i] |= facts[i]; }
        });
    }
    return {all, all, all, true};
}
CompletionSummary CompletionStorage::liftCoarse(const CompletionSummary& summary) const
{
    return {CompletionFacts(classCount, combined(summary.all)),
            CompletionFacts(classCount, combined(summary.exposed)),
            CompletionFacts(classCount, combined(summary.suffix)), summary.transparent};
}
CompletionSummary collapseCompletionSummary(const CompletionSummary& summary)
{
    return {{combined(summary.all)}, {combined(summary.exposed)}, {combined(summary.suffix)}, summary.transparent};
}
} // namespace mlir::pto::frontiersynch
