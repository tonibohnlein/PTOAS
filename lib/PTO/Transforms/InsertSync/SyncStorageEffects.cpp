// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Split physical ranges at endpoints, never by byte count or by SSA allocation.
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "SyncEffectRanges.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>

namespace mlir::pto {
LogicalResult SyncStorageEffects::build(const SyncInput& input)
{
    records.clear();
    partition.clear();
    phaseEffects.clear();
    SyncStorageEffects pending;
    pending.gmAliasPolicy = input.memory().gmPolicy();
    for (const auto* phase : input.instructions()) {
        if (!phase || !phase->elementOp) {
            return failure();
        }
        for (auto mode : {SyncAccessMode::Read, SyncAccessMode::Write}) {
            auto memories = mode == SyncAccessMode::Read ? ArrayRef(phase->useVec) : ArrayRef(phase->defVec);
            for (const auto* memory : memories) {
                if (!memory) {
                    return failure();
                }
                SyncStorageEffect effect;
                effect.phase = phase;
                effect.memory = memory;
                effect.mode = mode;
                detail::resolveEffectRanges(input, effect);
                pending.phaseEffects[phase].push_back(pending.records.size());
                pending.records.push_back(std::move(effect));
            }
        }
    }
    pending.partitionRanges();
    *this = std::move(pending);
    return success();
}

void SyncStorageEffects::partitionRanges()
{
    struct Event { std::size_t effect; bool start; };
    std::map<AddressSpace, std::map<uint64_t, SmallVector<Event>>> spaces;
    for (auto [id, effect] : llvm::enumerate(records)) {
        for (const auto& range : effect.ranges) {
            spaces[range.space][range.begin].push_back({id, true});
            spaces[range.space][range.end].push_back({id, false});
        }
    }
    for (const auto& [space, points] : spaces) {
        std::set<std::size_t> active;
        for (auto point = points.begin(); point != points.end(); ++point) {
            for (const auto& event : point->second) {
                if (event.start) {
                    active.insert(event.effect);
                } else {
                    active.erase(event.effect);
                }
            }
            auto next = std::next(point);
            if (next == points.end() || active.empty()) {
                continue;
            }
            const auto cell = partition.size();
            partition.push_back({space, point->first, next->first});
            for (auto effect : active) {
                records[effect].cells.push_back(cell);
            }
        }
    }
}

ArrayRef<std::size_t> SyncStorageEffects::effectsFor(const CompoundInstanceElement* phase) const
{
    auto found = phaseEffects.find(phase);
    return found == phaseEffects.end() ? ArrayRef<std::size_t>{} : found->second;
}

bool SyncStorageEffects::allAccessesExact() const
{
    return llvm::all_of(records, [](const auto& effect) { return effect.precision == SyncAccessPrecision::Exact; });
}

bool SyncStorageEffects::mayOverlap(std::size_t first, std::size_t second) const
{
    if (first >= records.size() || second >= records.size()) {
        return true;
    }
    const auto& a = records[first];
    const auto& b = records[second];
    if (a.memory->scope == AddressSpace::Zero || b.memory->scope == AddressSpace::Zero) {
        return true;
    }
    if (a.memory->scope != b.memory->scope) {
        return false;
    }
    if (a.region && b.region && regionsProvablyDisjoint(*a.region, *b.region)) {
        return false;
    }
    if ((a.region && a.region->empty()) || (b.region && b.region->empty())) {
        return false;
    }
    if (a.memory->scope == AddressSpace::GM) {
        // Only the declared root policy is shared. Footprint comparison belongs
        // to this storage analysis, not the legacy dependency analyzer.
        if (gmAliasPolicy == GMAliasPolicy::MayAlias || !a.descriptorRegion || !b.descriptorRegion ||
            !a.descriptorRegion->base || !b.descriptorRegion->base) {
            return true;
        }
        auto firstRoot = dyn_cast<BlockArgument>(a.descriptorRegion->base);
        auto secondRoot = dyn_cast<BlockArgument>(b.descriptorRegion->base);
        const bool independent = firstRoot && secondRoot &&
            isa<func::FuncOp>(firstRoot.getOwner()->getParentOp()) &&
            isa<func::FuncOp>(secondRoot.getOwner()->getParentOp());
        return !independent || firstRoot == secondRoot;
    }
    if (a.precision == SyncAccessPrecision::Unknown || b.precision == SyncAccessPrecision::Unknown) {
        return true;
    }
    if ((a.precision == SyncAccessPrecision::Exact && !a.exactRanges) ||
        (b.precision == SyncAccessPrecision::Exact && !b.exactRanges)) {
        return true;
    }
    std::size_t i = 0, j = 0;
    while (i < a.cells.size() && j < b.cells.size()) {
        if (a.cells[i] == b.cells[j]) {
            return true;
        }
        if (a.cells[i] < b.cells[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return false;
}
bool SyncStorageEffects::mayConflict(std::size_t first, std::size_t second) const
{
    if (first >= records.size() || second >= records.size()) {
        return true;
    }
    if (records[first].mode == SyncAccessMode::Read && records[second].mode == SyncAccessMode::Read) {
        return false;
    }
    return mayOverlap(first, second);
}
} // namespace mlir::pto
