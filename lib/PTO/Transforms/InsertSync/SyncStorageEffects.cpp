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

namespace mlir::pto {
namespace {
bool hasCompleteSharedProvenance(Value value, DenseMap<Value, bool>& cache)
{
    if (!value) {
        return false;
    }
    auto found = cache.find(value);
    if (found != cache.end()) {
        return found->second;
    }
    cache[value] = false;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
        return cache[value] = function && argument.getOwner() == &function.getBody().front();
    }
    auto* operation = value.getDefiningOp();
    if (!operation || operation->getNumRegions() != 0) {
        return false;
    }
    // Trace the descriptor's SSA dependencies, not just its cached initial root.
    // A region argument or result needs a separate all-iterations alias proof.
    return cache[value] = llvm::all_of(operation->getOperands(), [&](Value operand) {
        return hasCompleteSharedProvenance(operand, cache);
    });
}
} // namespace

LogicalResult SyncStorageEffects::build(const SyncInput& input)
{
    records.clear();
    partition.clear();
    phaseEffects.clear();
    SyncStorageEffects pending;
    pending.memory = input.memory();
    DenseMap<Value, bool> provenance;
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
                effect.sharedProvenanceComplete = hasCompleteSharedProvenance(memory->baseBuffer, provenance);
                effect.mode = mode;
                detail::resolveEffectRanges(input, effect);
                pending.phaseEffects[phase->GetIndex()].push_back(pending.records.size());
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
            if (range.begin >= range.end) {
                continue;
            }
            spaces[range.space][range.begin].push_back({id, true});
            spaces[range.space][range.end].push_back({id, false});
        }
    }
    for (const auto& [space, points] : spaces) {
        std::map<std::size_t, unsigned> active;
        for (auto point = points.begin(); point != points.end(); ++point) {
            for (const auto& event : point->second) {
                if (event.start) {
                    ++active[event.effect];
                } else {
                    if (--active[event.effect] == 0) {
                        active.erase(event.effect);
                    }
                }
            }
            auto next = std::next(point);
            if (next == points.end() || active.empty()) {
                continue;
            }
            const auto cell = partition.size();
            partition.push_back({space, point->first, next->first});
            for (auto effect : active) {
                records[effect.first].cells.push_back(cell);
            }
        }
    }
}

ArrayRef<std::size_t> SyncStorageEffects::effectsFor(const CompoundInstanceElement* phase) const
{
    if (!phase) {
        return {};
    }
    auto found = phaseEffects.find(phase->GetIndex());
    if (found == phaseEffects.end() || records[found->second.front()].phase->elementOp != phase->elementOp) {
        return {};
    }
    return found->second;
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
    // Buffer disjointness needs no full-write declaration. Reuse the shared
    // address/range analysis, including views, slot alternatives and GM policy.
    // Its historical distinct-root rule for unplanned local buffers is not a
    // physical disjointness proof, so use local ranges only after planning.
    // A supplied selection describes its own bytes and may use native offset
    // conversion; do not substitute an allocation bound for that access set.
    const bool comparable = a.memory->scope == AddressSpace::GM ||
        (a.memory->hasKnownPhysicalAddresses && b.memory->hasKnownPhysicalAddresses);
    if (comparable && a.regions.empty() && b.regions.empty() &&
        a.sharedProvenanceComplete && b.sharedProvenanceComplete &&
        !memory.MemAlias(a.memory, b.memory)) {
        return false;
    }
    if (!a.regions.empty() && !b.regions.empty() &&
        llvm::all_of(a.regions, [&](const auto& left) {
            return llvm::all_of(b.regions, [&](const auto& right) { return regionsProvablyDisjoint(left, right); });
        })) {
        return false;
    }
    auto empty = [](const auto& access) {
        return !access.regions.empty() &&
            llvm::all_of(access.regions, [](const auto& region) { return region.empty(); });
    };
    if (empty(a) || empty(b)) {
        return false;
    }
    // Distinct GM roots follow the user's alias policy even with exact maps.
    // Never apply the legacy same-root range test to an instruction selection.
    if (a.memory->scope == AddressSpace::GM && a.memory->rootBuffer != b.memory->rootBuffer &&
        a.sharedProvenanceComplete && b.sharedProvenanceComplete && !memory.MemAlias(a.memory, b.memory)) {
        return false;
    }
    if (a.memory->scope == AddressSpace::GM) {
        return true;
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
bool SyncStorageEffects::dependencies(const CompoundInstanceElement* first, SyncAccessMode firstMode,
                                      const CompoundInstanceElement* second, SyncAccessMode secondMode,
                                      DepBaseMemInfoPairVec& result, Value secondOperand) const
{
    bool found = false;
    for (auto left : effectsFor(first)) {
        const auto& a = records[left];
        if (a.mode != firstMode) {
            continue;
        }
        for (auto right : effectsFor(second)) {
            const auto& b = records[right];
            if (b.mode != secondMode || (secondOperand && b.memory->baseBuffer != secondOperand)) {
                continue;
            }
            if (mayOverlap(left, right)) {
                auto pair = std::make_pair(a.memory, b.memory);
                if (!llvm::is_contained(result, pair)) {
                    result.push_back(pair);
                }
                found = true;
            }
        }
    }
    return found;
}
} // namespace mlir::pto
