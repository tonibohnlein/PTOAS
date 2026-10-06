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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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
    independentEffects.clear();
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
    using Points = std::map<uint64_t, SmallVector<Event>>;
    // Preserve first-seen base order, rather than ordering SSA pointer addresses.
    std::map<AddressSpace, llvm::MapVector<Value, Points>> spaces;
    for (auto [id, effect] : llvm::enumerate(records)) {
        for (const auto& range : effect.ranges) {
            if (range.begin >= range.end) {
                continue;
            }
            spaces[range.space][range.base][range.begin].push_back({id, true});
            spaces[range.space][range.base][range.end].push_back({id, false});
        }
    }
    for (const auto& [space, bases] : spaces) {
        for (const auto& [base, points] : bases) {
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
                partition.push_back({space, point->first, next->first, base});
                for (auto effect : active) {
                    records[effect.first].cells.push_back(cell);
                }
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

bool SyncStorageEffects::allAccessesMaterialized() const
{
    return llvm::all_of(records, [](const auto& effect) { return effect.rangesMaterialized; });
}

bool storageBasesAreComparable(ArrayRef<SyncStorageCell> ranges, GMAliasPolicy policy)
{
    llvm::DenseSet<Value> bases;
    Operation* invocation = nullptr;
    bool absoluteGM = false;
    for (const auto& range : ranges) {
        if (range.begin > range.end) { return false; }
        if (range.begin == range.end) { continue; }
        if (range.space == AddressSpace::Zero) { return false; }
        if (range.space != AddressSpace::GM) {
            if (range.base) { return false; }
            continue;
        }
        if (!range.base) { absoluteGM = true; continue; }
        if (!isCanonicalGMBase(range.base)) { return false; }
        auto* owner = cast<BlockArgument>(range.base).getOwner()->getParentOp();
        if (invocation && invocation != owner) { return false; }
        invocation = owner;
        bases.insert(range.base);
    }
    return !(absoluteGM && !bases.empty()) &&
        (bases.size() <= 1 || policy == GMAliasPolicy::MayNotAlias);
}

bool SyncStorageEffects::hasMaterializedCellPartition(ArrayRef<std::size_t> effectIds) const
{
    SmallVector<SyncStorageCell> ranges;
    for (auto id : effectIds) {
        if (id >= records.size()) { return false; }
        const auto& effect = records[id];
        if (!effect.memory || !effect.rangesMaterialized) {
            return false;
        }
        for (const auto& range : effect.ranges) {
            if (range.space != effect.memory->scope ||
                (range.base && (!effect.sharedProvenanceComplete || range.base != effect.memory->rootBuffer))) {
                return false;
            }
            ranges.push_back(range);
        }
    }
    return storageBasesAreComparable(ranges, memory.gmPolicy());
}

bool SyncStorageEffects::mayOverlap(std::size_t first, std::size_t second) const
{
    if (first >= records.size() || second >= records.size()) {
        return true;
    }
    const auto& a = records[first];
    const auto& b = records[second];
    if ((a.rangesMaterialized && a.ranges.empty()) || (b.rangesMaterialized && b.ranges.empty())) {
        return false;
    }
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
    // Symbolic access maps retain pointer provenance even when their offsets
    // depend on an IV. Apply the existing distinct-argument GM policy to those
    // resolved bases; carried/select-produced pointers are not canonical bases.
    if (a.memory->scope == AddressSpace::GM && !a.regions.empty() && !b.regions.empty() &&
        llvm::all_of(a.regions, [&](const auto& left) {
            return llvm::all_of(b.regions, [&](const auto& right) {
                if (!left.base || !right.base || left.base == right.base) { return false; }
                SmallVector<SyncStorageCell> domains{{AddressSpace::GM, 0, 1, left.base},
                                                    {AddressSpace::GM, 0, 1, right.base}};
                return storageBasesAreComparable(domains, memory.gmPolicy());
            });
        })) {
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
    if (!a.rangesMaterialized || !b.rangesMaterialized) {
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
bool SyncStorageEffects::needsOverlapQueries(const CompoundInstanceElement* phase, bool unresolvedBases) const
{
    return llvm::any_of(effectsFor(phase), [&](std::size_t id) {
        const auto& effect = records[id];
        return !effect.rangesMaterialized || effect.memory->scope == AddressSpace::Zero ||
            (unresolvedBases && effect.memory->scope == AddressSpace::GM);
    });
}
bool SyncStorageEffects::residualConflict(std::size_t first, std::size_t second) const
{
    if (first >= records.size() || second >= records.size()) { return true; }
    const auto& a = records[first];
    const auto& b = records[second];
    if ((a.mode == SyncAccessMode::Read && b.mode == SyncAccessMode::Read) ||
        (a.memory->scope != AddressSpace::Zero && b.memory->scope != AddressSpace::Zero &&
         a.memory->scope != b.memory->scope)) { return false; }
    if (a.rangesMaterialized && b.rangesMaterialized) {
        SmallVector<SyncStorageCell> domains(a.ranges);
        llvm::append_range(domains, b.ranges);
        if (storageBasesAreComparable(domains, memory.gmPolicy())) { return false; }
    }
    return mayConflict(first, second);
}
bool SyncStorageEffects::hasUniformRelationships(ArrayRef<const CompoundInstanceElement*> phases) const
{
    SmallVector<SyncStorageCell> domains;
    for (const auto* phase : phases) {
        for (auto id : effectsFor(phase)) {
            const auto& effect = records[id];
            if (!effect.rangesMaterialized && effect.regions.empty()) { return true; }
            llvm::append_range(domains, effect.ranges);
            for (const auto& region : effect.regions) {
                if (!region.empty()) { domains.push_back({effect.memory->scope, 0, 1, region.base}); }
            }
        }
    }
    return !storageBasesAreComparable(domains, memory.gmPolicy());
}
bool SyncStorageEffects::uniformConflict(std::size_t first, std::size_t second) const
{
    if (first >= records.size() || second >= records.size()) { return true; }
    const auto& a = records[first];
    const auto& b = records[second];
    if ((a.mode == SyncAccessMode::Read && b.mode == SyncAccessMode::Read) ||
        (a.memory->scope != AddressSpace::Zero && b.memory->scope != AddressSpace::Zero &&
         a.memory->scope != b.memory->scope)) { return false; }
    if ((!a.rangesMaterialized && a.regions.empty()) ||
        (!b.rangesMaterialized && b.regions.empty())) { return mayConflict(first, second); }
    SmallVector<SyncStorageCell> domains;
    for (const auto* effect : {&a, &b}) {
        llvm::append_range(domains, effect->ranges);
        for (const auto& region : effect->regions) {
            if (!region.empty()) { domains.push_back({effect->memory->scope, 0, 1, region.base}); }
        }
    }
    return !storageBasesAreComparable(domains, memory.gmPolicy()) && mayConflict(first, second);
}
bool SyncStorageEffects::independentOfOtherPhases(std::size_t effect) const
{
    if (effect >= records.size() || !records[effect].memory ||
        records[effect].memory->scope != AddressSpace::GM) {
        return false;
    }
    auto cached = independentEffects.find(effect);
    if (cached != independentEffects.end()) {
        return cached->second;
    }
    bool independent = true;
    for (std::size_t other = 0; other < records.size(); ++other) {
        if (records[other].phase != records[effect].phase && mayConflict(effect, other)) {
            independent = false;
            break;
        }
    }
    independentEffects.try_emplace(effect, independent);
    return independent;
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
