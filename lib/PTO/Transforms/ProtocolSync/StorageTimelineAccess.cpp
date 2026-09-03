// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTimelineAccess.cpp - Group exact generation access sets ---===//

#include "StorageTimelineInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace mlir::pto::protocol_sync::detail;

namespace {

struct IntervalRecord {
    SyncStorageFamilyId family = kInvalidSyncId;
    unsigned addressSpace = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    unsigned accessClass = 0;
};

struct ClassMaximum {
    std::uint64_t end = 0;
    unsigned accessClass = 0;
};

void updateClassMaxima(
    std::optional<ClassMaximum>& first, std::optional<ClassMaximum>& second, const IntervalRecord& current)
{
    if (first && first->accessClass == current.accessClass) {
        first->end = std::max(first->end, current.end);
        return;
    }
    if (second && second->accessClass == current.accessClass) {
        second->end = std::max(second->end, current.end);
        if (second->end > first->end) {
            std::swap(first, second);
        }
        return;
    }
    ClassMaximum candidate{current.end, current.accessClass};
    if (!first || candidate.end > first->end) {
        second = first;
        first = candidate;
    } else if (!second || candidate.end > second->end) {
        second = candidate;
    }
}

std::uint64_t hashAccessClass(const SyncAccess& access)
{
    llvm::hash_code hash = llvm::hash_combine(access.family, access.storage.intervals.size());
    for (const SyncByteInterval& interval : access.storage.intervals) {
        hash = llvm::hash_combine(hash, interval.begin, interval.size);
    }
    if (!access.slot) {
        return static_cast<std::uint64_t>(static_cast<std::size_t>(llvm::hash_combine(hash, 0U)));
    }
    hash = llvm::hash_combine(
        hash, 1U, static_cast<unsigned>(access.slot->kind), access.slot->depth, access.slot->modulus);
    if (access.slot->kind == SyncSlotExpressionKind::Unknown) {
        hash = llvm::hash_combine(hash, access.slot->selector);
    } else {
        hash = llvm::hash_combine(hash, access.slot->loop, access.slot->coefficient, access.slot->offset);
    }
    return static_cast<std::uint64_t>(static_cast<std::size_t>(hash));
}

SmallVector<StorageAccessClass, 16> buildAccessClasses(const StructuredSyncIR& schedule)
{
    SmallVector<StorageAccessClass, 16> classes;
    llvm::DenseMap<std::uint64_t, SmallVector<unsigned, 1>> buckets;
    for (const SyncAccess& access : schedule.getAccesses()) {
        const SyncStorageFamily* family = schedule.findStorageFamily(access.family);
        if (!family || family->role != SyncStorageRole::LocalBuffer) {
            continue;
        }
        const std::uint64_t hash = hashAccessClass(access);
        unsigned classId = std::numeric_limits<unsigned>::max();
        for (unsigned candidate : buckets[hash]) {
            const StorageAccessClass& current = classes[candidate];
            const bool sameClass = current.family == access.family &&
                                   sameTimelineIntervals(current.slice, access.storage.intervals) &&
                                   sameTimelineSlotClass(current.slot, access.slot);
            if (sameClass) {
                classId = candidate;
                break;
            }
        }
        if (classId == std::numeric_limits<unsigned>::max()) {
            classId = classes.size();
            StorageAccessClass current;
            current.family = access.family;
            current.addressSpace = static_cast<unsigned>(family->space);
            current.slice = access.storage.intervals;
            current.slot = access.slot;
            classes.push_back(std::move(current));
            buckets[hash].push_back(classId);
        }
        StorageAccessClass& current = classes[classId];
        current.accesses.push_back(access.id);
        current.unknownRange = current.unknownRange || access.storage.unknownRange;
        current.aliasesUnknownRange = current.aliasesUnknownRange || access.storage.aliasesUnknownRange;
    }
    return classes;
}

StorageAccessClass makeEmptyGeneration(const StorageAccessClass& accessClass)
{
    StorageAccessClass generation = accessClass;
    generation.accesses.clear();
    generation.inPlace = false;
    return generation;
}

void appendGeneration(SmallVectorImpl<StorageAccessClass>& generations, StorageAccessClass& generation)
{
    if (!generation.accesses.empty()) {
        generations.push_back(std::move(generation));
    }
}

SmallVector<StorageAccessClass, 16> splitGenerationClasses(
    const StructuredSyncIR& schedule, ArrayRef<StorageAccessClass> accessClasses)
{
    SmallVector<StorageAccessClass, 16> generations;
    for (const StorageAccessClass& accessClass : accessClasses) {
        const std::size_t firstGeneration = generations.size();
        StorageAccessClass generation = makeEmptyGeneration(accessClass);
        for (auto groupBegin = accessClass.accesses.begin(); groupBegin != accessClass.accesses.end();) {
            const SyncAccess* first = schedule.findAccess(*groupBegin);
            if (!first) {
                generation.accesses.push_back(*groupBegin);
                ++groupBegin;
                continue;
            }
            auto groupEnd = std::find_if(groupBegin, accessClass.accesses.end(), [&](SyncAccessId id) {
                const SyncAccess* access = schedule.findAccess(id);
                return !access || access->phase != first->phase;
            });
            bool reads = false;
            bool writes = false;
            for (SyncAccessId id : llvm::make_range(groupBegin, groupEnd)) {
                const SyncAccess* access = schedule.findAccess(id);
                reads = reads ||
                        (access && (access->mode == SyncAccessMode::Read || access->mode == SyncAccessMode::ReadWrite));
                writes =
                    writes ||
                    (access && (access->mode == SyncAccessMode::Write || access->mode == SyncAccessMode::ReadWrite));
            }
            if (writes && !reads) {
                appendGeneration(generations, generation);
                generation = makeEmptyGeneration(accessClass);
            }
            generation.accesses.append(groupBegin, groupEnd);
            generation.inPlace = generation.inPlace || (writes && reads);
            groupBegin = groupEnd;
        }
        appendGeneration(generations, generation);
        const bool hasMultipleGenerations = generations.size() - firstGeneration > 1;
        for (StorageAccessClass& generated :
             MutableArrayRef<StorageAccessClass>(generations).drop_front(firstGeneration)) {
            generated.multipleGenerations = hasMultipleGenerations;
        }
    }
    return generations;
}

void markUnknownAliasConflicts(SmallVectorImpl<StorageAccessClass>& classes)
{
    llvm::SmallDenseSet<unsigned, 4> poisonedAddressSpaces;
    llvm::SmallDenseSet<SyncStorageFamilyId, 4> poisonedFamilies;
    for (const StorageAccessClass& accessClass : classes) {
        if (accessClass.aliasesUnknownRange) {
            poisonedAddressSpaces.insert(accessClass.addressSpace);
        }
        if (accessClass.unknownRange) {
            poisonedFamilies.insert(accessClass.family);
        }
    }
    for (StorageAccessClass& accessClass : classes) {
        const bool isPoisoned =
            poisonedAddressSpaces.contains(accessClass.addressSpace) || poisonedFamilies.contains(accessClass.family);
        if (isPoisoned) {
            accessClass.conflictingPhysicalRange = true;
        }
    }
}

void markPartialOverlaps(SmallVectorImpl<StorageAccessClass>& classes, ProtocolSyncStatistics* statistics)
{
    SmallVector<IntervalRecord, 32> intervals;
    for (auto [classId, accessClass] : llvm::enumerate(classes)) {
        for (const SyncByteInterval& interval : accessClass.slice) {
            const bool invalid =
                interval.size == 0 || interval.begin > std::numeric_limits<std::uint64_t>::max() - interval.size;
            if (invalid) {
                accessClass.unknownRange = true;
                continue;
            }
            intervals.push_back(
                {accessClass.family, accessClass.addressSpace, interval.begin, interval.begin + interval.size,
                 static_cast<unsigned>(classId)});
        }
    }
    llvm::sort(intervals, [](const IntervalRecord& left, const IntervalRecord& right) {
        return std::tie(left.addressSpace, left.begin, left.end, left.family, left.accessClass) <
               std::tie(right.addressSpace, right.begin, right.end, right.family, right.accessClass);
    });
    std::optional<ClassMaximum> first;
    std::optional<ClassMaximum> second;
    unsigned activeAddressSpace = std::numeric_limits<unsigned>::max();
    for (const IntervalRecord& current : intervals) {
        if (current.addressSpace != activeAddressSpace) {
            first.reset();
            second.reset();
            activeAddressSpace = current.addressSpace;
        }
        const ClassMaximum* overlapping = first && first->accessClass != current.accessClass ? &*first : nullptr;
        if (!overlapping && second && second->accessClass != current.accessClass) {
            overlapping = &*second;
        }
        if (statistics && overlapping) {
            ++statistics->intervalIndexQueries;
        }
        if (overlapping && overlapping->end > current.begin) {
            const bool crossesFamilies =
                classes[overlapping->accessClass].family != classes[current.accessClass].family;
            if (crossesFamilies) {
                classes[overlapping->accessClass].conflictingPhysicalRange = true;
                classes[current.accessClass].conflictingPhysicalRange = true;
            } else {
                classes[overlapping->accessClass].partialOverlap = true;
                classes[current.accessClass].partialOverlap = true;
            }
        }
        updateClassMaxima(first, second, current);
    }
}

} // namespace

bool mlir::pto::protocol_sync::detail::sameTimelineIntervals(
    ArrayRef<SyncByteInterval> first, ArrayRef<SyncByteInterval> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& left, const auto& right) {
               return left.begin == right.begin && left.size == right.size;
           });
}

bool mlir::pto::protocol_sync::detail::sameTimelineSlotClass(
    const std::optional<SyncSlotExpression>& first, const std::optional<SyncSlotExpression>& second)
{
    if (!first || !second) {
        return !first && !second;
    }
    if (first->kind != second->kind || first->depth != second->depth || first->modulus != second->modulus) {
        return false;
    }
    if (first->kind == SyncSlotExpressionKind::Unknown) {
        return first->selector == second->selector;
    }
    return first->loop == second->loop && first->coefficient == second->coefficient && first->offset == second->offset;
}

SmallVector<StorageAccessClass, 16> mlir::pto::protocol_sync::detail::buildGenerationAccessClasses(
    const StructuredSyncIR& schedule, ProtocolSyncStatistics* statistics)
{
    SmallVector<StorageAccessClass, 16> accessClasses = buildAccessClasses(schedule);
    markPartialOverlaps(accessClasses, statistics);
    markUnknownAliasConflicts(accessClasses);
    return splitGenerationClasses(schedule, accessClasses);
}
