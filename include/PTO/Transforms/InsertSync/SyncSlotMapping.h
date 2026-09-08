// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSLOTMAPPING_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSLOTMAPPING_H
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncPlanningPrimitives.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "PTO/Transforms/InsertSync/StorageFrontierRelations.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

namespace mlir::pto {
struct InsertSyncSlotRequirement {
    Operation* source;
    Operation* target;
    Value sourceBuffer;
    Value targetBuffer;
    bool carried;
};
// An unequal selector proves non-overlap only if every unequal-index pair is
// physically disjoint. Equal capacity or equal root names alone are insufficient.
// This same certificate is required when selectors index recurring event keys.
inline bool haveCompatibleInsertSyncSlots(const BaseMemInfo* first, const BaseMemInfo* second)
{
    if (!first || !second || first->scope != second->scope || first->scope == AddressSpace::GM ||
        first->aliasesUnknownRange || second->aliasesUnknownRange || first->baseAddresses.size() < 2 ||
        first->baseAddresses.size() > kMaxMultiBufferCount ||
        first->baseAddresses.size() != second->baseAddresses.size() || !first->allocateSize || !second->allocateSize) {
        return false;
    }
    if (!(first->hasKnownPhysicalAddresses && second->hasKnownPhysicalAddresses) &&
        (first->rootBuffer != second->rootBuffer ||
         first->hasKnownPhysicalAddresses != second->hasKnownPhysicalAddresses)) {
        return false;
    }
    for (size_t a = 0; a < first->baseAddresses.size(); ++a) {
        for (size_t b = 0; b < second->baseAddresses.size(); ++b) {
            if (a != b &&
                insert_sync_detail::addressRangesMayOverlap(
                    first->baseAddresses[a], first->allocateSize, second->baseAddresses[b], second->allocateSize)) {
                return false;
            }
        }
    }
    return true;
}

// A known delta counts invocations of the same supported loop, not byte offsets
// or static operation indices. The native caller must prove that correspondence.
inline insert_sync_frontier::AccessRelationResult compareInsertSyncSlotOccurrences(
    const BaseMemInfo* source, const BaseMemInfo* target, scf::ForOp loop,
    int64_t targetIterationMinusSource, insert_sync_frontier::Budget& budget)
{
    using namespace insert_sync_frontier;
    if (!loop || !source || !target || source->scope != target->scope || source->scope == AddressSpace::GM)
        return {};
    auto convert = [&](const BaseMemInfo* info) -> std::optional<AccessSlice> {
        if (info->aliasesUnknownRange || !info->hasKnownPhysicalAddresses || !info->allocateSize ||
            info->baseAddresses.empty() || info->baseAddresses.size() > 16) return std::nullopt;
        AccessSlice result;
        result.known = true; result.write = true; result.space = static_cast<unsigned>(info->scope);
        result.extent = info->allocateSize;
        SlotMap slots;
        slots.modulus = info->baseAddresses.size(); slots.extent = info->allocateSize;
        slots.addresses.assign(info->baseAddresses.begin(), info->baseAddresses.end());
        auto permutation = recoverLoopCarriedSlotPermutation(info->baseBuffer);
        if (permutation && permutation->loop == loop.getOperation() &&
            permutation->initialSlots.size() == slots.modulus) {
            // Translator supplies physical addresses in this exact recurrence
            // order. One logical iteration advances one permutation edge.
            slots.selector = {0, {{0, 1}}, true};
        } else {
            auto selector = normalizeSlotSSA(findMultiTileSlotExpr(info->baseBuffer), slots.modulus);
            if (!selector) return std::nullopt;
            slots.selector = {selector->offset, {}, true};
            if (selector->induction) {
                if (selector->induction != loop.getInductionVar()) return std::nullopt;
                IntegerAttr lower, step;
                if (!matchPattern(loop.getLowerBound(), m_Constant(&lower)) ||
                    !matchPattern(loop.getStep(), m_Constant(&step))) return std::nullopt;
                int64_t start;
                if (!checkedSigned(slots.selector.constant, lower.getInt(), false, start)) return std::nullopt;
                slots.selector.constant = start;
                slots.selector.terms.push_back({0, step.getInt()});
            }
        }
        result.slots = std::move(slots);
        return result;
    };
    auto x = convert(source), y = convert(target);
    if (!x || !y) return {};
    OccurrenceRelation relation;
    relation.kind = targetIterationMinusSource ? OccurrenceRelation::Kind::KnownDistance :
                                               OccurrenceRelation::Kind::SameInstance;
    relation.commonLoops = {0};
    relation.deltas.push_back({0, targetIterationMinusSource});
    return compareAccesses(*x, *y, relation, false, budget);
}

inline bool disjointInsertSyncSlotOccurrences(const BaseMemInfo* source, const BaseMemInfo* target,
                                              scf::ForOp loop, bool carried,
                                              insert_sync_frontier::Budget& budget)
{
    if (!source || !target) return false;
    // A legacy carried record also protects wraparound reuse. Check the entire
    // finite selector period, including recurrence to the original storage.
    size_t period = carried ? std::lcm(source->baseAddresses.size(), target->baseAddresses.size()) : 1;
    if (!period || period > 16) return false;
    for (unsigned d = 0; d < period; ++d)
        if (compareInsertSyncSlotOccurrences(source, target, loop, carried ? d + 1 : 0, budget).kind !=
            insert_sync_frontier::AccessRelationResult::Kind::Disjoint) return false;
    return true;
}
} // namespace mlir::pto
#endif
