// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- EventLifetime.cpp - Prove consumption before rearming ----------------===//
#include "PTO/Transforms/ProtocolSync/EventLifetime.h"
#include "PTO/IR/PTO.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

struct EventPoint {
    Operation* anchor = nullptr;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    // Planned waits precede their phase, planned signals follow it; concrete
    // synchronization instructions occupy their own exact program points.
    int offset = 0;
};

bool strictlyBefore(const EventPoint& first, const EventPoint& second)
{
    const bool differentBlocks = first.anchor->getBlock() != second.anchor->getBlock();
    if (differentBlocks) {
        return false;
    }
    return first.anchor == second.anchor ? first.offset < second.offset : first.anchor->isBeforeInBlock(second.anchor);
}

bool appendGeneration(const SyncEventGeneration& generation, SmallVectorImpl<EventPoint>& points)
{
    const bool onceOnlyKind =
        generation.kind == SyncEventGenerationKind::OneShot || generation.kind == SyncEventGenerationKind::DirectRepair;
    const bool unsupported = !onceOnlyKind || generation.recurring || generation.recurrenceOwner != kInvalidSyncId ||
                             !generation.guard.empty() || !generation.setAnchor || !generation.waitAnchor ||
                             !isa_and_nonnull<func::FuncOp>(generation.setAnchor->getParentOp()) ||
                             !isa_and_nonnull<func::FuncOp>(generation.waitAnchor->getParentOp());
    if (unsupported) {
        points.append(2, EventPoint{});
        return false;
    }
    auto function = cast<func::FuncOp>(generation.setAnchor->getParentOp());
    const bool onceOnlyBlock =
        function.getBody().hasOneBlock() && generation.waitAnchor->getParentOp() == function.getOperation();
    if (!onceOnlyBlock) {
        points.append(2, EventPoint{});
        return false;
    }
    const EventPoint set{
        generation.setAnchor, generation.core, generation.sourcePipe, isa<SetFlagOp>(generation.setAnchor) ? 0 : 1};
    const EventPoint wait{
        generation.waitAnchor, generation.core, generation.targetPipe, isa<WaitFlagOp>(generation.waitAnchor) ? 0 : -1};
    if (!strictlyBefore(set, wait)) {
        points.append(2, EventPoint{});
        return false;
    }
    points.push_back(set);
    points.push_back(wait);
    return true;
}

void closeOrder(SmallVectorImpl<llvm::BitVector>& before)
{
    for (unsigned middle = 0; middle < before.size(); ++middle) {
        for (auto& sources : before) {
            if (sources.test(middle)) {
                sources |= before[middle];
            }
        }
    }
}

} // namespace

SyncEventConsumptionOrder mlir::pto::protocol_sync::buildEventConsumptionOrder(
    ArrayRef<SyncEventGeneration> generations)
{
    SyncEventConsumptionOrder result;
    // Bounded dense closure; exceeding this limit disables reuse proofs only.
    constexpr unsigned maximumGenerations = 128;
    const bool exceedsLimit = generations.size() > maximumGenerations;
    if (exceedsLimit) {
        result.budgetExceeded = true;
        return result;
    }
    SmallVector<EventPoint, 16> points;
    result.before.assign(2 * generations.size(), llvm::BitVector(2 * generations.size()));
    for (auto [index, generation] : llvm::enumerate(generations)) {
        if (generation.id != index) {
            return SyncEventConsumptionOrder{};
        }
        if (appendGeneration(generation, points)) {
            result.before[2 * index].set(2 * index + 1);
        }
    }
    for (auto [first, source] : llvm::enumerate(points)) {
        for (auto [second, target] : llvm::enumerate(points)) {
            const bool sameLane = source.anchor && target.anchor && source.core == target.core &&
                                  source.core != SyncPhysicalCore::Unknown && source.pipe == target.pipe;
            if (sameLane && strictlyBefore(source, target)) {
                result.before[first].set(second);
            }
        }
    }
    closeOrder(result.before);
    return result;
}

bool SyncEventConsumptionOrder::provesConsumedBeforeSet(SyncEventGenerationId first, SyncEventGenerationId second) const
{
    const unsigned generationCount = before.size() / 2;
    const bool valid = first < generationCount && second < generationCount && first != second;
    return valid && before[2 * first + 1].test(2 * second);
}
