// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Event causality uses iteration distances, never lexical physical completion.
// Zero-distance closure plus one carried edge suffices for distance <= 1.
#include "PTO/Transforms/ProtocolSync/RecurringEventLifetime.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <utility>

using namespace mlir::pto::protocol_sync;

SyncRecurringEventProof mlir::pto::protocol_sync::proveRecurringEventLifetimes(
    llvm::ArrayRef<SyncRecurringEventChannel> channels)
{
    constexpr unsigned maximumChannels = 64;
    const bool exceedsBudget = channels.size() > maximumChannels;
    if (exceedsBudget) {
        return {SyncRecurringEventStatus::AnalysisLimit, 0};
    }
    const unsigned count = 2 * channels.size();
    llvm::SmallVector<llvm::BitVector, 16> zero(count, llvm::BitVector(count));
    llvm::SmallVector<std::pair<unsigned, unsigned>, 16> carried;
    std::map<unsigned, std::map<unsigned, unsigned>> lanes;
    for (auto [index, channel] : llvm::enumerate(channels)) {
        const bool boundary = channel.primed == (channel.distance == 1) && channel.drained == channel.primed;
        const bool invalid = channel.distance > 1 || channel.setLane == channel.waitLane || !boundary;
        if (invalid) {
            return {SyncRecurringEventStatus::InvalidContract, static_cast<unsigned>(index)};
        }
        const unsigned set = 2 * index;
        const unsigned wait = set + 1;
        const bool uniqueSet = lanes[channel.setLane].emplace(channel.setOrder, set).second;
        const bool uniqueWait = lanes[channel.waitLane].emplace(channel.waitOrder, wait).second;
        if (!uniqueSet || !uniqueWait) {
            return {SyncRecurringEventStatus::InvalidContract, static_cast<unsigned>(index)};
        }
        if (channel.distance == 0) {
            zero[set].set(wait);
        } else {
            carried.push_back({set, wait});
        }
    }
    for (const auto& lane : lanes) {
        const auto& actions = lane.second;
        std::optional<unsigned> previous;
        for (const auto& action : actions) {
            if (previous) {
                zero[*previous].set(action.second);
            }
            previous = action.second;
        }
        carried.push_back({actions.rbegin()->second, actions.begin()->second});
    }
    for (unsigned middle = 0; middle < count; ++middle) {
        for (auto& targets : zero) {
            if (targets.test(middle)) {
                targets |= zero[middle];
            }
        }
    }
    for (unsigned action = 0; action < count; ++action) {
        if (zero[action].test(action)) {
            return {SyncRecurringEventStatus::ZeroDistanceCycle, action / 2};
        }
        zero[action].set(action); // Reflexivity is only for path composition.
    }
    for (auto [index, channel] : llvm::enumerate(channels)) {
        const unsigned set = 2 * index;
        const unsigned wait = set + 1;
        bool consumed = zero[wait].test(set);
        if (channel.distance == 0) {
            consumed = llvm::any_of(
                carried, [&](const auto& edge) { return zero[wait].test(edge.first) && zero[edge.second].test(set); });
        }
        if (!consumed) {
            return {SyncRecurringEventStatus::UnprovedRearm, static_cast<unsigned>(index)};
        }
    }
    // All causal edges advance iteration or belong to a zero-distance DAG.
    // Initial credits ground iteration zero. Every next set has a consuming
    // predecessor, so induction preserves capacity one. Final drains consume
    // the remaining carried credits, including the zero-trip initial marking.
    return {SyncRecurringEventStatus::Proven, 0};
}
