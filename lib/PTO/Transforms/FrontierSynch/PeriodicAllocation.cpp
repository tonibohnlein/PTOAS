// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact return-order budgets from completion-origin quotient thresholds.
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
#include "llvm/ADT/APInt.h"
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <tuple>
#include <unordered_set>
namespace mlir::pto::frontiersynch {
namespace {
PeriodicAllocation reject(const char* message)
{
    PeriodicAllocation output;
    output.error = message;
    return output;
}
bool ordered(DirectedAllocation& pool)
{
    auto& values = pool.handoffs;
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.source < b.source; });
    std::unordered_set<uint32_t> targets;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto& current = values[i];
        if (!targets.insert(current.target).second) {
            return false;
        }
        if (i) {
            const auto& previous = values[i - 1];
            const bool strictlyOrdered = previous.source < current.source &&
                std::tie(previous.displacement, previous.target) < std::tie(current.displacement, current.target);
            if (!strictlyOrdered) {
                return false;
            }
        }
    }
    if (values.empty()) {
        return true;
    }
    const auto& first = values.front();
    const auto& last = values.back();
    // (last.delay,last.target) < (first.delay+1,first.target), without overflowing +1.
    return last.displacement <= first.displacement ||
        (last.displacement - first.displacement == 1 && last.target < first.target);
}
bool budgets(const PeriodicAnalysis& analysis, DirectedAllocation& pool)
{
    const uint64_t count = pool.handoffs.size();
    pool.uniformBudget = 0;
    for (uint32_t r = 0; r < count; ++r) {
        auto& current = pool.handoffs[r];
        std::optional<llvm::APInt> best;
        for (uint32_t s = 0; s < count; ++s) {
            const auto& next = pool.handoffs[s];
            const auto query = analysis.completionThreshold(current.target, {next.source, PeriodicEventKind::Start});
            if (query.error != PeriodicQueryError::None) {
                return false;
            }
            if (!query.displacement) {
                continue;
            }
            auto periods = llvm::APInt(128, current.displacement) + llvm::APInt(128, *query.displacement);
            if (periods.isZero() && s <= r) {
                periods = llvm::APInt(128, 1);
            }
            // At most 65+32 bits. Keep the subtraction widened: the final value
            // may fit even when the multiplication alone exceeds UINT64_MAX.
            auto separation = periods * llvm::APInt(128, count) + llvm::APInt(128, s) - llvm::APInt(128, r);
            if (!best || separation.ult(*best)) {
                best = std::move(separation);
            }
        }
        if (!best) {
            current.firstReuse = std::nullopt;
            pool.uniformBudget = std::nullopt;
            continue;
        }
        if (best->getActiveBits() > 64) {
            return false;
        }
        current.firstReuse = best->getZExtValue();
        if (pool.uniformBudget) {
            pool.uniformBudget = std::max(*pool.uniformBudget, *current.firstReuse);
        }
    }
    return true;
}
} // namespace
PeriodicAllocation buildPeriodicAllocation(const PeriodicAnalysis& analysis)
{
    if (!analysis.error.empty() || analysis.payloads.size() > UINT32_MAX || analysis.retained.size() > UINT32_MAX) {
        return reject("invalid periodic allocation input");
    }
    PeriodicAllocation output;
    std::unordered_map<uint64_t, uint32_t> poolIDs;
    std::unordered_set<uint32_t> seen;
    for (auto identity : analysis.retained) {
        if (identity >= analysis.generators.size() || !seen.insert(identity).second) {
            return reject("invalid retained handoff identity");
        }
        const auto& record = analysis.generators[identity];
        const bool invalidEndpoint = record.source >= analysis.payloads.size() ||
            record.target >= analysis.payloads.size() || (!record.displacement && record.source >= record.target);
        if (invalidEndpoint) {
            return reject("invalid handoff endpoint");
        }
        auto from = analysis.payloads[record.source].pipe, to = analysis.payloads[record.target].pipe;
        if (from == to) {
            continue; // Same-pipe barriers do not consume event IDs.
        }
        const uint64_t key = (uint64_t(from) << 32) | to;
        auto [entry, inserted] = poolIDs.emplace(key, output.directions.size());
        if (inserted) {
            output.directions.emplace_back();
        }
        auto& pool = output.directions[entry->second];
        pool.sourcePipe = from;
        pool.targetPipe = to;
        pool.payloadTypes = static_cast<uint32_t>(analysis.payloads.size());
        pool.handoffs.push_back({identity, record.source, record.target, record.displacement, std::nullopt});
    }
    for (auto& pool : output.directions) {
        if (!ordered(pool)) {
            return reject("handoff endpoints are not strictly ordered and unique");
        }
        if (!budgets(analysis, pool)) {
            return reject("periodic allocation arithmetic overflow or invalid threshold");
        }
    }
    return output;
}
} // namespace mlir::pto::frontiersynch
