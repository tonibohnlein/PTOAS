// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Threshold and finite-prefix queries for completion-origin reachability.
#include "PeriodicAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
namespace {
bool present(std::size_t types, uint32_t type, uint64_t period, uint64_t prefix)
{
    // Division avoids overflowing period*types on an invalid query.
    return types && type < types && type < prefix && period <= (prefix - 1 - type) / types;
}
bool validKind(PeriodicEventKind kind)
{
    return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
}
uint32_t vertex(PeriodicEvent event)
{
    return 2 * event.type + (event.kind == PeriodicEventKind::Completion ? 1 : 0);
}
} // namespace
PeriodicThreshold PeriodicAnalysis::completionThreshold(uint32_t source, PeriodicEvent target) const
{
    if (!error.empty() || source >= payloads.size() || target.type >= payloads.size() || !validKind(target.kind)) {
        return {PeriodicQueryError::InvalidInput, std::nullopt};
    }
    const auto& row = frontiers[sourceRows[source]];
    const auto distance = row.distances[vertex(target)];
    if (!distance) {
        return {};
    }
    return {PeriodicQueryError::None, periodic::threshold(*distance, localRanks[source], row.count)};
}
PeriodicPredicate PeriodicAnalysis::completionPrecedes(uint32_t source, uint64_t sourcePeriod,
                                                      PeriodicEvent target, uint64_t targetPeriod,
                                                      uint64_t payloadPrefixLength, bool strict) const
{
    const bool endpointsPresent = present(payloads.size(), source, sourcePeriod, payloadPrefixLength) &&
        present(payloads.size(), target.type, targetPeriod, payloadPrefixLength);
    if (!endpointsPresent) {
        return {PeriodicQueryError::InvalidInput, false};
    }
    const auto threshold = completionThreshold(source, target);
    if (threshold.error != PeriodicQueryError::None) {
        return {threshold.error, false};
    }
    const bool identity = source == target.type && sourcePeriod == targetPeriod &&
        target.kind == PeriodicEventKind::Completion;
    if (targetPeriod < sourcePeriod || (strict && identity)) {
        return {};
    }
    return {PeriodicQueryError::None,
            threshold.displacement && targetPeriod - sourcePeriod >= *threshold.displacement};
}
PeriodicRank PeriodicAnalysis::completionRank(uint32_t pipe, PeriodicEvent target, uint64_t targetPeriod,
                                             uint64_t payloadPrefixLength) const
{
    const bool valid = error.empty() && validKind(target.kind) &&
        present(payloads.size(), target.type, targetPeriod, payloadPrefixLength);
    if (!valid) {
        return {PeriodicQueryError::InvalidInput, 0};
    }
    const auto found = pipeRows.find(pipe);
    if (found == pipeRows.end()) {
        return {}; // A pipe absent from the word has no completion ancestors.
    }
    const auto& row = frontiers[found->second];
    const auto distance = row.distances[vertex(target)];
    if (!distance) {
        return {};
    }
    uint64_t start = 0, rank = 0;
    if (!periodic::multiply(row.count, targetPeriod, start)) {
        return {PeriodicQueryError::Overflow, 0};
    }
    // Subtract D before adding h, so a representable final rank is not rejected
    // just because the next period's boundary would exceed UINT64_MAX.
    if (*distance >= row.count) {
        const auto back = *distance - row.count;
        return {PeriodicQueryError::None, start > back ? start - back : 0};
    }
    if (!periodic::add(start, row.count - *distance, rank)) {
        return {PeriodicQueryError::Overflow, 0};
    }
    return {PeriodicQueryError::None, rank};
}
} // namespace mlir::pto::frontiersynch
