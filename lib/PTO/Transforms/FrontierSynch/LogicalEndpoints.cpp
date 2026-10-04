// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Uniform direct endpoint guards, matching identities and per-cut ordering.
#include "PTO/Transforms/FrontierSynch/LogicalEndpoints.h"
#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_set>
namespace mlir::pto::frontiersynch {
namespace {
LogicalEndpointPlan reject(const char* message)
{
    LogicalEndpointPlan result;
    result.error = message;
    return result;
}
bool validKind(EndpointKind kind)
{
    return kind == EndpointKind::Set || kind == EndpointKind::Barrier || kind == EndpointKind::Wait;
}
} // namespace
CountedLoopResult countedTrips(int64_t lower, int64_t upper, int64_t step)
{
    if (step <= 0) {
        return {EndpointError::InvalidInput, 0};
    }
    if (upper <= lower) {
        return {};
    }
    // Unsigned subtraction represents the positive mathematical span exactly,
    // including INT64_MIN..INT64_MAX; signed subtraction could overflow.
    const uint64_t span = static_cast<uint64_t>(upper) - static_cast<uint64_t>(lower);
    const auto stride = static_cast<uint64_t>(step);
    return {EndpointError::None, span / stride + (span % stride != 0 ? 1 : 0)};
}
CountedLoopResult countedOrdinal(int64_t lower, int64_t upper, int64_t step, int64_t induction)
{
    if (step <= 0 || induction < lower || induction >= upper) {
        return {EndpointError::InvalidInput, 0};
    }
    const uint64_t offset = static_cast<uint64_t>(induction) - static_cast<uint64_t>(lower);
    const auto stride = static_cast<uint64_t>(step);
    if (offset % stride) {
        return {EndpointError::InvalidInput, 0};
    }
    return {EndpointError::None, offset / stride};
}
LogicalEndpointPlan buildLogicalEndpoints(const PeriodicAnalysis& analysis)
{
    if (!analysis.error.empty()) {
        return reject("periodic analysis failed");
    }
    if (analysis.retained.size() > std::numeric_limits<uint32_t>::max() / 2) {
        return reject("logical recipe identity overflow");
    }
    LogicalEndpointPlan plan;
    std::unordered_set<uint32_t> seen;
    for (auto identity : analysis.retained) {
        if (identity >= analysis.generators.size() || !seen.insert(identity).second) {
            return reject("invalid retained generator identity");
        }
        const auto& record = analysis.generators[identity];
        const bool invalid = record.source >= analysis.payloads.size() || record.target >= analysis.payloads.size() ||
            (!record.displacement && record.source >= record.target);
        if (invalid) {
            return reject("invalid logical endpoint");
        }
        const auto producer = analysis.payloads[record.source].pipe, consumer = analysis.payloads[record.target].pipe;
        EndpointRecipe recipe{identity, record.source, record.target, consumer,
                              record.displacement, EndpointKind::Barrier};
        if (producer != consumer) {
            recipe.pipe = producer;
            recipe.kind = EndpointKind::Set;
            plan.recipes.push_back(recipe);
            recipe.pipe = consumer;
            recipe.kind = EndpointKind::Wait;
        }
        plan.recipes.push_back(recipe);
    }
    return plan;
}
EndpointEvaluation LogicalEndpointPlan::evaluate(uint32_t index, uint64_t trips, uint64_t ordinal) const
{
    if (!error.empty() || index >= recipes.size() || !validKind(recipes[index].kind)) {
        return {EndpointError::InvalidInput, false, {}};
    }
    const auto& recipe = recipes[index];
    if (ordinal >= trips) {
        return {};
    }
    uint64_t source = ordinal;
    if (recipe.kind == EndpointKind::Set) {
        // This subtraction form cannot overflow even when trips is UINT64_MAX.
        if (recipe.displacement >= trips || ordinal >= trips - recipe.displacement) {
            return {};
        }
    } else {
        if (ordinal < recipe.displacement) {
            return {};
        }
        source -= recipe.displacement;
    }
    const auto type = recipe.kind == EndpointKind::Set ? recipe.source : recipe.target;
    return {EndpointError::None, true, {recipe.kind, recipe.pipe, type, ordinal, {recipe.record, source}}};
}
EndpointError canonicalizeCoincidentCut(std::vector<EndpointInstance>& instances)
{
    if (instances.empty()) {
        return EndpointError::None;
    }
    for (const auto& instance : instances) {
        if (!validKind(instance.kind)) {
            return EndpointError::InvalidInput;
        }
    }
    std::sort(instances.begin(), instances.end(), [](const auto& a, const auto& b) {
        return std::tie(a.kind, a.pipe, a.identity.record, a.identity.sourceOrdinal, a.type, a.ordinal) <
               std::tie(b.kind, b.pipe, b.identity.record, b.identity.sourceOrdinal, b.type, b.ordinal);
    });
    auto equal = [](const EndpointInstance& a, const EndpointInstance& b) {
        return a.kind == b.kind && a.pipe == b.pipe && (a.kind == EndpointKind::Barrier ||
            (a.identity.record == b.identity.record && a.identity.sourceOrdinal == b.identity.sourceOrdinal &&
             a.type == b.type && a.ordinal == b.ordinal));
    };
    instances.erase(std::unique(instances.begin(), instances.end(), equal), instances.end());
    return EndpointError::None;
}
} // namespace mlir::pto::frontiersynch
