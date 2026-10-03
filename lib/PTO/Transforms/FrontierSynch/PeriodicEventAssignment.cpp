// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/PeriodicEventAssignment.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using llvm::DynamicAPInt;
DynamicAPInt integer(std::size_t value) { return DynamicAPInt(static_cast<int64_t>(value)); }
bool validatePool(const PeriodicDemandReduction& reduction, ArrayRef<std::size_t> ids)
{
    const auto records = reduction.generators();
    const auto sites = reduction.sites();
    llvm::DenseSet<std::size_t> selected;
    llvm::DenseSet<std::size_t> retained;
    for (auto id : reduction.retained()) { retained.insert(id); }
    for (auto id : ids) {
        if (id >= records.size() || !retained.contains(id) || !selected.insert(id).second) { return false; }
        const auto& edge = records[id].edge;
        if (edge.source >= sites.size() || edge.consumer >= sites.size() ||
            !sites[edge.source] || !sites[edge.consumer] || edge.distance < 0) { return false; }
    }
    if (ids.empty()) { return true; }
    const auto& first = records[ids.front()].edge;
    const auto sourcePipe = sites[first.source]->kPipeValue;
    const auto targetPipe = sites[first.consumer]->kPipeValue;
    if (sourcePipe == targetPipe) { return false; }
    for (auto id : reduction.retained()) {
        const auto& edge = records[id].edge;
        const bool member = sites[edge.source]->kPipeValue == sourcePipe &&
                            sites[edge.consumer]->kPipeValue == targetPipe;
        if (member != selected.contains(id)) { return false; }
    }
    return true;
}
bool orderedTargets(ArrayRef<PeriodicPrerequisite> phases, const DynamicAPInt& periodSize)
{
    if (phases.empty()) { return true; }
    auto target = [&](const PeriodicPrerequisite& phase) {
        return periodSize * phase.distance + integer(phase.consumer);
    };
    for (std::size_t i = 1; i < phases.size(); ++i) {
        if (phases[i - 1].source >= phases[i].source || target(phases[i - 1]) >= target(phases[i])) {
            return false;
        }
    }
    return target(phases.back()) < target(phases.front()) + periodSize;
}
FailureOr<SmallVector<PeriodicDistance>> reusableSeparations(
    const PeriodicDemandReduction& reduction, ArrayRef<PeriodicPrerequisite> phases)
{
    SmallVector<PeriodicDistance> result;
    const auto count = integer(phases.size());
    for (std::size_t r = 0; r < phases.size(); ++r) {
        PeriodicDistance earliest;
        for (std::size_t s = 0; s < phases.size(); ++s) {
            auto threshold = reduction.threshold(phases[r].consumer, phases[s].source, PeriodicEventKind::Start);
            if (failed(threshold)) { return failure(); }
            if (!*threshold) { continue; }
            const auto periods = std::max(phases[r].distance + **threshold, DynamicAPInt(s <= r ? 1 : 0));
            const auto separation = count * periods + integer(s) - integer(r);
            if (separation <= 0) { return failure(); }
            if (!earliest || separation < *earliest) { earliest = separation; }
        }
        result.push_back(std::move(earliest));
    }
    return result;
}
DynamicAPInt prefixHandoffs(ArrayRef<PeriodicPrerequisite> phases,
    const DynamicAPInt& periodSize, const DynamicAPInt& payloads)
{
    DynamicAPInt result(0);
    for (const auto& phase : phases) {
        const auto position = integer(phase.consumer);
        if (payloads <= position) { continue; }
        const auto count = (payloads - 1 - position) / periodSize - phase.distance + 1;
        result += std::max(DynamicAPInt(0), count);
    }
    return result;
}
PeriodicDistance budget(ArrayRef<PeriodicDistance> separations,
    const std::optional<DynamicAPInt>& handoffs)
{
    DynamicAPInt result(0);
    for (std::size_t r = 0; r < separations.size(); ++r) {
        if (handoffs && integer(r) >= *handoffs) { break; }
        if (!handoffs && !separations[r]) { return std::nullopt; }
        const auto value = handoffs ?
            (separations[r] ? std::min(*separations[r], *handoffs - integer(r)) : *handoffs - integer(r)) :
            *separations[r];
        result = std::max(result, value);
    }
    return result;
}
} // namespace
PeriodicEventAssignment assignPeriodicEvents(const PeriodicDemandReduction& reduction,
    ArrayRef<std::size_t> poolRecords, ArrayRef<unsigned> eligibleIds,
    std::optional<DynamicAPInt> prefixPayloads)
{
    PeriodicEventAssignment result;
    const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max());
    llvm::DenseSet<unsigned> distinct;
    for (auto id : eligibleIds) {
        if (!distinct.insert(id).second) { result.reason = "duplicate eligible physical ID"; return result; }
    }
    if (reduction.sites().size() > maximum || poolRecords.size() > maximum || eligibleIds.size() > maximum ||
        (prefixPayloads && *prefixPayloads < 0) || !validatePool(reduction, poolRecords)) {
        result.reason = "numerical periodic directed-cover pool premises unmet";
        return result;
    }
    for (auto id : poolRecords) { result.phases.push_back(reduction.generators()[id].edge); }
    llvm::sort(result.phases, [](const auto& left, const auto& right) { return left.source < right.source; });
    if (!orderedTargets(result.phases, integer(reduction.sites().size()))) {
        result.reason = "periodic acquisitions are not strictly source-ordered across period boundaries";
        return result;
    }
    auto separations = reusableSeparations(reduction, result.phases);
    if (failed(separations)) { result.reason = "quotient return-order query is unavailable"; return result; }
    result.firstReusableSeparations = std::move(*separations);
    result.eligibleIds.assign(eligibleIds.begin(), eligibleIds.end());
    if (prefixPayloads) {
        result.handoffCount = prefixHandoffs(result.phases, integer(reduction.sites().size()), *prefixPayloads);
    }
    result.uniformRequired = budget(result.firstReusableSeparations, std::nullopt);
    result.required = budget(result.firstReusableSeparations, result.handoffCount);
    if (!result.required || *result.required > integer(eligibleIds.size())) {
        result.status = PeriodicAssignmentStatus::Counterexample;
        llvm::raw_string_ostream message(result.reason);
        message << "selected periodic plan requires ";
        if (result.required) { message << *result.required; } else { message << "unboundedly many"; }
        message << " IDs; eligible pool contains " << eligibleIds.size();
        return result;
    }
    result.status = PeriodicAssignmentStatus::Certified;
    return result;
}
FailureOr<unsigned> PeriodicEventAssignment::sourceId(std::size_t phase, const DynamicAPInt& sourcePeriod) const
{
    if (status != PeriodicAssignmentStatus::Certified || phase >= phases.size() ||
        sourcePeriod < 0 || eligibleIds.empty()) { return failure(); }
    const auto ordinal = integer(phases.size()) * sourcePeriod + integer(phase);
    if (handoffCount && ordinal >= *handoffCount) { return failure(); }
    const auto index = static_cast<int64_t>(ordinal % integer(eligibleIds.size()));
    return eligibleIds[static_cast<std::size_t>(index)];
}
FailureOr<unsigned> PeriodicEventAssignment::consumerId(std::size_t phase, const DynamicAPInt& consumerPeriod) const
{
    if (phase >= phases.size() || consumerPeriod < phases[phase].distance) { return failure(); }
    return sourceId(phase, consumerPeriod - phases[phase].distance);
}
} // namespace mlir::pto::frontiersynch
