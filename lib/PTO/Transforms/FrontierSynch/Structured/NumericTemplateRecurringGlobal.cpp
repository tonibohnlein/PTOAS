// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Re-extract all original accesses of a writing base before publishing relative atoms.
#include "NumericTemplateRecurringGlobal.h"
namespace mlir::pto::frontiersynch::detail {
const RecurringGlobalEffect* RecurringGlobalTemplate::find(std::size_t payload, std::size_t effect) const
{
    auto found = effectIndex.find({payload, effect});
    return found == effectIndex.end() || found->second >= effects.size() ? nullptr : &effects[found->second];
}
namespace {
bool sameEffect(const TemplateEffect& a, const TemplateEffect& b)
{
    if (a.sourceEffect != b.sourceEffect || a.mode != b.mode || a.discharge != b.discharge ||
        a.outerStride != b.outerStride || a.pendingGlobalGeometry != b.pendingGlobalGeometry ||
        a.atoms != b.atoms || a.ranges.size() != b.ranges.size() || a.regions.size() != b.regions.size()) {
        return false;
    }
    for (auto [x, y] : llvm::zip(a.ranges, b.ranges)) {
        if (!sameStorageDomain(x, y) || x.begin != y.begin || x.end != y.end) { return false; }
    }
    for (auto [x, y] : llvm::zip(a.regions, b.regions)) {
        if (x.base != y.base || x.byteOffset != y.byteOffset || x.extents != y.extents ||
            x.elementBytes != y.elementBytes || x.invariantSymbols != y.invariantSymbols) { return false; }
    }
    return true;
}
bool samePayload(const TemplatePayload& a, const TemplatePayload& b)
{
    if (a.phase != b.phase || a.invocationProtection != b.invocationProtection ||
        a.coordinates.size() != b.coordinates.size() || a.effects.size() != b.effects.size()) { return false; }
    for (auto [x, y] : llvm::zip(a.coordinates, b.coordinates)) {
        if (x.loop != y.loop || x.induction != y.induction) { return false; }
    }
    for (auto [x, y] : llvm::zip(a.effects, b.effects)) { if (!sameEffect(x, y)) { return false; } }
    return true;
}
}
bool RecurringGlobalTemplate::applies(const NumericTemplate& form) const
{
    if (!input || !index || outer != form.outer || form.specializedBody || payloadCount != form.payloads.size() ||
        (inputOwner && inputOwner.get() != input) || (indexOwner && indexOwner.get() != index)) { return false; }
    if (effectIndex.size() != effects.size() || signature.size() != form.payloads.size() ||
        lower != form.lower || step != form.step || form.emptyInvocation) { return false; }
    for (auto [first, second] : llvm::zip(signature, form.payloads)) {
        if (!samePayload(first, second)) { return false; }
    }
    for (const auto& record : effects) {
        if (find(record.payload, record.effect) != &record) { return false; }
        if (record.payload >= form.payloads.size()) { return false; }
        const auto& payload = form.payloads[record.payload];
        if (record.effect >= payload.effects.size() || payload.phase != record.phase ||
            payload.coordinates.size() != record.coordinates.size()) { return false; }
        for (auto [first, second] : llvm::zip(payload.coordinates, record.coordinates)) {
            if (first.loop != second.loop || first.induction != second.induction) { return false; }
        }
        const auto& effect = payload.effects[record.effect];
        if (record.sourceEffect >= input->accesses().effects().size() ||
            effect.mode != input->accesses().effects()[record.sourceEffect].mode ||
            effect.sourceEffect != record.sourceEffect || effect.pendingGlobalGeometry ||
            effect.discharge != TemplateDischarge::None || !effect.atoms.empty() || !effect.ranges.empty()) {
            return false;
        }
        for (auto atom : record.atoms) { if (atom >= atomCount) { return false; } }
    }
    return true;
}
namespace {
void chargeAttempt(NumericTemplate& form, uint64_t count)
{
    auto& work = form.estimatedAttemptedFragments;
    work = count > UINT64_MAX - work ? UINT64_MAX : work + count;
}
struct BaseExtraction {
    uint64_t materializedFragments = 0;
    std::vector<AlignedModularMap> maps;
    std::vector<std::size_t> owners;
    std::vector<RecurringGlobalEffect> effects;
};
bool extractEffect(TemplateBuilder& builder, std::size_t payloadId, std::size_t effectId, BaseExtraction& output)
{
    const auto& payload = builder.output.payloads[payloadId];
    const auto& effect = payload.effects[effectId];
    const auto& original = builder.input.accesses().effects()[effect.sourceEffect];
    if (original.regions.empty() || original.phase != payload.phase ||
        builder.output.fragments > builder.output.limits.fragments ||
        original.regions.size() > builder.output.limits.fragments - builder.output.fragments) { return false; }
    chargeAttempt(builder.output, original.regions.size());
    RecurringGlobalEffect record{payloadId, effectId, effect.sourceEffect, payload.phase,
        payload.coordinates, {}, {}};
    record.originalMaps.assign(original.regions.begin(), original.regions.end());
    RegionExpressions expressions;
    ModularScalar scalar(expressions, builder.output.outer);
    DenseMap<Value, int64_t> fixed;
    for (auto coordinate : payload.coordinates) {
        if (coordinate.loop != builder.output.outer) {
            fixed[coordinate.loop.getInductionVar()] = coordinate.induction;
        }
    }
    auto binding = [&](Value value) -> std::optional<RegionExpressions::Id> {
        auto found = fixed.find(value);
        return found == fixed.end() ? std::nullopt :
            std::optional<RegionExpressions::Id>(expressions.constant(found->second));
    };
    auto constant = [&](Value value) -> std::optional<llvm::APInt> {
        auto id = scalar.value(value, binding);
        auto bits = id ? expressions.constantValue(*id) : std::nullopt;
        const auto width = scalar.width(value.getType());
        return bits && width && width <= 64 ? std::optional<llvm::APInt>(llvm::APInt(width, *bits)) : std::nullopt;
    };
    AlignedModularMapBuilder mapBuilder(builder.output.outer, scalar, constant);
    for (const auto& region : original.regions) {
        const auto limit = builder.output.limits.fragments;
        if (builder.output.fragments > limit || output.materializedFragments >= limit - builder.output.fragments) {
            return false;
        }
        const auto remaining = limit - builder.output.fragments - output.materializedFragments;
        auto map = mapBuilder.build(region, remaining - 1,
            [&](uint64_t count) { chargeAttempt(builder.output, count); });
        if (!map || map->access.intervals.size() >= remaining) { return false; }
        output.materializedFragments += 1 + map->access.intervals.size();
        output.maps.push_back(std::move(*map));
        output.owners.push_back(output.effects.size());
    }
    output.effects.push_back(std::move(record));
    return expressions.constructionError().empty();
}
}
bool prepareRecurringGlobalBase(TemplateBuilder& builder,
    ArrayRef<std::pair<std::size_t, std::size_t>> effects)
{
    if (!allowRecurringGlobal(builder) || effects.empty()) { return false; }
    BaseExtraction extracted;
    for (auto [payload, effect] : effects) {
        if (!extractEffect(builder, payload, effect, extracted)) { return false; }
    }
    if (builder.output.fragments > builder.output.limits.fragments) { return false; }
    const auto remaining = builder.output.limits.fragments - builder.output.fragments;
    auto group = partitionAlignedModularMaps(extracted.maps, remaining,
        [&](uint64_t count) { chargeAttempt(builder.output, count); });
    if (!group) { return false; }
    auto owner = builder.output.recurringGlobal ?
        std::make_shared<RecurringGlobalTemplate>(*builder.output.recurringGlobal) :
        std::make_shared<RecurringGlobalTemplate>();
    owner->input = &builder.input; owner->index = &builder.index; owner->outer = builder.output.outer;
    owner->payloadCount = builder.output.payloads.size();
    if (owner->atomCount > UINT32_MAX - group->partition.atoms.size()) { return false; }
    uint64_t charge = extracted.maps.size() + group->partition.atoms.size();
    for (const auto& map : extracted.maps) { charge += map.access.intervals.size(); }
    for (auto [id, references] : llvm::enumerate(group->partition.references)) {
        charge += references.size();
        auto& record = extracted.effects[extracted.owners[id]];
        for (auto atom : references) { record.atoms.push_back(static_cast<uint32_t>(owner->atomCount + atom)); }
    }
    if (charge > remaining) {
        builder.output.result.note(RecognitionIssue::TemplateExpansionLimit, builder.output.outer);
        return false;
    }
    builder.output.fragments += charge;
    for (const auto& record : extracted.effects) {
        if (owner->find(record.payload, record.effect)) { return false; }
    }
    for (auto& record : extracted.effects) {
        llvm::sort(record.atoms);
        record.atoms.erase(std::unique(record.atoms.begin(), record.atoms.end()), record.atoms.end());
        auto& effect = builder.output.payloads[record.payload].effects[record.effect];
        effect.regions.clear(); effect.ranges.clear(); effect.atoms.clear();
        effect.pendingGlobalGeometry = false; effect.discharge = TemplateDischarge::None; effect.outerStride = 0;
        owner->effectIndex.emplace(std::make_pair(record.payload, record.effect), owner->effects.size());
        owner->effects.push_back(std::move(record));
    }
    owner->atomCount += group->partition.atoms.size();
    owner->groups.push_back(std::move(*group));
    builder.output.recurringGlobal = std::move(owner);
    return true;
}
bool sealRecurringGlobal(NumericTemplate& form)
{
    if (!form.recurringGlobal) { return true; }
    for (const auto& payload : form.payloads) {
        for (const auto& effect : payload.effects) { if (effect.pendingGlobalGeometry) { return false; } }
    }
    auto owner = std::make_shared<RecurringGlobalTemplate>(*form.recurringGlobal);
    owner->signature = form.payloads;
    owner->lower = form.lower; owner->step = form.step;
    form.recurringGlobal = std::move(owner);
    return form.recurringGlobal->applies(form);
}
bool retainRecurringGlobalOwners(NumericTemplate& form, std::shared_ptr<const SyncInput> input,
    std::shared_ptr<const PhaseIndex> index)
{
    if (!form.recurringGlobal) { return true; }
    if (form.recurringGlobal->input != input.get() || form.recurringGlobal->index != index.get() ||
        !form.recurringGlobal->applies(form)) { return false; }
    auto owner = std::make_shared<RecurringGlobalTemplate>(*form.recurringGlobal);
    owner->inputOwner = std::move(input); owner->indexOwner = std::move(index);
    form.recurringGlobal = std::move(owner);
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
