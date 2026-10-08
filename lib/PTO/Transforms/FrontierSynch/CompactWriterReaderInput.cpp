// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "CompactWriterReaderInputInternal.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
bool uniformRanges(const SyncStorageEffect& effect)
{
    // Shared numeric materialization is a complete bound. Remaining affine
    // symbols could correlate only same-visit addresses, so do not use those
    // maps to separate the union of bytes from different visits.
    return effect.rangesMaterialized && llvm::all_of(effect.regions,
        [](const auto& region) { return region.symbols.empty(); });
}
bool interval(const OriginDistanceInterval& value)
{
    return !value.reachable || !value.maximum || value.minimum <= *value.maximum;
}
void include(OriginDistanceInterval& aggregate, const OriginDistanceInterval& value)
{
    if (!value.reachable) { return; }
    if (!aggregate.reachable) { aggregate = value; return; }
    aggregate.minimum = std::min(aggregate.minimum, value.minimum);
    aggregate.maximum = aggregate.maximum && value.maximum ?
        std::optional<uint64_t>(std::max(*aggregate.maximum, *value.maximum)) : std::nullopt;
}
class Builder {
public:
    Builder(scf::ForOp loop, const SyncInput& input, const PhaseIndex& index,
            llvm::ArrayRef<std::vector<const CompoundInstanceElement*>> slots,
            const CompactWriterReaderBindings& bindings)
        : loop(loop), input(input), index(index), slots(slots), bindings(bindings), model(input.accesses()),
          protection(structuredProtection(model)) {}
    CompactWriterReaderInput result;
    bool run();
private:
    bool fail(CompactInputIssue issue, const char* message)
    {
        result.issue = issue; result.error = message; return false;
    }
    bool skeleton();
    bool prerequisites();
    bool collect();
    bool mayMeet(std::size_t a, std::size_t b);
    std::size_t root(std::size_t value);
    bool classes();
    bool queries();
    std::optional<OriginDistanceInterval> pair(std::size_t a, std::size_t b, StorageHazard hazard);
    uint64_t group(const SyncStorageEffect& effect) const;
    scf::ForOp loop;
    const SyncInput& input;
    const PhaseIndex& index;
    llvm::ArrayRef<std::vector<const CompoundInstanceElement*>> slots;
    const CompactWriterReaderBindings& bindings;
    const SyncStorageEffects& model;
    StructuredProtection protection;
    llvm::DenseSet<const CompoundInstanceElement*> members;
    llvm::DenseMap<const CompoundInstanceElement*, uint32_t> phaseSlots, phaseIds;
    std::vector<std::size_t> effects, parents;
    std::vector<uint32_t> sites;
};
bool Builder::skeleton()
{
    if (!loop || slots.size() > UINT32_MAX) {
        return fail(CompactInputIssue::InvalidBinding, "compact input needs a representable original loop slot word");
    }
    llvm::DenseSet<const CompoundInstanceElement*> shared(input.instructions().begin(), input.instructions().end());
    bool singleton = true;
    for (uint32_t slot = 0; slot < slots.size(); ++slot) {
        const auto& alternatives = slots[slot];
        if (alternatives.empty()) {
            return fail(CompactInputIssue::InvalidBinding, "compact slot has no executed alternative");
        }
        std::optional<uint32_t> pipe;
        for (const auto* phase : alternatives) {
            if (!phase || !shared.contains(phase) || !phase->elementOp ||
                !loop->isProperAncestor(phase->elementOp) || !members.insert(phase).second ||
                phaseIds.size() == UINT32_MAX) {
                return fail(CompactInputIssue::InvalidBinding,
                            "compact slot alternatives do not partition shared phases");
            }
            const auto current = static_cast<uint32_t>(phase->kPipeValue);
            if (pipe && *pipe != current) {
                return fail(CompactInputIssue::InvalidBinding, "compact slot alternatives have different pipes");
            }
            pipe = current;
            phaseSlots[phase] = slot;
            phaseIds.try_emplace(phase, static_cast<uint32_t>(phaseIds.size()));
        }
        singleton &= alternatives.size() == 1;
        result.phaseAlternatives.push_back(alternatives);
        result.payloads.push_back({*pipe});
    }
    if (singleton) {
        for (const auto& alternatives : slots) { result.phases.push_back(alternatives.front()); }
    }
    for (const auto* phase : input.instructions()) {
        if (loop->isProperAncestor(phase->elementOp) && !members.contains(phase)) {
            return fail(CompactInputIssue::InvalidBinding, "compact body omitted a shared phase");
        }
    }
    return true;
}
bool Builder::prerequisites()
{
    auto function = loop->getParentOfType<func::FuncOp>();
    if (!function) { return fail(CompactInputIssue::InvalidBinding, "compact loop has no original function"); }
    function.walk([&](Operation* operation) {
        const auto targets = index.phasesFor(operation);
        const bool targetInside = llvm::any_of(targets, [&](const auto* phase) { return members.contains(phase); });
        for (const auto& edge : index.prerequisitesFor(operation)) {
            const bool interfaceInside = targets.empty() && loop->isProperAncestor(operation);
            if (members.contains(edge.producer) != targetInside || interfaceInside) {
                result.boundaryPrerequisites.push_back(edge);
            }
        }
    });
    if (bindings.prerequisites) {
        result.additional = bindings.prerequisites->demands;
        result.native = bindings.prerequisites->native;
    } else {
        if (index.hasRelevantCarriedState(loop)) {
            return fail(CompactInputIssue::PrerequisiteMapping, "compact carried prerequisites need supplied maps");
        }
        struct Requirement {
            std::set<std::pair<uint32_t, uint32_t>> nativePairs;
        };
        std::map<std::pair<uint32_t, uint32_t>, Requirement> requirements;
        for (uint32_t target = 0; target < slots.size(); ++target) {
            for (const auto* alternative : slots[target]) {
                for (const auto& edge : index.prerequisitesFor(alternative->elementOp)) {
                    auto source = phaseSlots.find(edge.producer);
                    if (source == phaseSlots.end()) { continue; }
                    if (source->second >= target) {
                        return fail(CompactInputIssue::PrerequisiteMapping,
                                    "compact internal prerequisites need forward occurrence maps");
                    }
                    auto& requirement = requirements[{source->second, target}];
                    if (edge.native) {
                        requirement.nativePairs.insert({phaseIds.lookup(edge.producer), phaseIds.lookup(alternative)});
                    }
                }
            }
        }
        for (const auto& [edge, requirement] : requirements) {
            // A native fact must hold for every alternative pair. Missing or
            // arm-specific facts remain software upper demands; they are never
            // promoted into the common native lower graph. No path enumeration.
            const auto sources = slots[edge.first].size(), targets = slots[edge.second].size();
            const auto covered = requirement.nativePairs.size();
            const bool universalNative = covered / sources == targets && covered % sources == 0;
            if (universalNative) { result.native.push_back({edge.first, edge.second, 0}); }
            else { result.additional.push_back({edge.first, edge.second, {true, 0, 0}}); }
        }
    }
    for (auto& edge : result.additional) {
        if (edge.source >= result.payloads.size() || edge.target >= result.payloads.size() || !interval(edge.bounds)) {
            return fail(CompactInputIssue::InvalidBinding, "invalid compact prerequisite distance binding");
        }
        edge.bounds.minimum = std::max(edge.bounds.minimum, uint64_t(edge.source >= edge.target));
        if (edge.bounds.maximum && edge.bounds.minimum > *edge.bounds.maximum) { edge.bounds.reachable = false; }
    }
    for (auto edge : result.native) {
        if (edge.source >= result.payloads.size() || edge.target >= result.payloads.size() ||
            (!edge.displacement && edge.source >= edge.target)) {
            return fail(CompactInputIssue::InvalidBinding, "invalid compact native prerequisite binding");
        }
    }
    return true;
}
bool Builder::collect()
{
    for (uint32_t site = 0; site < slots.size(); ++site) {
        for (const auto* phase : slots[site]) {
            for (auto id : model.effectsFor(phase)) {
                if (id >= model.effects().size() || !model.effects()[id].memory) {
                    return fail(CompactInputIssue::InvalidBinding, "invalid compact shared effect binding");
                }
                const auto& effect = model.effects()[id];
                if (effect.rangesMaterialized && effect.ranges.empty()) { continue; }
                if (effects.size() == UINT32_MAX) {
                    return fail(CompactInputIssue::InvalidBinding, "compact incidence count exceeds representation");
                }
                parents.push_back(parents.size()); effects.push_back(id); sites.push_back(site);
            }
        }
    }
    // Each original incidence belongs to one access. Across all hazards there
    // are at most n*n effect-pair checks plus n*(n-1)/2 class checks.
    if (!effects.empty() && effects.size() > UINT64_MAX / (2 * uint64_t(effects.size()))) {
        return fail(CompactInputIssue::InvalidBinding, "compact pair work counters exceed representation");
    }
    return true;
}
bool Builder::mayMeet(std::size_t a, std::size_t b)
{
    ++result.overlapQueries;
    return detail::compactEffectsMayMeet(model, a, b);
}
std::size_t Builder::root(std::size_t value)
{
    while (parents[value] != value) {
        parents[value] = parents[parents[value]]; value = parents[value];
    }
    return value;
}
bool Builder::classes()
{
    for (std::size_t a = 0; a < effects.size(); ++a) {
        for (std::size_t b = a + 1; b < effects.size(); ++b) {
            if (mayMeet(effects[a], effects[b])) { parents[root(b)] = root(a); }
        }
    }
    std::map<std::size_t, uint32_t> classes;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> accesses;
    for (std::size_t id = 0; id < effects.size(); ++id) {
        auto inserted = classes.emplace(root(id), static_cast<uint32_t>(classes.size()));
        const auto cell = inserted.first->second;
        if (inserted.second) { result.classEffects.emplace_back(); }
        result.classEffects[cell].push_back(effects[id]);
        auto entry = accesses.emplace(std::make_pair(sites[id], cell), static_cast<uint32_t>(result.accesses.size()));
        if (entry.second) {
            result.accesses.push_back({sites[id], cell, false, false, false}); result.accessEffects.emplace_back();
        }
        auto& access = result.accesses[entry.first->second];
        const auto mode = model.effects()[effects[id]].mode;
        access.read |= mode == SyncAccessMode::Read;
        access.write |= mode == SyncAccessMode::Write;
        result.accessEffects[entry.first->second].push_back(effects[id]);
    }
    return true;
}
uint64_t Builder::group(const SyncStorageEffect& effect) const
{
    return effect.memory->scope == AddressSpace::ACC ? protection.inLoop(effect.phase, loop) : 0;
}
std::optional<OriginDistanceInterval> Builder::pair(std::size_t a, std::size_t b, StorageHazard hazard)
{
    ++result.effectPairQueries;
    const auto& source = model.effects()[a]; const auto& target = model.effects()[b];
    const auto sourcePipe = static_cast<uint32_t>(source.phase->kPipeValue);
    const auto targetPipe = static_cast<uint32_t>(target.phase->kPipeValue);
    if (!mayMeet(a, b)) { return OriginDistanceInterval{}; }
    if (ptoStorageProtection().protectsScalar(sourcePipe, targetPipe)) {
        ++result.protectedPairs; return OriginDistanceInterval{};
    }
    const auto sg = group(source), tg = group(target);
    if ((sg & invocationProtectionBit) && (tg & invocationProtectionBit) &&
        hardwareProtectsConflict(sourcePipe, sg, targetPipe, tg)) {
        ++result.protectedPairs; return OriginDistanceInterval{};
    }
    OriginDistanceInterval bound{true, 0, std::nullopt};
    if (bindings.accessBounds) {
        ++result.suppliedBoundQueries;
        auto supplied = bindings.accessBounds(a, b, hazard);
        if (supplied) { bound = *supplied; }
        if (!interval(bound)) {
            fail(CompactInputIssue::InvalidBinding, "invalid supplied compact access distance interval");
            return std::nullopt;
        }
    }
    if (hardwareProtectsConflict(sourcePipe, sg & ~invocationProtectionBit,
                                 targetPipe, tg & ~invocationProtectionBit)) {
        ++result.visitProtectedPairs;
        bound.minimum = std::max(bound.minimum, uint64_t(1));
        if (bound.maximum && bound.minimum > *bound.maximum) { bound.reachable = false; }
    }
    return bound;
}
bool Builder::queries()
{
    for (uint32_t a = 0; a < result.accesses.size(); ++a) {
        for (uint32_t b = 0; b < result.accesses.size(); ++b) {
            if (result.accesses[a].storageClass != result.accesses[b].storageClass) { continue; }
            for (auto hazard : {StorageHazard::RAW, StorageHazard::WAR, StorageHazard::WAW}) {
                OriginDistanceInterval aggregate;
                for (auto x : result.accessEffects[a]) {
                    if ((model.effects()[x].mode == SyncAccessMode::Read) != (hazard == StorageHazard::WAR)) {
                        continue;
                    }
                    for (auto y : result.accessEffects[b]) {
                        if ((model.effects()[y].mode == SyncAccessMode::Read) != (hazard == StorageHazard::RAW)) {
                            continue;
                        }
                        auto bound = pair(x, y, hazard);
                        if (!bound) { return false; }
                        include(aggregate, *bound);
                    }
                }
                if (!aggregate.reachable) { continue; }
                if (result.queries.size() == UINT32_MAX) {
                    return fail(CompactInputIssue::InvalidBinding, "compact query count exceeds representation");
                }
                result.queries.push_back({a, b, hazard, aggregate});
            }
        }
    }
    return true;
}
bool Builder::run()
{
    return skeleton() && prerequisites() && collect() && classes() && queries();
}
} // namespace
bool detail::compactEffectsMayMeet(const SyncStorageEffects& model, std::size_t a, std::size_t b)
{
    if (a >= model.effects().size() || b >= model.effects().size()) { return true; }
    const auto& left = model.effects()[a]; const auto& right = model.effects()[b];
    if (!left.memory || !right.memory) { return true; }
    if (left.memory->scope != AddressSpace::Zero && right.memory->scope != AddressSpace::Zero &&
        left.memory->scope != right.memory->scope) { return false; }
    if (!uniformRanges(left) || !uniformRanges(right)) { return true; }
    return model.mayOverlap(a, b);
}
CompactWriterReaderInput detail::buildCompactWriterReaderSlotInput(scf::ForOp loop,
    const SyncInput& input, const PhaseIndex& index,
    llvm::ArrayRef<std::vector<const CompoundInstanceElement*>> slots,
    const CompactWriterReaderBindings& bindings)
{
    Builder builder(loop, input, index, slots, bindings);
    builder.run();
    return std::move(builder.result);
}
CompactWriterReaderInput buildCompactWriterReaderInput(scf::ForOp loop,
    const SyncInput& input, const PhaseIndex& index, const CompactWriterReaderBindings& bindings)
{
    CompactWriterReaderInput result;
    auto fail = [&](CompactInputIssue issue, const char* message) {
        result.issue = issue; result.error = message; return result;
    };
    if (!loop) { return fail(CompactInputIssue::InvalidBinding, "compact input requires an original loop"); }
    const auto contract = recognizeExplicit(*loop.getBody(), index, input.accesses());
    for (const auto& diagnostic : contract.diagnostics) {
        if (diagnostic.issue != RecognitionIssue::AdditionalPrerequisite) {
            return fail(CompactInputIssue::FixedSkeleton, "compact body has an unsupported shared leaf contract");
        }
    }
    auto phases = index.explicitSequence(*loop.getBody());
    if (failed(phases)) {
        return fail(CompactInputIssue::FixedSkeleton, "compact input needs a fixed executed single-phase body");
    }
    std::vector<std::vector<const CompoundInstanceElement*>> slots;
    for (const auto* phase : *phases) { slots.push_back({phase}); }
    return detail::buildCompactWriterReaderSlotInput(loop, input, index, slots, bindings);
}
} // namespace mlir::pto::frontiersynch
