// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "Control.h"
#include "ControlComponents.h"
#include <algorithm>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
void collectOperations(const Region& region, std::vector<std::size_t>& out)
{
    if (region.kind == Region::Operation) {
        out.push_back(region.operation);
    }
    for (const auto& child : region.children) {
        collectOperations(child, out);
    }
}
const Region* findOwner(const Region& region, std::size_t owner)
{
    if (region.originalOwner == owner) {
        return &region;
    }
    for (const auto& child : region.children) {
        if (auto* found = findOwner(child, owner)) {
            return found;
        }
    }
    return nullptr;
}
void collectMandatoryUses(const Region& region, std::set<std::size_t>& operations)
{
    if (region.kind == Region::Operation) {
        operations.insert(region.operation);
        return;
    }
    if (region.kind == Region::Choice || region.kind == Region::While || region.kind == Region::For) {
        return;
    }
    for (const auto& child : region.children) {
        collectMandatoryUses(child, operations);
    }
}
void indexOwners(const Region& region, std::map<std::size_t, const Region*>& owners)
{
    if (region.kind != Region::Operation && region.originalOwner != NoControlId) {
        owners.emplace(region.originalOwner, &region);
    }
    for (const auto& child : region.children) {
        indexOwners(child, owners);
    }
}
bool separated(const PhysicalAddressRelation& relation)
{
    const auto size = relation.memory->allocateSize;
    if (!size || relation.addresses.size() < 2) {
        return false;
    }
    // D2 needs one physical bank per residue. A finite may-address set still
    // supplies useful footprint information, but no unique predecessor bank.
    if (!llvm::all_of(relation.addresses, [](const auto& addresses) { return addresses.size() == 1; })) {
        return false;
    }
    for (std::size_t i = 0; i < relation.addresses.size(); ++i) {
        for (std::size_t j = i + 1; j < relation.addresses.size(); ++j) {
            for (auto a : relation.addresses[i]) {
                for (auto b : relation.addresses[j]) {
                    if (a <= b ? b - a < size : a - b < size) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
} // namespace
struct OccurrenceQueries::Impl {
    std::unique_ptr<OriginalValueQueries> ownedValues;
    const OriginalValueQueries& values;
    detail::ControlGraph control;
    std::vector<bool> reachable;
    std::vector<std::vector<std::size_t>> predecessors, components;
    std::vector<std::size_t> membership;
    std::vector<std::vector<std::size_t>> relationUses;
    std::map<std::size_t, const Region*> owners;
    mutable std::map<std::size_t, std::set<std::size_t>> mandatoryByOwner;
    explicit Impl(const OriginalStructure& original, const OriginalValueQueries* shared)
        : ownedValues(shared ? nullptr : std::make_unique<OriginalValueQueries>(original)),
          values(shared ? *shared : *ownedValues),
          control(detail::buildControlGraph(original))
    {
        relationUses.resize(original.physicalAddresses.size());
        indexOwners(original.body, owners);
        for (std::size_t operation = 0; operation < original.operations.size(); ++operation) {
            for (const auto& access : original.operations[operation].accesses) {
                if (access.physicalRelation >= relationUses.size()) {
                    continue;
                }
                auto& uses = relationUses[access.physicalRelation];
                const bool newOperation = uses.empty() || uses.back() != operation;
                if (newOperation) {
                    uses.push_back(operation);
                }
            }
        }
        reachable = detail::reachableSites(control);
        predecessors.resize(control.sites.size());
        for (std::size_t source = 0; source < control.sites.size(); ++source) {
            for (auto target : control.sites[source].successors) {
                predecessors[target].push_back(source);
            }
        }
        components = detail::strongComponents(control, predecessors, reachable, membership);
    }
    bool cyclic(std::size_t site) const
    {
        const auto group = membership[site];
        if (group == NoControlId) {
            return true;
        }
        return components[group].size() > 1 || llvm::is_contained(control.sites[site].successors, site);
    }
};
OccurrenceQueries::OccurrenceQueries(const OriginalStructure& original, const OriginalValueQueries* values)
    : original(original), impl(std::make_unique<Impl>(original, values))
{}
OccurrenceQueries::~OccurrenceQueries() = default;

PhysicalBankCorrespondence OccurrenceQueries::bank(std::size_t relationId) const
{
    PhysicalBankCorrespondence result;
    if (!impl->values.current()) {
        result.reason = "original program changed; rebuild occurrence queries";
        return result;
    }
    if (relationId >= original.physicalAddresses.size()) {
        result.reason = "invalid physical address relation";
        return result;
    }
    const auto& relation = original.physicalAddresses[relationId];
    result.owner = relation.owner;
    result.memory = relation.memory;
    result.reason = "bank relation lacks a qualified physical permutation";
    if (!relation.memory || relation.memory->scope == AddressSpace::GM ||
        relation.memory->scope == AddressSpace::Zero || !separated(relation)) {
        return result;
    }
    const auto foundOwner = impl->owners.find(relation.owner);
    const auto* owner = foundOwner == impl->owners.end() ? nullptr : foundOwner->second;
    if (!owner || owner->kind != Region::For || owner->children.size() != 1 ||
        relation.owner >= original.originalSites.size()) {
        return result;
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(original.originalSites[relation.owner]);
    if (!loop) {
        return result;
    }
    result.endpointQualification = impl->values.counted(loop);
    auto mandatory = impl->mandatoryByOwner.find(relation.owner);
    if (mandatory == impl->mandatoryByOwner.end()) {
        std::set<std::size_t> uses;
        collectMandatoryUses(owner->children.front(), uses);
        mandatory = impl->mandatoryByOwner.emplace(relation.owner, std::move(uses)).first;
    }
    for (auto operation : impl->relationUses[relationId]) {
        if (!mandatory->second.count(operation)) {
            result.participatingOperations.clear();
            result.reason = "bank use is not proved to participate in every visit";
            return result;
        }
        result.participatingOperations.push_back(operation);
        const auto cut = impl->values.before(original.operations[operation].instruction->elementOp);
        result.endpointQualification = OriginalValueQueries::combine(
            std::move(result.endpointQualification), impl->values.qualify(relation.address, cut));
    }
    if (!result.endpointQualification.executableAfterPrerequisites()) {
        result.reason = result.endpointQualification.reason();
        return result;
    }
    if (result.participatingOperations.empty()) {
        result.reason = "no original access uses the physical selector";
        return result;
    }
    result.exactPermutation = true;
    result.distance = relation.addresses.size();
    result.reason.clear();
    return result;
}
FixedVisitCorrespondence OccurrenceQueries::fixedVisit(std::size_t source, std::size_t target, std::size_t cell) const
{
    FixedVisitCorrespondence result;
    if (!impl->values.current()) {
        result.reason = "original program changed; rebuild occurrence queries";
        return result;
    }
    result.source = source;
    result.target = target;
    result.reason = "source/target are not unique original visits";
    const auto& g = impl->control;
    if (source >= original.operations.size() || target >= original.operations.size() || cell >= original.cells.size() ||
        source == target || !impl->reachable[source] || !impl->reachable[target] || impl->cyclic(source) ||
        impl->cyclic(target)) {
        return result;
    }
    // A source on every path to the target has a stable original predecessor.
    std::vector<bool> seen(g.sites.size());
    std::vector<std::size_t> pending{g.entry};
    seen[g.entry] = true;
    while (!pending.empty()) {
        auto at = pending.back();
        pending.pop_back();
        if (at == target) {
            return result;
        }
        if (at == source) {
            continue;
        }
        for (auto next : g.sites[at].successors) {
            if (!seen[next]) {
                seen[next] = true;
                pending.push_back(next);
            }
        }
    }
    result.reason = "no source-to-target path or an intervening physical access";
    seen.assign(g.sites.size(), false);
    pending.assign(g.sites[source].successors.begin(), g.sites[source].successors.end());
    bool reached = false;
    while (!pending.empty()) {
        auto at = pending.back();
        pending.pop_back();
        if (seen[at]) {
            continue;
        }
        seen[at] = true;
        if (at == target) {
            reached = true;
            continue;
        }
        if (at < original.operations.size()) {
            for (const auto& access : original.operations[at].accesses) {
                if (access.cell == cell) {
                    return result;
                }
            }
        }
        pending.insert(pending.end(), g.sites[at].successors.begin(), g.sites[at].successors.end());
    }
    result.exact = reached;
    if (reached) {
        result.reason.clear();
    }
    return result;
}
std::vector<ChildOccurrence> OccurrenceQueries::children(std::size_t owner) const
{
    std::vector<ChildOccurrence> result;
    const auto* region = findOwner(original.body, owner);
    if (!region) {
        return result;
    }
    for (auto [position, child] : llvm::enumerate(region->children)) {
        ChildOccurrence occurrence;
        occurrence.position = position;
        occurrence.owner = owner;
        occurrence.child = child.originalOwner;
        occurrence.canSkip = child.kind == Region::Choice || (child.kind == Region::For && child.zeroTripPossible) ||
                             child.kind == Region::While;
        occurrence.canRepeat = child.kind == Region::For || child.kind == Region::While;
        collectOperations(child, occurrence.operations);
        result.push_back(std::move(occurrence));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
