// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "Control.h"
#include "ControlComponents.h"
#include "D2Periodic.h"
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
void indexOwners(const Region& region, std::map<std::size_t, const Region*>& owners)
{
    if (region.kind != Region::Operation && region.originalOwner != NoControlId) {
        owners.emplace(region.originalOwner, &region);
    }
    for (const auto& child : region.children) {
        indexOwners(child, owners);
    }
}
} // namespace
struct OccurrenceQueries::Impl {
    std::unique_ptr<OriginalValueQueries> ownedValues;
    const OriginalValueQueries& values;
    std::unique_ptr<OriginalLifetimes> ownedStorage;
    const OriginalLifetimes& storage;
    mutable std::map<std::tuple<std::size_t, std::size_t, FactoredUseNode::Hazard>,
                     std::shared_ptr<const FixedVisitSourceFrontier>> fixedSources;
    std::vector<std::vector<std::size_t>> relationUses;
    std::map<std::size_t, const Region*> owners;
    mutable std::map<std::size_t, PhysicalBankCorrespondence> banks;
    using PeriodicKey = std::tuple<OriginalInterval, bool, bool>;
    mutable std::map<PeriodicKey, PeriodicUseCorrespondence> periodicUses;
    explicit Impl(
        const OriginalStructure& original, const OriginalValueQueries* shared, const OriginalLifetimes* sharedStorage)
        : ownedValues(shared ? nullptr : std::make_unique<OriginalValueQueries>(original)),
          values(shared ? *shared : *ownedValues),
          ownedStorage(sharedStorage ? nullptr : std::make_unique<OriginalLifetimes>(original, &values)),
          storage(sharedStorage ? *sharedStorage : *ownedStorage)
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
    }
};
OccurrenceQueries::OccurrenceQueries(
    const OriginalStructure& original, const OriginalValueQueries* values, const OriginalLifetimes* storage)
    : original(original), impl(std::make_unique<Impl>(original, values, storage))
{}
OccurrenceQueries::~OccurrenceQueries() = default;

PhysicalBankCorrespondence OccurrenceQueries::bank(std::size_t relationId) const
{
    if (!impl->values.current()) {
        return d2_detail::bank(original, impl->values, relationId);
    }
    auto found = impl->banks.find(relationId);
    if (found == impl->banks.end()) {
        found = impl->banks.emplace(relationId, d2_detail::bank(original, impl->values, relationId)).first;
    }
    return found->second;
}
PeriodicUseCorrespondence OccurrenceQueries::periodic(
    const OriginalInterval& interval, bool sourceWrites, bool targetWrites) const
{
    if (!impl->values.current() || interval.query.version != original.version) {
        PeriodicUseCorrespondence result;
        result.interval = interval;
        result.reason = "original program changed; rebuild periodic correspondence";
        return result;
    }
    const Impl::PeriodicKey key{interval, sourceWrites, targetWrites};
    auto found = impl->periodicUses.find(key);
    if (found == impl->periodicUses.end()) {
        found = impl->periodicUses.emplace(
            key, d2_detail::periodic(original, impl->values, interval, sourceWrites, targetWrites,
                                    [this](std::size_t relation) { return bank(relation); })).first;
    }
    return found->second;
}
std::shared_ptr<const FixedVisitSourceFrontier> OccurrenceQueries::fixedSourcesAt(
    std::size_t target, std::size_t cell, FactoredUseNode::Hazard hazard) const
{
    auto result = std::make_shared<FixedVisitSourceFrontier>();
    if (!impl->values.current()) {
        result->sources.reason = "original program changed; rebuild occurrence queries";
        return result;
    }
    const auto key = std::make_tuple(target, cell, hazard);
    const auto old = impl->fixedSources.find(key);
    if (old != impl->fixedSources.end()) {
        return old->second;
    }
    if (target >= original.operations.size() || cell >= original.cells.size()) {
        result->sources.reason = "invalid D1 target or physical cell";
        return result;
    }
    // Share the same old-state provenance used by the obligation universe.
    const auto& use = impl->storage.factoredAt(target, cell);
    result->sources = queryFixedVisitSources(use, target, hazard);
    if (!result->sources.complete) {
        return impl->fixedSources.emplace(key, std::move(result)).first->second;
    }
    auto obstruct = [](const std::string& reason) {
        OriginalValueQualification q;
        q.status = OriginalValueQualification::Status::Unresolved;
        q.obstructions.push_back(reason);
        return q;
    };
    auto qualify = [&](std::size_t condition, std::size_t phase, bool after) {
        const auto cut = impl->values.phaseCut(phase, after);
        OriginalValueQualification q;
        q.status = OriginalValueQualification::Status::Available;
        if (!impl->values.legal(cut) ||
            !resolveOriginalCut(original, {phase, after ? OriginalCut::After : OriginalCut::Before})) {
            q.status = OriginalValueQualification::Status::NotObservableHere;
            q.obstructions.emplace_back("D1 endpoint has no executable immediate original cut");
            return q;
        }
        std::set<std::size_t> seen;
        std::vector<std::size_t> todo{condition};
        while (!todo.empty()) {
            const auto id = todo.back();
            todo.pop_back();
            if (!seen.insert(id).second) {
                continue;
            }
            const auto& n = (*result->sources.predicates)[id];
            if (n.kind == FactoredUseNode::Kind::Test) {
                if (!n.guard.value) {
                    q = OriginalValueQueries::combine(std::move(q), obstruct("D1 guard has no original value"));
                    continue;
                }
                auto value = Value::getFromOpaquePointer(reinterpret_cast<void*>(n.guard.value));
                const auto identity = impl->values.identity(value);
                if (!identity.value || identity.occurrenceScope != n.guard.scope) {
                    q = OriginalValueQueries::combine(std::move(q), obstruct("D1 guard occurrence identity changed"));
                } else {
                    // This service retains independently dischargeable prerequisites.
                    // No packet is guarded circularly by its not-yet-available value.
                    q = OriginalValueQueries::combine(std::move(q), impl->values.qualify(identity.value, cut));
                }
            } else if (n.kind == FactoredUseNode::Kind::Choose) {
                todo.push_back(n.condition);
                todo.push_back(n.left);
                todo.push_back(n.right);
            }
        }
        return q;
    };
    auto fixedPhysicalRole = [&](std::size_t operation, bool write) {
        if (operation >= original.operations.size()) {
            return false;
        }
        bool found = false;
        for (const auto& access : original.operations[operation].accesses) {
            if (access.cell != cell || (write ? !access.write : !access.read)) {
                continue;
            }
            found = true;
            if (access.physicalRelation == NoControlId) {
                continue;
            }
            if (access.physicalRelation >= original.physicalAddresses.size()) {
                return false;
            }
            const auto& addresses = original.physicalAddresses[access.physicalRelation].addresses;
            if (addresses.empty() || addresses.front().size() != 1 ||
                !llvm::all_of(addresses, [&](const auto& a) { return a == addresses.front(); })) {
                return false; // D2 must qualify varying physical selections separately.
            }
        }
        return found;
    };
    const bool targetWrites = hazard != FactoredUseNode::Hazard::RAW;
    const bool sourceWrites = hazard != FactoredUseNode::Hazard::WAR;
    result->physicalRolesQualified = !original.cells[cell].unknownRange &&
                                    original.cells[cell].storage != Cell::Storage::OverlapWitness &&
                                    fixedPhysicalRole(target, targetWrites);
    result->localGuardsQualified = true;
    for (const auto& source : result->sources.alternatives) {
        FixedVisitEndpointPair pair;
        pair.source = source.operation;
        pair.incoming = source.incoming;
        pair.sourceCut = {source.operation, OriginalCut::After};
        pair.targetCut = {target, OriginalCut::Before};
        pair.sourceCondition = source.sourceCondition;
        pair.targetCondition = source.targetCondition;
        // Never use the source's qualification as the target's qualification.
        pair.targetQualification = qualify(source.targetCondition, target, false);
        if (source.incoming) {
            result->hasIncoming = true;
            pair.sourceQualification = obstruct("D1 incoming case needs its enclosing source interface");
            pair.sourceCut = OriginalCut::scope(use.frame.owner, OriginalCut::Before);
        } else {
            pair.sourceQualification = qualify(source.sourceCondition, source.operation, true);
            result->physicalRolesQualified &= fixedPhysicalRole(source.operation, sourceWrites);
            result->localGuardsQualified &= pair.sourceQualification.executableAfterPrerequisites();
        }
        result->localGuardsQualified &= pair.targetQualification.executableAfterPrerequisites();
        result->endpoints.push_back(std::move(pair));
    }
    return impl->fixedSources.emplace(key, std::move(result)).first->second;
}
FixedVisitCorrespondence OccurrenceQueries::fixedVisit(
    std::size_t source, std::size_t target, std::size_t cell, FactoredUseNode::Hazard hazard) const
{
    FixedVisitCorrespondence result;
    result.source = source;
    result.target = target;
    result.alternatives = fixedSourcesAt(target, cell, hazard);
    const auto& frontier = *result.alternatives;
    result.reason = frontier.sources.reason;
    if (!frontier.sources.complete) {
        return result;
    }
    const auto pair = std::find_if(frontier.endpoints.begin(), frontier.endpoints.end(), [&](const auto& p) {
        return !p.incoming && p.source == source;
    });
    if (pair == frontier.endpoints.end() || source == target) {
        result.reason = "no applicable local D1 origin on the target's original paths";
        return result;
    }
    result.sourceQualification = pair->sourceQualification;
    result.targetQualification = pair->targetQualification;
    result.sourceCondition = pair->sourceCondition;
    result.targetCondition = pair->targetCondition;
    if (!frontier.physicalRolesQualified) {
        result.reason = "D1 requires fixed qualified physical effects; may-footprints remain obligations";
    } else if (!frontier.sources.independentReaders && !frontier.sources.exclusiveWriters) {
        result.reason = "D1 cannot make simultaneous partial-write origins exclusive";
    } else if (!result.sourceQualification.executableAfterPrerequisites()) {
        result.reason = result.sourceQualification.reason();
    } else if (!result.targetQualification.executableAfterPrerequisites()) {
        result.reason = result.targetQualification.reason();
    } else {
        // Conditional on these two ORIGINAL predicates, in this one fixed-use
        // frame. Incoming alternatives retain their own separate interface case.
        result.exact = true;
        result.reason.clear();
    }
    return result;
}
FixedVisitCorrespondence OccurrenceQueries::fixedVisit(std::size_t source, std::size_t target, std::size_t cell) const
{
    // Compatibility query. New consumers supply the actual hazard explicitly.
    FixedVisitCorrespondence result;
    result.source = source;
    result.target = target;
    result.reason = "source/target have no qualified D1 conflict";
    for (auto hazard : {FactoredUseNode::Hazard::RAW, FactoredUseNode::Hazard::WAR, FactoredUseNode::Hazard::WAW}) {
        auto candidate = fixedVisit(source, target, cell, hazard);
        if (candidate.exact) {
            return candidate;
        }
        if (candidate.alternatives->sources.complete) {
            result = std::move(candidate);
        }
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
