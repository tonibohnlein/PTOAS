// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_D2PERIODIC_H
#define PTO_FRONTIERSYNCH_D2PERIODIC_H

#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "PTO/Transforms/FrontierSynch/SyncSlotMapping.h"
#include <functional>
#include <iterator>
#include <map>
#include <set>

namespace mlir::pto::frontiersynch::d2_detail {

inline const Region* findScope(const Region& region, std::size_t owner)
{
    if (region.kind != Region::Operation && region.originalOwner == owner) {
        return &region;
    }
    for (const auto& child : region.children) {
        if (auto* found = findScope(child, owner)) {
            return found;
        }
    }
    return nullptr;
}
inline void collect(const Region& region, bool mandatory, std::vector<std::pair<std::size_t, bool>>& operations)
{
    if (region.kind == Region::Operation) {
        operations.emplace_back(region.operation, mandatory);
        return;
    }
    // A nested repeated/optional participant requires D3/D4 qualification. It
    // is retained for interference, not replaced by one convenient visit.
    mandatory &= region.kind == Region::Sequence;
    for (const auto& child : region.children) {
        collect(child, mandatory, operations);
    }
}
inline bool overlap(const periodic_uses::Bank& a, const periodic_uses::Bank& b)
{
    return a.begin <= b.begin ? b.begin - a.begin < a.bytes : a.begin - b.begin < b.bytes;
}

inline PhysicalBankCorrespondence bank(
    const OriginalStructure& original, const OriginalValueQueries& values, std::size_t relationId)
{
    PhysicalBankCorrespondence result;
    result.version = original.version;
    result.relation = relationId;
    auto reject = [&](const std::string& reason) {
        result.exactPermutation = false;
        result.reason = reason;
        return result;
    };
    if (!values.current() || relationId >= original.physicalAddresses.size()) {
        return reject("stale or invalid original physical relation");
    }
    const auto& relation = original.physicalAddresses[relationId];
    result.owner = relation.owner;
    result.memory = relation.memory;
    if (!relation.memory || !relation.memory->allocateSize || relation.memory->aliasesUnknownRange ||
        relation.memory->scope == AddressSpace::GM || relation.memory->scope == AddressSpace::Zero ||
        relation.memory->baseAddresses.size() != 1 || relation.addresses.empty() ||
        relation.addresses.size() > std::numeric_limits<unsigned>::max()) {
        return reject("physical selector lacks one qualified footprint per represented state");
    }
    const auto* scope = findScope(original.body, relation.owner);
    auto loop = relation.owner < original.originalSites.size() ?
                    dyn_cast_or_null<scf::ForOp>(original.originalSites[relation.owner]) : scf::ForOp{};
    if (!scope || scope->kind != Region::For || scope->children.size() != 1 || !loop) {
        return reject("physical selector has no counted original owner");
    }
    // Recheck the ORIGINAL dependency-sliced scalar state. The bound is the
    // supplied explicit relation population, not a guessed trip count or bank
    // count. A repeated address alone does not close that dependency state.
    SyncSlotMapping::AnalysisContext scalarFacts;
    auto mapping = SyncSlotMapping::derive(loop, relation.address, scalarFacts, unsigned(relation.addresses.size()));
    if (!mapping || mapping->values.size() != relation.addresses.size()) {
        return reject("original scalar transition does not certify the represented cycle");
    }
    std::vector<periodic_uses::Bank> banks;
    std::vector<std::size_t> next;
    for (std::size_t i = 0; i < relation.addresses.size(); ++i) {
        const auto address = SyncSlotMapping::evaluate(relation.address, mapping->values[i]);
        const auto offset = relation.memory->baseAddresses.front();
        if (!address || offset > std::numeric_limits<uint64_t>::max() - *address ||
            relation.addresses[i].size() != 1 || relation.addresses[i].front() != *address + offset) {
            return reject("physical bank is not the qualified original address plus its effect offset");
        }
        banks.push_back({relation.addresses[i].front(), relation.memory->allocateSize});
        next.push_back((i + 1) % relation.addresses.size());
    }
    result.permutation = periodic_uses::certify(std::move(banks), std::move(next));
    if (!result.permutation.exact) {
        return reject(result.permutation.reason);
    }
    // Bank IDs are local to this independent selector; there is no LCM with
    // selectors used by other physical storage families.
    result.phaseBanks.resize(relation.addresses.size());
    std::iota(result.phaseBanks.begin(), result.phaseBanks.end(), 0);
    result.endpointQualification = values.counted(loop);
    std::vector<std::pair<std::size_t, bool>> operations;
    collect(scope->children.front(), true, operations);
    for (auto [operation, mandatory] : operations) {
        if (operation >= original.operations.size()) {
            return reject("invalid original role operation");
        }
        const auto& phase = original.operations[operation];
        bool uses = false;
        for (const auto& access : phase.accesses) {
            uses |= access.physicalRelation == relationId;
        }
        if (!uses) {
            continue;
        }
        if (!phase.instruction || !phase.instruction->elementOp) {
            return reject("bank incidence has no original operation");
        }
        result.participatingOperations.push_back(operation);
        if (!mandatory) {
            result.conditionalOrRepeatedOperations.push_back(operation);
        }
        result.endpointQualification = OriginalValueQueries::combine(
            std::move(result.endpointQualification),
            values.qualify(relation.address, values.before(phase.instruction->elementOp)));
    }
    if (result.participatingOperations.empty()) {
        return reject("no original access uses this physical selector");
    }
    if (!result.endpointQualification.executableAfterPrerequisites()) {
        return reject(result.endpointQualification.reason());
    }
    result.distance = result.permutation.cycleLength.front();
    result.exactPermutation = true;
    return result;
}

// A role-specific stream retains its original scalar value as an endpoint
// observation. Equal physical streams can have different SSA/iter-arg spellings.
struct Stream {
    std::vector<periodic_uses::Bank> sequence;
    Value address;
    uint64_t offset = 0;
    std::size_t relation = NoControlId;
    const BaseMemInfo* memory = nullptr;
};
inline std::optional<Stream> stream(
    const OriginalStructure& original, const Access& access, std::size_t owner,
    const std::function<PhysicalBankCorrespondence(std::size_t)>& getBank)
{
    if (!access.memory || access.cell >= original.cells.size()) {
        return {};
    }
    const auto& memory = *access.memory;
    if (access.physicalRelation != NoControlId) {
        if (access.physicalRelation >= original.physicalAddresses.size()) {
            return {};
        }
        const auto proof = getBank(access.physicalRelation);
        if (!proof.exactPermutation || proof.owner != owner || proof.memory != access.memory) {
            return {};
        }
        Stream result;
        result.address = original.physicalAddresses[access.physicalRelation].address;
        result.offset = memory.baseAddresses.front();
        result.relation = access.physicalRelation;
        result.memory = access.memory;
        for (auto bank : proof.phaseBanks) {
            result.sequence.push_back(proof.permutation.banks[bank]);
        }
        return result;
    }
    // The shared address importer may already expose an explicit finite set of
    // absolute addresses. Derive its ORIGINAL scalar transition within that
    // population too; do not require an importer-created relation ID. Constant
    // addresses are the identity permutation of one bank. Direct allocations
    // avoid guessing whether a subview's recorded address is absolute or relative.
    auto alloc = memory.rootBuffer ? memory.rootBuffer.getDefiningOp<AllocTileOp>() : AllocTileOp{};
    auto loop = owner < original.originalSites.size() ?
                    dyn_cast_or_null<scf::ForOp>(original.originalSites[owner]) : scf::ForOp{};
    if (!alloc || !alloc.getAddr() || !loop || !memory.hasKnownPhysicalAddresses || memory.aliasesUnknownRange ||
        memory.baseAddresses.empty() || memory.baseAddresses.size() > std::numeric_limits<unsigned>::max() ||
        !memory.allocateSize || memory.scope == AddressSpace::GM || memory.scope == AddressSpace::Zero ||
        memory.baseBuffer != memory.rootBuffer) {
        return {};
    }
    SyncSlotMapping::AnalysisContext scalarFacts;
    auto mapping = SyncSlotMapping::derive(loop, alloc.getAddr(), scalarFacts, unsigned(memory.baseAddresses.size()));
    if (!mapping) {
        return {};
    }
    Stream result;
    result.address = alloc.getAddr();
    result.memory = access.memory;
    std::vector<std::size_t> next;
    for (auto& state : mapping->values) {
        const auto address = SyncSlotMapping::evaluate(alloc.getAddr(), state);
        if (!address || !llvm::is_contained(memory.baseAddresses, *address)) {
            return {};
        }
        result.sequence.push_back({*address, memory.allocateSize});
        next.push_back((next.size() + 1) % mapping->values.size());
    }
    if (!periodic_uses::certify(result.sequence, std::move(next)).exact) {
        return {};
    }
    return result;
}

// May-footprint filtering needs neither exact occurrence matching nor a common
// selector period. Failure to identify a footprint means possible interference.
inline bool mayOverlap(const OriginalStructure& original, const Access& access,
                       AddressSpace domain, const std::vector<periodic_uses::Bank>& banks)
{
    if (!access.memory || access.memory->scope == AddressSpace::Zero) {
        return true;
    }
    const auto& memory = *access.memory;
    if (memory.scope != domain) {
        return false;
    }
    if (memory.aliasesUnknownRange || !memory.allocateSize) {
        return true;
    }
    std::vector<uint64_t> addresses;
    if (access.physicalRelation < original.physicalAddresses.size()) {
        const auto& relation = original.physicalAddresses[access.physicalRelation];
        if (relation.memory != access.memory) {
            return true;
        }
        for (const auto& footprint : relation.addresses) {
            addresses.insert(addresses.end(), footprint.begin(), footprint.end());
        }
    } else if (memory.hasKnownPhysicalAddresses) {
        addresses.assign(memory.baseAddresses.begin(), memory.baseAddresses.end());
    } else {
        return true;
    }
    if (addresses.empty()) {
        return true;
    }
    for (auto begin : addresses) {
        if (memory.allocateSize > std::numeric_limits<uint64_t>::max() - begin) {
            return true;
        }
        // banks is the sorted, disjoint query population, not a product of
        // selector states. Only the next interval and its predecessor can hit.
        auto at = std::lower_bound(banks.begin(), banks.end(), begin,
                                   [](const auto& bank, uint64_t address) { return bank.begin < address; });
        if ((at != banks.end() && overlap({begin, memory.allocateSize}, *at)) ||
            (at != banks.begin() && overlap({begin, memory.allocateSize}, *std::prev(at)))) {
            return true;
        }
    }
    return false;
}

inline PeriodicEndpointDomain endpoint(
    const OriginalStructure& original, const OriginalValueQueries& values, std::size_t owner,
    const Stream& stream, OriginalCut cut, periodic_uses::Domain domain)
{
    PeriodicEndpointDomain result;
    result.cut = cut;
    result.domain = domain;
    const auto valueCut = values.phaseCut(cut.operation, cut.side == OriginalCut::After);
    if (!resolveOriginalCut(original, cut) || !values.legal(valueCut) || domain.phase >= stream.sequence.size()) {
        result.selectorQualification.status = OriginalValueQualification::Status::Unresolved;
        result.selectorQualification.obstructions.push_back("periodic endpoint has no realizable original cut");
        return result;
    }
    result.testsSelector = stream.sequence.size() > 1;
    result.selector = stream.address;
    result.selectorEquals = stream.sequence[domain.phase].begin - stream.offset;
    // Even an identity selection checks its actual original value. This keeps
    // exact value availability separate from physical correspondence.
    result.selectorQualification = values.qualify(stream.address, valueCut);
    auto loop = dyn_cast_or_null<scf::ForOp>(original.originalSites[owner]);
    result.boundaryQualification = values.counted(loop);
    using Boundary = periodic_uses::Domain::Boundary;
    if (!domain.distance) {
        result.empty = domain.boundary == Boundary::Initial || domain.boundary == Boundary::Final;
        return result;
    }
    if (domain.boundary == Boundary::Any) {
        return result;
    }
    const bool previous = domain.boundary == Boundary::HasPrevious || domain.boundary == Boundary::Initial;
    const bool positive = domain.boundary == Boundary::HasPrevious || domain.boundary == Boundary::HasNext;
    result.boundaryTest = ObservationAtom{
        previous ? ObservationAtom::LoopHasPrevious : ObservationAtom::LoopHasNext,
        owner, domain.distance, uint64_t(positive)};
    result.boundaryQualification = values.atom(*result.boundaryTest, valueCut);
    return result;
}

inline PeriodicUseCorrespondence periodic(
    const OriginalStructure& original, const OriginalValueQueries& values, const OriginalInterval& interval,
    bool sourceWrites, bool targetWrites,
    const std::function<PhysicalBankCorrespondence(std::size_t)>& getBank)
{
    PeriodicUseCorrespondence result;
    result.interval = interval;
    result.owner = interval.owner;
    result.source = interval.query.occurrence.source;
    result.target = interval.query.occurrence.target;
    result.sourceWrites = sourceWrites;
    result.targetWrites = targetWrites;
    auto reject = [&](const std::string& reason) {
        result.exact = false;
        result.reason = reason;
        return result;
    };
    const auto& query = interval.query;
    if (!values.current() || query.version != original.version) {
        return reject("original program changed; rebuild periodic correspondence");
    }
    if (result.source >= original.operations.size() || result.target >= original.operations.size() ||
        query.selector.cell >= original.cells.size() || (!sourceWrites && !targetWrites)) {
        return reject("invalid original periodic conflict roles");
    }
    const auto* scope = findScope(original.body, interval.owner);
    auto loop = interval.owner < original.originalSites.size() ?
                    dyn_cast_or_null<scf::ForOp>(original.originalSites[interval.owner]) : scf::ForOp{};
    if (!scope || scope->kind != Region::For || scope->children.size() != 1 || !loop) {
        return reject("D2 requires one counted owner; enclosing transport needs D4");
    }
    // This query's domains describe the local nearest source/target roles, not
    // first-static-stop or arbitrary partial continuation semantics from step 2.
    if (query.start != OriginalCut{result.source, OriginalCut::After} ||
        query.stop != OriginalCut{result.target, OriginalCut::Before} || query.includeStoppingAccess ||
        query.occurrence.stopVisit != OriginalOccurrenceContext::StopVisit::Unqualified ||
        query.occurrence.incomingInterface != NoControlId || query.occurrence.backedgeOwner != NoControlId ||
        query.occurrence.qualification != NoControlId || query.selector.qualification != NoControlId ||
        !query.selector.predicateDependencies.empty() ||
        (query.continuationOwner && *query.continuationOwner != interval.owner)) {
        return reject("D2 local interval differs from the requested continuation; qualified D4 transport is required");
    }
    const auto& source = original.operations[result.source];
    const auto& target = original.operations[result.target];
    if (!source.instruction || !target.instruction ||
        (query.selector.engine && *query.selector.engine != unsigned(target.instruction->kPipeValue)) ||
        (targetWrites ? !query.selector.write : !query.selector.read)) {
        return reject("periodic target selector does not include the requested effect role");
    }
    std::map<std::pair<uintptr_t, std::size_t>, std::optional<Stream>> streams;
    auto getStream = [&](const Access& access) -> const std::optional<Stream>& {
        const auto key = std::make_pair(reinterpret_cast<uintptr_t>(access.memory), access.physicalRelation);
        auto found = streams.find(key);
        if (found == streams.end()) {
            found = streams.emplace(key, stream(original, access, interval.owner, getBank)).first;
        }
        return found->second;
    };
    std::optional<Stream> from, to;
    auto selectedStream = [&](const PhysicalOperation& op, bool write, std::optional<Stream>& out) {
        for (const auto& access : op.accesses) {
            if (access.cell != query.selector.cell || (write ? !access.write : !access.read)) {
                continue;
            }
            const auto& candidate = getStream(access);
            if (!candidate || (out && (candidate->sequence != out->sequence ||
                                      candidate->memory->scope != out->memory->scope))) {
                return false;
            }
            if (!out) {
                out = candidate;
            }
        }
        return bool(out);
    };
    if (!selectedStream(source, sourceWrites, from) || !selectedStream(target, targetWrites, to) ||
        from->memory->scope != to->memory->scope) {
        return reject("periodic hazard incidences lack matching qualified physical streams");
    }
    if (query.selector.physicalRelation != NoControlId && query.selector.physicalRelation != to->relation) {
        return reject("periodic target effect differs from the requested physical selector");
    }
    result.banks = from->sequence;
    std::vector<std::size_t> next(result.banks.size());
    for (std::size_t i = 0; i < next.size(); ++i) {
        next[i] = (i + 1) % next.size();
    }
    const auto permutation = periodic_uses::certify(result.banks, std::move(next));
    if (!permutation.exact) {
        return reject(permutation.reason);
    }
    std::map<std::pair<uint64_t, uint64_t>, std::size_t> bankIds;
    for (std::size_t b = 0; b < result.banks.size(); ++b) {
        bankIds.emplace(std::make_pair(result.banks[b].begin, result.banks[b].bytes), b);
    }
    std::vector<bool> selectedBanks(result.banks.size(), true);
    std::vector<periodic_uses::Bank> queryBanks;
    const auto& cell = original.cells[query.selector.cell];
    for (std::size_t b = 0; b < result.banks.size(); ++b) {
        if (cell.storage == Cell::Storage::CanonicalInterval && !cell.unknownRange) {
            selectedBanks[b] = false;
            for (const auto& range : cell.ranges) {
                selectedBanks[b] = selectedBanks[b] || overlap(result.banks[b], {range.first, range.second});
            }
        }
        if (selectedBanks[b]) {
            queryBanks.push_back(result.banks[b]);
        }
    }
    if (queryBanks.empty()) {
        return reject("queried physical cell has no matching periodic bank");
    }
    std::sort(queryBanks.begin(), queryBanks.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
    std::vector<std::pair<std::size_t, bool>> operations;
    collect(scope->children.front(), true, operations);
    std::vector<periodic_uses::Role> roles;
    std::size_t sourceRole = NoControlId, targetRole = NoControlId;
    std::set<std::size_t> seenOperations;
    for (auto [operation, mandatory] : operations) {
        if (operation >= original.operations.size() || !seenOperations.insert(operation).second) {
            return reject("original role word has invalid or duplicated static operations");
        }
        periodic_uses::Role role;
        role.operation = operation;
        for (const auto& access : original.operations[operation].accesses) {
            if ((!access.read && !access.write) || !mayOverlap(original, access, from->memory->scope, queryBanks)) {
                continue;
            }
            if (!mandatory) {
                return reject("interfering optional/repeated child access needs qualified participation or D4 transport");
            }
            const auto& candidate = getStream(access);
            if (!candidate || candidate->sequence.size() != result.banks.size()) {
                return reject("interfering physical access lacks the same explicit permutation population");
            }
            std::vector<std::size_t> selection;
            for (const auto& bank : candidate->sequence) {
                auto found = bankIds.find({bank.begin, bank.bytes});
                if (found == bankIds.end()) {
                    return reject("interfering access partially overlaps or selects different physical banks");
                }
                selection.push_back(found->second);
            }
            if (!role.selection.empty() && role.selection != selection) {
                return reject("one role has different overlapping physical selector incidences");
            }
            role.selection = std::move(selection);
            role.read |= access.read;
            role.write |= access.write;
            result.witnesses.push_back({operation, candidate->relation, access.memory, access.read, access.write});
        }
        if (role.selection.empty()) {
            continue;
        }
        if (operation == result.source) {
            sourceRole = roles.size();
        }
        if (operation == result.target) {
            targetRole = roles.size();
        }
        roles.push_back(std::move(role));
    }
    const auto paired = periodic_uses::relate(permutation, roles, sourceRole, targetRole, sourceWrites, targetWrites, selectedBanks);
    if (!paired.exact) {
        return reject(paired.reason);
    }
    result.entry = OriginalCut::scope(interval.owner, OriginalCut::Before);
    result.exit = OriginalCut::scope(interval.owner, OriginalCut::After);
    result.bypassTest = {ObservationAtom::LoopNonEmpty, interval.owner, 0, 0};
    result.bypassQualification = values.atom(result.bypassTest, values.before(loop.getOperation()));
    if (!resolveOriginalCut(original, result.entry) || !resolveOriginalCut(original, result.exit) ||
        !result.bypassQualification.executableAfterPrerequisites()) {
        return reject("periodic entry/exit or zero-trip predicate lacks original qualification");
    }
    for (const auto& link : paired.links) {
        PeriodicBankLink record;
        record.occurrence = link;
        record.predecessor = endpoint(original, values, interval.owner, *to, query.stop, link.predecessor());
        record.initial = endpoint(original, values, interval.owner, *to, query.stop, link.initial());
        record.successor = endpoint(original, values, interval.owner, *from, query.start, link.successor());
        record.final = endpoint(original, values, interval.owner, *from, query.start, link.final());
        const bool qualified = record.predecessor.qualified() && record.initial.qualified() &&
                               record.successor.qualified() && record.final.qualified();
        result.links.push_back(std::move(record));
        if (!qualified) {
            return reject("periodic endpoint value/cut/domain has an unresolved qualification");
        }
    }
    if (result.links.empty()) {
        return reject("queried physical cell has no matching periodic bank");
    }
    result.exact = true;
    return result;
}
} // namespace mlir::pto::frontiersynch::d2_detail
#endif
