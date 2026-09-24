// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "OriginalReadQueries.h"
#include "OriginalObligationAdapter.h"
#include <algorithm>
#include <iterator>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include "llvm/ADT/DenseSet.h"

namespace mlir::pto::frontiersynch {
namespace {
// Translate the existing D3 syntax slice into original cuts without changing
// its equations. This is a record adapter, not a new boundary search. Anonymous
// sequence wrappers contribute no anchor and do not widen an existing cut.
const Region* scopeRegion(const Region& region, std::size_t owner)
{
    if (region.kind != Region::Operation && region.originalOwner == owner) {
        return &region;
    }
    for (const auto& child : region.children) {
        if (const auto* found = scopeRegion(child, owner)) {
            return found;
        }
    }
    return nullptr;
}
std::optional<OriginalCut> subtreeBoundary(const Region& region, OriginalCut::Side side)
{
    if (region.kind == Region::Operation) {
        return OriginalCut{region.operation, side};
    }
    if (region.originalOwner != NoControlId) {
        return OriginalCut::scope(region.originalOwner, side);
    }
    for (std::size_t i = 0; i < region.children.size(); ++i) {
        const auto index = side == OriginalCut::Before ? i : region.children.size() - i - 1;
        if (auto cut = subtreeBoundary(region.children[index], side)) {
            return cut;
        }
    }
    return {};
}
std::optional<OriginalIntervalRequest> readerIntervalRequest(
    const OriginalStructure& original, const ReaderIntervalQuery& slice, OriginalIntervalRequest context)
{
    const auto* region = slice.owner == NoControlId ? &original.body : scopeRegion(original.body, slice.owner);
    if (!region) {
        return {};
    }
    context.selector.cell = slice.cell;
    context.selector.read = true;
    context.selector.write = false;
    context.selector.engine = unsigned(slice.reader);
    context.selector.physicalRelation = NoControlId;
    context.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
    context.occurrence.backedgeOwner = NoControlId;
    context.includeStoppingAccess = false;
    context.start = OriginalCut::scope(slice.owner, OriginalCut::Before);
    context.stop = OriginalCut::scope(slice.owner, OriginalCut::After);
    if (slice.scope == ReaderIntervalQuery::WholeRegion) {
        return context;
    }
    if (slice.scope != ReaderIntervalQuery::BodyInterval) {
        return {};
    }
    if (region->kind == Region::For && region->children.size() == 1) {
        context.start = OriginalCut::childBoundary(slice.owner, 0, OriginalCut::Before);
        context.stop = OriginalCut::childBoundary(slice.owner, 0, OriginalCut::After);
        region = &region->children.front();
    }
    if (region->kind != Region::Sequence) {
        return {};
    }
    const auto end = slice.end == NoControlId ? region->children.size() : slice.end;
    if (slice.begin > end || end > region->children.size()) {
        return {};
    }
    for (auto i = slice.begin; i > 0; --i) {
        if (auto cut = subtreeBoundary(region->children[i - 1], OriginalCut::After)) {
            context.start = *cut;
            break;
        }
    }
    for (auto i = end; i < region->children.size(); ++i) {
        if (auto cut = subtreeBoundary(region->children[i], OriginalCut::Before)) {
            context.stop = *cut;
            break;
        }
    }
    return context;
}
bool allRoleIncidencesUseRelation(
    const OriginalStructure& original, std::size_t operation, std::size_t cell, bool write, std::size_t relation)
{
    if (operation >= original.operations.size()) {
        return false;
    }
    bool hasRole = false;
    for (const auto& access : original.operations[operation].accesses) {
        if (access.cell != cell || (write ? !access.write : !access.read)) {
            continue;
        }
        hasRole = true;
        if (access.physicalRelation != relation) {
            return false;
        }
    }
    return hasRole;
}
} // namespace

struct ProgramAnalysis::Impl {
    std::unique_ptr<OriginalObligations> obligationModel = std::make_unique<OriginalObligations>();
    std::string obligationError = "original obligation universe not prepared";
    bool legacyRequirementIndexReady = false;
    OriginalReadQueries readers;
    std::vector<std::optional<PhysicalBankCorrespondence>> banks;
    using AlternativeKey = std::tuple<std::size_t, std::size_t, StorageRelationship::Kind>;
    std::map<AlternativeKey, std::vector<StorageOrigin>> alternatives;
    std::map<std::pair<unsigned, unsigned>, std::vector<OriginalRequirementId>> directions;
    std::vector<std::vector<TypedOriginalRequirement>> typedByDeadline;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> typedBySource;
    // Requirement IDs alone do not identify a continuation. All original
    // interval fields (including Unknown qualifications) participate in reuse.
    using QueryKey = std::tuple<std::size_t, std::size_t, OriginalInterval>;
    std::map<OriginalInterval, FixedVisitCorrespondence> fixedVisits;
    std::map<QueryKey, DecodedRequirement> cache;
    std::map<QueryKey, InterpretedRequirement> interpretations;
    std::map<std::pair<std::size_t, std::size_t>, OriginalAccessSummary> allUses;
    std::map<std::pair<std::size_t, std::size_t>, StorageLifecycle> lifecycles;
    std::map<std::tuple<std::size_t, std::size_t, unsigned>, std::vector<StorageOrigin>> readerUses;
    explicit Impl(const OriginalStructure& original, const OriginalValueQueries& values)
        : readers(original, &values),
          banks(original.physicalAddresses.size()),
          typedByDeadline(original.originalSites.size()),
          typedBySource(original.operations.size())
    {}
};

ProgramAnalysis::ProgramAnalysis(const SyncInput& source, OriginalStructure structure)
    : input(source),
      original(std::move(structure)),
      values(original),
      storage(original, &values),
      occurrenceQueries(original, &values),
      impl(std::make_unique<Impl>(original, values))
{
    if (!storage.complete()) {
        return;
    }
    DenseMap<Value, std::set<std::tuple<std::size_t, unsigned, std::size_t>>> typedSeen;
    auto collectTyped = [&](std::size_t target, Value required, TypedOriginalRequirement::Cause cause) {
        const auto qualification = values.qualify(required, {target, false});
        for (const auto& prerequisite : qualification.prerequisites) {
            const auto deadline = prerequisite.deadline.site;
            if (deadline >= impl->typedByDeadline.size() ||
                !typedSeen[prerequisite.producedValue]
                     .emplace(deadline, unsigned(cause), prerequisite.sourcePhase)
                     .second) {
                continue;
            }
            TypedOriginalRequirement request;
            request.cause = cause;
            request.source = {prerequisite.executableSourcePhase, SourceMilestone::After};
            request.deadlineOriginalSite = deadline;
            request.requiredValue = prerequisite.producedValue;
            request.sourceEngine = prerequisite.sourceEngine;
            request.sourceGapExecutable = values.legal(prerequisite.source);
            request.occurrenceQualified = qualification.executableAfterPrerequisites();
            request.availability = qualification.status;
            request.completion = prerequisite;
            request.qualificationObstructions = qualification.obstructions;
            const auto index = impl->typedByDeadline[deadline].size();
            impl->typedByDeadline[deadline].push_back(std::move(request));
            if (prerequisite.executableSourcePhase < impl->typedBySource.size()) {
                impl->typedBySource[prerequisite.executableSourcePhase].push_back({deadline, index});
            }
        }
        if (!qualification.executableAfterPrerequisites() &&
            typedSeen[required].emplace(target, unsigned(cause), NoControlId).second) {
            TypedOriginalRequirement request;
            request.cause = cause;
            request.deadlineOriginalSite = target;
            request.requiredValue = required;
            request.incomingOrUnknownSource = true;
            request.availability = qualification.status;
            request.qualificationObstructions = qualification.obstructions;
            impl->typedByDeadline[target].push_back(std::move(request));
        }
    };
    for (auto [site, operation] : llvm::enumerate(original.originalSites)) {
        if (auto choice = dyn_cast<scf::IfOp>(operation)) {
            collectTyped(site, choice.getCondition(), TypedOriginalRequirement::Cause::BranchCondition);
        } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            for (auto bound : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
                collectTyped(site, bound, TypedOriginalRequirement::Cause::LoopBound);
            }
        } else if (auto condition = dyn_cast<scf::ConditionOp>(operation)) {
            collectTyped(site, condition.getCondition(), TypedOriginalRequirement::Cause::WhileCondition);
        }
    }
    for (const auto& phase : original.operations) {
        llvm::DenseSet<Value> seenAddresses;
        SmallVector<Value> addresses;
        auto collectAddresses = [&](ArrayRef<const BaseMemInfo*> memories) {
            for (const auto* memory : memories) {
                if (!memory || !memory->rootBuffer) {
                    continue;
                }
                if (auto alloc = memory->rootBuffer.getDefiningOp<AllocTileOp>()) {
                    if (alloc.getAddr() && seenAddresses.insert(alloc.getAddr()).second) {
                        addresses.push_back(alloc.getAddr());
                    }
                }
            }
        };
        collectAddresses(phase.instruction->defVec);
        collectAddresses(phase.instruction->useVec);
        for (auto address : addresses) {
            collectTyped(phase.original, address, TypedOriginalRequirement::Cause::Address);
        }
    }
    impl->obligationModel = detail::buildOriginalObligations(
        original, storage, values,
        [this](std::size_t site) -> const std::vector<TypedOriginalRequirement>& { return typedRequirementsAt(site); });
    impl->obligationError = impl->obligationModel->complete() ? "" : "invalid original obligation universe";
}

ProgramAnalysis::~ProgramAnalysis() = default;
bool ProgramAnalysis::complete() const { return storage.complete() && impl->obligationModel->complete(); }
const std::string& ProgramAnalysis::reason() const
{
    return storage.complete() ? impl->obligationError : storage.reason();
}
const OriginalObligations& ProgramAnalysis::obligations() const { return *impl->obligationModel; }
const std::vector<OriginalObligationFamilyId>& ProgramAnalysis::obligationsAt(std::size_t originalSite) const
{
    return obligations().atOriginalSite(originalSite);
}
Value ProgramAnalysis::obligationGuardValue(std::size_t originalIfSite) const
{
    if (originalIfSite < original.originalSites.size()) {
        if (auto choice = dyn_cast<scf::IfOp>(original.originalSites[originalIfSite])) {
            return choice.getCondition();
        }
    }
    return {};
}
OriginalObligationWitness ProgramAnalysis::obligationWitness(OriginalObligationId id) const
{
    return obligationWitness(id.family, id.origin);
}
OriginalObligationWitness ProgramAnalysis::obligationWitness(
    OriginalObligationFamilyId id, ObligationOrigin source) const
{
    OriginalObligationWitness result;
    result.membership = obligations().witness(id, source);
    using Status = ObligationMembership::Status;
    if (result.membership.status == Status::Invalid || result.membership.status == Status::Excluded) {
        return result;
    }
    const auto* family = obligations().get(id);
    const auto& key = family->key;
    if (key.kind == OriginalObligationKey::Kind::Typed) {
        for (const auto& request : typedRequirementsAt(key.consumerOriginal)) {
            const auto value = reinterpret_cast<std::uintptr_t>(request.requiredValue.getAsOpaquePointer());
            if (value != key.typedValue || static_cast<std::size_t>(request.cause) != key.typedCause) {
                continue;
            }
            if (request.incomingOrUnknownSource == source.incoming &&
                (source.incoming || request.completion.sourcePhase == source.operation)) {
                result.typedRequirement = &request;
                break;
            }
        }
    } else {
        if (key.cell < original.cells.size()) {
            result.physicalCell = &original.cells[key.cell];
        }
        if (key.consumerOperation < original.operations.size()) {
            result.consumerOperation = &original.operations[key.consumerOperation];
        }
        result.consumerEffects = family->consumerEffects;
        if (!source.incoming && source.operation < original.operations.size()) {
            result.sourceEffects = &storage.effectIncidences(
                source.operation, key.cell, key.sourceRole() == OriginalObligationKey::Role::Writer);
        }
    }
    if (!source.incoming && source.operation < original.operations.size()) {
        result.sourceOperation = &original.operations[source.operation];
    }
    return result;
}
void ProgramAnalysis::ensureLegacyRequirementIndex() const
{
    if (!storage.complete() || impl->legacyRequirementIndexReady) {
        return;
    }
    for (std::size_t target = 0; target < original.operations.size(); ++target) {
        const auto& requests = storage.requirementsAt(target);
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto& request = requests[index];
            const auto& relation = request.relationship;
            impl->alternatives[{target, relation.cell, relation.kind}].push_back(relation.source);
            const auto source = relation.source.operation;
            if (source < original.operations.size()) {
                const auto from = unsigned(original.operations[source].instruction->kPipeValue);
                const auto to = unsigned(original.operations[target].instruction->kPipeValue);
                impl->directions[{from, to}].push_back({target, index});
            }
        }
    }
    impl->legacyRequirementIndexReady = true;
}
const PhysicalBankCorrespondence& ProgramAnalysis::bankRelation(std::size_t index) const
{
    static const PhysicalBankCorrespondence invalid;
    if (index >= impl->banks.size()) {
        return invalid;
    }
    if (!impl->banks[index]) {
        impl->banks[index] = occurrenceQueries.bank(index);
    }
    return *impl->banks[index];
}
const std::vector<StorageOrigin>& ProgramAnalysis::alternativeSourcesFor(const DecodedRequirement& requirement) const
{
    ensureLegacyRequirementIndex();
    static const std::vector<StorageOrigin> empty;
    const auto& relation = requirement.requirement.relationship;
    const auto key = Impl::AlternativeKey{relation.target.operation, relation.cell, relation.kind};
    const auto found = impl->alternatives.find(key);
    return found == impl->alternatives.end() ? empty : found->second;
}
const std::vector<OriginalRequirement>& ProgramAnalysis::requirementsAt(std::size_t operation) const
{
    return storage.requirementsAt(operation);
}
const std::vector<OriginalRequirementId>& ProgramAnalysis::requirementsFromTo(
    PipelineType source, PipelineType target) const
{
    ensureLegacyRequirementIndex();
    static const std::vector<OriginalRequirementId> empty;
    const auto found = impl->directions.find({unsigned(source), unsigned(target)});
    return found == impl->directions.end() ? empty : found->second;
}
const std::vector<TypedOriginalRequirement>& ProgramAnalysis::typedRequirementsAt(std::size_t originalSite) const
{
    static const std::vector<TypedOriginalRequirement> empty;
    return originalSite < impl->typedByDeadline.size() ? impl->typedByDeadline[originalSite] : empty;
}
const std::vector<std::pair<std::size_t, std::size_t>>& ProgramAnalysis::typedSubscriptionsAt(
    std::size_t sourceOperation) const
{
    static const std::vector<std::pair<std::size_t, std::size_t>> empty;
    return sourceOperation < impl->typedBySource.size() ? impl->typedBySource[sourceOperation] : empty;
}
OriginalIntervalRequest ProgramAnalysis::intervalFor(std::size_t target, std::size_t index) const
{
    OriginalIntervalRequest request;
    request.version = original.version;
    const auto& requirements = storage.requirementsAt(target);
    if (index >= requirements.size()) {
        return request;
    }
    const auto& requirement = requirements[index];
    const auto& relation = requirement.relationship;
    request.selector.cell = relation.cell;
    request.selector.read = relation.kind == StorageRelationship::RAW;
    request.selector.write = !request.selector.read;
    request.occurrence.source = relation.source.operation;
    request.occurrence.target = relation.target.operation;
    request.start = requirement.source;
    request.stop = requirement.deadline;
    return request;
}
OriginalIntervalResult ProgramAnalysis::prepareInterval(OriginalIntervalRequest request) const
{
    return storage.prepareInterval(std::move(request));
}
const DecodedRequirement& ProgramAnalysis::decodeAt(std::size_t target, std::size_t index) const
{
    return decodeAt(target, index, intervalFor(target, index));
}
const DecodedRequirement& ProgramAnalysis::decodeAt(
    std::size_t target, std::size_t index, OriginalIntervalRequest continuation) const
{
    const auto& requirements = storage.requirementsAt(target);
    if (!continuation.version) {
        continuation.version = original.version;
    }
    const auto prepared = prepareInterval(continuation);
    auto identity = prepared.interval;
    identity.query = continuation; // Keep the full request even on an invalid preparation.
    const auto key = Impl::QueryKey{target, index, std::move(identity)};
    const auto prior = impl->cache.find(key);
    if (prior != impl->cache.end()) {
        return prior->second;
    }
    DecodedRequirement answer;
    answer.continuation = continuation;
    const bool invalidIdentity = !complete() || index >= requirements.size();
    if (invalidIdentity) {
        answer.unresolved.emplace_back("invalid original requirement identity");
        return impl->cache.emplace(key, std::move(answer)).first->second;
    }
    const auto& requirement = requirements[index];
    answer.requirement = requirement;
    const auto source = requirement.relationship.source.operation;
    const auto cell = requirement.relationship.cell;
    if (source >= original.operations.size()) {
        answer.unresolved.emplace_back("source has no original operation");
        return impl->cache.emplace(key, std::move(answer)).first->second;
    }
    const bool sameRoles = continuation.selector.cell == cell && continuation.occurrence.source == source &&
                           continuation.occurrence.target == target;
    if (!sameRoles || !prepared.valid) {
        answer.unresolved.push_back(sameRoles ? prepared.reason : "continuation names different requirement roles");
        return impl->cache.emplace(key, std::move(answer)).first->second;
    }
    answer.owner = prepared.interval.owner;
    answer.interval = prepared.interval;
    answer.requirement.interval = prepared.interval;
    auto collectBanks = [&](std::size_t operation, bool write, std::vector<std::size_t>& out) {
        for (const auto& access : original.operations[operation].accesses) {
            if (access.cell != cell || access.physicalRelation == NoControlId ||
                (write ? !access.write : !access.read)) {
                continue;
            }
            out.push_back(access.physicalRelation);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    };
    const bool sourceWrites = requirement.relationship.kind != StorageRelationship::WAR;
    const bool targetWrites = requirement.relationship.kind != StorageRelationship::RAW;
    collectBanks(source, sourceWrites, answer.sourceBankRelations);
    collectBanks(target, targetWrites, answer.targetBankRelations);
    const auto defaultInterval = intervalFor(target, index);
    const bool defaultCuts = continuation.start == defaultInterval.start && continuation.stop == defaultInterval.stop &&
                             continuation.includeStoppingAccess == defaultInterval.includeStoppingAccess &&
                             continuation.selector == defaultInterval.selector &&
                             continuation.occurrence == defaultInterval.occurrence &&
                             continuation.continuationOwner == defaultInterval.continuationOwner;
    if (!defaultCuts) {
        answer.unresolved.emplace_back("reader-segment adapter does not qualify this explicit cut interval");
    }
    if (defaultCuts && requirement.relationship.kind != StorageRelationship::WAW) {
        const auto reader = requirement.relationship.kind == StorageRelationship::RAW ? target : source;
        const auto pipe = original.operations[reader].instruction->kPipeValue;
        const auto& segment = impl->readers.segment(answer.owner, reader, cell, pipe);
        if (segment.complete) {
            answer.readers = impl->readers.query(segment.interval);
            answer.hasReaderFrontier = true;
        } else {
            ReaderIntervalQuery whole;
            whole.owner = answer.owner;
            whole.cell = cell;
            whole.reader = pipe;
            answer.readers = impl->readers.query(whole);
            answer.hasReaderFrontier = answer.readers.complete();
        }
        if (answer.hasReaderFrontier) {
            if (auto request = readerIntervalRequest(original, answer.readers.interval, continuation)) {
                const auto interval = prepareInterval(std::move(*request));
                if (interval.valid) {
                    answer.readerInterval = interval.interval;
                } else {
                    answer.unresolved.push_back(interval.reason);
                }
            } else {
                answer.unresolved.emplace_back("reader syntax slice has no original cut interpretation");
            }
            if (answer.readers.status == OriginalReaderFrontiers::Status::Unknown) {
                answer.unresolved.push_back(answer.readers.reason);
            }
            if (answer.readers.status == OriginalReaderFrontiers::Status::NoHit) {
                answer.unresolved.emplace_back("qualified region has no participating reader for this requirement");
            }
            if (answer.readers.status == OriginalReaderFrontiers::Status::Exact &&
                !answer.readers.guardsAvailableAtReadSites) {
                answer.unresolved.emplace_back("reader guard is unavailable at an original read site");
            }
        } else {
            answer.unresolved.push_back(segment.reason);
            if (!answer.readers.reason.empty()) {
                answer.unresolved.push_back(answer.readers.reason);
            }
        }
    }
    if (requirement.relationship.kind == StorageRelationship::WAW) {
        answer.unresolved.emplace_back("content generation requires a scoped support query");
    }
    return impl->cache.emplace(key, std::move(answer)).first->second;
}
const InterpretedRequirement& ProgramAnalysis::interpretAt(std::size_t target, std::size_t index) const
{
    return interpretAt(target, index, intervalFor(target, index));
}
const InterpretedRequirement& ProgramAnalysis::interpretAt(
    std::size_t target, std::size_t index, OriginalIntervalRequest continuation) const
{
    if (!continuation.version) {
        continuation.version = original.version;
    }
    const auto prepared = prepareInterval(continuation);
    auto identity = prepared.interval;
    identity.query = continuation; // Keep the full request even on an invalid preparation.
    const auto key = Impl::QueryKey{target, index, std::move(identity)};
    const auto prior = impl->interpretations.find(key);
    if (prior != impl->interpretations.end()) {
        return prior->second;
    }
    InterpretedRequirement answer;
    answer.decoded = &decodeAt(target, index, continuation);
    answer.queryInterval = answer.decoded->interval;
    answer.occurrence.interval = answer.queryInterval;
    if (index >= storage.requirementsAt(target).size()) {
        return impl->interpretations.emplace(key, std::move(answer)).first->second;
    }
    const auto fixedVisit = fixedVisitFor(*answer.decoded);
    answer.occurrence.owner = answer.decoded->owner;
    answer.occurrence.source = answer.decoded->requirement.relationship.source.operation;
    answer.occurrence.target = answer.decoded->requirement.relationship.target.operation;
    std::set_intersection(
        answer.decoded->sourceBankRelations.begin(), answer.decoded->sourceBankRelations.end(),
        answer.decoded->targetBankRelations.begin(), answer.decoded->targetBankRelations.end(),
        std::back_inserter(answer.occurrence.sharedBankCandidates));
    if (fixedVisit.exact) {
        answer.occurrence.status = OriginalOccurrenceInterpretation::Status::FixedVisit;
    } else if (
        answer.occurrence.source == answer.occurrence.target && answer.occurrence.sharedBankCandidates.size() == 1) {
        const auto candidate = answer.occurrence.sharedBankCandidates.front();
        const auto& bank = bankRelation(candidate);
        const auto& relation = answer.decoded->requirement.relationship;
        const bool sourceWrites = relation.kind != StorageRelationship::WAR;
        const bool targetWrites = relation.kind != StorageRelationship::RAW;
        const bool covered =
            allRoleIncidencesUseRelation(original, answer.occurrence.source, relation.cell, sourceWrites, candidate) &&
            allRoleIncidencesUseRelation(original, answer.occurrence.target, relation.cell, targetWrites, candidate);
        const bool oneMatchingUse = bank.participatingOperations.size() == 1 &&
                                    bank.participatingOperations.front() == answer.occurrence.source;
        const bool localContext =
            answer.queryInterval && answer.queryInterval->owner == bank.owner &&
            continuation.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::Unqualified &&
            continuation.occurrence.incomingInterface == NoControlId && continuation == intervalFor(target, index);
        if (covered && bank.exactPermutation && oneMatchingUse && localContext) {
            answer.occurrence.status = OriginalOccurrenceInterpretation::Status::PeriodicSameRole;
            answer.occurrence.owner = bank.owner;
            answer.occurrence.period = bank.distance;
            answer.occurrence.firstOrdinalWithLocalPredecessor = bank.distance;
            answer.occurrence.earlierOrdinalsNeedIncomingCase = true;
        }
    }
    if (answer.occurrence.status == OriginalOccurrenceInterpretation::Status::Unknown) {
        answer.occurrence.reason = answer.occurrence.sharedBankCandidates.empty() ?
                                       fixedVisit.reason :
                                       "physical-bank candidate lacks qualified predecessor and child correspondence";
    }
    const auto& relation = answer.decoded->requirement.relationship;
    auto lifecycle = [&](std::size_t operation) -> const StorageLifecycle* {
        const auto use = std::make_pair(operation, relation.cell);
        auto found = impl->lifecycles.find(use);
        if (found == impl->lifecycles.end()) {
            found = impl->lifecycles.emplace(use, storage.lifecycleAt(operation, relation.cell)).first;
        }
        return &found->second;
    };
    answer.sourceUse = lifecycle(relation.source.operation);
    answer.targetUse = lifecycle(relation.target.operation);
    answer.mayHaveIncomingGeneration = answer.targetUse->mayHaveNoPriorFullWrite;
    answer.interval.owner = answer.decoded->owner;
    answer.interval.cell = relation.cell;
    answer.interval.start = continuation.start;
    answer.interval.stop = continuation.stop;
    answer.interval.read = continuation.selector.read;
    answer.interval.write = continuation.selector.write;
    answer.support.cell = relation.cell;
    if (relation.kind == StorageRelationship::WAW) {
        answer.support.producer = relation.source.operation;
        answer.support.reuse = relation.target.operation;
        answer.support.hasWriteDelimitedCandidate = true;
    } else if (relation.kind == StorageRelationship::RAW && answer.targetUse->nextWriters.size() == 1) {
        answer.support.producer = relation.source.operation;
        answer.support.reuse = answer.targetUse->nextWriters.front().operation;
        answer.support.hasWriteDelimitedCandidate = true;
    } else if (relation.kind == StorageRelationship::WAR && answer.sourceUse->previousWriters.size() == 1) {
        answer.support.producer = answer.sourceUse->previousWriters.front().operation;
        answer.support.reuse = relation.target.operation;
        answer.support.hasWriteDelimitedCandidate = true;
    } else {
        answer.support.reason = "producer or next conflicting write is not unique";
    }
    answer.firstConflict = firstConflict(*answer.decoded);
    answer.lastRelevantUse = lastRelevantUse(*answer.decoded);
    answer.mayAlternativeSources = &alternativeSourcesFor(*answer.decoded);
    answer.factoredUse = &storage.factoredAt(target, relation.cell);
    auto hazard = FactoredUseNode::Hazard::RAW;
    switch (relation.kind) {
        case StorageRelationship::RAW:
            hazard = FactoredUseNode::Hazard::RAW;
            break;
        case StorageRelationship::WAR:
            hazard = FactoredUseNode::Hazard::WAR;
            break;
        case StorageRelationship::WAW:
            hazard = FactoredUseNode::Hazard::WAW;
            break;
    }
    answer.factoredDemand = answer.factoredUse->requirement(target, hazard);
    // The origin group is a may-set. D1 path guards and the possible absence of
    // an incoming full writer remain explicit in targetUse and are not inferred
    // from this vector's cardinality.
    return impl->interpretations.emplace(key, std::move(answer)).first->second;
}
FixedVisitCorrespondence ProgramAnalysis::fixedVisitFor(const DecodedRequirement& requirement) const
{
    const auto& relation = requirement.requirement.relationship;
    if (!requirement.interval || requirement.interval->query.version != original.version ||
        requirement.interval->query.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge) {
        FixedVisitCorrespondence unknown;
        unknown.source = relation.source.operation;
        unknown.target = relation.target.operation;
        unknown.reason = "fixed-visit query lacks its original interval interpretation";
        return unknown;
    }
    const auto& key = *requirement.interval;
    const auto prior = impl->fixedVisits.find(key);
    if (prior != impl->fixedVisits.end()) {
        return prior->second;
    }
    auto result = occurrenceQueries.fixedVisit(relation.source.operation, relation.target.operation, relation.cell);
    return impl->fixedVisits.emplace(key, std::move(result)).first->second;
}
OriginalBoundaryResult ProgramAnalysis::boundary(const DecodedRequirement& requirement, bool first) const
{
    static const std::vector<StorageOrigin> empty;
    OriginalBoundaryResult result;
    result.mayAccesses = &empty;
    const auto& relation = requirement.requirement.relationship;
    result.owner = requirement.owner;
    result.cell = relation.cell;
    result.interval = requirement.readerInterval;
    if (result.interval) {
        result.owner = result.interval->owner;
    }
    const auto key = std::make_pair(result.owner, result.cell);
    auto found = impl->allUses.find(key);
    if (found == impl->allUses.end()) {
        found = impl->allUses.emplace(key, storage.all(result.owner, result.cell)).first;
    }
    const auto& allUses = found->second;
    if (relation.kind == StorageRelationship::WAW) {
        result.mayAccesses = &allUses.writers;
        result.reason = "write frontier needs a qualified occurrence and episode interval";
        return result;
    }
    if (first && relation.kind == StorageRelationship::WAR) {
        result.mayAccesses = &allUses.writers;
        result.reason = "first conflicting write needs a qualified occurrence interval";
        return result;
    }
    const auto reader =
        relation.kind == StorageRelationship::RAW ? relation.target.operation : relation.source.operation;
    if (reader >= original.operations.size()) {
        result.reason = "reader has no original operation";
        return result;
    }
    const auto pipe = original.operations[reader].instruction->kPipeValue;
    const auto readerKey = std::make_tuple(result.owner, result.cell, unsigned(pipe));
    auto readerUses = impl->readerUses.find(readerKey);
    if (readerUses == impl->readerUses.end()) {
        std::vector<StorageOrigin> matching;
        for (const auto& use : allUses.readers) {
            if (original.operations[use.operation].instruction->kPipeValue == pipe) {
                matching.push_back(use);
            }
        }
        readerUses = impl->readerUses.emplace(readerKey, std::move(matching)).first;
    }
    result.mayAccesses = &readerUses->second;
    if (!requirement.hasReaderFrontier || !requirement.readerInterval) {
        result.reason = "no qualified original reader cut interval";
        return result;
    }
    const auto& readers = requirement.readers;
    if (readers.status == OriginalReaderFrontiers::Status::Unknown) {
        result.reason = readers.reason;
        return result;
    }
    result.nonempty = readers.nonempty;
    result.frontier = first ? readers.first : readers.last;
    result.endpointQualification.status = OriginalValueQualification::Status::Available;
    for (const auto& [operation, guard] : impl->readers.guardedAccesses(result.frontier)) {
        result.endpointQualification = OriginalValueQueries::combine(
            std::move(result.endpointQualification), impl->readers.qualificationAt(guard, operation, !first));
    }
    result.guardsAvailableAtReadSites = result.endpointQualification.available();
    if (!result.endpointQualification.executableAfterPrerequisites()) {
        result.reason = result.endpointQualification.reason();
        return result;
    }
    if (!requirement.requirement.episodeKnown || !requirement.requirement.occurrenceKnown) {
        result.reason = "structural reader frontier lacks a qualified episode and occurrence interval";
        return result;
    }
    result.status = readers.status == OriginalReaderFrontiers::Status::NoHit ? OriginalBoundaryResult::Status::NoHit :
                                                                               OriginalBoundaryResult::Status::Exact;
    if (result.status == OriginalBoundaryResult::Status::Exact) {
        for (const auto& [operation, guard] : impl->readers.guardedAccesses(result.frontier)) {
            const OriginalCut cut{operation, first ? OriginalCut::Before : OriginalCut::After};
            if (guard == 0 || !resolveOriginalCut(original, cut)) {
                result.status = OriginalBoundaryResult::Status::Unknown;
                result.reason = "exact access frontier has no executable original cut";
                result.cuts.clear();
                return result;
            }
            result.cuts.push_back(cut);
        }
    }
    return result;
}
OriginalBoundaryResult ProgramAnalysis::firstConflict(const DecodedRequirement& requirement) const
{
    return boundary(requirement, true);
}
OriginalBoundaryResult ProgramAnalysis::lastRelevantUse(const DecodedRequirement& requirement) const
{
    return boundary(requirement, false);
}
PhysicalUseFrontier ProgramAnalysis::firstMayUse(const OriginalUseQuery& query) const
{
    return storage.firstUse(query);
}
PhysicalUseFrontier ProgramAnalysis::lastMayUse(const OriginalUseQuery& query) const { return storage.lastUse(query); }
PhysicalUseFrontier ProgramAnalysis::firstMayUse(const OriginalInterval& query) const
{
    return storage.firstUse(query);
}
PhysicalUseFrontier ProgramAnalysis::lastMayUse(const OriginalInterval& query) const { return storage.lastUse(query); }
const std::vector<SourceSubscription>& ProgramAnalysis::subscriptionsAt(std::size_t operation) const
{
    return storage.subscriptionsAt(operation);
}
OriginalAccessSummary ProgramAnalysis::all(const OriginalAllQuery& query) const
{
    const auto key = std::make_pair(query.owner, query.cell);
    auto found = impl->allUses.find(key);
    if (found == impl->allUses.end()) {
        found = impl->allUses.emplace(key, storage.all(query.owner, query.cell)).first;
    }
    auto result = found->second;
    if (!result.complete) {
        return result;
    }
    auto filter = [&](std::vector<StorageOrigin>& uses, bool enabled) {
        if (!enabled) {
            uses.clear();
            return;
        }
        if (query.engine) {
            uses.erase(
                std::remove_if(
                    uses.begin(), uses.end(),
                    [&](const StorageOrigin& use) {
                        return use.operation >= original.operations.size() ||
                               original.operations[use.operation].instruction->kPipeValue != *query.engine;
                    }),
                uses.end());
        }
    };
    filter(result.readers, query.read);
    filter(result.writers, query.write);
    return result;
}
OriginalMayAfter ProgramAnalysis::mayAfter(const OriginalContinuationQuery& query) const
{
    return storage.mayAfter(query);
}
OriginalMayAfter ProgramAnalysis::mayAfter(const OriginalInterval& query) const { return storage.mayAfter(query); }
OriginalIntervalResult ProgramAnalysis::supportInterval(const InterpretedRequirement& requirement) const
{
    if (!requirement.queryInterval || !requirement.support.hasWriteDelimitedCandidate) {
        OriginalIntervalResult result;
        result.reason = "missing original support interval";
        return result;
    }
    auto request = requirement.queryInterval->query;
    request.selector.cell = requirement.support.cell;
    request.selector.read = request.selector.write = true;
    request.selector.engine.reset();
    request.selector.physicalRelation = NoControlId;
    request.occurrence.source = requirement.support.producer;
    request.occurrence.target = requirement.support.reuse;
    request.start = {requirement.support.producer, OriginalCut::After};
    request.stop = {requirement.support.reuse, OriginalCut::Before};
    request.includeStoppingAccess = false;
    return prepareInterval(std::move(request));
}
OriginalSupportInterval ProgramAnalysis::qualifySupport(const InterpretedRequirement& requirement) const
{
    if (!requirement.support.hasWriteDelimitedCandidate) {
        OriginalSupportInterval result;
        result.reason = requirement.support.reason;
        return result;
    }
    const auto interval = supportInterval(requirement);
    if (!interval.valid) {
        OriginalSupportInterval result;
        result.reason = interval.reason;
        return result;
    }
    return storage.supportBetween(interval.interval);
}
OriginalBoundaryResult ProgramAnalysis::qualifiedBoundary(
    const InterpretedRequirement& requirement, const OriginalSupportInterval& support, bool first) const
{
    if (!requirement.decoded) {
        OriginalBoundaryResult unknown;
        unknown.reason = "missing original requirement";
        return unknown;
    }
    auto result = boundary(*requirement.decoded, first);
    if (!support.complete || !support.generationEstablished ||
        requirement.occurrence.status != OriginalOccurrenceInterpretation::Status::FixedVisit ||
        !requirement.decoded->hasReaderFrontier) {
        result.reason = "reader frontier lacks qualified generation or occurrence support";
        return result;
    }
    const auto& relation = requirement.decoded->requirement.relationship;
    const auto reader =
        relation.kind == StorageRelationship::RAW ? relation.target.operation : relation.source.operation;
    const bool participates =
        llvm::any_of(support.readers, [&](const StorageOrigin& use) { return use.operation == reader; });
    const auto expected = supportInterval(requirement);
    const bool sameInterval = expected.valid && support.interval && *support.interval == expected.interval;
    const auto& readInterval = requirement.decoded->readerInterval;
    // A first/last answer about an entire lexical owner must not be relabelled
    // as the answer about a smaller producer-to-reuse interval. Broader adapters
    // belong to the exact-frontier step, not to cache-key normalization.
    const bool sameReaderSpan =
        readInterval && support.interval && readInterval->owner == support.interval->owner &&
        readInterval->query.version == support.interval->query.version &&
        readInterval->query.start == support.interval->query.start &&
        readInterval->query.stop == support.interval->query.stop &&
        readInterval->query.includeStoppingAccess == support.interval->query.includeStoppingAccess &&
        readInterval->query.continuationOwner == support.interval->query.continuationOwner &&
        readInterval->query.occurrence.incomingInterface == support.interval->query.occurrence.incomingInterface &&
        !support.cases.backedge;
    if (!participates || !sameInterval || !sameReaderSpan) {
        result.reason = "reader summary and qualified support do not name the same original interval";
        return result;
    }
    auto qualified = *requirement.decoded;
    qualified.requirement.occurrenceKnown = true;
    qualified.requirement.episodeKnown = true;
    return boundary(qualified, first);
}
std::vector<OriginalEndpointCandidate> ProgramAnalysis::endpointCandidates(
    const OriginalBoundaryResult& boundary, SourceMilestone::Side side) const
{
    std::vector<OriginalEndpointCandidate> result;
    if (boundary.status != OriginalBoundaryResult::Status::Exact || !boundary.interval ||
        boundary.interval->query.version != original.version) {
        return result;
    }
    for (const auto& [operation, guard] : impl->readers.guardedAccesses(boundary.frontier)) {
        const bool invalidEndpoint = operation >= original.operations.size() || guard == 0;
        if (invalidEndpoint) {
            return {};
        }
        const OriginalCut cut{operation, side};
        if (!resolveOriginalCut(original, cut)) {
            return {};
        }
        auto qualification = guardQualificationAt(guard, operation, side);
        result.push_back({cut, guard, qualification.available(), true, std::move(qualification)});
    }
    return result;
}
ParticipationExpression ProgramAnalysis::predicate(std::size_t id) const { return impl->readers.predicate(id); }
GuardedReadFrontier ProgramAnalysis::readerFrontier(std::size_t id) const { return impl->readers.frontier(id); }
bool ProgramAnalysis::guardAvailableAt(std::size_t predicateId, std::size_t operation) const
{
    return impl->readers.availableAt(predicateId, operation);
}
OriginalValueQualification ProgramAnalysis::guardQualificationAt(
    std::size_t predicateId, std::size_t operation, SourceMilestone::Side side) const
{
    return impl->readers.qualificationAt(predicateId, operation, side == SourceMilestone::After);
}
std::vector<OriginalParticipationDemand> ProgramAnalysis::participationDemands() const
{
    return impl->readers.participationDemands();
}
} // namespace mlir::pto::frontiersynch
