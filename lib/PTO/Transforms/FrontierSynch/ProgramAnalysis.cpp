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
#include "OriginalExactQueries.h"
#include "OriginalObligationAdapter.h"
#include "OriginalRequestAdapter.h"
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
} // namespace

struct ProgramAnalysis::Impl {
    std::unique_ptr<OriginalObligations> obligationModel = std::make_unique<OriginalObligations>();
    std::unique_ptr<OriginalRequests> requestModel;
    std::unique_ptr<OriginalPreparation> preparationModel;
    std::string obligationError = "original obligation universe not prepared";
    bool legacyRequirementIndexReady = false;
    OriginalReadQueries readers;
    OriginalExactQueries exact;
    std::map<OriginalInterval, std::vector<StorageOrigin>> exactMay;
    std::vector<std::optional<PhysicalBankCorrespondence>> banks;
    using AlternativeKey = std::tuple<std::size_t, std::size_t, StorageRelationship::Kind>;
    std::map<AlternativeKey, std::vector<StorageOrigin>> alternatives;
    std::map<std::pair<unsigned, unsigned>, std::vector<OriginalRequirementId>> directions;
    std::vector<std::vector<TypedOriginalRequirement>> typedByDeadline;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> typedBySource;
    // Requirement IDs alone do not identify a continuation. All original
    // interval fields (including Unknown qualifications) participate in reuse.
    using QueryKey = std::tuple<std::size_t, std::size_t, OriginalInterval>;
    // RMW can give WAR and WAW under the same cut interval. They query
    // different old-state roots and must never share a fixed-visit answer.
    using FixedVisitKey = std::pair<OriginalInterval, StorageRelationship::Kind>;
    std::map<FixedVisitKey, FixedVisitCorrespondence> fixedVisits;
    std::map<QueryKey, DecodedRequirement> cache;
    std::map<QueryKey, InterpretedRequirement> interpretations;
    std::map<std::pair<std::size_t, std::size_t>, OriginalAccessSummary> allUses;
    std::map<std::pair<std::size_t, std::size_t>, StorageLifecycle> lifecycles;
    std::map<std::tuple<std::size_t, std::size_t, unsigned>, std::vector<StorageOrigin>> readerUses;
    explicit Impl(const OriginalStructure& original, const OriginalValueQueries& values)
        : readers(original, &values),
          exact(original, values, readers),
          banks(original.physicalAddresses.size()),
          typedByDeadline(original.originalSites.size()),
          typedBySource(original.operations.size())
    {}
};

ProgramAnalysis::ProgramAnalysis(
    const SyncInput& source, OriginalStructure structure, OriginalRequestBoundaryProvider requestBoundaries)
    : input(source),
      original([&] {
          normalizeOriginalSequences(structure.body);
          return std::move(structure);
      }()),
      values(original),
      storage(original, &values),
      occurrenceQueries(original, &values, &storage),
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
    if (impl->obligationModel->complete()) {
        impl->requestModel = OriginalRequests::build(*this, requestBoundaries);
        if (impl->requestModel->complete()) {
            impl->preparationModel = std::make_unique<OriginalPreparation>(*this);
        }
    }
}

ProgramAnalysis::~ProgramAnalysis() = default;
bool ProgramAnalysis::complete() const
{
    return storage.complete() && impl->obligationModel->complete() && impl->requestModel && impl->requestModel->complete() &&
           impl->preparationModel && impl->preparationModel->formed();
}
const std::string& ProgramAnalysis::reason() const
{
    if (!storage.complete()) { return storage.reason(); }
    if (!impl->obligationModel->complete()) { return impl->obligationError; }
    return impl->requestModel ? impl->requestModel->reason() : impl->obligationError;
}
const OriginalObligations& ProgramAnalysis::obligations() const { return *impl->obligationModel; }
const OriginalRequests* ProgramAnalysis::requests() const
{
    return impl->requestModel && impl->requestModel->complete() ? impl->requestModel.get() : nullptr;
}
const OriginalPreparation* ProgramAnalysis::preparation() const
{
    return impl->preparationModel && impl->preparationModel->formed() ? impl->preparationModel.get() : nullptr;
}
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
    if (!values.current() || index >= impl->banks.size()) {
        return invalid;
    }
    if (!impl->banks[index]) {
        impl->banks[index] = occurrenceQueries.bank(index);
    }
    return *impl->banks[index];
}
PeriodicUseCorrespondence ProgramAnalysis::periodicUseFor(OriginalObligationId obligation) const
{
    PeriodicUseCorrespondence result;
    const auto* family = obligations().get(obligation.family);
    const auto member = obligations().membership(obligation);
    if (!family || member.status == ObligationMembership::Status::Invalid ||
        member.status == ObligationMembership::Status::Excluded) {
        result.reason = "invalid or excluded original obligation";
        return result;
    }
    const auto& key = family->key;
    if (obligation.origin.incoming || key.kind == OriginalObligationKey::Kind::Typed) {
        result.reason = "incoming/typed obligation needs its own interface, not a fictitious periodic source";
        return result;
    }
    OriginalIntervalRequest query;
    query.version = key.originalVersion;
    query.selector.cell = key.cell;
    query.selector.read = key.kind == OriginalObligationKey::Kind::RAW;
    query.selector.write = !query.selector.read;
    query.occurrence.source = obligation.origin.operation;
    query.occurrence.target = key.consumerOperation;
    query.start = {query.occurrence.source, OriginalCut::After};
    query.stop = {query.occurrence.target, OriginalCut::Before};
    // This is a local PLACEMENT relationship queried under an unchanged ID,
    // not a restriction of that ID's complete incoming/original-control demands.
    const auto prepared = prepareInterval(std::move(query));
    if (!prepared.valid) {
        result.reason = prepared.reason;
        return result;
    }
    return occurrenceQueries.periodic(
        prepared.interval, key.sourceRole() == OriginalObligationKey::Role::Writer,
        key.consumerRole() == OriginalObligationKey::Role::Writer);
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
    } else if (answer.queryInterval) {
        const auto& relation = answer.decoded->requirement.relationship;
        answer.occurrence.periodic = occurrenceQueries.periodic(
            *answer.queryInterval, relation.kind != StorageRelationship::WAR, relation.kind != StorageRelationship::RAW);
        const auto& periodic = *answer.occurrence.periodic;
        if (periodic.exact) {
            answer.occurrence.status = answer.occurrence.source == answer.occurrence.target ?
                                          OriginalOccurrenceInterpretation::Status::PeriodicSameRole :
                                          OriginalOccurrenceInterpretation::Status::PeriodicRoles;
            answer.occurrence.owner = periodic.owner;
            const auto distance = periodic.links.front().occurrence.distance;
            const bool uniform = std::all_of(periodic.links.begin(), periodic.links.end(), [&](const auto& link) {
                return link.occurrence.distance == distance;
            });
            answer.occurrence.period = uniform ? distance : 0;
            answer.occurrence.firstOrdinalWithLocalPredecessor = uniform ? distance : 0;
            answer.occurrence.earlierOrdinalsNeedIncomingCase = std::any_of(
                periodic.links.begin(), periodic.links.end(), [](const auto& link) { return link.occurrence.distance > 0; });
        }
    }
    if (answer.occurrence.status == OriginalOccurrenceInterpretation::Status::Unknown) {
        answer.occurrence.reason = answer.occurrence.periodic ? answer.occurrence.periodic->reason : fixedVisit.reason;
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
    answer.fixedVisit = fixedVisit;
    if (fixedVisit.alternatives) {
        const auto& d1 = *fixedVisit.alternatives;
        // This flag is about a complete matched local alternative family. An
        // incoming path retains its own unresolved enclosing-source interface.
        answer.alternativeGuardsQualified = fixedVisit.exact && d1.sources.complete &&
                                           d1.physicalRolesQualified && d1.localGuardsQualified && !d1.hasIncoming;
    }
    // The marginal origin vector remains a may-set. D1 predicates live in the
    // retained fixedVisit result; do not treat its conditional exactness as an
    // unconditional edge or an already-established completion.
    return impl->interpretations.emplace(key, std::move(answer)).first->second;
}
FixedVisitCorrespondence ProgramAnalysis::fixedVisitFor(const DecodedRequirement& requirement) const
{
    const auto& relation = requirement.requirement.relationship;
    if (!values.current() || !requirement.interval || requirement.interval->query.version != original.version ||
        requirement.interval->query.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge ||
        requirement.interval->query.occurrence.incomingInterface != NoControlId ||
        requirement.interval->query.occurrence.qualification != NoControlId) {
        FixedVisitCorrespondence unknown;
        unknown.source = relation.source.operation;
        unknown.target = relation.target.operation;
        unknown.reason = "fixed-visit query lacks its original interval interpretation";
        return unknown;
    }
    const Impl::FixedVisitKey key{*requirement.interval, relation.kind};
    const auto prior = impl->fixedVisits.find(key);
    if (prior != impl->fixedVisits.end()) {
        return prior->second;
    }
    const auto hazard = relation.kind == StorageRelationship::RAW ? FactoredUseNode::Hazard::RAW :
                        relation.kind == StorageRelationship::WAR ? FactoredUseNode::Hazard::WAR :
                                                                    FactoredUseNode::Hazard::WAW;
    auto result = occurrenceQueries.fixedVisit(
        relation.source.operation, relation.target.operation, relation.cell, hazard);
    if (result.exact && result.alternatives->sources.frame.kind != FactoredUseFrame::Kind::Invocation) {
        const auto owner = result.alternatives->sources.frame.owner;
        const auto* frame = scopeRegion(original.body, owner);
        if (requirement.interval->owner != owner && (!frame || !scopeRegion(*frame, requirement.interval->owner))) {
            result.exact = false;
            result.reason = "fixed-use result needs qualified transport to the enclosing continuation";
        }
    }
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
    if (relation.kind == StorageRelationship::WAW || (first && relation.kind == StorageRelationship::WAR)) {
        if (requirement.interval) {
            return exactBoundary(*requirement.interval, first, false);
        }
        result.mayAccesses = &allUses.writers;
        result.reason = "conflict frontier has no validated original cut interval";
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
    // Boundary discovery does not itself pair a producer with these reads.
    // Keep D1/D2/generation qualification on the original relationship/support;
    // do not withhold an independently derivable D3/I.2 boundary because that
    // DIFFERENT query is unresolved. Complete obligations remain unchanged.
    return exactBoundary(*requirement.readerInterval, first, true);
}
OriginalExactFrontiers ProgramAnalysis::exactFrontiers(const OriginalInterval& interval, bool readOnly) const
{
    const auto prepared = prepareInterval(interval.query);
    if (!prepared.valid || prepared.interval != interval) {
        OriginalExactFrontiers unknown;
        unknown.interval = interval;
        unknown.readOnly = readOnly;
        unknown.reason = prepared.valid ? "inconsistent original frontier owner" : prepared.reason;
        return unknown;
    }
    return impl->exact.query(interval, readOnly);
}
OriginalBoundaryResult ProgramAnalysis::exactBoundary(const OriginalInterval& interval, bool first, bool readOnly) const
{
    static const std::vector<StorageOrigin> empty;
    OriginalBoundaryResult result;
    result.interval = interval;
    result.owner = interval.owner;
    result.cell = interval.query.selector.cell;
    result.mayAccesses = &empty;
    const auto& selector = interval.query.selector;
    if (interval.query.version == original.version && selector.cell < original.cells.size()) {
        auto found = impl->exactMay.find(interval);
        if (found == impl->exactMay.end()) {
            // This is a conservative owner-level view, not an exact positive
            // frontier. Keep it even when a cut or participation is unresolved.
            OriginalAllQuery mayQuery;
            mayQuery.owner = interval.owner;
            mayQuery.cell = selector.cell;
            mayQuery.read = selector.read;
            mayQuery.write = selector.write;
            if (selector.engine) {
                mayQuery.engine = static_cast<PipelineType>(*selector.engine);
            }
            const auto footprint = all(mayQuery);
            auto may = footprint.readers;
            may.insert(may.end(), footprint.writers.begin(), footprint.writers.end());
            found = impl->exactMay.emplace(interval, std::move(may)).first;
        }
        result.mayAccesses = &found->second;
    }
    const auto answer = exactFrontiers(interval, readOnly);
    result.nonempty = answer.nonempty;
    result.noHit = answer.noHit;
    result.frontier = first ? answer.first : answer.last;
    if (answer.status == OriginalExactFrontiers::Status::Unknown) {
        result.reason = answer.reason;
        return result;
    }
    result.endpointQualification.status = OriginalValueQualification::Status::Available;
    if (answer.status == OriginalExactFrontiers::Status::NoHit) {
        result.status = OriginalBoundaryResult::Status::NoHit;
        result.guardsAvailableAtReadSites = true;
        return result;
    }
    for (const auto& [operation, guard] : impl->readers.guardedAccesses(result.frontier)) {
        if (guard == 0) {
            continue;
        }
        const OriginalCut cut{operation, first ? OriginalCut::Before : OriginalCut::After};
        if (!resolveOriginalCut(original, cut)) {
            result.reason = "exact selected access has no executable original cut on this side";
            result.cuts.clear();
            return result;
        }
        result.endpointQualification = OriginalValueQueries::combine(
            std::move(result.endpointQualification), impl->readers.qualificationAt(guard, operation, !first));
        result.cuts.push_back(cut);
    }
    if (result.cuts.empty()) {
        result.reason = "nonempty exact frontier has no qualified endpoint references";
        return result;
    }
    result.guardsAvailableAtReadSites = result.endpointQualification.available();
    if (!result.endpointQualification.executableAfterPrerequisites()) {
        result.reason = result.endpointQualification.reason();
        result.cuts.clear();
        return result;
    }
    result.status = OriginalBoundaryResult::Status::Exact;
    return result;
}
OriginalBoundaryResult ProgramAnalysis::firstConflict(const OriginalInterval& interval, bool readOnly) const
{
    return exactBoundary(interval, true, readOnly);
}
OriginalBoundaryResult ProgramAnalysis::lastRelevantUse(const OriginalInterval& interval, bool readOnly) const
{
    return exactBoundary(interval, false, readOnly);
}
const OriginalIntervalParticipation* ProgramAnalysis::intervalParticipation(std::size_t id) const
{
    return impl->readers.intervalParticipation(id);
}
OriginalExactFrontierStats ProgramAnalysis::exactFrontierStats() const { return impl->exact.statistics(); }
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
    auto unknown = boundary(*requirement.decoded, first);
    unknown.status = OriginalBoundaryResult::Status::Unknown;
    unknown.cuts.clear();
    if (!support.complete || !support.generationEstablished || support.cases.backedge ||
        requirement.occurrence.status != OriginalOccurrenceInterpretation::Status::FixedVisit) {
        unknown.reason = "reader frontier lacks qualified generation or occurrence support";
        return unknown;
    }
    const auto& relation = requirement.decoded->requirement.relationship;
    const auto reader =
        relation.kind == StorageRelationship::RAW ? relation.target.operation : relation.source.operation;
    const auto expected = supportInterval(requirement);
    if (!expected.valid || !support.interval || *support.interval != expected.interval ||
        reader >= original.operations.size() ||
        !llvm::any_of(support.readers, [&](const StorageOrigin& use) { return use.operation == reader; })) {
        unknown.reason = "reader and support do not name the same original interval";
        return unknown;
    }
    auto query = support.interval->query;
    query.selector.read = true;
    query.selector.write = false;
    query.selector.engine = unsigned(original.operations[reader].instruction->kPipeValue);
    const auto prepared = prepareInterval(std::move(query));
    if (!prepared.valid) {
        unknown.reason = prepared.reason;
        return unknown;
    }
    return exactBoundary(prepared.interval, first, true);
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
        if (!resolveOriginalCut(original, cut) ||
            std::find(boundary.cuts.begin(), boundary.cuts.end(), cut) == boundary.cuts.end()) {
            return {}; // A first/before certificate is not a last/after certificate.
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
