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
#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include "llvm/ADT/DenseSet.h"

namespace mlir::pto::frontiersynch {
namespace {
void indexOwners(const Region &region, std::vector<std::size_t> &path,
                 std::vector<std::vector<std::size_t>> &byOperation) {
  // A payload identity is not the owner of its surrounding physical-use
  // sequence, even when one instruction translates to several phases.
  const bool owns = region.kind != Region::Operation && region.originalOwner != NoControlId;
  if (owns) {
    path.push_back(region.originalOwner);
  }
  if (region.kind == Region::Operation && region.operation < byOperation.size()) {
    byOperation[region.operation] = path;
  }
  for (const auto &child : region.children) {
    indexOwners(child, path, byOperation);
  }
  if (owns) {
    path.pop_back();
  }
}

std::size_t commonOwner(const std::vector<std::size_t> &source,
                        const std::vector<std::size_t> &target) {
  std::size_t owner = NoControlId;
  for (std::size_t i = 0; i < source.size() && i < target.size(); ++i) {
    if (source[i] != target[i]) {
      break;
    }
    owner = source[i];
  }
  return owner;
}
bool allRoleIncidencesUseRelation(const OriginalStructure &original, std::size_t operation,
                                  std::size_t cell, bool write, std::size_t relation) {
  if (operation >= original.operations.size()) {
    return false;
  }
  bool hasRole = false;
  for (const auto &access : original.operations[operation].accesses) {
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
  OriginalReadQueries readers;
  std::vector<std::vector<std::size_t>> owners;
  std::vector<std::optional<PhysicalBankCorrespondence>> banks;
  using AlternativeKey = std::tuple<std::size_t, std::size_t, StorageRelationship::Kind>;
  std::map<AlternativeKey, std::vector<StorageOrigin>> alternatives;
  std::map<std::pair<unsigned, unsigned>, std::vector<OriginalRequirementId>> directions;
  std::vector<std::vector<TypedOriginalRequirement>> typedByDeadline;
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> typedBySource;
  std::map<std::tuple<std::size_t, std::size_t, std::size_t>, FixedVisitCorrespondence> fixedVisits;
  std::map<std::pair<std::size_t, std::size_t>, DecodedRequirement> cache;
  std::map<std::pair<std::size_t, std::size_t>, InterpretedRequirement> interpretations;
  std::map<std::pair<std::size_t, std::size_t>, OriginalAccessSummary> allUses;
  std::map<std::pair<std::size_t, std::size_t>, StorageLifecycle> lifecycles;
  std::map<std::tuple<std::size_t, std::size_t, unsigned>, std::vector<StorageOrigin>> readerUses;
  explicit Impl(const OriginalStructure &original)
      : readers(original), owners(original.operations.size()),
        banks(original.physicalAddresses.size()),
        typedByDeadline(original.originalSites.size()), typedBySource(original.operations.size()) {
    std::vector<std::size_t> path;
    indexOwners(original.body, path, owners);
  }
};

ProgramAnalysis::ProgramAnalysis(const SyncInput &source, OriginalStructure structure)
    : input(source), original(std::move(structure)), storage(original), occurrenceQueries(original),
      impl(std::make_unique<Impl>(original)) {
  if (!storage.complete()) {
    return;
  }
  for (std::size_t target = 0; target < original.operations.size(); ++target) {
    const auto &requests = storage.requirementsAt(target);
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto &request = requests[index];
      const auto &relation = request.relationship;
      impl->alternatives[{target, relation.cell, relation.kind}].push_back(relation.source);
      const auto source = relation.source.operation;
      if (source < original.operations.size()) {
        const auto from = unsigned(original.operations[source].instruction->kPipeValue);
        const auto to = unsigned(original.operations[target].instruction->kPipeValue);
        impl->directions[{from, to}].push_back({target, index});
      }
    }
  }
  DenseMap<mlir::Operation *, SmallVector<std::size_t>> phases;
  for (std::size_t phase = 0; phase < original.operations.size(); ++phase) {
    phases[original.operations[phase].instruction->elementOp].push_back(phase);
  }
  DenseMap<Value, std::set<std::tuple<std::size_t, unsigned, std::size_t>>> typedSeen;
  auto collectTyped = [&](std::size_t target, Value required,
                          TypedOriginalRequirement::Cause cause) {
    llvm::DenseSet<Value> seen;
    std::set<std::size_t> emitted;
    SmallVector<Value> pending{required};
    bool incomingOrUnknown = false;
    while (!pending.empty()) {
      const auto value = pending.pop_back_val();
      if (!value || !seen.insert(value).second) {
        continue;
      }
      auto *producer = value.getDefiningOp();
      if (!producer) {
        incomingOrUnknown = true;
        continue;
      }
      const auto found = phases.find(producer);
      if (found != phases.end()) {
        for (auto phase : found->second) {
          if (!emitted.insert(phase).second) {
            continue;
          }
          if (!typedSeen[required].emplace(target, unsigned(cause), phase).second) {
            continue;
          }
          TypedOriginalRequirement request;
          request.cause = cause;
          request.source = {phase, SourceMilestone::After};
          request.deadlineOriginalSite = target;
          request.requiredValue = required;
          request.sourceEngine = original.operations[phase].instruction->kPipeValue;
          request.sourceGapExecutable = original.operations[phase].afterExecutable;
          const auto index = impl->typedByDeadline[target].size();
          impl->typedByDeadline[target].push_back(request);
          const auto executableSource = original.operations[phase].enclosingAfter;
          if (executableSource < impl->typedBySource.size()) {
            impl->typedBySource[executableSource].push_back({target, index});
          }
        }
        continue;
      }
      if (!isMemoryEffectFree(producer)) {
        incomingOrUnknown = true;
      }
      pending.append(producer->operand_begin(), producer->operand_end());
    }
    if (incomingOrUnknown) {
      if (!typedSeen[required].emplace(target, unsigned(cause), NoControlId).second) {
        return;
      }
      TypedOriginalRequirement request;
      request.cause = cause;
      request.deadlineOriginalSite = target;
      request.requiredValue = required;
      request.incomingOrUnknownSource = true;
      impl->typedByDeadline[target].push_back(request);
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
  for (const auto &phase : original.operations) {
    llvm::DenseSet<Value> seenAddresses;
    SmallVector<Value> addresses;
    auto collectAddresses = [&](ArrayRef<const BaseMemInfo *> memories) {
      for (const auto *memory : memories) {
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
}

ProgramAnalysis::~ProgramAnalysis() = default;
bool ProgramAnalysis::complete() const { return storage.complete(); }
const std::string &ProgramAnalysis::reason() const { return storage.reason(); }
const PhysicalBankCorrespondence &ProgramAnalysis::bankRelation(std::size_t index) const {
  static const PhysicalBankCorrespondence invalid;
  if (index >= impl->banks.size()) {
    return invalid;
  }
  if (!impl->banks[index]) {
    impl->banks[index] = occurrenceQueries.bank(index);
  }
  return *impl->banks[index];
}
const std::vector<StorageOrigin> &ProgramAnalysis::alternativeSourcesFor(
    const DecodedRequirement &requirement) const {
  static const std::vector<StorageOrigin> empty;
  const auto &relation = requirement.requirement.relationship;
  const auto key = Impl::AlternativeKey{relation.target.operation, relation.cell, relation.kind};
  const auto found = impl->alternatives.find(key);
  return found == impl->alternatives.end() ? empty : found->second;
}
const std::vector<OriginalRequirement> &ProgramAnalysis::requirementsAt(std::size_t operation) const {
  return storage.requirementsAt(operation);
}
const std::vector<OriginalRequirementId> &ProgramAnalysis::requirementsFromTo(
    PipelineType source, PipelineType target) const {
  static const std::vector<OriginalRequirementId> empty;
  const auto found = impl->directions.find({unsigned(source), unsigned(target)});
  return found == impl->directions.end() ? empty : found->second;
}
const std::vector<TypedOriginalRequirement> &ProgramAnalysis::typedRequirementsAt(
    std::size_t originalSite) const {
  static const std::vector<TypedOriginalRequirement> empty;
  return originalSite < impl->typedByDeadline.size() ? impl->typedByDeadline[originalSite] : empty;
}
const std::vector<std::pair<std::size_t, std::size_t>> &ProgramAnalysis::typedSubscriptionsAt(
    std::size_t sourceOperation) const {
  static const std::vector<std::pair<std::size_t, std::size_t>> empty;
  return sourceOperation < impl->typedBySource.size() ? impl->typedBySource[sourceOperation] : empty;
}
const DecodedRequirement &ProgramAnalysis::decodeAt(std::size_t target, std::size_t index) const {
  const auto &requirements = storage.requirementsAt(target);
  const auto key = std::make_pair(target, index);
  const auto prior = impl->cache.find(key);
  if (prior != impl->cache.end()) {
    return prior->second;
  }
  DecodedRequirement answer;
  const bool invalidIdentity = !complete() || index >= requirements.size();
  if (invalidIdentity) {
    answer.unresolved.emplace_back("invalid original requirement identity");
    return impl->cache.emplace(key, std::move(answer)).first->second;
  }
  const auto &requirement = requirements[index];
  answer.requirement = requirement;
  const auto source = requirement.relationship.source.operation;
  const auto cell = requirement.relationship.cell;
  if (source >= impl->owners.size()) {
    answer.unresolved.emplace_back("source has no original operation");
    return impl->cache.emplace(key, std::move(answer)).first->second;
  }
  answer.owner = commonOwner(impl->owners[source], impl->owners[target]);
  auto collectBanks = [&](std::size_t operation, bool write, std::vector<std::size_t> &out) {
    for (const auto &access : original.operations[operation].accesses) {
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
  if (requirement.relationship.kind != StorageRelationship::WAW) {
    const auto reader = requirement.relationship.kind == StorageRelationship::RAW ? target : source;
    const auto pipe = original.operations[reader].instruction->kPipeValue;
    const auto &segment = impl->readers.segment(answer.owner, reader, cell, pipe);
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
const InterpretedRequirement &ProgramAnalysis::interpretAt(std::size_t target, std::size_t index) const {
  const auto key = std::make_pair(target, index);
  const auto prior = impl->interpretations.find(key);
  if (prior != impl->interpretations.end()) {
    return prior->second;
  }
  InterpretedRequirement answer;
  answer.decoded = &decodeAt(target, index);
  if (index >= storage.requirementsAt(target).size()) {
    return impl->interpretations.emplace(key, std::move(answer)).first->second;
  }
  const auto fixedVisit = fixedVisitFor(*answer.decoded);
  answer.occurrence.owner = answer.decoded->owner;
  answer.occurrence.source = answer.decoded->requirement.relationship.source.operation;
  answer.occurrence.target = answer.decoded->requirement.relationship.target.operation;
  std::set_intersection(answer.decoded->sourceBankRelations.begin(),
                        answer.decoded->sourceBankRelations.end(),
                        answer.decoded->targetBankRelations.begin(),
                        answer.decoded->targetBankRelations.end(),
                        std::back_inserter(answer.occurrence.sharedBankCandidates));
  if (fixedVisit.exact) {
    answer.occurrence.status = OriginalOccurrenceInterpretation::Status::FixedVisit;
  } else if (answer.occurrence.source == answer.occurrence.target &&
             answer.occurrence.sharedBankCandidates.size() == 1) {
    const auto candidate = answer.occurrence.sharedBankCandidates.front();
    const auto &bank = bankRelation(candidate);
    const auto &relation = answer.decoded->requirement.relationship;
    const bool sourceWrites = relation.kind != StorageRelationship::WAR;
    const bool targetWrites = relation.kind != StorageRelationship::RAW;
    const bool covered = allRoleIncidencesUseRelation(
                             original, answer.occurrence.source, relation.cell, sourceWrites, candidate) &&
                         allRoleIncidencesUseRelation(
                             original, answer.occurrence.target, relation.cell, targetWrites, candidate);
    const bool oneMatchingUse = bank.participatingOperations.size() == 1 &&
                                bank.participatingOperations.front() == answer.occurrence.source;
    if (covered && bank.exactPermutation && oneMatchingUse) {
      answer.occurrence.status = OriginalOccurrenceInterpretation::Status::PeriodicSameRole;
      answer.occurrence.owner = bank.owner;
      answer.occurrence.period = bank.distance;
      answer.occurrence.firstOrdinalWithLocalPredecessor = bank.distance;
      answer.occurrence.earlierOrdinalsNeedIncomingCase = true;
    }
  }
  if (answer.occurrence.status == OriginalOccurrenceInterpretation::Status::Unknown) {
    answer.occurrence.reason = answer.occurrence.sharedBankCandidates.empty() ?
        fixedVisit.reason : "physical-bank candidate lacks qualified predecessor and child correspondence";
  }
  const auto &relation = answer.decoded->requirement.relationship;
  auto lifecycle = [&](std::size_t operation) -> const StorageLifecycle * {
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
  answer.interval.start = answer.decoded->requirement.source;
  answer.interval.stop = answer.decoded->requirement.deadline;
  answer.interval.read = relation.kind == StorageRelationship::RAW;
  answer.interval.write = !answer.interval.read;
  answer.support.cell = relation.cell;
  if (relation.kind == StorageRelationship::WAW) {
    answer.support.producer = relation.source.operation;
    answer.support.reuse = relation.target.operation;
    answer.support.hasWriteDelimitedCandidate = true;
  } else if (relation.kind == StorageRelationship::RAW &&
             answer.targetUse->nextWriters.size() == 1) {
    answer.support.producer = relation.source.operation;
    answer.support.reuse = answer.targetUse->nextWriters.front().operation;
    answer.support.hasWriteDelimitedCandidate = true;
  } else if (relation.kind == StorageRelationship::WAR &&
             answer.sourceUse->previousWriters.size() == 1) {
    answer.support.producer = answer.sourceUse->previousWriters.front().operation;
    answer.support.reuse = relation.target.operation;
    answer.support.hasWriteDelimitedCandidate = true;
  } else {
    answer.support.reason = "producer or next conflicting write is not unique";
  }
  answer.firstConflict = firstConflict(*answer.decoded);
  answer.lastRelevantUse = lastRelevantUse(*answer.decoded);
  answer.mayAlternativeSources = &alternativeSourcesFor(*answer.decoded);
  answer.factoredUse = &storage.factored(relation.cell);
  // The origin group is a may-set. D1 path guards and the possible absence of
  // an incoming full writer remain explicit in targetUse and are not inferred
  // from this vector's cardinality.
  return impl->interpretations.emplace(key, std::move(answer)).first->second;
}
FixedVisitCorrespondence ProgramAnalysis::fixedVisitFor(const DecodedRequirement &requirement) const {
  const auto &relation = requirement.requirement.relationship;
  const auto key = std::make_tuple(relation.source.operation, relation.target.operation, relation.cell);
  const auto prior = impl->fixedVisits.find(key);
  if (prior != impl->fixedVisits.end()) {
    return prior->second;
  }
  auto result = occurrenceQueries.fixedVisit(relation.source.operation, relation.target.operation,
                                             relation.cell);
  return impl->fixedVisits.emplace(key, std::move(result)).first->second;
}
OriginalBoundaryResult ProgramAnalysis::boundary(const DecodedRequirement &requirement, bool first) const {
  static const std::vector<StorageOrigin> empty;
  OriginalBoundaryResult result;
  result.mayAccesses = &empty;
  const auto &relation = requirement.requirement.relationship;
  result.owner = requirement.owner;
  result.cell = relation.cell;
  const auto key = std::make_pair(result.owner, result.cell);
  auto found = impl->allUses.find(key);
  if (found == impl->allUses.end()) {
    found = impl->allUses.emplace(key, storage.all(result.owner, result.cell)).first;
  }
  const auto &allUses = found->second;
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
  const auto reader = relation.kind == StorageRelationship::RAW ?
      relation.target.operation : relation.source.operation;
  if (reader >= original.operations.size()) {
    result.reason = "reader has no original operation";
    return result;
  }
  const auto pipe = original.operations[reader].instruction->kPipeValue;
  const auto readerKey = std::make_tuple(result.owner, result.cell, unsigned(pipe));
  auto readerUses = impl->readerUses.find(readerKey);
  if (readerUses == impl->readerUses.end()) {
    std::vector<StorageOrigin> matching;
    for (const auto &use : allUses.readers) {
      if (original.operations[use.operation].instruction->kPipeValue == pipe) {
        matching.push_back(use);
      }
    }
    readerUses = impl->readerUses.emplace(readerKey, std::move(matching)).first;
  }
  result.mayAccesses = &readerUses->second;
  if (!requirement.hasReaderFrontier) {
    result.reason = "no qualified original reader interval";
    return result;
  }
  const auto &readers = requirement.readers;
  if (readers.status == OriginalReaderFrontiers::Status::Unknown) {
    result.reason = readers.reason;
    return result;
  }
  result.nonempty = readers.nonempty;
  result.frontier = first ? readers.first : readers.last;
  result.guardsAvailableAtReadSites = readers.guardsAvailableAtReadSites;
  if (!readers.guardsAvailableAtReadSites) {
    result.reason = "reader participation is unavailable at an original read site";
    return result;
  }
  if (!requirement.requirement.episodeKnown || !requirement.requirement.occurrenceKnown) {
    result.reason = "structural reader frontier lacks a qualified episode and occurrence interval";
    return result;
  }
  result.status = readers.status == OriginalReaderFrontiers::Status::NoHit ?
      OriginalBoundaryResult::Status::NoHit : OriginalBoundaryResult::Status::Exact;
  return result;
}
OriginalBoundaryResult ProgramAnalysis::firstConflict(const DecodedRequirement &requirement) const {
  return boundary(requirement, true);
}
OriginalBoundaryResult ProgramAnalysis::lastRelevantUse(const DecodedRequirement &requirement) const {
  return boundary(requirement, false);
}
PhysicalUseFrontier ProgramAnalysis::firstMayUse(const OriginalUseQuery &query) const {
  return storage.firstUse(query);
}
PhysicalUseFrontier ProgramAnalysis::lastMayUse(const OriginalUseQuery &query) const {
  return storage.lastUse(query);
}
const std::vector<SourceSubscription> &ProgramAnalysis::subscriptionsAt(std::size_t operation) const {
  return storage.subscriptionsAt(operation);
}
OriginalAccessSummary ProgramAnalysis::all(const OriginalAllQuery &query) const {
  const auto key = std::make_pair(query.owner, query.cell);
  auto found = impl->allUses.find(key);
  if (found == impl->allUses.end()) {
    found = impl->allUses.emplace(key, storage.all(query.owner, query.cell)).first;
  }
  auto result = found->second;
  if (!result.complete) {
    return result;
  }
  auto filter = [&](std::vector<StorageOrigin> &uses, bool enabled) {
    if (!enabled) {
      uses.clear();
      return;
    }
    if (query.engine) {
      uses.erase(std::remove_if(uses.begin(), uses.end(), [&](const StorageOrigin &use) {
        return use.operation >= original.operations.size() ||
               original.operations[use.operation].instruction->kPipeValue != *query.engine;
      }), uses.end());
    }
  };
  filter(result.readers, query.read);
  filter(result.writers, query.write);
  return result;
}
OriginalMayAfter ProgramAnalysis::mayAfter(const OriginalContinuationQuery &query) const {
  return storage.mayAfter(query);
}
OriginalSupportInterval ProgramAnalysis::qualifySupport(const InterpretedRequirement &requirement) const {
  if (!requirement.support.hasWriteDelimitedCandidate) {
    OriginalSupportInterval result;
    result.reason = requirement.support.reason;
    return result;
  }
  return storage.supportBetween(requirement.support.producer, requirement.support.reuse,
                                requirement.support.cell);
}
OriginalBoundaryResult ProgramAnalysis::qualifiedBoundary(const InterpretedRequirement &requirement,
                                                 const OriginalSupportInterval &support,
                                                 bool first) const {
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
  const auto &relation = requirement.decoded->requirement.relationship;
  const auto reader = relation.kind == StorageRelationship::RAW ?
      relation.target.operation : relation.source.operation;
  const bool participates = llvm::any_of(support.readers, [&](const StorageOrigin &use) {
    return use.operation == reader;
  });
  const bool sameInterval = support.producer == requirement.support.producer &&
                            support.reuse == requirement.support.reuse &&
                            support.cell == requirement.support.cell;
  if (!participates || !sameInterval) {
    result.reason = "reader is not in the qualified write-delimited episode";
    return result;
  }
  auto qualified = *requirement.decoded;
  qualified.requirement.occurrenceKnown = true;
  qualified.requirement.episodeKnown = true;
  return boundary(qualified, first);
}
std::vector<OriginalEndpointCandidate> ProgramAnalysis::endpointCandidates(
    const OriginalBoundaryResult &boundary, SourceMilestone::Side side) const {
  std::vector<OriginalEndpointCandidate> result;
  if (boundary.status != OriginalBoundaryResult::Status::Exact) {
    return result;
  }
  for (const auto &[operation, guard] : impl->readers.guardedAccesses(boundary.frontier)) {
    const bool invalidEndpoint = operation >= original.operations.size() || guard == 0;
    if (invalidEndpoint) {
      return {};
    }
    const auto &use = original.operations[operation];
    const bool executable = side == SourceMilestone::Before ?
        use.beforeExecutable : use.afterExecutable;
    result.push_back({{operation, side}, guard, guardAvailableAt(guard, operation), executable});
  }
  return result;
}
ParticipationExpression ProgramAnalysis::predicate(std::size_t id) const { return impl->readers.predicate(id); }
GuardedReadFrontier ProgramAnalysis::readerFrontier(std::size_t id) const { return impl->readers.frontier(id); }
bool ProgramAnalysis::guardAvailableAt(std::size_t predicateId, std::size_t operation) const {
  return impl->readers.availableAt(predicateId, operation);
}
std::vector<OriginalParticipationDemand> ProgramAnalysis::participationDemands() const {
  return impl->readers.participationDemands();
}
} // namespace mlir::pto::frontiersynch
