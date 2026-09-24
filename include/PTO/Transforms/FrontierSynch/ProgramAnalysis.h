// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PROGRAMANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PROGRAMANALYSIS_H

#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "PTO/Transforms/FrontierSynch/ReaderFrontiers.h"
#include <optional>

namespace mlir::pto::frontiersynch {

// An original-program interpretation of one required physical ordering edge.
// Unknown qualification retains the edge and every independently established fact.
struct DecodedRequirement {
  OriginalRequirement requirement;
  std::size_t owner = NoControlId;
  std::vector<std::size_t> sourceBankRelations, targetBankRelations;
  OriginalReaderFrontiers readers;
  bool hasReaderFrontier = false;
  std::vector<std::string> unresolved;
};
struct OriginalBoundaryResult {
  enum class Status { Unknown, NoHit, Exact } status = Status::Unknown;
  std::size_t owner = NoControlId, cell = 0;
  std::size_t nonempty = 0, frontier = 0;
  bool guardsAvailableAtReadSites = false;
  // Shared original may-set owned by ProgramAnalysis; no per-requirement copy.
  const std::vector<StorageOrigin> *mayAccesses = nullptr;
  std::string reason;
};
struct OriginalEndpointCandidate {
  SourceMilestone position;
  std::size_t predicate = 0;
  bool guardAvailableAtOperation = false;
  bool executableInOriginalIR = false;
};
struct OriginalRequirementId {
  std::size_t deadline = NoControlId, index = 0;
};
struct TypedOriginalRequirement {
  enum class Cause { BranchCondition, LoopBound, WhileCondition, Address } cause = Cause::BranchCondition;
  SourceMilestone source;
  std::size_t deadlineOriginalSite = NoControlId;
  Value requiredValue;
  PipelineType sourceEngine = PipelineType::PIPE_UNASSIGNED;
  bool incomingOrUnknownSource = false;
  bool sourceGapExecutable = false;
  // The original SSA dependency is a requirement, not proof that the source
  // completion is natively available at this dynamic control occurrence.
  bool occurrenceQualified = false;
};
struct OriginalAllQuery {
  std::size_t owner = NoControlId, cell = 0;
  bool read = true, write = true;
  std::optional<PipelineType> engine;
};
struct OriginalOccurrenceInterpretation {
  enum class Status { Unknown, FixedVisit, PeriodicSameRole } status = Status::Unknown;
  std::size_t owner = NoControlId;
  std::size_t source = NoControlId, target = NoControlId;
  // For PeriodicSameRole, the predecessor in one invocation of owner is at
  // ordinal i - period when i >= period. Entry/re-entry remain unresolved.
  std::size_t period = 0;
  std::size_t firstOrdinalWithLocalPredecessor = 0;
  bool earlierOrdinalsNeedIncomingCase = false;
  // A bank relation is a physical candidate until predecessor/successor
  // occurrence matching, participation and child transport are proved.
  std::vector<std::size_t> sharedBankCandidates;
  std::string reason;
};
struct OriginalSupportRequest {
  bool hasWriteDelimitedCandidate = false;
  std::size_t producer = NoControlId, reuse = NoControlId, cell = 0;
  std::string reason;
};
// Complete original-only interpretation requested at a construction deadline.
// Each component keeps its own qualification: a may origin or structural
// frontier is never promoted to an exact generation or selected completion.
struct InterpretedRequirement {
  const DecodedRequirement *decoded = nullptr;
  OriginalOccurrenceInterpretation occurrence;
  const StorageLifecycle *sourceUse = nullptr, *targetUse = nullptr;
  // A missing incoming full writer is a separate, unresolved entry path; the
  // alternative-origin may-set does not silently cover it.
  bool mayHaveIncomingGeneration = true;
  OriginalContinuationQuery interval;
  OriginalSupportRequest support;
  OriginalBoundaryResult firstConflict, lastRelevantUse;
  const std::vector<StorageOrigin> *mayAlternativeSources = nullptr;
  bool alternativeGuardsQualified = false;
  // Shared conditional original-use expression. Acyclic formation alone is
  // not an exact occurrence or executable endpoint qualification.
  const FactoredUseResult *factoredUse = nullptr;
};

// Owns the immutable original structure and its construction-facing query services.
// The unchanged SyncInput and source function must outlive this result. Source
// subscriptions are installed by OriginalLifetimes before this result is read.
class ProgramAnalysis {
public:
  ProgramAnalysis(const SyncInput &input, OriginalStructure structure);
  ~ProgramAnalysis();
  ProgramAnalysis(const ProgramAnalysis &) = delete;
  ProgramAnalysis &operator=(const ProgramAnalysis &) = delete;

  // Preparation completeness; individual decoded fields may remain Unknown.
  bool complete() const;
  const std::string &reason() const;
  const OriginalStructure &structure() const { return original; }
  const SyncInput &translatedInput() const { return input; }
  const OriginalLifetimes &lifetimes() const { return storage; }
  const OccurrenceQueries &occurrences() const { return occurrenceQueries; }
  const PhysicalBankCorrespondence &bankRelation(std::size_t index) const;
  const std::vector<StorageOrigin> &alternativeSourcesFor(const DecodedRequirement &requirement) const;
  const std::vector<OriginalRequirement> &requirementsAt(std::size_t operation) const;
  const std::vector<OriginalRequirementId> &requirementsFromTo(PipelineType source,
                                                               PipelineType target) const;
  const std::vector<TypedOriginalRequirement> &typedRequirementsAt(std::size_t originalSite) const;
  const std::vector<std::pair<std::size_t, std::size_t>> &typedSubscriptionsAt(
      std::size_t sourceOperation) const;
  // Decode only an applicable indexed request. The immutable source positions
  // were subscribed before construction; decoding need not precede traversal.
  const DecodedRequirement &decodeAt(std::size_t operation, std::size_t index) const;
  // Lazily joins the shared original queries for a currently relevant request.
  // No selected ledger, event key, or predicted completion enters this result.
  const InterpretedRequirement &interpretAt(std::size_t operation, std::size_t index) const;
  FixedVisitCorrespondence fixedVisitFor(const DecodedRequirement &requirement) const;
  OriginalBoundaryResult firstConflict(const DecodedRequirement &requirement) const;
  OriginalBoundaryResult lastRelevantUse(const DecodedRequirement &requirement) const;
  // General may-frontier queries over a caller-qualified original interval.
  // Their Present result does not by itself establish guarded participation.
  PhysicalUseFrontier firstMayUse(const OriginalUseQuery &query) const;
  PhysicalUseFrontier lastMayUse(const OriginalUseQuery &query) const;
  const std::vector<SourceSubscription> &subscriptionsAt(std::size_t operation) const;
  OriginalAccessSummary all(const OriginalAllQuery &query) const;
  OriginalMayAfter mayAfter(const OriginalContinuationQuery &query) const;
  OriginalSupportInterval qualifySupport(const InterpretedRequirement &requirement) const;
  OriginalBoundaryResult qualifiedBoundary(const InterpretedRequirement &requirement,
                                           const OriginalSupportInterval &support,
                                           bool first) const;
  // A qualified original frontier expands to guarded legal payload cuts.
  // Selected-word positions and their causal source snapshots belong to Phase B.
  std::vector<OriginalEndpointCandidate> endpointCandidates(const OriginalBoundaryResult &boundary,
                                                            SourceMilestone::Side side) const;
  ParticipationExpression predicate(std::size_t id) const;
  GuardedReadFrontier readerFrontier(std::size_t id) const;
  bool guardAvailableAt(std::size_t predicateId, std::size_t operation) const;
  std::vector<OriginalParticipationDemand> participationDemands() const;

private:
  OriginalBoundaryResult boundary(const DecodedRequirement &requirement, bool first) const;
  const SyncInput &input;
  OriginalStructure original;
  OriginalLifetimes storage;
  OccurrenceQueries occurrenceQueries;
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::pto::frontiersynch
#endif
