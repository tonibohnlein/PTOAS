// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {

struct CanonicalEventDomainKey {
  PipelineType source = PipelineType::PIPE_UNASSIGNED;
  PipelineType target = PipelineType::PIPE_UNASSIGNED;

  bool operator<(const CanonicalEventDomainKey &other) const {
    return std::tie(source, target) < std::tie(other.source, other.target);
  }
};

struct CanonicalScarcityStats {
  std::size_t originalEventCount = 0;
  unsigned originalColorCount = 0;
  std::size_t serializationCost = 0;
  std::uint64_t originalCriticalPathWeight = 0;
  std::uint64_t criticalPathWeight = 0;
};

struct CanonicalEventBundleCandidate {
  std::size_t id = 0;
  std::size_t protocolIdentity = 0;
  CanonicalEventBundleKind kind = CanonicalEventBundleKind::Standalone;
  SmallVector<CanonicalEvent, 2> events;
  SmallVector<std::size_t, 2> conflicts;
  std::optional<CanonicalDependency> completionWitness;
  SmallVector<CanonicalDependency, 4> originRequirements;
  bool hasCompleteOriginProvenance = false;
};

struct CanonicalBarrierCandidate {
  std::size_t id = 0;
  CanonicalBarrier barrier;
  SmallVector<std::size_t, 2> conflicts;
  SmallVector<CanonicalDependency, 4> originRequirements;
  bool hasCompleteOriginProvenance = false;
};

struct CanonicalMechanismUniverse {
  std::vector<CanonicalBarrierCandidate> barriers;
  std::vector<CanonicalEventBundleCandidate> eventBundles;
};

struct CanonicalMechanismPlanScore {
  std::size_t usefulOwnershipBundles = 0;
  std::vector<std::size_t> ownershipSignature;
  std::vector<std::size_t> dynamicActionProfile;
  std::vector<std::size_t> barrierActionProfile;
  std::size_t waitDistance = 0;
  std::size_t intervalSpan = 0;
  unsigned peakColorPressure = 0;
  std::size_t directedDomains = 0;
  std::size_t barrierCount = 0;
  std::vector<std::size_t> candidateSignature;
};

struct CanonicalAffectedSliceEvictionMechanism {
  CanonicalSelectionMechanismRef mechanism;
  std::vector<std::size_t> actionProfile;
  std::vector<std::size_t> barrierProfile;
};

struct CanonicalAffectedSliceEvictionSeed {
  SmallVector<CanonicalSelectionMechanismRef, 2> mechanisms;
  std::vector<std::size_t> actionProfile;
  std::vector<std::size_t> barrierProfile;
};

struct CanonicalAffectedSliceSearchCandidate {
  CanonicalSelectionMechanismRef mechanism;
  std::vector<std::size_t> coveredRequirements;
};

struct CanonicalAffectedSliceSearchEvaluation {
  std::vector<std::size_t> uncoveredRequirements;
  CanonicalMechanismPlanScore score;
};

struct CanonicalAffectedSliceSearchComplete {
  std::vector<CanonicalSelectionMechanismRef> mechanisms;
  CanonicalMechanismPlanScore score;
};

struct CanonicalAffectedSliceSearchResult {
  std::vector<CanonicalAffectedSliceSearchComplete> complete;
  std::size_t evaluations = 0;
  bool budgetExhausted = false;
};

std::vector<CanonicalEventBundleCandidate>
buildCanonicalEventBundles(ArrayRef<CanonicalEvent> events);

std::vector<CanonicalEvent>
flattenCanonicalEventBundles(ArrayRef<CanonicalEventBundleCandidate> bundles);

bool canonicalEventBundleProjectionMatches(
    ArrayRef<CanonicalEventBundleCandidate> bundles,
    ArrayRef<CanonicalEvent> events);

LogicalResult restoreCanonicalEventBundleIdentities(
    std::vector<CanonicalEventBundleCandidate> &bundles,
    ArrayRef<CanonicalEventBundleCandidate> knownBundles,
    std::size_t &nextFreshId);

bool verifyCanonicalSyntheticRoundTripBundle(
    ArrayRef<const CanonicalEvent *> events);

bool verifyCanonicalSyntheticRoundTripWitness(
    ArrayRef<const CanonicalEvent *> events,
    const CanonicalDependency &requirement);

bool canonicalEventBundlesHaveNoConflicts(
    ArrayRef<CanonicalEventBundleCandidate> bundles);

bool canonicalDiagnosticEventBundlesEquivalent(
    const CanonicalEventBundleCandidate &first,
    const CanonicalEventBundleCandidate &second,
    ArrayRef<CanonicalEventBundleCandidate> universe);

bool canonicalDiagnosticEventBundleMatchesSelected(
    const CanonicalEventBundleCandidate &candidate,
    ArrayRef<CanonicalEventBundleCandidate> selected,
    ArrayRef<CanonicalEventBundleCandidate> universe);

bool exchangeCanonicalEventBundleCandidate(
    std::vector<CanonicalEventBundleCandidate> &selected,
    const CanonicalEventBundleCandidate &candidate);

bool appendCanonicalEventBundleCandidate(
    std::vector<CanonicalEventBundleCandidate> &selected,
    const CanonicalEventBundleCandidate &candidate);

bool canonicalMechanismOriginsAreInactive(
    bool hasCompleteOriginProvenance,
    ArrayRef<CanonicalDependency> originRequirements,
    ArrayRef<CanonicalDependency> activeRequirements);

std::vector<CanonicalAffectedSliceEvictionSeed>
buildCanonicalAffectedSliceEvictionSeeds(
    ArrayRef<CanonicalAffectedSliceEvictionMechanism> mechanisms,
    std::size_t exhaustiveMechanismThreshold,
    std::size_t mechanismFrontierLimit, std::size_t seedLimit);

CanonicalAffectedSliceSearchResult searchCanonicalAffectedSliceCandidates(
    ArrayRef<CanonicalAffectedSliceSearchCandidate> candidates,
    std::size_t requirementCount,
    const CanonicalMechanismPlanScore &initialScore, std::size_t beamWidth,
    std::size_t depthLimit, std::size_t evaluationLimit,
    std::size_t completeCandidateLimit,
    llvm::function_ref<std::optional<CanonicalAffectedSliceSearchEvaluation>(
        ArrayRef<CanonicalSelectionMechanismRef>)>
        evaluate);

std::size_t calculateCanonicalEventColorOverflow(
    ArrayRef<CanonicalEvent> events, unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds);

std::vector<std::size_t> buildCanonicalBarrierActionProfile(
    ArrayRef<CanonicalBarrier> barriers, std::size_t maxLoopDepth);

bool canonicalMechanismPlanScoreLess(
    const CanonicalMechanismPlanScore &first,
    const CanonicalMechanismPlanScore &second);

SmallVector<const CanonicalEventBundleCandidate *, 16>
selectCanonicalEventCandidateFrontier(
    ArrayRef<const CanonicalEventBundleCandidate *> candidates,
    std::size_t limit);

bool verifyCanonicalOwnershipEventPair(
    const CanonicalOwnershipCycle &cycle,
    ArrayRef<const CanonicalEvent *> events);

bool verifyCanonicalAlternatingPathMapping(
    const CanonicalOwnershipCycle &cycle);

std::pair<CanonicalEvent, CanonicalEvent>
buildCanonicalOwnershipProtocols(const CanonicalOwnershipCycle &cycle);

bool tryCommitCanonicalOwnershipCandidate(
    std::vector<CanonicalEvent> &acceptedOwnership,
    std::vector<CanonicalBarrier> &currentBarriers,
    std::vector<CanonicalEvent> &currentEvents, CanonicalEvent ready,
    CanonicalEvent release, llvm::function_ref<bool()> evaluate);

class CanonicalSyncLatencyContext {
public:
  CanonicalSyncLatencyContext(const CanonicalSyncPlan &plan,
                              ArrayRef<CanonicalEvent> domainEvents,
                              const CanonicalEventDomainKey &domain);

  std::uint64_t
  calculateCriticalPathWeight(ArrayRef<CanonicalEvent> domainEvents) const;

private:
  struct BlockGraph {
    std::vector<std::uint64_t> weights;
    std::vector<SyncGraphEdge> baseEdges;
  };

  using NodeLocation = std::pair<std::size_t, std::size_t>;

  std::optional<std::pair<std::size_t, SyncGraphEdge>>
  localizeEdge(std::size_t source, std::size_t target) const;
  void addNodes(const CanonicalSyncPlan &plan,
                const std::set<Block *> &affectedBlocks);
  void addBaseEdge(std::size_t source, std::size_t target);

  std::vector<BlockGraph> blocks_;
  std::map<std::size_t, NodeLocation> nodeLocations_;
};

class CanonicalSyncPlanBuilder {
public:
  CanonicalSyncPlanBuilder(func::FuncOp func, unsigned eventIdMax,
                           CanonicalGMAliasPolicy gmAliasPolicy,
                           const CanonicalSelectionDiagnosticRequest
                               *diagnosticRequest)
      : func_(func), funcOperation_(func.getOperation()),
        eventIdMax_(eventIdMax),
        gmAliasPolicy_(gmAliasPolicy),
        selectionDiagnosticsEnabled_(diagnosticRequest != nullptr),
        diagnosticRequest_(diagnosticRequest ? *diagnosticRequest
                                             : CanonicalSelectionDiagnosticRequest{}),
        translator_(syncIR_, memoryAnalyzer_, bufferMap_, func,
                    SyncAnalysisMode::CANONICALSYNC) {}

  FailureOr<CanonicalSyncPlan> build();

private:
  struct DependencyKey {
    std::size_t source = 0;
    std::size_t target = 0;
    CanonicalDependencyKind kind = CanonicalDependencyKind::SSA;
    unsigned iterationDistance = 0;
    Operation *recurrenceLoop = nullptr;

    bool operator<(const DependencyKey &other) const {
      const auto key = std::tie(source, target, kind, iterationDistance);
      const auto otherKey = std::tie(other.source, other.target, other.kind,
                                     other.iterationDistance);
      if (key != otherKey) {
        return key < otherKey;
      }
      return std::less<Operation *>{}(recurrenceLoop, other.recurrenceLoop);
    }
  };

  LogicalResult validateInput();
  LogicalResult parseNoAliasPairs();
  LogicalResult collectNodes();
  LogicalResult validateModeledEffects();
  LogicalResult addDependencies();
  void addFixedEdge(std::size_t source, std::size_t target,
                    SyncGraphEdgeKind kind);
  void addIssueOrderEdges();
  void addMemoryDependencies();
  LogicalResult addSSAAndRecurrenceDependencies();
  void discardImpossibleRecurrences();
  void preserveConservativeCompletionRequirements();
  void preserveForwardCompletionRequirements();
  void reduceForwardDependencies();
  void preserveRecurrenceCompletionRequirements();
  LogicalResult materializeSyncRequirements();
  void materializeBarriers();
  void materializeEvents();
  void materializeEventsFrom(ArrayRef<CanonicalDependency> dependencies,
                             std::vector<CanonicalEvent> &events);
  CanonicalEvent makeForwardEvent(std::size_t source, std::size_t target) const;
  CanonicalEvent
  makeRecurrenceEvent(const CanonicalDependency &dependency) const;
  void analyzeOwnershipCycles();
  std::optional<CanonicalEventBundleCandidate>
  buildOwnershipEventBundle(const CanonicalOwnershipCycle &cycle);
  void buildMechanismUniverse();
  LogicalResult refreshSelectedEventBundles();
  bool tryBuildConservativeIncumbent(
      std::vector<CanonicalBarrier> &barriers,
      std::vector<CanonicalEvent> &events);
  void removeRedundantMechanisms();
  LogicalResult optimizeMechanismSelection();
  void optimizeAffectedSliceExchanges(
      std::vector<CanonicalBarrier> &incumbentBarriers,
      std::vector<CanonicalEventBundleCandidate> &incumbentBundles) const;
  LogicalResult buildSelectionDiagnostics();
  void synthesizeOwnershipProtocols();
  LogicalResult verifyEventProtocols(ArrayRef<CanonicalEvent> events,
                                     bool requireAllocation,
                                     bool diagnose) const;
  bool verifyEventProtocol(const CanonicalEvent &event, bool requireAllocation,
                           bool diagnose) const;
  void optimizeBarriers();
  std::vector<SyncGraphEdge>
  buildBarrierCompletionEdges(ArrayRef<CanonicalBarrier> barriers) const;
  std::vector<SyncGraphEdge>
  buildEventCompletionEdges(ArrayRef<CanonicalEvent> events) const;
  std::vector<CanonicalEvent>
  selectRequiredEvents(ArrayRef<CanonicalBarrier> barriers,
                       ArrayRef<CanonicalEvent> candidates) const;
  bool isForwardVertexAvailable(const CanonicalDependency &requirement,
                                std::size_t vertex) const;
  bool isAnchorGuaranteedForRequirement(const CanonicalAnchor &anchor,
                                        std::size_t source,
                                        std::size_t target) const;
  bool isRecurrenceAnchorGuaranteedForEndpoint(const CanonicalAnchor &anchor,
                                               unsigned anchorOccurrence,
                                               std::size_t endpoint,
                                               unsigned endpointOccurrence,
                                               Operation *loop) const;
  bool planCoversRequirements(ArrayRef<CanonicalBarrier> barriers,
                              ArrayRef<CanonicalEvent> events,
                              bool diagnose = false) const;
  std::size_t countUncoveredRequirements(
      ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
      ArrayRef<CanonicalDependency> requirements, bool diagnose = false,
      SmallVectorImpl<std::size_t> *uncoveredRequirements = nullptr) const;
  std::size_t countUncoveredRecurrenceRequirements(
      ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
      ArrayRef<CanonicalDependency> requirements, bool diagnose,
      SmallVectorImpl<std::size_t> *uncoveredRequirements) const;
  bool isVacuousOwnedAlternatingRecurrence(
      ArrayRef<const CanonicalOwnershipCycle *> cycles,
      const CanonicalDependency &requirement) const;
  bool isRecurrenceVertexAvailable(const CanonicalDependency &requirement,
                                   std::size_t vertex,
                                   std::size_t nodeCount) const;
  bool eventsFitBudget(ArrayRef<CanonicalEvent> events) const;
  bool isCandidatePlanFeasible(
      ArrayRef<CanonicalBarrier> barriers,
      ArrayRef<CanonicalEventBundleCandidate> eventBundles,
      ArrayRef<CanonicalDependency> requirements,
      bool diagnose = false) const;
  bool isCandidatePlanWellFormed(
      ArrayRef<CanonicalBarrier> barriers,
      ArrayRef<CanonicalEventBundleCandidate> eventBundles,
      ArrayRef<CanonicalDependency> requirements,
      bool diagnose = false) const;
  bool bootstrapFeasibleMechanismPlan(
      std::vector<CanonicalBarrier> &barriers,
      std::vector<CanonicalEventBundleCandidate> &eventBundles) const;
  CanonicalMechanismPlanScore scoreCandidatePlan(
      ArrayRef<CanonicalBarrier> barriers,
      ArrayRef<CanonicalEventBundleCandidate> eventBundles) const;
  LogicalResult verifyFinalPlan();
  LogicalResult repairEventScarcity();
  LogicalResult repairEventDomain(const CanonicalEventDomainKey &key,
                                  unsigned availableIds);
  std::optional<CanonicalEvent>
  coalesceForwardEvents(ArrayRef<CanonicalEvent> events) const;
  bool coversCoalescedEvents(const CanonicalEvent &candidate,
                             ArrayRef<CanonicalEvent> originals) const;
  LogicalResult allocateEvents();
  void initializeForwardProtocol(CanonicalEvent &event) const;
  void initializeRecurrenceProtocol(CanonicalEvent &event) const;
  void deriveEventInterval(CanonicalEvent &event) const;

  void addDependency(std::size_t source, std::size_t target,
                     CanonicalDependencyKind kind,
                     unsigned iterationDistance = 0,
                     Operation *recurrenceLoop = nullptr,
                     bool activeWitness = true);
  void addAccessHazards(const CanonicalSyncNode &source,
                        const CanonicalSyncNode &target,
                        unsigned iterationDistance, Operation *loop,
                        bool compareSlots, bool honorNoAlias,
                        bool activeWitness);
  void addRecurrenceAccessHazards(const CanonicalSyncNode &source,
                                  const CanonicalSyncNode &target,
                                  Operation *loop);
  bool memoryAliases(const CanonicalMemoryAccess &first,
                     const CanonicalMemoryAccess &second,
                     bool compareSlots = true,
                     bool honorNoAlias = true) const;
  bool memoryAliasesAcrossIterations(const CanonicalMemoryAccess &first,
                                     const CanonicalMemoryAccess &second,
                                     Operation *loop,
                                     unsigned iterationDistance,
                                     bool honorNoAlias = true) const;
  bool rootsAreNoAlias(Value first, Value second) const;
  SlotRelation compareSlotsAcrossIterations(const CanonicalMemoryAccess &first,
                                            const CanonicalMemoryAccess &second,
                                            Operation *loop,
                                            unsigned iterationDistance) const;
  bool mayExecuteTogether(Operation *first, Operation *second) const;
  bool hasHardwareCompletion(PipelineType pipe) const;
  bool hasIntrinsicMmadAccumulatorOrdering(
      const CanonicalDependency &dependency) const;
  CanonicalAnchor getSetAnchor(Operation *source, Operation *target) const;
  CanonicalAnchor getWaitAnchor(Operation *source, Operation *target) const;
  std::size_t getAnchorPosition(const CanonicalAnchor &anchor) const;
  unsigned getRecurrenceWidth(const CanonicalDependency &dependency,
                              Value &setSlot, Value &waitSlot) const;
  void reserveHiddenEventIds();

  func::FuncOp func_;
  Operation *funcOperation_ = nullptr;
  unsigned eventIdMax_ = 0;
  CanonicalGMAliasPolicy gmAliasPolicy_ = CanonicalGMAliasPolicy::MayAlias;
  bool selectionDiagnosticsEnabled_ = false;
  CanonicalSelectionDiagnosticRequest diagnosticRequest_;
  CanonicalSyncPlan plan_;
  SyncIRs syncIR_;
  MemoryDependentAnalyzer memoryAnalyzer_;
  Buffer2MemInfoMap bufferMap_;
  PTOIRTranslator translator_;
  DenseMap<Operation *, SmallVector<std::size_t, 2>> operationNodes_;
  std::set<std::tuple<std::size_t, std::size_t, SyncGraphEdgeKind>>
      fixedEdgeKeys_;
  std::map<DependencyKey, std::size_t> dependencyIndices_;
  std::map<CanonicalEventDomainKey, std::set<unsigned>> reservedIds_;
  std::map<CanonicalEventDomainKey, CanonicalScarcityStats> scarcityStats_;
  std::set<std::pair<unsigned, unsigned>> noAliasArgPairs_;
  std::vector<CanonicalEvent> eventCandidates_;
  CanonicalMechanismUniverse mechanismUniverse_;
  std::vector<CanonicalEventBundleCandidate> selectedEventBundles_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
