// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverCandidateIndex.h"
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

struct SyncCoverSlotLifecycleResult;
struct SyncCoverSlotProtocolResult;

struct CanonicalEventDomainKey {
  PipelineType source = PipelineType::PIPE_UNASSIGNED;
  PipelineType target = PipelineType::PIPE_UNASSIGNED;

  bool operator<(const CanonicalEventDomainKey &other) const {
    return std::tie(source, target) < std::tie(other.source, other.target);
  }

  bool operator==(const CanonicalEventDomainKey &other) const {
    return source == other.source && target == other.target;
  }
};

struct CanonicalEventBundleCandidate {
  std::size_t id = 0;
  std::size_t protocolIdentity = 0;
  CanonicalEventBundleKind kind = CanonicalEventBundleKind::Standalone;
  CanonicalOwnershipProtocolKind ownershipProtocol =
      CanonicalOwnershipProtocolKind::RoundTrip;
  SmallVector<CanonicalEvent, 2> events;
  SmallVector<std::size_t, 2> conflicts;
};

struct CanonicalBarrierCandidate {
  std::size_t id = 0;
  CanonicalBarrier barrier;
  SmallVector<std::size_t, 2> conflicts;
};

struct CanonicalMechanismUniverse {
  std::vector<CanonicalBarrierCandidate> barriers;
  std::vector<CanonicalEventBundleCandidate> eventBundles;
};

std::vector<CanonicalEventBundleCandidate>
buildCanonicalEventBundles(ArrayRef<CanonicalEvent> events);

std::vector<CanonicalEvent>
flattenCanonicalEventBundles(ArrayRef<CanonicalEventBundleCandidate> bundles);

bool canonicalEventBundleProjectionMatches(
    ArrayRef<CanonicalEventBundleCandidate> bundles,
    ArrayRef<CanonicalEvent> events);

bool verifyCanonicalOwnershipEventPair(const CanonicalOwnershipCycle &cycle,
                                       ArrayRef<const CanonicalEvent *> events);

bool verifyCanonicalCompositeOwnershipBundle(
    const CanonicalEventBundleCandidate &bundle,
    ArrayRef<CanonicalOwnershipCycle> cycles,
    ArrayRef<CanonicalSyncNode> nodes);

bool verifyCanonicalAlternatingPathMapping(
    const CanonicalOwnershipCycle &cycle);

std::pair<CanonicalEvent, CanonicalEvent>
buildCanonicalOwnershipProtocols(const CanonicalOwnershipCycle &cycle);

class CanonicalSyncPlanBuilder {
public:
  CanonicalSyncPlanBuilder(func::FuncOp func,
                           const CanonicalSyncBuildOptions &options)
      : func_(func), funcOperation_(func.getOperation()),
        eventIdMax_(options.eventIdMax), gmAliasPolicy_(options.gmAliasPolicy),
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
  void materializeEventsFrom(ArrayRef<CanonicalDependency> dependencies,
                             std::vector<CanonicalEvent> &events);
  CanonicalEvent makeForwardEvent(std::size_t source, std::size_t target) const;
  CanonicalEvent
  makeRecurrenceEvent(const CanonicalDependency &dependency) const;
  void analyzeOwnershipCycles();
  std::optional<CanonicalEventBundleCandidate>
  buildOwnershipEventBundle(const CanonicalOwnershipCycle &cycle,
                            CanonicalOwnershipProtocolKind protocol);
  std::optional<CanonicalEventBundleCandidate>
  buildCompositeOwnershipEventBundle();
  void buildMechanismUniverse();
  LogicalResult verifyEventProtocols(ArrayRef<CanonicalEvent> events,
                                     bool requireAllocation,
                                     bool diagnose) const;
  bool verifyEventProtocol(const CanonicalEvent &event, bool requireAllocation,
                           bool diagnose) const;
  std::vector<SyncGraphEdge>
  buildBarrierCompletionEdges(ArrayRef<CanonicalBarrier> barriers) const;
  bool isAnchorGuaranteedForRequirement(const CanonicalAnchor &anchor,
                                        std::size_t source,
                                        std::size_t target) const;
  bool isRecurrenceAnchorGuaranteedForEndpoint(const CanonicalAnchor &anchor,
                                               unsigned anchorOccurrence,
                                               std::size_t endpoint,
                                               unsigned endpointOccurrence,
                                               Operation *loop) const;
  LogicalResult buildCoveringGraph();
  LogicalResult materializeCoveringSelection();
  void initializeForwardProtocol(CanonicalEvent &event) const;
  void initializeRecurrenceProtocol(CanonicalEvent &event) const;
  void deriveEventInterval(CanonicalEvent &event) const;

  void
  addDependency(std::size_t source, std::size_t target,
                CanonicalDependencyKind kind, unsigned iterationDistance = 0,
                Operation *recurrenceLoop = nullptr, bool activeWitness = true,
                CanonicalStorageProvenance storageProvenance =
                    CanonicalStorageProvenance::NotApplicable,
                ArrayRef<CanonicalMemoryHazardWitness> storageWitnesses = {});
  void addAccessHazards(const CanonicalSyncNode &source,
                        const CanonicalSyncNode &target,
                        unsigned iterationDistance, Operation *loop,
                        bool compareSlots, bool honorNoAlias,
                        bool activeWitness, bool captureStorageProvenance);
  void addRecurrenceAccessHazards(const CanonicalSyncNode &source,
                                  const CanonicalSyncNode &target,
                                  Operation *loop);
  bool memoryAliases(const CanonicalMemoryAccess &first,
                     const CanonicalMemoryAccess &second,
                     bool compareSlots = true, bool honorNoAlias = true) const;
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
  std::set<std::pair<unsigned, unsigned>> noAliasArgPairs_;
  CanonicalMechanismUniverse mechanismUniverse_;
  std::vector<CanonicalEventBundleCandidate> selectedEventBundles_;
};

LogicalResult runCanonicalSyncCoveringSelection(
    func::FuncOp func, const CanonicalSyncPlan &plan,
    const CanonicalMechanismUniverse &candidateUniverse,
    ArrayRef<CanonicalEventBundleCandidate> selectedEventBundles,
    unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds,
    SyncCoverGraph &graph, const SyncCoverCandidateIndex &candidateIndex,
    const SyncCoverSlotLifecycleResult &slotLifecycles,
    const SyncCoverSlotProtocolResult &slotProtocols,
    ArrayRef<SyncCoverDemandId> activeDemands,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
    std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
        getBarrierCompletionEdges,
    std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols,
    CanonicalSyncCoveringSnapshot &snapshot);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
