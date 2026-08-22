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
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

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
  CanonicalSyncPlanBuilder(func::FuncOp func, unsigned eventIdMax)
      : func_(func), funcOperation_(func.getOperation()),
        eventIdMax_(eventIdMax),
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
  void preserveForwardCompletionRequirements();
  void reduceForwardDependencies();
  void preserveRecurrenceCompletionRequirements();
  void materializeSyncRequirements();
  void materializeBarriers();
  void materializeEvents();
  void materializeEventsFrom(ArrayRef<CanonicalDependency> dependencies,
                             std::vector<CanonicalEvent> &events);
  CanonicalEvent makeForwardEvent(std::size_t source, std::size_t target) const;
  CanonicalEvent
  makeRecurrenceEvent(const CanonicalDependency &dependency) const;
  void synthesizeL0OwnershipProtocols();
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
      ArrayRef<CanonicalDependency> requirements, bool diagnose = false) const;
  std::size_t countUncoveredRecurrenceRequirements(
      ArrayRef<CanonicalBarrier> barriers, ArrayRef<CanonicalEvent> events,
      ArrayRef<CanonicalDependency> requirements, bool diagnose) const;
  bool isRecurrenceVertexAvailable(const CanonicalDependency &requirement,
                                   std::size_t vertex,
                                   std::size_t nodeCount) const;
  bool eventsFitBudget(ArrayRef<CanonicalEvent> events) const;
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
                     Operation *recurrenceLoop = nullptr);
  void addAccessHazards(const CanonicalSyncNode &source,
                        const CanonicalSyncNode &target,
                        unsigned iterationDistance, Operation *loop,
                        bool compareSlots);
  void addRecurrenceAccessHazards(const CanonicalSyncNode &source,
                                  const CanonicalSyncNode &target,
                                  Operation *loop);
  bool memoryAliases(const CanonicalMemoryAccess &first,
                     const CanonicalMemoryAccess &second,
                     bool compareSlots = true) const;
  bool memoryAliasesAcrossIterations(const CanonicalMemoryAccess &first,
                                     const CanonicalMemoryAccess &second,
                                     Operation *loop,
                                     unsigned iterationDistance) const;
  bool rootsAreNoAlias(Value first, Value second) const;
  SlotRelation compareSlotsAcrossIterations(const CanonicalMemoryAccess &first,
                                            const CanonicalMemoryAccess &second,
                                            Operation *loop,
                                            unsigned iterationDistance) const;
  bool mayExecuteTogether(Operation *first, Operation *second) const;
  bool hasHardwareCompletion(PipelineType pipe) const;
  CanonicalAnchor getSetAnchor(Operation *source, Operation *target) const;
  CanonicalAnchor getWaitAnchor(Operation *source, Operation *target) const;
  std::size_t getAnchorPosition(const CanonicalAnchor &anchor) const;
  unsigned getRecurrenceWidth(const CanonicalDependency &dependency,
                              Value &setSlot, Value &waitSlot) const;
  void reserveHiddenEventIds();

  func::FuncOp func_;
  Operation *funcOperation_ = nullptr;
  unsigned eventIdMax_ = 0;
  CanonicalSyncPlan plan_;
  SyncIRs syncIR_;
  MemoryDependentAnalyzer memoryAnalyzer_;
  Buffer2MemInfoMap bufferMap_;
  PTOIRTranslator translator_;
  DenseMap<Operation *, SmallVector<std::size_t, 2>> operationNodes_;
  std::set<std::tuple<std::size_t, std::size_t, SyncGraphEdgeKind>>
      fixedEdgeKeys_;
  std::set<DependencyKey> dependencyKeys_;
  std::map<CanonicalEventDomainKey, std::set<unsigned>> reservedIds_;
  std::map<CanonicalEventDomainKey, CanonicalScarcityStats> scarcityStats_;
  std::set<std::pair<unsigned, unsigned>> noAliasArgPairs_;
  std::vector<CanonicalEvent> eventCandidates_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
