// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSISINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSISINTERNAL_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {
namespace canonical_sync_detail {

constexpr std::size_t kMaximumSlotCount = 16;

struct RegionContext {
  SyncCoverScopeId scope = 0;
  SyncCoverGuard guard;
};

struct ExtractedAccess {
  SyncCoverNodeId node = 0;
  Value base;
  Value root;
  AddressSpace space = AddressSpace::Zero;
  std::vector<std::uint64_t> addresses;
  std::uint64_t size = 0;
  bool knownPhysical = false;
  bool unknownRange = false;
  std::optional<std::uint64_t> rootRelativeOffset;
  SyncCoverStorageAccessMode mode = SyncCoverStorageAccessMode::Read;
  std::vector<SyncCoverStorageAccessId> graphAccesses;
};

struct HazardWitnessPhaseState {
  SyncCoverStorageWitnessId witness = 0;
  std::uint16_t raw = 0;
  std::uint16_t war = 0;
  std::uint16_t waw = 0;
};

struct OrdinalPairPhaseState {
  unsigned first = 0;
  unsigned second = 0;
  std::uint16_t sourcePhases = 1;
};

struct HazardWitnessPhaseCandidate {
  SyncCoverStorageWitnessId witness = 0;
  std::uint16_t sourcePhases = 0;
};

struct HazardWitnesses {
  std::vector<HazardWitnessPhaseState> states;
};

struct RecurrencePhaseOrbit {
  bool staticallyReachable = true;
  std::size_t period = 1;
  std::uint16_t sourceActivePhases = 1;
  std::uint16_t targetActivePhases = 1;
};

using RecurrencePhaseOrbitCacheKey =
    std::tuple<SyncCoverScopeId, std::vector<SyncCoverGuardLiteral>,
               std::vector<SyncCoverGuardLiteral>>;

using IssueFrontier = std::map<std::uint32_t, std::vector<SyncCoverNodeId>>;

struct IssueHistoryNode;
using IssueHistory = std::shared_ptr<const IssueHistoryNode>;
using IssueHistoryHeads = std::map<std::uint32_t, IssueHistory>;

/// Persistent issue history. Ordinary nodes prepend one record and control
/// joins add one shared union record, so copying path state never copies a
/// cumulative node vector.
struct IssueHistoryNode {
  std::optional<SyncCoverNodeId> issued;
  IssueHistory first;
  IssueHistory second;
};

struct FixedBarrierBoundary {
  Operation *operation = nullptr;
  std::shared_ptr<const SyncCoverGuard> guard;
  std::shared_ptr<const std::vector<SyncCoverNodeId>> sources;
  std::shared_ptr<const std::set<std::uint32_t>> remainingTargetResources;
};

struct IssueOrderState {
  std::shared_ptr<const IssueFrontier> frontier;
  std::shared_ptr<const IssueHistoryHeads> issued;
  std::shared_ptr<const std::vector<FixedBarrierBoundary>> fixedBarriers;
};

bool isCanonicalSyncOwned(Operation *operation);
bool isTransparentRegionOperation(Operation *operation);
bool isCompletionOrdered(
    std::uint32_t resource,
    const CanonicalSyncTargetCapabilities &capabilities);
bool canSignalDirectCompletion(std::uint32_t resource);
bool canSignalPrefixCompletion(
    std::uint32_t resource,
    const CanonicalSyncTargetCapabilities &capabilities);

class ProgramBuilder {
public:
  ProgramBuilder(func::FuncOp function,
                 const CanonicalSyncAnalysisOptions &options);
  FailureOr<CanonicalSyncProgram> build();

private:
  LogicalResult validateInput();
  LogicalResult extract();
  LogicalResult buildScopes();
  LogicalResult addRegion(Region &region, const RegionContext &context,
                          std::size_t timelineEnd);
  std::optional<SyncCoverScopeId>
  getNearestLoopScope(SyncCoverScopeId scope) const;
  LogicalResult addPeriodicControlEvidence(scf::IfOp conditional,
                                           SyncCoverControlId control,
                                           SyncCoverScopeId occurrenceScope);
  LogicalResult addSuccessorControlEvidence(scf::IfOp conditional,
                                            SyncCoverControlId control,
                                            SyncCoverScopeId occurrenceScope);
  LogicalResult buildNodesAndStorage();
  LogicalResult validateControlDataflow();
  LogicalResult refineLoopTimelines();
  LogicalResult indexSsaCompletionNodes();
  void collectHiddenEventReservations();
  void indexNodesByLoop();

  void appendAccesses(SyncCoverNodeId node,
                      ArrayRef<const BaseMemInfo *> memoryInfos, bool writes);
  LogicalResult materializeNodeAccesses(SyncCoverNodeId node);
  LogicalResult addConservativeAccess(ExtractedAccess &access,
                                      SyncCoverStorageDomainId domain,
                                      SyncCoverStorageAccessFamilyId family);
  LogicalResult buildStorageConflictIndex();
  bool gmAccessesAreNoAlias(const ExtractedAccess &first,
                            const ExtractedAccess &second) const;
  FailureOr<std::vector<OrdinalPairPhaseState>> getOrdinalPairs(
      const ExtractedAccess &first, const ExtractedAccess &second,
      Operation *loop, unsigned distance,
      const std::vector<std::size_t> *reachableSourcePhases = nullptr,
      std::size_t phasePeriod = 1);
  std::optional<std::int64_t> getIterationOffset(Operation *loop,
                                                 unsigned distance) const;

  bool consumePairInspections(std::size_t amount = 1);
  bool consumePairInspection() { return consumePairInspections(); }
  LogicalResult collectEnclosingLoopControls(
      Operation *operation, llvm::DenseMap<Value, Value> &inductionLowerBounds);
  LogicalResult walkSsaProvenance(
      Value seed, const llvm::DenseMap<Value, Value> *trackedLoopInductions,
      llvm::SetVector<SyncCoverNodeId> *producers,
      llvm::DenseSet<Value> &discovered, bool *reachesTrackedLoopInduction);
  LogicalResult addFixedIssueOrder();
  LogicalResult addCertifiedCompletionFrontiers();
  LogicalResult addRegionIssueOrder(Region &region, IssueOrderState &state);
  LogicalResult recordBlockingBarrierPrefixes(SyncCoverNodeId target,
                                              IssueOrderState &state);
  LogicalResult addIssueNode(SyncCoverNodeId target, IssueOrderState &state);
  LogicalResult addFixedBarrier(BarrierOp barrier, IssueOrderState &state);
  LogicalResult mergeIssueStates(IssueOrderState &target,
                                 const IssueOrderState &source);
  LogicalResult collectIssuedSources(const IssueHistory &history,
                                     const SyncCoverGuard &barrierGuard,
                                     std::vector<SyncCoverNodeId> &sources);
  LogicalResult addForwardDependencies();
  LogicalResult addRecurrenceDependencies();
  LogicalResult addTargetCompletionCertificates(
      const CanonicalSyncTargetCapabilities &capabilities);
  LogicalResult discoverBasicOwnershipCertificates(
      const CanonicalSyncTargetCapabilities &capabilities);
  bool isDemandImplicitlyComplete(SyncCoverNodeId source,
                                  SyncCoverNodeId target);
  bool hasIntrinsicMmadAccumulatorOrdering(
      SyncCoverNodeId source, SyncCoverNodeId target,
      const SyncCoverStorageAccess &sourceAccess,
      const SyncCoverStorageAccess &targetAccess);
  FailureOr<RecurrencePhaseOrbit>
  buildRecurrencePhaseOrbit(SyncCoverScopeId loopScope,
                            const SyncCoverGuard &sourceGuard,
                            const SyncCoverGuard &targetGuard);
  std::vector<std::size_t>
  getReachableRecurrenceSourcePhases(const RecurrencePhaseOrbit &orbit,
                                     unsigned distance) const;
  FailureOr<unsigned>
  maximumRecurrenceDistance(SyncCoverNodeId source, SyncCoverNodeId target,
                            const RecurrencePhaseOrbit &orbit);
  LogicalResult addMemoryHazards(
      SyncCoverNodeId source, SyncCoverNodeId target, Operation *loop,
      unsigned distance, SyncCoverScopeId recurrenceScope = 0,
      HazardWitnesses *covered = nullptr,
      const std::vector<std::size_t> *reachableSourcePhases = nullptr,
      std::size_t phasePeriod = 1);
  LogicalResult addDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                          SyncCoverScopeId scope, unsigned distance,
                          std::vector<SyncCoverDemandKind> kinds,
                          std::vector<SyncCoverStorageWitnessId> witnesses,
                          std::size_t originalDemandCount = 1);

  func::FuncOp function_;
  const CanonicalSyncAnalysisOptions &options_;
  CanonicalSyncTargetCapabilities targetCapabilities_;
  SyncCoverGraph graph_;
  SyncIRs syncIR_;
  Buffer2MemInfoMap bufferMap_;
  OperationMemInfoStorage operationMemory_;
  std::vector<CompoundInstanceElement *> compounds_;
  std::vector<CanonicalSyncNodeBinding> nodeBindings_;
  std::vector<CanonicalSyncScopeBinding> scopeBindings_;
  std::vector<CanonicalSyncControlBinding> controlBindings_;
  llvm::DenseMap<Region *, RegionContext> contexts_;
  llvm::DenseMap<Operation *, SmallVector<SyncCoverNodeId, 2>> operationNodes_;
  llvm::DenseMap<Value, SyncCoverNodeId> ssaCompletionNodes_;
  std::vector<std::pair<Operation *, SyncCoverScopeId>> loopScopes_;
  llvm::DenseMap<Operation *, SmallVector<SyncCoverNodeId, 16>> loopNodes_;
  std::vector<ExtractedAccess> extractedAccesses_;
  std::vector<std::vector<std::size_t>> nodeAccessIndices_;
  std::vector<std::vector<SyncCoverNodeId>> storageConflictPeers_;
  std::map<AddressSpace, SyncCoverStorageDomainId> storageDomains_;
  llvm::DenseMap<Value, SyncCoverStorageAccessFamilyId> storageFamilies_;
  SyncCoverStorageAccessFamilyId nextStorageFamily_ = 1;
  std::set<std::pair<unsigned, unsigned>> noAliasArguments_;
  CanonicalSyncEventReservations eventReservations_;
  CanonicalSyncOwnershipDiscoveryStatistics ownershipDiscoveryStatistics_;
  std::map<RecurrencePhaseOrbitCacheKey, RecurrencePhaseOrbit>
      recurrencePhaseOrbitCache_;
  std::size_t pairInspections_ = 0;
};

} // namespace canonical_sync_detail
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSISINTERNAL_H
