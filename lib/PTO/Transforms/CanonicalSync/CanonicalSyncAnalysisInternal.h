// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

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
#include <optional>
#include <set>
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
  SyncCoverStorageAccessMode mode = SyncCoverStorageAccessMode::Read;
  std::vector<SyncCoverStorageAccessId> graphAccesses;
};

struct HazardKinds {
  bool raw = false;
  bool war = false;
  bool waw = false;
};

using IssueFrontier = std::map<std::uint32_t, std::vector<SyncCoverNodeId>>;

bool isTransparentRegionOperation(Operation *operation);
bool isCompletionOrdered(std::uint32_t resource, Operation *operation);
bool canSignalDirectCompletion(std::uint32_t resource);

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
  LogicalResult buildNodesAndStorage();
  LogicalResult validateControlDataflow();
  LogicalResult refineLoopTimelines();
  void collectHiddenEventReservations();
  void indexNodesByLoop();

  void appendAccesses(SyncCoverNodeId node,
                      ArrayRef<const BaseMemInfo *> memoryInfos, bool writes);
  LogicalResult materializeNodeAccesses(SyncCoverNodeId node);
  LogicalResult addConservativeAccess(ExtractedAccess &access,
                                      SyncCoverStorageDomainId domain,
                                      SyncCoverStorageAccessFamilyId family);
  bool gmAccessesAreNoAlias(const ExtractedAccess &first,
                            const ExtractedAccess &second) const;
  FailureOr<std::vector<std::pair<unsigned, unsigned>>>
  getOrdinalPairs(const ExtractedAccess &first, const ExtractedAccess &second,
                  Operation *loop, unsigned distance);
  std::optional<std::int64_t> getIterationOffset(Operation *loop,
                                                 unsigned distance) const;

  bool consumePairInspection();
  LogicalResult addFixedIssueOrder();
  LogicalResult addRegionIssueOrder(Region &region, IssueFrontier &frontier);
  LogicalResult addIssueNode(SyncCoverNodeId target, IssueFrontier &frontier);
  static void mergeFrontiers(IssueFrontier &target,
                             const IssueFrontier &source);
  LogicalResult addForwardDependencies();
  LogicalResult addRecurrenceDependencies();
  bool isDemandImplicitlyComplete(SyncCoverNodeId source,
                                  SyncCoverNodeId target);
  void collectScheduledProducers(Value value,
                                 llvm::SetVector<SyncCoverNodeId> &producers,
                                 llvm::DenseSet<Value> &visited) const;
  bool hasIntrinsicMmadAccumulatorOrdering(
      SyncCoverNodeId source, SyncCoverNodeId target,
      const SyncCoverStorageAccess &sourceAccess,
      const SyncCoverStorageAccess &targetAccess);
  unsigned maximumRecurrenceDistance(SyncCoverNodeId source,
                                     SyncCoverNodeId target) const;
  FailureOr<HazardKinds> addMemoryHazards(SyncCoverNodeId source,
                                          SyncCoverNodeId target,
                                          Operation *loop, unsigned distance,
                                          SyncCoverScopeId recurrenceScope = 0,
                                          HazardKinds alreadyCovered = {});
  LogicalResult addDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                          SyncCoverScopeId scope, unsigned distance,
                          SyncCoverDemandKind kind,
                          std::vector<SyncCoverStorageWitnessId> witnesses);

  func::FuncOp function_;
  const CanonicalSyncAnalysisOptions &options_;
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
  std::vector<std::pair<Operation *, SyncCoverScopeId>> loopScopes_;
  llvm::DenseMap<Operation *, SmallVector<SyncCoverNodeId, 16>> loopNodes_;
  std::vector<ExtractedAccess> extractedAccesses_;
  std::vector<std::vector<std::size_t>> nodeAccessIndices_;
  std::map<AddressSpace, SyncCoverStorageDomainId> storageDomains_;
  llvm::DenseMap<Value, SyncCoverStorageAccessFamilyId> storageFamilies_;
  SyncCoverStorageAccessFamilyId nextStorageFamily_ = 1;
  std::set<std::pair<unsigned, unsigned>> noAliasArguments_;
  CanonicalSyncEventReservations eventReservations_;
  std::size_t pairInspections_ = 0;
};

} // namespace canonical_sync_detail
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSISINTERNAL_H
