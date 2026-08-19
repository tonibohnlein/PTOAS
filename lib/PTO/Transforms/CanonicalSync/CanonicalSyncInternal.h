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

#include <map>
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

class CanonicalSyncPlanBuilder {
public:
  CanonicalSyncPlanBuilder(func::FuncOp func, unsigned eventIdMax)
      : func_(func), eventIdMax_(eventIdMax),
        translator_(syncIR_, memoryAnalyzer_, bufferMap_, func,
                    SyncAnalysisMode::NORMALSYNC) {}

  FailureOr<CanonicalSyncPlan> build();

private:
  LogicalResult validateInput();
  LogicalResult collectNodes();
  LogicalResult validateModeledEffects();
  LogicalResult addDependencies();
  void addFixedEdge(std::size_t source, std::size_t target,
                    SyncGraphEdgeKind kind);
  void addIssueOrderEdges();
  void addMemoryDependencies();
  LogicalResult addSSAAndRecurrenceDependencies();
  void reduceForwardDependencies();
  void materializeSyncRequirements();
  LogicalResult allocateEvents();

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
  SlotRelation compareSlotsAcrossIterations(const CanonicalMemoryAccess &first,
                                            const CanonicalMemoryAccess &second,
                                            Operation *loop,
                                            unsigned iterationDistance) const;
  bool mayExecuteTogether(Operation *first, Operation *second) const;
  bool hasHardwareCompletion(PipelineType pipe);
  CanonicalAnchor getSetAnchor(Operation *source, Operation *target) const;
  CanonicalAnchor getWaitAnchor(Operation *source, Operation *target) const;
  std::size_t getAnchorPosition(const CanonicalAnchor &anchor) const;
  unsigned getRecurrenceWidth(const CanonicalDependency &dependency,
                              Value &setSlot, Value &waitSlot) const;
  void reserveHiddenEventIds();

  func::FuncOp func_;
  unsigned eventIdMax_ = 0;
  CanonicalSyncPlan plan_;
  SyncIRs syncIR_;
  MemoryDependentAnalyzer memoryAnalyzer_;
  Buffer2MemInfoMap bufferMap_;
  PTOIRTranslator translator_;
  DenseMap<Operation *, SmallVector<std::size_t, 2>> operationNodes_;
  std::map<CanonicalEventDomainKey, std::set<unsigned>> reservedIds_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
