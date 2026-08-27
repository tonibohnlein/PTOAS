// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSyncAnalysis.h - Lean MLIR graph adapter ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSIS_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSIS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {

enum class CanonicalSyncGmAliasPolicy : std::uint8_t {
  MayAlias,
  DistinctArgumentsNoAlias,
};

struct CanonicalSyncAnalysisOptions {
  CanonicalSyncGmAliasPolicy gmAliasPolicy =
      CanonicalSyncGmAliasPolicy::MayAlias;
  std::size_t maximumNodes = 1U << 16;
  std::size_t maximumScopes = 1U << 14;
  std::size_t maximumControls = 1U << 14;
  std::size_t maximumStorageAccesses = 1U << 20;
  std::size_t maximumPairInspections = 1U << 24;
};

struct CanonicalSyncNodeBinding {
  Operation *operation = nullptr;
  int macroPhase = -1;
};

struct CanonicalSyncScopeBinding {
  Operation *owner = nullptr;
  Region *region = nullptr;
};

using CanonicalSyncEventReservations =
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<unsigned>>;

/// One authoritative synchronization graph plus the minimal MLIR side tables
/// needed to materialize graph anchors. The side tables contain no copied
/// dependency, mechanism, cost, or selection state. Their raw MLIR bindings
/// remain valid only while the function's operation and region structure is
/// unchanged.
class CanonicalSyncProgram {
public:
  CanonicalSyncProgram() = default;
  CanonicalSyncProgram(CanonicalSyncProgram &&) = default;
  CanonicalSyncProgram &operator=(CanonicalSyncProgram &&) = default;
  CanonicalSyncProgram(const CanonicalSyncProgram &) = delete;
  CanonicalSyncProgram &operator=(const CanonicalSyncProgram &) = delete;
  CanonicalSyncProgram(func::FuncOp function, SyncCoverGraph graph,
                       std::vector<CanonicalSyncNodeBinding> nodeBindings,
                       std::vector<CanonicalSyncScopeBinding> scopeBindings,
                       std::vector<AddressSpace> storageSpaces,
                       CanonicalSyncEventReservations eventReservations)
      : function_(function), graph_(std::move(graph)),
        nodeBindings_(std::move(nodeBindings)),
        scopeBindings_(std::move(scopeBindings)),
        storageSpaces_(std::move(storageSpaces)),
        eventReservations_(std::move(eventReservations)) {}

  SyncCoverGraph &getGraph() { return graph_; }
  const SyncCoverGraph &getGraph() const { return graph_; }
  func::FuncOp getFunction() const { return function_; }
  const std::vector<CanonicalSyncNodeBinding> &getNodeBindings() const {
    return nodeBindings_;
  }
  const std::vector<CanonicalSyncScopeBinding> &getScopeBindings() const {
    return scopeBindings_;
  }
  const std::vector<AddressSpace> &getStorageSpaces() const {
    return storageSpaces_;
  }
  const CanonicalSyncEventReservations &getEventReservations() const {
    return eventReservations_;
  }

private:
  func::FuncOp function_;
  SyncCoverGraph graph_;
  std::vector<CanonicalSyncNodeBinding> nodeBindings_;
  std::vector<CanonicalSyncScopeBinding> scopeBindings_;
  std::vector<AddressSpace> storageSpaces_;
  CanonicalSyncEventReservations eventReservations_;
};

FailureOr<CanonicalSyncProgram>
buildCanonicalSyncProgram(func::FuncOp function,
                          const CanonicalSyncAnalysisOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSIS_H
