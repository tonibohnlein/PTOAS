// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGEINTERNAL_H
#define PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGEINTERNAL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace mlir {
namespace pto {
namespace sync_cover_detail {

constexpr unsigned kStaticControlCopy = std::numeric_limits<unsigned>::max();

struct ContextLiteral {
  SyncCoverControlId control = 0;
  unsigned copy = 0;
  unsigned alternative = 0;

  bool operator<(const ContextLiteral &other) const {
    return std::tie(control, copy, alternative) <
           std::tie(other.control, other.copy, other.alternative);
  }

  bool operator==(const ContextLiteral &other) const {
    return control == other.control && copy == other.copy &&
           alternative == other.alternative;
  }
};

struct DemandContext {
  bool valid = false;
  std::vector<ContextLiteral> condition;
};

inline unsigned contextCopy(const SyncCoverGraph &graph,
                            const SyncCoverDemand &demand,
                            SyncCoverControlId control, unsigned copy) {
  const SyncCoverScopeId controlScope = graph.getControls()[control].scope;
  const bool perIteration =
      demand.distance != 0 && graph.scopeContains(demand.scope, controlScope);
  return perIteration ? copy : kStaticControlCopy;
}

inline bool appendGuard(const SyncCoverGraph &graph,
                        const SyncCoverDemand &demand,
                        const SyncCoverGuard &guard, unsigned copy,
                        std::vector<ContextLiteral> &context) {
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    context.push_back({literal.control,
                       contextCopy(graph, demand, literal.control, copy),
                       literal.alternative});
  }
  std::sort(context.begin(), context.end());
  context.erase(std::unique(context.begin(), context.end()), context.end());
  for (std::size_t index = 1; index < context.size(); ++index) {
    const ContextLiteral &previous = context[index - 1];
    const ContextLiteral &current = context[index];
    if (previous.control == current.control && previous.copy == current.copy &&
        previous.alternative != current.alternative) {
      return false;
    }
  }
  return true;
}

inline DemandContext makeDemandContext(const SyncCoverGraph &graph,
                                       const SyncCoverDemand &demand) {
  DemandContext result;
  result.valid =
      appendGuard(graph, demand, demand.sourceGuard, 0, result.condition) &&
      appendGuard(graph, demand, demand.targetGuard, demand.distance,
                  result.condition);
  return result;
}

inline bool guardIsImplied(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand,
                           const DemandContext &context,
                           const SyncCoverGuard &guard, unsigned copy) {
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    const ContextLiteral required{
        literal.control, contextCopy(graph, demand, literal.control, copy),
        literal.alternative};
    if (!std::binary_search(context.condition.begin(), context.condition.end(),
                            required)) {
      return false;
    }
  }
  return true;
}

inline bool nodeInstanceAvailable(const SyncCoverGraph &graph,
                                  const SyncCoverDemand &demand,
                                  SyncCoverNodeId node, unsigned copy) {
  if (copy > demand.distance) {
    return false;
  }
  const bool isSource = node == demand.source && copy == 0;
  const bool isTarget = node == demand.target && copy == demand.distance;
  if (isSource || isTarget) {
    return true;
  }
  const SyncCoverScopeId nodeScope = graph.getNodes()[node].scope;
  if (graph.scopeMustExecuteWithin(demand.scope, nodeScope)) {
    return true;
  }
  const SyncCoverScopeId sourceScope = graph.getNodes()[demand.source].scope;
  if (copy == 0 && graph.scopeContains(nodeScope, sourceScope)) {
    return true;
  }
  const SyncCoverScopeId targetScope = graph.getNodes()[demand.target].scope;
  return copy == demand.distance && graph.scopeContains(nodeScope, targetScope);
}

inline bool edgeGuardsActive(const SyncCoverGraph &graph,
                             const SyncCoverDemand &demand,
                             const DemandContext &context,
                             const SyncCoverEdge &edge, unsigned sourceCopy,
                             unsigned targetCopy) {
  return guardIsImplied(graph, demand, context, edge.sourceGuard, sourceCopy) &&
         guardIsImplied(graph, demand, context, edge.targetGuard, targetCopy);
}

} // namespace sync_cover_detail
} // namespace pto
} // namespace mlir

#endif // PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGEINTERNAL_H
