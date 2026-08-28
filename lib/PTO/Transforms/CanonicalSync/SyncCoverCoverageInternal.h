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

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

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

inline bool consumeWork(SyncCoverCoverageWorkBudget *budget,
                        std::size_t amount = 1) {
  return !budget || budget->consume(amount);
}

inline bool scopeContains(const SyncCoverGraph &graph,
                          SyncCoverScopeId ancestor,
                          SyncCoverScopeId descendant,
                          SyncCoverCoverageWorkBudget *budget) {
  const auto &scopes = graph.getScopes();
  const bool workUnavailable = !consumeWork(budget);
  const bool invalidScope =
      ancestor >= scopes.size() || descendant >= scopes.size();
  if (workUnavailable || invalidScope) {
    return false;
  }
  while (descendant != ancestor && descendant != 0) {
    if (!consumeWork(budget)) {
      return false;
    }
    descendant = scopes[descendant].parent;
  }
  return descendant == ancestor;
}

inline bool scopeMustExecuteWithin(const SyncCoverGraph &graph,
                                   SyncCoverScopeId ancestor,
                                   SyncCoverScopeId descendant,
                                   SyncCoverCoverageWorkBudget *budget) {
  if (!scopeContains(graph, ancestor, descendant, budget)) {
    return false;
  }
  const auto &scopes = graph.getScopes();
  while (descendant != ancestor) {
    const bool workUnavailable = !consumeWork(budget);
    const bool optionalScope = !scopes[descendant].mustExecuteWithinParent;
    if (workUnavailable || optionalScope) {
      return false;
    }
    descendant = scopes[descendant].parent;
  }
  return true;
}

inline unsigned contextCopy(const SyncCoverGraph &graph,
                            const SyncCoverDemand &demand,
                            SyncCoverControlId control, unsigned copy,
                            SyncCoverCoverageWorkBudget *budget = nullptr) {
  const SyncCoverScopeId controlScope = graph.getControls()[control].scope;
  const bool perIteration =
      demand.distance != 0 &&
      scopeContains(graph, demand.scope, controlScope, budget);
  return perIteration ? copy : kStaticControlCopy;
}

inline bool appendGuard(const SyncCoverGraph &graph,
                        const SyncCoverDemand &demand,
                        const SyncCoverGuard &guard, unsigned copy,
                        std::vector<ContextLiteral> &context,
                        SyncCoverCoverageWorkBudget *budget = nullptr) {
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    if (!consumeWork(budget)) {
      return false;
    }
    const ContextLiteral item{
        literal.control,
        contextCopy(graph, demand, literal.control, copy, budget),
        literal.alternative};
    auto position = context.begin();
    while (position != context.end()) {
      const bool precedes = std::tie(position->control, position->copy) <
                            std::tie(item.control, item.copy);
      if (!precedes) {
        break;
      }
      if (!consumeWork(budget)) {
        return false;
      }
      ++position;
    }
    const bool sameContext = position != context.end() &&
                             position->control == item.control &&
                             position->copy == item.copy;
    if (sameContext) {
      if (position->alternative != item.alternative) {
        return false;
      }
      continue;
    }
    const std::size_t shifted =
        static_cast<std::size_t>(context.end() - position);
    const bool reallocates = context.size() == context.capacity();
    const bool insertionWorkUnavailable =
        !consumeWork(budget, shifted) ||
        (reallocates && !consumeWork(budget, context.size() + 1));
    if (insertionWorkUnavailable) {
      return false;
    }
    context.insert(position, item);
  }
  return true;
}

inline DemandContext
makeDemandContext(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                  SyncCoverCoverageWorkBudget *budget = nullptr) {
  DemandContext result;
  result.valid = appendGuard(graph, demand, demand.sourceGuard, 0,
                             result.condition, budget) &&
                 appendGuard(graph, demand, demand.targetGuard, demand.distance,
                             result.condition, budget);
  return result;
}

inline bool guardIsImplied(const SyncCoverGraph &graph,
                           const SyncCoverDemand &demand,
                           const DemandContext &context,
                           const SyncCoverGuard &guard, unsigned copy,
                           SyncCoverCoverageWorkBudget *budget = nullptr) {
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    if (!consumeWork(budget)) {
      return false;
    }
    const ContextLiteral required{
        literal.control,
        contextCopy(graph, demand, literal.control, copy, budget),
        literal.alternative};
    bool found = false;
    for (const ContextLiteral &available : context.condition) {
      if (!consumeWork(budget)) {
        return false;
      }
      if (available == required) {
        found = true;
        break;
      }
      if (required < available) {
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

inline bool
nodeInstanceAvailable(const SyncCoverGraph &graph,
                      const SyncCoverDemand &demand, SyncCoverNodeId node,
                      unsigned copy,
                      SyncCoverCoverageWorkBudget *budget = nullptr) {
  const bool workUnavailable = !consumeWork(budget);
  const bool invalidCopy = copy > demand.distance;
  if (workUnavailable || invalidCopy) {
    return false;
  }
  const bool isSource = node == demand.source && copy == 0;
  const bool isTarget = node == demand.target && copy == demand.distance;
  if (isSource || isTarget) {
    return true;
  }
  const SyncCoverScopeId nodeScope = graph.getNodes()[node].scope;
  if (scopeMustExecuteWithin(graph, demand.scope, nodeScope, budget)) {
    return true;
  }
  const SyncCoverScopeId sourceScope = graph.getNodes()[demand.source].scope;
  if (copy == 0 && scopeContains(graph, nodeScope, sourceScope, budget)) {
    return true;
  }
  const SyncCoverScopeId targetScope = graph.getNodes()[demand.target].scope;
  return copy == demand.distance &&
         scopeContains(graph, nodeScope, targetScope, budget);
}

inline bool edgeGuardsActive(const SyncCoverGraph &graph,
                             const SyncCoverDemand &demand,
                             const DemandContext &context,
                             const SyncCoverEdge &edge, unsigned sourceCopy,
                             unsigned targetCopy,
                             SyncCoverCoverageWorkBudget *budget = nullptr) {
  return guardIsImplied(graph, demand, context, edge.sourceGuard, sourceCopy,
                        budget) &&
         guardIsImplied(graph, demand, context, edge.targetGuard, targetCopy,
                        budget);
}

} // namespace sync_cover_detail
} // namespace pto
} // namespace mlir

#endif // PTO_LIB_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGEINTERNAL_H
