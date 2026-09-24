// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License. Read-only extraction of SyncOriginClosure's original SSA origin graph. No translated
// phase or buffer is rewritten by this query.
#ifndef PTO_FRONTIERSYNCH_STORAGEORIGINS_H
#define PTO_FRONTIERSYNCH_STORAGEORIGINS_H
#include "PTO/Transforms/FrontierSynch/SyncOriginPropagation.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <set>
namespace mlir::pto::frontiersynch::origin_detail {
inline bool memoryType(Type type) {
  return isa<TileBufType, MultiTileBufType, PtrType, TensorViewType, PartitionTensorViewType,
             BaseMemRefType>(type);
}
struct Node {
  Value value;
  std::vector<unsigned> sources, users;
  std::set<unsigned> roots;
  bool affected = false, unknown = false;
};
struct Origins {
  std::vector<Value> roots;
  bool carried = false, unknown = false;
};
inline DenseMap<Value, Origins> collectOrigins(func::FuncOp function,
                                               const Buffer2MemInfoMap &buffers) {
  DenseMap<Value, unsigned> ids;
  std::vector<Node> nodes;
  auto id = [&](Value value) {
    auto found = ids.find(value);
    if (found != ids.end()) {
      return found->second;
    }
    unsigned next = nodes.size();
    ids[value] = next;
    nodes.push_back(Node{value, {}, {}, {}, false, false});
    return next;
  };
  auto edge = [&](Value to, Value from) {
    if (!to || !from || !memoryType(to.getType())) {
      return;
    }
    unsigned a = id(to), b = id(from);
    if (std::find(nodes[a].sources.begin(), nodes[a].sources.end(), b) == nodes[a].sources.end()) {
      nodes[a].sources.push_back(b);
      nodes[b].users.push_back(a);
    }
  };
  auto carried = [&](Value value) {
    if (value && memoryType(value.getType())) {
      nodes[id(value)].affected = true;
    }
  };
  function.walk([&](mlir::Operation *op) {
    for (Value value : op->getOperands()) {
      if (memoryType(value.getType())) {
        (void)id(value);
      }
    }
    for (Value value : op->getResults()) {
      if (memoryType(value.getType())) {
        (void)id(value);
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      for (unsigned i = 0; i < loop.getInitArgs().size(); ++i) {
        Value arg = loop.getRegionIterArgs()[i], result = loop.getResult(i);
        edge(arg, loop.getInitArgs()[i]);
        edge(arg, yield.getOperand(i));
        edge(result, loop.getInitArgs()[i]);
        edge(result, yield.getOperand(i));
        carried(arg);
        carried(result);
      }
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      auto condition = loop.getConditionOp();
      auto yield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
      for (unsigned i = 0; i < loop.getBeforeArguments().size(); ++i) {
        Value arg = loop.getBeforeArguments()[i];
        edge(arg, loop.getInits()[i]);
        edge(arg, yield.getOperand(i));
        carried(arg);
      }
      for (unsigned i = 0; i < condition.getArgs().size(); ++i) {
        edge(loop.getAfterArguments()[i], condition.getArgs()[i]);
        edge(loop.getResult(i), condition.getArgs()[i]);
        carried(loop.getAfterArguments()[i]);
        carried(loop.getResult(i));
      }
    } else if (auto choice = dyn_cast<scf::IfOp>(op)) {
      for (mlir::Region &region : choice->getRegions()) {
        if (region.empty()) {
          continue;
        }
        auto yield = cast<scf::YieldOp>(region.front().getTerminator());
        for (unsigned i = 0; i < choice.getNumResults(); ++i) {
          edge(choice.getResult(i), yield.getOperand(i));
        }
      }
    } else if (auto view = dyn_cast<MakeTensorViewOp>(op)) {
      edge(view.getResult(), view.getPtr());
    } else if (auto view = dyn_cast<PartitionViewOp>(op)) {
      edge(view.getResult(), view.getSource());
    } else if (auto view = dyn_cast<SubViewOp>(op)) {
      edge(view.getResult(), view.getSource());
    } else if (auto view = dyn_cast<AddPtrOp>(op)) {
      edge(view.getResult(), view.getPtr());
    } else if (auto view = dyn_cast<CastPtrOp>(op)) {
      // The generic scalar surface expresses pointer/integer round trips
      // with castptr in both directions. Match the shared translator contract.
      auto roundTrip = view.getInput().getDefiningOp<CastPtrOp>();
      if (isa<PtrType>(view.getResult().getType()) && roundTrip &&
          isa<IntegerType>(roundTrip.getResult().getType()) &&
          isa<PtrType>(roundTrip.getInput().getType())) {
        edge(view.getResult(), roundTrip.getInput());
      } else {
        edge(view.getResult(), view.getInput());
      }
    } else if (auto view = dyn_cast<TReshapeOp>(op)) {
      edge(view.getResult(), view.getSrc());
    } else if (auto view = dyn_cast<BitcastOp>(op)) {
      edge(view.getResult(), view.getSrc());
    } else if (auto view = dyn_cast<MultiTileGetOp>(op)) {
      edge(view.getResult(), view.getSource());
    } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      edge(select.getResult(), select.getTrueValue());
      edge(select.getResult(), select.getFalseValue());
    }
  });
  // Mark the entire forward slice of loop-carried handles before origin work.
  std::deque<unsigned> queue;
  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (nodes[i].affected) {
      queue.push_back(i);
    }
  }
  while (!queue.empty()) {
    unsigned i = queue.front();
    queue.pop_front();
    for (unsigned user : nodes[i].users) {
      if (!nodes[user].affected) {
        nodes[user].affected = true;
        queue.push_back(user);
      }
    }
  }
  // Origin identities are finite graph leaves, not successively shifted views.
  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (nodes[i].sources.empty()) {
      nodes[i].roots.insert(i);
      auto found = buffers.find(nodes[i].value);
      nodes[i].unknown = found == buffers.end() || found->second.empty();
    }
  }
  (void)propagateSyncOrigins(nodes);
  DenseMap<Value, Origins> result;
  for (const auto &node : nodes) {
    auto &entry = result[node.value];
    entry.carried = node.affected;
    entry.unknown = node.unknown || node.roots.empty();
    for (auto root : node.roots) {
      entry.roots.push_back(nodes[root].value);
    }
  }
  return result;
}
} // namespace mlir::pto::frontiersynch::origin_detail
#endif
