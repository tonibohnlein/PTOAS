// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCORIGINCLOSURE_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCORIGINCLOSURE_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncOriginPropagation.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/SyncProtocolModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <deque>
#include <set>
#include <vector>

namespace mlir::pto {
namespace sync_origin_detail {
inline bool memoryType(Type type) {
  return isa<TileBufType, MultiTileBufType, PtrType, TensorViewType,
             PartitionTensorViewType, BaseMemRefType>(type);
}
inline AddressSpace space(Type type) {
  if (auto pointer = dyn_cast<PtrType>(type))
    return pointer.getMemorySpace().getAddressSpace();
  if (auto multi = dyn_cast<MultiTileBufType>(type))
    type = multi.getSlotType();
  Attribute attr;
  if (auto tile = dyn_cast<TileBufType>(type)) attr = tile.getMemorySpace();
  else if (auto memref = dyn_cast<BaseMemRefType>(type))
    attr = memref.getMemorySpace();
  if (auto address = dyn_cast_or_null<AddressSpaceAttr>(attr))
    return address.getAddressSpace();
  return AddressSpace::GM; // Tensor views have a GM contract.
}
struct Node {
  Value value;
  std::vector<unsigned> sources, users;
  std::set<unsigned> roots;
  bool affected = false, unknown = false;
};
} // namespace sync_origin_detail

// Shared post-translation origin closure. Existing precise records survive;
// loop-derived values additionally cover every reaching whole-root footprint.
// No offset iteration: subviews of carried values deliberately widen to roots.
// Output records are append-only, so previously translated pointers stay valid.
// The final phase refresh also imports effects absent during the first traversal
// (notably while condition/result forwarding).
inline LogicalResult closeStructuredSyncOrigins(
    func::FuncOp function, SyncIRs &phases, Buffer2MemInfoMap &buffers) {
  using namespace sync_origin_detail;
  DenseMap<Value, unsigned> ids;
  std::vector<Node> nodes;
  auto id = [&](Value value) {
    auto found = ids.find(value);
    if (found != ids.end()) return found->second;
    unsigned next = nodes.size();
    ids[value] = next;
    nodes.push_back(Node{value, {}, {}, {}, false, false});
    return next;
  };
  auto edge = [&](Value to, Value from) {
    if (!to || !from || !memoryType(to.getType())) return;
    unsigned a = id(to), b = id(from);
    if (std::find(nodes[a].sources.begin(), nodes[a].sources.end(), b) ==
        nodes[a].sources.end()) {
      nodes[a].sources.push_back(b);
      nodes[b].users.push_back(a);
    }
  };
  auto carried = [&](Value value) {
    if (value && memoryType(value.getType())) nodes[id(value)].affected = true;
  };
  function.walk([&](mlir::Operation *op) {
    for (Value value : op->getOperands())
      if (memoryType(value.getType())) (void)id(value);
    for (Value value : op->getResults())
      if (memoryType(value.getType())) (void)id(value);
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      for (unsigned i = 0; i < loop.getInitArgs().size(); ++i) {
        Value arg = loop.getRegionIterArgs()[i], result = loop.getResult(i);
        edge(arg, loop.getInitArgs()[i]); edge(arg, yield.getOperand(i));
        edge(result, loop.getInitArgs()[i]); edge(result, yield.getOperand(i));
        carried(arg); carried(result);
      }
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      auto condition = loop.getConditionOp();
      auto yield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
      for (unsigned i = 0; i < loop.getBeforeArguments().size(); ++i) {
        Value arg = loop.getBeforeArguments()[i];
        edge(arg, loop.getInits()[i]); edge(arg, yield.getOperand(i));
        carried(arg);
      }
      for (unsigned i = 0; i < condition.getArgs().size(); ++i) {
        edge(loop.getAfterArguments()[i], condition.getArgs()[i]);
        edge(loop.getResult(i), condition.getArgs()[i]);
        carried(loop.getAfterArguments()[i]); carried(loop.getResult(i));
      }
    } else if (auto choice = dyn_cast<scf::IfOp>(op)) {
      for (mlir::Region &region : choice->getRegions()) {
        if (region.empty()) continue;
        auto yield = cast<scf::YieldOp>(region.front().getTerminator());
        for (unsigned i = 0; i < choice.getNumResults(); ++i)
          edge(choice.getResult(i), yield.getOperand(i));
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
  for (unsigned i = 0; i < nodes.size(); ++i)
    if (nodes[i].affected) queue.push_back(i);
  while (!queue.empty()) {
    unsigned i = queue.front(); queue.pop_front();
    for (unsigned user : nodes[i].users)
      if (!nodes[user].affected) {
        nodes[user].affected = true; queue.push_back(user);
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
  auto append = [&](Value value, std::unique_ptr<BaseMemInfo> info) {
    auto &entries = buffers[value];
    for (const auto &old : entries)
      if (*old == *info) return;
    entries.push_back(std::move(info));
  };
  // Gather copies before modifying the map: DenseMap insertion may rehash.
  std::vector<std::pair<Value, std::unique_ptr<BaseMemInfo>>> additions;
  for (const Node &node : nodes) {
    if (!node.affected) continue;
    bool unknown = node.unknown || node.roots.empty();
    for (unsigned root : node.roots) {
      auto found = buffers.find(nodes[root].value);
      if (found == buffers.end() || found->second.empty()) {
        unknown = true; continue;
      }
      for (const auto &info : found->second)
        additions.emplace_back(node.value, info->clone(node.value));
    }
    if (unknown)
      additions.emplace_back(node.value, std::make_unique<BaseMemInfo>(
          node.value, node.value, space(node.value.getType()),
          SmallVector<uint64_t>{0}, 0, false, true));
  }
  for (auto &addition : additions) append(addition.first, std::move(addition.second));
  // Refresh from original operation effects, not from the old (possibly empty)
  // translated use/def list. Match the translator's precedence: incomplete
  // optional protocol descriptions fall through to the ordinary interfaces.
  for (auto &entry : phases) {
    auto *phase = dyn_cast<CompoundInstanceElement>(entry.get());
    if (!phase) continue;
    SmallVector<Value> reads, writes;
    if (auto protocol = getSyncProtocolModel(phase->elementOp);
        protocol && protocol->complete()) {
      reads = protocol->reads;
      writes = protocol->writes;
    } else if (auto macro = getSyncMacroModel(phase->elementOp)) {
      if (phase->macroOpInstanceId < 0 ||
          unsigned(phase->macroOpInstanceId) >= macro->phases.size())
        return phase->elementOp->emitError("invalid phase during origin closure");
      const auto &model = macro->phases[phase->macroOpInstanceId];
      reads.append(model.useValues.begin(), model.useValues.end());
      writes.append(model.defValues.begin(), model.defValues.end());
    } else if (auto effects = dyn_cast<MemoryEffectOpInterface>(phase->elementOp)) {
      SmallVector<MemoryEffects::EffectInstance> list;
      effects.getEffects(list);
      for (const auto &effect : list) {
        if (!effect.getValue()) continue;
        if (isa<MemoryEffects::Read>(effect.getEffect())) reads.push_back(effect.getValue());
        if (isa<MemoryEffects::Write>(effect.getEffect())) writes.push_back(effect.getValue());
      }
    } else continue; // A helper's explicit translator contract is not replaced.
    auto refresh = [&](auto &to, const auto &values) {
      to.clear();
      for (Value value : values) {
        auto found = buffers.find(value);
        if (found == buffers.end()) continue;
        for (const auto &info : found->second)
          if (std::find(to.begin(), to.end(), info.get()) == to.end())
            to.push_back(info.get());
      }
    };
    refresh(phase->useVec, reads); refresh(phase->defVec, writes);
  }
  return success();
}
} // namespace mlir::pto
#endif
