// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/KernelScheduleGraph.h"
#include "PTO/Transforms/KernelScheduling/PTOISADuration.h"

#include "../Utils.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;

namespace {

using VertexIdx = pto::ScheduleGraph::VertexIdx;

struct MemoryAccess {
  Value root;
  bool reads = false;
  bool writes = false;
};

struct NodeMemoryAccesses {
  SmallVector<MemoryAccess, 4> accesses;
  bool unknown = false;
};

struct MemoryState {
  std::optional<VertexIdx> writer;
  SmallVector<VertexIdx, 4> readers;
};

static bool isSynchronizationOperation(Operation *op) {
  const StringRef name = op->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Cases(pto::RecordEventOp::getOperationName(),
             pto::WaitEventOp::getOperationName(),
             pto::BarrierSyncOp::getOperationName(), true)
      .Cases(pto::AicInitializePipeOp::getOperationName(),
             pto::AivInitializePipeOp::getOperationName(),
             pto::InitializeL2G2LPipeOp::getOperationName(), true)
      .Cases(pto::InitializeL2LPipeOp::getOperationName(),
             pto::DeclareEventIdArrayOp::getOperationName(),
             pto::EventIdArrayGetOp::getOperationName(), true)
      .Cases(pto::EventIdArraySetOp::getOperationName(),
             pto::SetFlagOp::getOperationName(),
             pto::WaitFlagOp::getOperationName(), true)
      .Cases(pto::SetFlagDynOp::getOperationName(),
             pto::WaitFlagDynOp::getOperationName(),
             pto::GetBufOp::getOperationName(), true)
      .Cases(pto::RlsBufOp::getOperationName(),
             pto::SyncSetOp::getOperationName(),
             pto::SyncWaitOp::getOperationName(), true)
      .Cases(pto::SetCrossBlockOp::getOperationName(),
             pto::WaitCrossBlockOp::getOperationName(),
             pto::SetIntraBlockOp::getOperationName(), true)
      .Cases(pto::WaitIntraBlockOp::getOperationName(),
             pto::BarrierOp::getOperationName(),
             pto::FenceBarrierAllOp::getOperationName(), true)
      .Cases(pto::TSyncOp::getOperationName(),
             pto::SyncAllOp::getOperationName(), pto::DsbOp::getOperationName(),
             true)
      .Default(false);
}

static bool isTransferPipe(pto::PIPE pipe) {
  switch (pipe) {
  case pto::PIPE::PIPE_MTE1:
  case pto::PIPE::PIPE_MTE2:
  case pto::PIPE::PIPE_MTE3:
  case pto::PIPE::PIPE_MTE4:
  case pto::PIPE::PIPE_MTE5:
  case pto::PIPE::PIPE_FIX:
  case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return true;
  default:
    return false;
  }
}

static std::optional<unsigned> getPyptoAccessOrder(Operation *op) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  op->getLoc().print(stream);
  stream.flush();
  constexpr StringLiteral marker = "pypto.access.";
  const size_t markerOffset = StringRef(printed).find(marker);
  if (markerOffset == StringRef::npos) {
    return std::nullopt;
  }
  StringRef suffix = StringRef(printed).drop_front(markerOffset);
  if (!suffix.consume_front(marker)) {
    return std::nullopt;
  }
  size_t digits = 0;
  while (digits < suffix.size() && suffix[digits] >= '0' && suffix[digits] <= '9') {
    ++digits;
  }
  if (digits == 0) {
    return std::nullopt;
  }
  unsigned order = 0;
  if (suffix.take_front(digits).getAsInteger(10, order)) {
    return std::nullopt;
  }
  return order;
}

static unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      ++depth;
    }
  }
  return depth;
}

static void appendUnique(SmallVectorImpl<VertexIdx> &values, VertexIdx value) {
  if (!llvm::is_contained(values, value)) {
    values.push_back(value);
  }
}

class KernelScheduleGraphBuilder {
public:
  explicit KernelScheduleGraphBuilder(func::FuncOp func,
                                      pto::KernelScheduleGraphBuildOptions options)
      : func_(func), options_(options) {}

  FailureOr<pto::KernelScheduleGraph> build() {
    if (failed(validateControlFlow())) {
      return failure();
    }
    collectNodes();
    if (durationUnavailable_) {
      return failure();
    }
    addSSADependencies();
    addMemoryDependencies();
    addControlDependencies();
    addLoopRecurrences();
    if (!graph_.isAcyclic()) {
      func_.emitError("kernel schedule graph contains a zero-distance cycle");
      return failure();
    }
    return std::move(graph_);
  }

private:
  LogicalResult validateControlFlow() {
    WalkResult result = func_.walk([&](Operation *op) {
      const bool hasSuccessors = op->getNumSuccessors() != 0;
      if (!hasSuccessors) {
        return WalkResult::advance();
      }
      op->emitError("kernel schedule graph does not yet support unstructured "
                    "control flow");
      return WalkResult::interrupt();
    });
    return failure(result.wasInterrupted());
  }

  std::optional<pto::PIPE> getSchedulablePipe(Operation *op) const {
    auto pipeOp = dyn_cast<pto::OpPipeInterface>(op);
    const bool hasRegions = op->getNumRegions() != 0;
    const bool isSynchronization = isSynchronizationOperation(op);
    if (!pipeOp || hasRegions || isSynchronization) {
      return std::nullopt;
    }
    const pto::PIPE pipe = pipeOp.getPipe();
    const bool isConcretePipe = pto::isConcreteSyncPipe(pipe);
    if (!isConcretePipe || pipe == pto::PIPE::PIPE_NUM) {
      return std::nullopt;
    }
    return pipe;
  }

  VertexIdx getBlockId(Block *block) {
    auto [iterator, inserted] = blockIds_.try_emplace(block, blockIds_.size());
    (void)inserted;
    return iterator->second;
  }

  void collectNodes() {
    VertexIdx order = 0;
    func_.walk([&](Operation *op) {
      std::optional<pto::PIPE> pipe = getSchedulablePipe(op);
      if (!pipe) {
        return;
      }
      const pto::ScheduleNodeKind kind =
          isa<pto::MteOpInterface, pto::LoadScalarOp, pto::StoreScalarOp>(op) ||
                  isTransferPipe(*pipe)
              ? pto::ScheduleNodeKind::Transfer
              : pto::ScheduleNodeKind::Compute;
      uint64_t durationCycles = 1;
      bool hasResolvedDuration = false;
      std::optional<pto::PTOISADurationSignature> signature;
      if (options_.durationTable) {
        signature = pto::getPTOISADurationSignature(op);
        std::optional<pto::PTOISADurationEstimate> estimate =
            signature ? options_.durationTable->estimate(*signature)
                      : std::nullopt;
        if (estimate) {
          durationCycles = estimate->cycles;
          hasResolvedDuration = true;
        } else if (options_.requireExactDurations) {
            op->emitError(
                "kernel schedule graph has no matching pinned PTO-ISA duration "
                "for this operation");
            durationUnavailable_ = true;
            return;
        }
      }
      const VertexIdx id = graph_.addNode(
          op, *pipe, kind, order++, getBlockId(op->getBlock()), getLoopDepth(op), durationCycles, hasResolvedDuration,
          hasResolvedDuration ? signature : std::nullopt, getPyptoAccessOrder(op));
      nodeIds_.try_emplace(op, id);
    });
    memoryAccesses_.resize(graph_.getNodes().size());
  }

  void collectScheduledProducers(Value value,
                                 llvm::SetVector<VertexIdx> &producers,
                                 DenseSet<Value> &visited) const {
    if (!value || !visited.insert(value).second) {
      return;
    }
    Operation *def = value.getDefiningOp();
    if (!def) {
      return;
    }
    if (auto iterator = nodeIds_.find(def); iterator != nodeIds_.end()) {
      producers.insert(iterator->second);
      return;
    }
    if (auto alias = pto::getOperationAliasInfo(def);
        alias && alias->first == value) {
      collectScheduledProducers(alias->second, producers, visited);
      return;
    }
    const bool isEffectFree = isMemoryEffectFree(def);
    const bool hasRegions = def->getNumRegions() != 0;
    if (!isEffectFree || hasRegions) {
      return;
    }
    for (Value operand : def->getOperands()) {
      collectScheduledProducers(operand, producers, visited);
    }
  }

  void addSSADependencies() {
    for (const pto::KernelScheduleNode &node : graph_.getNodes()) {
      for (Value operand : node.operation->getOperands()) {
        llvm::SetVector<VertexIdx> producers;
        DenseSet<Value> visited;
        collectScheduledProducers(operand, producers, visited);
        for (VertexIdx producer : producers) {
          if (producer != node.id) {
            graph_.addDependency(producer, node.id,
                                 pto::ScheduleDependencyKind::SSA);
          }
        }
      }
    }
  }

  Value getAliasRoot(Value value) const {
    DenseSet<Value> visited;
    while (value && visited.insert(value).second) {
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        Operation *parent = argument.getOwner()->getParentOp();
        if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
          const unsigned number = argument.getArgNumber();
          if (number > 0 && number <= forOp.getInitArgs().size()) {
            value = forOp.getInitArgs()[number - 1];
            continue;
          }
        }
        break;
      }

      Operation *def = value.getDefiningOp();
      if (!def) {
        break;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        const auto result = cast<OpResult>(value);
        const bool isInitResult =
            result.getResultNumber() < forOp.getInitArgs().size();
        if (isInitResult) {
          value = forOp.getInitArgs()[result.getResultNumber()];
          continue;
        }
      }
      if (auto alias = pto::getOperationAliasInfo(def);
          alias && alias->first == value) {
        value = alias->second;
        continue;
      }
      break;
    }
    return value;
  }

  static MemoryAccess &getOrAddAccess(NodeMemoryAccesses &result, Value root) {
    auto iterator = llvm::find_if(result.accesses, [root](MemoryAccess access) {
      return access.root == root;
    });
    if (iterator != result.accesses.end()) {
      return *iterator;
    }
    result.accesses.push_back({root, false, false});
    return result.accesses.back();
  }

  NodeMemoryAccesses collectMemoryAccesses(Operation *op) const {
    NodeMemoryAccesses result;
    auto effectsOp = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectsOp) {
      result.unknown = !isMemoryEffectFree(op);
      return result;
    }

    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 8> effects;
    effectsOp.getEffects(effects);
    for (const auto &effect : effects) {
      const bool reads = isa<MemoryEffects::Read>(effect.getEffect());
      const bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (!reads && !writes) {
        continue;
      }
      Value value = effect.getValue();
      if (!value) {
        result.unknown = true;
        continue;
      }
      MemoryAccess &access = getOrAddAccess(result, getAliasRoot(value));
      access.reads |= reads;
      access.writes |= writes;
    }
    return result;
  }

  void applyMemoryAccess(VertexIdx node, const MemoryAccess &access,
                         DenseMap<Value, MemoryState> &states) {
    MemoryState &state = states[access.root];
    if (access.reads) {
      if (state.writer && *state.writer != node) {
        graph_.addDependency(*state.writer, node,
                             pto::ScheduleDependencyKind::MemoryRAW);
      }
      appendUnique(state.readers, node);
    }
    if (!access.writes) {
      return;
    }
    if (state.writer && *state.writer != node) {
      graph_.addDependency(*state.writer, node,
                           pto::ScheduleDependencyKind::MemoryWAW);
    }
    for (VertexIdx reader : state.readers) {
      if (reader != node) {
        graph_.addDependency(reader, node,
                             pto::ScheduleDependencyKind::MemoryWAR);
      }
    }
    state.readers.clear();
    state.writer = node;
  }

  void addMemoryDependencies() {
    DenseMap<Block *, SmallVector<VertexIdx, 8>> nodesByBlock;
    SmallVector<Block *, 8> blockOrder;
    for (const pto::KernelScheduleNode &node : graph_.getNodes()) {
      if (!nodesByBlock.count(node.operation->getBlock())) {
        blockOrder.push_back(node.operation->getBlock());
      }
      nodesByBlock[node.operation->getBlock()].push_back(node.id);
      memoryAccesses_[node.id] = collectMemoryAccesses(node.operation);
    }

    for (Block *block : blockOrder) {
      DenseMap<Value, MemoryState> states;
      SmallVector<VertexIdx, 8> previousNodes;
      std::optional<VertexIdx> lastUnknown;
      for (VertexIdx node : nodesByBlock[block]) {
        const NodeMemoryAccesses &accesses = memoryAccesses_[node];
        if (lastUnknown) {
          graph_.addDependency(*lastUnknown, node,
                               pto::ScheduleDependencyKind::Control);
        }
        if (accesses.unknown) {
          for (VertexIdx previous : previousNodes) {
            graph_.addDependency(previous, node,
                                 pto::ScheduleDependencyKind::Control);
          }
          lastUnknown = node;
        }
        for (const MemoryAccess &access : accesses.accesses) {
          applyMemoryAccess(node, access, states);
        }
        previousNodes.push_back(node);
      }
    }
  }

  void collectDescendantNodes(Operation *op,
                              SmallVectorImpl<VertexIdx> &nodes) const {
    op->walk([&](Operation *nested) {
      if (auto iterator = nodeIds_.find(nested); iterator != nodeIds_.end()) {
        appendUnique(nodes, iterator->second);
      }
    });
  }

  bool isControlBoundary(Operation *op) const {
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return false;
    }
    const bool hasRegions = op->getNumRegions() != 0;
    const bool isCall = isa<CallOpInterface>(op);
    if (hasRegions || isCall) {
      return true;
    }
    const bool isNode = nodeIds_.count(op) != 0;
    const bool isAllocation =
        isa<pto::AllocTileOp, pto::AllocMultiTileOp, pto::ReserveBufferOp>(op);
    if (isNode || isAllocation) {
      return false;
    }
    return !isMemoryEffectFree(op);
  }

  void addControlBoundaryDependencies(Operation *boundary) {
    SmallVector<VertexIdx, 8> before;
    SmallVector<VertexIdx, 8> inside;
    SmallVector<VertexIdx, 8> after;
    bool passedBoundary = false;
    for (Operation &op : *boundary->getBlock()) {
      if (&op == boundary) {
        passedBoundary = true;
        collectDescendantNodes(boundary, inside);
        continue;
      }
      collectDescendantNodes(&op, passedBoundary ? after : before);
    }

    auto connect = [&](ArrayRef<VertexIdx> sources,
                       ArrayRef<VertexIdx> targets) {
      for (VertexIdx source : sources) {
        for (VertexIdx target : targets) {
          if (source != target) {
            graph_.addDependency(source, target,
                                 pto::ScheduleDependencyKind::Control);
          }
        }
      }
    };
    if (inside.empty()) {
      connect(before, after);
      return;
    }
    connect(before, inside);
    connect(inside, after);
  }

  void addControlDependencies() {
    SmallVector<Operation *, 8> boundaries;
    func_.walk([&](Operation *op) {
      const bool isFunction = op == func_.getOperation();
      const bool isBoundary = isControlBoundary(op);
      if (!isFunction && isBoundary) {
        boundaries.push_back(op);
      }
    });
    for (Operation *boundary : boundaries) {
      addControlBoundaryDependencies(boundary);
    }
  }

  SmallVector<VertexIdx, 8> getLoopNodes(Operation *loop) const {
    SmallVector<VertexIdx, 8> nodes;
    collectDescendantNodes(loop, nodes);
    llvm::sort(nodes, [&](VertexIdx lhs, VertexIdx rhs) {
      return graph_.getNode(lhs).originalOrder <
             graph_.getNode(rhs).originalOrder;
    });
    return nodes;
  }

  void addMemoryRecurrences(Operation *loop) {
    const SmallVector<VertexIdx, 8> nodes = getLoopNodes(loop);
    DenseMap<Value, MemoryState> finalStates;
    for (VertexIdx node : nodes) {
      for (const MemoryAccess &access : memoryAccesses_[node].accesses) {
        MemoryState &state = finalStates[access.root];
        if (access.reads) {
          appendUnique(state.readers, node);
        }
        if (access.writes) {
          state.readers.clear();
          state.writer = node;
        }
      }
    }

    struct TaggedNode {
      VertexIdx node;
      bool previousIteration;
    };
    struct RecurrenceState {
      std::optional<TaggedNode> writer;
      SmallVector<TaggedNode, 4> readers;
    };
    DenseMap<Value, RecurrenceState> states;
    for (const auto &entry : finalStates) {
      RecurrenceState &state = states[entry.first];
      if (entry.second.writer) {
        state.writer = TaggedNode{*entry.second.writer, true};
      }
      for (VertexIdx reader : entry.second.readers) {
        state.readers.push_back({reader, true});
      }
    }

    for (VertexIdx node : nodes) {
      for (const MemoryAccess &access : memoryAccesses_[node].accesses) {
        RecurrenceState &state = states[access.root];
        if (access.reads) {
          if (state.writer && state.writer->previousIteration) {
            graph_.addDependency(state.writer->node, node,
                                 pto::ScheduleDependencyKind::MemoryRAW,
                                 /*iterationDistance=*/1, loop);
          }
          state.readers.push_back({node, false});
        }
        if (!access.writes) {
          continue;
        }
        if (state.writer && state.writer->previousIteration) {
          graph_.addDependency(state.writer->node, node,
                               pto::ScheduleDependencyKind::MemoryWAW,
                               /*iterationDistance=*/1, loop);
        }
        for (TaggedNode reader : state.readers) {
          if (reader.previousIteration) {
            graph_.addDependency(reader.node, node,
                                 pto::ScheduleDependencyKind::MemoryWAR,
                                 /*iterationDistance=*/1, loop);
          }
        }
        state.readers.clear();
        state.writer = TaggedNode{node, false};
      }
    }
  }

  bool valueDependsOn(Value value, Value source,
                      DenseSet<Value> &visited) const {
    if (value == source) {
      return true;
    }
    if (!value || !visited.insert(value).second) {
      return false;
    }
    Operation *def = value.getDefiningOp();
    if (!def || nodeIds_.count(def)) {
      return false;
    }
    if (auto alias = pto::getOperationAliasInfo(def);
        alias && alias->first == value) {
      return valueDependsOn(alias->second, source, visited);
    }
    const bool isEffectFree = isMemoryEffectFree(def);
    const bool hasRegions = def->getNumRegions() != 0;
    if (!isEffectFree || hasRegions) {
      return false;
    }
    return llvm::any_of(def->getOperands(), [&](Value operand) {
      return valueDependsOn(operand, source, visited);
    });
  }

  void addForSSARecurrences(scf::ForOp forOp) {
    const bool hasMatchingYields =
        forOp.getRegionIterArgs().size() == forOp.getYieldedValues().size();
    if (!hasMatchingYields) {
      return;
    }
    const SmallVector<VertexIdx, 8> nodes = getLoopNodes(forOp);
    for (auto mapping : llvm::zip(forOp.getRegionIterArgs(), forOp.getYieldedValues())) {
        Value iterArg = std::get<0>(mapping);
        Value yielded = std::get<1>(mapping);
        llvm::SetVector<VertexIdx> producers;
        DenseSet<Value> producerVisited;
        collectScheduledProducers(yielded, producers, producerVisited);
        for (VertexIdx producer : producers) {
            if (!forOp->isAncestor(graph_.getNode(producer).operation)) {
                continue;
            }
            for (VertexIdx target : nodes) {
                bool consumesIterArg =
                    llvm::any_of(graph_.getNode(target).operation->getOperands(), [&](Value operand) {
                        DenseSet<Value> visited;
                        return valueDependsOn(operand, iterArg, visited);
                    });
                if (consumesIterArg) {
                    graph_.addDependency(
                        producer, target, pto::ScheduleDependencyKind::LoopCarriedSSA,
                        /*iterationDistance=*/1, forOp);
                }
            }
        }
    }
  }

  void addLoopRecurrences() {
    SmallVector<Operation *, 8> loops;
    SmallVector<scf::ForOp, 8> forOps;
    func_.walk([&](Operation *op) {
      if (isa<LoopLikeOpInterface>(op)) {
        loops.push_back(op);
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        forOps.push_back(forOp);
      }
    });
    for (Operation *loop : loops) {
      addMemoryRecurrences(loop);
    }
    for (scf::ForOp forOp : forOps) {
      addForSSARecurrences(forOp);
    }
  }

  func::FuncOp func_;
  pto::KernelScheduleGraph graph_;
  pto::KernelScheduleGraphBuildOptions options_;
  bool durationUnavailable_ = false;
  DenseMap<Operation *, VertexIdx> nodeIds_;
  DenseMap<Block *, VertexIdx> blockIds_;
  std::vector<NodeMemoryAccesses> memoryAccesses_;
};

} // namespace

FailureOr<pto::KernelScheduleGraph>
mlir::pto::buildKernelScheduleGraph(func::FuncOp func,
                                    KernelScheduleGraphBuildOptions options) {
  return KernelScheduleGraphBuilder(func, options).build();
}
