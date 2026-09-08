// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/InsertSync/MmadChainAnalysis.h"
#include "PTO/Transforms/InsertSync/MmadChainDomain.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::insert_sync_detail;

namespace {
using Tails = std::vector<uint32_t>;

std::optional<AddressSpace> space(Value value) {
  auto tile = dyn_cast<TileBufType>(value.getType());
  auto attr = tile ? dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace())
                   : AddressSpaceAttr();
  if (!attr)
    return std::nullopt;
  return attr.getAddressSpace();
}

bool equalsConstant(Value value, uint64_t expected) {
  IntegerAttr attr;
  return value && matchPattern(value, m_Constant(&attr)) &&
         !attr.getValue().isNegative() && attr.getValue().getActiveBits() <= 64 &&
         attr.getValue().getZExtValue() == expected;
}

// Full valid static descriptors on direct planned allocations only. A mutable
// valid-shape setter, view/alias, call, or forwarded handle prevents this local
// qualification. This is not a completeness gate for InsertSync itself.
bool staticDescriptor(Value value, AddressSpace expected) {
  auto type = dyn_cast<TileBufType>(value.getType());
  auto alloc = value.getDefiningOp<AllocTileOp>();
  if (!type || !alloc || space(value) != expected)
    return false;
  auto shape = type.getShape();
  auto valid = type.getValidShape();
  if (shape.size() != 2 || valid.size() != 2 || shape[0] <= 0 ||
      shape[1] <= 0 || shape != valid ||
      type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne))
    return false;
  IntegerAttr address;
  if (!alloc.getAddr() || !matchPattern(alloc.getAddr(), m_Constant(&address)) ||
      address.getValue().isNegative() || address.getValue().getActiveBits() > 64)
    return false;
  const uint64_t alignment = expected == AddressSpace::ACC ? 1024 : 512;
  if (address.getValue().getZExtValue() % alignment)
    return false;
  if ((alloc.getValidRow() && !equalsConstant(alloc.getValidRow(), valid[0])) ||
      (alloc.getValidCol() && !equalsConstant(alloc.getValidCol(), valid[1])))
    return false;
  for (Operation *user : value.getUsers())
    if (!isa<TMatmulOp, TMatmulAccOp, TExtractOp, TMovOp, TLoadOp, TStoreOp>(user))
      return false;
  return true;
}

struct Description {
  Value accumulator;
  Operation *context;
  uint64_t m, n, k;
  bool accumulate;
};

std::optional<Description> describe(Operation *op, Operation *context) {
  if (!context || op->getNumResults() || getSyncMacroModel(op))
    return std::nullopt;
  Value lhs, rhs, dst;
  bool accumulate = false;
  if (auto matrix = dyn_cast<TMatmulOp>(op)) {
    if (matrix.getAccPhase() != AccPhase::Unspecified)
      return std::nullopt;
    lhs = matrix.getLhs();
    rhs = matrix.getRhs();
    dst = matrix.getDst();
  } else if (auto matrix = dyn_cast<TMatmulAccOp>(op)) {
    if (matrix.getAccPhase() != AccPhase::Unspecified ||
        matrix.getAccIn() != matrix.getDst())
      return std::nullopt;
    lhs = matrix.getLhs();
    rhs = matrix.getRhs();
    dst = matrix.getDst();
    accumulate = true;
  } else {
    return std::nullopt;
  }
  if (!staticDescriptor(lhs, AddressSpace::LEFT) ||
      !staticDescriptor(rhs, AddressSpace::RIGHT) ||
      !staticDescriptor(dst, AddressSpace::ACC))
    return std::nullopt;
  auto a = cast<TileBufType>(lhs.getType());
  auto b = cast<TileBufType>(rhs.getType());
  auto c = cast<TileBufType>(dst.getType());
  // Initial target qualification is f16 x f16 -> f32, standard Cube layouts.
  // GEMV, BF16, bias, sparse/MX, partial shapes and explicit unit flags are not
  // silently generalized from this experiment.
  if (!a.getElementType().isF16() || !b.getElementType().isF16() ||
      !c.getElementType().isF32() ||
      a.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor) ||
      a.getSLayoutValueI32() != static_cast<int32_t>(SLayout::RowMajor) ||
      b.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor) ||
      b.getSLayoutValueI32() != static_cast<int32_t>(SLayout::ColMajor) ||
      c.getBLayoutValueI32() != static_cast<int32_t>(BLayout::ColMajor) ||
      c.getSLayoutValueI32() != static_cast<int32_t>(SLayout::RowMajor))
    return std::nullopt;
  uint64_t m = a.getValidShape()[0], k = a.getValidShape()[1];
  uint64_t n = b.getValidShape()[1];
  if (b.getValidShape()[0] != static_cast<int64_t>(k) ||
      c.getValidShape()[0] != static_cast<int64_t>(m) ||
      c.getValidShape()[1] != static_cast<int64_t>(n) || !isLargeMmadShape(m, n, k))
    return std::nullopt;
  return Description{dst, context, m, n, k, accumulate};
}

bool sameIdentity(const Description &a, const Description &b) {
  return a.accumulator == b.accumulator && a.context == b.context &&
         a.m == b.m && a.n == b.n && a.k == b.k;
}

class MmadGraphBuilder {
public:
  std::vector<MmadFlowNode> nodes;
  llvm::DenseMap<Operation *, uint32_t> operationNodes;
  llvm::DenseMap<Operation *, unsigned> keys;
  llvm::DenseMap<Operation *, Value> accumulators;
  bool limited = false;

  explicit MmadGraphBuilder(func::FuncOp function) {
    nodes.emplace_back(); // Entry's incoming state is explicitly unknown.
    auto kind = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
    Operation *context = kind && kind.getKernelKind() == FunctionKernelKind::Cube
                             ? function.getOperation() : nullptr;
    region(function.getBody(), {0}, context, 0);
  }

private:
  std::vector<Description> identities;

  Tails add(Tails predecessors, MmadFlowNode::Kind kind,
            unsigned key = kMmadUnknown, bool accumulate = false,
            Operation *operation = nullptr) {
    if (limited || nodes.size() >= 4096) {
      limited = true;
      return {};
    }
    uint32_t id = nodes.size();
    nodes.push_back({kind, key, accumulate, {}});
    for (uint32_t pred : predecessors)
      nodes[pred].successors.push_back(id);
    if (operation)
      operationNodes[operation] = id;
    return {id};
  }

  unsigned keyFor(const Description &description) {
    for (unsigned i = 0; i < identities.size(); ++i)
      if (sameIdentity(identities[i], description))
        return i + 2;
    identities.push_back(description);
    return identities.size() + 1;
  }

  Tails region(Region &body, Tails tails, Operation *context, unsigned depth) {
    if (limited)
      return {};
    if (depth > 32 || !llvm::hasSingleElement(body))
      return add(std::move(tails), MmadFlowNode::Kind::Reset);
    for (Operation &op : body.front()) {
      if (limited)
        return {};
      if (isa<scf::YieldOp, func::ReturnOp>(op))
        continue;
      if (auto choice = dyn_cast<scf::IfOp>(op)) {
        Tails left = region(choice.getThenRegion(), tails, context, depth + 1);
        Tails right = choice.getElseRegion().empty()
                          ? tails : region(choice.getElseRegion(), tails, context, depth + 1);
        left.insert(left.end(), right.begin(), right.end());
        tails = add(std::move(left), MmadFlowNode::Kind::PassThrough);
      } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
        // Header is both the body-entry join and the loop-exit frontier.
        // Keeping the zero-trip edge is conservative even for positive bounds.
        Tails header = add(std::move(tails), MmadFlowNode::Kind::PassThrough);
        Tails end = region(loop.getRegion(), header, context, depth + 1);
        if (limited)
          return {};
        for (uint32_t last : end)
          nodes[last].successors.push_back(header.front());
        tails = header;
      } else if (isa<SectionCubeOp>(op)) {
        tails = add(std::move(tails), MmadFlowNode::Kind::Reset);
        tails = region(op.getRegion(0), std::move(tails), &op, depth + 1);
        tails = add(std::move(tails), MmadFlowNode::Kind::Reset);
      } else if (op.getNumRegions()) {
        // No claims inside or across an unsupported region/physical section.
        tails = add(std::move(tails), MmadFlowNode::Kind::Reset);
      } else if (auto description = describe(&op, context)) {
        unsigned key = keyFor(*description);
        keys[&op] = key;
        accumulators[&op] = description->accumulator;
        tails = add(std::move(tails), MmadFlowNode::Kind::Matrix, key,
                    description->accumulate, &op);
      } else {
        bool breaks = static_cast<bool>(getSyncMacroModel(&op));
        if (auto pipe = dyn_cast<OpPipeInterface>(op))
          breaks |= pipe.getPipe() == PIPE::PIPE_M;
        if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> instances;
          effects.getEffects(instances);
          for (const auto &effect : instances) {
            Value value = effect.getValue();
            if (!isa<MemoryEffects::Read, MemoryEffects::Write,
                     MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect())) {
              breaks = true;
            } else if (!value || space(value) == AddressSpace::ACC) {
              breaks = true;
            }
          }
        } else if (!isa<SetFlagOp, WaitFlagOp, BarrierOp, AllocTileOp>(op) &&
                   !isMemoryEffectFree(&op)) {
          breaks = true;
        }
        if (breaks)
          tails = add(std::move(tails), MmadFlowNode::Kind::Reset);
      }
    }
    return tails;
  }
};
} // namespace

MmadChainAnalysis::MmadChainAnalysis(func::FuncOp function) {
  auto module = function->getParentOfType<ModuleOp>();
  auto arch = module ? module->getAttrOfType<StringAttr>("pto.target_arch") : StringAttr();
  if (!arch || (arch.getValue() != "a2" && arch.getValue() != "a3")) {
    reason = "unchanged: requires explicit A2/A3 target";
    return;
  }
  MmadGraphBuilder builder(function);
  if (builder.limited) {
    reason = "unchanged: matrix-chain graph budget";
    return;
  }
  auto result = analyzeMmadFlow(builder.nodes);
  transfers = result.transfers;
  if (result.status != MmadFlowResult::Status::Complete) {
    reason = "unchanged: incomplete matrix-chain fixed point";
    return;
  }
  keys = std::move(builder.keys);
  accumulators = std::move(builder.accumulators);
  for (auto [operation, node] : builder.operationNodes)
    if (result.eligible[node])
      eligible.insert(operation);
  complete = true;
  reason = "structured predecessor identities; no completion or token facts invented";
}

bool MmadChainAnalysis::discharges(
    const CompoundInstanceElement *source, const CompoundInstanceElement *target,
    const DepBaseMemInfoPairVec &dependencies) const {
  return target && eligible.contains(target->elementOp) && qualifiedPair(source, target, dependencies);
}

bool MmadChainAnalysis::dischargesWithPredecessors(
    const CompoundInstanceElement *source, const CompoundInstanceElement *target,
    const DepBaseMemInfoPairVec &dependencies, ArrayRef<Operation*> predecessors) const {
  if (!target || !isa<TMatmulAccOp>(target->elementOp) || predecessors.empty() ||
      !qualifiedPair(source, target, dependencies)) return false;
  auto key = keys.find(target->elementOp);
  for (Operation* predecessor : predecessors) {
    auto from = keys.find(predecessor);
    if (from == keys.end() || from->second != key->second) return false;
  }
  return true;
}

bool MmadChainAnalysis::qualifiedPair(
    const CompoundInstanceElement *source, const CompoundInstanceElement *target,
    const DepBaseMemInfoPairVec &dependencies) const {
  if (!complete || !source || !target || dependencies.empty() ||
      source->kPipeValue != PipelineType::PIPE_M ||
      target->kPipeValue != PipelineType::PIPE_M ||
      source->compoundCoreType != TCoreType::CUBE ||
      target->compoundCoreType != TCoreType::CUBE)
    return false;
  auto from = keys.find(source->elementOp), to = keys.find(target->elementOp);
  if (from == keys.end() || to == keys.end() || from->second != to->second)
    return false;
  Value accumulator = accumulators.lookup(target->elementOp);
  for (const auto &pair : dependencies) {
    const BaseMemInfo *a = pair.first, *b = pair.second;
    if (!a || !b || a->scope != AddressSpace::ACC || b->scope != AddressSpace::ACC ||
        a->baseBuffer != accumulator || b->baseBuffer != accumulator ||
        a->rootBuffer != accumulator || b->rootBuffer != accumulator ||
        a->aliasesUnknownRange || b->aliasesUnknownRange || !(*a == *b))
      return false;
  }
  // Only the in-place accumulator ordering is discharged. In particular DO
  // NOT update alreadySync[PIPE_M], retire L0 reads, or certify M->FIX output.
  // Earlier same-ACC writers are ordered through the intervening writer chain:
  // every link is either this intrinsic rule or an unchanged required repair.
  return true;
}
