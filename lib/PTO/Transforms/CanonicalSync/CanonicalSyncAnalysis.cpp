// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kHardwareEventIdCount = 8;

bool isManualSync(Operation *op) {
  const StringRef name = op->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Cases(RecordEventOp::getOperationName(), WaitEventOp::getOperationName(),
             BarrierSyncOp::getOperationName(), true)
      .Cases(SetFlagOp::getOperationName(), WaitFlagOp::getOperationName(),
             SetFlagDynOp::getOperationName(), true)
      .Cases(WaitFlagDynOp::getOperationName(), BarrierOp::getOperationName(),
             TSyncOp::getOperationName(), true)
      .Cases(FenceBarrierAllOp::getOperationName(),
             GetBufOp::getOperationName(), GetBufDynOp::getOperationName(),
             true)
      .Cases(RlsBufOp::getOperationName(), RlsBufDynOp::getOperationName(),
             DeclareEventIdArrayOp::getOperationName(), true)
      .Cases(EventIdArrayGetOp::getOperationName(),
             EventIdArraySetOp::getOperationName(),
             TPutAsyncOp::getOperationName(), true)
      .Cases(TGetAsyncOp::getOperationName(),
             WaitAsyncEventOp::getOperationName(),
             TestAsyncEventOp::getOperationName(), true)
      .Cases(MakePrefetchAsyncContextOp::getOperationName(),
             GetPrefetchAsyncSessionOp::getOperationName(),
             TPrefetchAsyncOp::getOperationName(), true)
      .Cases(SyncAllOp::getOperationName(), SyncSetOp::getOperationName(),
             SyncWaitOp::getOperationName(), true)
      .Cases(SetCrossBlockOp::getOperationName(),
             WaitCrossBlockOp::getOperationName(),
             SetIntraBlockOp::getOperationName(), true)
      .Case(WaitIntraBlockOp::getOperationName(), true)
      .Default(false);
}

bool isSupportedRegionContainer(Operation *op) {
  if (isa<func::FuncOp, scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
    return true;
  }
  return op->getName().getStringRef().starts_with("pto.section.");
}

bool isMemoryDefinition(Operation *op) {
  return isa<AllocTileOp, AllocMultiTileOp, DeclareTileOp, DeclareGlobalOp>(op);
}

bool hasNonPayloadMemoryEffects(Operation *op) {
  return isa<SetValidShapeOp, GetValidShapeOp, InitializeL2G2LPipeOp,
             InitializeL2LPipeOp, CmoCacheInvalidOp>(op);
}

bool isPhysicalMemoryValue(Value value) {
  if (!value) {
    return false;
  }
  return isa<PtrType, TileBufType, MultiTileBufType, TensorViewType,
             PartitionTensorViewType>(value.getType());
}

bool isTransferPipe(PipelineType pipe) {
  switch (static_cast<PIPE>(pipe)) {
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_MTE4:
  case PIPE::PIPE_MTE5:
  case PIPE::PIPE_FIX:
  case PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return true;
  default:
    return false;
  }
}

bool isGmPointerLikeArgument(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace().getAddressSpace() == AddressSpace::GM;
  }
  return isa<TensorViewType, PartitionTensorViewType>(type);
}

bool isKnownSyncHelper(func::CallOp call) {
  func::FuncOp callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  return callee && (callee->hasAttr("pto.tileop.helper") ||
                    callee->hasAttr("pto.ptodsl.subkernel_helper"));
}

bool sameAccessIdentity(const CanonicalMemoryAccess &first,
                        const CanonicalMemoryAccess &second) {
  return first.base == second.base && first.root == second.root &&
         first.space == second.space && first.addresses == second.addresses &&
         first.size == second.size &&
         first.knownPhysical == second.knownPhysical &&
         first.unknownRange == second.unknownRange;
}

void appendAccess(CanonicalSyncNode &node, const BaseMemInfo *info, bool reads,
                  bool writes) {
  if (!info) {
    return;
  }
  CanonicalMemoryAccess incoming;
  incoming.base = info->baseBuffer;
  incoming.root = info->rootBuffer;
  incoming.space = info->scope;
  incoming.addresses.append(info->baseAddresses.begin(),
                            info->baseAddresses.end());
  incoming.size = info->allocateSize;
  incoming.knownPhysical = info->hasKnownPhysicalAddresses;
  incoming.unknownRange = info->aliasesUnknownRange;
  incoming.reads = reads;
  incoming.writes = writes;
  auto existing = llvm::find_if(node.accesses, [&](const auto &access) {
    return sameAccessIdentity(access, incoming);
  });
  if (existing == node.accesses.end()) {
    node.accesses.push_back(std::move(incoming));
    return;
  }
  existing->reads |= reads;
  existing->writes |= writes;
}

bool intervalsOverlap(std::uint64_t firstStart, std::uint64_t firstSize,
                      std::uint64_t secondStart, std::uint64_t secondSize) {
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (firstSize == 0 || secondSize == 0 || firstSize > max - firstStart ||
      secondSize > max - secondStart) {
    return true;
  }
  const std::uint64_t firstEnd = firstStart + firstSize;
  const std::uint64_t secondEnd = secondStart + secondSize;
  return std::max(firstStart, secondStart) < std::min(firstEnd, secondEnd);
}

bool addressSetsOverlap(const CanonicalMemoryAccess &first,
                        const CanonicalMemoryAccess &second) {
  if (first.addresses.empty() || second.addresses.empty()) {
    return true;
  }
  for (std::uint64_t firstAddress : first.addresses) {
    for (std::uint64_t secondAddress : second.addresses) {
      if (intervalsOverlap(firstAddress, first.size, secondAddress,
                           second.size)) {
        return true;
      }
    }
  }
  return false;
}

Region *getRegionUnder(Operation *operation, Operation *ancestor) {
  for (Operation *current = operation; current && current != ancestor;
       current = current->getParentOp()) {
    if (current->getParentOp() == ancestor) {
      return current->getParentRegion();
    }
  }
  return nullptr;
}

} // namespace

FailureOr<CanonicalSyncPlan> CanonicalSyncPlanBuilder::build() {
  if (failed(validateInput())) {
    return failure();
  }
  translator_.Build();
  if (failed(collectNodes()) || failed(validateModeledEffects()) ||
      failed(addDependencies())) {
    return failure();
  }
  discardImpossibleRecurrences();
  preserveForwardCompletionRequirements();
  preserveRecurrenceCompletionRequirements();
  reduceForwardDependencies();
  reserveHiddenEventIds();
  materializeSyncRequirements();
  if (failed(repairEventScarcity())) {
    return failure();
  }
  if (failed(verifyFinalPlan())) {
    return failure();
  }
  if (failed(allocateEvents())) {
    return failure();
  }
  return std::move(plan_);
}

LogicalResult CanonicalSyncPlanBuilder::validateInput() {
  if (eventIdMax_ == 0 || eventIdMax_ > kHardwareEventIdCount) {
    return func_.emitError()
           << "canonical sync event-id maximum must be in [1, "
           << kHardwareEventIdCount << "]";
  }
  if (failed(parseNoAliasPairs())) {
    return failure();
  }
  WalkResult result = func_.walk([&](Operation *op) {
    if (isa<TAssignOp>(op)) {
      op->emitError("PTOCanonicalSync does not support pto.tassign");
      return WalkResult::interrupt();
    }
    if (auto alloc = dyn_cast<AllocMultiTileOp>(op)) {
      auto planned =
          alloc->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName);
      APInt base;
      const bool hasConstantBase =
          alloc.getAddr() &&
          matchPattern(alloc.getAddr(), m_ConstantInt(&base));
      const std::uint32_t slotCount = alloc.getResult().getType().getCount();
      if ((planned && planned.size() != slotCount) ||
          (!planned && !hasConstantBase)) {
        alloc.emitError("PTOCanonicalSync requires planner-assigned slot "
                        "addresses or a constant level3 base");
        return WalkResult::interrupt();
      }
    }
    if (isManualSync(op)) {
      op->emitError("PTOCanonicalSync does not accept input containing "
                    "manual synchronization");
      return WalkResult::interrupt();
    }
    if (op->getNumSuccessors() != 0) {
      op->emitError("PTOCanonicalSync does not support unstructured control "
                    "flow");
      return WalkResult::interrupt();
    }
    if (op->getNumRegions() != 0 && !isSupportedRegionContainer(op)) {
      op->emitError("PTOCanonicalSync cannot model this region operation");
      return WalkResult::interrupt();
    }
    if (auto call = dyn_cast<func::CallOp>(op);
        call && !isKnownSyncHelper(call)) {
      op->emitError("PTOCanonicalSync requires precise effects for helper "
                    "calls");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult CanonicalSyncPlanBuilder::parseNoAliasPairs() {
  Attribute rawPairs = func_->getAttr(kPtoNoAliasPairsAttrName);
  if (!rawPairs) {
    return success();
  }
  auto pairs = dyn_cast<DenseI64ArrayAttr>(rawPairs);
  if (!pairs) {
    return func_.emitError()
           << "expects '" << kPtoNoAliasPairsAttrName
           << "' to be a dense i64 array of argument-index pairs";
  }
  if (pairs.size() % 2 != 0) {
    return func_.emitError()
           << "expects '" << kPtoNoAliasPairsAttrName
           << "' to contain an even number of argument indices";
  }

  ArrayRef<int64_t> indices = pairs.asArrayRef();
  for (size_t index = 0; index < indices.size(); index += 2) {
    const int64_t first = indices[index];
    const int64_t second = indices[index + 1];
    if (first < 0 || second < 0 ||
        first >= static_cast<int64_t>(func_.getNumArguments()) ||
        second >= static_cast<int64_t>(func_.getNumArguments())) {
      return func_.emitError() << "'" << kPtoNoAliasPairsAttrName
                               << "' argument indices must be in [0, "
                               << func_.getNumArguments() << ")";
    }
    if (first == second) {
      return func_.emitError()
             << "'" << kPtoNoAliasPairsAttrName
             << "' cannot contain a self-pair for argument " << first;
    }
    if (!isGmPointerLikeArgument(func_.getArgument(first).getType()) ||
        !isGmPointerLikeArgument(func_.getArgument(second).getType())) {
      return func_.emitError()
             << "'" << kPtoNoAliasPairsAttrName << "' pair (" << first << ", "
             << second << ") must name GM pointer or view arguments";
    }

    const std::pair<unsigned, unsigned> pair = std::minmax(
        static_cast<unsigned>(first), static_cast<unsigned>(second));
    if (!noAliasArgPairs_.insert(pair).second) {
      return func_.emitError()
             << "'" << kPtoNoAliasPairsAttrName << "' repeats pair ("
             << pair.first << ", " << pair.second << ")";
    }
  }
  return success();
}

LogicalResult CanonicalSyncPlanBuilder::collectNodes() {
  for (const std::unique_ptr<InstanceElement> &element : syncIR_) {
    auto *compound = dyn_cast<CompoundInstanceElement>(element.get());
    if (!compound || !compound->elementOp) {
      continue;
    }
    CanonicalSyncNode node;
    node.id = plan_.nodes_.size();
    node.operation = compound->elementOp;
    node.pipe = compound->kPipeValue;
    node.macroPhase = compound->macroOpInstanceId;
    node.order = compound->GetIndex();
    if (isTransferPipe(node.pipe)) {
      node.transferWeight = 1;
    } else {
      node.computeWeight = 1;
    }
    for (const BaseMemInfo *read : compound->useVec) {
      appendAccess(node, read, true, false);
    }
    for (const BaseMemInfo *write : compound->defVec) {
      appendAccess(node, write, false, true);
    }
    operationNodes_[node.operation].push_back(node.id);
    plan_.nodes_.push_back(std::move(node));
  }
  return success();
}

LogicalResult CanonicalSyncPlanBuilder::validateModeledEffects() {
  WalkResult result = func_.walk([&](Operation *op) {
    if (isSupportedRegionContainer(op) || isMemoryDefinition(op) ||
        hasNonPayloadMemoryEffects(op)) {
      return WalkResult::advance();
    }
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) {
      if (isMemoryEffectFree(op) ||
          (isa<func::CallOp>(op) &&
           isKnownSyncHelper(cast<func::CallOp>(op)))) {
        return WalkResult::advance();
      }
      op->emitError("PTOCanonicalSync cannot prove this operation is free of "
                    "unmodeled memory effects");
      return WalkResult::interrupt();
    }
    SmallVector<MemoryEffects::EffectInstance, 4> instances;
    effects.getEffects(instances);
    const bool hasMemoryEffect =
        llvm::any_of(instances, [](const auto &effect) {
          return isa<MemoryEffects::Read, MemoryEffects::Write>(
              effect.getEffect());
        });
    if (!hasMemoryEffect) {
      return WalkResult::advance();
    }
    auto nodeIt = operationNodes_.find(op);
    if (nodeIt == operationNodes_.end()) {
      op->emitError("PTOCanonicalSync cannot model this memory-effecting "
                    "operation");
      return WalkResult::interrupt();
    }

    for (const MemoryEffects::EffectInstance &effect : instances) {
      const bool reads = isa<MemoryEffects::Read>(effect.getEffect());
      const bool writes = isa<MemoryEffects::Write>(effect.getEffect());
      if (!reads && !writes) {
        continue;
      }
      Value value = effect.getValue();
      if (!isPhysicalMemoryValue(value)) {
        continue;
      }
      const bool modeled = llvm::any_of(nodeIt->second, [&](auto node) {
        return llvm::any_of(plan_.nodes_[node].accesses,
                            [&](const CanonicalMemoryAccess &access) {
                              const bool sameValue =
                                  access.base == value || access.root == value;
                              return sameValue && (!reads || access.reads) &&
                                     (!writes || access.writes);
                            });
      });
      if (!modeled) {
        op->emitError("PTOCanonicalSync could not recover a declared memory "
                      "effect");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

bool CanonicalSyncPlanBuilder::memoryAliases(
    const CanonicalMemoryAccess &first, const CanonicalMemoryAccess &second,
    bool compareSlots) const {
  if (first.space != second.space) {
    return false;
  }
  if (first.unknownRange || second.unknownRange) {
    return true;
  }
  if (compareSlots && first.root == second.root) {
    const unsigned slots =
        std::max(first.addresses.size(), second.addresses.size());
    Value firstSlot = findMultiTileSlotExpr(first.base);
    Value secondSlot = findMultiTileSlotExpr(second.base);
    if (slots > 1 && firstSlot && secondSlot &&
        compareSlotSSA(firstSlot, secondSlot, slots) ==
            SlotRelation::kDisjoint) {
      return false;
    }
  }
  if (first.space == AddressSpace::GM) {
    if (rootsAreNoAlias(first.root, second.root)) {
      return false;
    }
    return first.root != second.root || addressSetsOverlap(first, second);
  }
  if (first.knownPhysical && second.knownPhysical) {
    return addressSetsOverlap(first, second);
  }
  if (first.root == second.root) {
    return addressSetsOverlap(first, second);
  }
  return true;
}

bool CanonicalSyncPlanBuilder::rootsAreNoAlias(Value first,
                                               Value second) const {
  auto firstArgument = dyn_cast<BlockArgument>(first);
  auto secondArgument = dyn_cast<BlockArgument>(second);
  if (!firstArgument || !secondArgument ||
      firstArgument.getOwner()->getParentOp() != funcOperation_ ||
      secondArgument.getOwner()->getParentOp() != funcOperation_) {
    return false;
  }
  const std::pair<unsigned, unsigned> pair =
      std::minmax(firstArgument.getArgNumber(), secondArgument.getArgNumber());
  return noAliasArgPairs_.find(pair) != noAliasArgPairs_.end();
}

bool CanonicalSyncPlanBuilder::memoryAliasesAcrossIterations(
    const CanonicalMemoryAccess &first, const CanonicalMemoryAccess &second,
    Operation *loop, unsigned iterationDistance) const {
  const bool aliases = memoryAliases(first, second, /*compareSlots=*/false);
  if (!aliases || first.root != second.root) {
    return aliases;
  }
  return compareSlotsAcrossIterations(first, second, loop, iterationDistance) !=
         SlotRelation::kDisjoint;
}

SlotRelation CanonicalSyncPlanBuilder::compareSlotsAcrossIterations(
    const CanonicalMemoryAccess &first, const CanonicalMemoryAccess &second,
    Operation *loop, unsigned iterationDistance) const {
  const std::size_t slots =
      std::max(first.addresses.size(), second.addresses.size());
  if (slots <= 1 || slots > kMaxMultiBufferCount) {
    return SlotRelation::kUnknown;
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(loop);
  if (!forOp) {
    return SlotRelation::kUnknown;
  }
  APInt step;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&step)) ||
      !step.isStrictlyPositive() || step.getActiveBits() > 63) {
    return SlotRelation::kUnknown;
  }
  const std::uint64_t unsignedStep = step.getZExtValue();
  const std::uint64_t maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (iterationDistance != 0 && unsignedStep > maximum / iterationDistance) {
    return SlotRelation::kUnknown;
  }
  Value firstSlot = findMultiTileSlotExpr(first.base);
  Value secondSlot = findMultiTileSlotExpr(second.base);
  if (!firstSlot || !secondSlot) {
    return SlotRelation::kUnknown;
  }
  const auto offset = static_cast<std::int64_t>(
      unsignedStep * static_cast<std::uint64_t>(iterationDistance));
  return compareSlotSSAWithOffset(firstSlot, secondSlot,
                                  static_cast<std::uint32_t>(slots),
                                  forOp.getInductionVar(), offset);
}

bool CanonicalSyncPlanBuilder::mayExecuteTogether(Operation *first,
                                                  Operation *second) const {
  for (Operation *parent = first->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(parent);
    if (!ifOp || !ifOp->isAncestor(second)) {
      continue;
    }
    return getRegionUnder(first, ifOp) == getRegionUnder(second, ifOp);
  }
  return true;
}

FailureOr<CanonicalSyncPlan>
mlir::pto::buildCanonicalSyncPlan(func::FuncOp func, unsigned eventIdMax) {
  return CanonicalSyncPlanBuilder(func, eventIdMax).build();
}
