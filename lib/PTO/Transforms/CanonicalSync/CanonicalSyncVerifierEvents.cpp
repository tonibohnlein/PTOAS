// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncVerifier.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.canonical_sync.generated";
constexpr StringLiteral kMechanismAttr = "pto.canonical_sync.mechanism";
constexpr StringLiteral kProtocolRoleAttr = "pto.canonical_sync.protocol_role";

struct ConcreteControlArm {
  Operation *choice = nullptr;
  unsigned arm = 0;

  bool operator==(const ConcreteControlArm &other) const {
    return choice == other.choice && arm == other.arm;
  }
};

struct ConcreteEventGeneration {
  int64_t mechanism = -1;
  Operation *set = nullptr;
  Operation *wait = nullptr;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  unsigned eventId = 0;
  SmallVector<ConcreteControlArm, 2> controlPath;
};

struct ConcreteRecurringProtocol {
  int64_t mechanism = -1;
  Operation *releasePrimeSet = nullptr;
  Operation *releaseBodyWait = nullptr;
  Operation *readyBodySet = nullptr;
  Operation *readyBodyWait = nullptr;
  Operation *releaseBodySet = nullptr;
  Operation *releaseDrainWait = nullptr;
};

SmallVector<ConcreteControlArm, 2> getControlPath(Operation *operation) {
  SmallVector<ConcreteControlArm, 2> result;
  Region *region = operation ? operation->getParentRegion() : nullptr;
  while (region) {
    Operation *parent = region->getParentOp();
    if (!parent) {
      break;
    }
    if (auto choice = dyn_cast<scf::IfOp>(parent)) {
      const unsigned arm = region == &choice.getThenRegion() ? 0U : 1U;
      result.push_back({parent, arm});
    }
    region = parent->getParentRegion();
  }
  return result;
}

bool controlsAreMutuallyExclusive(ArrayRef<ConcreteControlArm> first,
                                  ArrayRef<ConcreteControlArm> second) {
  return llvm::any_of(first, [&](const ConcreteControlArm &left) {
    return llvm::any_of(second, [&](const ConcreteControlArm &right) {
      return left.choice == right.choice && left.arm != right.arm;
    });
  });
}

bool hasRepeatingAncestor(Operation *operation) {
  Operation *parent = operation ? operation->getParentOp() : nullptr;
  while (parent) {
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

FailureOr<int64_t> getMechanismId(Operation *operation) {
  if (!operation->hasAttr(kGeneratedAttr)) {
    operation->emitError(
        "canonical sync event verifier found an unowned event operation");
    return failure();
  }
  auto mechanism = operation->getAttrOfType<IntegerAttr>(kMechanismAttr);
  const bool invalidMechanism = !mechanism || mechanism.getInt() < 0;
  if (invalidMechanism) {
    operation->emitError(
        "canonical sync event verifier found an invalid generation tag");
    return failure();
  }
  return mechanism.getInt();
}

LogicalResult
collectEventOperation(Operation *operation, bool isSet,
                      DenseMap<int64_t, ConcreteEventGeneration> &generations) {
  FailureOr<int64_t> mechanism = getMechanismId(operation);
  if (failed(mechanism)) {
    return failure();
  }
  ConcreteEventGeneration &generation = generations[*mechanism];
  generation.mechanism = *mechanism;
  Operation *&slot = isSet ? generation.set : generation.wait;
  if (slot) {
    InFlightDiagnostic diagnostic =
        operation->emitError("canonical sync event verifier found duplicate "
                             "actions for generation m")
        << *mechanism;
    diagnostic.attachNote(slot->getLoc()) << "previous action is here";
    return failure();
  }
  slot = operation;
  return success();
}

LogicalResult collectProtocolOperation(
    Operation *operation, StringRef role,
    DenseMap<int64_t, ConcreteRecurringProtocol> &protocols) {
  FailureOr<int64_t> mechanism = getMechanismId(operation);
  if (failed(mechanism)) {
    return failure();
  }
  ConcreteRecurringProtocol &protocol = protocols[*mechanism];
  protocol.mechanism = *mechanism;
  Operation **slot = nullptr;
  if (role == "release-prime-set") {
    slot = &protocol.releasePrimeSet;
  } else if (role == "release-body-wait") {
    slot = &protocol.releaseBodyWait;
  } else if (role == "ready-body-set") {
    slot = &protocol.readyBodySet;
  } else if (role == "ready-body-wait") {
    slot = &protocol.readyBodyWait;
  } else if (role == "release-body-set") {
    slot = &protocol.releaseBodySet;
  } else if (role == "release-drain-wait") {
    slot = &protocol.releaseDrainWait;
  } else {
    return operation->emitError(
        "canonical sync event verifier found an unknown protocol role");
  }
  if (*slot) {
    InFlightDiagnostic diagnostic = operation->emitError(
        "canonical sync event verifier found a duplicate protocol action");
    diagnostic.attachNote((*slot)->getLoc()) << "previous action is here";
    return failure();
  }
  *slot = operation;
  return success();
}

LogicalResult
collectGenerations(func::FuncOp function,
                   DenseMap<int64_t, ConcreteEventGeneration> &generations,
                   DenseMap<int64_t, ConcreteRecurringProtocol> &protocols) {
  WalkResult result = function.walk([&](Operation *operation) -> WalkResult {
    auto role = operation->getAttrOfType<StringAttr>(kProtocolRoleAttr);
    if (role && isa<SetFlagOp, WaitFlagOp>(operation)) {
      return failed(collectProtocolOperation(operation, role.getValue(),
                                             protocols))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    }
    if (isa<SetFlagOp>(operation)) {
      return failed(collectEventOperation(operation, true, generations))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    }
    if (isa<WaitFlagOp>(operation)) {
      return failed(collectEventOperation(operation, false, generations))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool eventActionsMatch(Operation *setOperation, Operation *waitOperation) {
  auto set = dyn_cast_or_null<SetFlagOp>(setOperation);
  auto wait = dyn_cast_or_null<WaitFlagOp>(waitOperation);
  return set && wait && set.getSrcPipe() == wait.getSrcPipe() &&
         set.getDstPipe() == wait.getDstPipe() &&
         set.getEventId() == wait.getEventId();
}

bool setActionsMatch(Operation *firstOperation, Operation *secondOperation) {
  auto first = dyn_cast_or_null<SetFlagOp>(firstOperation);
  auto second = dyn_cast_or_null<SetFlagOp>(secondOperation);
  return first && second && first.getSrcPipe() == second.getSrcPipe() &&
         first.getDstPipe() == second.getDstPipe() &&
         first.getEventId() == second.getEventId();
}

LogicalResult
validateEventKey(func::FuncOp function, const CanonicalSyncTarget &target,
                 Operation *operation, CanonicalPhysicalResource source,
                 CanonicalPhysicalResource destination, unsigned eventId) {
  const bool unsupported =
      !target.supportsEvent(source, destination) ||
      !llvm::is_contained(target.getCompilerEventIds(), eventId);
  if (unsupported) {
    return operation->emitError(
        "canonical sync event verifier found an unsupported event key");
  }
  const SmallVector<unsigned, 6> reserved =
      reservedEventIds(function, source, destination);
  if (llvm::is_contained(reserved, eventId)) {
    return operation->emitError(
        "canonical sync event verifier found a macro-reserved event ID");
  }
  return success();
}

FailureOr<ConcreteEventGeneration>
makeProtocolGeneration(func::FuncOp function, const CanonicalSyncTarget &target,
                       int64_t mechanism, Operation *setOperation,
                       Operation *waitOperation) {
  auto set = cast<SetFlagOp>(setOperation);
  const PIPE sourcePipe = set.getSrcPipe().getPipe();
  const PIPE targetPipe = set.getDstPipe().getPipe();
  FailureOr<CanonicalPhysicalResource> source =
      resolvePhysicalResource(function, setOperation, sourcePipe);
  FailureOr<CanonicalPhysicalResource> destination =
      resolvePhysicalResource(function, setOperation, targetPipe);
  const bool unresolved = failed(source) || failed(destination);
  if (unresolved) {
    return failure();
  }
  const unsigned eventId = static_cast<unsigned>(set.getEventId().getEvent());
  if (failed(validateEventKey(function, target, setOperation, *source,
                              *destination, eventId))) {
    return failure();
  }
  ConcreteEventGeneration result;
  result.mechanism = mechanism;
  result.set = setOperation;
  result.wait = waitOperation;
  result.source = *source;
  result.target = *destination;
  result.eventId = eventId;
  result.controlPath = getControlPath(setOperation);
  return result;
}

LogicalResult resolveRecurringProtocol(
    func::FuncOp function, const CanonicalSyncTarget &target,
    const ConcreteRecurringProtocol &protocol,
    SmallVectorImpl<ConcreteEventGeneration> &generations) {
  const bool incomplete = !protocol.releasePrimeSet ||
                          !protocol.releaseBodyWait || !protocol.readyBodySet ||
                          !protocol.readyBodyWait || !protocol.releaseBodySet ||
                          !protocol.releaseDrainWait;
  if (incomplete) {
    Operation *witness = protocol.releasePrimeSet ? protocol.releasePrimeSet
                                                  : function.getOperation();
    return witness->emitError(
        "canonical sync event verifier found an incomplete recurring "
        "protocol");
  }
  const bool mismatchedKeys =
      !eventActionsMatch(protocol.releasePrimeSet, protocol.releaseBodyWait) ||
      !eventActionsMatch(protocol.releaseBodySet, protocol.releaseDrainWait) ||
      !eventActionsMatch(protocol.readyBodySet, protocol.readyBodyWait) ||
      !setActionsMatch(protocol.releasePrimeSet, protocol.releaseBodySet);
  if (mismatchedKeys) {
    return protocol.releasePrimeSet->emitError(
        "canonical sync event verifier found mismatched recurring protocol "
        "keys");
  }
  auto loop =
      dyn_cast_or_null<scf::ForOp>(protocol.releaseBodyWait->getParentOp());
  Block *body = loop ? &loop.getRegion().front() : nullptr;
  const bool bodyPlacement = body &&
                             protocol.readyBodySet->getBlock() == body &&
                             protocol.readyBodyWait->getBlock() == body &&
                             protocol.releaseBodySet->getBlock() == body;
  const bool bodyOrder =
      bodyPlacement &&
      protocol.releaseBodyWait->isBeforeInBlock(protocol.readyBodySet) &&
      protocol.readyBodySet->isBeforeInBlock(protocol.readyBodyWait) &&
      protocol.readyBodyWait->isBeforeInBlock(protocol.releaseBodySet);
  const bool boundaryPlacement =
      loop && protocol.releasePrimeSet->getBlock() == loop->getBlock() &&
      protocol.releaseDrainWait->getBlock() == loop->getBlock() &&
      protocol.releasePrimeSet->isBeforeInBlock(loop) &&
      loop->isBeforeInBlock(protocol.releaseDrainWait);
  if (!bodyOrder || !boundaryPlacement) {
    return protocol.releasePrimeSet->emitError(
        "canonical sync event verifier found an invalid recurring protocol "
        "placement");
  }
  FailureOr<ConcreteEventGeneration> ready =
      makeProtocolGeneration(function, target, protocol.mechanism,
                             protocol.readyBodySet, protocol.readyBodyWait);
  FailureOr<ConcreteEventGeneration> release = makeProtocolGeneration(
      function, target, protocol.mechanism, protocol.releasePrimeSet,
      protocol.releaseDrainWait);
  const bool unresolvedGeneration = failed(ready) || failed(release);
  if (unresolvedGeneration) {
    return failure();
  }
  const bool unbalanced = ready->source != release->target ||
                          ready->target != release->source ||
                          getControlPath(protocol.releasePrimeSet) !=
                              getControlPath(protocol.releaseDrainWait);
  if (unbalanced) {
    return protocol.releasePrimeSet->emitError(
        "canonical sync event verifier found an unbalanced recurring "
        "protocol");
  }
  generations.push_back(std::move(*ready));
  generations.push_back(std::move(*release));
  return success();
}

LogicalResult resolveGeneration(func::FuncOp function,
                                const CanonicalSyncTarget &target,
                                ConcreteEventGeneration &generation) {
  if (!generation.set || !generation.wait) {
    Operation *witness = generation.set ? generation.set : generation.wait;
    return witness->emitError(
        "canonical sync event verifier found an unbalanced generation");
  }
  auto set = cast<SetFlagOp>(generation.set);
  auto wait = cast<WaitFlagOp>(generation.wait);
  const PIPE setSource = set.getSrcPipe().getPipe();
  const PIPE setTarget = set.getDstPipe().getPipe();
  const unsigned setId = static_cast<unsigned>(set.getEventId().getEvent());
  const bool mismatchedPair =
      setSource != wait.getSrcPipe().getPipe() ||
      setTarget != wait.getDstPipe().getPipe() ||
      setId != static_cast<unsigned>(wait.getEventId().getEvent());
  if (mismatchedPair) {
    return generation.wait->emitError(
        "canonical sync event verifier found mismatched generation actions");
  }
  FailureOr<CanonicalPhysicalResource> source =
      resolvePhysicalResource(function, generation.set, setSource);
  FailureOr<CanonicalPhysicalResource> destination =
      resolvePhysicalResource(function, generation.set, setTarget);
  const bool unresolvedResources = failed(source) || failed(destination);
  if (unresolvedResources) {
    return failure();
  }
  generation.source = *source;
  generation.target = *destination;
  generation.eventId = setId;
  generation.controlPath = getControlPath(generation.set);
  const bool unbalancedPath =
      generation.controlPath != getControlPath(generation.wait);
  if (unbalancedPath) {
    return generation.wait->emitError(
        "canonical sync event verifier found path-unbalanced generation");
  }
  const bool differentBlocks =
      generation.set->getBlock() != generation.wait->getBlock();
  const bool setDoesNotPrecedeWait =
      !generation.set->isBeforeInBlock(generation.wait);
  if (differentBlocks || setDoesNotPrecedeWait) {
    return generation.wait->emitError("canonical sync event verifier cannot "
                                      "prove set-before-wait issue order");
  }
  const bool repeatedSet = hasRepeatingAncestor(generation.set);
  const bool repeatedWait = hasRepeatingAncestor(generation.wait);
  if (repeatedSet || repeatedWait) {
    return generation.set->emitError("canonical sync event verifier rejects a "
                                     "generation repeated by a loop");
  }
  return validateEventKey(function, target, generation.set, generation.source,
                          generation.target, generation.eventId);
}

LogicalResult
verifyInterference(ArrayRef<ConcreteEventGeneration> generations) {
  for (size_t firstIndex = 0; firstIndex < generations.size(); ++firstIndex) {
    const ConcreteEventGeneration &first = generations[firstIndex];
    for (size_t secondIndex = firstIndex + 1; secondIndex < generations.size();
         ++secondIndex) {
      const ConcreteEventGeneration &second = generations[secondIndex];
      const bool sameKey = first.source == second.source &&
                           first.target == second.target &&
                           first.eventId == second.eventId;
      if (!sameKey ||
          controlsAreMutuallyExclusive(first.controlPath, second.controlPath)) {
        continue;
      }
      InFlightDiagnostic diagnostic = second.set->emitError(
          "canonical sync event verifier found coexecuting generations sharing "
          "one physical event key");
      diagnostic.attachNote(first.set->getLoc())
          << "conflicting generation m" << first.mechanism << " is here";
      return failure();
    }
  }
  return success();
}

} // namespace

LogicalResult mlir::pto::canonical_sync_detail::verifyConcreteEventGenerations(
    func::FuncOp function, const CanonicalSyncTarget &target) {
  DenseMap<int64_t, ConcreteEventGeneration> byMechanism;
  DenseMap<int64_t, ConcreteRecurringProtocol> protocols;
  if (failed(collectGenerations(function, byMechanism, protocols))) {
    return failure();
  }
  SmallVector<ConcreteEventGeneration, 8> generations;
  generations.reserve(byMechanism.size());
  for (auto &entry : byMechanism) {
    if (failed(resolveGeneration(function, target, entry.second))) {
      return failure();
    }
    generations.push_back(std::move(entry.second));
  }
  for (const auto &entry : protocols) {
    if (failed(resolveRecurringProtocol(function, target, entry.second,
                                        generations))) {
      return failure();
    }
  }
  llvm::sort(generations, [](const ConcreteEventGeneration &left,
                             const ConcreteEventGeneration &right) {
    return left.mechanism < right.mechanism;
  });
  return verifyInterference(generations);
}
