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
constexpr StringLiteral kReleaseOwnerAttr = "pto.canonical_sync.release_owner";

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
  bool crossCore = false;
  SmallVector<ConcreteControlArm, 2> controlPath;
};

struct ConcreteRecurringProtocol {
  int64_t mechanism = -1;
  int64_t releaseOwner = -1;
  Operation *releasePrimeSet = nullptr;
  Operation *releaseBodyWait = nullptr;
  Operation *readyPrimeSet = nullptr;
  Operation *readyBodySet = nullptr;
  Operation *readyBodyWait = nullptr;
  Operation *readyDrainWait = nullptr;
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

bool hasOrderedUnconditionalFftsSetup(func::FuncOp function, Operation *source,
                                      Operation *target) {
  Block &entry = function.getBody().front();
  bool found = false;
  function.walk([&](SetFFTsOp setup) {
    Operation *operation = setup.getOperation();
    const bool unconditional = operation->getBlock() == &entry;
    const bool precedesEndpoints =
        programPointMustPrecede(
            {operation, CanonicalProgramPointPosition::After},
            {source, CanonicalProgramPointPosition::Before}) &&
        programPointMustPrecede(
            {operation, CanonicalProgramPointPosition::After},
            {target, CanonicalProgramPointPosition::Before});
    found |= unconditional && precedesEndpoints;
  });
  return found;
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
  } else if (role == "ready-prime-set") {
    slot = &protocol.readyPrimeSet;
  } else if (role == "ready-body-set") {
    slot = &protocol.readyBodySet;
  } else if (role == "ready-body-wait") {
    slot = &protocol.readyBodyWait;
  } else if (role == "ready-drain-wait") {
    slot = &protocol.readyDrainWait;
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
  if (role.starts_with("ready-")) {
    auto releaseOwner =
        operation->getAttrOfType<IntegerAttr>(kReleaseOwnerAttr);
    const bool invalidReleaseOwner =
        !releaseOwner || releaseOwner.getInt() < 0 ||
        (protocol.releaseOwner >= 0 &&
         protocol.releaseOwner != releaseOwner.getInt());
    if (invalidReleaseOwner) {
      return operation->emitError(
          "canonical sync event verifier found an invalid release owner");
    }
    protocol.releaseOwner = releaseOwner.getInt();
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
    if (isa<SetFlagOp, SyncSetOp>(operation)) {
      return failed(collectEventOperation(operation, true, generations))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    }
    if (isa<WaitFlagOp, SyncWaitOp>(operation)) {
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
    const ConcreteRecurringProtocol &releaseProtocol,
    bool emitReleaseGeneration,
    SmallVectorImpl<ConcreteEventGeneration> &generations) {
  const bool boundaryReady = protocol.readyPrimeSet || protocol.readyDrainWait;
  const bool incomplete =
      !protocol.readyBodySet || !protocol.readyBodyWait ||
      !releaseProtocol.releasePrimeSet || !releaseProtocol.releaseBodyWait ||
      !releaseProtocol.releaseBodySet || !releaseProtocol.releaseDrainWait ||
      (boundaryReady && (!protocol.readyPrimeSet || !protocol.readyDrainWait));
  if (incomplete) {
    Operation *witness =
        protocol.readyBodySet
            ? protocol.readyBodySet
            : (releaseProtocol.releasePrimeSet ? releaseProtocol.releasePrimeSet
                                               : function.getOperation());
    return witness->emitError(
        "canonical sync event verifier found an incomplete recurring "
        "protocol");
  }
  const bool mismatchedKeys =
      !eventActionsMatch(releaseProtocol.releasePrimeSet,
                         releaseProtocol.releaseBodyWait) ||
      !eventActionsMatch(releaseProtocol.releaseBodySet,
                         releaseProtocol.releaseDrainWait) ||
      !setActionsMatch(releaseProtocol.releasePrimeSet,
                       releaseProtocol.releaseBodySet) ||
      (boundaryReady
           ? (!eventActionsMatch(protocol.readyPrimeSet,
                                 protocol.readyBodyWait) ||
              !eventActionsMatch(protocol.readyBodySet,
                                 protocol.readyDrainWait) ||
              !setActionsMatch(protocol.readyPrimeSet, protocol.readyBodySet))
           : !eventActionsMatch(protocol.readyBodySet, protocol.readyBodyWait));
  if (mismatchedKeys) {
    return releaseProtocol.releasePrimeSet->emitError(
        "canonical sync event verifier found mismatched recurring protocol "
        "keys");
  }
  auto loop = releaseProtocol.releaseBodyWait->getParentOfType<scf::ForOp>();
  Block *body = loop ? loop.getBody() : nullptr;
  Block *readyBody = protocol.readyBodySet->getBlock();
  const bool bodyPlacement =
      loop && body && releaseProtocol.releaseBodyWait->getBlock() == body &&
      releaseProtocol.releaseBodySet->getBlock() == body && readyBody &&
      protocol.readyBodyWait->getBlock() == readyBody &&
      protocol.readyBodySet->getParentOfType<scf::ForOp>() == loop &&
      protocol.readyBodyWait->getParentOfType<scf::ForOp>() == loop &&
      releaseProtocol.releaseBodySet->getParentOfType<scf::ForOp>() == loop;
  const bool readyControlBalanced =
      bodyPlacement && getControlPath(protocol.readyBodySet) ==
                           getControlPath(protocol.readyBodyWait);
  const bool releaseBodyOrder = programPointMustPrecede(
      {releaseProtocol.releaseBodyWait, CanonicalProgramPointPosition::After},
      {releaseProtocol.releaseBodySet, CanonicalProgramPointPosition::Before});
  const bool bodyOrder =
      readyControlBalanced && releaseBodyOrder &&
      (boundaryReady
           ? programPointMustPrecede(
                 {protocol.readyBodyWait, CanonicalProgramPointPosition::After},
                 {protocol.readyBodySet, CanonicalProgramPointPosition::Before})
           : (programPointMustPrecede(
                  {releaseProtocol.releaseBodyWait,
                   CanonicalProgramPointPosition::After},
                  {protocol.readyBodySet,
                   CanonicalProgramPointPosition::Before}) &&
              protocol.readyBodySet->isBeforeInBlock(protocol.readyBodyWait) &&
              programPointMustPrecede(
                  {protocol.readyBodyWait,
                   CanonicalProgramPointPosition::After},
                  {releaseProtocol.releaseBodySet,
                   CanonicalProgramPointPosition::Before})));
  const bool releaseBoundaryPlacement =
      loop && releaseProtocol.releasePrimeSet->getBlock() == loop->getBlock() &&
      releaseProtocol.releaseDrainWait->getBlock() == loop->getBlock() &&
      releaseProtocol.releasePrimeSet->isBeforeInBlock(loop) &&
      loop->isBeforeInBlock(releaseProtocol.releaseDrainWait);
  const bool readyBoundaryPlacement =
      !boundaryReady ||
      (protocol.readyPrimeSet->getBlock() == loop->getBlock() &&
       protocol.readyDrainWait->getBlock() == loop->getBlock() &&
       protocol.readyPrimeSet->isBeforeInBlock(loop) &&
       loop->isBeforeInBlock(protocol.readyDrainWait));
  if (!bodyOrder || !releaseBoundaryPlacement || !readyBoundaryPlacement) {
    return releaseProtocol.releasePrimeSet->emitError(
        "canonical sync event verifier found an invalid recurring protocol "
        "placement");
  }
  FailureOr<ConcreteEventGeneration> ready = makeProtocolGeneration(
      function, target, protocol.mechanism,
      boundaryReady ? protocol.readyPrimeSet : protocol.readyBodySet,
      boundaryReady ? protocol.readyDrainWait : protocol.readyBodyWait);
  FailureOr<ConcreteEventGeneration> releaseGeneration = makeProtocolGeneration(
      function, target, releaseProtocol.mechanism,
      releaseProtocol.releasePrimeSet, releaseProtocol.releaseDrainWait);
  const bool unresolvedGeneration = failed(ready) || failed(releaseGeneration);
  if (unresolvedGeneration) {
    return failure();
  }
  const bool unbalanced = ready->source != releaseGeneration->target ||
                          ready->target != releaseGeneration->source ||
                          getControlPath(releaseProtocol.releasePrimeSet) !=
                              getControlPath(releaseProtocol.releaseDrainWait);
  if (unbalanced) {
    return releaseProtocol.releasePrimeSet->emitError(
        "canonical sync event verifier found an unbalanced recurring "
        "protocol");
  }
  generations.push_back(std::move(*ready));
  if (emitReleaseGeneration) {
    generations.push_back(std::move(*releaseGeneration));
  }
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
  const bool crossCore =
      isa<SyncSetOp>(generation.set) || isa<SyncWaitOp>(generation.wait);
  if (crossCore) {
    auto set = dyn_cast<SyncSetOp>(generation.set);
    auto wait = dyn_cast<SyncWaitOp>(generation.wait);
    const bool mismatchedKinds = !set || !wait;
    const IntegerAttr setIdAttr = set ? set.getEventIdAttr() : IntegerAttr();
    const IntegerAttr waitIdAttr = wait ? wait.getEventIdAttr() : IntegerAttr();
    const IntegerAttr setModeAttr = set ? set.getFftsModeAttr() : IntegerAttr();
    const IntegerAttr waitModeAttr =
        wait ? wait.getFftsModeAttr() : IntegerAttr();
    const bool dynamicId =
        (set && set.getEventIdDyn()) || (wait && wait.getEventIdDyn());
    const bool invalidStaticKey = !setIdAttr || !waitIdAttr || !setModeAttr ||
                                  !waitModeAttr || setIdAttr.getInt() < 0 ||
                                  waitIdAttr.getInt() < 0;
    const bool mismatchedKey =
        !invalidStaticKey &&
        (setIdAttr.getInt() != waitIdAttr.getInt() ||
         setModeAttr.getInt() != 2 || waitModeAttr.getInt() != 2);
    if (mismatchedKinds || dynamicId || invalidStaticKey || mismatchedKey) {
      return generation.wait->emitError(
          "canonical sync event verifier found mismatched cross-core "
          "generation actions");
    }
    FailureOr<CanonicalPhysicalResource> source = resolvePhysicalResource(
        function, generation.set, set.getPipe().getPipe());
    FailureOr<CanonicalPhysicalResource> destination = resolvePhysicalResource(
        function, generation.wait, wait.getPipe().getPipe());
    const bool unresolvedResources = failed(source) || failed(destination);
    if (unresolvedResources) {
      return failure();
    }
    const unsigned eventId = static_cast<unsigned>(setIdAttr.getInt());
    const bool unsupported =
        !hasOrderedUnconditionalFftsSetup(function, generation.set,
                                          generation.wait) ||
        !target.supportsCrossCoreEvent(*source, *destination) ||
        !llvm::is_contained(target.getCompilerCrossCoreEventIds(), eventId);
    if (unsupported) {
      return generation.set->emitError(
          "canonical sync event verifier found an unsupported cross-core "
          "event key or missing FFTS setup");
    }
    generation.source = *source;
    generation.target = *destination;
    generation.eventId = eventId;
    generation.crossCore = true;
    generation.controlPath = getControlPath(generation.set);
    if (generation.controlPath != getControlPath(generation.wait)) {
      return generation.wait->emitError(
          "canonical sync event verifier found path-unbalanced cross-core "
          "generation");
    }
    const bool issueOrdered = programPointMustPrecede(
        {generation.set, CanonicalProgramPointPosition::After},
        {generation.wait, CanonicalProgramPointPosition::Before});
    if (!issueOrdered) {
      return generation.wait->emitError(
          "canonical sync event verifier cannot prove cross-core "
          "set-before-wait issue order");
    }
    const bool repeated = hasRepeatingAncestor(generation.set) ||
                          hasRepeatingAncestor(generation.wait);
    if (repeated) {
      return generation.set->emitError(
          "canonical sync event verifier rejects a cross-core generation "
          "repeated by a loop");
    }
    return success();
  }
  auto set = dyn_cast<SetFlagOp>(generation.set);
  auto wait = dyn_cast<WaitFlagOp>(generation.wait);
  if (!set || !wait) {
    return generation.wait->emitError(
        "canonical sync event verifier found mismatched generation kinds");
  }
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
      const bool sameCrossCoreKey = first.crossCore && second.crossCore &&
                                    first.eventId == second.eventId;
      const bool sameIntraCoreKey = !first.crossCore && !second.crossCore &&
                                    first.source == second.source &&
                                    first.target == second.target &&
                                    first.eventId == second.eventId;
      const bool sameKey = sameCrossCoreKey || sameIntraCoreKey;
      if (!sameKey ||
          controlsAreMutuallyExclusive(first.controlPath, second.controlPath)) {
        continue;
      }
      InFlightDiagnostic diagnostic =
          second.set->emitError("canonical sync event verifier found "
                                "coexecuting generations sharing "
                                "one physical event key; generation m")
          << second.mechanism << " conflicts with generation m"
          << first.mechanism;
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
    const ConcreteRecurringProtocol &protocol = entry.second;
    auto release = protocols.find(protocol.releaseOwner);
    if (protocol.releaseOwner < 0 || release == protocols.end()) {
      return function.emitError(
          "canonical sync event verifier cannot resolve a release group");
    }
    if (failed(resolveRecurringProtocol(
            function, target, protocol, release->second,
            protocol.mechanism == protocol.releaseOwner, generations))) {
      return failure();
    }
  }
  llvm::sort(generations, [](const ConcreteEventGeneration &left,
                             const ConcreteEventGeneration &right) {
    return left.mechanism < right.mechanism;
  });
  return verifyInterference(generations);
}
