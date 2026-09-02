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

#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.canonical_sync.generated";
constexpr StringLiteral kMechanismAttr = "pto.canonical_sync.mechanism";
constexpr StringLiteral kProtocolRoleAttr = "pto.canonical_sync.protocol_role";
constexpr StringLiteral kReleaseOwnerAttr = "pto.canonical_sync.release_owner";
constexpr StringLiteral kReleasePoolAttr = "pto.canonical_sync.release_pool";
constexpr StringLiteral kProtocolIndexAttr =
    "pto.canonical_sync.protocol_index";

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
  bool recurringReady = false;
  bool recurringRelease = false;
  bool boundaryReady = false;
  int64_t releaseOwner = -1;
  int64_t releasePool = -1;
  Operation *recurrenceLoop = nullptr;
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
  int64_t releasePool = -1;
};

struct ConcreteReleasePool {
  int64_t index = -1;
  int64_t mechanism = -1;
  Operation *primeSet = nullptr;
  Operation *drainWait = nullptr;
};

struct ConcreteSerializedStep {
  Operation *releaseWait = nullptr;
  Operation *readySet = nullptr;
  Operation *readyWait = nullptr;
  Operation *releaseSet = nullptr;
};

struct ConcreteSerializedProtocol {
  int64_t mechanism = -1;
  Operation *releasePrimeSet = nullptr;
  DenseMap<unsigned, ConcreteSerializedStep> steps;
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

bool hasRepeatingAncestor(Operation *operation);

bool controlsAreMutuallyExclusive(ArrayRef<ConcreteControlArm> first,
                                  ArrayRef<ConcreteControlArm> second) {
  return llvm::any_of(first, [&](const ConcreteControlArm &left) {
    return llvm::any_of(second, [&](const ConcreteControlArm &right) {
      return left.choice == right.choice && left.arm != right.arm &&
             !hasRepeatingAncestor(left.choice);
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
  if (role.starts_with("release-body-")) {
    auto releasePool = operation->getAttrOfType<IntegerAttr>(kReleasePoolAttr);
    if (releasePool) {
      const bool invalidPool =
          releasePool.getInt() < 0 ||
          (protocol.releasePool >= 0 &&
           protocol.releasePool != releasePool.getInt());
      if (invalidPool) {
        return operation->emitError(
            "canonical sync event verifier found an invalid release pool");
      }
      protocol.releasePool = releasePool.getInt();
    }
  }
  *slot = operation;
  return success();
}

LogicalResult collectReleasePoolOperation(
    Operation *operation, StringRef role,
    DenseMap<int64_t, ConcreteReleasePool> &releasePools) {
  FailureOr<int64_t> mechanism = getMechanismId(operation);
  auto index = operation->getAttrOfType<IntegerAttr>(kReleasePoolAttr);
  const bool valid = succeeded(mechanism) && index && index.getInt() >= 0;
  if (!valid) {
    return operation->emitError(
        "canonical sync event verifier found an invalid release-pool tag");
  }
  ConcreteReleasePool &pool = releasePools[index.getInt()];
  pool.index = index.getInt();
  if (pool.mechanism >= 0 && pool.mechanism != *mechanism) {
    return operation->emitError(
        "canonical sync event verifier found inconsistent release-pool "
        "ownership");
  }
  pool.mechanism = *mechanism;
  Operation **slot = role == "release-pool-prime-set"
                         ? &pool.primeSet
                         : role == "release-pool-drain-wait" ? &pool.drainWait
                                                              : nullptr;
  if (!slot) {
    return operation->emitError(
        "canonical sync event verifier found an unknown release-pool role");
  }
  if (*slot) {
    InFlightDiagnostic diagnostic = operation->emitError(
        "canonical sync event verifier found a duplicate release-pool "
        "action");
    diagnostic.attachNote((*slot)->getLoc()) << "previous action is here";
    return failure();
  }
  *slot = operation;
  return success();
}

LogicalResult collectSerializedProtocolOperation(
    Operation *operation, StringRef role,
    DenseMap<int64_t, ConcreteSerializedProtocol> &protocols) {
  FailureOr<int64_t> mechanism = getMechanismId(operation);
  auto index = operation->getAttrOfType<IntegerAttr>(kProtocolIndexAttr);
  const bool validIndex = succeeded(mechanism) && index && index.getInt() >= 0;
  if (!validIndex) {
    return operation->emitError(
        "canonical sync event verifier found an invalid serialized protocol "
        "index");
  }
  ConcreteSerializedProtocol &protocol = protocols[*mechanism];
  protocol.mechanism = *mechanism;
  Operation **slot = nullptr;
  if (role == "serial-release-prime-set") {
    slot = &protocol.releasePrimeSet;
  } else {
    ConcreteSerializedStep &step =
        protocol.steps[static_cast<unsigned>(index.getInt())];
    if (role == "serial-release-wait") {
      slot = &step.releaseWait;
    } else if (role == "serial-ready-set") {
      slot = &step.readySet;
    } else if (role == "serial-ready-wait") {
      slot = &step.readyWait;
    } else if (role == "serial-release-set") {
      slot = &step.releaseSet;
    }
  }
  if (!slot) {
    return operation->emitError(
        "canonical sync event verifier found an unknown serialized protocol "
        "role");
  }
  if (*slot) {
    InFlightDiagnostic diagnostic = operation->emitError(
        "canonical sync event verifier found a duplicate serialized "
        "protocol action");
    diagnostic.attachNote((*slot)->getLoc()) << "previous action is here";
    return failure();
  }
  *slot = operation;
  return success();
}

LogicalResult collectGenerations(
    func::FuncOp function,
    DenseMap<int64_t, ConcreteEventGeneration> &generations,
    DenseMap<int64_t, ConcreteRecurringProtocol> &protocols,
    DenseMap<int64_t, ConcreteSerializedProtocol> &serializedProtocols,
    DenseMap<int64_t, ConcreteReleasePool> &releasePools) {
  WalkResult result = function.walk([&](Operation *operation) -> WalkResult {
    auto role = operation->getAttrOfType<StringAttr>(kProtocolRoleAttr);
    if (role && isa<SetFlagOp, WaitFlagOp>(operation)) {
      if (role.getValue().starts_with("release-pool-")) {
        return failed(collectReleasePoolOperation(
                   operation, role.getValue(), releasePools))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      }
      const bool serialized = role.getValue().starts_with("serial-");
      const LogicalResult collected =
          serialized
              ? collectSerializedProtocolOperation(operation, role.getValue(),
                                                   serializedProtocols)
              : collectProtocolOperation(operation, role.getValue(), protocols);
      return failed(collected) ? WalkResult::interrupt()
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
  auto set = dyn_cast_or_null<SetFlagOp>(setOperation);
  auto wait = dyn_cast_or_null<WaitFlagOp>(waitOperation);
  if (!set || !wait || !eventActionsMatch(setOperation, waitOperation)) {
    Operation *witness = waitOperation ? waitOperation : setOperation;
    witness->emitError(
        "canonical sync event verifier found mismatched protocol endpoints");
    return failure();
  }
  const PIPE sourcePipe = set.getSrcPipe().getPipe();
  const PIPE targetPipe = set.getDstPipe().getPipe();
  FailureOr<CanonicalPhysicalResource> setSource =
      resolvePhysicalResource(function, setOperation, sourcePipe);
  FailureOr<CanonicalPhysicalResource> setTarget =
      resolvePhysicalResource(function, setOperation, targetPipe);
  FailureOr<CanonicalPhysicalResource> waitSource =
      resolvePhysicalResource(function, waitOperation, sourcePipe);
  FailureOr<CanonicalPhysicalResource> waitTarget =
      resolvePhysicalResource(function, waitOperation, targetPipe);
  const bool unresolved = failed(setSource) || failed(setTarget) ||
                          failed(waitSource) || failed(waitTarget);
  if (unresolved) {
    return failure();
  }
  const bool mismatchedContext =
      *setSource != *waitSource || *setTarget != *waitTarget;
  if (mismatchedContext) {
    waitOperation->emitError(
        "canonical sync event verifier found context-mismatched protocol "
        "endpoints");
    return failure();
  }
  const unsigned eventId = static_cast<unsigned>(set.getEventId().getEvent());
  if (failed(validateEventKey(function, target, setOperation, *setSource,
                              *setTarget, eventId))) {
    return failure();
  }
  ConcreteEventGeneration result;
  result.mechanism = mechanism;
  result.set = setOperation;
  result.wait = waitOperation;
  result.source = *setSource;
  result.target = *setTarget;
  result.eventId = eventId;
  result.controlPath = getControlPath(setOperation);
  return result;
}

bool eventGenerationKeysMatch(const ConcreteEventGeneration &first,
                              const ConcreteEventGeneration &second) {
  return first.source == second.source && first.target == second.target &&
         first.eventId == second.eventId;
}

bool protocolActionPrecedes(Operation *first, Operation *second);

LogicalResult resolveRecurringProtocol(
    func::FuncOp function, const CanonicalSyncTarget &target,
    const ConcreteRecurringProtocol &protocol,
    const ConcreteRecurringProtocol &releaseProtocol,
    bool emitReleaseGeneration,
    const DenseMap<int64_t, ConcreteReleasePool> &releasePools,
    SmallVectorImpl<ConcreteEventGeneration> &generations) {
  const bool boundaryReady = protocol.readyPrimeSet || protocol.readyDrainWait;
  const bool pooledRelease = releaseProtocol.releasePool >= 0;
  if (boundaryReady && pooledRelease) {
    Operation *witness = protocol.readyPrimeSet
                             ? protocol.readyPrimeSet
                             : function.getOperation();
    return witness->emitError(
        "canonical sync event verifier found a boundary recurrence using a "
        "release pool");
  }
  const auto pool = pooledRelease
                        ? releasePools.find(releaseProtocol.releasePool)
                        : releasePools.end();
  const bool completePool = !pooledRelease ||
                            (pool != releasePools.end() &&
                             pool->second.primeSet && pool->second.drainWait);
  const bool incomplete =
      !protocol.readyBodySet || !protocol.readyBodyWait ||
      !releaseProtocol.releaseBodyWait || !releaseProtocol.releaseBodySet ||
      !completePool ||
      (!pooledRelease && (!releaseProtocol.releasePrimeSet ||
                          !releaseProtocol.releaseDrainWait)) ||
      (pooledRelease && (releaseProtocol.releasePrimeSet ||
                         releaseProtocol.releaseDrainWait)) ||
      (boundaryReady && (!protocol.readyPrimeSet || !protocol.readyDrainWait));
  if (incomplete) {
    Operation *witness =
        protocol.readyBodySet
            ? protocol.readyBodySet
            : (releaseProtocol.releasePrimeSet
                   ? releaseProtocol.releasePrimeSet
                   : (pool != releasePools.end() && pool->second.primeSet
                          ? pool->second.primeSet
                          : function.getOperation()));
    return witness->emitError(
        "canonical sync event verifier found an incomplete recurring "
        "protocol");
  }
  Operation *releasePrime = pooledRelease ? pool->second.primeSet
                                          : releaseProtocol.releasePrimeSet;
  Operation *releaseDrain = pooledRelease ? pool->second.drainWait
                                          : releaseProtocol.releaseDrainWait;
  const bool mismatchedKeys =
      !eventActionsMatch(releasePrime, releaseProtocol.releaseBodyWait) ||
      !eventActionsMatch(releaseProtocol.releaseBodySet, releaseDrain) ||
      !setActionsMatch(releasePrime, releaseProtocol.releaseBodySet) ||
      (boundaryReady
           ? (!eventActionsMatch(protocol.readyPrimeSet,
                                 protocol.readyBodyWait) ||
              !eventActionsMatch(protocol.readyBodySet,
                                 protocol.readyDrainWait) ||
              !setActionsMatch(protocol.readyPrimeSet, protocol.readyBodySet))
           : !eventActionsMatch(protocol.readyBodySet, protocol.readyBodyWait));
  if (mismatchedKeys) {
    return releasePrime->emitError(
        "canonical sync event verifier found mismatched recurring protocol "
        "keys");
  }
  auto loop = releaseProtocol.releaseBodyWait->getParentOfType<scf::ForOp>();
  Block *body = loop ? loop.getBody() : nullptr;
  Block *readyBody = protocol.readyBodySet->getBlock();
  const bool bodyPlacement =
      loop && body && releaseProtocol.releaseBodyWait->getBlock() == body &&
      releaseProtocol.releaseBodySet->getBlock() == body &&
      readyBody == body &&
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
      pooledRelease ||
      (loop && releasePrime->getBlock() == loop->getBlock() &&
       releaseDrain->getBlock() == loop->getBlock() &&
       releasePrime->isBeforeInBlock(loop) &&
       loop->isBeforeInBlock(releaseDrain));
  const bool pooledBoundaryPlacement =
      !pooledRelease ||
      (protocolActionPrecedes(releasePrime,
                              releaseProtocol.releaseBodyWait) &&
       protocolActionPrecedes(releaseProtocol.releaseBodySet, releaseDrain));
  const bool readyBoundaryPlacement =
      !boundaryReady ||
      (loop && protocol.readyPrimeSet->getBlock() == loop->getBlock() &&
       protocol.readyDrainWait->getBlock() == loop->getBlock() &&
       protocol.readyPrimeSet->isBeforeInBlock(loop) &&
       loop->isBeforeInBlock(protocol.readyDrainWait));
  const bool repeatedBoundaryReady =
      boundaryReady && loop && hasRepeatingAncestor(loop.getOperation());
  if (!bodyOrder || !releaseBoundaryPlacement || !pooledBoundaryPlacement ||
      !readyBoundaryPlacement || repeatedBoundaryReady) {
    return releasePrime->emitError(
        "canonical sync event verifier found an invalid recurring protocol "
        "placement");
  }
  FailureOr<ConcreteEventGeneration> ready = makeProtocolGeneration(
      function, target, protocol.mechanism,
      boundaryReady ? protocol.readyPrimeSet : protocol.readyBodySet,
      boundaryReady ? protocol.readyDrainWait : protocol.readyBodyWait);
  FailureOr<ConcreteEventGeneration> releaseGeneration = makeProtocolGeneration(
      function, target, releaseProtocol.mechanism,
      releasePrime, releaseDrain);
  const bool unresolvedGeneration = failed(ready) || failed(releaseGeneration);
  if (unresolvedGeneration) {
    return failure();
  }
  const bool unbalanced = ready->source != releaseGeneration->target ||
                          ready->target != releaseGeneration->source ||
                          getControlPath(releasePrime) !=
                              getControlPath(releaseDrain);
  if (unbalanced) {
    return releasePrime->emitError(
        "canonical sync event verifier found an unbalanced recurring "
        "protocol");
  }
  ready->recurringReady = true;
  ready->boundaryReady = boundaryReady;
  ready->releaseOwner = protocol.releaseOwner;
  ready->releasePool = releaseProtocol.releasePool;
  ready->recurrenceLoop = loop.getOperation();
  generations.push_back(std::move(*ready));
  if (emitReleaseGeneration && !pooledRelease) {
    releaseGeneration->recurringRelease = true;
    generations.push_back(std::move(*releaseGeneration));
  }
  return success();
}

bool protocolActionPrecedes(Operation *first, Operation *second) {
  return first && second &&
         programPointMustPrecede(
             {first, CanonicalProgramPointPosition::After},
             {second, CanonicalProgramPointPosition::Before});
}

LogicalResult resolveReleasePool(
    func::FuncOp function, const CanonicalSyncTarget &target,
    const ConcreteReleasePool &pool,
    const DenseMap<int64_t, ConcreteRecurringProtocol> &protocols,
    SmallVectorImpl<ConcreteEventGeneration> &generations) {
  if (!pool.primeSet || !pool.drainWait || pool.mechanism < 0) {
    Operation *witness = pool.primeSet ? pool.primeSet : function.getOperation();
    return witness->emitError(
        "canonical sync event verifier found an incomplete recurring release "
        "pool");
  }
  const bool invalidBoundary =
      hasRepeatingAncestor(pool.primeSet) ||
      hasRepeatingAncestor(pool.drainWait) ||
      getControlPath(pool.primeSet) != getControlPath(pool.drainWait) ||
      !protocolActionPrecedes(pool.primeSet, pool.drainWait);
  if (invalidBoundary) {
    return pool.primeSet->emitError(
        "canonical sync event verifier found an invalid recurring release "
        "pool boundary");
  }
  FailureOr<ConcreteEventGeneration> generation = makeProtocolGeneration(
      function, target, pool.mechanism, pool.primeSet, pool.drainWait);
  if (failed(generation)) {
    return failure();
  }

  SmallVector<Operation *, 8> memberLoops;
  for (const auto &entry : protocols) {
    const ConcreteRecurringProtocol &protocol = entry.second;
    if (protocol.releasePool != pool.index) {
      continue;
    }
    auto loop = protocol.releaseBodyWait
                    ? protocol.releaseBodyWait->getParentOfType<scf::ForOp>()
                    : scf::ForOp();
    if (!loop || !protocol.releaseBodySet ||
        !eventActionsMatch(pool.primeSet, protocol.releaseBodyWait) ||
        !setActionsMatch(pool.primeSet, protocol.releaseBodySet) ||
        !protocolActionPrecedes(pool.primeSet, protocol.releaseBodyWait) ||
        !protocolActionPrecedes(protocol.releaseBodySet, pool.drainWait)) {
      return pool.primeSet->emitError(
          "canonical sync event verifier found a mismatched recurring "
          "release-pool member");
    }
    Operation *loopOperation = loop.getOperation();
    if (llvm::is_contained(memberLoops, loopOperation)) {
      return pool.primeSet->emitError(
          "canonical sync event verifier found multiple recurring release "
          "owners in one pooled loop");
    }
    memberLoops.push_back(loopOperation);
  }
  const bool validSingleton = memberLoops.size() == 1 &&
                              hasRepeatingAncestor(memberLoops.front());
  const bool degenerate =
      memberLoops.empty() || (memberLoops.size() == 1 && !validSingleton);
  if (degenerate) {
    return pool.primeSet->emitError(
        "canonical sync event verifier found a degenerate recurring release "
        "pool");
  }
  for (auto [index, loop] : llvm::enumerate(memberLoops)) {
    for (Operation *previous :
         ArrayRef<Operation *>(memberLoops).take_front(index)) {
      const bool nested =
          loop->isAncestor(previous) || previous->isAncestor(loop);
      if (nested) {
        return pool.primeSet->emitError(
            "canonical sync event verifier found nested recurring loops "
            "sharing one release pool");
      }
    }
  }
  generation->recurringRelease = true;
  generation->releasePool = pool.index;
  generations.push_back(std::move(*generation));
  return success();
}

LogicalResult resolveSerializedProtocol(
    func::FuncOp function, const CanonicalSyncTarget &target,
    const ConcreteSerializedProtocol &protocol,
    SmallVectorImpl<ConcreteEventGeneration> &generations) {
  const bool complete = protocol.releasePrimeSet && protocol.steps.size() >= 2;
  if (!complete) {
    Operation *witness = protocol.releasePrimeSet ? protocol.releasePrimeSet
                                                  : function.getOperation();
    return witness->emitError(
        "canonical sync event verifier found an incomplete serialized "
        "ready/release protocol");
  }
  const unsigned count = static_cast<unsigned>(protocol.steps.size());
  SmallVector<const ConcreteSerializedStep *, 8> steps;
  steps.reserve(count);
  for (unsigned index = 0; index < count; ++index) {
    auto found = protocol.steps.find(index);
    if (found == protocol.steps.end()) {
      return protocol.releasePrimeSet->emitError(
          "canonical sync event verifier found a non-contiguous serialized "
          "protocol");
    }
    steps.push_back(&found->second);
  }

  const ConcreteSerializedStep &first = *steps.front();
  if (!first.releaseWait || !first.readySet || !first.readyWait ||
      !eventActionsMatch(protocol.releasePrimeSet, first.releaseWait)) {
    return protocol.releasePrimeSet->emitError(
        "canonical sync event verifier found mismatched serialized protocol "
        "keys");
  }
  const SmallVector<ConcreteControlArm, 2> control =
      getControlPath(protocol.releasePrimeSet);
  Operation *previousReleaseSet = protocol.releasePrimeSet;
  std::optional<ConcreteEventGeneration> readyKey;
  std::optional<ConcreteEventGeneration> releaseKey;
  for (auto [index, stepPointer] : llvm::enumerate(steps)) {
    const ConcreteSerializedStep &step = *stepPointer;
    const bool last = index + 1 == steps.size();
    const bool incomplete = !step.releaseWait || !step.readySet ||
                            !step.readyWait || (!last && !step.releaseSet) ||
                            (last && step.releaseSet);
    const bool mismatched =
        incomplete ||
        !eventActionsMatch(previousReleaseSet, step.releaseWait) ||
        !eventActionsMatch(step.readySet, step.readyWait) ||
        !setActionsMatch(protocol.releasePrimeSet, previousReleaseSet) ||
        (!last && !setActionsMatch(protocol.releasePrimeSet, step.releaseSet));
    if (mismatched) {
      return step.releaseWait
                 ? step.releaseWait->emitError(
                       "canonical sync event verifier found mismatched "
                       "serialized protocol keys")
                 : protocol.releasePrimeSet->emitError(
                       "canonical sync event verifier found an incomplete "
                       "serialized protocol step");
    }
    const bool repeated =
        hasRepeatingAncestor(step.releaseWait) ||
        hasRepeatingAncestor(step.readySet) ||
        hasRepeatingAncestor(step.readyWait) ||
        (step.releaseSet && hasRepeatingAncestor(step.releaseSet));
    const bool balancedControl =
        getControlPath(step.releaseWait) == control &&
        getControlPath(step.readySet) == control &&
        getControlPath(step.readyWait) == control &&
        (!step.releaseSet || getControlPath(step.releaseSet) == control);
    const bool releaseArrives =
        protocolActionPrecedes(previousReleaseSet, step.releaseWait);
    const bool releaseBeforeReady =
        protocolActionPrecedes(step.releaseWait, step.readySet);
    const bool readyPairOrdered =
        protocolActionPrecedes(step.readySet, step.readyWait);
    const bool releaseReturned =
        last || (protocolActionPrecedes(step.readyWait, step.releaseSet) &&
                 protocolActionPrecedes(step.releaseSet,
                                        steps[index + 1]->releaseWait));
    const bool ordered = releaseArrives && releaseBeforeReady &&
                         readyPairOrdered && releaseReturned;
    if (repeated) {
      return step.releaseWait->emitError(
          "canonical sync event verifier found a loop-repeated serialized "
          "ready/release placement");
    }
    if (!balancedControl) {
      return step.releaseWait->emitError(
          "canonical sync event verifier found a control-unbalanced "
          "serialized ready/release placement");
    }
    if (!ordered) {
      return step.releaseWait->emitError(
                 "canonical sync event verifier found a misordered "
                 "serialized ready/release placement at step ")
             << index << " (release-arrives=" << releaseArrives
             << ", release-before-ready=" << releaseBeforeReady
             << ", ready-pair=" << readyPairOrdered
             << ", release-returned=" << releaseReturned << ')';
    }
    FailureOr<ConcreteEventGeneration> stepReady = makeProtocolGeneration(
        function, target, protocol.mechanism, step.readySet, step.readyWait);
    FailureOr<ConcreteEventGeneration> stepRelease =
        makeProtocolGeneration(function, target, protocol.mechanism,
                               previousReleaseSet, step.releaseWait);
    const bool validStep = succeeded(stepReady) && succeeded(stepRelease);
    if (!validStep) {
      return failure();
    }
    const bool readyMatches =
        !readyKey || eventGenerationKeysMatch(*readyKey, *stepReady);
    const bool releaseMatches =
        !releaseKey || eventGenerationKeysMatch(*releaseKey, *stepRelease);
    if (!readyMatches || !releaseMatches) {
      return step.releaseWait->emitError(
          "canonical sync event verifier found inconsistent physical keys "
          "across a serialized ready/release protocol");
    }
    readyKey = std::move(*stepReady);
    releaseKey = std::move(*stepRelease);
    if (!last) {
      previousReleaseSet = step.releaseSet;
    }
  }

  FailureOr<ConcreteEventGeneration> ready =
      makeProtocolGeneration(function, target, protocol.mechanism,
                             first.readySet, steps.back()->readyWait);
  FailureOr<ConcreteEventGeneration> release = makeProtocolGeneration(
      function, target, protocol.mechanism, protocol.releasePrimeSet,
      steps.back()->releaseWait);
  const bool generated = succeeded(ready) && succeeded(release);
  if (!generated) {
    return failure();
  }
  const bool unbalanced = ready->source != release->target ||
                          ready->target != release->source ||
                          ready->controlPath != release->controlPath;
  if (unbalanced) {
    return protocol.releasePrimeSet->emitError(
        "canonical sync event verifier found an unbalanced serialized "
        "ready/release protocol");
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
  const bool crossCore =
      isa<SyncSetOp>(generation.set) || isa<SyncWaitOp>(generation.wait);
  if (crossCore) {
    return generation.set->emitError(
        "canonical sync event verifier rejects cross-core synchronization "
        "because the target has no collective planner");
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
      differentBlocks || !generation.set->isBeforeInBlock(generation.wait);
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

bool concreteOnceOnlyLoopPrecedes(Operation *previousLoop,
                                  Operation *nextLoop) {
  const bool validLoops = isa_and_nonnull<scf::ForOp>(previousLoop) &&
                          isa_and_nonnull<scf::ForOp>(nextLoop);
  if (!validLoops) {
    return false;
  }
  SmallVector<Block *, 8> previousBlocks;
  for (Operation *current = previousLoop; current;
       current = current->getParentOp()) {
    if (current->getBlock()) {
      previousBlocks.push_back(current->getBlock());
    }
  }
  Block *commonBlock = nullptr;
  for (Operation *current = nextLoop; current;
       current = current->getParentOp()) {
    Block *block = current->getBlock();
    if (block && llvm::is_contained(previousBlocks, block)) {
      commonBlock = block;
      break;
    }
  }
  if (!commonBlock || !isa<func::FuncOp>(commonBlock->getParentOp())) {
    return false;
  }
  const auto liftToBlock = [](Operation *operation, Block *block) {
    while (operation) {
      Block *currentBlock = operation->getBlock();
      if (currentBlock == block) {
        return operation;
      }
      operation = operation->getParentOp();
    }
    return static_cast<Operation *>(nullptr);
  };
  Operation *previousAnchor = liftToBlock(previousLoop, commonBlock);
  Operation *nextAnchor = liftToBlock(nextLoop, commonBlock);
  return previousAnchor && nextAnchor && previousAnchor != nextAnchor &&
         previousAnchor->isBeforeInBlock(nextAnchor);
}

bool recurringReadyReuseIsCertified(
    const ConcreteEventGeneration &first, const ConcreteEventGeneration &second,
    ArrayRef<ConcreteEventGeneration> generations) {
  if (!first.recurringReady || !second.recurringReady || first.boundaryReady ||
      second.boundaryReady) {
    return false;
  }
  if (first.releasePool >= 0 && first.releasePool == second.releasePool &&
      first.recurrenceLoop != second.recurrenceLoop) {
    return true;
  }
  const auto ordered = [&](const ConcreteEventGeneration &previous,
                           const ConcreteEventGeneration &next) {
    auto release = llvm::find_if(
        generations, [&](const ConcreteEventGeneration &candidate) {
          return candidate.recurringRelease &&
                 candidate.mechanism == previous.releaseOwner;
        });
    const bool hasReverseRelease = release != generations.end() &&
                                   release->source == previous.target &&
                                   release->target == previous.source;
    return hasReverseRelease &&
           concreteOnceOnlyLoopPrecedes(previous.recurrenceLoop,
                                        next.recurrenceLoop) &&
           protocolActionPrecedes(release->wait, next.set);
  };
  return ordered(first, second) || ordered(second, first);
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
      if (recurringReadyReuseIsCertified(first, second, generations)) {
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
  DenseMap<int64_t, ConcreteSerializedProtocol> serializedProtocols;
  DenseMap<int64_t, ConcreteReleasePool> releasePools;
  if (failed(collectGenerations(function, byMechanism, protocols,
                                serializedProtocols, releasePools))) {
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
            protocol.mechanism == protocol.releaseOwner, releasePools,
            generations))) {
      return failure();
    }
  }
  for (const auto &entry : releasePools) {
    if (failed(resolveReleasePool(function, target, entry.second, protocols,
                                  generations))) {
      return failure();
    }
  }
  for (const auto &entry : serializedProtocols) {
    if (failed(resolveSerializedProtocol(function, target, entry.second,
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
