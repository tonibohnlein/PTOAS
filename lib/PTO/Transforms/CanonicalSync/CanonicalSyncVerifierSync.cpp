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

#include "PTO/IR/PTOTypeUtils.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr StringLiteral kGeneratedAttr = "pto.canonical_sync.generated";
constexpr StringLiteral kMechanismAttr = "pto.canonical_sync.mechanism";

VerifierResourceState &getResource(VerifierState &state,
                                   CanonicalPhysicalResource resource) {
  auto iterator =
      llvm::find_if(state.resources, [&](const VerifierResourceState &entry) {
        return entry.resource == resource;
      });
  if (iterator == state.resources.end()) {
    state.resources.push_back({resource, {}, {}, {}, {}});
    return state.resources.back();
  }
  return *iterator;
}

void addKey(SmallVectorImpl<VerifierEffectKey> &keys,
            const VerifierEffectKey &key) {
  if (!llvm::is_contained(keys, key)) {
    keys.push_back(key);
  }
}

void addPendingKeys(const VerifierResourceState &resource,
                    SmallVectorImpl<VerifierEffectKey> &keys) {
  for (const VerifierEffect &effect : resource.pending) {
    addKey(keys, effect.key);
  }
  for (const VerifierEffectKey &key : resource.pendingCompletions) {
    addKey(keys, key);
  }
}

bool cacheActionsEqual(const VerifierCacheAction &first,
                       const VerifierCacheAction &second) {
  return first.operation == second.operation && first.allGm == second.allGm &&
         first.access.value == second.access.value &&
         first.access.aliasRoot == second.access.aliasRoot;
}

void completeResource(VerifierResourceState &resource, VerifierState &state) {
  addPendingKeys(resource, resource.known);
  for (const VerifierEffect &effect : resource.pending) {
    addKey(state.exitComplete, effect.key);
  }
  for (const VerifierEffectKey &key : resource.pendingCompletions) {
    addKey(state.exitComplete, key);
  }
}

LogicalResult resolveEventResources(const VerifierProgram &program,
                                    Operation *operation, PIPE sourcePipe,
                                    PIPE targetPipe,
                                    CanonicalPhysicalResource &source,
                                    CanonicalPhysicalResource &target) {
  FailureOr<CanonicalPhysicalResource> sourceResult =
      resolvePhysicalResource(program.function, operation, sourcePipe);
  FailureOr<CanonicalPhysicalResource> targetResult =
      resolvePhysicalResource(program.function, operation, targetPipe);
  const bool invalidResources = failed(sourceResult) || failed(targetResult);
  if (invalidResources) {
    return failure();
  }
  source = *sourceResult;
  target = *targetResult;
  return success();
}

LogicalResult
validateEvent(const VerifierProgram &program, const CanonicalSyncTarget &target,
              Operation *operation, CanonicalPhysicalResource source,
              CanonicalPhysicalResource destination, unsigned eventId) {
  if (!target.supportsEvent(source, destination)) {
    return operation->emitError(
        "canonical sync verifier found an illegal directed hardware event");
  }
  if (!llvm::is_contained(target.getCompilerEventIds(), eventId)) {
    return operation->emitError(
        "canonical sync verifier found an event ID outside the compiler pool");
  }
  const SmallVector<unsigned, 6> reserved =
      reservedEventIds(program.function, source, destination);
  if (llvm::is_contained(reserved, eventId)) {
    return operation->emitError("canonical sync verifier found a compiler "
                                "event ID reserved by a macro");
  }
  return success();
}

LogicalResult applySet(const VerifierProgram &program,
                       const CanonicalSyncTarget &target, SetFlagOp operation,
                       VerifierState &state) {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource destination;
  const unsigned eventId =
      static_cast<unsigned>(operation.getEventId().getEvent());
  if (failed(resolveEventResources(
          program, operation, operation.getSrcPipe().getPipe(),
          operation.getDstPipe().getPipe(), source, destination)) ||
      failed(validateEvent(program, target, operation, source, destination,
                           eventId))) {
    return failure();
  }
  auto collision = llvm::find_if(state.tokens, [&](const VerifierToken &token) {
    return token.source == source && token.target == destination &&
           token.eventId == eventId;
  });
  if (collision != state.tokens.end()) {
    return operation.emitError(
        "canonical sync verifier found a flag set before its prior wait");
  }
  VerifierToken token;
  token.source = source;
  token.target = destination;
  token.eventId = eventId;
  const VerifierResourceState &resource = getResource(state, source);
  addPendingKeys(resource, token.payload);
  for (const VerifierEffectKey &key : resource.known) {
    addKey(token.payload, key);
  }
  for (const VerifierEffectKey &key : state.globalKnown) {
    addKey(token.payload, key);
  }
  state.tokens.push_back(std::move(token));
  return success();
}

LogicalResult applyWait(const VerifierProgram &program,
                        const CanonicalSyncTarget &target, WaitFlagOp operation,
                        VerifierState &state) {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource destination;
  const unsigned eventId =
      static_cast<unsigned>(operation.getEventId().getEvent());
  if (failed(resolveEventResources(
          program, operation, operation.getSrcPipe().getPipe(),
          operation.getDstPipe().getPipe(), source, destination)) ||
      failed(validateEvent(program, target, operation, source, destination,
                           eventId))) {
    return failure();
  }
  auto token = llvm::find_if(state.tokens, [&](const VerifierToken &entry) {
    return entry.source == source && entry.target == destination &&
           entry.eventId == eventId;
  });
  if (token == state.tokens.end()) {
    return operation.emitError(
        "canonical sync verifier found a wait without a matching set");
  }
  VerifierResourceState &resource = getResource(state, destination);
  for (const VerifierEffectKey &key : token->payload) {
    addKey(resource.known, key);
    addKey(state.exitComplete, key);
  }
  state.tokens.erase(token);
  return success();
}

template <typename CounterpartOp>
FailureOr<CounterpartOp>
findCrossCoreCounterpart(const VerifierProgram &program, Operation *operation) {
  if (!operation->hasAttr(kGeneratedAttr)) {
    operation->emitError(
        "canonical sync verifier found an unowned cross-core event");
    return failure();
  }
  auto mechanism = operation->getAttrOfType<IntegerAttr>(kMechanismAttr);
  if (!mechanism) {
    operation->emitError(
        "canonical sync verifier found an untagged cross-core event");
    return failure();
  }
  CounterpartOp result;
  func::FuncOp function = program.function;
  function.walk([&](CounterpartOp candidate) {
    auto candidateMechanism =
        candidate->template getAttrOfType<IntegerAttr>(kMechanismAttr);
    if (candidateMechanism && candidateMechanism == mechanism) {
      result = candidate;
    }
  });
  if (!result) {
    operation->emitError(
        "canonical sync verifier found an unbalanced cross-core event");
    return failure();
  }
  return result;
}

LogicalResult resolveCrossCoreEvent(const VerifierProgram &program,
                                    const CanonicalSyncTarget &target,
                                    SyncSetOp set, SyncWaitOp wait,
                                    CanonicalPhysicalResource &source,
                                    CanonicalPhysicalResource &destination,
                                    unsigned &eventId) {
  IntegerAttr setId = set.getEventIdAttr();
  IntegerAttr waitId = wait.getEventIdAttr();
  IntegerAttr setMode = set.getFftsModeAttr();
  IntegerAttr waitMode = wait.getFftsModeAttr();
  const bool invalid = !setId || !waitId || set.getEventIdDyn() ||
                       wait.getEventIdDyn() || !setMode || !waitMode ||
                       setId.getInt() < 0 || setId != waitId ||
                       setMode.getInt() != 2 || waitMode.getInt() != 2;
  if (invalid) {
    return set.emitError(
        "canonical sync verifier found an invalid cross-core event pair");
  }
  FailureOr<CanonicalPhysicalResource> sourceResult =
      resolvePhysicalResource(program.function, set, set.getPipe().getPipe());
  FailureOr<CanonicalPhysicalResource> destinationResult =
      resolvePhysicalResource(program.function, wait, wait.getPipe().getPipe());
  const bool unresolvedResources =
      failed(sourceResult) || failed(destinationResult);
  if (unresolvedResources) {
    return failure();
  }
  source = *sourceResult;
  destination = *destinationResult;
  eventId = static_cast<unsigned>(setId.getInt());
  const bool unsupported =
      !target.supportsCrossCoreEvent(source, destination) ||
      !llvm::is_contained(target.getCompilerCrossCoreEventIds(), eventId);
  if (unsupported) {
    return set.emitError(
        "canonical sync verifier found an unsupported cross-core event key");
  }
  return success();
}

LogicalResult applyCrossCoreSet(const VerifierProgram &program,
                                const CanonicalSyncTarget &target,
                                SyncSetOp operation, VerifierState &state) {
  FailureOr<SyncWaitOp> wait =
      findCrossCoreCounterpart<SyncWaitOp>(program, operation);
  if (failed(wait)) {
    return failure();
  }
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource destination;
  unsigned eventId = 0;
  if (failed(resolveCrossCoreEvent(program, target, operation, *wait, source,
                                   destination, eventId))) {
    return failure();
  }
  auto collision = llvm::find_if(state.tokens, [&](const VerifierToken &token) {
    return token.source == source && token.target == destination &&
           token.eventId == eventId;
  });
  if (collision != state.tokens.end()) {
    return operation.emitError(
        "canonical sync verifier found a cross-core counter set before its "
        "prior wait");
  }
  VerifierToken token;
  token.source = source;
  token.target = destination;
  token.eventId = eventId;
  const VerifierResourceState &resource = getResource(state, source);
  addPendingKeys(resource, token.payload);
  for (const VerifierEffectKey &key : resource.known) {
    addKey(token.payload, key);
  }
  for (const VerifierEffectKey &key : state.globalKnown) {
    addKey(token.payload, key);
  }
  state.tokens.push_back(std::move(token));
  return success();
}

LogicalResult applyCrossCoreWait(const VerifierProgram &program,
                                 const CanonicalSyncTarget &target,
                                 SyncWaitOp operation, VerifierState &state) {
  FailureOr<SyncSetOp> set =
      findCrossCoreCounterpart<SyncSetOp>(program, operation);
  if (failed(set)) {
    return failure();
  }
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource destination;
  unsigned eventId = 0;
  if (failed(resolveCrossCoreEvent(program, target, *set, operation, source,
                                   destination, eventId))) {
    return failure();
  }
  auto token = llvm::find_if(state.tokens, [&](const VerifierToken &entry) {
    return entry.source == source && entry.target == destination &&
           entry.eventId == eventId;
  });
  if (token == state.tokens.end()) {
    return operation.emitError(
        "canonical sync verifier found a cross-core wait without a matching "
        "set");
  }
  VerifierResourceState &resource = getResource(state, destination);
  for (const VerifierEffectKey &key : token->payload) {
    addKey(resource.known, key);
    addKey(state.exitComplete, key);
  }
  state.tokens.erase(token);
  return success();
}

LogicalResult applyBarrier(const VerifierProgram &program,
                           const CanonicalSyncTarget &target,
                           BarrierOp operation, VerifierState &state) {
  const PIPE pipe = operation.getPipe().getPipe();
  if (pipe == PIPE::PIPE_ALL) {
    const std::optional<bool> vectorExecution =
        resolvePTOExecutionVector(operation);
    if (!vectorExecution) {
      return operation.emitError(
          "canonical sync verifier cannot resolve the execution core for "
          "PIPE_ALL");
    }
    const CanonicalCore core =
        *vectorExecution ? CanonicalCore::AIV : CanonicalCore::AIC;
    for (VerifierResourceState &resource : state.resources) {
      if (resource.resource.core != core) {
        continue;
      }
      completeResource(resource, state);
      for (const VerifierEffect &effect : resource.pending) {
        addKey(state.globalKnown, effect.key);
      }
      for (const VerifierEffectKey &key : resource.pendingCompletions) {
        addKey(state.globalKnown, key);
      }
    }
    return success();
  }
  FailureOr<CanonicalPhysicalResource> physical =
      resolvePhysicalResource(program.function, operation, pipe);
  if (failed(physical)) {
    return failure();
  }
  if (!target.supportsPipeBarrier(*physical)) {
    return operation.emitError(
        "canonical sync verifier found an unsupported pipe barrier");
  }
  completeResource(getResource(state, *physical), state);
  return success();
}

void applyCmo(const VerifierProgram &program, CmoCacheInvalidOp operation,
              VerifierState &state) {
  auto found = program.cacheActions.find(operation);
  if (found == program.cacheActions.end()) {
    return;
  }
  for (const VerifierCacheAction &action : found->second) {
    const bool alreadyInvalidated = llvm::any_of(
        state.cacheInvalidations, [&](const VerifierCacheAction &existing) {
          return cacheActionsEqual(existing, action);
        });
    if (!alreadyInvalidated) {
      state.cacheInvalidations.push_back(action);
    }
    for (const VerifierResourceState &resource : state.resources) {
      if (resource.resource.pipe != PIPE::PIPE_S) {
        continue;
      }
      for (const VerifierEffect &effect : resource.pending) {
        const bool writes = accessWrites(effect.access.mode);
        const bool cacheCovered =
            verifierCacheActionCovers(action, effect.access);
        if (writes && cacheCovered) {
          addKey(state.cacheMaintained, effect.key);
        }
      }
    }
  }
}

LogicalResult applyFenceAll(const VerifierProgram &program,
                            const CanonicalSyncTarget &target,
                            FenceBarrierAllOp operation, VerifierState &state) {
  const FenceScope scope = operation.getScope().getScope();
  if (scope != FenceScope::GM && scope != FenceScope::All) {
    return operation.emitError(
        "canonical sync verifier rejects an unsupported physical fence scope");
  }
  FailureOr<SmallVector<CanonicalPhysicalResource, 8>> drained =
      target.getFenceDrainedResources(operation);
  if (failed(drained)) {
    return operation.emitError(
        "canonical sync verifier cannot resolve the physical execution "
        "context for fence");
  }
  for (VerifierResourceState &resource : state.resources) {
    if (!llvm::is_contained(*drained, resource.resource)) {
      continue;
    }
    completeResource(resource, state);
    for (const VerifierEffectKey &key : resource.known) {
      addKey(state.globalKnown, key);
    }
    for (const VerifierEffect &effect : resource.pending) {
      addKey(state.globalKnown, effect.key);
      const bool scalarWrite = resource.resource.pipe == PIPE::PIPE_S &&
                               accessWrites(effect.access.mode);
      if (!scalarWrite ||
          llvm::is_contained(state.cacheMaintained, effect.key)) {
        addKey(state.globalVisible, effect.key);
      }
    }
    for (const VerifierEffectKey &key : resource.pendingCompletions) {
      addKey(state.globalKnown, key);
    }
  }
  // A cache invalidation performed before this publication point cannot
  // acquire data that only becomes globally visible at this fence. Require a
  // fresh target-side invalidation after the fence.
  state.cacheInvalidations.clear();
  return success();
}

} // namespace

LogicalResult mlir::pto::canonical_sync_detail::applyVerifierSyncOperation(
    const VerifierProgram &program, const CanonicalSyncTarget &target,
    Operation *operation, VerifierState &state, bool &handled) {
  handled = true;
  if (auto set = dyn_cast<SetFlagOp>(operation)) {
    return applySet(program, target, set, state);
  }
  if (auto wait = dyn_cast<WaitFlagOp>(operation)) {
    return applyWait(program, target, wait, state);
  }
  if (auto set = dyn_cast<SyncSetOp>(operation)) {
    return applyCrossCoreSet(program, target, set, state);
  }
  if (auto wait = dyn_cast<SyncWaitOp>(operation)) {
    return applyCrossCoreWait(program, target, wait, state);
  }
  if (auto barrier = dyn_cast<BarrierOp>(operation)) {
    return applyBarrier(program, target, barrier, state);
  }
  if (auto cmo = dyn_cast<CmoCacheInvalidOp>(operation)) {
    applyCmo(program, cmo, state);
    return success();
  }
  if (auto fence = dyn_cast<FenceBarrierAllOp>(operation)) {
    return applyFenceAll(program, target, fence, state);
  }
  handled = false;
  return success();
}
