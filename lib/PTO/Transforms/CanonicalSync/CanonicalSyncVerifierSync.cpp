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

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

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

LogicalResult applyBarrier(const VerifierProgram &program,
                           const CanonicalSyncTarget &target,
                           BarrierOp operation, VerifierState &state) {
  const PIPE pipe = operation.getPipe().getPipe();
  if (pipe == PIPE::PIPE_ALL) {
    for (VerifierResourceState &resource : state.resources) {
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

bool fenceDrainsResource(const VerifierProgram &program,
                         CanonicalPhysicalResource resource) {
  auto kind = program.function->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  const bool vectorKernel =
      kind && kind.getKernelKind() == FunctionKernelKind::Vector;
  if (vectorKernel) {
    return true;
  }
  return resource.pipe == PIPE::PIPE_MTE2 || resource.pipe == PIPE::PIPE_MTE3 ||
         resource.pipe == PIPE::PIPE_FIX;
}

void applyFenceAll(const VerifierProgram &program, FenceBarrierAllOp operation,
                   VerifierState &state) {
  const FenceScope scope = operation.getScope().getScope();
  if (scope != FenceScope::GM && scope != FenceScope::All) {
    return;
  }
  for (VerifierResourceState &resource : state.resources) {
    if (!fenceDrainsResource(program, resource.resource)) {
      continue;
    }
    completeResource(resource, state);
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
  if (auto barrier = dyn_cast<BarrierOp>(operation)) {
    return applyBarrier(program, target, barrier, state);
  }
  if (auto cmo = dyn_cast<CmoCacheInvalidOp>(operation)) {
    applyCmo(program, cmo, state);
    return success();
  }
  if (auto fence = dyn_cast<FenceBarrierAllOp>(operation)) {
    applyFenceAll(program, fence, state);
    return success();
  }
  handled = false;
  return success();
}
