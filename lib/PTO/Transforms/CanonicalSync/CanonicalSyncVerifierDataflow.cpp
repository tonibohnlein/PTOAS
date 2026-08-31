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

const VerifierResourceState *getResource(const VerifierState &state,
                                         CanonicalPhysicalResource resource) {
  auto iterator =
      llvm::find_if(state.resources, [&](const VerifierResourceState &entry) {
        return entry.resource == resource;
      });
  return iterator == state.resources.end() ? nullptr : &*iterator;
}

bool containsKey(ArrayRef<VerifierEffectKey> keys,
                 const VerifierEffectKey &key) {
  return llvm::is_contained(keys, key);
}

void eraseKey(SmallVectorImpl<VerifierEffectKey> &keys,
              const VerifierEffectKey &key) {
  llvm::erase_if(keys, [&](const VerifierEffectKey &candidate) {
    return candidate == key;
  });
}

void invalidateCompletion(VerifierState &state, const VerifierEffectKey &key) {
  eraseKey(state.globalKnown, key);
  eraseKey(state.globalVisible, key);
  eraseKey(state.cacheMaintained, key);
  eraseKey(state.exitComplete, key);
  for (VerifierResourceState &resource : state.resources) {
    eraseKey(resource.known, key);
    eraseKey(resource.visible, key);
  }
}

bool isAccReadConflict(const VerifierEffect &source,
                       const VerifierEffect &target) {
  const bool sourceMayBeAcc =
      source.access.unknownSpace || source.access.space == AddressSpace::ACC;
  const bool targetMayBeAcc =
      target.access.unknownSpace || target.access.space == AddressSpace::ACC;
  return accessReads(source.access.mode) && accessReads(target.access.mode) &&
         sourceMayBeAcc && targetMayBeAcc &&
         source.resource.core == CanonicalCore::AIC &&
         target.resource.core == CanonicalCore::AIC &&
         source.resource.pipe != target.resource.pipe;
}

bool isHazard(const VerifierEffect &source, const VerifierEffect &target) {
  const bool sourceRead = accessReads(source.access.mode);
  const bool sourceWrite = accessWrites(source.access.mode);
  const bool targetRead = accessReads(target.access.mode);
  const bool targetWrite = accessWrites(target.access.mode);
  return source.access.ordered || target.access.ordered ||
         (sourceWrite && targetRead) || (sourceRead && targetWrite) ||
         (sourceWrite && targetWrite) || isAccReadConflict(source, target);
}

bool requiresVisibility(const VerifierEffect &source,
                        const VerifierEffect &target) {
  const bool sourceMayBeGm =
      source.access.unknownSpace || source.access.space == AddressSpace::GM;
  const bool targetMayBeGm =
      target.access.unknownSpace || target.access.space == AddressSpace::GM;
  return sourceMayBeGm && targetMayBeGm &&
         ((source.resource.pipe == PIPE::PIPE_S) !=
          (target.resource.pipe == PIPE::PIPE_S));
}

bool cacheActionCovers(const VerifierCacheAction &action,
                       const CanonicalAccess &access) {
  if (action.allGm) {
    return access.unknownSpace || access.space == AddressSpace::GM;
  }
  const bool exactAccessStart =
      access.addressByteOffset && *access.addressByteOffset == 0 &&
      access.addressByteSize && *access.addressByteSize > 0;
  return action.access.value == access.value && exactAccessStart;
}

bool completionKnown(const CanonicalSyncTarget &target,
                     CanonicalPhysicalResource sourceResource,
                     const VerifierEffectKey &source,
                     CanonicalPhysicalResource destinationResource,
                     const VerifierState &state) {
  if (sourceResource == destinationResource &&
      target.hasIntrinsicCompletion(sourceResource)) {
    return true;
  }
  if (containsKey(state.globalKnown, source)) {
    return true;
  }
  const VerifierResourceState *resource =
      getResource(state, destinationResource);
  return resource && containsKey(resource->known, source);
}

bool completionKnown(const CanonicalSyncTarget &target,
                     const VerifierEffect &source,
                     const VerifierEffect &destination,
                     const VerifierState &state) {
  return completionKnown(target, source.resource, source.key,
                         destination.resource, state);
}

bool visibilityKnown(const VerifierEffect &source,
                     const VerifierEffect &destination,
                     const VerifierState &state) {
  if (!containsKey(state.globalVisible, source.key)) {
    return false;
  }
  const bool scalarRead = destination.resource.pipe == PIPE::PIPE_S &&
                          accessReads(destination.access.mode);
  return !scalarRead ||
         llvm::any_of(state.cacheInvalidations,
                      [&](const VerifierCacheAction &action) {
                        return cacheActionCovers(action, destination.access);
                      });
}

LogicalResult reportHazard(const VerifierEffect &source,
                           const VerifierEffect &target, bool visibility) {
  InFlightDiagnostic diagnostic = target.key.operation->emitError(
      visibility
          ? "canonical sync verifier found an unsatisfied visibility hazard"
          : "canonical sync verifier found an unsatisfied completion hazard");
  diagnostic << " from " << stringifyCanonicalCore(source.resource.core) << ':'
             << stringifyPIPE(source.resource.pipe) << " to "
             << stringifyCanonicalCore(target.resource.core) << ':'
             << stringifyPIPE(target.resource.pipe);
  diagnostic.attachNote(source.key.operation->getLoc())
      << "conflicting source effect is here";
  return failure();
}

bool containsEffect(ArrayRef<VerifierEffect> effects,
                    const VerifierEffectKey &key) {
  return llvm::any_of(
      effects, [&](const VerifierEffect &effect) { return effect.key == key; });
}

bool tokensEqual(ArrayRef<VerifierToken> first,
                 ArrayRef<VerifierToken> second) {
  const bool differentSizes = first.size() != second.size();
  if (differentSizes) {
    return false;
  }
  return llvm::all_of(first, [&](const VerifierToken &token) {
    return llvm::any_of(second, [&](const VerifierToken &other) {
      return token.source == other.source && token.target == other.target &&
             token.eventId == other.eventId && token.payload == other.payload;
    });
  });
}

void collectLoopKeys(const VerifierProgram &program, Region &region,
                     SmallVectorImpl<VerifierEffectKey> &keys) {
  region.walk([&](Operation *operation) {
    auto phases = program.phases.find(operation);
    if (phases == program.phases.end()) {
      return;
    }
    for (const VerifierPhase &phase : phases->second) {
      if (!containsKey(keys, phase.completion.key)) {
        keys.push_back(phase.completion.key);
      }
      for (const VerifierEffect &effect : phase.effects) {
        if (!containsKey(keys, effect.key)) {
          keys.push_back(effect.key);
        }
      }
    }
  });
}

LogicalResult executeIf(const VerifierProgram &program,
                        const CanonicalSyncTarget &target, scf::IfOp operation,
                        VerifierState &state) {
  VerifierState thenState = state;
  if (failed(executeVerifierBlock(
          program, target, operation.getThenRegion().front(), thenState))) {
    return failure();
  }
  VerifierState elseState = state;
  const bool hasElse = !operation.getElseRegion().empty();
  if (hasElse &&
      failed(executeVerifierBlock(
          program, target, operation.getElseRegion().front(), elseState))) {
    return failure();
  }
  if (!tokensEqual(thenState.tokens, elseState.tokens)) {
    return operation.emitError("canonical sync verifier found an event "
                               "lifecycle crossing incompatible branches");
  }
  state = mergeVerifierStates(thenState, elseState);
  return success();
}

LogicalResult executeLoop(const VerifierProgram &program,
                          const CanonicalSyncTarget &target,
                          scf::ForOp operation, VerifierState &state) {
  const VerifierState entry = state;
  VerifierState iteration = state;
  SmallVector<VerifierEffectKey, 16> loopKeys;
  collectLoopKeys(program, operation.getRegion(), loopKeys);
  const unsigned limit = static_cast<unsigned>(loopKeys.size() * 4 + 16);
  bool converged = false;
  for (unsigned count = 0; count < limit; ++count) {
    const VerifierState before = iteration;
    if (failed(executeVerifierBlock(
            program, target, operation.getRegion().front(), iteration))) {
      return failure();
    }
    if (!tokensEqual(before.tokens, iteration.tokens)) {
      return operation.emitError("canonical sync verifier found an event "
                                 "lifecycle crossing a loop iteration");
    }
    if (iteration == before) {
      converged = true;
      break;
    }
    iteration.loopCarried.assign(loopKeys.begin(), loopKeys.end());
  }
  if (!converged) {
    return operation.emitError(
        "canonical sync verifier dataflow did not converge for this loop");
  }
  iteration.loopCarried = entry.loopCarried;
  state = mergeVerifierStates(entry, iteration);
  return success();
}

LogicalResult verifyReturn(func::ReturnOp operation,
                           const VerifierState &state) {
  if (!state.tokens.empty()) {
    return operation.emitError(
        "canonical sync verifier found an unconsumed event at function exit");
  }
  for (const VerifierResourceState &resource : state.resources) {
    for (const VerifierEffect &effect : resource.pending) {
      if (!containsKey(state.exitComplete, effect.key)) {
        return operation.emitError("canonical sync verifier found an undrained "
                                   "pipeline at function exit");
      }
    }
    for (const VerifierEffectKey &key : resource.pendingCompletions) {
      if (!containsKey(state.exitComplete, key)) {
        return operation.emitError("canonical sync verifier found an "
                                   "incomplete phase at function exit");
      }
    }
  }
  return success();
}

} // namespace

LogicalResult mlir::pto::canonical_sync_detail::verifyAndIssuePhase(
    const VerifierProgram &program, const CanonicalSyncTarget &target,
    const VerifierPhase &phase, VerifierState &state) {
  for (const VerifierCompletion &source : phase.ssaSources) {
    if (source.key.operation == phase.completion.key.operation) {
      continue;
    }
    if (!completionKnown(target, source.resource, source.key, phase.resource,
                         state)) {
      InFlightDiagnostic diagnostic = phase.completion.key.operation->emitError(
          "canonical sync verifier found an unsatisfied SSA completion");
      diagnostic.attachNote(source.key.operation->getLoc())
          << "SSA producer is here";
      return failure();
    }
  }
  for (const VerifierEffect &destination : phase.effects) {
    for (const VerifierResourceState &resource : state.resources) {
      for (const VerifierEffect &source : resource.pending) {
        const bool sameOperation =
            source.key.operation == destination.key.operation;
        const bool unresolvedSameOperation =
            sameOperation && !containsKey(state.loopCarried, source.key);
        const bool nonAliasing =
            !accessesMayAlias(source.access, destination.access);
        const bool nonHazard = !isHazard(source, destination);
        if (unresolvedSameOperation || nonAliasing || nonHazard) {
          continue;
        }
        const bool visibility = requiresVisibility(source, destination);
        const bool missingCompletion =
            !completionKnown(target, source, destination, state);
        if (missingCompletion ||
            (visibility && !visibilityKnown(source, destination, state))) {
          return reportHazard(source, destination, visibility);
        }
      }
    }
  }
  VerifierResourceState &resource = getResource(state, phase.resource);
  invalidateCompletion(state, phase.completion.key);
  if (!containsKey(resource.pendingCompletions, phase.completion.key)) {
    resource.pendingCompletions.push_back(phase.completion.key);
  }
  for (const VerifierEffect &effect : phase.effects) {
    invalidateCompletion(state, effect.key);
    if (accessWrites(effect.access.mode)) {
      llvm::erase_if(state.cacheInvalidations,
                     [&](const VerifierCacheAction &action) {
                       return cacheActionCovers(action, effect.access);
                     });
    }
    if (!containsEffect(resource.pending, effect.key)) {
      resource.pending.push_back(effect);
    }
  }
  return success();
}

LogicalResult mlir::pto::canonical_sync_detail::executeVerifierBlock(
    const VerifierProgram &program, const CanonicalSyncTarget &target,
    Block &block, VerifierState &state) {
  for (Operation &operation : block) {
    bool handled = false;
    if (failed(applyVerifierSyncOperation(program, target, &operation, state,
                                          handled))) {
      return failure();
    }
    if (handled) {
      continue;
    }
    if (auto ifOperation = dyn_cast<scf::IfOp>(operation)) {
      if (failed(executeIf(program, target, ifOperation, state))) {
        return failure();
      }
      continue;
    }
    if (auto forOperation = dyn_cast<scf::ForOp>(operation)) {
      if (failed(executeLoop(program, target, forOperation, state))) {
        return failure();
      }
      continue;
    }
    if (auto section = dyn_cast<SectionCubeOp>(operation)) {
      if (failed(executeVerifierBlock(program, target,
                                      section.getBody().front(), state))) {
        return failure();
      }
      continue;
    }
    if (auto section = dyn_cast<SectionVectorOp>(operation)) {
      if (failed(executeVerifierBlock(program, target,
                                      section.getBody().front(), state))) {
        return failure();
      }
      continue;
    }
    auto phases = program.phases.find(&operation);
    if (phases != program.phases.end()) {
      for (const VerifierPhase &phase : phases->second) {
        if (failed(verifyAndIssuePhase(program, target, phase, state))) {
          return failure();
        }
      }
    }
    if (auto returnOperation = dyn_cast<func::ReturnOp>(operation)) {
      if (failed(verifyReturn(returnOperation, state))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult mlir::pto::canonical_sync_detail::verifyMaterializedCanonicalSync(
    func::FuncOp function) {
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(function);
  FailureOr<std::unique_ptr<VerifierProgram>> program =
      buildVerifierProgram(function);
  const bool invalidInputs = failed(target) || failed(program);
  if (invalidInputs) {
    return failure();
  }
  VerifierState state;
  return executeVerifierBlock(**program, *target, function.getBody().front(),
                              state);
}
