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

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool lifetimesCanOverlap(const CanonicalMechanism &first,
                         const CanonicalMechanism &second) {
  // Scalar program order does not order the execution of different hardware
  // pipelines. In particular, a lexically earlier wait may not have consumed
  // its event before the source pipeline reaches a later set. Until a hardware
  // happens-before proof is available, only mutually exclusive generations may
  // share a physical event ID.
  return controlsCanCoexecute(first.guard, second.guard);
}

bool idAvailable(const CanonicalSyncProgram &program,
                 const CanonicalSetCoverSolution &solution,
                 const CanonicalMechanism &candidate,
                 CanonicalPhysicalResource source,
                 CanonicalPhysicalResource destination, unsigned eventId) {
  for (CanonicalMechanismId existingId : solution.mechanisms) {
    const CanonicalMechanism &existing = program.getMechanism(existingId);
    const bool readyCollision = existing.eventId == eventId &&
                                existing.source == source &&
                                existing.target == destination;
    const bool releaseCollision =
        existing.kind == CanonicalMechanismKind::RecurringEvent &&
        existing.releaseEventId == eventId && existing.target == source &&
        existing.source == destination;
    const bool collision = (readyCollision || releaseCollision) &&
                           lifetimesCanOverlap(existing, candidate);
    if (collision) {
      return false;
    }
  }
  return true;
}

FailureOr<unsigned> allocateEventId(CanonicalSyncProgram &program,
                                    const CanonicalSyncTarget &target,
                                    const CanonicalSetCoverSolution &solution,
                                    const CanonicalMechanism &mechanism,
                                    CanonicalPhysicalResource source,
                                    CanonicalPhysicalResource destination) {
  const SmallVector<unsigned, 6> reserved =
      reservedEventIds(program.getFunction(), source, destination);
  for (unsigned eventId : target.getCompilerEventIds()) {
    const bool available =
        !llvm::is_contained(reserved, eventId) &&
        idAvailable(program, solution, mechanism, source, destination, eventId);
    if (available) {
      return eventId;
    }
  }
  Operation *witness = mechanism.targetPoint.operation
                           ? mechanism.targetPoint.operation
                           : program.getFunction().getOperation();
  witness->emitError("canonical sync exhausted compiler event IDs after "
                     "applying hidden macro reservations")
      << "; mechanism m" << mechanism.id << " domain "
      << stringifyCanonicalCore(source.core) << ':'
      << stringifyPIPE(source.pipe) << " -> "
      << stringifyPIPE(destination.pipe);
  return failure();
}

bool crossCoreIdAvailable(const CanonicalSyncProgram &program,
                          const CanonicalSetCoverSolution &solution,
                          const CanonicalMechanism &candidate,
                          unsigned eventId) {
  return llvm::none_of(
      solution.mechanisms, [&](CanonicalMechanismId existingId) {
        const CanonicalMechanism &existing = program.getMechanism(existingId);
        return existing.kind == CanonicalMechanismKind::CrossCoreEvent &&
               existing.eventId == eventId &&
               lifetimesCanOverlap(existing, candidate);
      });
}

FailureOr<unsigned>
allocateCrossCoreEventId(CanonicalSyncProgram &program,
                         const CanonicalSyncTarget &target,
                         const CanonicalSetCoverSolution &solution,
                         const CanonicalMechanism &mechanism) {
  for (unsigned eventId : target.getCompilerCrossCoreEventIds()) {
    if (crossCoreIdAvailable(program, solution, mechanism, eventId)) {
      return eventId;
    }
  }
  Operation *witness = mechanism.targetPoint.operation
                           ? mechanism.targetPoint.operation
                           : program.getFunction().getOperation();
  witness->emitError("canonical sync exhausted cross-core counter IDs")
      << "; mechanism m" << mechanism.id << " domain "
      << stringifyCanonicalCore(mechanism.source.core) << ':'
      << stringifyPIPE(mechanism.source.pipe) << " -> "
      << stringifyCanonicalCore(mechanism.target.core) << ':'
      << stringifyPIPE(mechanism.target.pipe);
  return failure();
}

} // namespace

LogicalResult
mlir::pto::allocateCanonicalSyncEvents(CanonicalSyncProgram &program) {
  if (!program.getSetCoverSolution()) {
    return program.getFunction().emitError(
        "canonical sync event allocation requires a selected cover");
  }
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }
  const CanonicalSetCoverSolution &solution = *program.getSetCoverSolution();
  for (CanonicalMechanismId mechanismId : solution.mechanisms) {
    const CanonicalMechanism &mechanism = program.getMechanism(mechanismId);
    const bool event = mechanism.kind == CanonicalMechanismKind::Event;
    const bool crossCore =
        mechanism.kind == CanonicalMechanismKind::CrossCoreEvent;
    const bool recurring =
        mechanism.kind == CanonicalMechanismKind::RecurringEvent;
    if (!event && !crossCore && !recurring) {
      continue;
    }
    if (crossCore) {
      FailureOr<unsigned> eventId =
          allocateCrossCoreEventId(program, *target, solution, mechanism);
      if (failed(eventId)) {
        return failure();
      }
      program.setMechanismEventId(mechanism.id, *eventId);
      continue;
    }
    FailureOr<unsigned> ready =
        allocateEventId(program, *target, solution, mechanism, mechanism.source,
                        mechanism.target);
    if (failed(ready)) {
      return failure();
    }
    program.setMechanismEventId(mechanism.id, *ready);
    if (recurring) {
      FailureOr<unsigned> release =
          allocateEventId(program, *target, solution, mechanism,
                          mechanism.target, mechanism.source);
      if (failed(release)) {
        return failure();
      }
      program.setMechanismReleaseEventId(mechanism.id, *release);
    }
  }
  return success();
}
