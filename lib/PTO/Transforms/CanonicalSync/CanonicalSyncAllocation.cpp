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

bool sameDomain(const CanonicalMechanism &first,
                const CanonicalMechanism &second) {
  return first.source == second.source && first.target == second.target;
}

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
                 const CanonicalMechanism &candidate, unsigned eventId) {
  for (CanonicalMechanismId existingId : solution.mechanisms) {
    const CanonicalMechanism &existing = program.getMechanism(existingId);
    if (existing.id >= candidate.id ||
        existing.kind != CanonicalMechanismKind::Event ||
        existing.eventId != eventId || !sameDomain(existing, candidate)) {
      continue;
    }
    if (lifetimesCanOverlap(existing, candidate)) {
      return false;
    }
  }
  return true;
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
    if (mechanism.kind != CanonicalMechanismKind::Event) {
      continue;
    }
    const SmallVector<unsigned, 6> reserved = reservedEventIds(
        program.getFunction(), mechanism.source, mechanism.target);
    std::optional<unsigned> assigned;
    for (unsigned eventId : target->getCompilerEventIds()) {
      const bool eventAvailable =
          !llvm::is_contained(reserved, eventId) &&
          idAvailable(program, solution, mechanism, eventId);
      if (eventAvailable) {
        assigned = eventId;
        break;
      }
    }
    if (!assigned) {
      Operation *witness = mechanism.targetPoint.operation
                               ? mechanism.targetPoint.operation
                               : program.getFunction().getOperation();
      witness->emitError("canonical sync exhausted event IDs 0-5 after "
                         "applying hidden macro reservations")
          << "; mechanism m" << mechanism.id << " domain "
          << stringifyCanonicalCore(mechanism.source.core) << ':'
          << stringifyPIPE(mechanism.source.pipe) << " -> "
          << stringifyPIPE(mechanism.target.pipe);
      return failure();
    }
    program.setMechanismEventId(mechanism.id, *assigned);
  }
  return success();
}
