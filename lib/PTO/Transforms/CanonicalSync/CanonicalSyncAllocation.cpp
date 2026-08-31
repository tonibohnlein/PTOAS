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

bool waitPrecedesSet(const CanonicalMechanism &waiter,
                     const CanonicalMechanism &setter) {
  Operation *wait = waiter.waitBefore;
  Operation *set = setter.setAfter;
  return wait && set && wait->getBlock() == set->getBlock() &&
         (wait == set || wait->isBeforeInBlock(set));
}

bool lifetimesCanOverlap(const CanonicalMechanism &first,
                         const CanonicalMechanism &second) {
  if (!controlsCanCoexecute(first.guard, second.guard)) {
    return false;
  }
  return !waitPrecedesSet(first, second) && !waitPrecedesSet(second, first);
}

bool idAvailable(const CanonicalSyncProgram &program,
                 const CanonicalMechanism &candidate, unsigned eventId) {
  for (const CanonicalMechanism &existing : program.getMechanisms()) {
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
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }
  for (const CanonicalMechanism &mechanism : program.getMechanisms()) {
    if (mechanism.kind != CanonicalMechanismKind::Event) {
      continue;
    }
    const SmallVector<unsigned, 6> reserved = reservedEventIds(
        program.getFunction(), mechanism.source, mechanism.target);
    std::optional<unsigned> assigned;
    for (unsigned eventId : target->getCompilerEventIds()) {
      const bool eventAvailable = !llvm::is_contained(reserved, eventId) &&
                                  idAvailable(program, mechanism, eventId);
      if (eventAvailable) {
        assigned = eventId;
        break;
      }
    }
    if (!assigned) {
      Operation *witness = mechanism.waitBefore
                               ? mechanism.waitBefore
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
