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

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

Block *findCommonBlock(Operation *first, Operation *second) {
  SmallVector<Block *, 8> firstBlocks;
  for (Operation *current = first; current; current = current->getParentOp()) {
    if (current->getBlock()) {
      firstBlocks.push_back(current->getBlock());
    }
  }
  for (Operation *current = second; current; current = current->getParentOp()) {
    Block *block = current->getBlock();
    if (block && llvm::is_contained(firstBlocks, block)) {
      return block;
    }
  }
  return nullptr;
}

Operation *liftToBlock(Operation *operation, Block *block) {
  while (operation) {
    const bool reachedBlock = operation->getBlock() == block;
    if (reachedBlock) {
      break;
    }
    operation = operation->getParentOp();
  }
  return operation;
}

} // namespace

bool mlir::pto::canonical_sync_detail::programPointMustPrecede(
    CanonicalProgramPoint first, CanonicalProgramPoint second) {
  if (!first.operation || !second.operation) {
    return false;
  }
  if (first.operation == second.operation) {
    return first.position == CanonicalProgramPointPosition::Before ||
           second.position == CanonicalProgramPointPosition::After;
  }
  if (second.operation->isProperAncestor(first.operation)) {
    return second.position == CanonicalProgramPointPosition::After;
  }
  if (first.operation->isProperAncestor(second.operation)) {
    return first.position == CanonicalProgramPointPosition::Before;
  }
  Block *block = findCommonBlock(first.operation, second.operation);
  Operation *firstAnchor = liftToBlock(first.operation, block);
  Operation *secondAnchor = liftToBlock(second.operation, block);
  return block && firstAnchor && secondAnchor && firstAnchor != secondAnchor &&
         firstAnchor->isBeforeInBlock(secondAnchor);
}

bool mlir::pto::canonical_sync_detail::phaseMayPrecedePoint(
    const CanonicalPhase &phase, CanonicalProgramPoint point) {
  return programPointMustPrecede(
      {phase.operation, CanonicalProgramPointPosition::After}, point);
}

bool mlir::pto::canonical_sync_detail::pointMustPrecedePhase(
    CanonicalProgramPoint point, const CanonicalPhase &phase) {
  return programPointMustPrecede(
      point, {phase.operation, CanonicalProgramPointPosition::Before});
}
