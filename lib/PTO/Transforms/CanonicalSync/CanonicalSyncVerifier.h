// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCVERIFIER_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCVERIFIER_H

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace pto {
namespace canonical_sync_detail {

struct VerifierEffectKey {
  Operation *operation = nullptr;
  unsigned phase = 0;
  unsigned access = 0;

  bool operator==(const VerifierEffectKey &other) const {
    return operation == other.operation && phase == other.phase &&
           access == other.access;
  }
};

struct VerifierEffect {
  VerifierEffectKey key;
  CanonicalPhysicalResource resource;
  CanonicalAccess access;
};

struct VerifierCompletion {
  VerifierEffectKey key;
  CanonicalPhysicalResource resource;
};

struct VerifierPhase {
  CanonicalPhysicalResource resource;
  VerifierCompletion completion;
  llvm::SmallVector<VerifierCompletion, 4> ssaSources;
  llvm::SmallVector<VerifierEffect, 4> effects;
};

struct VerifierCacheAction {
  Operation *operation = nullptr;
  CanonicalCore core = CanonicalCore::AIV;
  bool allGm = false;
  CanonicalAccess access;
};

struct VerifierProgram {
  func::FuncOp function;
  llvm::DenseMap<Operation *, llvm::SmallVector<VerifierPhase, 2>> phases;
  llvm::DenseMap<Operation *, llvm::SmallVector<VerifierCacheAction, 2>>
      cacheActions;
  llvm::DenseSet<Operation *> normalizedOperations;
};

FailureOr<std::unique_ptr<VerifierProgram>>
buildVerifierProgram(func::FuncOp function);

LogicalResult verifyConcreteEventGenerations(func::FuncOp function,
                                             const CanonicalSyncTarget &target);

struct VerifierResourceState {
  CanonicalPhysicalResource resource;
  llvm::SmallVector<VerifierEffect, 8> pending;
  llvm::SmallVector<VerifierEffectKey, 8> pendingCompletions;
  llvm::SmallVector<VerifierEffectKey, 8> known;
  llvm::SmallVector<VerifierEffectKey, 8> visible;
};

struct VerifierToken {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  unsigned eventId = 0;
  llvm::SmallVector<VerifierEffectKey, 8> payload;
};

struct VerifierState {
  llvm::SmallVector<VerifierResourceState, 8> resources;
  llvm::SmallVector<VerifierToken, 8> tokens;
  llvm::SmallVector<VerifierEffectKey, 16> globalKnown;
  llvm::SmallVector<VerifierEffectKey, 16> globalVisible;
  llvm::SmallVector<VerifierEffectKey, 16> cacheMaintained;
  llvm::SmallVector<VerifierCacheAction, 8> cacheInvalidations;
  llvm::SmallVector<VerifierEffectKey, 16> exitComplete;
  llvm::SmallVector<VerifierEffectKey, 16> loopCarried;
};

bool verifierCacheActionCovers(const VerifierCacheAction &action,
                               CanonicalCore core,
                               const CanonicalAccess &access);

bool operator==(const VerifierState &first, const VerifierState &second);
VerifierState mergeVerifierStates(const VerifierState &first,
                                  const VerifierState &second);
LogicalResult applyVerifierSyncOperation(const VerifierProgram &program,
                                         const CanonicalSyncTarget &target,
                                         Operation *operation,
                                         VerifierState &state, bool &handled);
LogicalResult verifyAndIssuePhase(const VerifierProgram &program,
                                  const CanonicalSyncTarget &target,
                                  const VerifierPhase &phase,
                                  VerifierState &state);
LogicalResult executeVerifierBlock(const VerifierProgram &program,
                                   const CanonicalSyncTarget &target,
                                   Block &block, VerifierState &state);

} // namespace canonical_sync_detail
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCVERIFIER_H
