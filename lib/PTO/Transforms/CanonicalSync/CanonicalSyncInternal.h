// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncTarget.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"

#include <cstddef>

namespace mlir {
namespace pto {
namespace canonical_sync_detail {

struct AliasFact {
  Value root;
  AddressSpace space = AddressSpace::Zero;
  bool unknownSpace = false;
  llvm::SmallVector<CanonicalByteInterval, 2> intervals;
  bool physical = false;
  bool unknownRange = true;
  Value slotExpression;
};

class AliasAnalysis {
public:
  explicit AliasAnalysis(func::FuncOp function);

  LogicalResult solve();
  LogicalResult observe(Operation *operation);
  llvm::ArrayRef<AliasFact> lookup(Value value) const;
  llvm::SmallVector<AliasFact, 2> describe(Value value) const;
  void bind(Value value, llvm::ArrayRef<AliasFact> facts);

private:
  func::FuncOp function;
  llvm::DenseMap<Value, llvm::SmallVector<AliasFact, 2>> facts;
  std::size_t revision = 0;

  void initializeArguments();
  void bindAlias(Value result, Value source);
  void bindLoopCarried(Value value, llvm::ArrayRef<AliasFact> aliases);
  LogicalResult bindAllocation(Operation *operation);
};

bool areMemoryLikeTypes(Type type);
bool accessesMayAlias(const CanonicalAccess &first,
                      const CanonicalAccess &second,
                      CanonicalGmAliasPolicy gmAliasPolicy);
bool accessReads(CanonicalAccessMode mode);
bool accessWrites(CanonicalAccessMode mode);
bool controlsCanCoexecute(llvm::ArrayRef<CanonicalControlAtom> first,
                          llvm::ArrayRef<CanonicalControlAtom> second);
llvm::SmallVector<CanonicalControlAtom, 2>
conjoinCompatibleControlPaths(llvm::ArrayRef<CanonicalControlAtom> first,
                              llvm::ArrayRef<CanonicalControlAtom> second);
CanonicalRegionId findRegionLca(const CanonicalSyncProgram &program,
                                CanonicalRegionId first,
                                CanonicalRegionId second);
bool programPointMustPrecede(CanonicalProgramPoint first,
                             CanonicalProgramPoint second);
bool phaseMayPrecedePoint(const CanonicalPhase &phase,
                          CanonicalProgramPoint point);
bool pointMustPrecedePhase(CanonicalProgramPoint point,
                           const CanonicalPhase &phase);
FailureOr<CanonicalPhysicalResource>
resolvePhysicalResource(func::FuncOp function, Operation *operation, PIPE pipe);
LogicalResult rejectUnsupportedCanonicalSyncInput(func::FuncOp function);
llvm::SmallVector<unsigned, 6>
reservedEventIds(func::FuncOp function, CanonicalPhysicalResource source,
                 CanonicalPhysicalResource target);
LogicalResult
buildCanonicalStructureAndAccesses(CanonicalSyncProgram &program,
                                   const CanonicalSyncTarget &target);
LogicalResult deriveCanonicalDemands(CanonicalSyncProgram &program,
                                     const CanonicalSyncTarget &target);
struct CanonicalUnrolledCoverageResult {
  llvm::SmallVector<CanonicalDemandId, 16> covered;
  bool exhaustive = true;
};

FailureOr<CanonicalUnrolledCoverageResult> evaluateCanonicalSyncUnrolledOracle(
    const CanonicalSyncProgram &program,
    llvm::ArrayRef<CanonicalMechanismId> selected);
FailureOr<CanonicalCoverageWorld>
evaluateCanonicalSyncGroup(const CanonicalSyncProgram &program,
                           llvm::StringRef name,
                           llvm::ArrayRef<CanonicalMechanismId> selected);
LogicalResult verifyMaterializedCanonicalSync(
    func::FuncOp function, CanonicalGmAliasPolicy gmAliasPolicy);

} // namespace canonical_sync_detail
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCINTERNAL_H
