// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Recover arithmetic primitives from a checked subset of original MLIR.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICPROGRAM_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICPROGRAM_H
#include "PTO/Transforms/FrontierSynch/ArithmeticRecognition.h"
namespace mlir::pto::frontiersynch {
struct ArithmeticGuard {
    scf::IfOp branch;
    bool takeThen = true;
};
struct ArithmeticSite {
    const CompoundInstanceElement* phase = nullptr;
    SmallVector<scf::ForOp> loops; // Outer to inner, one coordinate per loop.
    SmallVector<ArithmeticGuard> guards; // Enclosing branch arms, outer to inner.
};
struct ArithmeticProgram {
    RecognitionResult extraction;
    ArithmeticRecognition recognition;
    ArithmeticPrimitives primitives;
    SmallVector<ArithmeticSite> sites;
    SmallVector<Value> parameters; // Matches primitives.parameters.
};
// No unfolding: counted nests with constant nonnegative lower bounds and
// positive steps dividing the configured residue period. Upper bounds use
// proved affine expressions. Exact shared access maps retain symbolic origins
// and finite within-origin byte unions. GM regions retain canonical function
// entry pointer identities. Distinct GM bases require the shared MayNotAlias
// policy; one base works under either policy. Unresolved bases and mixtures of
// absolute and based GM addresses are rejected. Local allocation SSA roots
// never distinguish physical bytes. Carried scalar state requires an exact
// shared recurrence; loop-result uses in domains/accesses remain unsupported.
// Branch domains admit signed index comparisons and supported Boolean formulas;
// their exact finite unions are charged to the output size. Metadata follows
// the shared leaf contract. No synchronization or additional prerequisites are
// accepted by this producer.
// The shared producer must supply all payload effects. Output borrows input/IR.
// Failure clears primitive/site/parameter exports; diagnostics remain available.
ArithmeticProgram recognizeArithmeticProgram(func::FuncOp function, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits);
} // namespace mlir::pto::frontiersynch
#endif
