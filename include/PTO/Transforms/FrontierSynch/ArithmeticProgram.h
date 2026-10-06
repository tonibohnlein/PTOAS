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
    SmallVector<scf::ForOp> loops; // Root-relative, outer to inner.
    SmallVector<ArithmeticGuard> guards; // Enclosing branch arms, outer to inner.
};
// One execution of root, conditional on reaching its entry. Loops/branches
// enclosing root belong to the parent; their values are shared parameters,
// never independent coordinates of the two endpoint occurrences. Repeated
// visits require parent substitution/certification. Parameter relations are
// pointwise exact at actual entry bindings; arbitrary assignments to distinct
// SSA parameters need not be jointly realizable.
struct ArithmeticRegionContext {
    func::FuncOp function;
    Operation* root = nullptr;
};
struct ArithmeticProgram {
    ArithmeticRegionContext context;
    // Incoming scalar prerequisites remain obligations of the composing parent.
    SmallVector<ValuePrerequisite> incomingPrerequisites;
    RecognitionResult extraction;
    ArithmeticRecognition recognition;
    ArithmeticPrimitives primitives;
    SmallVector<ArithmeticSite> sites;
    SmallVector<std::pair<uint32_t, uint32_t>> uniformConflicts;
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
// shared recurrence; unnormalized local loop-result uses remain unsupported.
// Branch domains admit signed index comparisons and supported Boolean formulas;
// their exact finite unions are charged to the output size. Metadata follows
// the shared leaf contract. Classified internal scalar prerequisites are retained;
// existing synchronization and unclassified prerequisites remain unsupported.
// The shared producer must supply all payload effects. Output borrows input/IR.
// Failure clears primitive/site/parameter exports; diagnostics remain available.
// Region extraction preserves original SSA bindings. Only index/i1 values
// available before root can become extra parameters; local unsupported values
// remain unsupported. This produces primitives, not a composable regional plan:
// storage selectors, query adapters and boundary discharge are separate steps.
ArithmeticProgram recognizeArithmeticProgram(ArithmeticRegionContext context, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits);
ArithmeticProgram recognizeArithmeticProgram(func::FuncOp function, const PhaseIndex& index,
                                             const SyncInput& input, const SyncStorageEffects& effects,
                                             const ArithmeticLimits& limits);
} // namespace mlir::pto::frontiersynch
#endif
