// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Syntactic route recognition on original IR; no backend or demand generation.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_RECOGNITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_RECOGNITION_H
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::pto::frontiersynch {
enum class RecognitionState { Applicable, NotApplicable, MissingPremise };
enum class RecognitionIssue {
    StructuredBody, MultiplePhases, UnmodeledOperation, UnknownPipe,
    InexactFootprint, UnknownGeometry, LoopDomain, LoopCarriedState,
    SlotExpression, IndexArithmetic, CommonStride, OverlappingFamilies,
    AliasedOperand, UnsupportedView, GuardInvariance, UnsupportedControl,
    ArithmeticDimension, ArithmeticPeriod, ArithmeticPipeLimit, ArithmeticConfiguration, AdditionalPrerequisite
};
struct RecognitionDiagnostic {
    RecognitionIssue issue;
    Operation* anchor = nullptr;
};
struct RotatingAccess {
    std::size_t effect = 0;
    Value family;
    uint64_t slots = 1;
    uint64_t stride = 0;
    uint64_t offset = 0;
    uint64_t refresh = 1;
    // Exact within-slot scalar bytes, when proved. Empty for unresolved effects.
    std::optional<std::pair<uint64_t, uint64_t>> atom;
    std::optional<std::size_t> guard;
};
struct RecognitionResult {
    RecognitionState state = RecognitionState::Applicable;
    SmallVector<RecognitionDiagnostic> diagnostics;
    SmallVector<RotatingAccess> accesses;
    void note(RecognitionIssue issue, Operation* anchor, bool outsideClass = false);
};

// Guard nodes are shared conjunctions: parent AND (condition == takeThen).
// A missing parent denotes true. Conditions need evaluation only on their path.
struct GuardNode {
    std::optional<std::size_t> parent;
    Value condition;
    bool takeThen = true;
};
struct GuardedPhase {
    const CompoundInstanceElement* phase = nullptr;
    std::optional<std::size_t> guard;
};
struct GuardedRecognition {
    RecognitionResult result;
    SmallVector<GuardNode> guards;
    SmallVector<GuardedPhase> phases;
    bool entryGuardsAvailable = true;
};

// Index and effects must come from the same unchanged SyncInput and function.
// Results borrow unchanged input/IR. Applicable establishes this recognizer's
// input contract, relative to completeness of the supplied effects. It is not
// Section 8 Ready: no demands, endpoint code, selectors or queries are produced.
RecognitionResult recognizeExplicit(Block& block, const PhaseIndex& index,
                                    const SyncStorageEffects& effects);
RecognitionResult recognizeRotating(scf::ForOp loop, const PhaseIndex& index,
                                    const SyncInput& input, const SyncStorageEffects& effects);
GuardedRecognition recognizeFiniteGuarded(Region& region, const PhaseIndex& index,
                                          const SyncStorageEffects& effects);
GuardedRecognition recognizeGuardedRotating(scf::ForOp loop, const PhaseIndex& index,
                                            const SyncInput& input, const SyncStorageEffects& effects);
StringRef recognitionName(RecognitionState state);
StringRef recognitionName(RecognitionIssue issue);
} // namespace mlir::pto::frontiersynch
#endif
