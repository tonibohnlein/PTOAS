// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Original finite alternatives, separate from arbitrary-word query/endpoint exports.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEVISITRECOGNITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEVISITRECOGNITION_H
#include "PTO/Transforms/FrontierSynch/FiniteVisitDemands.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
namespace mlir::pto::frontiersynch {
struct FiniteVisitAlternative {
    std::size_t node = 0;
    std::vector<ArithmeticGuard> selection; // Original decision path; not a shared cross-visit predicate.
};
struct FiniteVisitRecognition {
    scf::ForOp loop;
    std::vector<FiniteVisitAlternative> alternatives;
    ProgramContractCandidate contract;
    std::string error;
};
struct FiniteVisitAnalysis {
    FiniteVisitRecognition recognition;
    std::vector<SequenceAnalysis> children;
    std::optional<FiniteVisitDemands> demands;
    // There is deliberately no whole-loop RegionalAnalysis, prepare callback,
    // storage/rank summary or allocation certificate for an arbitrary type word.
};
// Read-only structural/invariance qualification. Class membership remains
// unproved until exact common refresh/native-presence obligations are checked.
// One exhaustive original decision tree selects the type; its predicates may
// vary by visit. Every selected interior is invariant relative to that visit.
FiniteVisitRecognition recognizeFiniteVisitLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, const PhaseIndex& index);
// Analyzes each unmasked original arm once, then the h^2 ordered pair reductions.
// No runtime visit sequence is enumerated. Borrowed input/IR/program must remain
// unchanged and outlive the result, as for the underlying regional analyses.
FiniteVisitAnalysis analyzeFiniteVisitLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node);
} // namespace mlir::pto::frontiersynch
#endif
