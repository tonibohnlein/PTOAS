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
    std::vector<std::size_t> nodes; // Ordered original nodes forming one complete visit.
    std::vector<ArithmeticGuard> selection; // Original decision path; not a shared cross-visit predicate.
};
struct FiniteVisitRecognition {
    scf::ForOp loop;
    std::vector<FiniteVisitAlternative> alternatives;
    // Explicit producer work, including intermediate choice-prefix descriptions.
    uint64_t typeDescriptions = 0, nodeReferences = 0;
    bool storageProjectionRequired = false;
    ProgramContractCandidate contract;
    std::string error;
};
struct FiniteVisitAnalysis {
    FiniteVisitRecognition recognition;
    // Each original atomic node is analyzed once; whole types reuse these owners.
    std::map<std::size_t, SequenceAnalysis> cachedNodes;
    std::vector<SequenceAnalysis> children;
    std::optional<FiniteVisitDemands> demands;
    // There is deliberately no whole-loop RegionalAnalysis, prepare callback,
    // storage/rank summary or allocation certificate for an arbitrary type word.
};
// Read-only structural/invariance qualification. Class membership remains
// unproved until exact common refresh/native-presence obligations are checked.
// Exhaustive original choices select whole visits, including common prefix,
// suffix and intervening nodes. Choice predicates may vary by visit; every
// selected interior is invariant. Explicit type expansion has a producer limit.
FiniteVisitRecognition recognizeFiniteVisitLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, const PhaseIndex& index);
// Analyzes each unmasked original node once, composes the whole-visit types,
// then constructs the h^2 ordered pair reductions.
// No runtime visit sequence is enumerated. Borrowed input/IR/program must remain
// unchanged and outlive the result, as for the underlying regional analyses.
FiniteVisitAnalysis analyzeFiniteVisitLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node);
} // namespace mlir::pto::frontiersynch
#endif
