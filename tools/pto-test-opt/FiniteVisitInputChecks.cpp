// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
int runFiniteVisitInputChecks(func::FuncOp function)
{
    fs::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize()) || failed(analysis.analyzeFiniteVisitCandidates())) {
        llvm::errs() << "finite visit original-input analysis failed\n"; return 1;
    }
    const auto& results = analysis.finiteVisitDemands();
    if (results.size() != 1) { llvm::errs() << "expected one original finite visit loop\n"; return 1; }
    const auto& result = *results.begin()->second;
    const auto& contract = result.recognition.contract;
    if (!result.demands || !result.demands->error.empty()) {
        llvm::errs() << "finite visit demands: " <<
            (result.demands ? result.demands->error : result.recognition.error) << "\n"; return 1;
    }
    if (result.children.size() != 2 || result.demands->boundaries.size() != 4 ||
        result.demands->cost.pairReductions != 4 || contract.membership != fs::ContractStatus::Established ||
        contract.demands != fs::ContractImplementation::Available ||
        contract.endpointRecipes != fs::ContractImplementation::Unavailable ||
        contract.allocation != fs::ContractImplementation::NotRequested) {
        llvm::errs() << "finite visit stage contract lost\n"; return 1;
    }
    // Distinct visit choices must retain nonempty cross-arm storage demands.
    for (const auto& boundary : result.demands->boundaries) {
        if (boundary.demands.empty()) { llvm::errs() << "missing finite visit boundary demands\n"; return 1; }
    }
    const auto* before = results.begin()->second.get();
    if (failed(analysis.analyzeFiniteVisitCandidates()) ||
        analysis.finiteVisitDemands().begin()->second.get() != before) {
        llvm::errs() << "finite visit child/pair analysis was not cached\n"; return 1;
    }
    bool audited = false;
    for (const auto& candidate : analysis.result()->contractAudit) {
        if (candidate.kind == fs::ContractClass::FiniteVisitTypes &&
            candidate.membership == fs::ContractStatus::Established &&
            candidate.demands == fs::ContractImplementation::Available) { audited = true; }
    }
    if (!audited) { llvm::errs() << "finite visit success missing from contract audit\n"; return 1; }
    llvm::outs() << "finite visit original input: two types, four cached boundaries, demand-only exports\n";
    return 0;
}
