// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_RETAINED_CONTROL_SELECTION_CHECKS_H
#define PTO_RETAINED_CONTROL_SELECTION_CHECKS_H
namespace mlir::pto::test {
inline LogicalResult checkRetainedControlSelection(func::FuncOp function, GMAliasPolicy policy)
{
    using namespace frontiersynch;
    FrontierAnalysis session(function);
    session.enableStageProfiling();
    if (failed(session.initialize(policy))) { return failure(); }
    auto loop = *function.getOps<scf::ForOp>().begin();
    const auto& nodes = session.result()->nodes;
    auto node = llvm::find_if(nodes, [&](const auto& entry) { return entry.anchor == loop; });
    if (node == nodes.end()) { return failure(); }
    AnalysisRequest request;
    request.region = std::distance(nodes.begin(), node);
    request.needs.synchronization = true;
    const auto incumbent = session.analyze(request);
    if (incumbent.status != AnalysisStatus::Ready) { return failure(); }
    // A stronger request can reuse either completed owner. The original exact
    // request keeps its established selection; no failed-candidate walk repeats.
    request.needs.queries = request.needs.selectors = true;
    auto arithmetic = session.analyzeArithmeticRegional(request);
    if (arithmetic.status != AnalysisStatus::Ready) {
        return function.emitError("retained selection fixture has no qualified arithmetic preparation");
    }
    const auto answer = session.analyze(request);
    if (answer.status != AnalysisStatus::Ready) { return failure(); }
    bool compared = false, cheaper = false;
    for (const auto& stage : answer.ledger->stages) {
        if (stage.stage != "retained-control-comparison") { continue; }
        uint64_t known = 0, incumbent = 0, candidate = 0;
        for (const auto& [key, count] : stage.work) {
            if (key == "known") { known = count; }
            if (key == "incumbent") { incumbent = count; }
            if (key == "candidate") { candidate = count; }
        }
        compared |= known != 0;
        cheaper |= known && candidate < incumbent;
        llvm::errs() << "retained selection cost " << incumbent << " -> " << candidate << " known=" << known << "\n";
    }
    if (!compared || !cheaper || answer.mathematical != arithmetic.mathematical) {
        llvm::errs() << "incumbent=" << incumbent.mathematical->backend << " answer="
            << answer.mathematical->backend << " arithmetic=" << arithmetic.mathematical->backend << "\n";
        for (auto outcome : {incumbent, arithmetic}) {
            auto p = session.prepareLogical(outcome);
            if (failed(p)) { continue; }
            const auto cost = logicalFamilyConstructionOperations(**p);
            llvm::errs() << "candidate " << outcome.mathematical->backend << " guarantee="
                << static_cast<int>(describeResultGuarantees(**p).realization) << " cost="
                << cost.value_or(UINT64_MAX) << "\n";
        }
        return function.emitError("retained selection did not choose the cheaper qualified arithmetic recipe");
    }
    const auto stages = answer.ledger->stages.size();
    const auto repeat = session.analyze(request);
    if (repeat.mathematical != answer.mathematical || repeat.ledger->stages.size() != stages + 1 ||
        repeat.ledger->stages.back().cacheHits != 1) {
        return function.emitError("retained selection repeated construction on identical request");
    }
    auto prepared = session.prepareLogical(answer);
    if (failed(prepared) || describeResultGuarantees(**prepared).realization != RealizationOrder::Equivalent ||
        failed(materializeLogicalFamilies(**prepared))) { return failure(); }
    llvm::outs() << "retained-control-selection: cheaper-qualified-owner cached-request\n";
    return success();
}
} // namespace mlir::pto::test
#endif
