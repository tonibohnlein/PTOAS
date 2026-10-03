// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Dump shared F*, then atomically publish direct logical endpoint mechanisms.
#include "PTO/Transforms/FrontierSynch/FrontierSynch.h"
#include "DirectEmissionInternal.h"
#include "ExplicitPhysicalEmission.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::pto::frontiersynch {
namespace {
llvm::json::Array dumpSites(ArrayRef<StructuredSite> input)
{
    llvm::json::Array sites;
    DenseMap<Operation*, std::size_t> loops;
    for (auto [id, site] : llvm::enumerate(input)) {
        std::string location;
        llvm::raw_string_ostream os(location);
        site.anchor->getLoc().print(os);
        llvm::json::Array coordinates;
        for (Operation* loop : site.loops) {
            auto found = loops.try_emplace(loop, loops.size());
            coordinates.push_back("iteration" + std::to_string(found.first->second));
        }
        sites.push_back(
            llvm::json::Object{
                {"id", static_cast<int64_t>(id)},
                {"operation", site.phase->opName.getStringRef()},
                {"pipe", static_cast<int64_t>(site.phase->kPipeValue)},
                {"location", std::move(location)},
                {"coordinates", std::move(coordinates)},
                {"local_phase", static_cast<int64_t>(site.localPhase)}});
    }
    return sites;
}
LogicalResult dump(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& analysis,
    const DirectEmissionResult& emission,
    const llvm::json::Value& selectedReport)
{
    AsmState asmState(function);
    llvm::json::Array parameters, conflicts, pipes;
    for (Value argument : function.getArguments()) {
        if (isa<IntegerType, IndexType>(argument.getType())) {
            std::string text;
            llvm::raw_string_ostream os(text);
            argument.printAsOperand(os, asmState);
            parameters.push_back(std::move(text));
        }
    }
    for (PipelineType pipe : analysis.pipes()) {
        pipes.push_back(static_cast<int64_t>(pipe));
    }
    for (auto [consumer, incoming] : llvm::enumerate(analysis.conflicts())) {
        for (std::size_t source : incoming) {
            conflicts.push_back(llvm::json::Array{static_cast<int64_t>(source), static_cast<int64_t>(consumer)});
        }
    }
    llvm::json::Object report{
        {"function", function.getSymName()},
        {"algorithm", "frontier-synch"},
        {"requirements", "shared-modeled"},
        {"gm_alias_policy", input.memory().gmPolicy() == GMAliasPolicy::MayAlias ? "may-alias" : "may-not-alias"},
        {"representation", "selected-regional-analysis"},
        {"endpoint_types", "completion,start"},
        {"coordinate_semantics", "executed iteration ordinals"},
        {"sites", dumpSites(analysis.sites())},
        {"parameters", std::move(parameters)},
        {"pipes", std::move(pipes)},
        {"G_site_pairs", std::move(conflicts)}};
    report["selected_analysis"] = selectedReport;
    report["endpoints"] = llvm::json::Object{
        {"status", emission.emitted ? "emitted" : "unresolved"},
        {"route", emission.route},
        {"local_barrier_policy", "before-consumer"},
        {"ordering_guarantee",
         emission.emitted ? "coverage; order equality requires adjacent local demands" : "not established"},
        {"coordinate_semantics", "source induction values"},
        {"counts", "static internal operations; invocation epilogue excluded"},
        {"reason", emission.reason},
        {"sets", static_cast<int64_t>(emission.sets)},
        {"waits", static_cast<int64_t>(emission.waits)},
        {"barriers", static_cast<int64_t>(emission.barriers)}};
    bool physical = emission.pending && emission.pending.get()->hasAttr("pto.frontier.physical_status");
    bool finite = physical && emission.selected && emission.selected->kind == SelectedAnalysis::Kind::Explicit;
    report["physical_ids"] = physical ? (finite ? "assigned; finite directed pools under recorded target/ABI premises" :
                                                 "assigned; directed pools under recorded target/ABI premises") :
                             !emission.emitted ? "not attempted; endpoint realization unresolved" :
                             emission.sets     ? "unassigned; separate assignment stage is not implemented" :
                                                 "no cross-pipe handoffs";
    if (emission.selected && emission.selected->kind == SelectedAnalysis::Kind::Explicit) {
        llvm::json::Array pairs;
        for (auto id : emission.selected->explicitReduction.retained()) {
            const auto& edge = emission.selected->generators[id];
            pairs.push_back(
                llvm::json::Array{
                    static_cast<int64_t>(emission.selected->sites[edge.source]),
                    static_cast<int64_t>(emission.selected->sites[edge.consumer])});
        }
        report["F_star_pairs"] = std::move(pairs);
    }
    return printFrontierReport(function, llvm::json::Value(std::move(report)));
}
} // namespace
LogicalResult run(func::FuncOp function, const SyncInput& input, DiagnosticOptions options, CostLedger& costs)
{
    // Both algorithms consume the shared phase/effect/native model directly.
    // Backend applicability and generated-plan validation happen downstream.
    auto analysis = [&]() {
        CostScope effect(costs, CostStage::Effects);
        return TraceDemandAnalysis::build(function, input);
    }();
    if (failed(analysis)) {
        if (options.reportCosts) {
            reportImportFailure(
                function, costs, "shared source requirements could not be imported", options.dumpDemands);
        }
        return function.emitError("frontier-synch: could not import shared source requirements");
    }
    auto emission = emitDirectDemands(function, input, *analysis, costs);
    if (emission.emitted &&
        failed(emitExplicitPhysical(input, *analysis, emission, costs, options.finiteOneWayFamily))) {
        emission.emitted = false;
        emission.pending = nullptr;
    }
    if (!emission.emitted) {
        if (options.reportCosts) {
            auto report = costReport(function, input, *analysis, emission, options.dumpDemands);
            report["qualification"] = "unmet-realization";
            if (costs.stage(CostStage::Allocation).invocations) {
                report["logical_insertion"] = "prepared-logical; pending clone discarded";
            }
            finishCostReport(report, costs, emission);
            (void)printFrontierReport(function, llvm::json::Value(std::move(report)));
        }
        return function.emitError("frontier-synch: no certified physical plan: ") << emission.reason;
    }
    // All unsuccessful plans returned above; retain the accepted summaries once.
    Attribute retained;
    {
        CostScope evidence(costs, CostStage::Insertion);
        retained = retainSelectedAnalysis(function, *analysis, emission);
        NamedAttrList withTarget(cast<DictionaryAttr>(retained));
        withTarget.set("shared_target", input.target().attribute());
        withTarget.set("gm_alias_policy", StringAttr::get(function.getContext(),
            input.memory().gmPolicy() == GMAliasPolicy::MayAlias ? "may-alias" : "may-not-alias"));
        retained = withTarget.getDictionary(function.getContext());
    }
    if (options.dumpDemands) {
        CostScope diagnostics(costs, CostStage::Diagnostics);
        if (failed(dump(function, input, *analysis, emission, selectedAnalysisReport(emission, retained)))) {
            return failure();
        }
    }
    // Collect identities before source replacement invalidates borrowed anchors.
    llvm::json::Object report;
    if (options.reportCosts) {
        report = costReport(function, input, *analysis, emission, options.dumpDemands);
    }
    Region originalBody;
    auto originalAttrs = function->getAttrDictionary();
    {
        CostScope publication(costs, CostStage::Insertion);
        if (options.reportCosts) {
            originalBody.takeBody(function.getBody());
        }
        function.getBody().takeBody(emission.pending->getBody());
        function->setAttr("pto.frontier.source", emission.pending->getOperation()->getAttr("pto.frontier.source"));
        function->setAttr("pto.frontier.analysis", retained);
        function->setAttr("pto.frontier.endpoint_status", StringAttr::get(function.getContext(), "emitted"));
        function->setAttr("pto.frontier.physical_status", emission.pending->getOperation()->getAttr(
            "pto.frontier.physical_status"));
        function->setAttr("pto.frontier.physical",
                          emission.pending->getOperation()->getAttr("pto.frontier.physical"));
        function->removeAttr("pto.frontier.endpoint_reason");
        if (emission.privateSelectors) {
            function->setAttr("pto.frontier.selector_lowering_pending", UnitAttr::get(function.getContext()));
        } else {
            function->removeAttr("pto.frontier.selector_lowering_pending");
        }
    }
    if (options.reportCosts) {
        finishCostReport(report, costs, emission);
        if (failed(printFrontierReport(function, llvm::json::Value(std::move(report))))) {
            function.getBody().takeBody(originalBody);
            function->setAttrs(originalAttrs);
            return failure();
        }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
