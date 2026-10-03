// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compact diagnostic evidence only. Missing target/counter capabilities remain
// null, not guessed from logical command cardinality or shared numeric ID enums.
#include "DirectEmissionInternal.h"
#include "GeneralQueries.h"
#include "SingleStreamLoop.h"
#include "PTO/Transforms/FrontierSynch/FrontierSynch.h"
#include "llvm/Support/FormatVariadic.h"
#include <mutex>
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
llvm::StringRef stageName(CostStage stage)
{
    constexpr llvm::StringLiteral names[] = {"effect_recovery", "recognition", "backend",  "merge",      "selectors",
                                             "insertion",       "allocation",  "lowering", "diagnostics"};
    return names[static_cast<unsigned>(stage)];
}
llvm::StringRef stageStatus(CostStage stage, const CostLedger& costs, const DirectEmissionResult& result)
{
    if (stage == CostStage::Allocation) {
        if (result.pending && result.pending.get()->hasAttr("pto.frontier.physical_status")) {
            return "certified-under-recorded-premises";
        }
        return costs.stage(stage).invocations ? "failed" : "not-attempted";
    }
    if (stage == CostStage::Lowering) {
        return result.privateSelectors ? "pending-selector-lowering" : "not-attempted";
    }
    return costs.stage(stage).invocations ? "executed" : "not-attempted";
}
llvm::json::Array attempts(ArrayRef<AnalysisAttempt> input)
{
    llvm::json::Array result;
    for (const auto& attempt : input) {
        result.push_back(
            llvm::json::Object{
                {"route", attempt.route},
                {"outcome", attempt.outcome},
                {"obligation", attempt.obligation},
                {"inclusive_ns", attempt.inclusiveNanoseconds}});
    }
    return result;
}
std::size_t origins(const SyncInput& input)
{
    DenseSet<Value> roots;
    for (const auto* phase : input.instructions()) {
        for (const auto* memory : phase->defVec) {
            roots.insert(memory->rootBuffer);
        }
        for (const auto* memory : phase->useVec) {
            roots.insert(memory->rootBuffer);
        }
    }
    return roots.size();
}
std::size_t crossings(func::FuncOp function, const SelectedAnalysis& selected)
{
    if (selected.kind != SelectedAnalysis::Kind::Signed) {
        return 0;
    }
    SmallVector<Operation*> children;
    for (const auto& site : selected.structured->sites()) {
        Operation* child = site.anchor;
        Operation* parent = child->getParentOp();
        while (parent != function) {
            child = parent;
            parent = child->getParentOp();
        }
        children.push_back(child);
    }
    std::size_t count = 0;
    for (const auto& piece : selected.minimum->pieces()) {
        if (piece.domain.site && piece.range.site && children[*piece.domain.site] != children[*piece.range.site]) {
            ++count;
        }
    }
    return count;
}
void selectedMetrics(func::FuncOp function, const SelectedAnalysis& selected, llvm::json::Object& metrics)
{
    metrics["inspected_regions"] = static_cast<int64_t>(selected.inspectedRegions);
    metrics["import_cache_hits"] = static_cast<int64_t>(selected.cacheHits);
    metrics["crossing_relation_pieces"] = selected.kind == SelectedAnalysis::Kind::Signed ?
                                              llvm::json::Value(static_cast<int64_t>(crossings(function, selected))) :
                                              llvm::json::Value(nullptr);
    metrics["endpoint_maps"] = static_cast<int64_t>(selected.endpoints.size());
    std::size_t selectors = 0;
    for (const auto& endpoint : selected.endpoints) {
        selectors += endpoint.second->pieces().size();
    }
    metrics["selector_pieces"] = static_cast<int64_t>(selectors);
    switch (selected.kind) {
        case SelectedAnalysis::Kind::BoundaryLoop:
            metrics["retained_demand_families"] = static_cast<int64_t>(
                selected.boundaryLoop->bodies.size() + (selected.boundaryLoop->repeatedPair() ? 1 : 2) +
                (selected.boundaryLoop->hasPrefix ? 1 : 0) +
                (selected.boundaryLoop->frames.empty() ? 0 : selected.boundaryLoop->frames.size() - 1));
            metrics["existing_prefix_crossing"] = selected.boundaryLoop->hasPrefix;
            metrics["frame_depth"] = static_cast<int64_t>(
                selected.boundaryLoop->frames.empty() ? 1 : selected.boundaryLoop->frames.size());
            if (!selected.boundaryLoop->frames.empty()) {
                metrics["compact_child_indices"] = selected.boundaryLoop->loopChild ? 1 : 0;
                metrics["frame_domain"] = "original rectangular tuples; no numerical unfolding";
                metrics["dynamic_generation_formula"] = selected.boundaryLoop->repeatedPair() ?
                    "READY=T; RELEASE=max(T-1,0); final=int(T>0), T=positive rectangular occurrence count" :
                    "entry=exit=int(T>0), T=positive rectangular occurrence count";
            }
            metrics["boundary_generations"] = selected.boundaryLoop->repeatedPair() ? 1 : 2;
            if (selected.boundaryLoop->protocol == SingleStreamProtocol::SequentialBoundaries ||
                selected.boundaryLoop->protocol == SingleStreamProtocol::ExclusiveArmBoundaries) {
                metrics["retained_demand_families"] =
                    static_cast<int64_t>(selected.boundaryLoop->bodies.size()) +
                    (selected.boundaryLoop->protocol == SingleStreamProtocol::ExclusiveArmBoundaries ? 4 : 5);
                metrics["closure_updates"] = static_cast<int64_t>(selected.boundaryLoop->portClosureUpdates);
                metrics["closure_stored_cell_maximum_pieces"] =
                    static_cast<int64_t>(selected.boundaryLoop->portClosureMaximumPieces);
                metrics["static_boundary_pair_alternatives"] = 4;
                metrics["static_endpoint_templates"] = 8;
                metrics["compact_child_indices"] = 2;
            }
            metrics["repeated_handoff_families"] = selected.boundaryLoop->repeatedPair() ? 2 : 0;
            metrics["original_barrier_templates"] = selected.boundaryLoop->repeatedPair() ? 1 : 0;
            metrics["inserted_local_barrier_templates"] = selected.boundaryLoop->repeatedPair() ?
                static_cast<int64_t>(selected.boundaryLoop->bodies.size() - 2) :
                static_cast<int64_t>(selected.boundaryLoop->bodies.size());
            break;
        case SelectedAnalysis::Kind::Explicit:
            metrics["retained_records"] = static_cast<int64_t>(selected.explicitReduction.retained().size());
            metrics["dynamic_counts"] =
                "one occurrence per retained explicit demand before physical repair "
                "(includes original local prerequisites)";
            break;
        case SelectedAnalysis::Kind::Periodic:
            metrics["retained_records"] = static_cast<int64_t>(selected.periodic.retained().size());
            metrics["quotient_vertices"] = static_cast<int64_t>(selected.periodic.vertexCount());
            metrics["quotient_edges"] = static_cast<int64_t>(selected.periodic.edgeCount());
            break;
        case SelectedAnalysis::Kind::General:
            metrics["minimum_relation_pieces"] = static_cast<int64_t>(selected.general->minimum.getNumDisjuncts());
            metrics["native_relation_pieces"] = static_cast<int64_t>(selected.general->native.getNumDisjuncts());
            metrics["reachability_relation_pieces"] =
                static_cast<int64_t>(selected.general->reachability.getNumDisjuncts());
            if (selected.generalEndpoints) {
                metrics["endpoint_maps"] = static_cast<int64_t>(selected.generalEndpoints->endpoints.size());
                std::size_t pieces = 0;
                for (const auto& endpoint : selected.generalEndpoints->endpoints) {
                    pieces += endpoint.function.getNumPieces();
                }
                metrics["selector_pieces"] = static_cast<int64_t>(pieces);
            }
            break;
        case SelectedAnalysis::Kind::Signed:
            metrics["minimum_relation_pieces"] = static_cast<int64_t>(selected.minimum->pieces().size());
            metrics["native_relation_pieces"] = static_cast<int64_t>(selected.native->pieces().size());
            metrics["reachability_relation_pieces"] = static_cast<int64_t>(selected.reachability->pieces().size());
            break;
        case SelectedAnalysis::Kind::Guarded:
            metrics["retained_records"] = static_cast<int64_t>(selected.guarded.retained().size());
            metrics["predicate_circuit_nodes"] = static_cast<int64_t>(selected.guarded.predicates().nodes().size());
            break;
    }
}
} // namespace
llvm::json::Object costReport(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& source,
    const DirectEmissionResult& result, bool dumpedDemands)
{
    std::size_t pairs = 0, rejected = 0;
    for (auto incoming : source.conflicts()) {
        pairs += incoming.size();
    }
    for (const auto& attempt : result.attempts) {
        rejected += attempt.outcome != "ready";
    }
    llvm::json::Object metrics{
        {"sites", static_cast<int64_t>(source.sites().size())},
        {"shared_storage_origins", static_cast<int64_t>(origins(input))},
        {"modeled_conflict_site_pairs", static_cast<int64_t>(pairs)},
        {"endpoint_sites", static_cast<int64_t>(result.sets + result.waits + result.barriers)},
        {"sets", static_cast<int64_t>(result.sets)},
        {"waits", static_cast<int64_t>(result.waits)},
        {"barriers", static_cast<int64_t>(result.barriers)},
        {"invocation_epilogue", "excluded"},
        {"dynamic_counts", "not-supplied"},
        {"pool_capacities", nullptr},
        {"physical_ids", nullptr}};
    if (result.selected) {
        selectedMetrics(function, *result.selected, metrics);
        if (result.selected->kind == SelectedAnalysis::Kind::Explicit && result.pending &&
            result.pending.get()->hasAttr("pto.frontier.physical_status")) {
            metrics["physical_dynamic_pairs"] = result.sets;
        }
    }
    if (result.pending) {
        auto physical = result.pending.get()->getAttrOfType<DictionaryAttr>("pto.frontier.physical");
        if (physical) {
            if (auto excess = physical.getAs<StringAttr>("repair_excess")) {
                metrics["repair_added_excess"] = excess.getValue();
                metrics["repair_guarantee"] = physical.getAs<StringAttr>("guarantee").getValue();
            }
            llvm::json::Array capacities, ids;
            for (auto item : physical.getAs<ArrayAttr>("pools")) {
                auto pool = cast<DictionaryAttr>(item);
                capacities.push_back(llvm::json::Object{
                    {"allocator", pool.getAs<StringAttr>("allocator").getValue()},
                    {"source", pool.getAs<IntegerAttr>("source").getInt()},
                    {"target", pool.getAs<IntegerAttr>("target").getInt()},
                    {"available", static_cast<int64_t>(pool.getAs<ArrayAttr>("eligible").size())},
                    {"required", pool.getAs<IntegerAttr>("required").getInt()}});
            }
            for (auto item : physical.getAs<ArrayAttr>("matching")) {
                ids.push_back(cast<DictionaryAttr>(item).getAs<IntegerAttr>("id").getInt());
            }
            metrics["pool_capacities"] = std::move(capacities);
            metrics["physical_ids"] = std::move(ids);
        }
    }
    return llvm::json::Object{
        {"report", "frontier-costs-v1"},
        {"function", function.getSymName()},
        {"algorithm", "frontier-synch"},
        {"route", result.route},
        {"analysis", result.selected ? (result.selected->contract.closure == SelectedClosure::SoundUpper ?
            "selected-upper-covers" : "selected-minimum") : "unmet-obligation"},
        {"selected_closure", result.selected && result.selected->contract.closure == SelectedClosure::SoundUpper ?
            "sound-upper" : "modeled-requirements"},
        {"requirements", "shared-modeled"},
        {"gm_alias_policy", input.memory().gmPolicy() == GMAliasPolicy::MayAlias ? "may-alias" : "may-not-alias"},
        {"reduction_quality", result.selected ? "selected-order-covers" : "not-established"},
        {"logical_order_equality", realizedOrderStatement(result)},
        {"logical_insertion", result.emitted ? "prepared-logical" : "unresolved"},
        {"physical_realization", result.pending && result.pending.get()->hasAttr("pto.frontier.physical_status") ?
            "physical assignment certified under recorded target/ABI premises" : "not-certified"},
        {"failure_category", result.emitted ? llvm::json::Value(nullptr) : llvm::json::Value("NoCertifiedPlan")},
        {"reason", result.reason},
        {"accounting", "exclusive nested stages; inclusive attempts are non-additive"},
        {"excluded_costs", "cost-report metrics/serialization/IO; pass dispatch overhead; work outside this pass"},
        {"attempts", attempts(result.attempts)},
        {"rejected_attempts", static_cast<int64_t>(rejected)},
        {"metrics", std::move(metrics)},
        {"demand_dump_enabled", dumpedDemands}};
}
void finishCostReport(llvm::json::Object& report, const CostLedger& costs, const DirectEmissionResult& result)
{
    llvm::json::Object stages;
    int64_t total = 0;
    for (unsigned id = 0; id < static_cast<unsigned>(CostStage::Count); ++id) {
        auto stage = static_cast<CostStage>(id);
        const auto& measured = costs.stage(stage);
        total += measured.nanoseconds;
        stages[stageName(stage)] = llvm::json::Object{
            {"exclusive_ns", measured.nanoseconds},
            {"invocations", measured.invocations},
            {"status", stageStatus(stage, costs, result)}};
    }
    report["stages"] = std::move(stages);
    report["stage_total_ns"] = total;
}
LogicalResult printFrontierReport(func::FuncOp function, llvm::json::Value report)
{
    static std::mutex outputMutex;
    std::lock_guard<std::mutex> lock(outputMutex);
    llvm::errs() << llvm::formatv("{0}\n", report);
    if (llvm::errs().has_error()) {
        return function.emitError("could not write frontier diagnostic report");
    }
    return success();
}
namespace {
void reportPreanalysisFailure(func::FuncOp function, const CostLedger& costs, llvm::StringRef reason,
                              bool dumpDemands)
{
    llvm::json::Object report{
        {"report", "frontier-costs-v1"},
        {"function", function.getSymName()},
        {"algorithm", "frontier-synch"},
        {"analysis", "import-failed"},
        {"requirements", "not-imported"},
        {"reduction_quality", "not-established"},
        {"logical_order_equality", "not-certified"},
        {"logical_insertion", "not-attempted"},
        {"physical_realization", "not-certified"},
        {"failure_category", "ImportFailure"},
        {"reason", reason},
        {"attempts", llvm::json::Array{}},
        {"rejected_attempts", 0},
        {"metrics", nullptr},
        {"demand_dump_enabled", dumpDemands},
        {"accounting", "exclusive nested stages; inclusive attempts are non-additive"},
        {"excluded_costs", "cost-report metrics/serialization/IO; pass dispatch overhead; work outside this pass"}};
    finishCostReport(report, costs, DirectEmissionResult{});
    auto* stages = report["stages"].getAsObject();
    (*(*stages)["effect_recovery"].getAsObject())["status"] = "failed";
    (void)printFrontierReport(function, llvm::json::Value(std::move(report)));
}
} // namespace
void reportImportFailure(func::FuncOp function, const CostLedger& costs, llvm::StringRef reason, bool dumpDemands)
{
    reportPreanalysisFailure(function, costs, reason, dumpDemands);
}
} // namespace mlir::pto::frontiersynch
