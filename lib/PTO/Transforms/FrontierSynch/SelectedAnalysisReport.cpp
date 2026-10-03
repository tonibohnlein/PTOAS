// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Diagnostic serialization of durable evidence. Optional periodic all-pairs
// queries are confined here and never needed by semantic retention.
#include "DirectEmissionInternal.h"
#include "StationaryCells.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
namespace {
llvm::json::Value json(Attribute value)
{
    if (auto dictionary = dyn_cast<DictionaryAttr>(value)) {
        llvm::json::Object result;
        for (auto named : dictionary) {
            result[named.getName().strref()] = json(named.getValue());
        }
        return result;
    }
    if (auto array = dyn_cast<ArrayAttr>(value)) {
        llvm::json::Array result;
        for (auto element : array) {
            result.push_back(json(element));
        }
        return result;
    }
    if (auto array = dyn_cast<DenseI64ArrayAttr>(value)) {
        llvm::json::Array result;
        for (int64_t element : array.asArrayRef()) { result.push_back(element); }
        return result;
    }
    if (auto integer = dyn_cast<IntegerAttr>(value)) {
        return integer.getInt();
    }
    return cast<StringAttr>(value).getValue();
}
void periodicThresholds(const SelectedAnalysis& selected, llvm::json::Object& report)
{
    llvm::json::Array thresholds;
    for (std::size_t a = 0; a < selected.sites.size(); ++a) {
        for (std::size_t b = 0; b < selected.sites.size(); ++b) {
            for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                auto value = selected.periodic.threshold(a, b, kind);
                if (succeeded(value) && *value) {
                    std::string text;
                    llvm::raw_string_ostream os(text);
                    os << **value;
                    thresholds.push_back(
                        llvm::json::Array{
                            static_cast<int64_t>(a), static_cast<int64_t>(b), static_cast<int64_t>(kind),
                            std::move(text)});
                }
            }
        }
    }
    report["completion_thresholds"] = std::move(thresholds);
}
} // namespace
llvm::json::Object selectedAnalysisReport(
    const DirectEmissionResult& result, Attribute retained)
{
    llvm::json::Object report;
    if (retained) {
        auto serialized = json(retained);
        report = std::move(*serialized.getAsObject());
    }
    llvm::json::Array attempts;
    for (const auto& attempt : result.attempts) {
        attempts.push_back(
            llvm::json::Object{
                {"route", attempt.route}, {"outcome", attempt.outcome}, {"obligation", attempt.obligation}});
    }
    report["attempts"] = std::move(attempts);
    report["requirements"] = "shared-modeled";
    bool upper = result.selected && result.selected->contract.closure == SelectedClosure::SoundUpper;
    report["selected_closure"] = upper ? "sound-upper" : "modeled-requirements";
    report["demand_name"] = upper ? "F_hat" : "F_star";
    report["status"] = result.emitted ? "prepared-logical" : "unmet-obligation";
    report["endpoint_backend"] = "mlir-arith-scf";
    bool boundaryGuards = result.selected && (result.selected->kind == SelectedAnalysis::Kind::BoundaryLoop ||
                                              result.selected->kind == SelectedAnalysis::Kind::Periodic);
    report["target_lowering"] = result.privateSelectors ? "pending" :
        boundaryGuards ? "supported-source-arith-scf-guards" : "no-generated-guards";
    report["logical_order_equality"] =
        realizedOrderStatement(result);
    if (result.selected) {
        const auto& selected = *result.selected;
        report["cost"] = llvm::json::Object{
            {"attempts", static_cast<int64_t>(selected.attempts.size())},
            {"inspected_regions", static_cast<int64_t>(selected.inspectedRegions)},
            {"cache_hits", static_cast<int64_t>(selected.cacheHits)},
            {"static_commands", static_cast<int64_t>(result.sets + result.waits + result.barriers)}};
        if (selected.stationary) {
            report["stationary_cells"] = llvm::json::Object{
                {"requirements", "shared-modeled"},
                {"physical_overwrite", "not-qualified"},
                {"site_pair_checks", static_cast<int64_t>(selected.stationary->sitePairChecks())},
                {"cells", static_cast<int64_t>(selected.stationary->cells().size())},
                {"fragments", static_cast<int64_t>(selected.stationary->storage().fragments().size())}};
        }
        if (selected.kind == SelectedAnalysis::Kind::Periodic) {
            periodicThresholds(selected, report);
        }
    }
    return report;
}
} // namespace mlir::pto::frontiersynch
