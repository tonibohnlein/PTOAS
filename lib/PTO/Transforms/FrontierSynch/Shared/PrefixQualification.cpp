// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PrefixPreparation.h"
#include "PrefixContracts.h"
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
#include "llvm/ADT/DenseSet.h"
#include <chrono>
namespace mlir::pto::frontiersynch::prefix {
LogicalResult Preparation::qualify()
{
    core = pto::recoverSyncPhysicalCore(function);
    for (Operation* op = function; op; op = op->getParentOp()) {
        if (auto attr = op->getAttrOfType<StringAttr>("pto.target_arch")) {
            architecture = attr.str();
            break;
        }
    }
    const auto start = std::chrono::steady_clock::now();
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes))) {
        return refuse(fs::PrefixRefusal::Admission, "shared access recovery failed");
    }
    const auto recovered = std::chrono::steady_clock::now();
    if (failed(index.build(function, input))) {
        return refuse(fs::PrefixRefusal::Readiness, "phase/prerequisite index unavailable");
    }
    recoverySeconds = std::chrono::duration<double>(recovered - start).count();
    indexSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - recovered).count();
    if (failed(fs::preflightPhysicalOutput(function, {}))) {
        return refuse(fs::PrefixRefusal::CutOrDirection, "physical target preflight failed");
    }
    const auto admissionStart = std::chrono::steady_clock::now();
    auto admitted = fs::prefix::checkContracts(function, input, index, architecture, core);
    work["contract_admission_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - admissionStart).count();
    work["contract_operation_visits"] = admitted.operationVisits;
    work["contract_effect_visits"] = admitted.effectVisits;
    work["contract_readiness_operation_visits"] = admitted.readinessVisits;
    work["contract_readiness_use_visits"] = admitted.readinessUses;
    work["gm_visibility_inventory_visits"] = admitted.visibilityInventoryVisits;
    work["gm_read_independence_queries"] = admitted.visibilityQueries;
    if (!admitted.issues.empty()) {
        for (const auto& issue : admitted.issues) {
            issue.operation->emitError("current-cut: ")
                << issue.operation->getName() << ": " << issue.contract << ": " << issue.reason;
        }
        result.refusal = fs::PrefixRefusal::Admission;
        result.reason = admitted.issues.front().reason;
        for (const auto& issue : admitted.issues) {
            if (llvm::StringRef(issue.contract) == "gm-read-visibility") {
                result.refusal = fs::PrefixRefusal::Visibility;
                result.reason = issue.reason;
                break;
            }
            if (llvm::StringRef(issue.contract).contains("readiness") ||
                llvm::StringRef(issue.contract) == "structured-control") {
                result.refusal = fs::PrefixRefusal::Readiness;
            }
        }
        return failure();
    }
    llvm::SmallDenseSet<int64_t> unique;
    for (auto id : pool) {
        if (!fs::isNativeEventId(id) || !unique.insert(id).second) {
            return refuse(fs::PrefixRefusal::Resources, "invalid native eligible pool");
        }
    }
    return success();
}
bool Preparation::protectPaths(
    const fs::PrefixInput& description, const fs::ServiceMask& baseline, fs::ServiceMask& paths)
{
    // Admission already excludes completion-requiring payload results/cuts.
    // Native scalar SSA/control paths remain in the original IR. Preserve all
    // incident services too, rather than infer their support from storage.
    bool qualified = true;
    uint64_t native = 0;
    function.walk([&](Operation* op) {
        for (const auto& prerequisite : index.prerequisitesFor(op)) {
            ++native;
            qualified &= prerequisite.native && prerequisite.producer &&
                         fs::prefix::nativeOrder(*prerequisite.producer, architecture, core) ==
                             fs::prefix::NativeOrder::Serialized;
        }
    });
    const auto& possible = baseline;
    uint64_t protectedCount = 0;
    for (std::size_t id = 0; id < possible.size(); ++id) {
        std::vector<uint32_t> row;
        for (auto source : possible[id]) {
            if (source == static_cast<uint32_t>(pto::PIPE::PIPE_S) ||
                description.sites[id].pipe == static_cast<uint32_t>(pto::PIPE::PIPE_S)) {
                row.push_back(source);
                ++protectedCount;
            }
        }
        paths.push_back(std::move(row));
    }
    work["protected_scalar_services"] = protectedCount;
    work["protected_native_prerequisites"] = native;
    return qualified;
}
} // namespace mlir::pto::frontiersynch::prefix
namespace mlir::pto::frontiersynch {
llvm::json::Object inspectPrefixContracts(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    PhaseIndex index;
    if (!llvm::hasSingleElement(function.getBody())) {
        return llvm::json::Object{{"error", "one structured entry block required"}};
    }
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index.build(function, input))) {
        return llvm::json::Object{{"error", "shared input/phase preparation failed"}};
    }
    std::string architecture;
    for (Operation* op = function; op; op = op->getParentOp()) {
        if (auto attr = op->getAttrOfType<StringAttr>("pto.target_arch")) {
            architecture = attr.str();
            break;
        }
    }
    auto result = prefix::checkContracts(function, input, index, architecture, pto::recoverSyncPhysicalCore(function));
    llvm::json::Array issues;
    for (const auto& issue : result.issues) {
        std::string location;
        llvm::raw_string_ostream stream(location);
        issue.operation->getLoc().print(stream);
        issues.push_back(
            llvm::json::Object{
                {"operation", issue.operation->getName().getStringRef()},
                {"location", location},
                {"contract", issue.contract},
                {"reason", issue.reason}});
    }
    return llvm::json::Object{
        {"error", ""},
        {"issues", std::move(issues)},
        {"operation_visits", result.operationVisits},
        {"effect_visits", result.effectVisits},
        {"readiness_operation_visits", result.readinessVisits},
        {"readiness_use_visits", result.readinessUses},
        {"gm_visibility_inventory_visits", result.visibilityInventoryVisits},
        {"gm_read_independence_queries", result.visibilityQueries},
        {"phases", input.instructions().size()}};
}

} // namespace mlir::pto::frontiersynch
