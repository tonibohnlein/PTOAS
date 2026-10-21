// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PrefixPreparation.h"
#include "PTO/Transforms/FrontierSynch/Shared/FunctionStaging.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Verifier.h"
#include <chrono>
namespace mlir::pto::frontiersynch::prefix {
bool Preparation::verifyProtocol() const
{
    // Check the actual prepared command sequence and decoded constant IDs.
    // Whole acknowledgments rearm each directional role before its next use.
    std::map<std::pair<uint32_t, uint32_t>, std::map<bool, int64_t>> roles;
    uint64_t checked = 0;
    for (std::size_t i = 0; i < commands.size();) {
        if (commands[i].barrier) {
            ++i;
            continue;
        }
        if (commands.size() - i < 4) {
            return false;
        }
        const auto& a = commands[i];
        const auto& b = commands[i + 1];
        const auto& c = commands[i + 2];
        const auto& d = commands[i + 3];
        if (b.barrier || c.barrier || d.barrier || !a.publish || b.publish || !c.publish || d.publish ||
            a.cut != b.cut || a.cut != c.cut || a.cut != d.cut || a.source != b.source || a.target != b.target ||
            c.source != a.target || c.target != a.source || d.source != c.source || d.target != c.target ||
            a.physicalId != b.physicalId || c.physicalId != d.physicalId || a.source == a.target) {
            return false;
        }
        for (auto entry : {std::make_pair(&a, false), std::make_pair(&c, true)}) {
            auto& assigned = roles[{entry.first->source, entry.first->target}];
            auto old = assigned.find(entry.second);
            if (old != assigned.end() && old->second != entry.first->physicalId) {
                return false;
            }
            assigned[entry.second] = entry.first->physicalId;
            if (assigned.size() == 2 && assigned[false] == assigned[true]) {
                return false;
            }
        }
        ++checked;
        i += 4;
    }
    return checked == selectedCross;
}
LogicalResult Preparation::emit()
{
    const auto start = std::chrono::steady_clock::now();
    // This ordered immutable list is complete before any IR changes. Constant
    // maps ignore logical coordinates; matching identity remains (site, IVs, p).
    for (const auto& command : commands) {
        OpBuilder builder(command.cut);
        auto location = command.cut->getLoc();
        if (command.barrier) {
            builder.create<pto::BarrierOp>(
                location, pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(command.source)));
        } else {
            auto id = fs::emitPhysicalEndpointMap(builder, location, *command.mapping, {Value{}}, {});
            fs::emitPhysicalCommand(builder, location, command.publish, command.source, command.target, id);
        }
    }
    // Whole invocation only: drain command completion before handing ownership
    // back to the caller. This does not consume any unmatched notification.
    for (auto& block : function.getBody()) {
        auto* end = block.getTerminator();
        if (isa<func::ReturnOp>(end)) {
            OpBuilder builder(end);
            builder.create<pto::BarrierOp>(
                end->getLoc(), pto::PipeAttr::get(builder.getContext(), pto::PIPE::PIPE_ALL));
        }
    }
    if (failed(verify(function))) {
        return failure();
    }
    work["access_recovery_seconds"] = recoverySeconds;
    work["phase_index_seconds"] = indexSeconds;
    work["map_preparation_seconds"] = mapsSeconds;
    work["native_discharge_query_seconds"] = nativeSeconds;
    work["physical_map_construction_seconds"] = physicalMapSeconds;
    work["ordered_command_sites"] = commands.size();
    work["return_completion_sites"] = 1;
    work["selected_self_prefixes"] = selectedSelf;
    work["selected_cross_services"] = selectedCross;
    work["native_self_discharges"] = nativeDischarges;
    work["emission_verification_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return success();
}
} // namespace mlir::pto::frontiersynch::prefix
namespace mlir::pto::frontiersynch {
llvm::StringRef prefixRefusalName(PrefixRefusal reason)
{
    switch (reason) {
        case PrefixRefusal::None:
            return "none";
        case PrefixRefusal::Admission:
            return "access-operation-admission";
        case PrefixRefusal::Visibility:
            return "scalar-gm-visibility";
        case PrefixRefusal::Readiness:
            return "control-readiness";
        case PrefixRefusal::CutOrDirection:
            return "cuts-directions";
        case PrefixRefusal::Resources:
            return "resource-reservation";
        case PrefixRefusal::Certificate:
            return "certificate";
        case PrefixRefusal::Emission:
            return "emission";
    }
    llvm_unreachable("unknown prefix refusal");
}
PrefixResult insertConservativePrefix(func::FuncOp original, GMAliasPolicy policy, const PrefixOptions& options)
{
    if (!llvm::hasSingleElement(original.getBody())) {
        original.emitError("current-cut: one entry block required");
        return {PrefixRefusal::Readiness, "one entry block required", {}, {}};
    }
    FunctionStaging staging(original);
    if (!staging.function) {
        return {PrefixRefusal::Admission, "symbol context unavailable for staging", {}, {}};
    }
    prefix::Preparation adapter(staging.function, policy, options);
    if (failed(adapter.prepare()) || failed(adapter.emit())) {
        if (adapter.result.accepted()) {
            adapter.result.refusal = PrefixRefusal::Emission;
            adapter.result.reason = "input, physical preflight or IR verification failed";
        }
        return adapter.takeResult();
    }
    auto report = adapter.takeResult();
    report.guarantees.modeledOrder = InputOrderGuarantee::InputOrderCovering;
    report.guarantees.realization = RealizationOrder::CoversSelectedOrder;
    report.guarantees.completedExit = true;
    staging.commit(original);
    return report;
}

LogicalResult insertModuleConservativePrefix(ModuleOp module, GMAliasPolicy policy, const PrefixOptions& options)
{
    // The module pass exclusively owns symbol context. Stage the entire
    // request so even a later function refusal leaves the input intact.
    OwningOpRef<ModuleOp> staging(cast<ModuleOp>(module->clone()));
    SmallVector<func::FuncOp> functions;
    staging->walk([&](func::FuncOp function) {
        if (!function.isDeclaration()) {
            functions.push_back(function);
        }
    });
    llvm::json::Array reports;
    for (auto function : functions) {
        auto result = insertConservativePrefix(function, policy, options);
        if (!result.accepted()) {
            module.emitError("conservative-prefix refused [")
                << prefixRefusalName(result.refusal) << "]: " << result.reason;
            return failure();
        }
        reports.push_back(
            llvm::json::Object{
                {"guarantee", "conservative-modeled-storage-coverage"},
                {"realization_order", "covers-selected-prefix-order"},
                {"notification_exit", "closed"},
                {"payload_return", "completed-by-return-ALL"},
                {"function", function.getSymName()},
                {"work", std::move(result.work)}});
    }
    module.getBodyRegion().takeBody(staging->getBodyRegion());
    for (auto& report : reports) {
        llvm::errs() << "frontier-prefix-result " << report << "\n";
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
