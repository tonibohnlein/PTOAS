// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PrefixPreparation.h"
#include "PrefixNativeOrder.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include <chrono>
namespace mlir::pto::frontiersynch::prefix {
bool Preparation::refine(
    const fs::PrefixInput& description, const fs::SourcePipeServices& baseline, fs::ServiceMask& selected)
{
    bool acceptedPruning = false;
    {
        fs::ServiceMask paths;
        const auto supportStart = std::chrono::steady_clock::now();
        const bool supported = protectPaths(description, baseline.possible, paths);
        work["pruning_support_seconds"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - supportStart).count();
        llvm::json::Object stats;
        if (supported) {
            const auto proposalStart = std::chrono::steady_clock::now();
            auto proposal = fs::proposeLocalServices(description.graph, description.sites, baseline.possible, paths);
            const auto proposed = std::chrono::steady_clock::now();
            fs::DeliveryCertificate certificate;
            if (proposal.error.empty()) {
                certificate = fs::certifyServiceDelivery(
                    description.graph, description.sites, baseline.possible, proposal.kept, paths);
            }
            stats = llvm::json::Object{
                {"accepted", certificate.accepted},
                {"reason", proposal.error.empty() ? certificate.error : proposal.error},
                {"certificate_runs", proposal.error.empty() ? 1 : 0},
                {"input_preparation_seconds", 0.0},
                {"proposal_seconds", std::chrono::duration<double>(proposed - proposalStart).count()},
                {"verification_seconds",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - proposed).count()},
                {"proposal_nodes", proposal.nodes},
                {"service_tests", proposal.tested},
                {"class_tests", proposal.classTests},
                {"class_updates", proposal.updates},
                {"clear_entries", proposal.cleared},
                {"proposed_deletions", proposal.deleted},
                {"facts_retained", certificate.retained},
                {"fact_transmissions", certificate.transmissions},
                {"fact_inspections", certificate.inspected},
                {"violations", certificate.violations}};
            if (certificate.accepted) {
                selected = proposal.kept;
                acceptedPruning = true;
            }
            llvm::json::Array kept;
            for (const auto& row : proposal.kept) {
                llvm::json::Array values;
                for (auto pipe : row) {
                    values.push_back(pipe);
                }
                kept.push_back(std::move(values));
            }
            stats["kept"] = std::move(kept);
        } else {
            stats = llvm::json::Object{{"accepted", false}, {"reason", "unqualified non-storage supporting path"}};
        }
        work["pruning"] = std::move(stats);
    }
    return acceptedPruning;
}
LogicalResult Preparation::reserveRoles(const fs::PrefixInput& description, const fs::ServiceMask& baseline)
{
    int64_t record = 0;
    llvm::json::Array reservedRoles;
    for (const auto& role : fs::prefixRoles(description.sites, baseline)) {
        fs::PhysicalRecordAllocation allocation;
        allocation.record = record++;
        allocation.sourcePipe = role.source;
        allocation.targetPipe = role.target;
        const auto reserved = pto::SyncEventIdAllocation::GetReservedEventIdNum(
            static_cast<pto::PipelineType>(role.source), static_cast<pto::PipelineType>(role.target));
        SmallVector<int64_t> directional;
        for (auto id : pool) {
            if (reserved < pto::kTotalEventIdNum && static_cast<uint64_t>(id) < pto::kTotalEventIdNum - reserved) {
                directional.push_back(id);
            }
        }
        if (role.ordinal >= directional.size()) {
            return refuse(
                fs::PrefixRefusal::Resources, "eligible role reservation unavailable (directional reserved tail)");
        }
        allocation.ids = {directional[role.ordinal]};
        assignments[{role.source, role.target, role.reverse}] = allocation;
        reservedRoles.push_back(
            llvm::json::Object{
                {"source", role.source},
                {"target", role.target},
                {"role", role.reverse ? "R" : "F"},
                {"id", allocation.ids.front()}});
    }
    work["reserved_role_ids"] = std::move(reservedRoles);
    // The existing A3 invocation-completion contract covers ALL. The local
    // per-pipe barrier availability table intentionally does not model ALL.
    if (architecture != "a3") {
        return refuse(fs::PrefixRefusal::CutOrDirection, "A3 invocation-return contract required");
    }
    return success();
}
LogicalResult Preparation::constructCommands(const fs::PrefixInput& description, const fs::ServiceMask& masks)
{
    DominanceInfo dominance;
    for (auto id : description.order) {
        auto* cut = input.instructions()[id]->elementOp;
        for (const auto& command : fs::prefixCommands(description.sites[id].pipe, masks[id])) {
            if (command.kind == fs::PrefixCommandKind::Barrier) {
                ++selectedSelf;
                const auto nativeStart = std::chrono::steady_clock::now();
                const auto order = fs::prefix::nativeOrder(*input.instructions()[id], architecture, core);
                nativeSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - nativeStart).count();
                if (order == fs::prefix::NativeOrder::Serialized) {
                    ++nativeDischarges;
                    continue;
                }
                auto available =
                    pto::getSyncBarrierAvailability(architecture, core, static_cast<pto::PIPE>(command.source));
                if (!available.value_or(false)) {
                    return refuse(fs::PrefixRefusal::CutOrDirection, "local barrier unavailable");
                }
                commands.push_back({cut, true, false, command.source, command.target, {}});
                continue;
            }
            auto found = assignments.find({command.source, command.target, command.reverse});
            if (found == assignments.end()) {
                return refuse(fs::PrefixRefusal::Certificate, "missing role assignment");
            }
            const auto& allocation = found->second;
            auto source = allocation.sourcePipe;
            auto target = allocation.targetPipe;
            bool publish = command.kind == fs::PrefixCommandKind::Set;
            if (publish && !command.reverse) {
                ++selectedCross;
            }
            auto capability = pto::getSyncEventAvailability(
                architecture, core, static_cast<pto::PIPE>(source), static_cast<pto::PIPE>(target));
            if (!capability.available.value_or(false)) {
                return refuse(fs::PrefixRefusal::CutOrDirection, "direction unavailable or endpoint inconsistent");
            }
            const auto mapStart = std::chrono::steady_clock::now();
            auto mapping = fs::planPhysicalEndpointMap(cut, {allocation}, {{}}, 1, false, dominance);
            physicalMapSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - mapStart).count();
            if (failed(mapping)) {
                return refuse(fs::PrefixRefusal::CutOrDirection, "physical endpoint map unavailable");
            }
            commands.push_back({cut, false, publish, source, target, *mapping, allocation.ids.front()});
        }
    }
    return success();
}
LogicalResult Preparation::prepare()
{
    if (failed(qualify())) {
        return failure();
    }
    auto description = fs::buildPrefixInput(function, input.instructions());
    if (!description.error.empty()) {
        return refuse(fs::PrefixRefusal::CutOrDirection, description.error);
    }
    work["class_preparation_seconds"] = description.classSeconds;
    work["descriptor_tree_seconds"] = description.graphSeconds;
    work["cfg_preparation_seconds"] = description.graphSeconds;
    work["storage_classes"] = description.classes;
    work["cfg_nodes"] = description.graph.size();
    const auto solveStart = std::chrono::steady_clock::now();
    const auto baseline = fs::solveSourcePipeServices(description.graph, description.sites, description.classes);
    work["solve_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - solveStart).count();
    if (!baseline.error.empty()) {
        return refuse(fs::PrefixRefusal::Certificate, baseline.error);
    }
    work["prepared_modes"] = baseline.work.preparedModes;
    work["cfg_edges"] = baseline.work.preparedEdges;
    work["payload_sites"] = description.sites.size();
    work["out_facts"] = baseline.work.outgoing;
    work["edge_fact_transmissions"] = baseline.work.transmissions;
    fs::ServiceMask selected = baseline.possible;
    const bool acceptedPruning = pruning && refine(description, baseline, selected);
    const auto mappingStart = std::chrono::steady_clock::now();
    if (failed(reserveRoles(description, baseline.possible))) {
        return failure();
    }
    if (failed(constructCommands(description, selected))) {
        return failure();
    }
    const auto protocolStart = std::chrono::steady_clock::now();
    if (!verifyProtocol()) {
        if (!acceptedPruning) {
            return refuse(fs::PrefixRefusal::Certificate, "ordered role protocol certificate failed");
        }
        commands.clear();
        nativeDischarges = 0;
        selectedSelf = 0;
        selectedCross = 0;
        work["pruning"] = llvm::json::Object{{"accepted", false}, {"reason", "physical protocol certificate failed"}};
        if (failed(constructCommands(description, baseline.possible)) || !verifyProtocol()) {
            return refuse(fs::PrefixRefusal::Certificate, "baseline protocol certificate failed");
        }
    }
    work["protocol_verification_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - protocolStart).count();
    work["protocol_certified"] = true;
    mapsSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mappingStart).count();
    return success();
}
} // namespace mlir::pto::frontiersynch::prefix
