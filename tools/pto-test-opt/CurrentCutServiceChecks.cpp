// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Experimental whole-invocation adapter. No production dispatch uses this file.
// Ordered current-cut commands intentionally bypass SET/barrier/WAIT coalescing.
// Only command-free, qualified single-phase payloads with ready structured
// control qualify. The common staging/map planner/emitter retain their contracts.
#include "CompletionDataflowInput.h"
#include "CurrentCutContracts.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/Transforms/FrontierSynch/Shared/FunctionStaging.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionReadiness.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <chrono>
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
struct Command {
    Operation* cut = nullptr;
    bool barrier = false, publish = false;
    uint32_t source = 0, target = 0;
    std::shared_ptr<const fs::PhysicalEndpointMap> mapping;
    int64_t physicalId = -1;
};
void collectServices(const llvm::json::Array& tree, SmallVector<const llvm::json::Object*>& services)
{
    for (const auto& value : tree) {
        const auto& item = *value.getAsObject();
        auto kind = *item.getString("kind");
        if (kind == "service") {
            services.push_back(&item);
        } else if (kind == "loop") {
            collectServices(*item.getArray("body"), services);
        } else {
            collectServices(*item.getArray("then"), services);
            collectServices(*item.getArray("else"), services);
        }
    }
}
class Adapter {
public:
    explicit Adapter(func::FuncOp function, pto::GMAliasPolicy policy, bool pruning)
        : function(function), input(policy), pruning(pruning)
    {}
    LogicalResult prepare();
    LogicalResult emit();

private:
    func::FuncOp function;
    pto::SyncInput input;
    fs::PhaseIndex index;
    SmallVector<Command> commands;
    std::string architecture;
    pto::SyncPhysicalCore core = pto::SyncPhysicalCore::Unknown;
    bool pruning = false;
    uint64_t nativeDischarges = 0, selectedSelf = 0, selectedCross = 0;
    double recoverySeconds = 0, indexSeconds = 0, mapsSeconds = 0;
    double nativeSeconds = 0, physicalMapSeconds = 0;
    llvm::json::Object work;
    SmallVector<int64_t> pool{0, 1, 2, 3, 4, 5};
    LogicalResult qualify();
    bool verifyProtocol() const;
    bool protectPaths(llvm::json::Object& description);
};
LogicalResult Adapter::qualify()
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
        return failure();
    }
    const auto recovered = std::chrono::steady_clock::now();
    if (failed(index.build(function, input))) {
        return failure();
    }
    recoverySeconds = std::chrono::duration<double>(recovered - start).count();
    indexSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - recovered).count();
    if (failed(fs::preflightPhysicalOutput(function, {}))) {
        return failure();
    }
    const auto admissionStart = std::chrono::steady_clock::now();
    auto admitted = current_cut::checkContracts(function, input, index, architecture, core);
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
        return failure();
    }
    if (auto eligible = function->getAttrOfType<DenseI64ArrayAttr>("test.current_cut_ids")) {
        pool.assign(eligible.asArrayRef().begin(), eligible.asArrayRef().end());
    }
    llvm::SmallDenseSet<int64_t> unique;
    for (auto id : pool) {
        if (id < 0 || id >= 6 || !unique.insert(id).second) {
            return function.emitError("current-cut: invalid six-ID eligible pool");
        }
    }
    return success();
}
bool Adapter::protectPaths(llvm::json::Object& description)
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
                         current_cut::nativeOrder(*prerequisite.producer, architecture, core) ==
                             current_cut::NativeOrder::Serialized;
        }
    });
    llvm::json::Array protectedPaths;
    const auto& possible = *description.getObject("projected")->getArray("possible");
    uint64_t protectedCount = 0;
    for (std::size_t id = 0; id < possible.size(); ++id) {
        llvm::json::Array row;
        for (const auto& value : *possible[id].getAsArray()) {
            auto source = *value.getAsInteger();
            if (source == static_cast<uint32_t>(pto::PIPE::PIPE_S) ||
                input.instructions()[id]->kPipeValue == pto::PipelineType::PIPE_S) {
                row.push_back(source);
                ++protectedCount;
            }
        }
        protectedPaths.push_back(std::move(row));
    }
    description["protected_services"] = std::move(protectedPaths);
    work["protected_scalar_services"] = protectedCount;
    work["protected_native_prerequisites"] = native;
    return qualified;
}
bool Adapter::verifyProtocol() const
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
LogicalResult Adapter::prepare()
{
    if (failed(qualify())) {
        return failure();
    }
    auto description = completion_dataflow::Input(input.instructions()).exportFunction(function);
    auto error = description.getString("error");
    if (!error || !error->empty()) {
        return function.emitError("current-cut: structured CFG unavailable");
    }
    work["class_preparation_seconds"] = *description.getNumber("class_preparation_seconds");
    work["descriptor_tree_seconds"] = *description.getNumber("descriptor_tree_seconds");
    bool supported = false;
    if (pruning) {
        const auto supportingStart = std::chrono::steady_clock::now();
        supported = protectPaths(description);
        work["pruning_support_seconds"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - supportingStart).count();
    }
    // Adding protection can grow the root JSON DenseMap. Acquire references
    // to its values only after that mutation, independently of bucket capacity.
    const auto& projected = *description.getObject("projected");
    const auto& metrics = *projected.getObject("metrics");
    work["cfg_preparation_seconds"] = *metrics.getNumber("cfg_preparation_seconds");
    work["solve_seconds"] = *metrics.getNumber("solve_seconds");
    work["plan_serialization_seconds"] = *metrics.getNumber("plan_serialization_seconds");
    work["out_facts"] = *metrics.getInteger("facts_retained");
    work["edge_fact_transmissions"] = *metrics.getInteger("edge_fact_transmissions");
    auto* plan = projected.getArray("physical_plan");
    std::optional<llvm::json::Object> pruned;
    if (pruning) {
        if (supported) {
            pruned = exportSourcePipePruning(description, projected);
            if (pruned->getBoolean("accepted").value_or(false)) {
                plan = pruned->getArray("physical_plan");
            }
        } else {
            pruned = llvm::json::Object{{"accepted", false}, {"reason", "unqualified non-storage supporting path"}};
        }
        auto stats = *pruned;
        stats.erase("physical_plan");
        work["pruning"] = std::move(stats);
    }
    const auto mappingStart = std::chrono::steady_clock::now();
    auto* roles = projected.getArray("role_ids");
    if (!plan || !roles) {
        return function.emitError("current-cut: projected analysis unavailable");
    }
    std::map<std::string, fs::PhysicalRecordAllocation> assignments;
    int64_t record = 0;
    for (const auto& value : *roles) {
        const auto& role = *value.getAsObject();
        auto ordinal = *role.getInteger("id");
        if (ordinal < 0 || uint64_t(ordinal) >= pool.size()) {
            return function.emitError("current-cut: eligible role reservation unavailable");
        }
        fs::PhysicalRecordAllocation allocation;
        allocation.record = record++;
        allocation.sourcePipe = *role.getInteger("source");
        allocation.targetPipe = *role.getInteger("target");
        allocation.ids = {pool[ordinal]};
        auto label = role.getString("role")->str() + ":" + std::to_string(allocation.sourcePipe) + ":" +
                     std::to_string(allocation.targetPipe);
        assignments[label] = allocation;
    }
    if (architecture != "a3") {
        return function.emitError("current-cut: A3 invocation-return contract required");
    }
    DominanceInfo dominance;
    auto constructCommands = [&](const llvm::json::Array& selected) -> LogicalResult {
        SmallVector<const llvm::json::Object*> services;
        collectServices(selected, services);
        // Consume the one generated structured plan. Original SCF still owns its
        // control and occurrence coordinates; no endpoint or guard is relocated.
        for (const auto* service : services) {
            auto id = *service->getInteger("site");
            if (id < 0 || uint64_t(id) >= input.instructions().size()) {
                return function.emitError("current-cut: invalid planned occurrence");
            }
            auto* cut = input.instructions()[id]->elementOp;
            for (const auto& value : *service->getArray("commands")) {
                const auto& command = *value.getAsObject();
                auto op = *command.getString("op");
                auto pipe = static_cast<uint32_t>(*command.getInteger("pipe"));
                if (op == "B") {
                    ++selectedSelf;
                    const auto nativeStart = std::chrono::steady_clock::now();
                    const auto order = current_cut::nativeOrder(*input.instructions()[id], architecture, core);
                    nativeSeconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - nativeStart).count();
                    if (order == current_cut::NativeOrder::Serialized) {
                        ++nativeDischarges;
                        continue;
                    }
                    auto available = pto::getSyncBarrierAvailability(architecture, core, static_cast<pto::PIPE>(pipe));
                    if (!available.value_or(false)) {
                        return function.emitError("current-cut: local barrier unavailable");
                    }
                    commands.push_back({cut, true, false, pipe, pipe, {}});
                    continue;
                }
                auto found = assignments.find(command.getString("source")->str());
                if (found == assignments.end()) {
                    return function.emitError("current-cut: missing role assignment");
                }
                const auto& allocation = found->second;
                auto source = allocation.sourcePipe;
                auto target = allocation.targetPipe;
                if (op == "S" && command.getString("source")->starts_with("F:")) {
                    ++selectedCross;
                }
                const auto reserved = pto::SyncEventIdAllocation::GetReservedEventIdNum(
                    static_cast<pto::PipelineType>(source), static_cast<pto::PipelineType>(target));
                if (reserved >= pto::kTotalEventIdNum || llvm::any_of(allocation.ids, [&](int64_t physical) {
                        return static_cast<uint64_t>(physical) >= pto::kTotalEventIdNum - reserved;
                    })) {
                    return function.emitError("current-cut: role collides with reserved directional tail");
                }
                auto capability = pto::getSyncEventAvailability(
                    architecture, core, static_cast<pto::PIPE>(source), static_cast<pto::PIPE>(target));
                if (!capability.available.value_or(false) || pipe != (op == "S" ? source : target)) {
                    return function.emitError("current-cut: direction unavailable or endpoint inconsistent");
                }
                const auto mapStart = std::chrono::steady_clock::now();
                auto mapping = fs::planPhysicalEndpointMap(cut, {allocation}, {{}}, 1, false, dominance);
                physicalMapSeconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - mapStart).count();
                if (failed(mapping)) {
                    return failure();
                }
                commands.push_back({cut, false, op == "S", source, target, *mapping, allocation.ids.front()});
            }
        }
        return success();
    };
    if (failed(constructCommands(*plan))) {
        return failure();
    }
    const auto protocolStart = std::chrono::steady_clock::now();
    if (!verifyProtocol()) {
        if (!pruned || !pruned->getBoolean("accepted").value_or(false)) {
            return function.emitError("current-cut: ordered role protocol certificate failed");
        }
        commands.clear();
        nativeDischarges = 0;
        selectedSelf = 0;
        selectedCross = 0;
        work["pruning"] = llvm::json::Object{{"accepted", false}, {"reason", "physical protocol certificate failed"}};
        if (failed(constructCommands(*projected.getArray("physical_plan"))) || !verifyProtocol()) {
            return function.emitError("current-cut: baseline protocol certificate failed");
        }
    }
    work["protocol_verification_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - protocolStart).count();
    work["protocol_certified"] = true;
    mapsSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - mappingStart).count();
    return success();
}
LogicalResult Adapter::emit()
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
    work["selected_self_prefixes"] = selectedSelf;
    work["selected_cross_services"] = selectedCross;
    work["native_self_discharges"] = nativeDischarges;
    work["emission_verification_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    llvm::errs() << "current-cut-work " << llvm::json::Value(std::move(work)) << "\n";
    return success();
}
} // namespace
LogicalResult emitCurrentCutServices(func::FuncOp original, pto::GMAliasPolicy policy, bool pruning)
{
    if (!llvm::hasSingleElement(original.getBody())) {
        return original.emitError("current-cut: one entry block required");
    }
    fs::FunctionStaging staging(original);
    Adapter adapter(staging.function, policy, pruning);
    if (failed(adapter.prepare()) || failed(adapter.emit())) {
        return failure();
    }
    staging.commit(original);
    return success();
}

llvm::json::Object inspectCurrentCutContracts(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    fs::PhaseIndex index;
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
    auto result =
        current_cut::checkContracts(function, input, index, architecture, pto::recoverSyncPhysicalCore(function));
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
