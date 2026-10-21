// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only admission adapter over existing effect/phase/readiness contracts.
// No operation-name registry: unknown effects are not an empty access set.
#ifndef PTO_TEST_CURRENTCUTCONTRACTS_H
#define PTO_TEST_CURRENTCUTCONTRACTS_H
#include "CurrentCutNativeOrder.h"
#include "PTO/IR/PTOAccessRegion.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionReadiness.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace current_cut {
struct ContractIssue {
    mlir::Operation* operation;
    const char* contract;
    const char* reason;
};
struct ContractCheck {
    llvm::SmallVector<ContractIssue> issues;
    uint64_t operationVisits = 0, effectVisits = 0, readinessVisits = 0, readinessUses = 0;
    uint64_t visibilityInventoryVisits = 0, visibilityQueries = 0;
    bool possibleGmWrites = false;
    void note(mlir::Operation* op, const char* contract, const char* reason)
    {
        issues.push_back({op, contract, reason});
    }
};
inline bool representedEffect(
    const mlir::pto::SyncInput& input, const mlir::pto::CompoundInstanceElement& phase,
    const mlir::pto::SyncMemoryEffect& effect)
{
    auto found = input.buffers().find(effect.getValue());
    if (found == input.buffers().end() || found->second.empty()) {
        return false;
    }
    auto memories = mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()) ? llvm::ArrayRef(phase.useVec) :
                                                                               llvm::ArrayRef(phase.defVec);
    return llvm::all_of(
        found->second, [&](const auto& memory) { return memory && llvm::is_contained(memories, memory.get()); });
}
inline void checkEffects(
    mlir::Operation* op, const mlir::pto::SyncInput& input, const mlir::pto::CompoundInstanceElement& phase,
    ContractCheck& result)
{
    using namespace mlir;
    if (!isa<MemoryEffectOpInterface>(op)) {
        result.note(op, "effects", "payload has no declared read/write effect contract");
        return;
    }
    for (const auto& effect : input.effectsFor(op)) {
        ++result.effectVisits;
        auto parameters = dyn_cast_or_null<DictionaryAttr>(effect.getParameters());
        if (parameters && parameters.getAs<UnitAttr>("pto.descriptor_state")) {
            continue;
        }
        if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()) || !effect.getValue() ||
            !representedEffect(input, phase, effect)) {
            result.note(op, "effect-coverage", "declared effect is not covered by retained read/write storage facts");
        }
    }
}
// SET/WAIT availability is not a cache-maintenance contract. Supported
// scalar GM reads are read-only relative to this invocation's modeled writes,
// under the supplied alias policy and ordinary coherent-input entry contract.
inline void checkReadVisibility(
    mlir::Operation* op, const mlir::pto::SyncInput& input, const mlir::pto::CompoundInstanceElement& phase,
    ContractCheck& result)
{
    using namespace mlir::pto;
    if (phase.kPipeValue != PipelineType::PIPE_S || !result.possibleGmWrites) {
        return;
    }
    for (auto id : input.accesses().effectsFor(&phase)) {
        const auto& effect = input.accesses().effects()[id];
        if (effect.mode != SyncAccessMode::Read ||
            (effect.memory->scope != AddressSpace::GM && effect.memory->scope != AddressSpace::Zero)) {
            continue;
        }
        ++result.visibilityQueries;
        if (!phase.defVec.empty() || !input.accesses().independentOfOtherPhases(id)) {
            result.note(
                op, "gm-read-visibility",
                "possible GM writer needs an explicit scalar read/cache-visibility contract; event availability alone "
                "is insufficient");
        }
    }
}
inline bool checkControl(mlir::Operation* op, ContractCheck& result)
{
    using namespace mlir;
    if (isa<func::FuncOp, func::ReturnOp, scf::YieldOp, scf::IfOp>(op)) {
        return true;
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (loop.getNumResults() || loop.getNumRegionIterArgs()) {
            result.note(op, "structured-control", "loop-carried values need an explicit readiness/coordinate contract");
        }
        return true;
    }
    if (isa<func::CallOp>(op)) {
        result.note(op, "child-ownership", "call needs an admitted invocation and notification-ownership contract");
        return true;
    }
    if (op->getNumRegions()) {
        result.note(op, "structured-control", "region control is outside the supported SCF for/if contract");
        return true;
    }
    if (pto::isPreservedSyncProtocol(op) || isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp,
                                                pto::RecordEventOp, pto::WaitEventOp>(op)) {
        result.note(op, "event-ownership", "existing synchronization needs a separate entry/exit ownership contract");
        return true;
    }
    return false;
}
inline void checkPayload(
    mlir::Operation* op, const mlir::pto::SyncInput& input, const mlir::pto::CompoundInstanceElement& phase,
    llvm::StringRef architecture, mlir::pto::SyncPhysicalCore core, ContractCheck& result)
{
    using namespace mlir;
    for (Value value : op->getResults()) {
        if (pto::resultAvailability(phase, value) == pto::SyncResultAvailability::NonScalar) {
            result.note(op, "result-readiness", "non-scalar payload results need a qualified value-readiness contract");
        }
    }
    auto pipe = dyn_cast<pto::OpPipeInterface>(op);
    if (!pipe || static_cast<pto::PipelineType>(pipe.getPipe()) != phase.kPipeValue) {
        result.note(op, "pipe", "original pipe contract does not agree with the retained phase");
    }
    if (!pto::getSyncPipelinePresence(architecture, core, static_cast<pto::PIPE>(phase.kPipeValue)).value_or(false)) {
        result.note(op, "target-pipe", "modeled pipe is unavailable on this target/core");
    }
    checkEffects(op, input, phase, result);
    checkReadVisibility(op, input, phase, result);
    if (phase.kPipeValue == pto::PipelineType::PIPE_S &&
        nativeOrder(phase, architecture, core) != NativeOrder::Serialized) {
        result.note(op, "native-prefix", "scalar phase lacks a synchronous native-prefix contract for this form");
    }
}
inline void checkOperation(
    mlir::Operation* op, const mlir::pto::SyncInput& input, const mlir::pto::frontiersynch::PhaseIndex& index,
    llvm::StringRef architecture, mlir::pto::SyncPhysicalCore core, ContractCheck& result)
{
    using namespace mlir;
    ++result.operationVisits;
    if (checkControl(op, result)) {
        return;
    }
    auto phases = index.phasesFor(op);
    if (phases.empty()) {
        if (isa<pto::OpPipeInterface>(op) && !pto::hasOnlyDescriptorEffects(op)) {
            result.note(op, "pipe-phase", "pipe-bearing operation has no retained execution phase");
            return;
        }
        // alloc_tile is an existing declarative allocation without a purity
        // interface. This exception does not make other unknown operations pure.
        if (!isMemoryEffectFree(op) && !pto::hasOnlyDescriptorEffects(op) && !isa<pto::AllocTileOp>(op)) {
            result.note(
                op, "effects-or-phase", "operation lacks proved purity/descriptor semantics or a modeled effect phase");
        }
        return;
    }
    if (phases.size() != 1 || phases.front()->macroOpInstanceId >= 0) {
        result.note(op, "phase-cut", "compound/envelope phases need a qualified shared protocol cut");
        return;
    }
    checkPayload(op, input, *phases.front(), architecture, core, result);
}
inline ContractCheck checkContracts(
    mlir::func::FuncOp function, const mlir::pto::SyncInput& input, const mlir::pto::frontiersynch::PhaseIndex& index,
    llvm::StringRef architecture, mlir::pto::SyncPhysicalCore core)
{
    using namespace mlir;
    ContractCheck result;
    for (const auto& effect : input.accesses().effects()) {
        ++result.visibilityInventoryVisits;
        result.possibleGmWrites |=
            effect.mode == pto::SyncAccessMode::Write &&
            (effect.memory->scope == pto::AddressSpace::GM || effect.memory->scope == pto::AddressSpace::Zero);
    }
    function.walk([&](Operation* op) { checkOperation(op, input, index, architecture, core, result); });
    llvm::SmallVector<Operation*> roots;
    for (auto& op : function.front().without_terminator()) {
        roots.push_back(&op);
    }
    auto readiness = pto::frontiersynch::CompletionReadiness::qualify(roots, index);
    if (succeeded(readiness)) {
        result.readinessVisits = readiness->operationVisits();
        result.readinessUses = readiness->useVisits();
    }
    if (failed(readiness) || readiness->size()) {
        result.note(
            function, "result-readiness",
            "non-storage result/control readiness needs an additional qualified completion cut");
    }
    return result;
}
} // namespace current_cut
#endif
