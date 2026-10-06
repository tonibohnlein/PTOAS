// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Prove a fixed accumulator chain once over structured control, without
// unfolding trips. Region analysis consumes the proof before demand reduction.
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace mlir::pto::frontiersynch {
namespace {
class InvocationBuilder {
public:
    explicit InvocationBuilder(const SyncStorageEffects& model) : storage(model)
    {
        for (const auto& effect : storage.effects()) {
            auto& list = phases[effect.phase->elementOp];
            if (!llvm::is_contained(list, effect.phase)) { list.push_back(effect.phase); }
        }
    }
    InvocationProtection run(func::FuncOp function)
    {
        HardwareProtectionBuilder state;
        InvocationProtection groups;
        for (auto& op : function.front()) {
            if (op.getNumRegions()) {
                InvocationProtection pending;
                if (state.currentGroup() && preserve(op, state, pending, 0)) {
                    for (const auto& entry : pending) { groups.insert(entry); }
                } else { state.endScope(); }
            } else {
                observe(op, state, groups);
            }
        }
        return groups;
    }
private:
    const SyncStorageEffects& storage;
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> phases;

    bool metadata(Operation& op) const
    {
        // Descriptor construction has no physical access; mutations are not
        // transparent. Reuse the same metadata distinction as recognition.
        if (isa<SetValidShapeOp>(op)) { return false; }
        return isa<AllocTileOp, AllocMultiTileOp, MultiTileGetOp, SubViewOp>(op) ||
            op.hasTrait<OpTrait::IsTerminator>() || isMemoryEffectFree(&op);
    }
    bool accesses(const CompoundInstanceElement* phase, ExplicitEffects& occurrence,
                  SmallVectorImpl<uint32_t>& atoms) const
    {
        occurrence.pipe = static_cast<uint32_t>(phase->kPipeValue);
        for (auto id : storage.effectsFor(phase)) {
            const auto& effect = storage.effects()[id];
            if (!effect.memory || effect.memory->scope == AddressSpace::Zero) { return false; }
            if (effect.memory->scope != AddressSpace::ACC) { continue; }
            // A finite union of possible banks is not a fixed accumulator.
            // Require materialized maps with no runtime symbols or selections.
            if (!effect.rangesMaterialized || effect.regions.empty() || effect.selection ||
                llvm::any_of(effect.regions, [](const auto& region) {
                    return region.base || !region.symbols.empty();
                })) { return false; }
            for (auto cell : effect.cells) {
                if (cell > UINT32_MAX) { return false; }
                auto atom = static_cast<uint32_t>(cell);
                atoms.push_back(atom);
                occurrence.accesses.push_back({atom, effect.mode == SyncAccessMode::Read,
                                               effect.mode == SyncAccessMode::Write});
            }
        }
        return true;
    }
    void observe(Operation& op, HardwareProtectionBuilder& state, InvocationProtection& groups)
    {
        auto found = phases.find(&op);
        if (found == phases.end()) {
            if (!metadata(op)) { state.endScope(); }
            return;
        }
        // Macro operations with several phases need a phase-specific hardware
        // contract; do not interpret the enclosing operation as one MMAD.
        if (found->second.size() != 1) { state.endScope(); return; }
        const auto* phase = found->second.front();
        ExplicitEffects occurrence;
        SmallVector<uint32_t> atoms;
        if (!accesses(phase, occurrence, atoms)) { state.endScope(); return; }
        state.observe(&op, occurrence, atoms);
        if (!occurrence.accesses.empty() && occurrence.accesses.front().protectionGroup) {
            groups[phase] = invocationProtectionBit | occurrence.accesses.front().protectionGroup;
        }
    }
    bool preserve(Operation& op, const HardwareProtectionBuilder& entry,
                  InvocationProtection& groups, unsigned depth)
    {
        // Every arm/body must be the identity on the incoming protected chain.
        // This proves zero trips, arbitrarily many visits and either branch.
        // Do not establish a new invocation-wide group inside repeated control.
        constexpr unsigned maxDepth = 64;
        if (depth >= maxDepth || !isa<scf::ForOp, scf::IfOp>(op)) { return false; }
        for (auto& region : op.getRegions()) {
            if (region.empty()) { continue; }
            if (!region.hasOneBlock()) { return false; }
            auto state = entry;
            for (auto& child : region.front()) {
                if (child.getNumRegions()) {
                    if (!preserve(child, state, groups, depth + 1)) { return false; }
                } else {
                    observe(child, state, groups);
                    if (state.currentGroup() != entry.currentGroup()) { return false; }
                }
            }
        }
        return true;
    }
};
} // namespace
InvocationProtection invocationProtectionGroups(const SyncStorageEffects& storage)
{
    if (storage.effects().empty()) { return InvocationProtection{}; }
    auto function = storage.effects().front().phase->elementOp->getParentOfType<func::FuncOp>();
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock()) { return InvocationProtection{}; }
    // TASSIGN mutates the original descriptor even when later uses retain its
    // old SSA identity. Until shared geometry models rebinding, no initializer
    // in this function can certify a fixed physical accumulator.
    bool rebound = false;
    function.walk([&](TAssignOp) { rebound = true; });
    if (rebound) { return InvocationProtection{}; }
    return InvocationBuilder(storage).run(function);
}
} // namespace mlir::pto::frontiersynch
