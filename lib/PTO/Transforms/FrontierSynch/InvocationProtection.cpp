// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared structured accumulator analysis. Summaries describe one body visit;
// uniform children preserve a chain, while nonuniform loops scope local facts
// to a single visit. No loop iterations or branch valuations are enumerated.
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/IR/PTO.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
uint64_t StructuredProtection::lookup(const CompoundInstanceElement* phase) const
{
    auto found = facts.find(phase);
    return found != facts.end() && !found->second.scope ? found->second.group | invocationProtectionBit : 0;
}
uint64_t StructuredProtection::within(const CompoundInstanceElement* phase,
                                     ArrayRef<scf::ForOp> enclosing) const
{
    auto found = facts.find(phase);
    if (found == facts.end()) { return 0; }
    if (found->second.scope && !llvm::any_of(enclosing, [&](scf::ForOp loop) {
            return loop.getOperation() == found->second.scope;
        })) { return 0; }
    return found->second.group | invocationProtectionBit;
}
uint64_t StructuredProtection::at(const CompoundInstanceElement* phase) const
{
    SmallVector<scf::ForOp> enclosing;
    for (auto* parent = phase->elementOp->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) { enclosing.push_back(loop); }
    }
    return within(phase, enclosing);
}
uint64_t StructuredProtection::inLoop(const CompoundInstanceElement* phase, scf::ForOp loop) const
{
    auto found = facts.find(phase);
    if (found == facts.end()) { return 0; }
    const auto fact = found->second;
    if (!fact.scope || fact.scope->isProperAncestor(loop)) { return fact.group | invocationProtectionBit; }
    return fact.scope == loop.getOperation() ? fact.group : 0;
}
namespace {
struct Summary {
    bool valid = true;
    Type type;
    SmallVector<uint32_t> atoms;
    std::optional<std::size_t> matrixEffect;
    Value symbolicBuffer;
    SmallVector<std::size_t> otherEffects;
};
struct Leaf {
    const CompoundInstanceElement* phase = nullptr;
    ExplicitEffects occurrence;
    SmallVector<uint32_t> atoms;
    Value symbolicBuffer;
};
class StructuredBuilder {
public:
    explicit StructuredBuilder(const SyncStorageEffects& model) : storage(model)
    {
        for (const auto& effect : storage.effects()) {
            auto& list = phases[effect.phase->elementOp];
            if (!llvm::is_contained(list, effect.phase)) { list.push_back(effect.phase); }
        }
    }
    StructuredProtection run(func::FuncOp function)
    {
        summarize(*function.getOperation());
        walk(function.front(), nullptr);
        return std::move(result);
    }
private:
    const SyncStorageEffects& storage;
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> phases;
    DenseMap<Operation*, Leaf> leaves;
    DenseMap<Operation*, Summary> summaries;
    DenseMap<Value, uint32_t> symbolicAtoms;
    HardwareProtectionBuilder state;
    StructuredProtection result;
    std::optional<std::size_t> activeEffect;

    bool metadata(Operation& op) const
    {
        if (isa<SetValidShapeOp>(op)) { return false; }
        return isa<AllocTileOp, AllocMultiTileOp, MultiTileGetOp, SubViewOp>(op) ||
            op.hasTrait<OpTrait::IsTerminator>() || isMemoryEffectFree(&op);
    }
    bool accesses(Leaf& leaf)
    {
        const auto effects = storage.effects();
        for (auto id : storage.effectsFor(leaf.phase)) {
            const auto& effect = effects[id];
            if (!effect.memory || effect.memory->scope == AddressSpace::Zero) { return false; }
            if (effect.memory->scope != AddressSpace::ACC) { continue; }
            SmallVector<uint32_t> atoms;
            if (effect.rangesMaterialized && !effect.regions.empty() && !effect.selection &&
                llvm::all_of(effect.regions, [](const auto& region) {
                    return !region.base && region.symbols.empty();
                })) {
                for (auto cell : effect.cells) {
                    if (cell > UINT32_MAX) { return false; }
                    atoms.push_back(static_cast<uint32_t>(cell));
                }
            } else {
                if (!effect.sharedProvenanceComplete || effect.selection || !effect.memory->baseBuffer ||
                    storage.cells().size() + symbolicAtoms.size() >= UINT32_MAX) { return false; }
                leaf.symbolicBuffer = effect.memory->baseBuffer;
                auto atom = symbolicAtoms.try_emplace(effect.memory->baseBuffer,
                    static_cast<uint32_t>(storage.cells().size() + symbolicAtoms.size())).first->second;
                atoms.push_back(atom);
            }
            for (auto atom : atoms) {
                leaf.atoms.push_back(atom);
                leaf.occurrence.accesses.push_back({atom, effect.mode == SyncAccessMode::Read,
                                                    effect.mode == SyncAccessMode::Write});
            }
        }
        llvm::sort(leaf.atoms);
        leaf.atoms.erase(std::unique(leaf.atoms.begin(), leaf.atoms.end()), leaf.atoms.end());
        return true;
    }
    Summary leafSummary(Operation& op)
    {
        Summary summary;
        auto found = phases.find(&op);
        if (found == phases.end()) { summary.valid = metadata(op); return summary; }
        if (found->second.size() != 1) { summary.valid = false; return summary; }
        Leaf leaf;
        leaf.phase = found->second.front();
        leaf.occurrence.pipe = static_cast<uint32_t>(leaf.phase->kPipeValue);
        summary.valid = accesses(leaf);
        auto matrix = matrixProtectionInfo(&op);
        if (matrix && summary.valid && !leaf.atoms.empty() &&
            leaf.occurrence.pipe == static_cast<uint32_t>(PipelineType::PIPE_M)) {
            summary.type = matrix->accumulator;
            summary.atoms = leaf.atoms;
            summary.symbolicBuffer = leaf.symbolicBuffer;
            for (auto id : storage.effectsFor(leaf.phase)) {
                if (storage.effects()[id].memory->scope == AddressSpace::ACC) { summary.matrixEffect = id; break; }
            }
        } else {
            if (leaf.occurrence.pipe == static_cast<uint32_t>(PipelineType::PIPE_M)) { summary.valid = false; }
            for (auto id : storage.effectsFor(leaf.phase)) { summary.otherEffects.push_back(id); }
        }
        leaves[&op] = std::move(leaf);
        return summary;
    }
    void merge(Summary& into, const Summary& child)
    {
        into.valid &= child.valid;
        if (child.type) {
            if (into.type && (into.type != child.type || into.atoms != child.atoms)) { into.valid = false; }
            if (!into.type) { into.type = child.type; into.atoms = child.atoms; into.matrixEffect = child.matrixEffect;
                into.symbolicBuffer = child.symbolicBuffer; }
        }
        llvm::append_range(into.otherEffects, child.otherEffects);
    }
    Summary summarize(Operation& op)
    {
        Summary summary;
        if (!op.getNumRegions()) { summary = leafSummary(op); }
        else if (!isa<scf::ForOp, scf::IfOp, func::FuncOp>(op)) { summary.valid = false; }
        else {
            for (auto& region : op.getRegions()) {
                if (!region.empty() && !region.hasOneBlock()) { summary.valid = false; continue; }
                for (auto& block : region) {
                    for (auto& child : block) { merge(summary, summarize(child)); }
                }
            }
            // A symbolic buffer defined inside a repeated body may select a
            // different physical resource on each visit. Its SSA identity only
            // proves equality within that visit, even without a selection map.
            if (isa<scf::ForOp>(op) && summary.symbolicBuffer) {
                auto* owner = summary.symbolicBuffer.getDefiningOp();
                if (auto arg = dyn_cast<BlockArgument>(summary.symbolicBuffer)) {
                    owner = arg.getOwner()->getParentOp();
                }
                if (!owner || owner == &op || op.isProperAncestor(owner)) { summary.valid = false; }
            }
            if (summary.matrixEffect && llvm::any_of(summary.otherEffects, [&](auto id) {
                    return storage.mayOverlap(*summary.matrixEffect, id);
                })) { summary.valid = false; }
        }
        summaries[&op] = summary;
        return summary;
    }
    void observe(Operation& op, Operation* scope)
    {
        auto found = leaves.find(&op);
        if (found == leaves.end()) {
            if (!metadata(op)) { state.endScope(); }
            return;
        }
        auto& leaf = found->second;
        if (!summaries.lookup(&op).valid) { state.endScope(); return; }
        // An unresolved alias to the active resource must also break a chain.
        // Uniform summaries discharge such interference before flattening.
        if (activeEffect && !matrixProtectionInfo(&op) &&
            llvm::any_of(storage.effectsFor(leaf.phase), [&](auto id) {
                return storage.mayOverlap(*activeEffect, id);
            })) { state.endScope(); activeEffect.reset(); }
        state.observe(&op, leaf.occurrence, leaf.atoms);
        if (auto effect = summaries.lookup(&op).matrixEffect) { activeEffect = effect; }
        else if (!state.currentGroup()) { activeEffect.reset(); }
        if (!leaf.occurrence.accesses.empty() && leaf.occurrence.accesses.front().protectionGroup) {
            result.facts[leaf.phase] = {leaf.occurrence.accesses.front().protectionGroup, scope};
        }
    }
    void uniform(Operation& op, Operation* scope)
    {
        if (!op.getNumRegions()) { observe(op, scope); return; }
        for (auto& region : op.getRegions()) {
            for (auto& block : region) {
                for (auto& child : block) { uniform(child, scope); }
            }
        }
    }
    void walk(Block& block, Operation* scope)
    {
        for (auto& op : block) {
            if (!op.getNumRegions()) { observe(op, scope); continue; }
            const auto summary = summaries.lookup(&op);
            if (summary.valid) { uniform(op, scope); continue; }
            state.endScope();
            if (isa<scf::ForOp, scf::IfOp>(op)) {
                for (auto& region : op.getRegions()) {
                    if (region.empty() || !region.hasOneBlock()) { continue; }
                    walk(region.front(), isa<scf::ForOp>(op) ? &op : scope);
                    state.endScope();
                }
            }
        }
    }
};
} // namespace
StructuredProtection structuredProtection(const SyncStorageEffects& storage)
{
    if (storage.effects().empty()) { return {}; }
    auto function = storage.effects().front().phase->elementOp->getParentOfType<func::FuncOp>();
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock()) { return {}; }
    bool rebound = false;
    function.walk([&](TAssignOp) { rebound = true; });
    if (rebound) { return {}; }
    return StructuredBuilder(storage).run(function);
}
} // namespace mlir::pto::frontiersynch
