// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
// q=1 syntax check over shared effects. No instruction-specific footprint rules.
#include "SequenceAnalysisInternal.h"
#include "CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
namespace mlir::pto::frontiersynch {
namespace {
class Invariance {
public:
    Invariance(scf::ForOp loop, const PhaseIndex& index) : loop(loop), index(index) {}
    bool value(Value value)
    {
        if (!value) { return true; }
        if (auto found = memo.find(value); found != memo.end()) { return found->second; }
        // Mark before traversal: evolving/cyclic SSA is not an invariance proof.
        memo[value] = false;
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto* owner = argument.getOwner()->getParentOp();
            if (!owner || owner == loop) { return false; }
            if (!loop->isProperAncestor(owner)) { return memo[value] = true; }
            auto inner = dyn_cast<scf::ForOp>(owner);
            return memo[value] = inner && argument == inner.getInductionVar();
        }
        auto* definition = value.getDefiningOp();
        if (!definition) { return false; }
        if (!loop->isProperAncestor(definition)) { return memo[value] = true; }
        if (definition->getNumRegions() || !index.phasesFor(definition).empty()) { return false; }
        // Restrict syntactic scalar normalization to deterministic arithmetic.
        // This is not an operation-effect registry: physical effects stay shared.
        auto dialect = definition->getName().getDialectNamespace();
        if ((dialect != "arith" && dialect != "index") || !isMemoryEffectFree(definition) ||
            !isSpeculatable(definition)) { return false; }
        for (auto operand : definition->getOperands()) { if (!this->value(operand)) { return false; } }
        return memo[value] = true;
    }
    bool descriptor(Operation* operation)
    {
        if (operation->getNumRegions() || !index.phasesFor(operation).empty()) { return true; }
        auto storage = [](Type type) {
            return isa<TileBufType, MultiTileBufType, PtrType, MemRefType>(type);
        };
        if (!llvm::any_of(operation->getOperandTypes(), storage) &&
            !llvm::any_of(operation->getResultTypes(), storage)) { return true; }
        // Metadata can change a handle in place even if its result is unused.
        // Check its scalar geometry inputs, not an instruction-name whitelist.
        for (auto operand : operation->getOperands()) {
            if (operand.getType().isIntOrIndex() && !value(operand)) { return false; }
        }
        return true;
    }
    bool region(const SyncAccessRegion& region)
    {
        if (!value(region.base)) { return false; }
        bool valid = true;
        auto check = [&](AffineExpr expression) {
            expression.walk([&](AffineExpr term) {
                if (auto symbol = dyn_cast<AffineSymbolExpr>(term)) {
                    valid &= symbol.getPosition() < region.symbols.size() &&
                             value(region.symbols[symbol.getPosition()]);
                }
            });
        };
        check(region.byteOffset);
        for (auto extent : region.extents) { check(extent); }
        return valid;
    }
private:
    scf::ForOp loop;
    const PhaseIndex& index;
    DenseMap<Value, bool> memo;
};
} // namespace
bool SequenceAnalysisState::repeatedChild(const StructureNode& node, Expr trips)
{
    auto unavailable = [&](const std::string& reason) {
        if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
        repeatedAttempt += reason;
        return false;
    };
    auto loop = dyn_cast<scf::ForOp>(node.anchor);
    if (!loop || index.hasUnprovedCarriedState(loop) || node.children.size() != 1) {
        return unavailable("q1 repeat recognition: single body with proved carried state required");
    }
    if (llvm::any_of(input->instructions(), [&](auto* phase) {
            return phase->macroOpInstanceId >= 0 && loop->isProperAncestor(phase->elementOp);
        })) {
        return unavailable("repeated macro envelope and hidden-event adapter not implemented yet");
    }
    bool nested = false, uniform = true;
    Invariance invariant(loop, index);
    loop.getBody()->walk([&](Operation* operation) {
        uniform &= invariant.descriptor(operation);
        if (auto inner = dyn_cast<scf::ForOp>(operation)) {
            nested = true;
            uniform &= invariant.value(inner.getLowerBound()) && invariant.value(inner.getUpperBound()) &&
                       invariant.value(inner.getStep());
        } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            uniform &= invariant.value(branch.getCondition());
        }
    });
    if (!nested) { return unavailable("q1 repeat recognition: no nested body"); }
    if (!uniform) { return unavailable("q1 repeat recognition: control or descriptor invariance unavailable"); }
    bool evolving = false;
    for (const auto& effect : input->accesses().effects()) {
        if (!effect.phase || !loop->isProperAncestor(effect.phase->elementOp)) { continue; }
        if (effect.selection && !invariant.value(effect.selection->selector)) {
            return unavailable("q1 repeat recognition: storage selection varies with outer visit");
        }
        for (const auto& region : effect.regions) { evolving |= !invariant.region(region); }
    }
    if (evolving) {
        bool numericalDomains = true;
        loop.getBody()->walk([&](scf::ForOp inner) {
            auto domain = CountedLoop::get(inner);
            numericalDomains &= domain && expressions.constantValue(domain->trips(expressions)).has_value();
        });
        if (numericalDomains) {
            // Prefer a finite whole-region representation over speculative
            // symbolic child selectors. This is one nonrecursive request with
            // the ordinary arithmetic class check and an explicit output cost.
            // A deferred finite export leaves all compact/symbolic routes open.
            std::string diagnostic;
            const auto id = static_cast<std::size_t>(&node - program->nodes.data());
            auto finite = resolveOriginal.finiteArithmetic ? resolveOriginal.finiteArithmetic(id, diagnostic) :
                analyzeFiniteArithmeticRegionWithProfiles({function, loop}, index, *input, arena, diagnostic,
                    program->regionalArithmeticProfiles);
            if (succeeded(finite)) {
                Child child;
                child.regional = std::move(*finite);
                child.anchors = child.regional.anchors;
                children.push_back(std::move(child));
                return true;
            }
        }
    }
    std::string diagnostic;
    auto analyzed = originalRegion(node.children.front(), diagnostic);
    if (failed(analyzed)) { return unavailable("q1 repeat body interface: " + diagnostic); }
    auto body = std::move(*analyzed);
    if (evolving || !body.deferredAccessBoundary.empty()) {
        auto exported = body;
        exported.accessBoundary.insert(exported.accessBoundary.end(), exported.deferredAccessBoundary.begin(),
                                       exported.deferredAccessBoundary.end());
        exported.deferredAccessBoundary.clear();
        auto storage = recognizeRepeatedStorage(exported, loop, trips);
        if (storage.storage) {
            auto repeated = repeatEvolvingRegion(function, std::move(storage.storage));
            if (!repeated.error.empty()) {
                return unavailable("evolving repeat interface: " + repeated.error);
            }
            Child child;
            child.regional = std::move(repeated.regional);
            child.anchors = child.regional.anchors;
            children.push_back(std::move(child));
            return true;
        }
        unavailable("evolving repeat recognition: " + storage.error);
        if (evolving) { return false; }
    }
    // A child may discharge a streaming effect within one visit. A write can
    // still conflict with itself after re-entry; it needs actual storage selectors.
    DenseSet<std::size_t> exported;
    for (const auto& access : body.accessBoundary) { exported.insert(access.effect); }
    for (const auto& payload : body.anchors) {
        for (auto effect : input->accesses().effectsFor(payload.phase)) {
            if (input->accesses().effects()[effect].mode == SyncAccessMode::Write && !exported.count(effect)) {
                return unavailable("q1 repeat interface: discharged writer lacks outer re-entry selectors");
            }
        }
    }
    auto repeated = repeatInvariantRegion(function, loop, std::move(body), trips);
    if (!repeated.error.empty()) {
        return unavailable("q1 repeat interface: " + repeated.error);
    }
    Child child;
    child.regional = std::move(repeated.regional);
    child.anchors = child.regional.anchors;
    children.push_back(std::move(child));
    return true;
}
} // namespace mlir::pto::frontiersynch
