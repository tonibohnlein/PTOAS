// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/IR/PTOAccessRegion.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <numeric>

namespace mlir::pto::frontiersynch {
LogicalResult PhaseIndex::build(func::FuncOp source, const SyncInput& input)
{
    return build(source, input.instructions());
}

LogicalResult PhaseIndex::build(func::FuncOp source, ArrayRef<const CompoundInstanceElement*> phases)
{
    anchorPhases.clear();
    valuePrerequisites.clear();
    prerequisites.clear();
    relevantValues.clear();
    carriedOrdinals.clear();
    dominance.invalidate();
    function = {};
    if (!source) {
        return failure();
    }
    DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>> pending;
    for (const auto* phase : phases) {
        Operation* anchor = phase->elementOp;
        if (!anchor || !anchor->getBlock() || !source->isProperAncestor(anchor)) {
            return source.emitError("shared synchronization phase has no anchor in this function");
        }
        pending[anchor].push_back(phase);
    }
    anchorPhases = std::move(pending);
    function = source;
    computeRelevance();
    computeCarriedOrdinals();
    DenseMap<Operation*, bool> carriedRelevance;
    for (const auto& entry : anchorPhases) {
        if (entry.second.size() != 1) {
            continue;
        }
        for (Value value : entry.first->getResults()) {
            traceResult(entry.second.front(), value, carriedRelevance);
        }
    }
    return success();
}

// Treat structured SSA edges as dataflow, without rewriting carried values or
// dropping their initializers. Worklist visitation handles cross-carried cycles.
void PhaseIndex::computeRelevance()
{
    SmallVector<Value> work;
    DenseSet<Operation*> expandedDefinitions;
    function.walk([&](Operation* op) {
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            work.append({loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
        } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
            work.push_back(branch.getCondition());
        } else if (isa<scf::YieldOp>(op) && isa<scf::ForOp, scf::IfOp>(op->getParentOp())) {
            // A live block argument or result requests its incoming value below.
        } else if (isa<func::ReturnOp>(op) || !phasesFor(op).empty() || !isMemoryEffectFree(op)) {
            llvm::append_range(work, op->getOperands());
        }
    });
    auto incoming = [&](scf::ForOp loop, unsigned position) {
        work.push_back(loop.getInitArgs()[position]);
        work.push_back(cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(position));
    };
    while (!work.empty()) {
        Value value = work.pop_back_val();
        if (!relevantValues.insert(value).second) { continue; }
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
            if (loop && argument.getArgNumber()) { incoming(loop, argument.getArgNumber() - 1); }
            continue;
        }
        auto result = cast<OpResult>(value);
        Operation* op = result.getOwner();
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            incoming(loop, result.getResultNumber());
        } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
            for (Region& arm : branch->getRegions()) {
                if (!arm.empty()) {
                    work.push_back(cast<scf::YieldOp>(arm.front().getTerminator())
                                       .getOperand(result.getResultNumber()));
                }
            }
        } else if (expandedDefinitions.insert(op).second) {
            llvm::append_range(work, op->getOperands());
        }
    }
}

bool PhaseIndex::hasRelevantResults(Operation* operation) const
{
    return llvm::any_of(operation->getResults(), [&](Value result) { return isRelevant(result); });
}

bool PhaseIndex::hasRelevantCarriedState(Operation* operation) const
{
    auto loop = dyn_cast<scf::ForOp>(operation);
    return loop && (hasRelevantResults(loop) || llvm::any_of(loop.getRegionIterArgs(),
        [&](Value argument) { return isRelevant(argument); }));
}

void PhaseIndex::computeCarriedOrdinals()
{
    function.walk([&](scf::ForOp loop) {
        for (auto argument : loop.getRegionIterArgs()) {
            if (!isRelevant(argument)) { continue; }
            auto* context = function.getContext();
            mlir::pto::detail::ScalarEvolution evolution(context, loop);
            auto proof = evolution.modularRecurrence(argument);
            if (!proof) { continue; }
            auto ordinal = getAffineDimExpr(0, context);
            auto scaled = mlir::pto::detail::checkedMul(ordinal, getAffineConstantExpr(proof->stride, context));
            auto expression = mlir::pto::detail::checkedAdd(getAffineConstantExpr(proof->seed, context), scaled);
            if (!expression) { continue; }
            carriedOrdinals.try_emplace(argument, CarriedOrdinalProof{expression % proof->modulus,
                static_cast<uint64_t>(proof->seed), static_cast<uint64_t>(proof->stride),
                static_cast<uint64_t>(proof->modulus)});
        }
    });
}

uint64_t PhaseIndex::CarriedOrdinalProof::period() const
{
    return modulus / std::gcd(modulus, stride);
}

uint64_t PhaseIndex::CarriedOrdinalProof::atOrdinal(uint64_t ordinal) const
{
    // The product of two uint64_t values plus a signed-64 seed fits 128
    // unsigned bits here: stride is bounded by the signed machine proof.
    auto value = APInt(128, ordinal) * APInt(128, stride) + APInt(128, seed);
    return value.urem(APInt(128, modulus)).getZExtValue();
}

const PhaseIndex::CarriedOrdinalProof* PhaseIndex::carriedOrdinal(Value argument) const
{
    auto found = carriedOrdinals.find(argument);
    return found == carriedOrdinals.end() ? nullptr : &found->second;
}

bool PhaseIndex::hasUnprovedCarriedState(Operation* operation) const
{
    auto loop = dyn_cast_or_null<scf::ForOp>(operation);
    // Final values need a separate exit/zero-trip interpretation. A formula
    // for the body argument alone does not certify an escaping loop result.
    return loop && (hasRelevantResults(loop) || llvm::any_of(loop.getRegionIterArgs(), [&](Value argument) {
        return isRelevant(argument) && !carriedOrdinal(argument);
    }));
}

void PhaseIndex::traceResult(const CompoundInstanceElement* producer, Value result,
                             DenseMap<Operation*, bool>& carriedRelevance)
{
    const auto availability = resultAvailability(*producer, result);
    if (availability == SyncResultAvailability::NonScalar) {
        return;
    }
    const bool native = availability == SyncResultAvailability::SynchronousScalar;
    SmallVector<std::pair<Value, bool>> work{{result, true}};
    DenseSet<Value> seenDirect, seenIndirect;
    auto record = [&](Operation* consumer, bool direct) {
        auto& entries = prerequisites[consumer];
        auto found = llvm::find_if(entries, [producer](const auto& entry) { return entry.producer == producer; });
        if (found == entries.end()) { entries.push_back({producer, consumer, native, direct}); }
        else { found->directSSA &= direct; }
    };
    while (!work.empty()) {
        auto [value, direct] = work.pop_back_val();
        if (!(direct ? seenDirect : seenIndirect).insert(value).second) {
            continue;
        }
        for (OpOperand& use : value.getUses()) {
            Operation* user = use.getOwner();
            if (!phasesFor(user).empty()) {
                record(user, direct);
            } else if (auto branch = dyn_cast<scf::IfOp>(user)) {
                if (!native) {
                    valuePrerequisites.insert(user);
                }
                record(user, false);
                branch.walk([&](Operation* nested) {
                    if (!phasesFor(nested).empty()) {
                        record(nested, false);
                    }
                });
            } else if (isa<scf::YieldOp>(user) && isa<scf::IfOp>(user->getParentOp())) {
                work.push_back({user->getParentOp()->getResult(use.getOperandNumber()), false});
            } else if (auto loop = dyn_cast<scf::ForOp>(user); loop && use.getOperandNumber() < 3 && native) {
                // Native scalar bounds are available before loop execution.
                // They order body users without becoming storage demands.
                record(user, false);
                loop.walk([&](Operation* nested) {
                    if (!phasesFor(nested).empty()) { record(nested, false); }
                });
            } else if (user->getNumRegions() || isa<scf::YieldOp>(user)) {
                // A carried value needs an occurrence map, not a static SSA edge.
                auto* owner = user->getNumRegions() ? user : user->getParentOp();
                if (!native || !isa<scf::ForOp>(owner)) {
                    valuePrerequisites.insert(owner);
                } else {
                    auto found = carriedRelevance.find(owner);
                    if (found == carriedRelevance.end()) {
                        found = carriedRelevance.try_emplace(owner, hasRelevantCarriedState(owner)).first;
                    }
                    if (found->second) { valuePrerequisites.insert(owner); }
                }
            } else if (isa<func::ReturnOp>(user)) {
                if (!native) {
                    valuePrerequisites.insert(user);
                }
            } else if (hasOnlyDescriptorEffects(user) || isPreservedSyncProtocol(user)) {
                // Scalar descriptor updates execute natively. A result from an
                // asynchronous pipe still needs an explicit completion mapping.
                record(user, false);
                if (!native) { valuePrerequisites.insert(user); }
            } else if (isMemoryEffectFree(user)) {
                for (Value next : user->getResults()) { work.push_back({next, direct}); }
            } else {
                valuePrerequisites.insert(user);
            }
        }
    }
}

ArrayRef<ValuePrerequisite> PhaseIndex::prerequisitesFor(Operation* operation) const
{
    auto found = prerequisites.find(operation);
    return found == prerequisites.end() ? ArrayRef<ValuePrerequisite>{} : found->second;
}

PhasePrerequisiteEdges PhaseIndex::mapPrerequisites(ArrayRef<const CompoundInstanceElement*> phases) const
{
    PhasePrerequisiteEdges output;
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    if (phases.size() > UINT32_MAX) {
        output.error = "value prerequisite identity overflow";
        return output;
    }
    for (uint32_t i = 0; i < phases.size(); ++i) {
        if (!positions.try_emplace(phases[i], i).second) {
            output.error = "repeated scalar producer needs an occurrence mapping";
            return output;
        }
    }
    for (uint32_t target = 0; target < phases.size(); ++target) {
        for (const auto& requirement : prerequisitesFor(phases[target]->elementOp)) {
            auto source = positions.find(requirement.producer);
            if (source == positions.end()) {
                continue;
            }
            if (source->second >= target) {
                output.error = "value prerequisite needs a forward occurrence mapping";
                return output;
            }
            auto& edges = requirement.native ? output.native : output.demands;
            edges.push_back({source->second, target});
        }
    }
    return output;
}

bool PhaseIndex::needsValuePrerequisite(Operation* operation) const
{
    return valuePrerequisites.contains(operation);
}

bool PhaseIndex::contains(Operation* operation) const
{
    return function && operation && function->isProperAncestor(operation);
}

ArrayRef<const CompoundInstanceElement*> PhaseIndex::phasesFor(Operation* anchor) const
{
    auto found = anchorPhases.find(anchor);
    if (found == anchorPhases.end()) {
        return {};
    }
    return found->second;
}

SmallVector<Region*> PhaseIndex::controlPath(const CompoundInstanceElement& phase) const
{
    SmallVector<Region*> path;
    if (!contains(phase.elementOp) || !llvm::is_contained(phasesFor(phase.elementOp), &phase)) {
        return path;
    }
    for (Region* region = phase.elementOp->getParentRegion(); region != &function->getRegion(0);
         region = region->getParentRegion()) {
        if (isa<RegionBranchOpInterface>(region->getParentOp())) {
            path.push_back(region);
        }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

FailureOr<SmallVector<const CompoundInstanceElement*>> PhaseIndex::explicitSequence(Block& block) const
{
    Region* region = block.getParent();
    if (!function || !region || !function->getRegion(0).isAncestor(region) || !region->hasOneBlock() ||
        !dominance.hasSSADominance(&block)) {
        return failure();
    }
    SmallVector<const CompoundInstanceElement*> sequence;
    for (Operation& operation : block) {
        auto phases = phasesFor(&operation);
        if (operation.getNumRegions() != 0 || phases.size() > 1) {
            return failure();
        }
        llvm::append_range(sequence, phases);
    }
    return sequence;
}

bool PhaseIndex::valueAvailable(Value value, Operation* anchor, Boundary boundary) const
{
    if (!contains(anchor) || !value || !value.getParentRegion() ||
        !function->getRegion(0).isAncestor(value.getParentRegion()) ||
        !dominance.hasSSADominance(anchor->getBlock())) {
        return false;
    }
    if (boundary == Boundary::After) {
        if (anchor->hasTrait<OpTrait::IsTerminator>()) {
            return false;
        }
        anchor = anchor->getNextNode();
    }
    return anchor && dominance.properlyDominates(value, anchor);
}
} // namespace mlir::pto::frontiersynch
