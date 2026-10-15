// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structural singleton selection proof shared by SSA, arena and counted views.
#ifndef PTO_FRONTIER_CLEANUPSELECTIONPROOF_H
#define PTO_FRONTIER_CLEANUPSELECTIONPROOF_H
#include "PTO/Transforms/FrontierSynch/Shared/EndpointCutChoices.h"
#include "CountedLoop.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Dominance.h"
namespace mlir::pto::frontiersynch::detail {
enum class CleanupPredicateBinary { And, Equal, Subtract, Divide };
template<class View>
void cleanupConjuncts(const View& view, typename View::Ref root, SmallVectorImpl<typename View::Ref>& atoms)
{
    using Ref = typename View::Ref;
    SmallVector<Ref> work{root};
    llvm::DenseSet<Ref> seen;
    while (!work.empty()) {
        auto ref = work.pop_back_val();
        if (!view.valid(ref) || !seen.insert(ref).second) { continue; }
        auto both = view.binary(ref, CleanupPredicateBinary::And);
        if (!both) { both = view.masked(ref); }
        if (both) { work.push_back(both->first); work.push_back(both->second); }
        else { atoms.push_back(ref); }
    }
}
template<class View>
bool cleanupOrdinal(const View& view, typename View::Ref ref, scf::ForOp loop)
{
    // Unsigned distance/positive step preserves the exact visit coordinate,
    // including signed counted-loop intervals crossing zero.
    auto iv = view.get(loop.getInductionVar());
    if (ref == iv) { return true; }
    if (auto division = view.binary(ref, CleanupPredicateBinary::Divide)) {
        auto domain = CountedLoop::get(loop);
        if (!domain || view.number(division->second) != static_cast<uint64_t>(domain->step)) { return false; }
        ref = division->first;
        if (ref == iv && view.number(view.get(loop.getLowerBound())) == 0) { return true; }
    }
    auto difference = view.binary(ref, CleanupPredicateBinary::Subtract);
    if (!difference || difference->first != iv) { return false; }
    auto lower = view.get(loop.getLowerBound());
    return difference->second == lower ||
        (view.number(difference->second) && view.number(difference->second) == view.number(lower));
}
template<class View>
bool cleanupInvariant(const View& view, typename View::Ref root, Operation* outer, DominanceInfo& dominance)
{
    using Ref = typename View::Ref;
    SmallVector<Ref> work{root};
    llvm::DenseSet<Ref> seen;
    while (!work.empty()) {
        auto ref = work.pop_back_val();
        if (!view.valid(ref)) { return false; }
        if (!seen.insert(ref).second) { continue; }
        if (auto original = view.original(ref)) {
            auto* region = original.getParentRegion();
            if (region && region->getParentOp() && dominance.properlyDominates(original, outer)) { continue; }
        }
        if (!view.appendOperands(ref, work)) { return false; }
    }
    return true;
}
template<class View>
bool singletonCleanupConsumer(Operation* cut, const View& view, typename View::Ref predicate,
    func::FuncOp function)
{
    using Ref = typename View::Ref;
    if (view.valid(predicate) && !view.boolean(predicate)) { return false; }
    SmallVector<scf::ForOp> loops;
    SmallVector<Ref> atoms;
    cleanupConjuncts(view, predicate, atoms);
    auto* descendant = cut;
    for (auto* parent = cut->getParentOp(); parent != function; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            if (!CountedLoop::get(loop)) { return false; }
            loops.push_back(loop);
        } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
            if (descendant->getParentRegion() == &branch.getThenRegion()) {
                cleanupConjuncts(view, view.get(branch.getCondition()), atoms);
            }
        } else if (!isEndpointSelectionContainer(parent)) { return false; }
        descendant = parent;
    }
    if (loops.empty()) { return true; }
    if (!view.valid(predicate)) { return false; }
    if (view.number(predicate) == 0) { return true; }
    DominanceInfo dominance(function);
    for (auto loop : loops) {
        bool selected = llvm::any_of(atoms, [&](Ref atom) {
            auto equality = view.binary(atom, CleanupPredicateBinary::Equal);
            if (!equality) { return false; }
            return (cleanupOrdinal(view, equality->first, loop) &&
                    cleanupInvariant(view, equality->second, loops.back(), dominance)) ||
                (cleanupOrdinal(view, equality->second, loop) &&
                 cleanupInvariant(view, equality->first, loops.back(), dominance));
        });
        if (!selected) { return false; }
    }
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
