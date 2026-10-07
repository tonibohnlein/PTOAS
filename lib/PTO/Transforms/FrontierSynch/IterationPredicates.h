// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_FRONTIERSYNCH_ITERATIONPREDICATES_H
#define PTO_FRONTIERSYNCH_ITERATIONPREDICATES_H
#include "CountedLoop.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
namespace mlir::pto::frontiersynch {
// A placeholder denotes a *particular evaluation*, not the static SSA value.
// This distinction is essential when the same branch changes across iterations.
class IterationPredicates {
public:
    using Id = RegionExpressions::Id;
    IterationPredicates(scf::ForOp loop, RegionExpressions& e, ArrayRef<const CompoundInstanceElement*> phases)
        : loop(loop), e(e), dominance(loop->getParentOfType<func::FuncOp>())
    {
        for (auto* phase : phases) {
            forbidden.insert(phase->elementOp);
        }
    }
    RegionExpressions::Id at(Value value, Id ordinal)
    {
        auto key = std::make_pair(value, ordinal);
        auto found = identities.find(key);
        if (found != identities.end()) {
            return found->second;
        }
        auto argument = symbols.addArgument(value.getType(), loop.getLoc());
        auto id = e.input(argument);
        identities.try_emplace(key, id);
        recipes.try_emplace(argument, std::make_pair(value, ordinal));
        return id;
    }
    LogicalResult recover(Id root, OpBuilder& builder, Operation* cut, RegionExpressions::CutEmission& context)
    {
        for (auto [id, value] : e.referencedInputs(root)) {
            auto found = recipes.find(value);
            if (found == recipes.end() || context.values.count(id)) {
                continue;
            }
            auto [original, ordinal] = found->second;
            auto domain = CountedLoop::get(loop);
            if (!domain) {
                return failure();
            }
            auto current =
                e.div(e.sub(e.input(loop.getInductionVar()), e.input(loop.getLowerBound())), e.constant(domain->step));
            if (ordinal == current) {
                // Reuse the actual evaluated predicate when this recipe asks
                // for the current iteration. This may include a memory-derived
                // value already available at the cut; it is never speculated.
                auto actual = e.emitContextual(e.input(original), builder, cut, context);
                if (failed(actual)) {
                    return failure();
                }
                context.values[id] = *actual;
                context.cofactors[id] = id;
                continue;
            }
            auto iv = e.input(loop.getLowerBound()), multiple = e.constant(0), factor = ordinal;
            auto step = static_cast<uint64_t>(domain->step);
            while (step) {
                if (step & 1) {
                    multiple = e.add(multiple, factor);
                }
                step >>= 1;
                if (step) {
                    factor = e.add(factor, factor);
                }
            }
            iv = e.add(iv, multiple);
            auto induction = e.emitContextual(iv, builder, cut, context);
            if (failed(induction)) {
                return failure();
            }
            IRMapping mapping;
            mapping.map(loop.getInductionVar(), *induction);
            // Values inside the loop must be replayed even if their current
            // evaluation dominates the cut: this recipe may ask for i+d.
            SmallVector<std::pair<Value, bool>> work{{original, false}};
            DenseSet<Operation*> scheduled;
            DenseSet<Value> complete;
            SmallVector<Operation*> order;
            while (!work.empty()) {
                auto [v, done] = work.pop_back_val();
                if (mapping.contains(v) || complete.contains(v)) {
                    continue;
                }
                auto* op = v.getDefiningOp();
                auto* owner = v.getParentRegion() ? v.getParentRegion()->getParentOp() : nullptr;
                if (owner && !loop->isProperAncestor(owner) && owner != loop && dominance.properlyDominates(v, cut)) {
                    mapping.map(v, v);
                    continue;
                }
                if (!mlir::pto::detail::canReplayScalar(op) || forbidden.contains(op)) {
                    error = "iteration-indexed guard needs an unavailable value or non-replayable operation";
                    return failure();
                }
                if (done) {
                    complete.insert(v);
                    if (scheduled.insert(op).second) {
                        order.push_back(op);
                    }
                } else {
                    work.push_back({v, true});
                    for (auto operand : op->getOperands()) {
                        work.push_back({operand, false});
                    }
                }
            }
            for (auto* op : order) {
                auto* clone = builder.clone(*op, mapping);
                // Off-domain window coordinates are masked. Drop poison-only
                // overflow promises there; retained machine integer bits agree
                // on every defined source evaluation, including narrow integers.
                clone->removeAttr("overflowFlags");
            }
            auto recovered = mapping.lookupOrNull(original);
            if (!recovered || !recovered.getType().isInteger(1)) {
                return failure();
            }
            context.values[id] = recovered;
            context.cofactors[id] = id;
        }
        return success();
    }
    std::string error;

private:
    scf::ForOp loop;
    RegionExpressions& e;
    DominanceInfo dominance;
    Block symbols;
    DenseSet<Operation*> forbidden;
    DenseMap<std::pair<Value, Id>, Id> identities;
    DenseMap<Value, std::pair<Value, Id>> recipes;
};
} // namespace mlir::pto::frontiersynch
#endif
