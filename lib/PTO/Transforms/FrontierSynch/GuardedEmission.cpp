// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Materialize shared Boolean retention circuits only at qualified endpoint cuts.
#include "DirectEmissionInternal.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
namespace mlir::pto::frontiersynch {
namespace {
Value compilePredicate(
    Predicate id, ArrayRef<PredicateNode> nodes, IRMapping& mapping, OpBuilder& builder, Location loc,
    DenseMap<Predicate, Value>& cache)
{
    auto found = cache.find(id);
    if (found != cache.end()) {
        return found->second;
    }
    const auto& node = nodes[id];
    Value result;
    if (node.kind == PredicateKind::Atom) {
        result = mapping.lookup(node.condition);
    } else if (node.kind == PredicateKind::False || node.kind == PredicateKind::True) {
        result = builder.create<arith::ConstantIntOp>(loc, node.kind == PredicateKind::True, 1);
    } else {
        auto a = compilePredicate(node.first, nodes, mapping, builder, loc, cache);
        if (node.kind == PredicateKind::Not) {
            Value one = builder.create<arith::ConstantIntOp>(loc, 1, 1);
            result = builder.create<arith::XOrIOp>(loc, a, one);
        } else {
            auto b = compilePredicate(node.second, nodes, mapping, builder, loc, cache);
            result = node.kind == PredicateKind::And ? Value(builder.create<arith::AndIOp>(loc, a, b)) :
                                                       Value(builder.create<arith::OrIOp>(loc, a, b));
        }
    }
    cache[id] = result;
    return result;
}
} // namespace
LogicalResult emitGuardedDemands(
    const TraceDemandAnalysis& trace, const SelectedAnalysis& selected, IRMapping& mapping,
    DirectEmissionResult& result)
{
    DenseMap<const CompoundInstanceElement*, std::size_t> ids;
    for (auto [id, site] : llvm::enumerate(trace.sites())) {
        ids[site.phase] = id;
    }
    const auto& analysis = selected.guarded;
    auto predicates = analysis.predicates();
    DenseMap<std::size_t, Predicate> local;
    for (const auto& edge : analysis.retained()) {
        if (analysis.phases()[edge.source]->kPipeValue == analysis.phases()[edge.consumer]->kPipeValue) {
            auto found = local.try_emplace(edge.consumer, edge.predicate);
            if (!found.second) {
                found.first->second = predicates.disjunction(found.first->second, edge.predicate);
            }
        }
    }
    for (auto [consumer, predicate] : local) {
        auto* anchor = mapping.lookup(analysis.phases()[consumer]->elementOp);
        OpBuilder builder(anchor);
        DenseMap<Predicate, Value> cache;
        auto guard = compilePredicate(predicate, predicates.nodes(), mapping, builder, anchor->getLoc(), cache);
        auto selected = builder.create<scf::IfOp>(anchor->getLoc(), guard, false);
        selected.getThenBodyBuilder().create<pto::BarrierOp>(
            anchor->getLoc(),
            pto::PipeAttr::get(anchor->getContext(), static_cast<pto::PIPE>(analysis.phases()[consumer]->kPipeValue)));
        ++result.barriers;
    }
    for (bool outgoing : {false, true}) {
        for (const auto& edge : analysis.retained()) {
            auto* source = analysis.phases()[edge.source];
            auto* target = analysis.phases()[edge.consumer];
            if (source->kPipeValue == target->kPipeValue) {
                continue;
            }
            auto* anchor = mapping.lookup((outgoing ? source : target)->elementOp);
            OpBuilder builder(anchor);
            if (outgoing) {
                builder.setInsertionPointAfter(anchor);
            }
            DenseMap<Predicate, Value> cache;
            auto loc = anchor->getLoc();
            auto guard = compilePredicate(edge.predicate, predicates.nodes(), mapping, builder, loc, cache);
            auto selected = builder.create<scf::IfOp>(loc, guard, false);
            auto body = selected.getThenBodyBuilder();
            auto src = pto::PipeAttr::get(anchor->getContext(), static_cast<pto::PIPE>(source->kPipeValue));
            auto dst = pto::PipeAttr::get(anchor->getContext(), static_cast<pto::PIPE>(target->kPipeValue));
            auto key = body.getI64IntegerAttr(ids.lookup(source) * trace.sites().size() + ids.lookup(target));
            if (outgoing) {
                body.create<pto::LogicalSetOp>(loc, src, dst, key, ValueRange{});
                ++result.sets;
            } else {
                body.create<pto::LogicalWaitOp>(loc, src, dst, key, ValueRange{});
                ++result.waits;
            }
        }
    }
    result.privateSelectors = !analysis.retained().empty();
    return success();
}
} // namespace mlir::pto::frontiersynch
