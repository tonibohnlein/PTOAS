// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> SequenceAnalysisState::prepare()
{
    auto result = std::make_unique<PreparedLogicalPlan>(0);
    result->completeInvocation = llvm::any_of(children, [](const Child& child) { return !child.anchors.empty(); });
    result->groupedFamilies = true;
    result->independentPieces = true;
    uint32_t nextRecord = 0;
    int64_t nextPiece = 0;
    for (auto& child : children) {
        for (auto loop : child.regional.occurrenceLoops) {
            if (!loop) { continue; }
            auto step = sequenceInteger(loop.getStep());
            if (!step || *step <= 0 || !loop.getInductionVar().getType().isIndex() ||
                loop->getParentOfType<func::FuncOp>() != function) {
                fail("regional endpoint loop requires an available constant positive index step"); return failure();
            }
        }
        auto supplied = child.regional.prepare();
        if (failed(supplied)) { fail("regional child endpoint preparation failed"); return failure(); }
        auto prepared = std::move(*supplied);
        uint32_t maximum = 0;
        bool any = false;
        for (const auto& family : prepared->families) {
            maximum = std::max(maximum, family.id);
            for (const auto& member : family.members) { maximum = std::max(maximum, member.record); any = true; }
        }
        if (any && maximum >= UINT32_MAX - nextRecord) {
            fail("sequence record identity overflow"); return failure();
        }
        for (auto& family : prepared->families) {
            family.id += nextRecord;
            for (auto& member : family.members) {
                maximum = std::max(maximum, member.record);
                any = true;
                member.record += nextRecord;
            }
            result->families.push_back(std::move(family));
        }
        for (auto& stage : prepared->preparation) { result->preparation.push_back(std::move(stage)); }
        for (auto& endpoint : prepared->endpoints) {
            endpoint.record += nextRecord;
            endpoint.piece = nextPiece++;
            if (!prepared->independentPieces) {
                endpoint.records.push_back(static_cast<uint32_t>(endpoint.record - nextRecord));
                if (endpoint.kind != LogicalCommandKind::Barrier) {
                    OpBuilder builder(function.getContext());
                    builder.setInsertionPointToEnd(&result->addPreparation(endpoint.before));
                    endpoint.memberCoordinates.push_back(builder.create<arith::ConstantIndexOp>(
                        endpoint.before->getLoc(), endpoint.record - nextRecord));
                }
            }
            for (auto& record : endpoint.records) { record += nextRecord; }
            if (!endpoint.memberCoordinates.empty() && nextRecord) {
                OpBuilder builder(function.getContext());
                builder.setInsertionPointToEnd(&result->addPreparation(endpoint.before));
                auto shift = builder.create<arith::ConstantIndexOp>(endpoint.before->getLoc(), nextRecord);
                endpoint.memberCoordinates[0] = builder.create<arith::AddIOp>(endpoint.before->getLoc(),
                    endpoint.memberCoordinates[0], shift);
            }
            result->endpoints.push_back(std::move(endpoint));
        }
        if (any) {
            if (maximum >= UINT32_MAX - nextRecord) { fail("sequence record identity overflow"); return failure(); }
            nextRecord += maximum + 1;
        }
    }
    std::map<Operation*, std::unique_ptr<DenseMap<Expr, Value>>> memos;
    std::map<Operation*, Block*> stages;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    const bool contextual = llvm::any_of(children, [](const Child& child) {
        return child.regional.capabilities.contextualGuards;
    });
    auto emit = [&](Expr expression, Operation* cut) -> FailureOr<Value> {
        auto& memo = memos[cut];
        if (!memo) { memo = std::make_unique<DenseMap<Expr, Value>>(); stages[cut] = &result->addPreparation(cut); }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(stages[cut]);
        return contextual ? expressions.emitContextual(expression,builder,cut,contexts[cut]) :
                            expressions.emit(expression,builder,cut,*memo);
    };
    auto endpointGuard = [&](const Port& point, Expr guard) {
        const auto& child = children[point.child];
        const auto& anchor = child.anchors[point.type];
        auto loop = child.regional.occurrenceLoops[point.type];
        if (loop) {
            auto ordinal = expressions.div(expressions.sub(expressions.input(loop.getInductionVar()),
                expressions.input(loop.getLowerBound())), c(*sequenceInteger(loop.getStep())));
            guard = both(guard, expressions.eq(ordinal, point.ordinal));
        }
        for (auto coordinate : anchor.coordinates) {
            guard = both(guard, expressions.eq(expressions.input(coordinate.loop.getInductionVar()),
                                               c(coordinate.induction)));
        }
        return guard;
    };
    auto addDemand = [&](const Port& a, const Port& b, Expr retained) -> bool {
        if (expressions.constantValue(retained) == 0) { return true; }
        const auto& ca = children[a.child];
        const auto& cb = children[b.child];
        const auto& aa = ca.anchors[a.type];
        const auto& ab = cb.anchors[b.type];
        auto p = static_cast<uint32_t>(aa.phase->kPipeValue);
        auto q = static_cast<uint32_t>(ab.phase->kPipeValue);
        if (p == q) {
            // Adjacency is an insertion premise. A crossing local cover must
            // connect the last occurrence on p to the next first one on p.
            {
                auto selectedAs = [&](const std::map<uint32_t, std::vector<RegionalSelector>>& selectors,
                                      const Port& occurrence) {
                    Expr selected = no();
                    auto found = selectors.find(p);
                    if (found != selectors.end()) {
                        for (const auto& candidate : found->second) {
                            if (candidate.event.type == occurrence.type) {
                                selected = either(selected, both(candidate.present,
                                    expressions.eq(candidate.event.ordinal, occurrence.ordinal)));
                            }
                        }
                    }
                    return selected;
                };
                auto adjacent = both(selectedAs(ca.regional.lastPayloads, a),
                                     selectedAs(cb.regional.firstPayloads, b));
                for (uint32_t child = a.child + 1; child < b.child; ++child) {
                    auto found = children[child].regional.firstPayloads.find(p);
                    if (found != children[child].regional.firstPayloads.end()) {
                        for (auto selected : found->second) { adjacent = both(adjacent, negate(selected.present)); }
                    }
                }
                if (!expressions.implies(retained, adjacent)) {
                    return fail("sequence local cover needs an unproved occurrence-adjacency predicate: child " +
                        std::to_string(a.child) + ":" + std::to_string(a.type) + " to " +
                        std::to_string(b.child) + ":" + std::to_string(b.type));
                }
            }
        }

        if (nextRecord == UINT32_MAX) { return fail("sequence endpoint identity overflow"); }
        const auto record = nextRecord++;
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = aa.after; family.targetCut = ab.before;
        family.members.push_back({record, a.type, b.type, aa.coordinates, ab.coordinates});
        result->families.push_back(std::move(family));
        auto targetGuard = emit(endpointGuard(b, retained), ab.before.before);
        if (failed(targetGuard)) { return fail("sequence WAIT/barrier predicate unavailable at its cut"); }
        if (p == q) {
            PreparedLogicalEndpoint endpoint{ab.before.before, LogicalCommandKind::Barrier, p, q,
                                              record, *targetGuard, {}};
            endpoint.records.push_back(record); endpoint.piece = nextPiece++;
            result->endpoints.push_back(std::move(endpoint));
            return true;
        }
        auto sourceGuard = emit(endpointGuard(a, retained), aa.after.before);
        auto sourceIdentity = emit(c(0), aa.after.before);
        auto targetIdentity = emit(c(0), ab.before.before);
        auto sourceMember = emit(c(record), aa.after.before);
        auto targetMember = emit(c(record), ab.before.before);
        if (failed(sourceGuard) || failed(sourceIdentity) || failed(targetIdentity) ||
            failed(sourceMember) || failed(targetMember)) {
            return fail("sequence SET predicate or matching identity unavailable at its cut");
        }
        PreparedLogicalEndpoint set{aa.after.before, LogicalCommandKind::Set, p, q, record,
                                     *sourceGuard, *sourceIdentity};
        set.memberCoordinates.push_back(*sourceMember); set.records.push_back(record); set.piece = nextPiece++;
        PreparedLogicalEndpoint wait{ab.before.before, LogicalCommandKind::Wait, p, q, record,
                                      *targetGuard, *targetIdentity};
        wait.memberCoordinates.push_back(*targetMember); wait.records.push_back(record); wait.piece = nextPiece++;
        result->endpoints.push_back(std::move(set)); result->endpoints.push_back(std::move(wait));
        return true;
    };
    for (const auto& edge : crossings) {
        if (!addDemand(ports[edge.source], ports[edge.target], edge.guard)) { return failure(); }
    }
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    std::map<uint32_t, Namespace> recordNamespace;
    for (const auto& family : result->families) {
        Namespace key{family.sourcePipe, family.targetPipe, family.displacement};
        for (const auto& member : family.members) {
            auto [it, added] = namespaces.emplace(key, member.record);
            it->second = std::min(it->second, member.record);
            recordNamespace.emplace(member.record, key);
        }
    }
    for (auto& endpoint : result->endpoints) {
        endpoint.record = namespaces.at(recordNamespace.at(endpoint.records.front()));
    }
    if (!expressions.error().empty()) { fail(expressions.error()); return failure(); }
    return result;
}
} // namespace mlir::pto::frontiersynch
