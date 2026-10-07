// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_FRONTIERSYNCH_CIRCUITENDPOINTS_H
#define PTO_FRONTIERSYNCH_CIRCUITENDPOINTS_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include <map>
namespace mlir::pto::frontiersynch {
// Shared detached emitter for paired arithmetic recipes. Producers supply the
// same canonical source coordinates at both cuts; availability is checked here.
class CircuitEndpoints {
public:
    using Id = RegionExpressions::Id;
    CircuitEndpoints(
        func::FuncOp function, RegionExpressions& arena, const std::vector<TemplateEndpointAnchor>& anchors)
        : function(function), arena(arena), anchors(anchors), plan(std::make_unique<PreparedLogicalPlan>(0))
    {
        plan->nestedIdentities = true;
        plan->independentPieces = true;
        plan->groupedFamilies = true;
        for (const auto& anchor : anchors) {
            arena.forbidRecomputation(anchor.phase->elementOp);
        }
    }
    std::function<LogicalResult(Id, OpBuilder&, Operation*, RegionExpressions::CutEmission&)> recover;
    bool add(
        uint32_t source, uint32_t target, Id sourceGuard, Id targetGuard, ArrayRef<Id> sourceIdentity,
        ArrayRef<Id> targetIdentity)
    {
        if (plan->families.size() >= UINT32_MAX || source >= anchors.size() || target >= anchors.size()) {
            return false;
        }
        const auto record = static_cast<uint32_t>(plan->families.size());
        const auto p = static_cast<uint32_t>(anchors[source].phase->kPipeValue);
        const auto q = static_cast<uint32_t>(anchors[target].phase->kPipeValue);
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = p;
        family.targetPipe = q;
        family.local = p == q;
        family.sourceCut = anchors[source].after;
        family.targetCut = anchors[target].before;
        family.members.push_back({record, source, target, {}, {}});
        plan->families.push_back(std::move(family));
        for (bool publish : {true, false}) {
            if (p == q && publish) {
                continue;
            }
            auto cut = publish ? anchors[source].after.before : anchors[target].before.before;
            auto guard = emit(publish ? sourceGuard : targetGuard, cut);
            if (failed(guard)) {
                return false;
            }
            PreparedLogicalEndpoint endpoint{
                cut,
                p == q ? LogicalCommandKind::Barrier : publish ? LogicalCommandKind::Set : LogicalCommandKind::Wait,
                p,
                q,
                record,
                *guard,
                {}};
            endpoint.records = {record};
            endpoint.piece = plan->endpoints.size();
            if (p != q) {
                auto identity = publish ? sourceIdentity : targetIdentity;
                if (identity.empty()) {
                    return false;
                }
                auto last = emit(identity.back(), cut), member = emit(arena.constant(record), cut);
                if (failed(last) || failed(member)) {
                    return false;
                }
                endpoint.identity = *last;
                endpoint.memberCoordinates.push_back(*member);
                for (auto coordinate : identity.drop_back()) {
                    auto value = emit(coordinate, cut);
                    if (failed(value)) {
                        return false;
                    }
                    endpoint.memberCoordinates.push_back(*value);
                }
            }
            plan->endpoints.push_back(std::move(endpoint));
        }
        return true;
    }
    std::unique_ptr<PreparedLogicalPlan> take()
    {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> names;
        for (const auto& family : plan->families) {
            names.emplace(std::make_pair(family.sourcePipe, family.targetPipe), family.id);
        }
        for (auto& endpoint : plan->endpoints) {
            endpoint.record = names.find({endpoint.sourcePipe, endpoint.targetPipe})->second;
        }
        return std::move(plan);
    }

private:
    func::FuncOp function;
    RegionExpressions& arena;
    const std::vector<TemplateEndpointAnchor>& anchors;
    std::unique_ptr<PreparedLogicalPlan> plan;
    std::map<Operation*, Block*> blocks;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    FailureOr<Value> emit(Id expression, Operation* cut)
    {
        if (!cut) {
            return failure();
        }
        auto [it, added] = blocks.try_emplace(cut);
        if (added) {
            it->second = &plan->addPreparation(cut);
        }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(it->second);
        if (recover && failed(recover(expression, builder, cut, contexts[cut]))) {
            return failure();
        }
        return arena.emitContextual(expression, builder, cut, contexts[cut]);
    }
};
} // namespace mlir::pto::frontiersynch
#endif
