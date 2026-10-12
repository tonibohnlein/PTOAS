// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "InvocationPrefixCollection.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "SequenceAnalysisInternal.h"
#include "../WholeInvocationIdentity.h"
#include "../Structured/BoundedPrefixRealization.h"
#include "PTO/IR/PTO.h"
namespace mlir::pto::frontiersynch {
LogicalResult InvocationPrefixCollection::bindInput(std::shared_ptr<const SyncInput> owner, std::string& error)
{
    if (!owner || (input && input != owner)) {
        error = "prefix contribution belongs to a different semantic input";
        return failure();
    }
    input = std::move(owner);
    return success();
}
std::shared_ptr<const RegionalPrefixContribution> sequencePrefixContribution(
    const RegionalAnalysis& region, std::shared_ptr<SequenceAnalysisState> state)
{
    if (!state || !state->error.empty() || state->relationalResult ||
        !region.expressions || region.expressions != state->arena) { return {}; }
    std::shared_ptr<const SyncInput> input;
    for (const auto& child : state->children) {
        auto owner = child.regional.invocationPrefix ? child.regional.invocationPrefix->input :
            (child.prefixMathematics ? child.prefixMathematics->input : nullptr);
        if (!owner) {
            const bool empty = child.anchors.empty() && child.regional.anchors.empty() &&
                child.explicitAnalysis.phases.empty();
            if (!empty) { return {}; }
            continue;
        }
        if (input && input != owner) { return {}; }
        input = std::move(owner);
    }
    // Legacy borrowed analyses can still prepare ordinary endpoints. Do not
    // publish a detached prefix capability without a retained semantic owner.
    if (!input || (state->input && state->input != input.get()) ||
        (region.accessModel && region.accessModel != &input->accesses())) { return {}; }
    auto result = std::make_shared<RegionalPrefixContribution>();
    result->input = std::move(input);
    result->expressions = region.expressions;
    result->function = state->function;
    result->accessModel = region.accessModel;
    result->aliasPolicy = region.gmAliasPolicy;
    result->collect = [state = std::move(state)](InvocationPrefixCollection& collection, std::string& error) {
        return state->collectPrefix(collection, error);
    };
    return result;
}
LogicalResult collectRegionalInvocationPrefix(const RegionalAnalysis& region,
    InvocationPrefixCollection& collection, std::string& error)
{
    const auto& contribution = region.invocationPrefix;
    const bool valid = contribution && contribution->collect && contribution->function == collection.function() &&
        contribution->expressions == region.expressions && contribution->accessModel == region.accessModel &&
        contribution->aliasPolicy == region.gmAliasPolicy;
    if (!valid) {
        error = "regional prefix contribution has stale invocation or semantic context";
        return failure();
    }
    if (failed(collection.bindInput(contribution->input, error))) { return failure(); }
    return contribution->collect(collection, error);
}
LogicalResult InvocationPrefixCollection::source(
    const CompoundInstanceElement* phase,
    std::function<LogicalResult(OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value>, Value, std::string&)>
        snapshot,
    std::string& error)
{
    if (!original || !phase || !phase->elementOp || !original->isProperAncestor(phase->elementOp)) {
        error = "prefix source phase belongs to a foreign invocation";
        return failure();
    }
    auto key = std::make_pair(phase->elementOp, static_cast<uint32_t>(phase->kPipeValue));
    auto existing = sourceCuts.find(key);
    if (existing != sourceCuts.end() && existing->second != phase) {
        error = "prefix source needs a simultaneous-phase identity certificate";
        return failure();
    }
    sourceCuts.emplace(key, phase);
    auto [entry, inserted] = sources.emplace(phase, pending.sources.size());
    if (inserted) {
        pending.sources.push_back({phase->elementOp, key.second, std::move(snapshot)});
        return success();
    }
    if (!snapshot) {
        return success();
    }
    auto previous = std::move(pending.sources[entry->second].snapshot);
    pending.sources[entry->second].snapshot = [previous, snapshot](
                                                  OpBuilder& b, const IRMapping& mapping,
                                                  llvm::MutableArrayRef<Value> state, Value ordinal,
                                                  std::string& error) {
        if (previous && failed(previous(b, mapping, state, ordinal, error))) {
            return failure();
        }
        return snapshot(b, mapping, state, ordinal, error);
    };
    return success();
}
LogicalResult InvocationPrefixCollection::consumer(
    Operation* anchor, PrefixDirection direction, PrefixIncomingSelector selector, std::string& error)
{
    const bool valid =
        anchor && original->isProperAncestor(anchor) && selector && direction.sourcePipe != direction.targetPipe;
    if (!valid) {
        error = "prefix consumer has invalid invocation or direction";
        return failure();
    }
    if (directions.insert(direction).second) {
        pending.directions.push_back(direction);
    }
    auto [entry, inserted] = consumers.emplace(std::make_pair(anchor, direction), pending.consumers.size());
    if (inserted) {
        pending.consumers.push_back({anchor, direction, std::move(selector)});
        return success();
    }
    auto previous = std::move(pending.consumers[entry->second].incoming);
    pending.consumers[entry->second].incoming = [previous, selector, anchor](
                                                    OpBuilder& b, const IRMapping& mapping, llvm::ArrayRef<Value> state,
                                                    std::string& error) -> FailureOr<Value> {
        auto a = previous(b, mapping, state, error);
        if (failed(a)) {
            return failure();
        }
        auto c = selector(b, mapping, state, error);
        const bool valid =
            succeeded(a) && succeeded(c) && *a && *c && (*a).getType().isIndex() && (*c).getType().isIndex();
        if (!valid) {
            return failure();
        }
        auto later = b.create<arith::CmpIOp>(anchor->getLoc(), arith::CmpIPredicate::ugt, *a, *c);
        return b.create<arith::SelectOp>(anchor->getLoc(), later, *a, *c).getResult();
    };
    return success();
}
LogicalResult appendExplicitPrefixRealization(
    const ExplicitAnalysis& retained, InvocationPrefixCollection& collection, std::string& error)
{
    if (!retained.error.empty() || !retained.reduction.error.empty() ||
        retained.phases.size() != retained.occurrences.size()) {
        error = "explicit prefix contribution requires exact retained occurrences";
        return failure();
    }
    auto& recipe = collection.recipe();
    auto type = IndexType::get(collection.function().getContext());
    std::map<uint32_t, std::size_t> positions, slots;
    Block* block = nullptr;
    Operation* previous = nullptr;
    for (std::size_t i = 0; i < retained.phases.size(); ++i) {
        auto* phase = retained.phases[i];
        auto* op = phase ? phase->elementOp : nullptr;
        const bool valid = op && (!block || op->getBlock() == block) && (!previous || previous->isBeforeInBlock(op)) &&
                           positions.emplace(retained.occurrences[i].payload, i).second;
        if (!valid) {
            error = "explicit prefix occurrence order or block is invalid";
            return failure();
        }
        block = op->getBlock();
        previous = op;
        const auto slot = recipe.slots.size();
        slots.emplace(retained.occurrences[i].payload, slot);
        recipe.slots.push_back({type, IntegerAttr::get(type, 0)});
        if (failed(collection.source(
                phase,
                [slot](OpBuilder&, const IRMapping&, MutableArrayRef<Value> state, Value ordinal, std::string&) {
                    state[slot] = ordinal;
                    return success();
                },
                error))) {
            return failure();
        }
    }
    for (auto edge : retained.reduction.retained) {
        if (!positions.count(edge.source) || !positions.count(edge.target) ||
            positions.at(edge.source) >= positions.at(edge.target)) {
            error = "explicit prefix demand is not an original forward pair";
            return failure();
        }
        auto* source = retained.phases[positions.at(edge.source)];
        auto* target = retained.phases[positions.at(edge.target)];
        auto p = static_cast<uint32_t>(source->kPipeValue), q = static_cast<uint32_t>(target->kPipeValue);
        auto* cut = target->elementOp;
        if (p == q) {
            if (p == static_cast<uint32_t>(PIPE::PIPE_S)) {
                error = "scalar prefix completion requires its native prerequisite proof";
                return failure();
            }
            recipe.actions.push_back(
                {cut, PrivateStateCut::Before,
                 [cut, p](OpBuilder& b, const IRMapping&, MutableArrayRef<Value>, std::string&) {
                     b.create<BarrierOp>(cut->getLoc(), PipeAttr::get(b.getContext(), static_cast<PIPE>(p)));
                     return success();
                 }});
        } else {
            const auto slot = slots.at(edge.source);
            if (failed(collection.consumer(
                    cut, {p, q},
                    [slot](OpBuilder&, const IRMapping&, ArrayRef<Value> state, std::string&) -> FailureOr<Value> {
                        return state[slot];
                    },
                    error))) {
                return failure();
            }
        }
    }
    return success();
}
LogicalResult collectInvocationPrefix(
    std::shared_ptr<const MathematicalResult> retained, InvocationPrefixCollection& collection, std::string& error)
{
    if (!retained || !retained->input) {
        error = "prefix collection requires its retained input owner";
        return failure();
    }
    if (failed(collection.bindInput(retained->input, error))) { return failure(); }
    if (retained->identityComposition) {
        const auto& owner = *retained->identityComposition;
        const bool valid =
            retained->region == owner.region && owner.child &&
            owner.applies(retained->input.get(), owner.index.get(), owner.program.get(), collection.function());
        if (!valid) {
            error = "prefix identity owner has stale original scope";
            return failure();
        }
        return collectInvocationPrefix(owner.child, collection, error);
    }
    if (retained->boundedDemands) {
        return appendBoundedPrefixRealization(retained->boundedDemands, collection, error);
    }
    if (retained->explicitDemands) {
        return appendExplicitPrefixRealization(*retained->explicitDemands, collection, error);
    }
    if (retained->sequenceDemands && retained->sequenceDemands->state) {
        return retained->sequenceDemands->state->collectPrefix(collection, error);
    }
    error = "selected mathematical backend " + retained->backend + " at region " + std::to_string(retained->region) +
            " has no certified prefix record adapter";
    return failure();
}
} // namespace mlir::pto::frontiersynch
