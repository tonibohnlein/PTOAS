// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "InvocationPrefixCollection.h"
#include "PrefixDemand.h"
#include "mlir/IR/Matchers.h"
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
    if (!state || !state->error.empty() || !region.expressions || region.expressions != state->arena) {
        return {};
    }
    std::shared_ptr<const SyncInput> input;
    for (const auto& child : state->children) {
        auto owner = child.regional.invocationPrefix ? child.regional.invocationPrefix->input :
            (child.prefixMathematics ? child.prefixMathematics->input : nullptr);
        if (!owner && child.finite && child.finite->originalInput()) {
            owner = child.finite->originalInput()->storageOwner();
        }
        if (!owner) {
            const bool empty = child.anchors.empty() && child.regional.anchors.empty() &&
                (!child.finite || !child.finite->concrete() || child.finite->concrete()->phases.empty());
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
    if (state->relationalResult) {
        return relationalPrefixContribution(
            region, std::move(input), [state](InvocationPrefixCollection& collection, std::string& error) {
                return collectSequencePrefixChildren(*state, collection, error);
            });
    }
    auto result = std::make_shared<RegionalPrefixContribution>();
    result->input = std::move(input);
    result->expressions = region.expressions;
    result->function = state->function;
    result->accessModel = region.accessModel;
    result->aliasPolicy = region.gmAliasPolicy;
    attachSequenceOccurrenceRecipes(region, state, *result);
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
        contribution->aliasPolicy == region.gmAliasPolicy && contribution->matchesOccurrenceFrame &&
        contribution->matchesOccurrenceFrame(region);
    if (!valid) {
        error = "regional prefix contribution has stale invocation or semantic context";
        return failure();
    }
    if (failed(collection.bindInput(contribution->input, error))) { return failure(); }
    return contribution->collect(collection, error);
}
LogicalResult InvocationPrefixCollection::source(
    const CompoundInstanceElement* phase, PrefixSourceSnapshot snapshot, std::string& error)
{
    if (!original || !phase || !phase->elementOp || !original->isProperAncestor(phase->elementOp) ||
        pending.sources.size() != sourceBatches.size()) {
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
        auto batch = std::make_shared<std::vector<PrefixSourceSnapshot>>();
        sourceBatches.push_back(batch);
        pending.sources.push_back(
            {phase->elementOp, key.second,
             [batch](
                 OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value> state, Value ordinal,
                 std::string& error) {
                 for (const auto& snapshot : *batch) {
                     if (failed(snapshot(b, mapping, state, ordinal, error))) {
                         return failure();
                     }
                 }
                 return success();
             }});
    }
    if (snapshot) {
        sourceBatches[entry->second]->push_back(std::move(snapshot));
    }
    return success();
}
LogicalResult InvocationPrefixCollection::consumer(
    Operation* anchor, PrefixDirection direction, PrefixIncomingSelector selector, std::string& error)
{
    const bool valid = original && anchor && original->isProperAncestor(anchor) && selector &&
                       direction.sourcePipe != direction.targetPipe &&
                       pending.consumers.size() == consumerBatches.size();
    if (!valid) {
        error = "prefix consumer has invalid invocation or direction";
        return failure();
    }
    if (directions.insert(direction).second) {
        pending.directions.push_back(direction);
    }
    auto [entry, inserted] = consumers.emplace(std::make_pair(anchor, direction), pending.consumers.size());
    if (inserted) {
        auto batch = std::make_shared<std::vector<PrefixIncomingSelector>>();
        consumerBatches.push_back(batch);
        pending.consumers.push_back(
            {anchor, direction,
             [batch, anchor](OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error)
                 -> FailureOr<Value> {
                 Value result;
                 for (const auto& selector : *batch) {
                     auto value = selector(b, mapping, state, error);
                     if (failed(value) || !*value || !(*value).getType().isIndex()) {
                         if (error.empty()) {
                             error = "prefix incoming selector must produce an index rank";
                         }
                         return failure();
                     }
                     if (!result) {
                         result = *value;
                         continue;
                     }
                     auto later = b.create<arith::CmpIOp>(anchor->getLoc(), arith::CmpIPredicate::ugt, result, *value);
                     result = b.create<arith::SelectOp>(anchor->getLoc(), later, result, *value);
                 }
                 return result;
             }});
    }
    consumerBatches[entry->second]->push_back(std::move(selector));
    return success();
}
LogicalResult appendPrefixIncoming(
    InvocationPrefixCollection& collection, Operation* cut, PrefixDirection direction, PrefixIncomingSelector active,
    PrefixIncomingSelector sourceRank, std::string& error)
{
    if (!collection.function() || !cut || !collection.function()->isProperAncestor(cut) || !active) {
        error = "prefix requirement has no valid original consumer or exact activity";
        return failure();
    }
    if (direction.sourcePipe == direction.targetPipe) {
        if (direction.sourcePipe == static_cast<uint32_t>(PIPE::PIPE_S)) {
            error = "scalar prefix completion requires its native prerequisite certificate";
            return failure();
        }
        collection.recipe().actions.push_back(
            {cut, PrivateStateCut::Before,
             [active = std::move(active), cut, pipe = direction.sourcePipe](
                 OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value> state, std::string& error) {
                 auto guard = active(b, mapping, state, error);
                 if (failed(guard) || !*guard || !(*guard).getType().isInteger(1)) {
                     return failure();
                 }
                 if (matchPattern(*guard, m_Zero())) {
                     return success();
                 }
                 auto branch = b.create<scf::IfOp>(cut->getLoc(), *guard, false);
                 OpBuilder::InsertionGuard restore(b);
                 b.setInsertionPointToStart(branch.thenBlock());
                 b.create<BarrierOp>(cut->getLoc(), PipeAttr::get(b.getContext(), static_cast<PIPE>(pipe)));
                 return success();
             }});
        return success();
    }
    if (!sourceRank) {
        error = "prefix requirement has no constructive source rank";
        return failure();
    }
    return collection.consumer(
        cut, direction,
        [active = std::move(active), sourceRank = std::move(sourceRank), cut](
            OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) -> FailureOr<Value> {
            auto guard = active(b, mapping, state, error);
            if (failed(guard) || !*guard || !(*guard).getType().isInteger(1)) {
                return failure();
            }
            auto zero = b.create<arith::ConstantIndexOp>(cut->getLoc(), 0);
            if (matchPattern(*guard, m_Zero())) {
                return zero.getResult();
            }
            auto prepareRank = [&]() -> FailureOr<Value> {
                auto rank = sourceRank(b, mapping, state, error);
                if (failed(rank) || !*rank || !(*rank).getType().isIndex()) { return failure(); }
                return *rank;
            };
            if (matchPattern(*guard, m_One())) {
                auto rank = prepareRank();
                if (failed(rank)) { return failure(); }
                // Keep the prior constant-true form: direct rank forwarding
                // would also change structural full-acquisition qualification.
                return b.create<arith::SelectOp>(cut->getLoc(), *guard, *rank, zero).getResult();
            }
            // Rank selectors only read captured state and construct owned pure
            // scalar expressions. Qualify the callback once as before, while
            // executing its code only when this exact incoming edge is active.
            // Prefix expression caches include the insertion block; then-only
            // values cannot be reused by a later outer or sibling callback.
            auto selected = b.create<scf::IfOp>(cut->getLoc(), TypeRange{b.getIndexType()}, *guard, true);
            {
                OpBuilder::InsertionGuard restore(b);
                if (!selected.thenBlock()->empty()) { selected.thenBlock()->getTerminator()->erase(); }
                b.setInsertionPointToStart(selected.thenBlock());
                auto rank = prepareRank();
                if (failed(rank)) { return failure(); }
                b.create<scf::YieldOp>(cut->getLoc(), *rank);
                if (!selected.elseBlock()->empty()) { selected.elseBlock()->getTerminator()->erase(); }
                b.setInsertionPointToStart(selected.elseBlock());
                b.create<scf::YieldOp>(cut->getLoc(), zero.getResult());
            }
            return selected.getResult(0);
        },
        error);
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
    if (retained->concreteDemands()) {
        return appendExplicitPrefixRealization(*retained->concreteDemands(), collection, error);
    }
    if (retained->sequenceDemands && retained->sequenceDemands->state) {
        return retained->sequenceDemands->state->collectPrefix(collection, error);
    }
    error = "selected mathematical backend " + retained->backend + " at region " + std::to_string(retained->region) +
            " has no certified prefix record adapter";
    return failure();
}
} // namespace mlir::pto::frontiersynch
