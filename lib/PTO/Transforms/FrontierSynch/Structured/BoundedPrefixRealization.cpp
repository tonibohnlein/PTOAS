// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "BoundedPrefixRealization.h"
#include "BoundedLifetimeInactiveRecords.h"
#include "../Shared/BoundedPredicateHistory.h"
#include "../Shared/CountedLoop.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct Leaf {
    Id id;
    Value original;
    uint64_t offset;
    std::size_t requirement;
};
struct Rank {
    std::size_t first;
    uint64_t lag;
};
struct Owner {
    std::shared_ptr<BoundedLifetimeDemandResult> demands;
    std::optional<BoundedPredicateHistory> history;
    std::vector<Leaf> leaves;
    std::map<uint32_t, Rank> ranks;
};
Id definitionPresence(Owner& owner, Value value, uint64_t offset)
{
    auto& demands = *owner.demands;
    auto& e = demands.expressions;
    auto present = e.boolean(true);
    auto* child = value.getDefiningOp();
    if (!child) {
        return e.invalid;
    }
    for (auto* parent = child->getParentOp(); parent != demands.loop; child = parent, parent = parent->getParentOp()) {
        auto branch = dyn_cast_or_null<scf::IfOp>(parent);
        if (!branch) {
            return e.invalid;
        }
        auto condition = e.invalid;
        for (auto [id, recipe] : demands.predicateBindings) {
            if (recipe.first == branch.getCondition() && recipe.second == offset) {
                condition = id;
                break;
            }
        }
        if (condition == e.invalid) {
            return e.invalid;
        }
        const bool thenArm = child->getParentRegion() == &branch.getThenRegion();
        present = e.land(present, thenArm ? condition : e.lnot(condition));
    }
    return present;
}
std::string proofDiagnostic(
    const char* reason, std::size_t record, GuardedRankEdge edge, uint64_t sites, const Leaf* leaf = nullptr)
{
    std::string result;
    llvm::raw_string_ostream out(result);
    out << "bounded prefix " << reason << ": record=" << record << " source=" << edge.source
        << " target=" << edge.target % sites << " target_window=" << edge.target << " distance=" << edge.target / sites
        << " guard=" << edge.guard;
    if (leaf) {
        out << " leaf=" << leaf->id << " leaf_offset=" << leaf->offset << " predicate={";
        Value original = leaf->original;
        original.print(out);
        out << "}";
    }
    return result;
}
bool evaluationPrecedes(Value original, Operation* cut, DominanceInfo& dominance)
{
    if (dominance.properlyDominates(original, cut)) {
        return true;
    }
    // A conditionally executed definition may precede a cut outside its arm
    // without dominating it. Branch absence is checked separately below.
    for (auto* op = original.getDefiningOp(); op; op = op->getParentOp()) {
        if (op->getBlock() == cut->getBlock()) {
            return op != cut && op->isBeforeInBlock(cut);
        }
    }
    return false;
}
bool absentInsensitive(Owner& owner, Id root, const Leaf& leaf)
{
    auto& e = owner.demands->expressions;
    auto present = definitionPresence(owner, leaf.original, leaf.offset);
    if (present == e.invalid) {
        return false;
    }
    RegionExpressions::Substitution no({{leaf.id, e.boolean(false)}}), yes({{leaf.id, e.boolean(true)}});
    auto a = e.substitute(root, no), b = e.substitute(root, yes);
    auto absent = e.lnot(present);
    return e.implies(e.land(absent, a), b) && e.implies(e.land(absent, b), a);
}
LogicalResult appendHistory(Owner& owner, InvocationPrefixRecipe& recipe, std::string& error)
{
    auto& demands = *owner.demands;
    auto& e = demands.expressions;
    llvm::DenseMap<Value, std::size_t> requirements;
    std::vector<BoundedPredicateHistoryRequirement> history;
    for (auto [id, binding] : demands.predicateBindings) {
        const auto inputs = e.referencedInputs(id);
        if (inputs.size() != 1 || inputs.front().first != id) {
            continue;
        }
        Value input = inputs.front().second;
        if (input.getParentRegion()) {
            continue;
        }
        auto [entry, added] = requirements.try_emplace(binding.first, requirements.size());
        if (added) {
            history.push_back({binding.first, demands.loop, demands.window.span});
        }
        owner.leaves.push_back({id, binding.first, binding.second, entry->second});
    }
    auto appended = BoundedPredicateHistory::append(demands.function, history, recipe.slots, recipe.actions, error);
    if (failed(appended)) {
        return failure();
    }
    owner.history.emplace(std::move(*appended));
    return success();
}
LogicalResult appendRank(
    Owner& owner, uint32_t site, uint64_t lag, InvocationPrefixCollection& collection, std::string& error)
{
    auto& recipe = collection.recipe();
    auto& demands = *owner.demands;
    auto type = IndexType::get(demands.function.getContext());
    const auto first = recipe.slots.size();
    if (lag >= recipe.slots.max_size() - first) {
        error = "source rank history extent is not representable";
        return failure();
    }
    for (uint64_t j = 0; j <= lag; ++j) {
        recipe.slots.push_back({type, IntegerAttr::get(type, 0)});
    }
    owner.ranks.emplace(site, Rank{first, lag});
    auto clear = [first, lag](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        auto zero = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0);
        for (uint64_t j = 0; j <= lag; ++j) {
            state[first + j] = zero;
        }
        return success();
    };
    auto shift = [first, lag](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        for (uint64_t j = lag; j > 0; --j) {
            state[first + j] = state[first + j - 1];
        }
        state[first] = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0);
        return success();
    };
    recipe.actions.push_back({demands.loop, PrivateStateCut::Before, clear});
    recipe.actions.push_back({&demands.loop.getBody()->front(), PrivateStateCut::Before, shift});
    return collection.source(
        demands.anchors[site].phase,
        [first](OpBuilder&, const IRMapping&, MutableArrayRef<Value> state, Value ordinal, std::string&) {
            state[first] = ordinal;
            return success();
        },
        error);
}
FailureOr<Value> guard(
    const std::shared_ptr<Owner>& owner, GuardedRankEdge edge, Operation* cut, OpBuilder& builder,
    const IRMapping& mapping, ArrayRef<Value> state, std::string& error)
{
    auto& demands = *owner->demands;
    auto& e = demands.expressions;
    const auto distance = edge.target / demands.anchors.size();
    RegionExpressions::Transaction transaction(e);
    std::vector<std::pair<Id, Id>> substitutions{{demands.base, e.sub(demands.ordinal, e.constant(distance))}};
    RegionExpressions::CutEmission context;
    DominanceInfo dominance(demands.function);
    const auto referenced = e.referencedInputs(edge.guard);
    for (const auto& leaf : owner->leaves) {
        const bool used = llvm::any_of(referenced, [&](auto input) { return input.first == leaf.id; });
        if (!used) {
            continue;
        }
        if (leaf.offset > distance) {
            error = "retained consumer guard requires a future evaluation";
            return failure();
        }
        auto read = owner->history->read(leaf.requirement, distance - leaf.offset, cut, mapping, state, error);
        if (failed(read)) {
            return failure();
        }
        context.values[leaf.id] = read->value;
        context.cofactors[leaf.id] = leaf.id;
    }
    RegionExpressions::Substitution binding(substitutions);
    auto root = e.substitute(edge.guard, binding);
    root = e.land(e.le(e.constant(distance), demands.ordinal), root);
    for (auto [id, original] : e.referencedInputs(root)) {
        if (context.values.count(id)) {
            continue;
        }
        auto actual = mapping.lookupOrNull(original);
        if (actual && dominance.properlyDominates(original, cut)) {
            context.values[id] = actual;
            context.cofactors[id] = id;
        }
    }
    for (const auto& item : mapping.getValueMap()) {
        if (dominance.properlyDominates(item.first, cut)) {
            context.inputs[item.first] = item.second;
        }
    }
    Block scratch;
    OpBuilder detached(builder.getContext());
    detached.setInsertionPointToEnd(&scratch);
    auto emitted = e.emitContextual(root, detached, cut, context);
    if (succeeded(emitted)) {
        builder.getInsertionBlock()->getOperations().splice(builder.getInsertionPoint(), scratch.getOperations());
    }
    if (failed(emitted) && error.empty()) {
        error = e.lastEmissionError();
    }
    return emitted;
}
LogicalResult appendRecord(
    const std::shared_ptr<Owner>& owner, GuardedRankEdge edge, InvocationPrefixCollection& collection,
    std::string& error)
{
    auto& demands = *owner->demands;
    const auto distance = edge.target / demands.anchors.size();
    const auto target = edge.target % demands.anchors.size();
    auto* cut = demands.anchors[target].before.before;
    auto p = static_cast<uint32_t>(demands.anchors[edge.source].phase->kPipeValue);
    auto q = static_cast<uint32_t>(demands.anchors[target].phase->kPipeValue);
    if (p == q) {
        if (p == static_cast<uint32_t>(PIPE::PIPE_S)) {
            error = "scalar completion requires its existing prerequisite contract";
            return failure();
        }
        collection.recipe().actions.push_back(
            {cut, PrivateStateCut::Before,
             [owner, edge, cut, p](
                 OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value> state, std::string& error) {
                 auto active = guard(owner, edge, cut, b, mapping, state, error);
                 if (failed(active)) {
                     return failure();
                 }
                 auto branch = b.create<scf::IfOp>(cut->getLoc(), *active, false);
                 OpBuilder::InsertionGuard restore(b);
                 b.setInsertionPointToStart(branch.thenBlock());
                 b.create<BarrierOp>(cut->getLoc(), PipeAttr::get(b.getContext(), static_cast<PIPE>(p)));
                 return success();
             }});
        return success();
    }
    const auto rank = owner->ranks.at(edge.source);
    return collection.consumer(
        cut, {p, q},
        [owner, edge, cut, rank, distance](
            OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) -> FailureOr<Value> {
            auto active = guard(owner, edge, cut, b, mapping, state, error);
            if (failed(active)) {
                return failure();
            }
            auto zero = b.create<arith::ConstantIndexOp>(cut->getLoc(), 0);
            return b.create<arith::SelectOp>(cut->getLoc(), *active, state[rank.first + distance], zero).getResult();
        },
        error);
}
} // namespace
LogicalResult appendBoundedPrefixRealization(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, InvocationPrefixCollection& collection, std::string& error)
{
    const bool valid = demands && demands->function == collection.function() && demands->loop &&
                       demands->window.sites == demands->anchors.size();
    if (!valid) {
        error = "bounded prefix contribution has foreign original context";
        return failure();
    }
    for (const auto& anchor : demands->anchors) {
        auto* op = anchor.phase ? anchor.phase->elementOp : nullptr;
        const bool originalCut = op && anchor.before.block == op->getBlock() && anchor.before.before == op &&
                                 anchor.after.block == op->getBlock() && anchor.after.before == op->getNextNode();
        if (!originalCut) {
            error = "bounded prefix requires its original phase cut adapter";
            return failure();
        }
    }
    auto inactive = detail::boundedInactiveEmissionRecipes(
        demands->expressions, demands->window, demands->unconditional, demands->analysis.sourceDemands);
    if (!inactive) {
        error = "bounded prefix contribution has invalid retained records";
        return failure();
    }
    auto owner = std::make_shared<Owner>();
    owner->demands = std::move(demands);
    if (failed(appendHistory(*owner, collection.recipe(), error))) {
        return failure();
    }
    std::map<uint32_t, uint64_t> ranks;
    auto& e = owner->demands->expressions;
    {
        RegionExpressions::Transaction proof(e);
        DominanceInfo dominance(owner->demands->function);
        for (auto [record, edge] : llvm::enumerate(owner->demands->analysis.sourceDemands)) {
            if ((*inactive)[record]) {
                continue;
            }
            const auto distance = edge.target / owner->demands->anchors.size();
            if (!e.implies(edge.guard, owner->demands->window.payloads[edge.source].present)) {
                error =
                    proofDiagnostic("source-presence proof unavailable", record, edge, owner->demands->anchors.size());
                return failure();
            }
            auto inputs = e.referencedInputs(edge.guard);
            for (const auto& leaf : owner->leaves) {
                const bool used = llvm::any_of(inputs, [&](auto input) { return input.first == leaf.id; });
                const auto target = edge.target % owner->demands->anchors.size();
                auto* cut = owner->demands->anchors[target].before.before;
                const bool tooEarly = leaf.offset == distance && !evaluationPrecedes(leaf.original, cut, dominance);
                if (!used) {
                    continue;
                }
                const char* reason = nullptr;
                if (leaf.offset > distance) {
                    reason = "future predicate evaluation";
                } else if (tooEarly) {
                    reason = "same-visit predicate unavailable at consumer cut";
                } else if (!absentInsensitive(*owner, edge.guard, leaf)) {
                    reason = "skipped-definition masking proof unavailable";
                }
                if (reason) {
                    error = proofDiagnostic(reason, record, edge, owner->demands->anchors.size(), &leaf);
                    return failure();
                }
            }
            ranks[edge.source] = std::max(ranks[edge.source], distance);
        }
    }
    for (auto [site, lag] : ranks) {
        if (failed(appendRank(*owner, site, lag, collection, error))) {
            return failure();
        }
    }
    // Every executed payload phase advances its pipe count, not merely phases
    // that happen to source a retained record in this window.
    for (const auto& anchor : owner->demands->anchors) {
        if (failed(collection.source(anchor.phase, {}, error))) {
            return failure();
        }
    }
    for (auto [record, edge] : llvm::enumerate(owner->demands->analysis.sourceDemands)) {
        if (!(*inactive)[record] && failed(appendRecord(owner, edge, collection, error))) {
            return failure();
        }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
