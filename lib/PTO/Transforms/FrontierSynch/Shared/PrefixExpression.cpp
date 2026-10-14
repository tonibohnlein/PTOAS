// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Realize retained Boolean evaluations without evaluating opaque producers again.
#include "PrefixExpression.h"
#include "MappedCutInputs.h"
#include "PrefixExpressionProfile.h"
#include "PrefixExpressionSession.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
struct Binding {
    Expr input;
    scf::ForOp loop;
    Operation* definition;
    int64_t lower;
    Expr ordinal;
    std::size_t valueSlot, validSlot;
};
std::optional<int64_t> literal(Value value)
{
    APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer)) || !integer.isSignedIntN(64)) { return {}; }
    return integer.getSExtValue();
}
bool sameFrame(scf::ForOp loop, Operation* cut, DominanceInfo& dominance)
{
    if (!loop->isProperAncestor(cut) && !dominance.properlyDominates(loop.getOperation(), cut)) { return false; }
    for (auto* parent = loop->getParentOp(); !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
        if (isa<scf::ForOp, scf::WhileOp>(parent) && !parent->isProperAncestor(cut)) { return false; }
    }
    return true;
}
std::vector<Binding> collectBindings(const std::shared_ptr<RegionExpressions>& owner, Expr root, Operation* cut,
                                    InvocationPrefixCollection& collection)
{
    auto& arena = *owner;
    std::vector<Binding> bindings;
    DominanceInfo dominance(collection.function());
    for (auto [id, unused] : arena.referencedInputs(root)) {
        auto evaluation = arena.evaluation(id);
        if (!evaluation) { continue; }
        auto loop = dyn_cast_or_null<scf::ForOp>(evaluation->loop);
        auto* definition = evaluation->original.getDefiningOp();
        if (!loop || !definition || definition->getBlock() != loop.getBody() ||
            !collection.function()->isProperAncestor(loop) || !sameFrame(loop, cut, dominance)) { continue; }
        const auto lower = literal(loop.getLowerBound()), step = literal(loop.getStep());
        // Unit-step signed loops cannot wrap on the final increment: IV<upper
        // implies IV+1<=upper<=INT64_MAX. Other strides need a retained proof.
        if (!lower || step != 1 || !loop.getInductionVar().getType().isIndex()) { continue; }
        std::string error;
        auto slots = collection.snapshotPredicate(evaluation->original, loop, owner, evaluation->ordinal, error);
        if (failed(slots)) { continue; }
        bindings.push_back({id, loop, definition, *lower, evaluation->ordinal, slots->first, slots->second});
    }
    return bindings;
}
// Each proof leaves every other historical input unconstrained. The premise
// contains only numeric control bounds, so the certificates compose when
// several unavailable evaluations are replaced simultaneously.
bool independentWhenUnavailable(RegionExpressions& arena, Expr root, Operation* cut, Binding binding)
{
    RegionExpressions::Transaction scratch(arena);
    auto available = arena.boolean(false), domain = arena.boolean(true);
    const auto selected = binding.ordinal;
    const auto lower = arena.constant(static_cast<uint64_t>(binding.lower));
    const auto upper = arena.input(binding.loop.getUpperBound());
    const auto exists = arena.land(arena.slt(lower, upper), arena.lt(selected, arena.sub(upper, lower)));
    if (binding.loop->isProperAncestor(cut)) {
        const auto current = arena.input(binding.loop.getInductionVar());
        const auto ordinal = arena.sub(current, lower);
        available = arena.lt(selected, ordinal);
        DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
        if (dominance.properlyDominates(binding.definition, cut)) { available = arena.le(selected, ordinal); }
        domain = arena.land(arena.sle(arena.constant(static_cast<uint64_t>(binding.lower)), current),
                            arena.slt(current, arena.input(binding.loop.getUpperBound())));
    } else {
        available = exists;
    }
    // Keep a simple signed comparison when an absolute selected IV is known.
    // This is equivalent on an executed visit, and avoids asking the proof
    // engine to rediscover the unsigned-distance identity for constant visits.
    if (auto ordinal = arena.constantValue(selected)) {
        const APInt actual = APInt(128, static_cast<uint64_t>(binding.lower), true) + APInt(128, *ordinal);
        if (actual.isSignedIntN(64)) {
            const auto induction = arena.constant(static_cast<uint64_t>(actual.getSExtValue()));
            if (binding.loop->isProperAncestor(cut)) {
                const auto current = arena.input(binding.loop.getInductionVar());
                DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
                available = dominance.properlyDominates(binding.definition, cut) ?
                    arena.sle(induction, current) : arena.slt(induction, current);
            } else { available = arena.slt(induction, upper); }
        }
    }
    if (arena.constantUnder(domain, available) == 1) { return true; }
    RegionExpressions::Substitution no({{binding.input, arena.boolean(false)}});
    RegionExpressions::Substitution yes({{binding.input, arena.boolean(true)}});
    const auto a = arena.substitute(root, no), b = arena.substitute(root, yes);
    const auto differs = arena.land(domain, arena.land(arena.lnot(available), arena.lnot(arena.eq(a, b))));
    return arena.constantValue(differs) == 0 || arena.inactiveByThresholds(differs);
}
RegionExpressions::CutEmission contextFor(RegionExpressions& arena, Expr root, Operation* cut,
                                          const IRMapping& mapping)
{
    RegionExpressions::CutEmission context;
    DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
    detail::bindMappedCutInputs(arena, root, cut, mapping, dominance, context);
    return context;
}
struct ExpressionRecipe {
    std::shared_ptr<RegionExpressions> arena;
    Expr root;
    Operation* cut;
    std::vector<Binding> bindings;
    std::optional<bool> historyProved;
    std::shared_ptr<PrefixExpressionProfile> profile;
    std::shared_ptr<PrefixExpressionSession> session;
    PrefixExpressionProfile::Entry* work = nullptr;
    FailureOr<Value> emit(OpBuilder& builder, const IRMapping& mapping,
                          ArrayRef<Value> state, std::string& error)
    {
        auto& e = *arena;
        accumulateCost(work->emissions, 1);
        Block scratch;
        OpBuilder detached(builder.getContext()); detached.setInsertionPointToEnd(&scratch);
        auto attempt = [&]() {
            accumulateCost(work->attempts, 1);
            accumulateCost(work->constructed, scratch.getOperations().size());
        };
        auto context = contextFor(e, root, cut, mapping);
        RegionExpressions::CutEmission* accepted = nullptr;
        auto lower = [&](PrefixExpressionSession::Environment environment) -> FailureOr<Value> {
            auto* shared = session->bindings(&e, cut, builder.getInsertionBlock(), mapping, std::move(environment));
            auto lookupValue = [&](Expr id) { return shared ? shared->values.lookup(id) : Value{}; };
            auto lookupInput = [&](Value input) { return shared ? shared->inputs.lookup(input) : Value{}; };
            auto hasValue = [&](Expr id) { return bool(lookupValue(id)); };
            RegionExpressions::SharedCutPlanning planning{hasValue};
            RegionExpressions::SharedCutBindings emitted{lookupValue, lookupInput};
            auto recipe = e.prepareContextual(root, cut, context, shared ? &planning : nullptr);
            if (failed(recipe)) { return failure(); }
            auto result = e.emitPreparedContextual(**recipe, detached, cut, context, shared ? &emitted : nullptr);
            if (succeeded(result)) { accepted = shared; }
            return result;
        };
        auto value = lower({});
        attempt();
        if (failed(value)) {
            accumulateCost(work->directFailures, 1);
            const std::string directError = e.lastEmissionError();
            scratch.clear(); detached.setInsertionPointToEnd(&scratch);
            if (e.inactiveByThresholds(root)) {
                accumulateCost(work->inactiveAttempts, 1);
                RegionExpressions::CutEmission inactive;
                value = e.emitContextual(e.boolean(false), detached, cut, inactive);
                attempt();
            } else {
                if (!historyProved) {
                    historyProved = !bindings.empty() && llvm::all_of(bindings, [&](const Binding& binding) {
                        return independentWhenUnavailable(e, root, cut, binding);
                    });
                }
                if (!*historyProved) { error = directError; return failure(); }
                accumulateCost(work->historyAttempts, 1);
                context = contextFor(e, root, cut, mapping);
                PrefixExpressionSession::Environment environment;
                for (const auto& binding : bindings) {
                    if (binding.validSlot >= state.size() || !state[binding.valueSlot].getType().isInteger(1) ||
                        !state[binding.validSlot].getType().isInteger(1)) {
                        error = "historical expression lost its invocation state"; return failure();
                    }
                    // The independent cofactor proof justifies any defined bit
                    // while invalid. Validity is retained, never assumed true.
                    context.values[binding.input] = state[binding.valueSlot];
                    context.cofactors[binding.input] = binding.input;
                    environment.emplace_back(binding.input, state[binding.valueSlot].getAsOpaquePointer());
                }
                value = lower(std::move(environment));
                attempt();
            }
            if (failed(value)) { error = e.lastEmissionError(); return failure(); }
        }
        const auto kept = scratch.getOperations().size();
        accumulateCost(work->retained, kept);
        work->maximumRetained = std::max(work->maximumRetained, static_cast<uint64_t>(kept));
        builder.getInsertionBlock()->getOperations().splice(builder.getInsertionPoint(), scratch.getOperations());
        if (accepted) {
            for (auto item : context.values) { accepted->values.try_emplace(item.first, item.second); }
            for (auto item : context.inputs) { accepted->inputs.try_emplace(item.first, item.second); }
        }
        return *value;
    }
};
} // namespace
PrefixExpressionProfile::Entry& PrefixExpressionProfile::request(
    const void* arena, uint32_t root, Operation* cut, const std::string& producer, uint64_t distance)
{
    auto [position, added] = entries.try_emplace({arena, root, cut, producer, distance});
    auto& entry = position->second;
    if (added) {
        entry.arena = arenas.try_emplace(arena, arenas.size()).first->second;
        entry.root = root; entry.producer = producer; entry.distance = distance;
        entry.cut = cut->getName().getStringRef().str();
        llvm::raw_string_ostream location(entry.location);
        cut->getLoc().print(location);
    }
    accumulateCost(entry.requests, 1);
    return entry;
}
void PrefixExpressionProfile::resetEmission()
{
    for (auto& [key, entry] : entries) {
        entry.emissions = entry.attempts = entry.directFailures = entry.inactiveAttempts = entry.historyAttempts = 0;
        entry.constructed = entry.retained = entry.maximumRetained = 0;
    }
}
llvm::json::Object PrefixExpressionProfile::report() const
{
    Entry totals;
    std::vector<const Entry*> largest;
    for (const auto& [key, entry] : entries) {
        accumulateCost(totals.requests, entry.requests);
        accumulateCost(totals.emissions, entry.emissions);
        accumulateCost(totals.attempts, entry.attempts);
        accumulateCost(totals.directFailures, entry.directFailures);
        accumulateCost(totals.inactiveAttempts, entry.inactiveAttempts);
        accumulateCost(totals.historyAttempts, entry.historyAttempts);
        accumulateCost(totals.constructed, entry.constructed);
        accumulateCost(totals.retained, entry.retained);
        largest.push_back(&entry);
    }
    llvm::sort(largest, [](const Entry* a, const Entry* b) {
        if (a->constructed != b->constructed) { return a->constructed > b->constructed; }
        return std::tie(a->arena, a->root, a->location) < std::tie(b->arena, b->root, b->location);
    });
    auto counts = [](const Entry& entry) {
        return llvm::json::Object{{"requests", entry.requests}, {"emissions", entry.emissions},
            {"attempts", entry.attempts}, {"direct_failures", entry.directFailures},
            {"inactive_attempts", entry.inactiveAttempts}, {"history_attempts", entry.historyAttempts},
            {"constructed_operations", entry.constructed}, {"retained_operations", entry.retained}};
    };
    auto result = counts(totals);
    result["unique_recipes"] = entries.size();
    result["scope"] = "latest-materialization-attempt";
    llvm::json::Array top;
    for (const auto* entry : ArrayRef<const Entry*>(largest).take_front(16)) {
        auto item = counts(*entry);
        item["arena"] = entry->arena; item["root"] = entry->root;
        item["producer"] = entry->producer; item["distance"] = entry->distance;
        item["cut"] = entry->cut; item["location"] = entry->location;
        item["maximum_retained_operations"] = entry->maximumRetained;
        top.push_back(std::move(item));
    }
    result["largest_recipes"] = std::move(top);
    return result;
}
PrefixIncomingSelector preparePrefixExpression(std::shared_ptr<RegionExpressions> arena,
    Expr root, Operation* cut, InvocationPrefixCollection& collection)
{
    auto recipe = std::make_shared<ExpressionRecipe>();
    recipe->arena = std::move(arena); recipe->root = root; recipe->cut = cut;
    auto& profile = collection.recipe().expressionProfile;
    if (!profile) { profile = std::make_shared<PrefixExpressionProfile>(); }
    recipe->profile = profile;
    auto& session = collection.recipe().expressionSession;
    if (!session) { session = std::make_shared<PrefixExpressionSession>(); }
    recipe->session = session;
    recipe->work = &profile->request(recipe->arena.get(), root, cut);
    recipe->bindings = collectBindings(recipe->arena, root, cut, collection);
    return [recipe](OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) {
        return recipe->emit(b, mapping, state, error);
    };
}
} // namespace mlir::pto::frontiersynch
