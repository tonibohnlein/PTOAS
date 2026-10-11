// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Fixed first/tail cells certify primitive facts before inspecting rank circuits.
#include "BoundedLifetimeAllocationContext.h"
#include "BoundedLifetimeInactiveRecords.h"
#include "../Shared/CountedLoop.h"
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch::detail {
namespace {
using Id = RegionExpressions::Id;
using Record = std::tuple<uint32_t, uint32_t, Id>;
std::vector<Record> records(llvm::ArrayRef<GuardedRankEdge> edges)
{
    std::vector<Record> result;
    for (const auto& edge : edges) { result.emplace_back(edge.source, edge.target, edge.guard); }
    return result;
}
std::vector<std::pair<uint32_t, Id>> payloads(const LifetimeWindowInput& window)
{
    std::vector<std::pair<uint32_t, Id>> result;
    for (const auto& payload : window.payloads) { result.emplace_back(payload.pipe, payload.present); }
    return result;
}
struct ScalarDefinition {
    Operation* operation;
    DictionaryAttr attributes;
    SmallVector<Value> operands;
};
void captureScalar(Value value, std::set<Operation*>& seen, std::vector<ScalarDefinition>& result)
{
    auto* operation = value.getDefiningOp();
    if (!operation || !seen.insert(operation).second) { return; }
    result.push_back({operation, operation->getAttrDictionary(),
        SmallVector<Value>(operation->getOperands().begin(), operation->getOperands().end())});
    for (auto operand : operation->getOperands()) { captureScalar(operand, seen, result); }
}
std::vector<ScalarDefinition> scalarDefinitions(const BoundedLifetimeDemandResult& owner)
{
    std::set<Operation*> seen;
    std::vector<ScalarDefinition> result;
    auto loop = owner.loop;
    for (auto value : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
        captureScalar(value, seen, result);
    }
    for (auto [id, recipe] : owner.predicateBindings) { (void)id; captureScalar(recipe.first, seen, result); }
    return result;
}
bool unchanged(llvm::ArrayRef<ScalarDefinition> definitions)
{
    return llvm::all_of(definitions, [](const ScalarDefinition& definition) {
        return definition.operation->getAttrDictionary() == definition.attributes &&
            llvm::equal(definition.operation->getOperands(), definition.operands);
    });
}
class CellProof {
public:
    explicit CellProof(const BoundedLifetimeDemandResult& owner)
        : owner(owner), e(owner.expressions), uniform(e, owner.window, owner.unconditional)
    {
        for (uint32_t site = 0; site < owner.window.sites; ++site) {
            if (owner.unconditional[site]) { legal = owner.window.payloads[site].present; break; }
        }
        collectInputs(legal);
        for (const auto& edge : owner.analysis.sourceDemands) { collectInputs(edge.guard); }
        for (auto [id, recipe] : owner.predicateBindings) { (void)recipe; collectInputs(id); }
        variables.assign(inputs.begin(), inputs.end());
    }
    std::vector<uint8_t> run()
    {
        std::vector<uint8_t> result(owner.analysis.sourceDemands.size(), true);
        for (bool first : {true, false}) {
            auto cell = e.eq(owner.base, e.constant(0));
            if (!first) { cell = e.lnot(cell); }
            auto universe = IntegerSystem::create(variables.size(), {});
            if (failed(universe)) { result.assign(result.size(), false); return result; }
            auto domains = e.integerRelation(e.land(legal, cell), variables, *universe, diagnostic);
            apply(domains, result);
        }
        return result;
    }
private:
    void collectInputs(Id root)
    {
        for (auto [id, value] : e.referencedInputs(root)) { (void)value; inputs.insert(id); }
    }
    bool empty(Id predicate, llvm::ArrayRef<IntegerSystem> domains)
    {
        for (const auto& domain : domains) {
            auto pieces = e.integerRelation(predicate, variables, domain, diagnostic);
            if (failed(pieces) || !llvm::all_of(*pieces, [](const auto& piece) { return piece.isEmpty(); })) {
                return false;
            }
        }
        return true;
    }
    std::vector<std::pair<Id, Id>> facts(llvm::ArrayRef<IntegerSystem> domains)
    {
        std::vector<std::pair<Id, Id>> result;
        std::set<Id> seen;
        for (auto [id, recipe] : owner.predicateBindings) {
            (void)recipe;
            if (!seen.insert(id).second) { continue; }
            if (empty(id, domains)) { result.emplace_back(id, e.boolean(false)); }
            else if (empty(e.lnot(id), domains)) { result.emplace_back(id, e.boolean(true)); }
        }
        return result;
    }
    void apply(const FailureOr<std::vector<IntegerSystem>>& domains, std::vector<uint8_t>& result)
    {
        if (failed(domains)) {
            for (std::size_t i = 0; i < result.size(); ++i) {
                result[i] &= uniform.proves(owner.analysis.sourceDemands[i].guard);
            }
            return;
        }
        if (domains->empty()) { return; }
        RegionExpressions::Substitution substitution(facts(*domains));
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto guard = e.substitute(owner.analysis.sourceDemands[i].guard, substitution);
            // No rank circuit is converted to an integer relation. The existing
            // prefix proof specializes only certified Boolean facts in its DAG.
            result[i] &= uniform.proves(guard);
        }
    }
    const BoundedLifetimeDemandResult& owner;
    RegionExpressions& e;
    BoundedInactiveRecordProof uniform;
    Id legal = RegionExpressions::invalid;
    std::set<Id> inputs;
    SmallVector<Id> variables;
    std::string diagnostic;
};
bool valid(const BoundedLifetimeDemandResult& owner)
{
    const auto& w = owner.window;
    if (!owner.function || !owner.loop || !CountedLoop::get(owner.loop) || owner.analysis.error.size() ||
        owner.loop->getParentOfType<func::FuncOp>() != owner.function || !w.sites ||
        w.sites != owner.anchors.size() || owner.unconditional.size() != w.sites ||
        w.span >= UINT32_MAX || w.sites > UINT32_MAX / (w.span + 1) ||
        w.payloads.size() != w.sites * (w.span + 1) || owner.base >= owner.expressions.size() ||
        owner.expressions.isBoolean(owner.base)) { return false; }
    for (const auto& payload : w.payloads) {
        if (!owner.expressions.isBoolean(payload.present)) { return false; }
    }
    for (auto [id, recipe] : owner.predicateBindings) {
        if (!owner.expressions.isBoolean(id) || !recipe.first || recipe.second > w.span) { return false; }
    }
    return llvm::any_of(owner.unconditional, [](uint8_t value) { return value != 0; });
}
} // namespace
struct BoundedLifetimeAllocationContext::State {
    State(std::shared_ptr<const BoundedLifetimeDemandResult> owner, std::shared_ptr<const SyncInput> input)
        : owner(std::move(owner)), input(std::move(input)), loop(this->owner->loop),
          lower(loop.getLowerBound()), upper(loop.getUpperBound()), step(loop.getStep()), base(this->owner->base),
          originalRecords(records(this->owner->analysis.sourceDemands)),
          originalPayloads(payloads(this->owner->window)), sites(this->owner->window.sites),
          span(this->owner->window.span), unconditional(this->owner->unconditional),
          bindings(this->owner->predicateBindings), definitions(scalarDefinitions(*this->owner)) {}
    bool matches(RegionExpressions& arena, const LifetimeWindowInput& window, llvm::ArrayRef<uint8_t> flags,
                 llvm::ArrayRef<GuardedRankEdge> demands) const
    {
        auto current = loop;
        return &arena == owner->arena.get() && &window == &owner->window && owner->input == input.get() &&
            owner->loop == loop && current.getLowerBound() == lower && current.getUpperBound() == upper &&
            current.getStep() == step && owner->base == base && valid(*owner) &&
            demands.data() == owner->analysis.sourceDemands.data() && records(demands) == originalRecords &&
            flags.data() == owner->unconditional.data() && flags == llvm::ArrayRef<uint8_t>(unconditional) &&
            owner->window.sites == sites && owner->window.span == span &&
            payloads(window) == originalPayloads && owner->predicateBindings == bindings && unchanged(definitions);
    }
    std::shared_ptr<const BoundedLifetimeDemandResult> owner;
    std::shared_ptr<const SyncInput> input;
    scf::ForOp loop;
    Value lower, upper, step;
    Id base;
    std::vector<Record> originalRecords;
    std::vector<std::pair<uint32_t, Id>> originalPayloads;
    uint32_t sites;
    uint64_t span;
    std::vector<uint8_t> unconditional;
    std::vector<std::pair<Id, std::pair<Value, uint64_t>>> bindings;
    std::vector<ScalarDefinition> definitions;
    std::optional<std::vector<uint8_t>> proofs;
    uint64_t builds = 0;
};
std::shared_ptr<BoundedLifetimeAllocationContext> BoundedLifetimeAllocationContext::create(
    std::shared_ptr<const BoundedLifetimeDemandResult> owner, std::shared_ptr<const SyncInput> input)
{
    if (!owner || !input || owner->input != input.get() || !valid(*owner)) { return {}; }
    return std::shared_ptr<BoundedLifetimeAllocationContext>(
        new BoundedLifetimeAllocationContext(std::make_shared<State>(std::move(owner), std::move(input))));
}
std::optional<llvm::ArrayRef<uint8_t>> BoundedLifetimeAllocationContext::inactive(
    RegionExpressions& arena, const LifetimeWindowInput& window, llvm::ArrayRef<uint8_t> unconditional,
    llvm::ArrayRef<GuardedRankEdge> demands)
{
    if (!state->matches(arena, window, unconditional, demands)) { return {}; }
    if (!state->proofs) {
        // Only owned booleans escape. All temporary IDs and proof-local caches
        // die before rollback; retained producer/query caches are never written.
        RegionExpressions::Transaction transaction(arena);
        state->proofs = CellProof(*state->owner).run();
        if (!arena.constructionError().empty()) {
            state->proofs->assign(demands.size(), false);
        }
        ++state->builds;
    }
    return llvm::ArrayRef<uint8_t>(*state->proofs);
}
uint64_t BoundedLifetimeAllocationContext::proofBuilds() const { return state->builds; }
} // namespace mlir::pto::frontiersynch::detail
