// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One logical slot identity distributed independently to each original arm cut.
#include "PTO/Transforms/FrontierSynch/Shared/BalancedCompactBody.h"
#include "PTO/IR/PTO.h"
#include "CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
bool validPath(scf::ForOp loop, const BalancedCompactAlternative& alternative)
{
    auto* region = alternative.phase->elementOp->getParentRegion();
    for (auto* expected : llvm::reverse(alternative.path)) {
        if (region != expected || !isa<scf::IfOp>(region->getParentOp())) { return false; }
        region = region->getParentOp()->getParentRegion();
    }
    return region == &loop.getRegion();
}
bool validBody(func::FuncOp function, const BalancedCompactBody& body, const PeriodicAnalysis& analysis,
               ArrayRef<scf::ForOp> enclosing)
{
    if (!function || !body.error.empty() || !body.loop || !analysis.error.empty() ||
        body.slots.size() != analysis.payloads.size() || !function->isProperAncestor(body.loop)) { return false; }
    SmallVector<scf::ForOp> actual;
    for (auto* parent = body.loop->getParentOp(); parent != function; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast_or_null<scf::ForOp>(parent)) { actual.push_back(loop); }
        else if (!parent || !isa<scf::IfOp>(parent)) { return false; }
    }
    if (!llvm::equal(llvm::reverse(actual), enclosing) || (!actual.empty() &&
        llvm::any_of(body.slots, [](const auto& slot) { return slot.alternatives.size() != 1; }))) { return false; }
    llvm::DenseSet<const CompoundInstanceElement*> seen;
    DominanceInfo dominance(function);
    auto loop = body.loop;
    for (uint32_t slot = 0; slot < body.slots.size(); ++slot) {
        const auto& entry = body.slots[slot];
        if (entry.alternatives.empty() || entry.pipe != analysis.payloads[slot].pipe ||
            entry.pipe > static_cast<uint32_t>(PIPE::PIPE_FIX) || entry.pipe == static_cast<uint32_t>(PIPE::PIPE_ALL)) {
            return false;
        }
        for (const auto& alternative : entry.alternatives) {
            const auto mapping = body.phaseSlots.find(alternative.phase);
            if (!alternative.phase || !alternative.phase->elementOp || !seen.insert(alternative.phase).second ||
                mapping == body.phaseSlots.end() || mapping->second != slot) { return false; }
            auto* operation = alternative.phase->elementOp;
            if (static_cast<uint32_t>(alternative.phase->kPipeValue) != entry.pipe ||
                alternative.before.before != operation || alternative.before.block != operation->getBlock() ||
                !alternative.after.before || alternative.after.before != operation->getNextNode() ||
                alternative.after.block != operation->getBlock() || !body.loop->isProperAncestor(operation) ||
                !validPath(loop, alternative) || !dominance.properlyDominates(loop.getInductionVar(), operation)) {
                return false;
            }
        }
    }
    return seen.size() == body.phaseSlots.size();
}
class Preparer {
public:
    Preparer(const BalancedCompactBody& body, const LogicalEndpointPlan& logical, const CountedLoop& domain,
             int64_t planId)
        : body(body), logical(logical), domain(domain),
          plan(std::make_unique<PreparedLogicalPlan>(planId))
    {}
    std::unique_ptr<PreparedLogicalPlan> run();
private:
    bool families();
    void endpoint(const EndpointRecipe& recipe, const BalancedCompactAlternative& alternative);
    const BalancedCompactBody& body;
    const LogicalEndpointPlan& logical;
    const CountedLoop& domain;
    std::unique_ptr<PreparedLogicalPlan> plan;
    std::shared_ptr<RegionExpressions> expressions = std::make_shared<RegionExpressions>();
    RegionExpressions::Id trips = RegionExpressions::invalid, ordinal = RegionExpressions::invalid;
    SmallVector<ContextualEndpointFamily> endpoints;
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    std::map<Operation*, uint64_t> orders;
};
void Preparer::endpoint(const EndpointRecipe& recipe, const BalancedCompactAlternative& alternative)
{
    const bool publish = recipe.kind == EndpointKind::Set, local = recipe.kind == EndpointKind::Barrier;
    const auto cut = publish ? alternative.after : alternative.before;
    auto& arena = *expressions;
    const auto delay = arena.constant(recipe.displacement);
    ContextualEndpointFamily endpoint;
    endpoint.piece.cut = cut;
    endpoint.piece.kind = recipe.kind;
    endpoint.piece.sourcePipe = body.slots[recipe.source].pipe;
    endpoint.piece.targetPipe = body.slots[recipe.target].pipe;
    endpoint.piece.displacement = recipe.displacement;
    endpoint.piece.namespaceId = namespaces.at({endpoint.piece.sourcePipe, endpoint.piece.targetPipe,
                                               recipe.displacement});
    endpoint.piece.order = orders[cut.before]++;
    endpoint.piece.members.push_back({recipe.record, {}});
    if (publish) {
        endpoint.activity = arena.land(arena.lt(delay, trips), arena.lt(ordinal, arena.sub(trips, delay)));
        endpoint.ordinal = ordinal;
    } else {
        endpoint.activity = arena.le(delay, ordinal);
        if (!local) { endpoint.ordinal = arena.sub(ordinal, delay); }
    }
    // One owned side recipe per original alternative; never form source/target pairs.
    endpoints.push_back(std::move(endpoint));
}
bool Preparer::families()
{
    plan->groupedFamilies = true;
    plan->independentPieces = true;
    std::map<uint32_t, const EndpointRecipe*> records;
    for (const auto& recipe : logical.recipes) { records.try_emplace(recipe.record, &recipe); }
    for (const auto& [record, recipe] : records) {
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = body.slots[recipe->source].pipe;
        family.targetPipe = body.slots[recipe->target].pipe;
        family.local = family.sourcePipe == family.targetPipe;
        family.displacement = recipe->displacement;
        SmallVector<TemplateEndpointCut> sources, targets;
        for (const auto& alternative : body.slots[recipe->source].alternatives) {
            sources.push_back(alternative.after);
        }
        for (const auto& alternative : body.slots[recipe->target].alternatives) {
            targets.push_back(alternative.before);
        }
        auto sourceChoices = qualifyEndpointCutChoices(body.loop, sources);
        auto targetChoices = qualifyEndpointCutChoices(body.loop, targets);
        if (!sourceChoices || !targetChoices) { return false; }
        family.sourceCut = sources.front(); family.targetCut = targets.front();
        if (sources.size() > 1 && !family.local) { family.sourceChoices = std::move(sourceChoices); }
        if (targets.size() > 1) { family.targetChoices = std::move(targetChoices); }
        family.members.push_back({record, recipe->source, recipe->target, {}, {}});
        namespaces.try_emplace({family.sourcePipe, family.targetPipe, family.displacement}, record);
        plan->families.push_back(std::move(family));
    }
    return true;
}
std::unique_ptr<PreparedLogicalPlan> Preparer::run()
{
    if (!families()) { return nullptr; }
    if (logical.recipes.empty()) { return std::move(plan); }
    trips = domain.trips(*expressions);
    ordinal = domain.ordinal(*expressions);
    for (const auto& recipe : logical.recipes) {
        const auto slot = recipe.kind == EndpointKind::Set ? recipe.source : recipe.target;
        for (const auto& alternative : body.slots[slot].alternatives) { endpoint(recipe, alternative); }
    }
    auto loop = body.loop;
    auto recipe = buildContextualFamilyEndpointRecipe(expressions, endpoints,
        {{trips, loop}, {ordinal, &loop.getBody()->front()}});
    if (failed(recipe)) { return nullptr; }
    plan->familyRecipe = std::move(*recipe);
    return std::move(plan);
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBalancedCompactInsertion(
    func::FuncOp function, const BalancedCompactBody& body, const PeriodicAnalysis& analysis,
    int64_t planId, std::string& error, ArrayRef<scf::ForOp> enclosing)
{
    error.clear();
    if (planId < 0 || !validBody(function, body, analysis, enclosing)) {
        error = "balanced endpoint preparation has a stale body, incompatible slot word or enclosing invocation";
        return failure();
    }
    const auto domain = CountedLoop::get(body.loop);
    if (!domain) {
        error = "balanced endpoint arithmetic needs a represented 64-bit counted domain";
        return failure();
    }
    auto logical = buildLogicalEndpoints(analysis);
    if (!logical.error.empty()) { error = logical.error; return failure(); }
    for (const auto& recipe : logical.recipes) {
        if (recipe.kind == EndpointKind::Barrier && recipe.pipe == static_cast<uint32_t>(PIPE::PIPE_S)) {
            error = "balanced endpoint preparation cannot emit a scalar-pipe barrier";
            return failure();
        }
    }
    auto plan = Preparer(body, logical, *domain, planId).run();
    if (!plan) {
        error = "balanced slot cuts or counted coordinates are unavailable at original cuts";
        return failure();
    }
    return std::move(plan);
}
} // namespace mlir::pto::frontiersynch
