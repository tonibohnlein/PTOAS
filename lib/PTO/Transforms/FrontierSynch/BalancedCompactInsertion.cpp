// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One logical slot identity distributed independently to each original arm cut.
#include "PTO/Transforms/FrontierSynch/BalancedCompactBody.h"
#include "PTO/IR/PTO.h"
#include "CountedLoop.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
        : body(body), logical(logical), domain(domain), builder(body.loop->getContext()),
          plan(std::make_unique<PreparedLogicalPlan>(planId))
    {}
    std::unique_ptr<PreparedLogicalPlan> run();
private:
    Value number(uint64_t value)
    {
        auto type = builder.getIndexType();
        return builder.create<arith::ConstantOp>(body.loop->getLoc(), type, IntegerAttr::get(type, APInt(64, value)));
    }
    Value compare(arith::CmpIPredicate predicate, Value left, Value right)
    {
        return builder.create<arith::CmpIOp>(body.loop->getLoc(), predicate, left, right);
    }
    bool initialize();
    bool families();
    void endpoint(const EndpointRecipe& recipe, const BalancedCompactAlternative& alternative);
    const BalancedCompactBody& body;
    const LogicalEndpointPlan& logical;
    const CountedLoop& domain;
    OpBuilder builder;
    std::unique_ptr<PreparedLogicalPlan> plan;
    Value trips, ordinal;
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    std::map<Operation*, Block*> blocks;
};
bool Preparer::initialize()
{
    RegionExpressions expressions;
    const auto count = domain.trips(expressions), visit = domain.ordinal(expressions);
    auto loop = body.loop;
    builder.setInsertionPointToEnd(&plan->addPreparation(loop));
    llvm::DenseMap<RegionExpressions::Id, Value> memo;
    auto emittedTrips = expressions.emit(count, builder, loop, memo);
    if (failed(emittedTrips)) { return false; }
    trips = *emittedTrips;
    auto* bodyCut = &loop.getBody()->front();
    builder.setInsertionPointToEnd(&plan->addPreparation(bodyCut));
    memo.clear();
    auto emittedOrdinal = expressions.emit(visit, builder, bodyCut, memo);
    if (failed(emittedOrdinal)) { return false; }
    ordinal = *emittedOrdinal;
    return true;
}
void Preparer::endpoint(const EndpointRecipe& recipe, const BalancedCompactAlternative& alternative)
{
    const bool publish = recipe.kind == EndpointKind::Set, local = recipe.kind == EndpointKind::Barrier;
    auto* cut = publish ? alternative.after.before : alternative.before.before;
    auto [block, added] = blocks.try_emplace(cut);
    if (added) { block->second = &plan->addPreparation(cut); }
    builder.setInsertionPointToEnd(block->second);
    auto location = cut->getLoc();
    auto delay = number(recipe.displacement);
    Value guard, identity;
    if (publish) {
        auto remaining = builder.create<arith::SubIOp>(location, trips, delay);
        auto hasTarget = compare(arith::CmpIPredicate::ult, delay, trips);
        auto beforeEnd = compare(arith::CmpIPredicate::ult, ordinal, remaining);
        guard = builder.create<arith::AndIOp>(location, hasTarget, beforeEnd);
        identity = ordinal;
    } else {
        guard = compare(arith::CmpIPredicate::uge, ordinal, delay);
        if (!local) { identity = builder.create<arith::SubIOp>(location, ordinal, delay); }
    }
    const auto kind = local ? LogicalCommandKind::Barrier :
        publish ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
    // No source/target alternative pair is formed. The slot record, source
    // coordinate and pipes are identical at every copy of this one side.
    plan->endpoints.push_back({cut, kind, body.slots[recipe.source].pipe,
                              body.slots[recipe.target].pipe,
                              namespaces.at({body.slots[recipe.source].pipe, body.slots[recipe.target].pipe,
                                             recipe.displacement}), guard, identity});
    auto& endpoint = plan->endpoints.back();
    if (!local) { endpoint.memberCoordinates.push_back(number(recipe.record)); }
    endpoint.records.push_back(recipe.record);
    endpoint.piece = static_cast<int64_t>(plan->endpoints.size() - 1);
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
    if (!initialize()) { return nullptr; }
    for (const auto& recipe : logical.recipes) {
        const auto slot = recipe.kind == EndpointKind::Set ? recipe.source : recipe.target;
        for (const auto& alternative : body.slots[slot].alternatives) { endpoint(recipe, alternative); }
    }
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
