// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/StorageLaneAllocation.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "CountedLoop.h"
#include "IterationPredicates.h"
#include "PhysicalRefreshLimits.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "CircuitEndpoints.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include <algorithm>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
BoundedLifetimeDemandResult::BoundedLifetimeDemandResult(std::shared_ptr<RegionExpressions> arena)
    : arena(std::move(arena)), expressions(*this->arena) {}
BoundedLifetimeDemandResult::~BoundedLifetimeDemandResult() = default;
using Id = RegionExpressions::Id;
FailureOr<std::shared_ptr<BoundedLifetimeDemandResult>> analyzeBoundedLifetimeRegion(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error, std::shared_ptr<RegionExpressions> arena)
{
    if (!function || !loop || loop->getParentOfType<func::FuncOp>() != function) {
        error = "bounded lifetime region must belong to its original function"; return failure();
    }
    const auto& skeleton = recognized.skeleton;
    auto domain = CountedLoop::get(loop);
    if (skeleton.result.state != RecognitionState::Applicable || !recognized.refresh.error.empty() || !domain) {
        error = "bounded lifetime requires a certified skeleton, refresh and counted domain";
        return failure();
    }
    const auto m = skeleton.phases.size(), span = recognized.refresh.span;
    if (!m || span >= UINT32_MAX || m > UINT32_MAX / (span + 1)) {
        error = "bounded lifetime window cannot represent its potential occurrences";
        return failure();
    }
    if (recognized.physical && !detail::physicalRefreshWindowFits(span,
            skeleton.phases.size(), skeleton.guards.size(), recognized.physical->accesses.size())) {
        error = "physical refresh window exceeds numerical fragment representation"; return failure();
    }
    if (!arena) { arena = std::make_shared<RegionExpressions>(); }
    auto demands = std::make_shared<BoundedLifetimeDemandResult>(std::move(arena));
    demands->function = function; demands->loop = loop; demands->input = &input;
    auto& e = demands->expressions;
    auto& symbols = demands->symbols;
    const auto base = demands->base =
        e.input(symbols.addArgument(IndexType::get(function.getContext()), function.getLoc()));
    auto trips = domain->trips(e);
    demands->ordinal =
        e.div(e.sub(e.input(loop.getInductionVar()), e.input(loop.getLowerBound())), e.constant(domain->step));
    SmallVector<const CompoundInstanceElement*> phases;
    auto& anchors = demands->anchors;
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    DenseMap<Operation*, uint32_t> operationIds;
    for (const auto& site : skeleton.phases) {
        positions[site.phase] = phases.size();
        phases.push_back(site.phase);
        auto* op = site.phase->elementOp;
        operationIds.try_emplace(op, operationIds.size() + 1);
        anchors.push_back({site.phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
    }
    demands->predicates = std::make_unique<IterationPredicates>(loop, e, phases);
    auto& predicates = *demands->predicates;
    auto& window = demands->window;
    window.sites = m;
    window.span = span;
    window.storageProtection = ptoStorageProtection();
    const auto protection = structuredProtection(input.accesses());
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> groups;
    auto& predicateBindings = demands->predicateBindings;
    for (uint64_t d = 0; d <= span; ++d) {
        std::vector<Id> guards;
        for (const auto& guard : skeleton.guards) {
            auto value = predicates.at(guard.condition, e.add(base, e.constant(d)));
            predicateBindings.push_back({value, {guard.condition, d}});
            if (!guard.takeThen) {
                value = e.lnot(value);
            }
            if (guard.parent) {
                value = e.select(guards[*guard.parent], value, e.boolean(false));
            }
            guards.push_back(value);
        }
        for (const auto& site : skeleton.phases) {
            auto active = e.land(e.lt(base, trips), e.lt(e.constant(d), e.sub(trips, base)));
            if (site.guard) {
                active = e.select(active, guards[*site.guard], e.boolean(false));
            }
            window.payloads.push_back({static_cast<uint32_t>(site.phase->kPipeValue), active});
            window.operations.push_back(d * operationIds.size() + operationIds.lookup(site.phase->elementOp));
        }
    }
    DenseMap<Value, uint32_t> families;
    std::map<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>, uint32_t> cells;
    auto& physicalCells = demands->physicalCells;
    auto protectionGroup = [&](const RotatingAccess& access, uint64_t distance) {
        const auto& effect = input.accesses().effects()[access.effect];
        uint64_t group = effect.memory && effect.memory->scope == AddressSpace::ACC && access.stride == 0 ?
                             protection.inLoop(effect.phase, loop) : 0;
        if (group) {
            auto scope = group & invocationProtectionBit ? 0 : distance;
            auto reset = group & protectionResetBit;
            group = groups.emplace(std::make_pair(group & ~protectionResetBit, scope), groups.size() + 1)
                        .first->second | reset;
        }
        return group;
    };
    if (recognized.physical) {
        const auto& physical = *recognized.physical;
        if (!physical.period || physical.period > span || physical.atoms.size() > UINT32_MAX) {
            error = "physical refresh certificate has invalid period or atom dimensions"; return failure();
        }
        for (auto [cell, atom] : llvm::enumerate(physical.atoms)) {
            // These are absolute physical cells. No source-relative family
            // renaming is needed; the fixed map also serves allocation.
            physicalCells.push_back({static_cast<uint32_t>(cell), static_cast<uint32_t>(atom.space),
                                     atom.begin, atom.end, 1, 0, 0});
        }
        for (uint64_t d = 0; d <= span; ++d) {
            auto phase = e.rem(e.add(base, e.constant(d)), e.constant(physical.period));
            for (const auto& projection : physical.accesses) {
                if (projection.access >= skeleton.result.accesses.size() || projection.atom >= physical.atoms.size()) {
                    error = "physical refresh certificate has invalid access identity"; return failure();
                }
                const auto& access = skeleton.result.accesses[projection.access];
                const auto& effect = input.accesses().effects()[access.effect];
                auto found = positions.find(effect.phase);
                if (found == positions.end()) { error = "physical refresh access has no payload"; return failure(); }
                auto active = e.eq(phase, e.constant(projection.phase));
                window.accesses.push_back({static_cast<uint32_t>(d * m + found->second), projection.atom,
                    e.land(active, e.boolean(access.reads)), e.land(active, e.boolean(access.writes)),
                    protectionGroup(access, d)});
            }
        }
    } else {
        for (uint64_t d = 0; d <= span; ++d) {
            for (const auto& access : skeleton.result.accesses) {
                const auto& effect = input.accesses().effects()[access.effect];
                auto found = positions.find(effect.phase);
                if (found == positions.end() || !access.atom || !access.slots) {
                    error = "bounded lifetime access has no normalized occurrence";
                    return failure();
                }
                // Common-stride storage renaming cancels the unknown source index.
                auto slot = (APInt(128, access.stride) * APInt(128, d) + APInt(128, access.offset))
                                .urem(APInt(128, access.slots))
                                .getZExtValue();
                const auto family = families.try_emplace(access.family, families.size()).first->second;
                auto [entry, added] = cells.emplace(
                    std::make_tuple(family, access.atom->first, access.atom->second, slot), cells.size());
                const auto cell = entry->second;
                if (added) {
                    physicalCells.push_back({cell, family, access.atom->first, access.atom->second,
                        access.slots, access.stride % access.slots, slot});
                }
                auto group = protectionGroup(access, d);
                window.accesses.push_back(
                    {static_cast<uint32_t>(d * m + found->second), cell,
                     e.boolean(access.reads), e.boolean(access.writes),
                     group});
            }
        }
    }
    auto prerequisites = index.mapPrerequisites(phases);
    if (!prerequisites.error.empty()) {
        error = prerequisites.error;
        return failure();
    }
    for (uint64_t d = 0; d <= span; ++d) {
        for (auto edge : prerequisites.native) {
            window.nativePrerequisites.push_back(
                {static_cast<uint32_t>(d * m + edge.source), static_cast<uint32_t>(d * m + edge.target),
                 e.boolean(true)});
        }
        for (auto edge : prerequisites.demands) {
            window.prerequisites.push_back(
                {static_cast<uint32_t>(d * m + edge.source), static_cast<uint32_t>(d * m + edge.target),
                 e.boolean(true)});
        }
    }
    demands->analysis = analyzeLifetimeWindow(e, window);
    auto& analysis = demands->analysis;
    if (!analysis.error.empty()) {
        error = analysis.error;
        return failure();
    }
    for (const auto& site : skeleton.phases) { demands->unconditional.push_back(!site.guard); }
    return demands;
}
FailureOr<std::shared_ptr<BoundedLifetimeDemandResult>> cachedBoundedLifetimeRegion(
    func::FuncOp function, const StructureNode& node, const PhaseIndex& index,
    const SyncInput& input, std::string& error)
{
    auto loop = dyn_cast_or_null<scf::ForOp>(node.anchor);
    if (!loop || !node.boundedLifetime) { error = "no bounded loop certificate"; return failure(); }
    if (!node.boundedDemandAttempted) {
        auto analyzed = analyzeBoundedLifetimeRegion(function, loop, index, input,
            *node.boundedLifetime, node.boundedDemandError);
        node.boundedDemandAttempted = true;
        if (succeeded(analyzed)) { node.boundedDemands = *analyzed; }
    }
    error = node.boundedDemandError;
    if (!node.boundedDemands) { return failure(); }
    if (node.boundedDemands->function != function || node.boundedDemands->loop != loop ||
        node.boundedDemands->input != &input) {
        error = "bounded cache belongs to different original modeled input"; return failure();
    }
    return node.boundedDemands;
}
struct BoundedLifetimeAllocationRecipe {
    StorageLaneAllocation storage;
    DictionaryAttr storageCertificate;
    std::optional<DictionaryAttr> ordinaryCertificate;
    std::optional<bool> storageCoordinatesAvailable;
};
namespace {
Id bindBoundedExpression(BoundedLifetimeDemandResult& demands, Id root, uint64_t distance, bool consumer)
{
    auto& e = demands.expressions;
    const auto ordinal = demands.ordinal;
    auto target = consumer ? e.sub(ordinal, e.constant(distance)) : ordinal;
    std::vector<std::pair<Id, Id>> bindings{{demands.base, target}};
    for (auto [id, recipe] : demands.predicateBindings) {
        auto shift = static_cast<int64_t>(recipe.second) - (consumer ? int64_t(distance) : 0);
        auto evaluation = shift < 0 ? e.sub(ordinal, e.constant(-shift)) : e.add(ordinal, e.constant(shift));
        bindings.push_back({id, demands.predicates->at(recipe.first, evaluation)});
    }
    RegionExpressions::Substitution substitution(bindings);
    return e.substitute(root, substitution);
}
std::shared_ptr<BoundedLifetimeAllocationRecipe> boundedAllocationRecipe(BoundedLifetimeDemandResult& demands)
{
    if (demands.allocationRecipe) { return demands.allocationRecipe; }
    auto recipe = std::make_shared<BoundedLifetimeAllocationRecipe>();
    demands.allocationRecipe = recipe;
    auto& storage = recipe->storage;
    storage = buildStorageLaneAllocation(demands.expressions, demands.window, demands.analysis,
                                         demands.physicalCells, demands.base, true);
    const bool storageAvailable = storage.error.empty() && storage.budget && !storage.records.empty();
    if (!storageAvailable) { return recipe; }
    SmallVector<GuardedRankEdge> residual;
    SmallVector<uint32_t> originalRecords;
    const auto& edges = demands.analysis.sourceDemands;
    for (std::size_t r = 0; r < edges.size(); ++r) {
        if (!std::binary_search(storage.records.begin(), storage.records.end(), r)) {
            if (r > UINT32_MAX) { return recipe; }
            residual.push_back(edges[r]); originalRecords.push_back(r);
        }
    }
    auto storageOnly = storageLaneAllocationCertificate(demands.function, demands.window, edges, storage, 0);
    auto remainder = boundedLifetimeAllocationCertificate(demands.function, demands.expressions, demands.window,
                                                          demands.unconditional, residual, 0, originalRecords);
    recipe->storageCertificate = combineStorageLaneCertificates(storageOnly, remainder);
    return recipe;
}
DictionaryAttr bindAllocationPlan(DictionaryAttr certificate, int64_t plan, MLIRContext* context)
{
    if (!certificate) { return {}; }
    NamedAttrList attributes(certificate);
    attributes.set("plan", Builder(context).getI64IntegerAttr(plan));
    return attributes.getDictionary(context);
}
bool attachStorageCoordinates(BoundedLifetimeDemandResult& demands, const StorageLaneAllocation& storage,
                              PreparedLogicalPlan& plan)
{
    // Stage every extra value before changing a selected endpoint. Destruction
    // drops all detached uses if any original cut cannot evaluate its lane.
    PreparedLogicalPlan staged(plan.planId);
    std::vector<std::pair<std::size_t, Value>> coordinates;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    for (auto [index, endpoint] : llvm::enumerate(plan.endpoints)) {
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        const bool singleRecord = endpoint.records.size() == 1;
        if (!singleRecord) { return false; }
        const auto record = endpoint.records.front();
        if (!std::binary_search(storage.records.begin(), storage.records.end(), record)) { continue; }
        const bool validRecord = record < demands.analysis.sourceDemands.size() && record < storage.lanes.size();
        if (!validRecord) { return false; }
        const auto& edge = demands.analysis.sourceDemands[record];
        auto lane = bindBoundedExpression(demands, storage.lanes[record], edge.target / demands.anchors.size(),
                                         endpoint.kind == LogicalCommandKind::Wait);
        auto& block = staged.addPreparation(endpoint.before);
        OpBuilder builder(demands.function.getContext());
        builder.setInsertionPointToEnd(&block);
        auto& context = contexts[endpoint.before];
        if (failed(demands.predicates->recover(lane, builder, endpoint.before, context))) { return false; }
        auto value = demands.expressions.emitContextual(lane, builder, endpoint.before, context);
        if (failed(value)) { return false; }
        coordinates.emplace_back(index, *value);
    }
    for (auto& stage : staged.preparation) { plan.preparation.push_back(std::move(stage)); }
    for (auto [index, value] : coordinates) { plan.endpoints[index].memberCoordinates.push_back(value); }
    return true;
}
void attachBoundedAllocation(BoundedLifetimeDemandResult& demands, PreparedLogicalPlan& plan)
{
    auto recipe = boundedAllocationRecipe(demands);
    DictionaryAttr certificate;
    const bool tryStorage = recipe->storageCertificate && recipe->storageCoordinatesAvailable.value_or(true);
    const bool storageReady = tryStorage && attachStorageCoordinates(demands, recipe->storage, plan);
    if (tryStorage) { recipe->storageCoordinatesAvailable = storageReady; }
    if (storageReady) {
        certificate = recipe->storageCertificate;
    } else {
        if (!recipe->ordinaryCertificate) {
            recipe->ordinaryCertificate = boundedLifetimeAllocationCertificate(demands.function,
                demands.expressions, demands.window, demands.unconditional, demands.analysis.sourceDemands, 0);
        }
        certificate = *recipe->ordinaryCertificate;
    }
    plan.allocationCertificate = bindAllocationPlan(certificate, plan.planId, demands.function.getContext());
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareEndpoints(
    const std::shared_ptr<BoundedLifetimeDemandResult>& demands, std::string& error, bool completeInvocation)
{
    auto& e = demands->expressions;
    const auto m = demands->anchors.size();
    const auto ordinal = demands->ordinal;
    CircuitEndpoints emitter(demands->function, e, demands->anchors);
    emitter.recover = [&](Id root, OpBuilder& builder, Operation* cut, RegionExpressions::CutEmission& context) {
        return demands->predicates->recover(root, builder, cut, context);
    };
    for (const auto& edge : demands->analysis.sourceDemands) {
        auto distance = edge.target / m;
        auto target = e.sub(ordinal, e.constant(distance));
        auto sourceGuard = bindBoundedExpression(*demands, edge.guard, distance, false);
        auto targetGuard = bindBoundedExpression(*demands, edge.guard, distance, true);
        targetGuard = e.land(e.le(e.constant(distance), ordinal), targetGuard);
        if (!emitter.add(edge.source, edge.target % m, sourceGuard, targetGuard, {ordinal}, {target})) {
            error = demands->predicates->error.empty() ? e.lastEmissionError() : demands->predicates->error;
            return failure();
        }
    }
    auto result = emitter.take();
    result->completeInvocation = completeInvocation;
    if (completeInvocation) {
        result->allocationPreparation = [demands](PreparedLogicalPlan& plan) {
            attachBoundedAllocation(*demands, plan);
        };
    }
    return result;
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeLogicalResult(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, std::string& error, bool completeInvocation)
{
    const bool valid = demands && demands->predicates && demands->analysis.error.empty() && !demands->anchors.empty();
    if (!valid) {
        error = "bounded endpoint preparation requires an owned successful analysis"; return failure();
    }
    if (completeInvocation && llvm::any_of(demands->input->instructions(), [&](const auto* phase) {
        return !demands->loop->isProperAncestor(phase->elementOp);
    })) {
        error = "bounded regional result does not cover the whole invocation"; return failure();
    }
    return prepareEndpoints(demands, error, completeInvocation);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeResult(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, std::string& error, bool completeInvocation)
{
    auto prepared = prepareBoundedLifetimeLogicalResult(std::move(demands), error, completeInvocation);
    if (succeeded(prepared)) { prepareAllocationSupport(**prepared); }
    return prepared;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error,
    std::shared_ptr<BoundedLifetimeDemandResult>* demands)
{
    if (demands) {
        demands->reset();
    }
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        if (node.kind != StructureKind::Loop || node.anchor->getParentOp() != function) {
            continue;
        }
        if (selected || !node.boundedLifetime ||
            node.boundedLifetime->skeleton.result.state != RecognitionState::Applicable) {
            return failure();
        }
        selected = &node;
    }
    if (!selected) {
        return failure();
    }
    auto loop = dyn_cast<scf::ForOp>(selected->anchor);
    if (llvm::any_of(input.instructions(), [&](const auto* p) { return !loop->isProperAncestor(p->elementOp); })) {
        return failure();
    }
    PhaseIndex index;
    if (failed(index.build(function, input))) {
        return failure();
    }
    auto analyzed = cachedBoundedLifetimeRegion(function, *selected, index, input, error);
    if (failed(analyzed)) { return failure(); }
    if (demands) { *demands = *analyzed; }
    return prepareBoundedLifetimeResult(*analyzed, error, true);
}
} // namespace mlir::pto::frontiersynch
