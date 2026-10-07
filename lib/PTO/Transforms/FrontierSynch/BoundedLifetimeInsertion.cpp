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
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "CountedLoop.h"
#include "IterationPredicates.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "CircuitEndpoints.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareEndpoints(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error, bool completeInvocation)
{
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
    RegionExpressions e;
    Block symbols;
    auto base = e.input(symbols.addArgument(IndexType::get(function.getContext()), function.getLoc()));
    auto trips = domain->trips(e);
    auto ordinal =
        e.div(e.sub(e.input(loop.getInductionVar()), e.input(loop.getLowerBound())), e.constant(domain->step));
    SmallVector<const CompoundInstanceElement*> phases;
    std::vector<TemplateEndpointAnchor> anchors;
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    DenseMap<Operation*, uint32_t> operationIds;
    for (const auto& site : skeleton.phases) {
        positions[site.phase] = phases.size();
        phases.push_back(site.phase);
        auto* op = site.phase->elementOp;
        operationIds.try_emplace(op, operationIds.size() + 1);
        anchors.push_back({site.phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
    }
    IterationPredicates predicates(loop, e, phases);
    LifetimeWindowInput window;
    window.sites = m;
    window.span = span;
    window.storageProtection = ptoStorageProtection();
    const auto protection = structuredProtection(input.accesses());
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> groups;
    std::vector<std::pair<Id, std::pair<Value, uint64_t>>> predicateBindings;
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
            auto cell = cells
                            .emplace(
                                std::make_tuple(
                                    families.try_emplace(access.family, families.size()).first->second,
                                    access.atom->first, access.atom->second, slot),
                                cells.size())
                            .first->second;
            uint64_t group = effect.memory && effect.memory->scope == AddressSpace::ACC && access.stride == 0 ?
                                 protection.inLoop(effect.phase, loop) :
                                 0;
            if (group) {
                auto scope = group & invocationProtectionBit ? 0 : d;
                auto reset = group & protectionResetBit;
                group = groups.emplace(std::make_pair(group & ~protectionResetBit, scope), groups.size() + 1)
                            .first->second |
                        reset;
            }
            window.accesses.push_back(
                {static_cast<uint32_t>(d * m + found->second), cell, e.boolean(access.reads), e.boolean(access.writes),
                 group});
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
    auto analysis = analyzeLifetimeWindow(e, window);
    if (!analysis.error.empty()) {
        error = analysis.error;
        return failure();
    }
    CircuitEndpoints emitter(function, e, anchors);
    emitter.recover = [&](Id root, OpBuilder& builder, Operation* cut, RegionExpressions::CutEmission& context) {
        return predicates.recover(root, builder, cut, context);
    };
    for (auto edge : analysis.sourceDemands) {
        auto distance = edge.target / m;
        auto source = ordinal, target = e.sub(ordinal, e.constant(distance));
        auto bind = [&](bool consumer) {
            std::vector<std::pair<Id, Id>> bindings{{base, consumer ? target : source}};
            for (auto [id, recipe] : predicateBindings) {
                auto shift = static_cast<int64_t>(recipe.second) - (consumer ? int64_t(distance) : 0);
                auto evaluation = shift < 0 ? e.sub(ordinal, e.constant(-shift)) : e.add(ordinal, e.constant(shift));
                bindings.push_back({id, predicates.at(recipe.first, evaluation)});
            }
            RegionExpressions::Substitution substitution(bindings);
            return e.substitute(edge.guard, substitution);
        };
        auto sourceGuard = bind(false), targetGuard = bind(true);
        targetGuard = e.land(e.le(e.constant(distance), ordinal), targetGuard);
        if (!emitter.add(edge.source, edge.target % m, sourceGuard, targetGuard, {source}, {target})) {
            error = predicates.error.empty() ? e.lastEmissionError() : predicates.error;
            return failure();
        }
    }
    auto result = emitter.take();
    result->completeInvocation = completeInvocation;
    if (completeInvocation) {
        SmallVector<uint8_t> unconditional;
        for (const auto& site : skeleton.phases) { unconditional.push_back(!site.guard); }
        result->allocationCertificate = boundedLifetimeAllocationCertificate(
            function, e, window, unconditional, analysis.sourceDemands, result->planId);
    }
    return result;
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeEndpoints(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error)
{
    return prepareEndpoints(function, loop, index, input, recognized, error, false);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error)
{
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
    return prepareEndpoints(function, loop, index, input, *selected->boundedLifetime, error, true);
}
} // namespace mlir::pto::frontiersynch
