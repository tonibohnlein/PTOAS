// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "RecognitionInternal.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
void protection(RotatingAnalysis& result, const SyncInput& input, const RecognitionResult& recognized)
{
    // A group is confined to this body visit. Only stationary accumulator
    // fragments can share a physical identity across sites here; rotating
    // accumulators retain ordinary software demands.
    HardwareProtectionBuilder builder;
    std::vector<ExplicitEffects> word(result.phases.size());
    for (uint32_t i = 0; i < word.size(); ++i) {
        word[i].payload = i;
        word[i].pipe = static_cast<uint32_t>(result.phases[i]->kPipeValue);
    }
    std::map<std::tuple<uint32_t, uint32_t, uint64_t>, uint32_t> identities;
    std::vector<uint32_t> atoms;
    for (const auto& fragment : result.fragments) {
        auto key = std::make_tuple(fragment.family, fragment.atom, fragment.offset);
        auto atom = identities.emplace(key, identities.size()).first->second;
        atoms.push_back(atom);
        word[fragment.payload].accesses.push_back({atom, fragment.read, fragment.write});
    }
    std::vector<SmallVector<uint32_t>> accumulatorAtoms(word.size());
    std::vector<bool> stationary(word.size(), true);
    for (std::size_t i = 0; i < result.fragments.size(); ++i) {
        const auto& fragment = result.fragments[i];
        const auto& effect = input.accesses().effects()[recognized.accesses[i].effect];
        if (effect.memory->scope == AddressSpace::ACC) {
            accumulatorAtoms[fragment.payload].push_back(atoms[i]);
            stationary[fragment.payload] = stationary[fragment.payload] && fragment.stride == 0;
        }
    }
    for (uint32_t i = 0; i < word.size(); ++i) {
        if (!stationary[i]) {
            builder.endScope();
        } else {
            builder.observe(result.phases[i]->elementOp, word[i], accumulatorAtoms[i]);
        }
    }
    std::vector<std::map<uint32_t, uint64_t>> groups(word.size());
    for (const auto& occurrence : word) {
        for (const auto& access : occurrence.accesses) {
            groups[occurrence.payload][access.atom] = access.protectionGroup;
        }
    }
    for (std::size_t i = 0; i < result.fragments.size(); ++i) {
        auto& fragment = result.fragments[i];
        fragment.protectionGroup = groups[fragment.payload][atoms[i]];
    }
}
} // namespace
RotatingAnalysis analyzeRotating(scf::ForOp loop, const PhaseIndex& index,
    const SyncInput& input, const RecognitionResult& recognized)
{
    RotatingAnalysis result;
    result.loop = loop;
    if (!loop || recognized.state != RecognitionState::Applicable) {
        result.error = "rotating route requires a certified fixed-body footprint contract";
        return result;
    }
    auto sequence = index.explicitSequence(*loop.getBody());
    if (failed(sequence) || sequence->size() > UINT32_MAX || recognized.accesses.size() > UINT32_MAX) {
        result.error = "rotating payload sequence unavailable or too large";
        return result;
    }
    result.phases = *sequence;
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    DenseMap<Value, uint32_t> families;
    std::map<std::tuple<uint32_t, uint64_t, uint64_t>, uint32_t> atoms;
    std::vector<PeriodicPayload> payloads;
    for (auto* phase : result.phases) {
        positions[phase] = payloads.size();
        payloads.push_back({static_cast<uint32_t>(phase->kPipeValue)});
    }
    for (const auto& access : recognized.accesses) {
        if (!access.atom || access.guard || access.parameterOffset ||
            access.effect >= input.accesses().effects().size()) {
            result.error = "rotating fragments require fixed unguarded within-slot atoms";
            return result;
        }
        auto found = positions.find(input.accesses().effects()[access.effect].phase);
        if (found == positions.end()) {
            result.error = "rotating fragment has no body occurrence";
            return result;
        }
        const auto family = families.try_emplace(access.family, families.size()).first->second;
        const auto atom = atoms.emplace(std::make_tuple(family, access.atom->first, access.atom->second),
                                       atoms.size()).first->second;
        result.fragments.push_back({found->second, family, atom, access.slots, access.stride, access.offset,
                                     access.reads, access.writes, 0});
    }
    protection(result, input, recognized);
    result.extraction = extractRotatingGenerators(payloads, result.fragments);
    if (!result.extraction.error.empty()) {
        result.error = result.extraction.error;
        return result;
    }
    if (input.accesses().hasUniformRelationships(result.phases)) {
        for (uint32_t a = 0; a < result.phases.size(); ++a) {
            for (uint32_t b = 0; b < result.phases.size(); ++b) {
                bool conflict = false;
                for (auto x : input.accesses().effectsFor(result.phases[a])) {
                    for (auto y : input.accesses().effectsFor(result.phases[b])) {
                        if (!llvm::is_contained(recognized.dischargedEffects, x) &&
                            !llvm::is_contained(recognized.dischargedEffects, y)) {
                            conflict |= input.accesses().uniformConflict(x, y);
                        }
                    }
                }
                if (conflict) {
                    result.extraction.generators.push_back({a, b, a < b ? 0U : 1U});
                    result.extraction.refreshBound = std::max<uint64_t>(result.extraction.refreshBound, 1);
                }
            }
        }
    }
    const auto prerequisites = index.mapPrerequisites(result.phases);
    if (!prerequisites.error.empty()) { result.error = prerequisites.error; return result; }
    std::vector<PeriodicRecord> native;
    for (const auto& edge : prerequisites.native) { native.push_back({edge.source, edge.target, 0}); }
    for (const auto& edge : prerequisites.demands) {
        result.extraction.generators.push_back({edge.source, edge.target, 0});
    }
    result.periodic = analyzePeriodicDemands(payloads, result.extraction.generators, native);
    result.error = result.periodic.error;
    if (!result.error.empty()) {
        return result;
    }
    SmallVector<TemplateEndpointAnchor> anchors;
    for (auto* phase : result.phases) {
        auto* op = phase->elementOp;
        anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
    }
    result.endpoints = bindPeriodicEndpoints(loop, anchors, result.periodic);
    result.error = result.endpoints.logical.error;
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareRotatingInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program)
{
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        return failure();
    }
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        if (node.kind == StructureKind::Loop && node.anchor->getParentOp() == function) {
            if (selected || !node.rotatingResult || node.rotatingResult->state != RecognitionState::Applicable) {
                return failure();
            }
            selected = &node;
        }
    }
    if (!selected) {
        return failure();
    }
    auto loop = dyn_cast<scf::ForOp>(selected->anchor);
    if (!loop || llvm::any_of(input.instructions(), [&](const auto* phase) {
        return !loop->isProperAncestor(phase->elementOp);
    })) {
        return failure();
    }
    PhaseIndex index;
    if (failed(index.build(function, input))) {
        return failure();
    }
    RecognitionResult outside;
    for (auto& op : function.front()) {
        if (&op != loop.getOperation()) {
            detail::inspectLeaf(op, index, outside);
            if (op.getNumRegions()) {
                return failure();
            }
        }
    }
    if (outside.state != RecognitionState::Applicable) {
        return failure();
    }
    auto analysis = analyzeRotating(loop, index, input, *selected->rotatingResult);
    if (!analysis.error.empty()) {
        return failure();
    }
    auto prepared = std::make_unique<PreparedLogicalPlan>(0);
    prepared->completeInvocation = !analysis.phases.empty();
    if (failed(prepareCountedEndpointCode(function, analysis.endpoints, *prepared))) {
        return failure();
    }
    auto allocation = buildPeriodicAllocation(analysis.periodic);
    if (allocation.error.empty()) {
        prepared->allocationCertificate = encodeCyclicAllocation(allocation, prepared->planId, function.getContext());
    }
    return prepared;
}
} // namespace mlir::pto::frontiersynch
