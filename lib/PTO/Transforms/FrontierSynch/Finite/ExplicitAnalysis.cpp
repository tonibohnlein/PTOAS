// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/Recognition.h"
#include "../Shared/WriterPairIndex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
void boundary(ExplicitAnalysis& result)
{
    std::unordered_map<uint32_t, std::size_t> cells;
    for (const auto& occurrence : result.occurrences) {
        std::unordered_map<uint32_t, CellAccess> modes;
        for (const auto& access : occurrence.accesses) {
            auto& mode = modes[access.atom];
            mode.read |= access.read;
            mode.write |= access.write;
        }
        for (const auto& [atom, mode] : modes) {
            auto [position, inserted] = cells.emplace(atom, result.storageBoundary.size());
            if (inserted) {
                ExplicitCellBoundary cell;
                cell.atom = atom;
                result.storageBoundary.push_back(std::move(cell));
            }
            auto& cell = result.storageBoundary[position->second];
            if (mode.write) {
                if (!cell.firstWriter) {
                    cell.firstWriter = occurrence.payload;
                }
                cell.lastWriter = occurrence.payload;
                std::unordered_map<uint32_t, uint32_t>().swap(cell.lastReaders);
            } else if (mode.read) {
                if (!cell.firstWriter) {
                    cell.firstReaders.emplace(occurrence.pipe, occurrence.payload);
                }
                cell.lastReaders[occurrence.pipe] = occurrence.payload;
            }
        }
    }
}
ExplicitAnalysis analyzeSpan(ArrayRef<const CompoundInstanceElement*> phases, const PhaseIndex& index,
                             const SyncInput& input,
                             ArrayRef<std::size_t> discharged = {})
{
    ExplicitAnalysis result;
    if (!input.hasFineStorage()) {
        result.error = "explicit analysis requires fine storage facts";
        return result;
    }
    if (phases.size() > std::numeric_limits<uint32_t>::max() ||
        input.accesses().cells().size() > std::numeric_limits<uint32_t>::max()) {
        result.error = "explicit occurrence or cell identity overflow";
        return result;
    }
    result.phases.assign(phases.begin(), phases.end());
    result.dischargedEffects.assign(discharged.begin(), discharged.end());
    llvm::DenseSet<std::size_t> omitted(discharged.begin(), discharged.end());
    const auto invocation = structuredProtection(input.accesses());
    for (const auto* phase : result.phases) {
        ExplicitEffects occurrence;
        occurrence.payload = static_cast<uint32_t>(result.occurrences.size());
        occurrence.pipe = static_cast<uint32_t>(phase->kPipeValue);
        for (auto id : input.accesses().effectsFor(phase)) {
            if (omitted.contains(id)) {
                continue;
            }
            const auto& effect = input.accesses().effects()[id];
            for (auto cell : effect.cells) {
                const auto atom = static_cast<uint32_t>(cell);
                occurrence.accesses.push_back({atom, effect.mode == SyncAccessMode::Read,
                                               effect.mode == SyncAccessMode::Write});
            }
        }
        if (auto group = invocation.at(phase)) {
            for (auto& access : occurrence.accesses) {
                if (input.accesses().cells()[access.atom].space == AddressSpace::ACC) {
                    access.protectionGroup = group;
                }
            }
        }
        result.occurrences.push_back(std::move(occurrence));
    }
    const auto modeledProtection = modeledProtectionGroups(input, phases, invocation);
    const bool unresolvedBases = input.accesses().hasUniformRelationships(phases);
    std::vector<StorageGenerator> residual;
    std::vector<ArrayRef<std::size_t>> effects;
    std::vector<WriterPairIndex> roles;
    std::vector<bool> needsOverlap;
    for (auto* phase : phases) {
        effects.push_back(input.accesses().effectsFor(phase));
        needsOverlap.push_back(input.accesses().needsOverlapQueries(phase, unresolvedBases));
        const auto ids = effects.back();
        roles.emplace_back(
            ids.size(), [&](std::size_t i) { return input.accesses().effects()[ids[i]].mode == SyncAccessMode::Read; });
    }
    WriterPairIndex candidates(
        phases.size(), [&](std::size_t i) { return !roles[i].hasWriters(); },
        [&](std::size_t i) { return needsOverlap[i]; });
    candidates.pairs([&](std::size_t a, std::size_t b) {
        const auto p = static_cast<uint32_t>(phases[a]->kPipeValue);
        const auto q = static_cast<uint32_t>(phases[b]->kPipeValue);
        if (ptoStorageProtection().protectsScalar(p, q)) {
            return true;
        }
        bool conflict = false;
        roles[a].cross(roles[b], [&](std::size_t i, std::size_t j) {
            const auto x = effects[a][i], y = effects[b][j];
            if (!hardwareProtectsConflict(p, modeledProtection[x], q, modeledProtection[y]) &&
                input.accesses().residualConflict(x, y)) {
                conflict = true;
                return false;
            }
            return true;
        });
        if (conflict) {
            residual.push_back({static_cast<uint32_t>(a), static_cast<uint32_t>(b)});
        }
        return true;
    });
    const auto prerequisites = index.mapPrerequisites(phases);
    if (!prerequisites.error.empty()) {
        result.error = prerequisites.error;
        return result;
    }
    llvm::append_range(residual, prerequisites.demands);
    result.scan = scanStorageLifetimes(result.occurrences, residual, ptoStorageProtection());
    if (!result.scan.error.empty()) {
        result.error = result.scan.error;
        return result;
    }
    result.reduction = reduceExplicitDemands(result.occurrences, result.scan.generators, prerequisites.native);
    result.error = result.reduction.error;
    if (result.error.empty()) {
        boundary(result);
    }
    return result;
}
bool independentGM(std::size_t id, const SyncInput& input)
{
    const auto effects = input.accesses().effects();
    const auto& effect = effects[id];
    auto* anchor = effect.phase->elementOp;
    auto function = anchor->getParentOfType<func::FuncOp>();
    if (!effect.memory || effect.memory->scope != AddressSpace::GM || !function ||
        function.isDeclaration() || anchor->getBlock() != &function.front()) {
        return false;
    }
    return input.accesses().independentOfOtherPhases(id);
}
bool spanEffects(ArrayRef<const CompoundInstanceElement*> phases, const SyncInput& input,
                 bool dischargeIndependentGM, SmallVectorImpl<std::size_t>& discharged)
{
    for (const auto* phase : phases) {
        for (auto id : input.accesses().effectsFor(phase)) {
            if (dischargeIndependentGM && !input.accesses().effects()[id].rangesMaterialized &&
                independentGM(id, input)) { discharged.push_back(id); }
        }
    }
    return true;
}
bool validSpan(ArrayRef<const CompoundInstanceElement*> phases, const PhaseIndex& index,
               const SyncInput& input, bool dischargeIndependentGM,
               SmallVectorImpl<std::size_t>& discharged)
{
    if (phases.empty()) {
        return true;
    }
    if (!phases.front() || !phases.back() || !phases.front()->elementOp || !phases.back()->elementOp) {
        return false;
    }
    SmallVector<Operation*> operations;
    std::size_t expected = 0;
    for (auto* op = phases.front()->elementOp; op; op = op->getNextNode()) {
        operations.push_back(op);
        for (const auto* phase : index.phasesFor(op)) {
            if (expected == phases.size() || phases[expected] != phase) {
                return false;
            }
            ++expected;
        }
        if (op == phases.back()->elementOp) {
            if (expected != phases.size()) {
                return false;
            }
            const auto recognized = recognizeExplicitRun(operations, index, input.accesses());
            if (recognized.state != RecognitionState::Applicable) { return false; }
            return spanEffects(phases, input, dischargeIndependentGM, discharged);
        }
    }
    return false;
}
} // namespace
ExplicitAnalysis analyzeExplicit(Block& block, const PhaseIndex& index, const SyncInput& input)
{
    if (!input.hasFineStorage()) {
        ExplicitAnalysis result;
        result.error = "explicit analysis requires fine storage facts";
        return result;
    }
    const auto recognized = recognizeExplicit(block, index, input.accesses());
    const auto sequence = index.explicitSequence(block);
    if (failed(sequence) || recognized.state != RecognitionState::Applicable) {
        ExplicitAnalysis result;
        result.error = "explicit route requires resolved control and shared modeled effects";
        return result;
    }
    // Function roots and regional spans use the same effect contract. A GM
    // access proved independent of every other phase needs no storage edge.
    return analyzeExplicit(*sequence, index, input, true);
}
ExplicitAnalysis analyzeExplicit(ArrayRef<const CompoundInstanceElement*> phases,
                                 const PhaseIndex& index, const SyncInput& input, bool dischargeIndependentGM)
{
    if (!input.hasFineStorage()) {
        ExplicitAnalysis result;
        result.error = "explicit analysis requires fine storage facts";
        return result;
    }
    SmallVector<std::size_t> discharged;
    if (!validSpan(phases, index, input, dischargeIndependentGM, discharged)) {
        ExplicitAnalysis result;
        result.error = "explicit span requires consecutive payloads and no additional prerequisites";
        return result;
    }
    return analyzeSpan(phases, index, input, discharged);
}
std::optional<bool> explicitEventPrecedes(const ExplicitAnalysis& analysis,
                                        PeriodicEvent source, PeriodicEvent target)
{
    if (!analysis.error.empty()) { return std::nullopt; }
    return explicitEventPrecedes(analysis.reduction, source, target);
}
std::optional<bool> explicitEventPrecedes(const ExplicitReduction& reduction,
                                        PeriodicEvent source, PeriodicEvent target)
{
    auto validKind = [](PeriodicEventKind kind) {
        return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
    };
    if (!reduction.error.empty() || source.type >= reduction.payloads.size() ||
        target.type >= reduction.payloads.size() || !validKind(source.kind) || !validKind(target.kind)) {
        return std::nullopt;
    }
    const auto sourceColumn = reduction.pipeColumns[source.type];
    if (source.kind == PeriodicEventKind::Start && target.kind == PeriodicEventKind::Start &&
        sourceColumn == reduction.pipeColumns[target.type] && source.type <= target.type) {
        return true;
    }
    const auto& ranks = target.kind == PeriodicEventKind::Completion ?
        reduction.completionRanks[target.type] : reduction.startRanks[target.type];
    return reduction.localRanks[source.type] <= ranks[sourceColumn];
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareExplicitInsertion(
    func::FuncOp function, const ExplicitAnalysis& analysis, bool includeAllocation)
{
    if (!analysis.error.empty() || function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) ||
        analysis.reduction.retained.size() > UINT32_MAX) {
        return failure();
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !analysis.phases.empty();
    if (includeAllocation) {
        plan->allocationCertificate = explicitAllocationCertificate(analysis, plan->planId, function.getContext());
    }
    if (analysis.reduction.retained.empty()) {
        return plan;
    }
    auto* first = &function.getBody().front().front();
    auto arena = std::make_shared<RegionExpressions>();
    const auto enabled = arena->boolean(true), identity = arena->constant(0);
    std::vector<ContextualEndpointFamily> endpoints;
    auto append = [&](Operation* cut, EndpointKind kind, uint32_t source, uint32_t target, uint32_t record) {
        ContextualEndpointFamily endpoint;
        endpoint.piece.cut = {cut->getBlock(), cut};
        endpoint.piece.kind = kind;
        endpoint.piece.namespaceId = record;
        endpoint.piece.sourcePipe = source; endpoint.piece.targetPipe = target;
        endpoint.piece.members.push_back({record, {}});
        endpoint.activity = enabled; endpoint.ordinal = identity;
        endpoints.push_back(std::move(endpoint));
    };
    uint32_t record = 0;
    for (const auto& edge : analysis.reduction.retained) {
        const auto sourcePipe = analysis.occurrences[edge.source].pipe;
        const auto targetPipe = analysis.occurrences[edge.target].pipe;
        auto* source = analysis.phases[edge.source]->elementOp;
        auto* target = analysis.phases[edge.target]->elementOp;
        if (source->getParentOfType<func::FuncOp>() != function ||
            target->getBlock() != source->getBlock() || !source->getNextNode()) {
            return failure();
        }
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = sourcePipe;
        family.targetPipe = targetPipe;
        family.local = sourcePipe == targetPipe;
        family.sourceCut = {source->getBlock(), source->getNextNode()};
        family.targetCut = {target->getBlock(), target};
        family.members.push_back({static_cast<uint32_t>(record), edge.source, edge.target, {}, {}});
        plan->families.push_back(std::move(family));
        if (sourcePipe == targetPipe) {
            append(target, EndpointKind::Barrier, sourcePipe, targetPipe, record);
        } else {
            append(source->getNextNode(), EndpointKind::Set, sourcePipe, targetPipe, record);
            append(target, EndpointKind::Wait, sourcePipe, targetPipe, record);
        }
        ++record;
    }
    // Match the shared insertion rule: SETs, local barriers, then WAITs.
    llvm::DenseMap<Operation*, std::size_t> orders;
    for (auto kind : {EndpointKind::Set, EndpointKind::Barrier, EndpointKind::Wait}) {
        for (auto& endpoint : endpoints) {
            if (endpoint.piece.kind != kind) { continue; }
            auto& family = plan->families[endpoint.piece.members.front().record];
            const auto order = orders[endpoint.piece.cut.before]++;
            endpoint.piece.order = order;
            if (kind == EndpointKind::Set) { family.sourceOrder = order; }
            else { family.targetOrder = order; }
        }
    }
    // Preserve the existing first-cut constant sharing, but retain owned roots
    // until allocation selects which matching inputs need scalar construction.
    auto recipe = buildContextualFamilyEndpointRecipe(arena, endpoints, {{enabled, first}, {identity, first}});
    if (failed(recipe)) { return failure(); }
    plan->familyRecipe = std::move(*recipe);
    plan->groupedFamilies = plan->independentPieces = true;
    return plan;
}
} // namespace mlir::pto::frontiersynch
