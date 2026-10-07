// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
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
    for (uint32_t a = 0; a < phases.size(); ++a) {
        if (!input.accesses().needsOverlapQueries(phases[a], unresolvedBases)) { continue; }
        for (uint32_t b = 0; b < phases.size(); ++b) {
            if (a == b || (b < a && input.accesses().needsOverlapQueries(phases[b], unresolvedBases))) { continue; }
            if (ptoStorageProtection().protectsScalar(static_cast<uint32_t>(phases[a]->kPipeValue),
                                                      static_cast<uint32_t>(phases[b]->kPipeValue))) { continue; }
            bool conflict = false;
            for (auto x : input.accesses().effectsFor(phases[a])) {
                for (auto y : input.accesses().effectsFor(phases[b])) {
                    if (input.accesses().residualConflict(x, y) &&
                        !hardwareProtectsConflict(static_cast<uint32_t>(phases[a]->kPipeValue), modeledProtection[x],
                            static_cast<uint32_t>(phases[b]->kPipeValue), modeledProtection[y])) {
                        conflict = true; break;
                    }
                }
                if (conflict) { break; }
            }
            if (conflict) { residual.push_back({std::min(a, b), std::max(a, b)}); }
        }
    }
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
    const auto& reduction = analysis.reduction;
    auto validKind = [](PeriodicEventKind kind) {
        return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
    };
    if (!analysis.error.empty() || !reduction.error.empty() || source.type >= reduction.payloads.size() ||
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
    func::FuncOp function, const ExplicitAnalysis& analysis)
{
    if (!analysis.error.empty() || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        return failure();
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !analysis.phases.empty();
    plan->allocationCertificate = explicitAllocationCertificate(analysis, plan->planId, function.getContext());
    if (analysis.reduction.retained.empty()) {
        return plan;
    }
    auto* first = &function.getBody().front().front();
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&plan->addPreparation(first));
    auto enabled = builder.create<arith::ConstantIntOp>(first->getLoc(), 1, 1);
    auto identity = builder.create<arith::ConstantIndexOp>(first->getLoc(), 0);
    int64_t record = 0;
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
            plan->endpoints.push_back({target, LogicalCommandKind::Barrier, sourcePipe, targetPipe,
                                       record, enabled, {}});
        } else {
            plan->endpoints.push_back({source->getNextNode(), LogicalCommandKind::Set, sourcePipe, targetPipe,
                                       record, enabled, identity});
            plan->endpoints.push_back({target, LogicalCommandKind::Wait, sourcePipe, targetPipe,
                                       record, enabled, identity});
        }
        ++record;
    }
    // Match the shared insertion rule: SETs, local barriers, then WAITs.
    llvm::DenseMap<Operation*, std::size_t> orders;
    for (auto kind : {LogicalCommandKind::Set, LogicalCommandKind::Barrier, LogicalCommandKind::Wait}) {
        for (const auto& endpoint : plan->endpoints) {
            if (endpoint.kind != kind) {
                continue;
            }
            auto& family = plan->families[endpoint.record];
            auto order = orders[endpoint.before]++;
            if (kind == LogicalCommandKind::Set) {
                family.sourceOrder = order;
            } else {
                family.targetOrder = order;
            }
        }
    }
    return plan;
}
} // namespace mlir::pto::frontiersynch
