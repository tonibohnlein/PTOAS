// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/NumericTemplateRegional.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateAnalysis.h"
#include "SequenceAnalysisInternal.h"
#include "NumericTemplateInternal.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
namespace mlir::pto::frontiersynch {
FailureOr<RegionalAnalysis> numericBodyRegionalResult(func::FuncOp function,
    const SyncInput& input, const NumericTemplate& body, std::shared_ptr<RegionExpressions> arena,
    ArrayRef<scf::ForOp> enclosing, std::string& error)
{
    if (!arena || body.result.state != RecognitionState::Applicable || !body.specializedBody) {
        error = "finite body requires certified specialized effects"; return failure();
    }
    auto word = numericTemplateOccurrences(body, 1);
    if (failed(word)) { error = "finite body contains invalid physical atoms"; return failure(); }
    auto analysis = std::make_shared<ExplicitAnalysis>();
    analysis->occurrences = std::move(*word);
    std::vector<StorageGenerator> supplied, native;
    for (auto [a, b] : body.uniformConflicts) {
        if (a != b) { supplied.push_back({std::min(a, b), std::max(a, b)}); }
    }
    for (auto [a, b] : body.valueDemands) { supplied.push_back({a, b}); }
    for (auto [a, b] : body.nativePrerequisites) { native.push_back({a, b}); }
    analysis->scan = scanStorageLifetimes(analysis->occurrences, supplied);
    if (!analysis->scan.error.empty()) { error = analysis->scan.error; return failure(); }
    analysis->reduction = reduceExplicitDemands(analysis->occurrences, analysis->scan.generators, native);
    if (!analysis->reduction.error.empty()) { error = analysis->reduction.error; return failure(); }
    RegionalAnalysis out;
    out.expressions = arena; out.accessModel = &input.accesses(); out.gmAliasPolicy = input.memory().gmPolicy();
    out.capabilities = {true, true, true, true};
    out.cost.children = 1; out.cost.numericVisits = body.countedVisits; out.cost.physicalFragments = body.fragments;
    auto zero = arena->constant(0), yes = arena->boolean(true);
    auto selector = [&](uint32_t type) { return RegionalSelector{{type, zero, PeriodicEventKind::Start}, yes}; };
    for (uint32_t type = 0; type < body.payloads.size(); ++type) {
        const auto& payload = body.payloads[type];
        auto* op = payload.phase->elementOp;
        analysis->phases.push_back(payload.phase);
        out.anchors.push_back({payload.phase, payload.coordinates, {op->getBlock(), op},
                               {op->getBlock(), op->getNextNode()}});
        out.occurrenceLoops.push_back({});
        out.firstSitePayloads[type].push_back(selector(type));
        auto pipe = static_cast<uint32_t>(payload.phase->kPipeValue);
        if (out.firstPayloads[pipe].empty()) { out.firstPayloads[pipe].push_back(selector(type)); }
        out.lastPayloads[pipe] = {selector(type)};
        for (const auto& effect : payload.effects) {
            auto& destination = effect.discharge == TemplateDischarge::None ?
                out.accessBoundary : out.deferredAccessBoundary;
            destination.push_back({effect.sourceEffect, selector(type), selector(type),
                                    effect.discharge == TemplateDischarge::None});
        }
    }
    std::map<uint32_t, RegionalStorageBoundary> cells;
    for (const auto& occurrence : analysis->occurrences) {
        std::map<uint32_t, CellAccess> modes;
        for (const auto& access : occurrence.accesses) {
            auto& mode = modes[access.atom]; mode.read |= access.read; mode.write |= access.write;
        }
        for (const auto& [atom, mode] : modes) {
            auto& cell = cells[atom]; cell.cell = body.atoms[atom];
            if (mode.write) {
                if (cell.firstWriters.empty()) { cell.firstWriters.push_back(selector(occurrence.payload)); }
                cell.lastWriters = {selector(occurrence.payload)};
                cell.lastReaders.clear();
            } else if (mode.read) {
                if (cell.firstWriters.empty() && cell.firstReaders[occurrence.pipe].empty()) {
                    cell.firstReaders[occurrence.pipe].push_back(selector(occurrence.payload));
                }
                cell.lastReaders[occurrence.pipe] = {selector(occurrence.payload)};
            }
        }
    }
    for (auto& [atom, cell] : cells) { out.storageBoundary.push_back(std::move(cell)); }
    out.presence = [analysis, arena, zero](RegionalEvent event) -> std::optional<Expr> {
        if (event.type >= analysis->phases.size() || !event.visits.empty() ||
            event.ordinal >= arena->size() || arena->isBoolean(event.ordinal)) { return std::nullopt; }
        return arena->eq(event.ordinal, zero);
    };
    auto presence = out.presence;
    out.reachability = [analysis, arena, presence](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        auto pa = presence(a), pb = presence(b);
        if (!pa || !pb) { return std::nullopt; }
        auto reaches = explicitEventPrecedes(*analysis, {a.type, a.kind}, {b.type, b.kind});
        if (!reaches) { return std::nullopt; }
        return *reaches ? arena->land(*pa, *pb) : arena->boolean(false);
    };
    // Sequence preparation already knows tuple-qualified original cuts. Reuse
    // that emitter for the reduced internal edges, with no child commands and
    // no second reduction. The query export above retains the complete order.
    out.prepare = []() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        auto plan = std::make_unique<PreparedLogicalPlan>(0); plan->completeInvocation = false; return plan;
    };
    auto state = std::make_shared<SequenceAnalysisState>(function, arena);
    state->completeInvocation = false;
    state->requiredOuterLoops.assign(enclosing.begin(), enclosing.end());
    Child child; child.regional = out; state->children.push_back(std::move(child));
    if (!state->importSummaries(true)) { error = state->error; return failure(); }
    for (auto edge : analysis->reduction.retained) {
        auto source = state->port(0, selector(edge.source).event);
        auto target = state->port(0, selector(edge.target).event);
        state->crossings.push_back({source, target, yes});
    }
    out.prepare = {};
    out.prepareWithVisits = [state](ArrayRef<scf::ForOp> visits) { return state->prepare(visits); };
    return out;
}
FailureOr<RegionalAnalysis> specializedExplicitRunRegional(func::FuncOp function, scf::ForOp outer,
    ArrayRef<Operation*> operations, const PhaseIndex& index, const SyncInput& input,
    std::shared_ptr<RegionExpressions> arena, ArrayRef<scf::ForOp> enclosing,
    TemplateGeometryConstant geometry, std::string& error)
{
    if (!outer || !geometry || !arena) { error = "explicit phase context unavailable"; return failure(); }
    NumericTemplate body;
    body.outer = outer; body.specializedBody = true;
    auto lower = sequenceInteger(outer.getLowerBound()), step = sequenceInteger(outer.getStep());
    if (!lower || *lower < 0 || !step || *step <= 0) {
        error = "explicit phase requires a positive constant loop step"; return failure();
    }
    body.lower = *lower; body.step = *step;
    detail::TemplateBuilder builder{body, index, input, DenseMap<Value, int64_t>(), {}};
    builder.geometryConstant = std::move(geometry);
    SmallVector<const CompoundInstanceElement*> phases;
    Operation* previous = nullptr;
    for (auto* operation : operations) {
        if (!operation || !outer->isProperAncestor(operation) || operation->getNumRegions() ||
            (previous && previous->getNextNode() != operation)) {
            error = "explicit phase requires consecutive original leaf operations"; return failure();
        }
        previous = operation;
        detail::inspectLeaf(*operation, index, body.result);
        if (body.result.state != RecognitionState::Applicable) {
            error = "explicit phase contains an unsupported leaf"; return failure();
        }
        for (const auto* phase : index.phasesFor(operation)) {
            if (!builder.charge(1, builder.phases, body.limits.payloads, operation) || !builder.payload(phase)) {
                error = "explicit phase geometry is not concrete"; return failure();
            }
            phases.push_back(phase);
        }
    }
    if (!detail::prepareTemplateEffects(builder)) {
        error = "explicit phase lacks a finite shared storage partition"; return failure();
    }
    // A single run cannot discharge a reader merely because its writer is in
    // a sibling region. Only the shared whole-input no-writer proof is enough
    // here; all other GM geometry stays with the general regional provider.
    for (const auto& payload : body.payloads) {
        for (const auto& effect : payload.effects) {
            if (effect.discharge != TemplateDischarge::None &&
                !detail::dischargeGlobalReadOnlyEffect(effect.sourceEffect, input.accesses())) {
                error = "explicit phase discharge needs an external storage selector"; return failure();
            }
        }
    }
    const auto prerequisites = index.mapPrerequisites(phases);
    if (!prerequisites.error.empty()) { error = prerequisites.error; return failure(); }
    for (auto edge : prerequisites.native) { body.nativePrerequisites.emplace_back(edge.source, edge.target); }
    for (auto edge : prerequisites.demands) { body.valueDemands.emplace_back(edge.source, edge.target); }
    const auto protection = invocationProtectionGroups(input.accesses());
    for (auto& payload : body.payloads) { payload.invocationProtection = protection.lookup(payload.phase); }
    // Metadata/access normalization is charged by its finite input and fragment
    // counts. numericVisits stays zero: these are original payload occurrences.
    body.countedPayloads = phases.size();
    return numericBodyRegionalResult(function, input, body, std::move(arena), enclosing, error);
}
} // namespace mlir::pto::frontiersynch
