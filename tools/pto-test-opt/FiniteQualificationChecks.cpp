// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent closure oracle and actual-IR insertion checks for finite overlays.
#include "FiniteRegionalChecks.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Finite/FiniteGuardedInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "llvm/Support/JSON.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
using Phase = const pto::CompoundInstanceElement*;
struct Demand { Phase source, target; Id active; };
struct View { fs::RegionalAnalysis region; std::vector<Demand> demands; };
View leaf(const std::shared_ptr<fs::FiniteAnalysis>& owner)
{
    View out{fs::finiteRegionalResult(owner), {}};
    const auto& state = *owner->guarded()->state;
    for (const auto& edge : state.retained) {
        out.demands.push_back({state.anchors[edge.source].phase, state.anchors[edge.target].phase, edge.guard});
    }
    return out;
}
View combine(func::FuncOp function, std::shared_ptr<fs::RegionExpressions> arena,
             const fs::PhaseIndex& index, View left, View right)
{
    std::vector<fs::RegionalAnalysis> children{left.region, right.region};
    auto composed = fs::composeRegionalSequence(function, arena, std::move(children), true, true, index.sourceFacts());
    if (!composed.error.empty() || !composed.state || composed.state->relationalResult) { return {}; }
    auto& state = *composed.state;
    View out{fs::sequenceRegionalResult(composed), std::move(left.demands)};
    llvm::append_range(out.demands, right.demands);
    bool valid = true;
    auto expand = [&](uint32_t port) {
        std::vector<std::pair<uint32_t, Id>> pending{{port, arena->boolean(true)}};
        std::vector<std::pair<Phase, Id>> result;
        while (!pending.empty()) {
            auto [id, active] = pending.back(); pending.pop_back();
            auto choice = state.portChoices.find(id);
            if (choice != state.portChoices.end()) {
                pending.push_back({choice->second.yes, arena->land(active, choice->second.choose)});
                pending.push_back({choice->second.no, arena->land(active, arena->lnot(choice->second.choose))});
            } else {
                const auto& point = state.ports.at(id);
                valid &= arena->constantValue(point.ordinal) == 0 && point.visits.empty();
                result.push_back({state.children.at(point.child).anchors.at(point.type).phase, active});
            }
        }
        return result;
    };
    for (const auto& edge : state.crossings) {
        for (auto a : expand(edge.source)) {
            for (auto b : expand(edge.target)) {
                out.demands.push_back({a.first, b.first, arena->land(edge.guard, arena->land(a.second, b.second))});
            }
        }
    }
    return valid ? out : View{};
}
using Boundary = std::tuple<unsigned, uint32_t, Phase>;
std::vector<std::pair<Boundary, Id>> boundary(
    const fs::RegionalAnalysis& region, const pto::SyncStorageCell& point, bool& valid)
{
    std::vector<std::pair<Boundary, Id>> out;
    auto add = [&](const auto& selected, unsigned kind, uint32_t pipe) {
        for (const auto& value : selected) {
            valid &= region.expressions->constantValue(value.event.ordinal) == 0 && value.event.visits.empty();
            out.push_back({{kind, pipe, region.anchors.at(value.event.type).phase}, value.present});
        }
    };
    for (const auto& cell : region.storageBoundary) {
        if (!pto::sameStorageDomain(point, cell.cell) || point.begin < cell.cell.begin ||
            point.begin >= cell.cell.end) {
            continue;
        }
        add(cell.firstWriters, 0, 0); add(cell.lastWriters, 1, 0);
        for (const auto& entry : cell.firstReaders) { add(entry.second, 2, entry.first); }
        for (const auto& entry : cell.lastReaders) { add(entry.second, 3, entry.first); }
    }
    return out;
}
bool equivalent(func::FuncOp function, const View& flat, const View& composed, uint64_t& comparisons)
{
    if (!flat.region.expressions || !composed.region.reachability ||
        flat.region.anchors.size() != composed.region.anchors.size()) { return false; }
    auto& arena = *flat.region.expressions;
    const auto zero = arena.constant(0);
    std::vector<uint32_t> mapped;
    for (const auto& anchor : flat.region.anchors) {
        auto it = llvm::find_if(composed.region.anchors,
            [&](const auto& other) { return other.phase == anchor.phase; });
        if (it == composed.region.anchors.end()) { return false; }
        mapped.push_back(it - composed.region.anchors.begin());
    }
    // Construct every cached query before rollback-scoped valuation substitution.
    std::vector<std::pair<Id, Id>> queries;
    for (uint32_t a = 0; a < mapped.size(); ++a) {
        auto pa = flat.region.presence({a, zero, fs::PeriodicEventKind::Start});
        auto pb = composed.region.presence({mapped[a], zero, fs::PeriodicEventKind::Start});
        if (!pa || !pb) { return false; }
        queries.push_back({*pa, *pb});
        for (uint32_t b = 0; b < mapped.size(); ++b) {
            for (auto ka : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                for (auto kb : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                    auto x = flat.region.reachability({a, zero, ka}, {b, zero, kb});
                    auto y = composed.region.reachability({mapped[a], zero, ka}, {mapped[b], zero, kb});
                    if (!x || !y) { return false; }
                    queries.push_back({*x, *y});
                }
            }
        }
    }
    using Entries = std::vector<std::pair<Boundary, Id>>;
    std::vector<std::pair<Entries, Entries>> boundaries;
    bool valid = true;
    for (const auto* region : {&flat.region, &composed.region}) {
        for (const auto& cell : region->storageBoundary) {
            boundaries.push_back({boundary(flat.region, cell.cell, valid),
                                  boundary(composed.region, cell.cell, valid)});
        }
    }
    if (!valid) { return false; }
    SmallVector<Id> inputs;
    for (auto argument : function.getArguments()) {
        if (argument.getType().isInteger(1)) { inputs.push_back(arena.input(argument)); }
    }
    if (inputs.size() > 3) { return false; }
    for (unsigned mask = 0; mask < (1U << inputs.size()); ++mask) {
        fs::RegionExpressions::Transaction transaction(arena);
        SmallVector<std::pair<Id, Id>> bindings;
        for (unsigned i = 0; i < inputs.size(); ++i) {
            bindings.push_back({inputs[i], arena.boolean(mask & (1U << i))});
        }
        fs::RegionExpressions::Substitution substitution(bindings);
        bool total = true;
        auto active = [&](Id id) {
            auto value = arena.constantValue(arena.substitute(id, substitution));
            total &= value.has_value();
            return value == 1;
        };
        for (auto [x, y] : queries) {
            ++comparisons;
            if (active(x) != active(y)) { return false; }
        }
        auto selected = [&](const Entries& entries) {
            std::set<Boundary> result;
            for (const auto& entry : entries) { if (active(entry.second)) { result.insert(entry.first); } }
            return result;
        };
        for (const auto& entry : boundaries) {
            ++comparisons;
            if (selected(entry.first) != selected(entry.second)) { return false; }
        }
        auto demands = [&](const View& view) {
            std::set<std::pair<Phase, Phase>> result;
            for (const auto& edge : view.demands) {
                if (active(edge.active)) { result.emplace(edge.source, edge.target); }
            }
            return result;
        };
        ++comparisons;
        if (demands(flat) != demands(composed) || !total) { return false; }
    }
    return true;
}
LogicalResult partitions(std::shared_ptr<const fs::FiniteOccurrenceInput> input)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto owner = fs::analyzeFinite(input, fs::FiniteStrategy::Guarded, arena);
    if (!owner || !owner->error().empty()) { return failure(); }
    const auto flat = leaf(owner);
    uint64_t comparisons = 0, partitions = 0;
    const auto roots = input->roots();
    auto slice = [&](std::size_t begin, std::size_t size) {
        auto child = fs::analyzeFinite(fs::finiteRegionInput(input, roots.slice(begin, size)),
                                       fs::FiniteStrategy::Guarded, arena);
        return child && child->error().empty() ? leaf(child) : View{};
    };
    for (std::size_t cut = 0; cut <= roots.size(); ++cut) {
        auto left = slice(0, cut), right = slice(cut, roots.size() - cut);
        if (!left.region.presence || !right.region.presence ||
            !equivalent(input->function(), flat,
                combine(input->function(), arena, input->index(), left, right), comparisons)) { return failure(); }
        ++partitions;
    }
    // Both binary parenthesizations, retaining empty and whole structured children.
    const auto a = roots.size() / 3, b = 2 * roots.size() / 3;
    auto x = slice(0, a), y = slice(a, b - a), z = slice(b, roots.size() - b);
    for (bool leftFirst : {false, true}) {
        auto combined = leftFirst ? combine(input->function(), arena, input->index(),
            combine(input->function(), arena, input->index(), x, y), z) :
            combine(input->function(), arena, input->index(), x,
                combine(input->function(), arena, input->index(), y, z));
        if (!equivalent(input->function(), flat, combined, comparisons)) { return failure(); }
        ++partitions;
    }
    llvm::outs() << llvm::json::Value(llvm::json::Object{{"partitions", partitions}, {"comparisons", comparisons}})
                 << "\n";
    return success();
}
LogicalResult concreteWork(const std::shared_ptr<fs::FiniteAnalysis>& result,
                           const fs::RegionalAnalysis& region, std::size_t beforeQueries, llvm::json::Object& report)
{
    const auto& value = *result->concrete();
    auto first = result->prepare(), second = result->prepare();
    if (failed(first) || failed(second) || (*first)->familyRecipe != (*second)->familyRecipe ||
        !(*first)->preparation.empty() || !(*second)->preparation.empty() || result->guarded() ||
        region.expressions->size() != beforeQueries || result->work.recipeBuilds != 1 ||
        result->work.recipeHits != 1 || value.scanWork.counterOverflow) { return failure(); }
    report["incidences"] = value.scanWork.incidenceVisits;
    report["hazard_visits"] = value.scanWork.hazardWitnesses;
    report["retained_witnesses"] = value.scanWork.retainedWitnesses;
    report["residual_phases"] = value.residualPhaseVisits;
    report["residual_effects"] = value.residualEffectVisits;
    report["numerical_queries"] = result->work.numericalQueries;
    report["boundary_entries"] = result->work.boundaryEntries;
    report["recipe_hits"] = result->work.recipeHits;
    return success();
}
LogicalResult guardedWork(const std::shared_ptr<fs::FiniteAnalysis>& result, llvm::json::Object& report)
{
    const auto& state = *result->guarded()->state;
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> cells;
    for (const auto& occurrence : state.effects) {
        for (const auto& access : occurrence.accesses) {
            ++cells[access.atom].first;
            cells[access.atom].second += access.write;
        }
    }
    uint64_t visits = 0, candidates = 0, incidences = 0, w = 0;
    for (auto [id, counts] : cells) {
        const auto [m, writers] = counts;
        incidences += m; w += m * writers;
        visits += writers * (m - 1);
        candidates += writers * (m - writers) + (writers ? writers * (writers - 1) / 2 : 0);
    }
    for (const auto* work : {&state.qualificationWork, &state.constructionWork}) {
        if (work->sourceVisits != incidences || work->pairVisits != visits ||
            work->candidateVisits != candidates) { return failure(); }
    }
    report["incidences"] = incidences; report["W"] = w;
    report["pair_visits"] = visits * 2; report["candidates"] = candidates;
    report["residual_phases"] = state.residualPhaseVisits;
    report["residual_effects"] = state.residualEffectVisits;
    report["expressions"] = result->guarded()->cost.expressionNodes;
    report["selectors"] = state.cost.selectorComparisons;
    return success();
}
} // namespace
LogicalResult runFiniteQualificationChecks(func::FuncOp function, pto::GMAliasPolicy policy)
{
    auto storage = std::make_shared<pto::SyncInput>(policy);
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(storage->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index->build(function, *storage))) { return failure(); }
    bool envelopes = true;
    function.walk([&](Operation* operation) {
        if (operation->getName().getStringRef() == "pto.tscatter") {
            std::set<uint32_t> pipes;
            for (auto* phase : index->phasesFor(operation)) { pipes.insert(static_cast<uint32_t>(phase->kPipeValue)); }
            envelopes &= pipes == std::set<uint32_t>{static_cast<uint32_t>(pto::PIPE::PIPE_V),
                                                       static_cast<uint32_t>(pto::PIPE::PIPE_S)};
        }
    });
    if (!envelopes) { return function.emitError("test requires both simultaneous scatter pipe envelopes"); }
    auto input = fs::finiteFunctionInput(function, storage, index);
    if (!input) { return failure(); }
    const auto mode = function->getAttrOfType<StringAttr>("test.finite_qualification");
    if (!mode) { return failure(); }
    if (mode.getValue() == "partitions") {
        if (failed(partitions(input))) { return function.emitError("guarded finite partition oracle disagrees"); }
        return success();
    }
    const auto beforeSlices = input->rootCopies();
    for (auto* root : input->roots()) {
        auto slice = fs::finiteRegionInput(input, ArrayRef<Operation*>{root});
        if (!slice || slice->roots().size() != 1) { return failure(); }
    }
    const auto sliceCopies = input->rootCopies() - beforeSlices;
    if (sliceCopies != input->roots().size()) { return failure(); }
    const bool concrete = mode.getValue() == "concrete";
    auto result = fs::analyzeFinite(input, concrete ? fs::FiniteStrategy::Concrete : fs::FiniteStrategy::Guarded);
    if (!result || !result->error().empty()) { return function.emitError("finite scaling analysis failed"); }
    auto region = fs::finiteRegionalResult(result);
    const auto beforeQueries = region.expressions->size();
    const auto zero = region.expressions->constant(0);
    for (uint32_t i = 0; i < region.anchors.size(); ++i) {
        auto answer = region.reachability({i, zero, fs::PeriodicEventKind::Start},
                                         {i, zero, fs::PeriodicEventKind::Completion});
        if (!answer) { return failure(); }
    }
    llvm::json::Object report{{"sites", region.anchors.size()}, {"retained_roots", input->roots().size()},
        {"slice_root_copies", sliceCopies}, {"root_copy_bytes", input->rootCopies() * sizeof(Operation*)},
        {"query_nodes", region.expressions->size() - beforeQueries}};
    if (failed(concrete ? concreteWork(result, region, beforeQueries, report) : guardedWork(result, report))) {
        return function.emitError("finite inspected-work or recipe reuse check failed");
    }
    llvm::outs() << llvm::json::Value(std::move(report)) << "\n";
    return success();
}
