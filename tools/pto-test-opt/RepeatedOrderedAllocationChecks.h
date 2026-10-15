// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exercise the production ordered-span allocator and its exported lane bounds.
#ifndef PTO_REPEATED_ORDERED_ALLOCATION_CHECKS_H
#define PTO_REPEATED_ORDERED_ALLOCATION_CHECKS_H
#include "llvm/ADT/BitVector.h"
namespace {
bool checkOrderedLaneAllocation(mlir::func::FuncOp function, mlir::Operation* anchor,
                                mlir::scf::ForOp loop, unsigned count, unsigned width,
                                unsigned trips, unsigned mode, uint64_t& queries)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto zero = e.constant(0), yes = e.boolean(true);
    const unsigned delay = mode == 5 ? 1 : 0, budget = delay * count + width;
    fs::RepeatedRegionState state;
    state.function = function; state.loop = loop; state.body.expressions = arena;
    state.body.capabilities = {true, true, true, false}; state.trips = e.constant(trips);
    std::vector<mlir::pto::CompoundInstanceElement> phases;
    phases.reserve(2 * count);
    for (unsigned t = 0; t < 2 * count; ++t) {
        const auto pipe = t % 2 ? mlir::pto::PipelineType::PIPE_M : mlir::pto::PipelineType::PIPE_MTE1;
        phases.emplace_back(t, mlir::SmallVector<const mlir::pto::BaseMemInfo*>{},
                            mlir::SmallVector<const mlir::pto::BaseMemInfo*>{}, pipe, anchor->getName());
        phases.back().elementOp = anchor;
        state.body.anchors.push_back({&phases.back(), {}, {anchor->getBlock(), anchor},
                                     {anchor->getBlock(), anchor->getNextNode()}});
        state.body.occurrenceLoops.push_back({});
    }
    state.body.presence = [yes](auto) { return std::optional(yes); };
    // Independently unfold a finite event graph. P and Q each have native
    // start/completion chains; P_i -> Q_i and Q_i -> P_(i+width) are supplied
    // completion-to-start edges. Reverse closure uses only these explicit edges.
    auto graph = [&](unsigned handoffs) {
        std::vector<llvm::BitVector> reach(4 * handoffs, llvm::BitVector(4 * handoffs));
        for (unsigned i = 0; i < handoffs; ++i) {
            reach[4*i].set(4*i+1); reach[4*i+2].set(4*i+3);
            if (i + delay * count < handoffs) { reach[4*i+1].set(4*(i+delay*count)+2); }
            if (i + width < handoffs) { reach[4*i+3].set(4*(i+width)); }
            if (i + 1 < handoffs) {
                for (unsigned k = 0; k < 4; ++k) { reach[4*i+k].set(4*(i+1)+k); }
            }
        }
        for (unsigned i = reach.size(); i-- > 0;) {
            for (int j = reach[i].find_first(); j >= 0; j = reach[i].find_next(j)) { reach[i] |= reach[j]; }
            reach[i].set(i);
        }
        return reach;
    };
    const auto local = graph(count);
    state.body.reachability = [&e, local](auto a, auto b) {
        const auto i = 2*a.type + (a.kind == fs::PeriodicEventKind::Completion);
        const auto j = 2*b.type + (b.kind == fs::PeriodicEventKind::Completion);
        return std::optional(e.boolean(local[i][j]));
    };
    state.weightedAcross = [&e, count, width, mode, gate = function.getArgument(1)](auto a, auto b, auto gap) {
        const auto g = e.constantValue(gap);
        if (!g) { return std::optional<fs::RegionExpressions::Id>(); }
        if (mode == 2) { return std::optional(e.input(gate)); }
        // Translation-invariant exact query for the Q-completion/P-start
        // endpoints used by this allocator. Other endpoints are not queried.
        if (a.type % 2 != 1 || b.type % 2 != 0 ||
            a.kind != fs::PeriodicEventKind::Completion || b.kind != fs::PeriodicEventKind::Start) {
            return std::optional<fs::RegionExpressions::Id>();
        }
        return std::optional(e.boolean(*g * count + b.type / 2 >= a.type / 2 + width));
    };
    fs::RegionalAllocationSummary input;
    std::vector<std::pair<uint32_t, std::size_t>> crossings;
    for (unsigned i = count; i-- > 0;) {
        fs::RegionalAllocationMember member{uint32_t(1000 + 3*i), 0, 0,
            {2*i, zero, fs::PeriodicEventKind::Start}, {2*i+1, zero, fs::PeriodicEventKind::Completion}, yes};
        member.tupleRule = fs::PhysicalTupleRule{1, 0, {}}; member.singletonHandoff = true;
        if (mode == 5) {
            crossings.push_back({member.record, state.crossings.size()});
            state.crossings.push_back({member.firstSource, member.lastTarget, yes});
        } else if (mode == 4 && i % 2 == 0) { input.groups.back().members.push_back(member); }
        else { input.groups.push_back({uint32_t(phases[0].kPipeValue), uint32_t(phases[1].kPipeValue), 1, {member}}); }
    }
    if (mode == 6) { state.originalTrips = state.trips; }
    if (mode == 1) { state.originalBegin = zero; } // force the general clipped route
    if (mode == 7) { state.body.capabilities.exactQueries = false; }
    if (mode == 8) { state.trips = e.input(function.getArgument(0)); }
    fs::RepeatedLaneProofCost cost;
    auto outcome = fs::periodicLaneAllocation(state, input, crossings, &cost);
    auto result = outcome.assignment;
    if (mode == 7 || mode == 8 || mode == 9) {
        return !outcome.overCapacity && !cost.orderedOverCapacity && cost.orderedUnresolved && cost.matrixPairs;
    }
    if (mode == 2) { return !cost.orderedAccepted && cost.matrixPairs != 0; }
    if (mode == 3) {
        const auto& witness = outcome.overCapacity;
        if (result || outcome.status != fs::RepeatedLaneAllocationStatus::CertifiedOverCapacity ||
            !witness || witness->spans.size() != 7 || !cost.orderedOverCapacity || cost.matrixPairs ||
            cost.orderedWitnessQueries > 21 || cost.orderedReuseQueries > 2 * count + 6) { return false; }
        // Validate the finite witness against the independent unfolded graph,
        // not the production suffix-query callback or stored negative IDs.
        const auto reach = graph(count * trips);
        for (std::size_t i = 0; i < witness->spans.size(); ++i) {
            const auto& a = witness->spans[i];
            if (a.sourceVisit >= trips || a.targetVisit >= trips || a.records.empty()) { return false; }
            for (std::size_t j = 0; j < witness->spans.size(); ++j) {
                if (i == j) { continue; }
                const auto& b = witness->spans[j];
                const auto end = 4 * (a.targetVisit * count + a.target.type / 2) + 3;
                const auto begin = 4 * (b.sourceVisit * count + b.source.type / 2);
                if (reach[end][begin]) { return false; }
            }
        }
        return true;
    }
    if (!result || result->groups.size() != 1) { return false; }
    if (trips > delay && (mode == 0 || mode == 4 || mode == 5) && (!cost.orderedAccepted || cost.matrixPairs ||
        cost.orderedReuseQueries > 2 * count + 6)) { return false; }
    if (trips && (mode == 1 || mode == 6) && (cost.orderedAccepted || !cost.matrixPairs)) { return false; }
    queries += cost.orderedReuseQueries;
    const auto& palette = result->groups.front();
    if ((mode == 0 || mode == 5) && trips > delay && palette.budget != budget) { return false; }
    const auto unfolded = count <= 16 ? graph(count * trips) : std::vector<llvm::BitVector>();
    std::map<uint64_t, unsigned> previous;
    std::vector<std::vector<unsigned>> uses(palette.budget);
    // Execute in publication order, independently of reverse input/provenance order.
    for (unsigned visit = 0; visit + delay < trips; ++visit) {
        for (unsigned i = 0; i < count; ++i) {
            const auto record = uint32_t(1000 + 3*i);
            const auto found = std::find_if(palette.members.begin(), palette.members.end(),
                [&](const auto& member) { return member.record == record; });
            if (found == palette.members.end() || !found->tupleRule) { return false; }
            const auto& rule = *found->tupleRule;
            uint64_t id = rule.base;
            for (const auto& term : rule.terms) {
                const uint64_t coordinate = term.coordinate == 0 && !delay ? 0 : visit;
                id += term.scale * ((term.stride * coordinate + term.phase) % term.modulus);
            }
            if (id >= palette.budget || ((mode == 0 || mode == 5) && id != (visit * count + i) % budget)) {
                return false;
            }
            const unsigned occurrence = visit * count + i;
            if (!unfolded.empty() && previous.count(id) &&
                !unfolded[4*(previous[id]+delay*count)+3][4*occurrence]) { return false; }
            previous[id] = occurrence; uses[id].push_back(occurrence);
        }
    }
    for (unsigned lane = 0; lane < palette.budget; ++lane) {
        std::vector<unsigned> first, last;
        for (const auto& selected : palette.lanes[lane].firstSources) {
            const auto active = e.constantValue(selected.present);
            if (!active) { return false; }
            if (*active) {
                auto visit = e.constantValue(selected.event.visits.front());
                if (!visit) { return false; }
                first.push_back(*visit * count + selected.event.type / 2);
            }
        }
        for (const auto& selected : palette.lanes[lane].lastTargets) {
            const auto active = e.constantValue(selected.present);
            if (!active) { return false; }
            if (*active) {
                auto visit = e.constantValue(selected.event.visits.front());
                if (!visit) { return false; }
                last.push_back((*visit - delay) * count + selected.event.type / 2);
            }
        }
        if (uses[lane].empty() != first.empty() || first.empty() != last.empty()) { return false; }
        if (!uses[lane].empty() && (*std::min_element(first.begin(), first.end()) != uses[lane].front() ||
            *std::max_element(last.begin(), last.end()) != uses[lane].back())) { return false; }
    }
    return true;
}
bool checkOrderedLaneAllocations(mlir::func::FuncOp function, mlir::Operation* anchor, mlir::scf::ForOp loop)
{
    uint64_t queries = 0;
    for (unsigned count : {1U, 2U, 7U, 16U, 64U, 256U}) {
        for (unsigned width = 1; width <= 6; ++width) {
            for (unsigned trips : {0U, 1U, 2U, 7U}) {
                if (!checkOrderedLaneAllocation(function, anchor, loop, count, width, trips, 0, queries)) {
                    llvm::errs() << "ordered repeated allocation mismatch "
                                 << count << "," << width << "," << trips << "\n";
                    return false;
                }
            }
        }
    }
    for (unsigned count = 1; count <= 3; ++count) {
        for (unsigned width = 1; width <= 3; ++width) {
            for (unsigned trips : {0U, 1U, 2U, 7U}) {
                if (!checkOrderedLaneAllocation(function, anchor, loop, count, width, trips, 5, queries)) {
                    llvm::errs() << "ordered delayed allocation mismatch "
                                 << count << "," << width << "," << trips << "\n";
                    return false;
                }
            }
        }
    }
    if (!checkOrderedLaneAllocation(function, anchor, loop, 2, 3, 7, 1, queries) ||
        !checkOrderedLaneAllocation(function, anchor, loop, 2, 3, 7, 2, queries) ||
        !checkOrderedLaneAllocation(function, anchor, loop, 2, 7, 8, 3, queries) ||
        !checkOrderedLaneAllocation(function, anchor, loop, 8, 1, 7, 4, queries) ||
        !checkOrderedLaneAllocation(function, anchor, loop, 2, 3, 7, 6, queries)) { return false; }
    for (unsigned count : {2U, 7U, 64U, 256U}) {
        if (!checkOrderedLaneAllocation(function, anchor, loop, count, 7, 8, 3, queries)) { return false; }
    }
    if (!checkOrderedLaneAllocation(function, anchor, loop, 2, 7, 8, 7, queries) ||
        !checkOrderedLaneAllocation(function, anchor, loop, 2, 7, 8, 8, queries) ||
        !checkOrderedLaneAllocation(function, anchor, loop, 2, 7, 1, 9, queries)) { return false; }
    llvm::outs() << "ordered repeated allocation: 185 finite/scaling cases, " << queries << " threshold queries\n";
    return true;
}
} // namespace
#endif
