// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedAnalysis.h"
#include "AccessIncidences.h"
#include <algorithm>
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
class ConflictBuilder {
public:
    ConflictBuilder(PredicateArena& arena, ArrayRef<Predicate> occurrences) : arena(arena), presence(occurrences) {}
    void add(std::size_t source, std::size_t consumer, Predicate predicate, std::optional<StorageWitness> witness)
    {
        predicate = arena.conjunction(predicate, arena.conjunction(presence[source], presence[consumer]));
        if (predicate == 0) {
            return;
        }
        auto [entry, added] = endpoints.try_emplace(std::make_pair(source, consumer), edges.size());
        if (added) {
            edges.push_back({source, consumer, predicate, {}});
        } else {
            auto& edge = edges[entry->second];
            edge.predicate = arena.disjunction(edge.predicate, predicate);
        }
        if (witness) {
            edges[entry->second].witnesses.push_back({*witness, predicate});
        }
    }
    void conflicts(const detail::Incidences& incidences, const detail::Neighbors& related)
    {
        for (std::size_t consumer = 0; consumer < incidences.size(); ++consumer) {
            for (std::size_t source = 0; source < consumer; ++source) {
                for (const auto& earlier : incidences[source]) {
                    for (const auto& later : incidences[consumer]) {
                        const auto& neighbors = related[earlier.footprint];
                        if (std::binary_search(neighbors.begin(), neighbors.end(), later.footprint)) {
                            hazards(source, consumer, earlier, later);
                        }
                    }
                }
            }
        }
    }
    SmallVector<GuardedDemand> edges;

private:
    void hazards(
        std::size_t source, std::size_t consumer, const detail::Incidence& earlier, const detail::Incidence& later)
    {
        if (earlier.write && later.read) {
            add(source, consumer, 1, StorageWitness{Hazard::RAW, earlier.footprint, later.footprint, {}});
        }
        if (earlier.read && later.write) {
            add(source, consumer, 1, StorageWitness{Hazard::WAR, earlier.footprint, later.footprint, {}});
        }
        if (earlier.write && later.write) {
            add(source, consumer, 1, StorageWitness{Hazard::WAW, earlier.footprint, later.footprint, {}});
        }
    }
    PredicateArena& arena;
    ArrayRef<Predicate> presence;
    DenseMap<std::pair<std::size_t, std::size_t>, std::size_t> endpoints;
};
} // namespace

LogicalResult GuardedDemandAnalysis::build(
    ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<Predicate> occurrences,
    ArrayRef<StorageFootprint> footprints, ArrayRef<StorageAlias> aliases, PredicateArena predicates,
    ArrayRef<GuardedDemand> additional)
{
    *this = GuardedDemandAnalysis();
    if (sequence.size() != occurrences.size() || sequence.size() > std::numeric_limits<std::size_t>::max() / 2) {
        return failure();
    }
    for (auto predicate : occurrences) {
        if (!predicates.valid(predicate)) {
            return failure();
        }
    }
    for (const auto& edge : additional) {
        const bool valid = edge.source < edge.consumer && edge.consumer < sequence.size() &&
                           predicates.valid(edge.predicate) && edge.witnesses.empty();
        if (!valid) {
            return failure();
        }
    }
    auto incidences = detail::indexAccesses(sequence, footprints);
    auto neighbors = detail::indexAliases(footprints.size(), aliases);
    if (failed(incidences) || failed(neighbors)) {
        return failure();
    }
    GuardedDemandAnalysis pending;
    pending.conditions = std::move(predicates);
    pending.sites.assign(sequence.begin(), sequence.end());
    pending.presence.assign(occurrences.begin(), occurrences.end());
    ConflictBuilder builder(pending.conditions, pending.presence);
    builder.conflicts(*incidences, *neighbors);
    for (const auto& edge : additional) {
        builder.add(edge.source, edge.consumer, edge.predicate, {});
    }
    return buildPrepared(pending.sites, pending.presence, std::move(pending.conditions), builder.edges);
}

LogicalResult GuardedDemandAnalysis::buildPrepared(
    ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<Predicate> occurrences, PredicateArena predicates,
    ArrayRef<GuardedDemand> generators)
{
    *this = GuardedDemandAnalysis();
    if (sequence.size() != occurrences.size() || sequence.size() > std::numeric_limits<std::size_t>::max() / 2) {
        return failure();
    }
    for (auto* phase : sequence) {
        if (!phase) {
            return failure();
        }
    }
    for (auto root : occurrences) {
        if (!predicates.valid(root)) {
            return failure();
        }
    }
    for (const auto& edge : generators) {
        if (edge.source >= edge.consumer || edge.consumer >= sequence.size() || !predicates.valid(edge.predicate)) {
            return failure();
        }
        for (const auto& witness : edge.witnesses) {
            if (!predicates.valid(witness.predicate)) {
                return failure();
            }
        }
    }
    GuardedDemandAnalysis pending;
    pending.conditions = std::move(predicates);
    pending.sites.assign(sequence.begin(), sequence.end());
    pending.presence.assign(occurrences.begin(), occurrences.end());
    pending.demands.assign(generators.begin(), generators.end());
    pending.initializeNative();
    pending.reduce();
    *this = std::move(pending);
    return success();
}

LogicalResult GuardedDemandAnalysis::adjacentLocalUpper()
{
    PredicateArena predicates = conditions;
    ConflictBuilder builder(predicates, presence);
    for (const auto& edge : covers) {
        if (sites[edge.source]->kPipeValue != sites[edge.consumer]->kPipeValue) {
            builder.add(edge.source, edge.consumer, edge.predicate, {});
            continue;
        }
        Predicate later = 0;
        for (auto previous = edge.consumer; previous > edge.source;) {
            --previous;
            if (sites[previous]->kPipeValue != sites[edge.consumer]->kPipeValue) { continue; }
            auto actual = predicates.conjunction(presence[previous], predicates.negate(later));
            builder.add(previous, edge.consumer, predicates.conjunction(edge.predicate, actual), {});
            later = predicates.disjunction(later, presence[previous]);
        }
    }
    GuardedDemandAnalysis pending;
    if (failed(pending.buildPrepared(sites, presence, std::move(predicates), builder.edges))) { return failure(); }
    *this = std::move(pending);
    return success();
}

void GuardedDemandAnalysis::initializeNative()
{
    reach.assign(2 * sites.size(), SmallVector<Predicate>(2 * sites.size(), 0));
    for (std::size_t source = 0; source < sites.size(); ++source) {
        reach[2 * source][2 * source + 1] = presence[source];
        for (std::size_t consumer = source + 1; consumer < sites.size(); ++consumer) {
            if (sites[source]->kPipeValue == sites[consumer]->kPipeValue) {
                const auto predicate = conditions.conjunction(presence[source], presence[consumer]);
                // Every ordered pair is labeled: absent intermediate sites do
                // not break the executed native start/completion chains.
                reach[2 * source][2 * consumer] = predicate;
                reach[2 * source + 1][2 * consumer + 1] = predicate;
            }
        }
    }
}

void GuardedDemandAnalysis::closeRequired()
{
    for (const auto& edge : demands) {
        auto& label = reach[2 * edge.source + 1][2 * edge.consumer];
        label = conditions.disjunction(label, edge.predicate);
    }
    // Strict guarded closure. Potential-event indices provide a common
    // topological order, not additional edges or a flattened execution trace.
    for (std::size_t middle = 0; middle < reach.size(); ++middle) {
        for (std::size_t source = 0; source < middle; ++source) {
            for (std::size_t target = middle + 1; target < reach.size(); ++target) {
                const auto path = conditions.conjunction(reach[source][middle], reach[middle][target]);
                reach[source][target] = conditions.disjunction(reach[source][target], path);
            }
        }
    }
}

void GuardedDemandAnalysis::reduce()
{
    closeRequired();
    for (const auto& edge : demands) {
        const auto source = 2 * edge.source + 1;
        const auto target = 2 * edge.consumer;
        Predicate alternative = 0;
        for (auto middle = source + 1; middle < target; ++middle) {
            alternative = conditions.disjunction(
                alternative, conditions.conjunction(reach[source][middle], reach[middle][target]));
        }
        // Native b_uv is false for every completion-to-start candidate in
        // the core graph. Thus g_uv AND NOT b_uv AND NOT A_uv simplifies here.
        const auto required = conditions.conjunction(edge.predicate, conditions.negate(alternative));
        if (required == 0) {
            continue;
        }
        auto retained = edge;
        retained.predicate = required;
        for (auto& witness : retained.witnesses) {
            witness.predicate = conditions.conjunction(witness.predicate, required);
        }
        covers.push_back(std::move(retained));
        recordLocal(covers.back());
    }
}

void GuardedDemandAnalysis::recordLocal(const GuardedDemand& edge)
{
    if (sites[edge.source]->kPipeValue != sites[edge.consumer]->kPipeValue) {
        return;
    }
    Predicate intervening = 0;
    for (auto site = edge.source + 1; site < edge.consumer; ++site) {
        if (sites[site]->kPipeValue == sites[edge.source]->kPipeValue) {
            intervening = conditions.disjunction(intervening, presence[site]);
        }
    }
    locals.push_back({covers.size() - 1, conditions.conjunction(edge.predicate, intervening)});
}

std::optional<Predicate> GuardedDemandAnalysis::completionBeforeStart(std::size_t source, std::size_t consumer) const
{
    if (source >= sites.size() || consumer >= sites.size()) {
        return {};
    }
    return reach[2 * source + 1][2 * consumer];
}
} // namespace mlir::pto::frontiersynch
