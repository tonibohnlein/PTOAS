// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspection never calls semantic callbacks: lazy exports/caches stay untouched.
#include "PTO/Transforms/FrontierSynch/Shared/RetainedInventory.h"
#include "SequenceAnalysisInternal.h"
#include "RepeatedRegionInternal.h"
#include "../WholeInvocationIdentity.h"
#include <deque>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
void origin(Object& row, Operation* operation)
{
    if (!operation) {
        return;
    }
    std::string location;
    llvm::raw_string_ostream stream(location);
    operation->getLoc().print(stream);
    row["original_operation"] = operation->getName().getStringRef();
    row["original_location"] = stream.str();
}
void periodic(Object& row, const PeriodicAnalysis& value)
{
    row["input_sites"] = value.payloads.size();
    row["generators"] = value.generators.size();
    row["reduced_records"] = value.retained.size();
    row["native_prerequisites"] = value.nativePrerequisites.size();
    row["frontier_rows"] = value.frontiers.size();
    row["reducer"] = "numerical-periodic-frontiers";
}
void concrete(Object& row, const ExplicitAnalysis& value)
{
    row["input_sites"] = value.occurrences.size();
    row["generators"] = value.scan.generators.size();
    row["reduced_records"] = value.reduction.retained.size();
    row["generator_producer"] = "concrete-storage-lifetime-scan";
    row["reducer"] = "explicit-per-pipe-rank";
    row["storage_cells"] = value.storageBoundary.size();
}
void regional(Object& row, const RegionalAnalysis& region)
{
    row["query_interface"] = region.capabilities.exactQueries;
    row["selector_interface"] = region.capabilities.exactSelectors;
    row["endpoint_interface"] = region.capabilities.endpointRecipes;
    row["storage_cells"] = region.storageBoundary.size();
    row["symbolic_effects"] = region.symbolicStorageEffects.size();
    row["query_types"] = region.anchors.size();
    row["arena_nodes"] = region.expressions ? region.expressions->size() : 0;
    row["numeric_visit_preflight_bound_inclusive"] = region.cost.numericVisits;
    row["rotating_residues_inclusive"] = region.cost.rotatingResidues;
    row["phase_descriptions_inclusive"] = region.cost.phaseDescriptions;
    row["physical_fragments_inclusive"] = region.cost.physicalFragments;
    row["selector_partition_visits_inclusive"] = region.cost.selectorPartitionVisits;
    row["selector_partition_atoms_inclusive"] = region.cost.selectorPartitionAtoms;
    if (!region.construction) {
        return;
    }
    const auto& value = *region.construction;
    row["representation"] = value.representation;
    row["generator_producer"] = value.generatorProducer;
    row["reducer"] = value.reducer;
    row["input_sites"] = value.inputSites;
    row["visit_preflight_bound"] = value.enumeratedVisits;
    row["storage_cells"] = value.cells;
    if (value.generators) {
        row["generators"] = *value.generators;
    }
    if (value.reduced) {
        row["reduced_records"] = *value.reduced;
    }
    origin(row, value.original);
}
struct Task {
    uint64_t parent;
    const RegionalAnalysis* region = nullptr;
    const Child* child = nullptr;
    std::shared_ptr<const SequenceAnalysisState> sequence;
    std::shared_ptr<const RepeatedRegionState> repeat;
};
class Inventory {
public:
    Array rows;
    std::deque<Task> pending;
    std::set<const void*> seen;
    void visit(Task task)
    {
        Object row{
            {"id", rows.size()},
            {"parent", task.parent},
            {"representation", "unavailable"},
            {"generator_producer", "unavailable"},
            {"reducer", "unavailable"},
            {"generators", nullptr},
            {"reduced_records", nullptr}};
        if (task.region) {
            regional(row, *task.region);
            task.repeat = task.region->retainedRepetition.lock();
            if (!task.repeat) {
                task.sequence = task.region->retainedSequence.lock();
            }
        }
        if (task.child) {
            leaf(row, *task.child);
        }
        const auto id = rows.size();
        if (task.repeat) {
            repetition(row, task, id);
        } else if (task.sequence) {
            sequence(row, task, id);
        }
        rows.push_back(std::move(row));
    }

private:
    void leaf(Object& row, const Child& child)
    {
        // An adapter may retain its producer without copying its inventories.
        // Report that selection, but never call it to fill unavailable counts.
        const auto owner = child.prefixMathematics ? child.prefixMathematics :
            child.regional.retainedMathematics.lock();
        if (owner) {
            row["provider_backend"] = owner->backend;
            row["provider_region"] = owner->region;
        }
        if (child.originalNode) {
            row["original_region"] = *child.originalNode;
        }
        if (child.loop) {
            origin(row, child.loop);
        }
        if (child.numericTemplate) {
            const auto& value = *child.numericTemplate;
            row["representation"] = "numerical-periodic-template";
            row["generator_producer"] = "two-copy-numeric-atom-lifetime-scan";
            row["visit_preflight_bound"] = value.countedVisits;
            row["enumerated_cells"] = value.atoms.size();
            row["normalized_fragments"] = value.fragments;
            periodic(row, child.periodic);
        } else if (child.finite) {
            if (auto value = child.finite->concrete()) {
                row["representation"] = "finite-concrete";
                concrete(row, *value);
            } else {
                row["representation"] = "finite-guarded";
            }
        } else if (!child.periodic.payloads.empty()) {
            row["representation"] = "periodic-child";
            periodic(row, child.periodic);
        }
    }
    void sequence(Object& row, const Task& task, uint64_t id)
    {
        const auto& state = *task.sequence;
        if (!task.region || !task.region->construction) {
            row["representation"] = "regional-sequence";
            row["generator_producer"] = "regional-storage-and-prerequisite-bridges";
            row["reducer"] = state.numerical ? "numerical-chain-composition" : "guarded-regional-frontiers";
        } else {
            row["endpoint_adapter"] = "regional-sequence";
        }
        row["construction_origin"] = state.constructionOrigin;
        row["children"] = state.children.size();
        row["crossing_records"] = state.crossings.size();
        row["bridge_pair_visits"] = state.bridgeWork.candidates;
        row["bridge_pairs_appended"] = state.bridgeWork.retained;
        row["bridge_residual_pair_visits"] = state.bridgeWork.residualCandidates;
        row["boundary_summary_visits"] = state.boundarySummaryVisits;
        row["ports"] = state.ports.size();
        row["storage_cells"] = state.cells.size();
        row["query_cache_entries"] = state.reachabilityCache.size();
        row["arena_nodes"] = state.expressions.size();
        if (state.originalRegionId) {
            row["original_region"] = *state.originalRegionId;
        }
        if (state.diagnosticRegion) {
            row["diagnostic_region"] = *state.diagnosticRegion;
        }
        origin(row, state.constructionAnchor);
        if (!seen.insert(task.sequence.get()).second) {
            row["shared_owner"] = true;
            return;
        }
        for (const auto& child : state.children) {
            pending.push_back({id, &child.regional, &child, {}, {}});
        }
    }
    void repetition(Object& row, const Task& task, uint64_t id)
    {
        const auto& state = *task.repeat;
        row["representation"] = "regional-repetition";
        row["generator_producer"] = "cross-visit-storage-and-native-boundaries";
        row["reducer"] = "repeated-boundary-frontiers";
        const auto nativeQueries = llvm::count_if(state.queryCrossings, [](const auto& edge) { return edge.native; });
        const auto nativeRetained = llvm::count_if(state.crossings, [](const auto& edge) { return edge.native; });
        row["generators"] = state.queryCrossings.size() - nativeQueries;
        row["reduced_records"] = state.crossings.size() - nativeRetained;
        row["query_crossings_including_native"] = state.queryCrossings.size();
        row["retained_crossings_including_native"] = state.crossings.size();
        row["query_native_boundaries"] = nativeQueries;
        row["retained_native_boundaries"] = nativeRetained;
        row["phase_count"] = state.phaseCount;
        row["query_cache_entries"] = state.queryMemo.size();
        row["body_query_cache_entries"] = state.bodyPortMemo.size();
        row["distance_cache_entries"] = state.distanceMemo.size();
        origin(row, state.loop);
        if (!seen.insert(task.repeat.get()).second) {
            row["shared_owner"] = true;
            return;
        }
        pending.push_back({id, &state.body, nullptr, {}, {}});
    }
};
} // namespace
llvm::json::Object retainedAnalysisInventory(const MathematicalResult& result)
{
    Inventory inventory;
    const auto* selected = &result;
    while (selected->identityComposition && selected->identityComposition->child) {
        inventory.rows.push_back(
            Object{
                {"id", inventory.rows.size()},
                {"representation", selected->backend},
                {"original_region", selected->region}});
        selected = selected->identityComposition->child.get();
    }
    if (selected->sequenceDemands && selected->sequenceDemands->state) {
        inventory.pending.push_back(
            {inventory.rows.empty() ? 0 : inventory.rows.size() - 1,
             nullptr,
             nullptr,
             selected->sequenceDemands->state,
             {}});
    } else if (selected->regionalDemands) {
        inventory.pending.push_back(
            {inventory.rows.empty() ? 0 : inventory.rows.size() - 1, &*selected->regionalDemands, nullptr, {}, {}});
    } else {
        Object row{
            {"id", inventory.rows.size()},
            {"representation", selected->backend},
            {"original_region", selected->region}};
        if (auto value = selected->concreteDemands()) {
            concrete(row, *value);
        }
        inventory.rows.push_back(std::move(row));
    }
    while (!inventory.pending.empty()) {
        auto task = inventory.pending.front();
        inventory.pending.pop_front();
        inventory.visit(std::move(task));
    }
    return Object{
        {"version", 1},
        {"scope", "selected-retained-owners-only"},
        {"inspection", "no-semantic-callbacks"},
        {"rows", std::move(inventory.rows)}};
}
} // namespace mlir::pto::frontiersynch
