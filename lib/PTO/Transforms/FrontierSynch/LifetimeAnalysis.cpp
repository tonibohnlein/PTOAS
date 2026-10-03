// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "llvm/ADT/MapVector.h"
#include "AccessIncidences.h"

namespace mlir::pto::frontiersynch {
namespace {
using detail::Incidence;
using detail::Neighbors;
struct Frontier {
    std::optional<std::size_t> writer;
    llvm::MapVector<PipelineType, std::size_t> readers;
};
class GeneratorBuilder {
public:
    GeneratorBuilder(std::size_t count, const Neighbors& neighbors) : related(neighbors), frontiers(count) {}

    void query(std::size_t consumer, const Incidence& access)
    {
        for (auto sourceFootprint : related[access.footprint]) {
            const auto& frontier = frontiers[sourceFootprint];
            StorageWitness witness{Hazard::RAW, sourceFootprint, access.footprint, frontier.writer};
            if (access.read && frontier.writer) {
                add(*frontier.writer, consumer, witness);
            }
            if (access.write) {
                queryWrite(consumer, access, frontier, witness);
            }
        }
    }

    void update(std::size_t site, PipelineType pipe, const Incidence& access)
    {
        auto& frontier = frontiers[access.footprint];
        if (access.write) {
            frontier.writer = site;
            frontier.readers.clear();
        } else {
            frontier.readers[pipe] = site;
        }
    }

    SmallVector<Demand> edges;
    SmallVector<ReaderBypass> relays;

private:
    void add(std::size_t source, std::size_t consumer, const StorageWitness& witness)
    {
        auto [found, inserted] = endpoints.try_emplace(std::make_pair(source, consumer), edges.size());
        if (inserted) {
            edges.push_back({source, consumer, {}});
        }
        edges[found->second].witnesses.push_back(witness);
    }

    void queryWrite(std::size_t consumer, const Incidence& access, const Frontier& frontier, StorageWitness witness)
    {
        witness.hazard = Hazard::WAR;
        for (const auto& reader : frontier.readers) {
            add(reader.second, consumer, witness);
        }
        if (access.read || !frontier.writer) {
            return;
        }
        if (frontier.readers.empty()) {
            witness.hazard = Hazard::WAW;
            add(*frontier.writer, consumer, witness);
        } else {
            relays.push_back(
                {*frontier.writer, frontier.readers.begin()->second, consumer, witness.sourceFootprint,
                 witness.consumerFootprint});
        }
    }

    const Neighbors& related;
    SmallVector<Frontier> frontiers;
    DenseMap<std::pair<std::size_t, std::size_t>, std::size_t> endpoints;
};
void scan(
    GeneratorBuilder& builder, ArrayRef<const CompoundInstanceElement*> sequence, const detail::Incidences& incidences)
{
    for (auto [site, phase] : llvm::enumerate(sequence)) {
        // Query the old state for all cells before any update, including RMW.
        for (const auto& access : incidences[site]) {
            builder.query(site, access);
        }
        for (const auto& access : incidences[site]) {
            builder.update(site, phase->kPipeValue, access);
        }
    }
}
} // namespace

LogicalResult LifetimeAnalysis::build(
    ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<StorageFootprint> footprints,
    ArrayRef<StorageAlias> aliases)
{
    edges.clear();
    relays.clear();
    auto incidences = detail::indexAccesses(sequence, footprints);
    auto neighbors = detail::indexAliases(footprints.size(), aliases);
    if (failed(incidences) || failed(neighbors)) {
        return failure();
    }
    GeneratorBuilder builder(footprints.size(), *neighbors);
    scan(builder, sequence, *incidences);
    edges = std::move(builder.edges);
    relays = std::move(builder.relays);
    return success();
}
LogicalResult LifetimeAnalysis::build(
    ArrayRef<const CompoundInstanceElement*> sequence, const StorageAnalysis& storage)
{
    if (failed(build(sequence, storage.footprints(), storage.aliases()))) { return failure(); }
    for (auto& edge : edges) {
        for (auto& witness : edge.witnesses) {
            witness.sourceMemory = storage.footprints()[witness.sourceFootprint].sharedMemory;
            witness.consumerMemory = storage.footprints()[witness.consumerFootprint].sharedMemory;
        }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
