// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A finite weighted event quotient; weights count whole period advancement.
// Each pipe runs scaled multisource Dijkstra once. Distinct incoming-edge minima implement
// candidate exclusion without rerunning paths or unfolding encoded distances.
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "PeriodicNativeGraph.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
using llvm::DynamicAPInt;
namespace quotient = mlir::pto::frontiersynch::detail;
namespace {
// All conversions of site/pipe sizes to signed integers follow validateSizes.
DynamicAPInt integer(std::size_t value) { return DynamicAPInt(static_cast<std::int64_t>(value)); }

bool validateSizes(std::size_t sites, std::size_t generators)
{
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (sites > maximum / 3 || sites > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return false;
    }
    return generators <= maximum - 3 * sites;
}
LogicalResult validate(ArrayRef<const CompoundInstanceElement*> sites, ArrayRef<PeriodicPrerequisite> generators)
{
    if (!validateSizes(sites.size(), generators.size())) {
        return failure();
    }
    for (const auto* phase : sites) {
        if (!phase) {
            return failure();
        }
    }
    for (const auto& edge : generators) {
        if (edge.source >= sites.size() || edge.consumer >= sites.size() || edge.distance < 0 ||
            (edge.distance == 0 && edge.source >= edge.consumer)) {
            return failure();
        }
    }
    return success();
}

auto endpointKey(const PeriodicPrerequisite& edge) { return std::tie(edge.source, edge.consumer, edge.distance); }
SmallVector<CanonicalPeriodicPrerequisite> canonicalize(ArrayRef<PeriodicPrerequisite> supplied)
{
    SmallVector<CanonicalPeriodicPrerequisite> ordered;
    for (auto [id, edge] : llvm::enumerate(supplied)) {
        ordered.push_back({edge, {id}});
    }
    llvm::stable_sort(
        ordered, [](const auto& left, const auto& right) { return endpointKey(left.edge) < endpointKey(right.edge); });
    SmallVector<CanonicalPeriodicPrerequisite> result;
    for (auto& record : ordered) {
        if (!result.empty() && endpointKey(result.back().edge) == endpointKey(record.edge)) {
            result.back().origins.push_back(record.origins.front());
        } else {
            result.push_back(std::move(record));
        }
    }
    return result;
}

bool validKind(PeriodicEventKind kind)
{
    return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
}
} // namespace

LogicalResult PeriodicDemandReduction::build(
    ArrayRef<const CompoundInstanceElement*> sites, ArrayRef<PeriodicPrerequisite> generators)
{
    PeriodicDemandReduction pending;
    if (failed(validate(sites, generators))) {
        *this = std::move(pending);
        return failure();
    }
    pending.phaseSites.assign(sites.begin(), sites.end());
    pending.records = canonicalize(generators);
    quotient::QuotientGraph graph(2 * sites.size());
    auto metadata = quotient::addNative(graph, sites);
    const auto nativeEdges = graph.edges.size();
    const auto vertices = graph.outgoing.size();
    if (vertices && metadata.pipes.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(PeriodicDistance) / vertices) {
        *this = PeriodicDemandReduction();
        return failure();
    }
    SmallVector<SmallVector<std::size_t>> candidates(metadata.pipes.size());
    for (auto [id, record] : llvm::enumerate(pending.records)) {
        graph.add(2 * record.edge.source + 1, 2 * record.edge.consumer, record.edge.distance);
        candidates[metadata.sitePipes[record.edge.source]].push_back(id);
    }
    SmallVector<unsigned char> retained(pending.records.size(), 0);
    for (std::size_t pipe = 0; pipe < metadata.pipes.size(); ++pipe) {
        const auto scale = integer(metadata.counts[pipe]);
        SmallVector<PeriodicDistance> seeds(vertices);
        for (std::size_t site = 0; site < sites.size(); ++site) {
            if (metadata.sitePipes[site] == pipe) {
                seeds[2 * site + 1] = scale - integer(metadata.siteRanks[site]);
            }
        }
        auto row = quotient::shortestPaths(graph, std::move(seeds), scale);
        const auto minima = quotient::incomingMinima(graph, row, scale);
        for (auto id : candidates[pipe]) {
            const auto& edge = pending.records[id].edge;
            const auto alternative = minima[2 * edge.consumer].excluding(nativeEdges + id);
            // D_p=h_p-beta_p: the minimum form is exactly the draft's
            // maximum-beta exclusion test, including distinct equal edges.
            const auto bound = scale * edge.distance + scale - integer(metadata.siteRanks[edge.source]);
            retained[id] = !alternative || *alternative > bound;
        }
        for (auto& distance : row) {
            if (distance) {
                distance = scale - *distance;
            }
        }
        pending.offsets.push_back(std::move(row));
    }
    for (std::size_t id = 0; id < retained.size(); ++id) {
        if (retained[id]) {
            pending.minimum.push_back(id);
        }
    }
    pending.columns = std::move(metadata.pipes);
    pending.sitePipes = std::move(metadata.sitePipes);
    pending.siteRanks = std::move(metadata.siteRanks);
    pending.pipeSizes = std::move(metadata.counts);
    pending.vertexTotal = graph.outgoing.size();
    pending.edgeTotal = graph.edges.size();
    *this = std::move(pending);
    return success();
}

LogicalResult PeriodicDemandReduction::build(
    const RotatingFootprintAnalysis& storage, ArrayRef<PeriodicPrerequisite> extras)
{
    if (!storage.valid() || extras.size() > std::numeric_limits<std::size_t>::max() - storage.generators().size()) {
        *this = PeriodicDemandReduction();
        return failure();
    }
    SmallVector<PeriodicPrerequisite> generators;
    for (const auto& edge : storage.generators()) {
        generators.push_back({edge.source, edge.consumer, edge.distance});
    }
    generators.append(extras.begin(), extras.end());
    return build(storage.sites(), generators);
}

FailureOr<PeriodicDistance> PeriodicDemandReduction::threshold(
    std::size_t source, std::size_t consumer, PeriodicEventKind kind) const
{
    if (source >= phaseSites.size() || consumer >= phaseSites.size() || !validKind(kind)) {
        return failure();
    }
    const auto pipe = sitePipes[source];
    const auto offset = offsets[pipe][2 * consumer + (kind == PeriodicEventKind::Completion ? 1 : 0)];
    if (!offset) {
        return PeriodicDistance{};
    }
    const auto scale = integer(pipeSizes[pipe]);
    // beta<=h, r>=1: this numerator is nonnegative, so ordinary division
    // implements ceil((r-beta)/h) exactly, including a within-period zero.
    return PeriodicDistance((integer(siteRanks[source]) - *offset + scale - DynamicAPInt(1)) / scale);
}

FailureOr<bool> PeriodicDemandReduction::reaches(
    std::size_t source, std::size_t consumer, PeriodicEventKind kind, const DynamicAPInt& displacement) const
{
    if (displacement < 0) {
        return failure();
    }
    const auto value = threshold(source, consumer, kind);
    if (failed(value)) {
        return failure();
    }
    return value->has_value() && displacement >= **value;
}

FailureOr<SmallVector<DynamicAPInt>> PeriodicDemandReduction::frontier(
    std::size_t consumer, PeriodicEventKind kind, const DynamicAPInt& period, const DynamicAPInt& prefixPayloads) const
{
    if (consumer >= phaseSites.size() || !validKind(kind) || period < 0 || prefixPayloads < 0 ||
        period * integer(phaseSites.size()) + integer(consumer) >= prefixPayloads) {
        return failure();
    }
    SmallVector<DynamicAPInt> result(columns.size());
    const auto target = 2 * consumer + (kind == PeriodicEventKind::Completion ? 1 : 0);
    for (std::size_t pipe = 0; pipe < columns.size(); ++pipe) {
        const auto& offset = offsets[pipe][target];
        if (offset) {
            result[pipe] = std::max(DynamicAPInt(0), integer(pipeSizes[pipe]) * period + *offset);
        }
    }
    return result;
}

FailureOr<PeriodicDistance> PeriodicDemandReduction::frontierOffset(
    std::size_t pipe, std::size_t consumer, PeriodicEventKind kind) const
{
    if (pipe >= columns.size() || consumer >= phaseSites.size() || !validKind(kind)) {
        return failure();
    }
    return offsets[pipe][2 * consumer + (kind == PeriodicEventKind::Completion ? 1 : 0)];
}
FailureOr<PeriodicDistance> PeriodicDemandReduction::threshold(
    std::size_t source, PeriodicEventKind sourceKind, std::size_t consumer, PeriodicEventKind targetKind) const
{
    if (!validKind(sourceKind)) {
        return failure();
    }
    auto result = threshold(source, consumer, targetKind);
    if (failed(result) || sourceKind == PeriodicEventKind::Completion) {
        return result;
    }
    // A path starting at I_a either follows only native starts, or first
    // enters some C_x on a's pipe. C_a reaches that C_x at the same shift,
    // so every path of the latter kind is already in the completion index.
    if (targetKind == PeriodicEventKind::Start && sitePipes[source] == sitePipes[consumer]) {
        DynamicAPInt native(source <= consumer ? 0 : 1);
        if (!*result || native < **result) {
            *result = native;
        }
    }
    return result;
}
