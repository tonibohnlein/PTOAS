// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Analysis of one qualified explicit occurrence sequence. Endpoints index that
// sequence, not SyncIR record IDs. Shared phases/effects remain borrowed.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_DEMANDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_DEMANDANALYSIS_H

#include "PTO/Transforms/FrontierSynch/StorageAnalysis.h"
#include <optional>

namespace mlir::pto::frontiersynch {
enum class Hazard { RAW, WAR, WAW };
struct StorageWitness {
    Hazard hazard = Hazard::RAW;
    std::size_t sourceFootprint = 0;
    std::size_t consumerFootprint = 0;
    std::optional<std::size_t> previousWriter;
    // Shared immutable memory IDs resolve coordinate-qualified bound cells.
    // They never qualify actual overlap, exact effects or full-cell overwrite.
    std::optional<std::size_t> sourceMemory = std::nullopt;
    std::optional<std::size_t> consumerMemory = std::nullopt;
};
struct Demand {
    std::size_t source = 0;
    std::size_t consumer = 0;
    SmallVector<StorageWitness> witnesses;
    Operation* originalBarrier = nullptr;
};
struct ReaderBypass {
    std::size_t writer = 0;
    std::size_t reader = 0;
    std::size_t consumer = 0;
    std::size_t sourceFootprint = 0;
    std::size_t consumerFootprint = 0;
};

class LifetimeAnalysis {
public:
    // The modeled graph has every RAW/WAR/WAW pair over identical or supplied
    // may-alias footprints. Aliases stay pairwise, not equivalence classes.
    // Frontiers are replaced through paths of modeled conflicts, never through
    // inferred full-write coverage. Physical-cell precision is not claimed.
    LogicalResult build(
        ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<StorageFootprint> footprints,
        ArrayRef<StorageAlias> aliases);
    // Same modeled graph, enriched with shared immutable bound-cell witnesses.
    LogicalResult build(ArrayRef<const CompoundInstanceElement*> sequence, const StorageAnalysis& storage);
    ArrayRef<Demand> generators() const { return edges; }
    ArrayRef<ReaderBypass> bypasses() const { return relays; }

private:
    SmallVector<Demand> edges;
    SmallVector<ReaderBypass> relays;
};

struct CompletionSummary {
    const CompoundInstanceElement* phase = nullptr;
    std::size_t pipe = 0;
    std::size_t rank = 0;
    SmallVector<std::size_t> S;
    SmallVector<std::size_t> T;
};
class RankReduction {
public:
    // Core native edges: I->C, preceding I->I, preceding C->C on each pipe.
    // Additional prerequisites may be supplied as forward demands without
    // storage witnesses. Duplicate endpoints are permitted. Build is atomic.
    LogicalResult build(ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<Demand> generators);
    ArrayRef<PipelineType> pipes() const { return columns; }
    ArrayRef<CompletionSummary> summaries() const { return ranks; }
    // Indices into the supplied generator array, preserving its witnesses.
    ArrayRef<std::size_t> retained() const { return minimum; }
    ArrayRef<std::size_t> nonadjacentLocal() const { return nonadjacent; }

private:
    SmallVector<PipelineType> columns;
    SmallVector<CompletionSummary> ranks;
    SmallVector<std::size_t> minimum;
    SmallVector<std::size_t> nonadjacent;
};
} // namespace mlir::pto::frontiersynch
#endif
