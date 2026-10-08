// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Fixed executed body: supplied class incidences and distance facts, never
// footprint recovery. All construction is independent of runtime trip counts.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTWRITERREADER_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTWRITERREADER_H
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include "PTO/Transforms/FrontierSynch/LogicalEndpoints.h"
#include "PTO/Transforms/FrontierSynch/OriginDistances.h"
namespace mlir::pto::frontiersynch {
struct CompactClassAccess {
    uint32_t site = 0, storageClass = 0;
    bool read = false, write = false;
    // A supplied proof of a definite full-class overwrite, not a materialized
    // range or a possible write. Queries precede this kill and origin generation.
    bool fullOverwrite = false;
};
struct FixedBodyOverwriteIndex {
    std::string error;
    uint32_t sites = 0;
    std::vector<CompactClassAccess> accesses;
    // Parallel to accesses: positive site advances, including the next visit
    // to the origin itself. Null means no definite overwrite of its class.
    std::vector<std::optional<uint64_t>> nextOverwrite;
    uint64_t indexingSteps = 0;
    // Null is malformed input; reachable=false is a valid empty interval.
    std::optional<OriginDistanceInterval> distances(uint32_t originAccess, uint32_t querySite) const;
};
// Accesses must be in body order and unique per (site,class). Sparse class/pipe
// labels do not determine allocation sizes. Expected O(m+z) work/storage.
FixedBodyOverwriteIndex indexFixedBodyOverwrites(
    uint32_t sites, llvm::ArrayRef<CompactClassAccess> accesses);

struct CompactSourceQuery {
    uint32_t sourceAccess = 0, targetAccess = 0;
    StorageHazard hazard = StorageHazard::RAW;
    OriginDistanceInterval bounds{true, 0, std::nullopt};
};
struct CompactAdditionalRequirement {
    uint32_t source = 0, target = 0;
    OriginDistanceInterval bounds{true, 0, std::nullopt};
};
struct CompactSelectedCandidate {
    // Query/extra indices retain the original supplied namespace. Coarse
    // candidates instead identify their original access pair and hazard.
    std::optional<uint32_t> query, additional;
    std::optional<uint32_t> sourceAccess, targetAccess;
    StorageHazard hazard = StorageHazard::Supplied;
    OriginDistanceInterval distances;
    PeriodicRecord selected; // Meaningful only when distances.reachable.
    std::optional<uint32_t> consolidated; // Covering winner, not identity equality.
};
struct CompactAdjacentReplacement {
    uint32_t firstRecord = 0; // Original firstReduction.generators index.
    uint32_t upperRecord = 0; // Canonical upper.generators index, even if reduced.
    // Both record versions are retained in the corresponding analyses. Their
    // distance difference includes the explicitly added startup synchronization.
};
struct CompactWriterReaderCost {
    uint64_t indexingSteps = 0, distanceQueries = 0, historyVisits = 0;
    uint64_t candidates = 0, consolidationSlots = 0;
    // Sizes of the two numerical quotient problems, not relaxation counts.
    uint64_t quotientPasses = 0, quotientEdges = 0;
};
struct CompactWriterReaderAnalysis {
    std::string error;
    std::vector<CompactSelectedCandidate> candidates;
    std::vector<PeriodicRecord> consolidated;
    std::vector<uint32_t> consolidatedToFirst;
    PeriodicAnalysis firstReduction, upper;
    std::vector<CompactAdjacentReplacement> adjacency;
    LogicalEndpointPlan endpoints;
    CompactWriterReaderCost cost;
};
// Caller supplies complete sound class/mode and query coverage of the unchanged
// modeled requirements. Origin-specific disjointness, protection or distance
// tests belong in this explicit query list/bounds. Indexed intervals intersect
// the supplied bounds; the chosen lower distance selects a later same-pipe source.
// Additional requirements are supplied independently; native prerequisites stay
// fixed through both reductions. Unreachable intervals add no candidate edge.
CompactWriterReaderAnalysis analyzeCompactWriterReader(llvm::ArrayRef<PeriodicPayload> payloads,
    llvm::ArrayRef<CompactClassAccess> accesses, llvm::ArrayRef<CompactSourceQuery> queries,
    llvm::ArrayRef<CompactAdditionalRequirement> additional = {},
    llvm::ArrayRef<PeriodicRecord> nativePrerequisites = {});
// Qualified specialization ONLY when no origin-specific filtering/bounds apply.
// Two sweeps with latest writer/reader per class/pipe emit the same consolidated
// candidates as the indexed class/kill analysis. A may-write alone never clears
// histories. Expected O(k(m+z)) before numerical reduction; no visit expansion.
CompactWriterReaderAnalysis analyzeCoarseCompactWriterReader(llvm::ArrayRef<PeriodicPayload> payloads,
    llvm::ArrayRef<CompactClassAccess> accesses, llvm::ArrayRef<CompactAdditionalRequirement> additional = {},
    llvm::ArrayRef<PeriodicRecord> nativePrerequisites = {});
// Both paths consolidate by (consumer,source pipe), then reduce, replace only
// surviving local records by immediate pipe predecessors, and reduce again.
// The upper graph covers the supplied input requirements, with minimum
// generators for its own closure. Entry/exit effects are separate obligations.
} // namespace mlir::pto::frontiersynch
#endif
