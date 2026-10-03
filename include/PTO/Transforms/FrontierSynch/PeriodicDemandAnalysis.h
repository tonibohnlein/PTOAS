// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numeric periodic analysis over supplied exact generators and borrowed shared
// phases. The fixed executed period and exactness certificate belong to the
// supplier; this class neither imports MLIR control nor reconstructs effects.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICDEMANDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICDEMANDANALYSIS_H

#include "PTO/Transforms/FrontierSynch/RotatingFootprintAnalysis.h"

namespace mlir::pto::frontiersynch {
struct PeriodicPrerequisite {
    std::size_t source = 0;
    std::size_t consumer = 0;
    llvm::DynamicAPInt distance;
};
struct CanonicalPeriodicPrerequisite {
    PeriodicPrerequisite edge;
    SmallVector<std::size_t> origins;
};
// Infinity is explicit: no finite machine integer is a sentinel distance.
using PeriodicDistance = std::optional<llvm::DynamicAPInt>;
enum class PeriodicEventKind { Start, Completion };

class PeriodicDemandReduction {
public:
    // Sites are one complete, fixed executed period in reference order, with
    // fixed pipe assignments. The site identity is its period-position index;
    // distinct positions may borrow the same shared phase in a grouped period.
    // This allows p>1 skeletons without cloning compiler phase records. Records
    // describe C_source(t)->I_consumer(t+distance) for every present instance.
    // The caller certifies that these and the core native graph generate the
    // exact required relation for EVERY admitted finite prefix. General guards,
    // exceptional boundaries and variable skeletons need a different adapter.
    // Source IR and phases must remain alive and unchanged. Original records
    // need only be retained when interpreting origins/witnesses later.
    // Invalid IDs, phases or displacements clear every published result.
    // For h supplied records and g canonical records, comparison sorting costs
    // O(h log(h+1)) and origins occupy O(h). Thereafter V=2m, E=3m+g:
    // O(k(V+E)log(V+1)+kE+g) graph/arithmetic work and O(V+E+km)
    // working storage, in addition to origins. Numeric distances are never
    // expanded. The algorithm imposes no local-adjacency acceptance gate.
    LogicalResult build(ArrayRef<const CompoundInstanceElement*> sites, ArrayRef<PeriodicPrerequisite> generators);
    // Origins first index storage.generators(), then extras. Existing storage
    // witnesses remain in the caller-owned extractor; no effect data is copied.
    // Reject default/failed extractors; a successfully built empty one is valid.
    LogicalResult build(const RotatingFootprintAnalysis& storage, ArrayRef<PeriodicPrerequisite> extras = {});

    ArrayRef<const CompoundInstanceElement*> sites() const { return phaseSites; }
    ArrayRef<CanonicalPeriodicPrerequisite> generators() const { return records; }
    ArrayRef<std::size_t> retained() const { return minimum; }
    ArrayRef<PipelineType> pipes() const { return columns; }
    // Diagnostic graph size; independent of numeric displacements and trips.
    std::size_t vertexCount() const { return vertexTotal; }
    std::size_t edgeCount() const { return edgeTotal; }
    std::size_t shortestPathRuns() const { return offsets.size(); }
    std::size_t indexEntryCount() const { return offsets.size() * vertexTotal; }
    ArrayRef<std::size_t> pipePayloadCounts() const { return pipeSizes; }
    // Immutable beta_p(event) offsets; nullopt means minus infinity. Bounding
    // geometry and physical-effect exactness are not established by this index.
    FailureOr<PeriodicDistance> frontierOffset(
        std::size_t pipe, std::size_t consumer, PeriodicEventKind kind) const;
    // All four reflexive event-kind relations. Native-only start paths are
    // combined with completion-seeded thresholds without another dense index.
    // This uses paired native I/I and C/C chains and reference-forward edges
    // from the supplied fixed-period certificate, not a target/macro claim.
    FailureOr<PeriodicDistance> threshold(
        std::size_t source, PeriodicEventKind sourceKind, std::size_t consumer, PeriodicEventKind targetKind) const;


    // Failure means an invalid site or event kind; successful nullopt means
    // unreachable. C->C includes the length-zero identity (threshold(a,a)=0).
    FailureOr<PeriodicDistance> threshold(std::size_t source, std::size_t consumer, PeriodicEventKind kind) const;
    // Endpoints are assumed present. Negative displacement is invalid; a valid
    // query with no path returns false, including impossible within-period order.
    FailureOr<bool> reaches(
        std::size_t source, std::size_t consumer, PeriodicEventKind kind, const llvm::DynamicAPInt& displacement) const;
    // prefixPayloads counts payloads from the period-zero beginning, allowing
    // incomplete final periods. The queried consumer must be present. Start
    // yields the required S profile; Completion yields T, including its own
    // completion. Entries are one-based completion ranks per pipes(), or zero.
    FailureOr<SmallVector<llvm::DynamicAPInt>> frontier(
        std::size_t consumer, PeriodicEventKind kind, const llvm::DynamicAPInt& period,
        const llvm::DynamicAPInt& prefixPayloads) const;

private:
    SmallVector<const CompoundInstanceElement*> phaseSites;
    SmallVector<CanonicalPeriodicPrerequisite> records;
    SmallVector<std::size_t> minimum;
    SmallVector<PipelineType> columns;
    SmallVector<std::size_t> sitePipes;
    SmallVector<std::size_t> siteRanks;
    SmallVector<std::size_t> pipeSizes;
    SmallVector<SmallVector<PeriodicDistance>> offsets;
    std::size_t vertexTotal = 0;
    std::size_t edgeTotal = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
