// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Fixed executed-body adapter over the unchanged shared modeled effects.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTWRITERREADERINPUT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTWRITERREADERINPUT_H
#include "PTO/Transforms/FrontierSynch/CompactWriterReader.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <functional>
namespace mlir::pto::frontiersynch {
struct CompactPrerequisiteBindings {
    std::vector<CompactAdditionalRequirement> demands;
    std::vector<PeriodicRecord> native;
};
struct CompactWriterReaderBindings {
    // If supplied, these templates cover ALL internal original prerequisites,
    // including carried ones. Native entries must be permanent native facts.
    // The provider certifies their occurrence maps on this same loop/input.
    // Otherwise PhaseIndex maps same-visit edges; relevant carried state is a
    // separate unmet mapping obligation. Incoming/outgoing edges remain boundary obligations.
    std::optional<CompactPrerequisiteBindings> prerequisites;
    // Optional sound distance restriction for two ORIGINAL shared effect IDs.
    // All admitted visits/access choices of the pair must satisfy the interval.
    // nullopt means unknown; reachable=false certifies no conflict. This is a
    // supplied analysis interface, not an instruction-specific recovery path.
    std::function<std::optional<OriginDistanceInterval>(std::size_t, std::size_t, StorageHazard)> accessBounds;
};
enum class CompactInputIssue { None, InvalidBinding, FixedSkeleton, PrerequisiteMapping };
struct CompactWriterReaderInput {
    std::string error;
    CompactInputIssue issue = CompactInputIssue::None;
    std::vector<const CompoundInstanceElement*> phases;
    std::vector<PeriodicPayload> payloads;
    std::vector<CompactClassAccess> accesses;
    std::vector<CompactSourceQuery> queries;
    std::vector<CompactAdditionalRequirement> additional;
    std::vector<PeriodicRecord> native;
    // Stable shared effect IDs in each conservative physical class/access.
    // Components join unresolved overlaps; class IDs are local to this input.
    std::vector<std::vector<std::size_t>> classEffects, accessEffects;
    // Includes incoming/outgoing and phase-less descriptor/control consumers.
    std::vector<ValuePrerequisite> boundaryPrerequisites;
    // overlapQueries counts EVERY class/pair overlap attempt, including unknown
    // geometry and address-space shortcuts, not just calls to mayOverlap.
    uint64_t overlapQueries = 0, effectPairQueries = 0, suppliedBoundQueries = 0;
    uint64_t protectedPairs = 0, visitProtectedPairs = 0;
};
// Shared input and PhaseIndex must be built on the same unchanged original IR.
// This adapter requires a fixed single-phase-per-site executed body; nested or
// simultaneous envelopes return a structural obligation, never a precision gate.
// It consumes every modeled effect in that body. Empty modeled sets are empty;
// missing geometry joins wider classes. Numeric ranges separate classes only
// when their shared materialization is uniform across visits. Same-visit affine
// cancellation is NOT a cross-visit disjointness proof.
// No descriptor/range is promoted to a definite full-class overwrite: this
// adapter exports no kills until a separate full-overwrite proof is supplied.
// Storage protection constrains only storage candidates, never prerequisites.
// Boundary requirements are returned separately; this is not a whole-function
// closure or endpoint certificate. Existing manual synchronization earns no
// reachability credit; dispatch still owns its normal manual-protocol policy.
// Source-specific queries require the indexed
// core path, not the unqualified two-sweep specialization.
// Costs charge all effect-pair comparisons (quadratic worst case), independently
// of runtime trips, storage bytes and numerical distance bounds. Boundary capture
// additionally walks original function syntax and prerequisite incidences once;
// the existing structured-protection prepass has its own whole-input summary
// and overlap-query costs, not included in these local pair counters.
// A nonempty error makes the partial output unusable. No IR mutation.
CompactWriterReaderInput buildCompactWriterReaderInput(scf::ForOp loop,
    const SyncInput& input, const PhaseIndex& index, const CompactWriterReaderBindings& bindings = {});
} // namespace mlir::pto::frontiersynch
#endif
