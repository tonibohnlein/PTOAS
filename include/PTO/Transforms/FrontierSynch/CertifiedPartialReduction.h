// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_CERTIFIEDPARTIALREDUCTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_CERTIFIEDPARTIALREDUCTION_H
#include "PTO/Transforms/FrontierSynch/BoundingRegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
#include "llvm/ADT/ArrayRef.h"
namespace mlir::pto::frontiersynch {
struct PartialReductionGenerator {
    RequirementRecordKey record;
    RegionalEvent source, target;
    RegionExpressions::Id active = RegionExpressions::invalid;
};
// Freeze this object before issuing attempts. All callbacks and borrowed IR
// must describe this same immutable B union G for the lifetime of the result.
// actualPaths certifies actual paths in that graph (unavailable answers are
// permitted). nativePaths, when supplied, certifies only paths in permanent B.
// A possible/overapproximated path is never admissible deletion evidence.
struct CertifiedPartialGraph {
    RegionalOrderView actualPaths;
    std::optional<RegionalOrderView> nativePaths;
    std::vector<PartialReductionGenerator> generators;
    std::shared_ptr<const RequirementProvenance> provenance;
};
using PartialGraphSnapshot = std::shared_ptr<const CertifiedPartialGraph>;
struct PartialReductionAttempt {
    PartialGraphSnapshot graph;
    std::size_t generator = 0; // Original vector index, independent of record identity.
    RegionExpressions::Id guard = RegionExpressions::invalid;
    // Missing intermediate requests only the separate permanent-native proof.
    // Otherwise its payload must lie strictly between the two endpoint payloads.
    std::optional<RegionalEvent> intermediate;
};
struct PartialReductionCost {
    uint64_t attempts = 0, presenceQueries = 0, orderQueries = 0, pathQueries = 0;
    uint64_t implicationChecks = 0, expressionNodes = 0;
};
struct CertifiedPartialReduction {
    std::string error;
    PartialGraphSnapshot graph;
    // Same order and cardinality as graph->generators. Original active guards,
    // endpoints, record keys and unreduced group provenance stay in graph.
    std::vector<RegionExpressions::Id> retained, removed;
    PartialReductionCost cost;
    ReductionQuality reduction = ReductionQuality::Partial;
    // Success proves closure preservation, never full cover reduction or
    // equality of the selected input order with the original required order.
};
// Requires forward C->S generators on the captured domain; original active
// guards need not include endpoint presence. All proof guards include it.
// Absent query answers retain cases. Malformed contexts, events, expression
// types, callback answers and explicit forwardness contradictions are errors.
// Unknown forwardness disables attempts for that record, retaining its guard.
// O(|G|desc + (g+q)*C) work for g records and q supplied attempts with bounded
// callback/implication cost C. No closure, witness search, adjacency filtering or identity merging.
// Strict intermediate payloads prove non-cover status in the frozen DAG, so
// all deletions are simultaneous even when proofs use other removed records.
CertifiedPartialReduction reduceCertifiedPartial(
    PartialGraphSnapshot graph, llvm::ArrayRef<PartialReductionAttempt> attempts);
} // namespace mlir::pto::frontiersynch
#endif
