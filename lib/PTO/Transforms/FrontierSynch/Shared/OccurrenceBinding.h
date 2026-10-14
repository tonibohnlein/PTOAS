// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared original-occurrence interpretation. These recipes contain arena IDs,
// never generated SSA. Protocol owners still supply site/invocation activity,
// availability, publication and cleanup; coordinate constraints are exact tests,
// not the necessary-domain approximation used by endpoint scheduling.
#ifndef PTO_FRONTIERSYNCH_OCCURRENCEBINDING_H
#define PTO_FRONTIERSYNCH_OCCURRENCEBINDING_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
#include <map>
namespace mlir::pto::frontiersynch {
std::optional<RegionExpressions::Id> originalLoopOrdinal(
    RegionExpressions& expressions, scf::ForOp loop, uint64_t divisor = 1);
// Event predicate, leaf ordinal, outer quotient coordinates, then fixed anchor
// coordinates, in that order. The caller combines these with its activity
// using its existing masking semantics. No original values are evaluated here.
FailureOr<SmallVector<RegionExpressions::Id>> regionalOccurrenceConstraints(
    const RegionalAnalysis& region, const RegionalEvent& event, std::string& error);
// Attempt-owned exact finite-point activity proof. Both sequence and repeated
// producers use original cut facts and the authoritative occurrence constraints.
// Failure means the compact point representation is unavailable, not inactivity.
class FixedPointActivity {
public:
    explicit FixedPointActivity(RegionExpressions& arena) : arena(arena) {}
    bool prove(const RegionalAnalysis& region, const RegionalEvent& event,
               const TemplateEndpointAnchor& anchor, Operation* cut, RegionExpressions::Id active);
private:
    RegionExpressions& arena;
    std::map<std::pair<const TemplateEndpointAnchor*, Operation*>,
             std::unique_ptr<RegionExpressions::Substitution>> contexts;
};
} // namespace mlir::pto::frontiersynch
#endif
