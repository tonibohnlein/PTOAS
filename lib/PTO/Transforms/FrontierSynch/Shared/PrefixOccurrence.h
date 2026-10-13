// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Producer-owned occurrence choices. Identity matching is evaluated at the
// original payload; selection conditions are evaluated only at the consumer.
#ifndef PTO_FRONTIERSYNCH_PREFIXOCCURRENCE_H
#define PTO_FRONTIERSYNCH_PREFIXOCCURRENCE_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
struct SequenceAnalysisState;
struct PrefixOccurrence;
using PrefixOccurrencePtr = std::shared_ptr<const PrefixOccurrence>;
// Nodes borrow phase anchors from the contributing semantic input. Consumers
// retain that input and the common expression arena through prefix collection.
struct PrefixOccurrence {
    TemplateEndpointAnchor anchor;
    RegionExpressions::Id match = RegionExpressions::invalid;
    std::optional<RegionExpressions::Id> choose;
    PrefixOccurrencePtr yes, no;
};
using PrefixOccurrenceCache = std::map<uint32_t, PrefixOccurrencePtr>;
PrefixOccurrencePtr sequencePrefixOccurrence(const std::shared_ptr<SequenceAnalysisState>& owner,
    uint32_t port, PrefixOccurrenceCache& cache, std::string& error);
std::optional<RegionExpressions::Id> prefixOccurrenceMatch(RegionExpressions& expressions,
    const PrefixOccurrencePtr& root);
// Only the already certified singleton wrapper may forward collection without
// adding intervisit demands. Lift every primitive occurrence through its frame.
void liftSingletonPrefixContribution(const RegionalAnalysis& body, RegionalAnalysis& lifted);
void attachSequenceOccurrenceRecipes(const RegionalAnalysis& region,
    const std::shared_ptr<SequenceAnalysisState>& state, struct RegionalPrefixContribution& result);
} // namespace mlir::pto::frontiersynch
#endif
