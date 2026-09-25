// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_OCCURRENCEQUERIES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_OCCURRENCEQUERIES_H
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include "PTO/Transforms/FrontierSynch/FixedVisitSources.h"
#include "PTO/Transforms/FrontierSynch/PeriodicCorrespondence.h"
#include <string>
namespace mlir::pto::frontiersynch {
// These are original-use facts. Unknown retains the original physical effects.
struct PhysicalBankCorrespondence {
    bool exactPermutation = false;
    std::size_t owner = NoControlId;
    const BaseMemInfo* memory = nullptr;
    std::size_t distance = 0;
    std::vector<std::size_t> participatingOperations;
    std::string reason;
    OriginalValueQualification endpointQualification;
    OriginalProgramVersion version;
    std::size_t relation = NoControlId;
    periodic_uses::Permutation permutation;
    std::vector<std::size_t> phaseBanks;
    // Geometry does not assert exactly-once participation of every static use.
    std::vector<std::size_t> conditionalOrRepeatedOperations;
};
// The incoming source is an interface placeholder, not an executable publication.
// Its condition and target qualification remain inspectable independently.
struct FixedVisitEndpointPair {
    std::size_t source = NoControlId;
    bool incoming = false;
    OriginalCut sourceCut, targetCut;
    std::size_t sourceCondition = 0, targetCondition = 0;
    OriginalValueQualification sourceQualification, targetQualification;
};
struct FixedVisitSourceFrontier {
    FixedVisitSources sources;
    std::vector<FixedVisitEndpointPair> endpoints;
    bool physicalRolesQualified = false, localGuardsQualified = false, hasIncoming = false;
};
struct FixedVisitCorrespondence {
    bool exact = false;
    std::size_t source = NoControlId, target = NoControlId;
    std::string reason;
    // exact is conditional on these predicates, not an unconditional edge.
    std::size_t sourceCondition = 0, targetCondition = 0;
    OriginalValueQualification sourceQualification, targetQualification;
    std::shared_ptr<const FixedVisitSourceFrontier> alternatives;
};
struct ChildOccurrence {
    std::size_t owner = NoControlId, child = NoControlId, position = 0;
    std::vector<std::size_t> operations;
    bool canSkip = false, canRepeat = false;
};
class OriginalLifetimes;
class OccurrenceQueries {
public:
    explicit OccurrenceQueries(
        const OriginalStructure& original, const OriginalValueQueries* values = nullptr,
        const OriginalLifetimes* storage = nullptr);
    ~OccurrenceQueries();
    PhysicalBankCorrespondence bank(std::size_t relationId) const;
    // D2 local physical-use matching, with directional boundary domains. This
    // neither kills provenance nor transports an unfinished use across re-entry.
    PeriodicUseCorrespondence periodic(
        const OriginalInterval& interval, bool sourceWrites, bool targetWrites) const;
    FixedVisitCorrespondence fixedVisit(std::size_t source, std::size_t target, std::size_t cell) const;
    FixedVisitCorrespondence fixedVisit(
        std::size_t source, std::size_t target, std::size_t cell, FactoredUseNode::Hazard hazard) const;
    // Complete target-family query, including targets having ONLY incoming origins.
    // Uses the target's declared single-visit frame, not an enclosing recurrence.
    std::shared_ptr<const FixedVisitSourceFrontier> fixedSourcesAt(
        std::size_t target, std::size_t cell, FactoredUseNode::Hazard hazard) const;
    std::vector<ChildOccurrence> children(std::size_t owner) const;

private:
    const OriginalStructure& original;
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::frontiersynch
#endif
