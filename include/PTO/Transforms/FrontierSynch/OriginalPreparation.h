// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALPREPARATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALPREPARATION_H
#include "PTO/Transforms/FrontierSynch/OriginalRequests.h"
#include <set>

namespace mlir::pto::frontiersynch {
// Conservative source adjacency is factored through cell/role buckets: it does
// not materialize source x deadline pairs. Membership remains a separate query.
struct OriginalSourceBucket {
    std::size_t cell = NoControlId;
    OriginalObligationKey::Role role = OriginalObligationKey::Role::Writer;
    std::vector<OriginalObligationFamilyId> families;
};
struct OriginalPreparedHook {
    OriginalCut sufficient;
    std::optional<OriginalCut> executable;
    bool conservative = false;
    std::vector<DescriptorFactRef> references;
    std::string obstruction;
};
struct OriginalConsequence {
    OriginalObligationId due, first, second;
    std::size_t middle = NoControlId;
    OriginalCut source;
    std::size_t condition = ObligationConditions::no;
};
struct OriginalConsequenceResult {
    // Each opportunity is conditional on its condition in this owned arena.
    // Unknown cases are retained; no candidate grants completion or deletes due.
    ObligationConditions predicates;
    std::vector<OriginalConsequence> opportunities;
    std::vector<std::string> obstructions;
};
class OriginalPreparation {
public:
    struct Stats {
        std::size_t descriptorNodes = 0, references = 0, roles = 0, links = 0;
        std::size_t accessIncidences = 0, consequenceQueries = 0, consequenceComputations = 0;
    };
    explicit OriginalPreparation(const ProgramAnalysis& analysis);
    bool current() const;
    bool formed() const { return current() && frozen; }
    bool repertoireComplete() const { return formed() && failures.empty(); }
    const std::vector<std::string>& obstructions() const { return failures; }
    const std::vector<OriginalPreparedHook>& hooks() const;
    const std::vector<OriginalSourceBucket>& sourceBuckets() const;
    const std::vector<std::size_t>& bucketsAt(std::size_t phase) const;
    const std::vector<OriginalObligationFamilyId>& atDeadline(std::size_t site) const;
    const std::map<DescriptorFactRef, OriginalSupportRole>& supportRoles() const
    {
        static const std::map<DescriptorFactRef, OriginalSupportRole> empty;
        return formed() ? roles : empty;
    }
    const Stats& stats() const { return work; }
    // Full original interval is part of the cache key. This initial native
    // matcher qualifies invocation D1 chains; recurring/D4 joins need an
    // explicit common dynamic middle relation and remain obstructed.
    const OriginalConsequenceResult& consequences(OriginalObligationId due, const OriginalInterval& interval) const;
private:
    void subscribe(OriginalCut cut, bool conservative, DescriptorFactRef ref = {});
    void prepare();
    const ProgramAnalysis& analysis;
    OriginalProgramVersion version;
    bool frozen = false;
    std::vector<std::string> failures;
    std::vector<OriginalPreparedHook> preparedHooks;
    std::map<OriginalCut, std::size_t> hookIndex;
    std::vector<OriginalSourceBucket> buckets;
    std::map<std::size_t, std::vector<std::size_t>> bySource;
    std::map<std::size_t, std::vector<OriginalObligationFamilyId>> byDeadline;
    std::map<DescriptorFactRef, OriginalSupportRole> roles;
    using ConsequenceKey = std::tuple<std::size_t, ObligationOrigin, OriginalInterval>;
    mutable std::map<ConsequenceKey, OriginalConsequenceResult> consequenceCache;
    mutable Stats work;
};
} // namespace mlir::pto::frontiersynch
#endif
