// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Cumulative diagnostics only: no expression IDs, proofs or admission policy.
#ifndef PTO_FRONTIER_REGIONALTERNATIVEWORK_H
#define PTO_FRONTIER_REGIONALTERNATIVEWORK_H
#include <cstdint>
#include <map>
#include <string>
namespace mlir::pto::frontiersynch::detail {
struct RegionAlternativeWork {
    uint64_t graphs = 0, graphOccurrences = 0, graphEdges = 0, guardQualification = 0;
    uint64_t queries = 0, numericConstructed = 0, occurrences = 0, edges = 0, dependencies = 0;
    uint64_t candidateConstructed = 0, retainedCandidateConstructed = 0;
    uint64_t registrationAttempts = 0, registrationVisits = 0, registrations = 0;
    uint64_t bindingVisits = 0, bindingConstructed = 0;
    uint64_t finalLookups = 0, finalLookupVisits = 0, finalReached = 0, finalRebuilt = 0;
    std::map<std::string, uint64_t> outcomes;
    static void add(uint64_t& to, uint64_t from = 1)
    {
        to = from > UINT64_MAX - to ? UINT64_MAX : to + from;
    }
    void merge(const RegionAlternativeWork& other)
    {
        add(graphs, other.graphs); add(graphOccurrences, other.graphOccurrences); add(graphEdges, other.graphEdges);
        add(guardQualification, other.guardQualification); add(queries, other.queries);
        add(numericConstructed, other.numericConstructed); add(occurrences, other.occurrences);
        add(edges, other.edges); add(dependencies, other.dependencies);
        add(candidateConstructed, other.candidateConstructed);
        add(retainedCandidateConstructed, other.retainedCandidateConstructed);
        add(registrationAttempts, other.registrationAttempts); add(registrationVisits, other.registrationVisits);
        add(registrations, other.registrations);
        add(bindingVisits, other.bindingVisits); add(bindingConstructed, other.bindingConstructed);
        add(finalLookups, other.finalLookups); add(finalLookupVisits, other.finalLookupVisits);
        add(finalReached, other.finalReached); add(finalRebuilt, other.finalRebuilt);
        for (const auto& [reason, count] : other.outcomes) { add(outcomes[reason], count); }
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
