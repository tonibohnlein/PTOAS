// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Attempt-owned diagnostics; no generated values or admission policy.
#ifndef PTO_FRONTIERSYNCH_PREFIXEXPRESSIONPROFILE_H
#define PTO_FRONTIERSYNCH_PREFIXEXPRESSIONPROFILE_H
#include "llvm/Support/JSON.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
namespace mlir {
class Operation;
namespace pto::frontiersynch {
struct PrefixOccurrenceRestrictionWork;
class PrefixExpressionProfile {
public:
    // Diagnostic snapshots own strings/counts only. Query outputs can alias or
    // overlap, so this is source-DAG lineage, never exclusive emitted work.
    struct QueryGroup {
        uint64_t child = 0, outputs = 0;
        std::string producer, kinds;
    };
    struct CoordinateInput {
        uint64_t expression = 0;
        std::string original, evaluationLoop;
        std::optional<uint64_t> evaluationOrdinal;
    };
    struct Coordinate {
        uint64_t expression = 0, nodes = 0, inputCount = 0;
        std::optional<uint64_t> constant;
        std::vector<CoordinateInput> inputs;
    };
    struct QueryEvent {
        uint64_t type = 0, visitCount = 0, outerLoopCount = 0;
        std::string kind, anchor, occurrenceLoop;
        Coordinate ordinal;
        std::vector<Coordinate> visits;
        std::vector<std::string> outerLoops;
    };
    struct QueryPair {
        uint64_t child = 0, output = 0;
        std::string producer;
        QueryEvent source, target;
    };
    static constexpr std::size_t maxQueryDetails = 64;
    static constexpr std::size_t maxCoordinateInputs = 8;
    static constexpr std::size_t maxFrameCoordinates = 8;
    struct SequenceLineage {
        std::string origin, anchor, source;
        std::optional<uint64_t> region;
        std::vector<std::string> frame;
        uint64_t sourceNodes = 0, cachedQueries = 0, reachableQueries = 0;
        uint64_t queryGroups = 0, omittedQueryPairs = 0;
        uint64_t entries = 0, prefixQueries = 0, suffixQueries = 0;
        std::vector<QueryGroup> groups;
        // Expression IDs are identities in the entry's arena, not executable
        // values. Details own snapshots only; omitted counts preserve scope.
        uint64_t omittedQueryDetails = 0;
        std::vector<QueryPair> queries;
    };
    struct Entry {
        uint64_t arena = 0, root = 0, requests = 0, emissions = 0, attempts = 0;
        uint64_t directFailures = 0, inactiveAttempts = 0, historyAttempts = 0;
        uint64_t constructed = 0, retained = 0, maximumRetained = 0;
        uint64_t inactiveSuccesses = 0, preparationNodes = 0, preparationInputs = 0, sharedReturns = 0,
                 bindingNodes = 0;
        uint64_t thresholdDiscovery = 0, thresholdSubstitution = 0, proofVisits = 0, historySubstitution = 0;
        uint64_t thresholdDependencies = 0, thresholdReused = 0, thresholdSkipped = 0, thresholdPreparationHits = 0;
        uint64_t distance = 0;
        std::string cut, location, producer;
        uint64_t lineageRequests = 0;
        std::vector<SequenceLineage> sequenceLineage;
    };
    Entry& request(const void* arena, uint32_t root, Operation* cut,
                   const std::string& producer = "regional", uint64_t distance = 0);
    // Called after request(); does not charge another recipe request. Retains
    // at most four original-source samples per shared expression entry.
    void annotateSequence(const void* arena, uint32_t root, Operation* cut, SequenceLineage lineage);
    void recordOccurrenceRestriction(const PrefixOccurrenceRestrictionWork& work);
    void resetEmission();
    llvm::json::Object report() const;
private:
    uint64_t incomingRequests = 0, incomingChoices = 0, incomingPruned = 0, incomingResidual = 0;
    uint64_t incomingProofVisits = 0;
    using Key = std::tuple<const void*, uint32_t, Operation*, std::string, uint64_t>;
    std::map<Key, Entry> entries;
    std::map<const void*, uint64_t> arenas;
};
} // namespace pto::frontiersynch
} // namespace mlir
#endif
