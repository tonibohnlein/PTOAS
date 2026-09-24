// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALLIFETIMES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALLIFETIMES_H
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include <memory>

namespace mlir::pto::frontiersynch {
// Adapted from OAHS StorageFrontiers.h. These are original-program may facts;
// none represents completion acquired by a selected synchronization command.
struct StorageOrigin {
    std::size_t site = NoControlId, operation = NoControlId;
};
struct StorageRelationship {
    enum Kind { RAW, WAR, WAW } kind = RAW;
    std::size_t cell = 0;
    StorageOrigin source, target;
};
struct StorageLifecycle {
    bool reachable = false;
    std::size_t cell = 0;
    StorageOrigin access;
    std::vector<StorageOrigin> previousWriters, previousReaders;
    std::vector<StorageOrigin> nextWriters, nextReaders;
    bool mayHaveNoPriorFullWrite = true;
    bool mayExitWithoutFurtherAccess = true;
};
struct OriginalUseQuery {
    std::size_t cell = 0, owner = NoControlId;
    std::vector<std::size_t> starts, stops;
    bool includeStops = false;
    bool read = true, write = true, stopAtOtherAccess = true;
};
struct PhysicalUseFrontier {
    enum class Status { Unknown, NoHit, Present } status = Status::Unknown;
    std::vector<StorageOrigin> accesses;
    bool reachesBoundary = false;
    std::string reason;
};
struct OriginalAccessSummary {
    bool complete = false;
    std::vector<StorageOrigin> readers, writers;
};
// Stable original positions are kept separately from a future selected word.
struct SourceMilestone {
    std::size_t operation = NoControlId;
    enum Side { Before, After } side = After;
};
// A query between original operation boundaries. A missing stop means the
// designated owner's continuation through exit. Selected words are absent.
struct OriginalContinuationQuery {
    std::size_t cell = 0, owner = NoControlId;
    SourceMilestone start, stop;
    bool read = true, write = true;
};
struct OriginalMayAfter {
    enum class Status { Unknown, NoHit, May } status = Status::Unknown;
    std::vector<StorageOrigin> witnesses;
    bool mayBypassStop = false;
    std::string reason;
};
struct OriginalRequirement {
    StorageRelationship relationship;
    SourceMilestone source, deadline;
    // Every original translated access incidence represented by this physical
    // demand. Coalescing duplicate cell/kind reasons must not erase a witness.
    // Stable shared incidence lists owned by OriginalLifetimes; requests and
    // support inventories do not copy them for every may-origin relationship.
    const std::vector<std::size_t>* sourceEffects = nullptr;
    const std::vector<std::size_t>* targetEffects = nullptr;
    PipelineType sourceEngine = PipelineType::PIPE_UNASSIGNED;
    PipelineType targetEngine = PipelineType::PIPE_UNASSIGNED;
    // The ordinary physical demand is for whole-operation completion. A
    // separately qualified native access protection may later discharge only
    // its particular access obligation.
    enum class Scope { WholeOperation } completionScope = Scope::WholeOperation;
    // A source boundary is a subscription, not a publication or causal credit.
    bool sourceSubscribed = false;
    bool sourceGapExecutable = false, deadlineGapExecutable = false;
    bool occurrenceKnown = false;
    bool episodeKnown = false;
};
struct SourceSubscription {
    SourceMilestone position;
    // Earliest analytically sufficient source, which can be inside a
    // multi-phase instruction and therefore not directly executable.
    SourceMilestone sufficientPosition;
    std::size_t deadlineOperation = NoControlId, cell = 0;
    StorageRelationship::Kind kind = StorageRelationship::RAW;
    // Stable index in requirementsAt(deadlineOperation).
    std::size_t requirementIndex = 0;
    bool executableInOriginalIR = false;
};
// The affected original continuation between a producer and a possible reuse.
// Complete means every reachable original site in this interval was inspected;
// it does not assert that the producer established a full content generation.
struct OriginalSupportInterval {
    bool complete = false, reachesReuse = false, mayBypass = false;
    bool mayReload = false, mayReenter = false;
    // An uninterrupted physical interval does not imply that its producer
    // overwrote all previous contents. Keep that stronger generation proof
    // separate from the interval and its outstanding older obligations.
    bool stablePhysicalInterval = false, generationEstablished = false;
    std::size_t producer = NoControlId, reuse = NoControlId, cell = 0;
    std::vector<StorageOrigin> readers;
    std::vector<OriginalRequirement> affectedRequirements;
    std::string reason;
};
struct OriginalLifetimeStats {
    std::size_t originalSites = 0, accessIncidences = 0, storageWords = 0;
    std::size_t forwardEvaluations = 0, backwardEvaluations = 0;
    std::size_t requirements = 0, frontierQueries = 0, frontierSites = 0;
};
class OriginalLifetimes {
public:
    explicit OriginalLifetimes(const OriginalStructure&);
    ~OriginalLifetimes();
    OriginalLifetimes(const OriginalLifetimes&) = delete;
    OriginalLifetimes& operator=(const OriginalLifetimes&) = delete;
    bool complete() const;
    const std::string& reason() const;
    const OriginalLifetimeStats& stats() const;
    StorageLifecycle lifecycleAt(std::size_t operation, std::size_t cell) const;
    const std::vector<OriginalRequirement>& requirementsAt(std::size_t operation) const;
    const std::vector<SourceSubscription>& subscriptionsAt(std::size_t operation) const;
    PhysicalUseFrontier firstUse(const OriginalUseQuery&) const;
    PhysicalUseFrontier lastUse(const OriginalUseQuery&) const;
    OriginalSupportInterval supportBetween(std::size_t producer, std::size_t reuse, std::size_t cell) const;
    // All represented accesses of this cell in the original region. This is a
    // may-set, not proof that any particular member participates in an episode.
    OriginalAccessSummary all(std::size_t owner, std::size_t cell) const;
    OriginalMayAfter mayAfter(const OriginalContinuationQuery&) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::frontiersynch
#endif
