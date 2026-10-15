// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Attempt-local aggregate profiling. Bucket times are exclusive, not spans.
#ifndef PTO_FRONTIERSYNCH_ALLOCATIONPROFILE_H
#define PTO_FRONTIERSYNCH_ALLOCATIONPROFILE_H
#include "PTO/Transforms/FrontierSynch/AnalysisLedger.h"
#include <array>
#include <chrono>
#include <memory>
namespace mlir::pto::frontiersynch {
class AllocationProfile {
public:
    enum Stage {
        Summary, Groups, Queries, Proofs, Lanes, Serialization, GroupPairs, LanePairs, BoundaryPairs,
        ArithmeticProjection, ArithmeticComposition, ArithmeticDifference, ArithmeticExtrema, ArithmeticOrder,
        PublicationQualification, ArithmeticSelectorExport, ArithmeticContainment, Count
    };
    explicit AllocationProfile(std::shared_ptr<AnalysisLedger> owner) : ledger(std::move(owner)) {}
    ~AllocationProfile()
    {
        if (!enabled() || ledger->activeStages.empty()) {
            return;
        }
        const auto parentIndex = ledger->activeStages.back();
        const auto parent = ledger->stages[parentIndex];
        const char* names[] = {
            "summary",    "groups",        "queries", "proofs", "lanes", "serialization", "group-member-pairs",
            "lane-pairs", "boundary-pairs", "arithmetic-projection", "arithmetic-composition",
            "arithmetic-difference", "arithmetic-extrema", "arithmetic-order",
            "publication-qualification", "arithmetic-selector-export", "arithmetic-containment"};
        for (unsigned i = 0; i < Count; ++i) {
            const auto& bucket = buckets[i];
            if (!bucket.calls) {
                continue;
            }
            AnalysisStageRecord record;
            record.id = ledger->stages.size() + 1;
            record.parent = parent.id;
            record.request = parent.request;
            record.parentRequest = parent.request;
            record.region = parent.region;
            record.method = parent.method;
            record.stage = std::string("allocation-") + names[i] + "-aggregate";
            record.elapsedNanoseconds = record.exclusiveNanoseconds = bucket.nanoseconds;
            record.aggregateCalls = bucket.calls;
            record.aggregateCacheHits = bucket.hits;
            record.aggregateCacheMisses = bucket.misses;
            accumulateCost(ledger->stages[parentIndex].exclusiveNanoseconds, bucket.nanoseconds);
            ledger->stages.push_back(std::move(record));
        }
    }
    AllocationProfile(const AllocationProfile&) = delete;
    AllocationProfile& operator=(const AllocationProfile&) = delete;
    bool enabled() const { return ledger && ledger->profiling; }
    void recordWork(std::string method, std::vector<std::pair<std::string, uint64_t>> work)
    {
        if (!enabled() || ledger->activeStages.empty()) { return; }
        const auto parent = ledger->stages[ledger->activeStages.back()];
        AnalysisStageRecord record;
        record.id = ledger->stages.size() + 1;
        record.parent = parent.id;
        record.request = record.parentRequest = parent.request;
        record.region = parent.region;
        record.stage = "repeated-allocation-work";
        record.method = std::move(method);
        record.work = std::move(work);
        ledger->stages.push_back(std::move(record));
    }
    void visited(Stage stage)
    {
        if (enabled()) {
            accumulateCost(buckets[stage].calls, 1);
        }
    }
    class Scope {
    public:
        Scope(AllocationProfile* owner, Stage stage) : owner(owner), stage(stage)
        {
            if (!owner || !owner->enabled()) {
                this->owner = nullptr;
                return;
            }
            previous = owner->active;
            owner->active = this;
            accumulateCost(owner->buckets[stage].calls, 1);
            started = Clock::now();
        }
        ~Scope()
        {
            if (!owner) {
                return;
            }
            const auto elapsed = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
            accumulateCost(owner->buckets[stage].nanoseconds, elapsed - children);
            owner->active = previous;
            if (previous) {
                accumulateCost(previous->children, elapsed);
            }
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        void cacheHit()
        {
            if (owner) {
                accumulateCost(owner->buckets[stage].hits, 1);
            }
        }
        void constructed()
        {
            if (owner) {
                accumulateCost(owner->buckets[stage].misses, 1);
            }
        }

    private:
        using Clock = std::chrono::steady_clock;
        AllocationProfile* owner = nullptr;
        Stage stage;
        Scope* previous = nullptr;
        uint64_t children = 0;
        Clock::time_point started;
    };

private:
    struct Bucket {
        uint64_t nanoseconds = 0, calls = 0, hits = 0, misses = 0;
    };
    std::shared_ptr<AnalysisLedger> ledger;
    std::array<Bucket, Count> buckets{};
    Scope* active = nullptr;
};
} // namespace mlir::pto::frontiersynch
#endif
