// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Attempt-local observation only; no allocation decision depends on counters.
#ifndef PTO_FRONTIERSYNCH_REGIONALCOALESCINGPROFILE_H
#define PTO_FRONTIERSYNCH_REGIONALCOALESCINGPROFILE_H
#include "PTO/Transforms/FrontierSynch/Shared/AllocationProfile.h"
namespace mlir::pto::frontiersynch {
struct RegionalCoalescingProfile {
    AllocationProfile* profile;
    std::string method;
    uint64_t groups = 0, members = 0, memberPairs = 0, envelopeRequests = 0, envelopeAccepted = 0;
    uint64_t queryRequests = 0, queryHits = 0, queryConstructions = 0;
    uint64_t proofRequests = 0, proofHits = 0, envelopeQueryRequests = 0;
    uint64_t summaryNs = 0, envelopeNs = 0, queryNs = 0, proofNs = 0;
    using Clock = std::chrono::steady_clock;
    Clock::time_point started;
    RegionalCoalescingProfile(AllocationProfile* owner, std::string label)
        : profile(owner && owner->enabled() ? owner : nullptr), method(std::move(label))
    {
        if (profile) {
            started = Clock::now();
        }
    }
    ~RegionalCoalescingProfile()
    {
        if (!profile) {
            return;
        }
        profile->recordWork(
            "coalescer:" + method, {{"elapsed_ns", duration(started)},
                                    {"groups", groups},
                                    {"members", members},
                                    {"member_pair_visits", memberPairs},
                                    {"envelope_requests", envelopeRequests},
                                    {"envelope_accepted", envelopeAccepted},
                                    {"query_requests", queryRequests},
                                    {"query_cache_hits", queryHits},
                                    {"unique_query_constructions", queryConstructions},
                                    {"proof_requests", proofRequests},
                                    {"proof_cache_hits", proofHits},
                                    {"summary_construction_ns", summaryNs},
                                    {"envelope_qualification_ns", envelopeNs},
                                    {"envelope_query_requests", envelopeQueryRequests},
                                    {"query_construction_ns", queryNs},
                                    {"proof_ns", proofNs}});
    }
    static uint64_t duration(Clock::time_point start)
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }
    void visit(uint64_t& count) const
    {
        if (profile) {
            accumulateCost(count, 1);
        }
    }
    class Scope {
    public:
        Scope(RegionalCoalescingProfile& owner, uint64_t& destination)
            : enabled(owner.profile != nullptr), destination(destination)
        {
            if (enabled) {
                started = Clock::now();
            }
        }
        ~Scope()
        {
            if (enabled) {
                accumulateCost(destination, duration(started));
            }
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        bool enabled;
        uint64_t& destination;
        Clock::time_point started;
    };
};
} // namespace mlir::pto::frontiersynch
#endif
