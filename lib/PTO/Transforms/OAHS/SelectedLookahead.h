// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_SELECTED_LOOKAHEAD_H
#define PTO_OAHS_SELECTED_LOOKAHEAD_H

#include <algorithm>
#include <cstddef>
#include <deque>
#include <map>
#include <utility>
#include <vector>

namespace mlir::pto::oahs::selected {

// Immutable original-program facts. Neither index contains selected events,
// completion receipts, inferred iteration identities, or executable guards.
class LookaheadIndex {
public:
    using Index = std::size_t;
    struct ClassIssue {
        Index frame, position, access;
    };

    bool build(const std::vector<std::vector<Index>>& successors,
               const std::vector<bool>& payload,
               const std::vector<ClassIssue>& issues)
    {
        ready = false;
        byClass.clear();
        futurePayload.clear();
        const auto size = successors.size();
        if (payload.size() != size) return false;
        std::vector<std::vector<Index>> predecessors(size);
        for (Index site = 0; site < size; ++site) {
            for (auto next : successors[site]) {
                if (next >= size) return false;
                predecessors[next].push_back(site);
            }
        }
        for (const auto& issue : issues) {
            if (issue.frame >= size || issue.position >= size) return false;
            byClass[{issue.frame, issue.access}].push_back(issue.position);
        }
        for (auto& entry : byClass) {
            auto& values = entry.second;
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
        // A STRICT future: initialize at predecessors, not at payload sites.
        // A backedge can nevertheless make a site its own future. Traverse the
        // original graph, not the construction-only acyclic loop summaries.
        futurePayload.assign(size, false);
        std::deque<Index> queue;
        auto add = [&](Index site) {
            if (!futurePayload[site]) {
                futurePayload[site] = true;
                queue.push_back(site);
            }
        };
        for (Index site = 0; site < size; ++site) {
            if (payload[site]) {
                for (auto before : predecessors[site]) add(before);
            }
        }
        while (!queue.empty()) {
            const auto site = queue.front();
            queue.pop_front();
            for (auto before : predecessors[site]) add(before);
        }
        ready = true;
        return true;
    }

    // The source cut is before its payload, whereas the consumer is excluded.
    // This is exactly the old [source, consumer) scan inside ONE proven frame.
    bool hasIssueBetween(Index frame, Index access, Index begin, Index end) const
    {
        if (!ready || frame >= futurePayload.size() || begin > end) return true;
        const auto found = byClass.find({frame, access});
        if (found == byClass.end()) return false;
        const auto& values = found->second;
        const auto at = std::lower_bound(values.begin(), values.end(), begin);
        return at != values.end() && *at < end;
    }

    bool mayIssueAfter(Index site) const
    {
        // An unavailable summary must never justify dropping an acknowledgment.
        return !ready || site >= futurePayload.size() || futurePayload[site];
    }


private:
    bool ready = false;
    std::map<std::pair<Index, Index>, std::vector<Index>> byClass;
    std::vector<bool> futurePayload;
};

} // namespace mlir::pto::oahs::selected
#endif
