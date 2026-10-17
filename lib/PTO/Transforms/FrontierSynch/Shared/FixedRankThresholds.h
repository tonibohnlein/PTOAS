// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Reducer-local projection of constant thresholds through certified total rank
// choices and masked maxima. No symbolic/symbolic comparison distribution.
#ifndef PTO_FRONTIER_FIXEDRANKTHRESHOLDS_H
#define PTO_FRONTIER_FIXEDRANKTHRESHOLDS_H
#include "PTO/Transforms/FrontierSynch/Shared/GuardedRanks.h"
#include <map>
namespace mlir::pto::frontiersynch::detail {
class FixedRankThresholds {
    using Id = RegionExpressions::Id;
    struct Node {
        Id guard, yes, no;
        bool maximum;
        uint64_t lower, upper;
    };
    RegionExpressions& arena;
    GuardedRankProjectionWork& work;
    std::map<Id, Node> nodes;
    std::map<std::pair<Id, uint64_t>, Id> memo;
    std::pair<uint64_t, uint64_t> range(Id value) const
    {
        if (auto constant = arena.constantValue(value)) {
            return {*constant, *constant};
        }
        auto found = nodes.find(value);
        return found == nodes.end() ? std::make_pair(uint64_t(0), UINT64_MAX) :
                                      std::make_pair(found->second.lower, found->second.upper);
    }
    Id choose(Id guard, Id yes, Id no)
    {
        // Dominating inputs may still be poison. Keep the original lazy arm
        // semantics; total-on-defined-inputs is not a poison-freedom certificate.
        return arena.select(guard, yes, no);
    }

public:
    FixedRankThresholds(RegionExpressions& arena, GuardedRankProjectionWork& work) : arena(arena), work(work) {}
    void record(Id value, Id guard, Id yes, Id no, bool maximum = false)
    {
        if (value == yes || value == no || arena.constantValue(value)) {
            return;
        }
        // Reduced numeric nodes can alias an earlier interned node. Keep the
        // first equivalent description, whose dependencies already precede it.
        if (nodes.count(value) || value <= guard || value <= yes || value <= no) {
            return;
        }
        auto a = range(yes), b = range(no);
        nodes.emplace(
            value,
            Node{
                guard, yes, no, maximum, maximum ? b.first : std::min(a.first, b.first), std::max(a.second, b.second)});
        ++work.descriptions;
    }
    Id below(Id value, uint64_t threshold)
    {
        ++work.requests;
        const auto key = std::make_pair(value, threshold);
        if (auto found = memo.find(key); found != memo.end()) {
            ++work.hits;
            return found->second;
        }
        SmallVector<Id> pending{value};
        while (!pending.empty()) {
            const auto current = pending.back();
            const auto currentKey = std::make_pair(current, threshold);
            if (memo.count(currentKey)) {
                pending.pop_back();
                continue;
            }
            // Charge failed attempts as well as successful entries. The total
            // projection traversal stays linear in retained numeric decisions
            // plus requests, even with independently changing thresholds. This
            // is a representation budget, never an analysis admission limit:
            // retain the exact incumbent comparison when it is exhausted.
            const uint64_t size = nodes.size() + work.requests + 1;
            const uint64_t budget = size > UINT64_MAX / 8 ? UINT64_MAX : 8 * size;
            if (work.visits >= budget) {
                memo.emplace(key, arena.lt(value, arena.constant(threshold)));
                break;
            }
            ++work.visits;
            auto bounds = range(current);
            if (threshold <= bounds.first || threshold > bounds.second) {
                memo.emplace(currentKey, arena.boolean(threshold > bounds.second));
                ++work.rangeFolds;
                pending.pop_back();
                continue;
            }
            auto found = nodes.find(current);
            if (found == nodes.end()) {
                memo.emplace(currentKey, arena.lt(current, arena.constant(threshold)));
                pending.pop_back();
                continue;
            }
            const auto node = found->second;
            bool ready = true;
            for (auto operand : {node.yes, node.no}) {
                if (!memo.count({operand, threshold})) {
                    pending.push_back(operand);
                    ready = false;
                }
            }
            if (!ready) {
                continue;
            }
            auto yes = memo.at({node.yes, threshold}), no = memo.at({node.no, threshold});
            // max(old, active ? incoming : old) < K iff old<K and
            // (active implies incoming<K). The original guards are total.
            auto result = node.maximum ? arena.land(no, choose(node.guard, yes, arena.boolean(true))) :
                                         choose(node.guard, yes, no);
            memo.emplace(currentKey, result);
            pending.pop_back();
        }
        work.entries = memo.size();
        return memo.at(key);
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
