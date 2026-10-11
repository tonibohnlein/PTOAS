// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Immutable atom-index backing scoped to one finalized numerical partition.
// Every requested range must be covered exactly by complete disjoint atoms.
#ifndef PTO_FRONTIERSYNCH_NUMERICTEMPLATEATOMREFERENCES_H
#define PTO_FRONTIERSYNCH_NUMERICTEMPLATEATOMREFERENCES_H
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplate.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch::detail {
class NumericAtomReferences {
public:
    using Backing = std::shared_ptr<const std::vector<std::size_t>>;
    using Charge = std::function<bool(uint64_t)>;
    using Attempt = std::function<void(uint64_t)>;
    using Cost = NumericAtomReferenceCost;
    NumericAtomReferences(ArrayRef<SyncStorageCell> partition, uint64_t maximumIds)
        : partition(partition), maximumIds(maximumIds)
    {
        for (auto [id, cell] : llvm::enumerate(partition)) { domains[domain(cell)].push_back(id); }
        for (auto& item : domains) {
            llvm::sort(item.second, [&](auto a, auto b) { return partition[a].begin < partition[b].begin; });
            uint64_t end = 0;
            for (auto id : item.second) {
                valid &= partition[id].begin < partition[id].end && end <= partition[id].begin;
                end = partition[id].end;
            }
        }
    }
    const Cost& cost() const { return work; }
    Backing get(ArrayRef<SyncStorageCell> ranges, const Charge& charge, const Attempt& attempted)
    {
        increment(work.requests);
        attempted(ranges.size());
        if (!valid) { return {}; }
        auto key = canonical(ranges);
        if (!key) { return {}; }
        auto found = cache.find(*key);
        if (found != cache.end()) {
            increment(work.hits);
            return charge(1) ? found->second : Backing{};
        }
        increment(work.expansions);
        auto values = expand(*key, attempted);
        if (!values) { return {}; }
        if (values->size() == UINT64_MAX || !charge(static_cast<uint64_t>(values->size()) + 1)) { return {}; }
        auto result = std::make_shared<const std::vector<std::size_t>>(std::move(*values));
        cache.emplace(std::move(*key), result);
        return result;
    }
private:
    static void increment(uint64_t& value) { if (value != UINT64_MAX) { ++value; } }
    using Domain = std::pair<unsigned, uintptr_t>;
    using Range = std::tuple<unsigned, uintptr_t, uint64_t, uint64_t>;
    using Key = std::vector<Range>;
    static Domain domain(const SyncStorageCell& cell)
    {
        return {static_cast<unsigned>(cell.space), reinterpret_cast<uintptr_t>(cell.base.getAsOpaquePointer())};
    }
    static std::optional<Key> canonical(ArrayRef<SyncStorageCell> ranges)
    {
        Key key;
        for (const auto& range : ranges) {
            if (range.begin > range.end) { return std::nullopt; }
            if (range.begin == range.end) { continue; }
            const auto d = domain(range);
            key.emplace_back(d.first, d.second, range.begin, range.end);
        }
        llvm::sort(key);
        Key merged;
        for (const auto& range : key) {
            if (!merged.empty() && std::get<0>(range) == std::get<0>(merged.back()) &&
                std::get<1>(range) == std::get<1>(merged.back()) &&
                std::get<2>(range) <= std::get<3>(merged.back())) {
                std::get<3>(merged.back()) = std::max(std::get<3>(merged.back()), std::get<3>(range));
            } else {
                merged.push_back(range);
            }
        }
        return merged;
    }
    std::optional<std::vector<std::size_t>> expand(const Key& key, const Attempt& attempted)
    {
        std::vector<std::size_t> result;
        for (const auto& range : key) {
            auto found = domains.find({std::get<0>(range), std::get<1>(range)});
            if (found == domains.end()) { return std::nullopt; }
            const auto& ids = found->second;
            auto first = std::lower_bound(ids.begin(), ids.end(), std::get<2>(range),
                [&](auto id, uint64_t point) { return partition[id].begin < point; });
            uint64_t expected = std::get<2>(range);
            for (; first != ids.end() && partition[*first].begin < std::get<3>(range); ++first) {
                if (partition[*first].begin != expected || partition[*first].end > std::get<3>(range)) {
                    return std::nullopt;
                }
                expected = partition[*first].end;
                attempted(1);
                if (result.size() >= maximumIds) { return std::nullopt; }
                increment(work.expandedIds);
                result.push_back(*first);
            }
            if (expected != std::get<3>(range)) { return std::nullopt; }
        }
        llvm::sort(result);
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
    ArrayRef<SyncStorageCell> partition;
    uint64_t maximumIds;
    Cost work;
    std::map<Domain, std::vector<std::size_t>> domains;
    std::map<Key, Backing> cache;
    bool valid = true;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
