// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compact sorted cell memberships. Iteration expands IDs only at the consumer.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCCELLMEMBERSHIP_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCCELLMEMBERSHIP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <iterator>
#include <utility>

namespace mlir::pto {
class SyncStorageEffects;
class SyncCellMembership {
public:
    using Interval = std::pair<std::size_t, std::size_t>;
    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = std::size_t;
        const_iterator() = default;
        std::size_t operator*() const { return cell; }
        const_iterator& operator++()
        {
            ++cell;
            if (cell == owner->runs[run].second) {
                ++run;
                cell = run == owner->runs.size() ? 0 : owner->runs[run].first;
            }
            return *this;
        }
        const_iterator operator++(int) { auto old = *this; ++*this; return old; }
        bool operator==(const const_iterator& other) const
        {
            return owner == other.owner && run == other.run && cell == other.cell;
        }
        bool operator!=(const const_iterator& other) const { return !(*this == other); }
    private:
        friend class SyncCellMembership;
        const_iterator(const SyncCellMembership* owner, std::size_t run)
            : owner(owner), run(run), cell(run == owner->runs.size() ? 0 : owner->runs[run].first) {}
        const SyncCellMembership* owner = nullptr;
        std::size_t run = 0, cell = 0;
    };
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, runs.size()); }
    std::size_t size() const { return count; }
    bool empty() const { return runs.empty(); }
    llvm::ArrayRef<Interval> intervals() const { return runs; }
    void clear() { runs.clear(); count = 0; }
    // Both memberships must refer to the same immutable cell partition.
    bool overlaps(const SyncCellMembership& other) const
    {
        std::size_t left = 0, right = 0;
        while (left < runs.size() && right < other.runs.size()) {
            const auto& a = runs[left]; const auto& b = other.runs[right];
            if (a.first < b.second && b.first < a.second) { return true; }
            if (a.second <= b.first) { ++left; } else { ++right; }
        }
        return false;
    }
private:
    friend class SyncStorageEffects;
    // The partition sweep appends disjoint sorted runs. Their total length is
    // bounded by the partition size, even when adjacent domains are coalesced.
    void append(std::size_t begin, std::size_t end)
    {
        assert(begin <= end && (runs.empty() || runs.back().second <= begin));
        if (begin == end) { return; }
        count += end - begin;
        if (!runs.empty() && runs.back().second == begin) { runs.back().second = end; }
        else { runs.push_back({begin, end}); }
    }
    llvm::SmallVector<Interval, 1> runs;
    std::size_t count = 0;
};
} // namespace mlir::pto
#endif
