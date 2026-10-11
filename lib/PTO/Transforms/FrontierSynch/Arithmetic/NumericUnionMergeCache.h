// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICUNIONMERGECACHE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICUNIONMERGECACHE_H
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
namespace mlir::pto::frontiersynch {
// IDs name immutable numerical unions in one reducer context. Event tags are
// deliberately absent: attaching the same union to another event changes no
// containment fact. This cache never crosses contexts or owns borrowed IR.
class NumericUnionMergeCache {
public:
    struct Cost {
        uint64_t attempts = 0, hits = 0, builds = 0;
    };
    template<class Build>
    std::size_t getOrBuild(std::size_t left, std::size_t right, Build&& build)
    {
        increment(counts.attempts);
        const auto key = std::make_pair(left, right);
        if (auto found = merged.find(key); found != merged.end()) {
            increment(counts.hits);
            return found->second;
        }
        increment(counts.builds);
        const auto result = build();
        merged.emplace(key, result);
        return result;
    }
    const Cost& cost() const { return counts; }
    void clear() { merged.clear(); counts = {}; }
private:
    static void increment(uint64_t& value)
    {
        if (value != std::numeric_limits<uint64_t>::max()) { ++value; }
    }
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> merged;
    Cost counts;
};
} // namespace mlir::pto::frontiersynch
#endif
