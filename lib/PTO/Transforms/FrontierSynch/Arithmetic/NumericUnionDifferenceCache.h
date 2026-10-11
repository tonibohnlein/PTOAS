// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICUNIONDIFFERENCECACHE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICUNIONDIFFERENCECACHE_H
#include "mlir/Support/LogicalResult.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
// IDs name immutable numerical unions in one reducer context. Successful empty
// differences are ordinary cached unions; failed construction remains distinct.
class NumericUnionDifferenceCache {
public:
    struct Cost { uint64_t attempts = 0, hits = 0, builds = 0; };
    template<class Build>
    FailureOr<std::size_t> getOrBuild(std::size_t left, std::size_t right, unsigned dimensions, Build&& build)
    {
        increment(counts.attempts);
        const auto key = std::make_tuple(left, right, dimensions);
        if (auto found = differences.find(key); found != differences.end()) {
            increment(counts.hits); return found->second;
        }
        increment(counts.builds);
        auto result = build();
        if (failed(result)) { return failure(); }
        differences.emplace(key, *result); return *result;
    }
    const Cost& cost() const { return counts; }
    void clear() { differences.clear(); counts = {}; }
private:
    static void increment(uint64_t& value)
    {
        if (value != std::numeric_limits<uint64_t>::max()) { ++value; }
    }
    std::map<std::tuple<std::size_t, std::size_t, unsigned>, std::size_t> differences;
    Cost counts;
};
} // namespace mlir::pto::frontiersynch
#endif
