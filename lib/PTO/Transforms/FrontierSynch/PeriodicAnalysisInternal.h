// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private quotient construction and checked arithmetic.
#ifndef PTO_FRONTIERSYNCH_PERIODICANALYSISINTERNAL_H
#define PTO_FRONTIERSYNCH_PERIODICANALYSISINTERNAL_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include <limits>
namespace mlir::pto::frontiersynch::periodic {
struct Edge {
    uint32_t source = 0;
    uint32_t target = 0;
    uint64_t weight = 0;
};
struct Graph {
    std::vector<Edge> edges;
    std::vector<std::vector<uint32_t>> outgoing;
    std::vector<uint32_t> recordEdges;
};
inline bool add(uint64_t a, uint64_t b, uint64_t& result)
{
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}
inline bool multiply(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}
// floor((distance + localRank - 1) / count), without overflowing the sum.
inline uint64_t threshold(uint64_t distance, uint32_t localRank, uint32_t count)
{
    return distance / count + (distance % count + localRank - 1 >= count ? 1 : 0);
}
bool computeFrontiers(const Graph& graph, PeriodicAnalysis& output);
} // namespace mlir::pto::frontiersynch::periodic
#endif
