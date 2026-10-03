// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared fixed-period native event chains and pipe/rank metadata.
#ifndef PTO_FRONTIERSYNCH_PERIODICNATIVEGRAPH_H
#define PTO_FRONTIERSYNCH_PERIODICNATIVEGRAPH_H
#include "PeriodicShortestPaths.h"
namespace mlir::pto::frontiersynch::detail {
using llvm::DynamicAPInt;
struct NativeMetadata {
    SmallVector<PipelineType> pipes;
    SmallVector<std::size_t> sitePipes;
    SmallVector<std::size_t> siteRanks;
    SmallVector<std::size_t> counts;
};
inline NativeMetadata addNative(QuotientGraph& graph, ArrayRef<const CompoundInstanceElement*> sites)
{
    NativeMetadata result;
    DenseMap<PipelineType, std::size_t> pipeIds;
    SmallVector<std::size_t> first, previous;
    for (auto [site, phase] : llvm::enumerate(sites)) {
        graph.add(2 * site, 2 * site + 1, DynamicAPInt(0));
        auto [entry, added] = pipeIds.try_emplace(phase->kPipeValue, result.pipes.size());
        const auto pipe = entry->second;
        if (added) {
            result.pipes.push_back(phase->kPipeValue);
            result.counts.push_back(0);
            first.push_back(site);
            previous.push_back(site);
        } else {
            graph.add(2 * previous[pipe], 2 * site, DynamicAPInt(0));
            graph.add(2 * previous[pipe] + 1, 2 * site + 1, DynamicAPInt(0));
        }
        previous[pipe] = site;
        result.sitePipes.push_back(pipe);
        result.siteRanks.push_back(++result.counts[pipe]);
    }
    for (std::size_t pipe = 0; pipe < first.size(); ++pipe) {
        graph.add(2 * previous[pipe], 2 * first[pipe], DynamicAPInt(1));
        graph.add(2 * previous[pipe] + 1, 2 * first[pipe] + 1, DynamicAPInt(1));
    }
    return result;
}

} // namespace mlir::pto::frontiersynch::detail
#endif
