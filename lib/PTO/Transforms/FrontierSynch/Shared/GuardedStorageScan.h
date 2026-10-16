// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Writer-centered traversal of an ordered cell access list. Callback state
// contains only accesses strictly between the supplied endpoint positions.
#ifndef PTO_FRONTIERSYNCH_GUARDED_STORAGE_SCAN_H
#define PTO_FRONTIERSYNCH_GUARDED_STORAGE_SCAN_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include <cstddef>
#include <map>
#include <vector>
namespace mlir::pto::frontiersynch::detail {
struct GuardedStorageScanWork {
    uint64_t sourceVisits = 0, pairVisits = 0, candidateVisits = 0;
};
template <typename State, typename Visit, typename Advance>
void scanWriterPairs(const std::vector<bool>& writers, const State& initial,
                     Visit visit, Advance advance, GuardedStorageScanWork* work = nullptr)
{
    for (std::size_t writer = 0; writer < writers.size(); ++writer) {
        if (work) { accumulateCost(work->sourceVisits, 1); }
        if (!writers[writer]) { continue; }
        auto forward = initial;
        for (std::size_t other = writer + 1; other < writers.size(); ++other) {
            if (work) { accumulateCost(work->pairVisits, 1); accumulateCost(work->candidateVisits, 1); }
            visit(writer, other, forward);
            advance(forward, other);
        }
        auto backward = initial;
        for (std::size_t end = writer; end; --end) {
            const auto other = end - 1;
            // Writer/writer pairs were already visited in the forward scan.
            if (work) { accumulateCost(work->pairVisits, 1); }
            if (!writers[other]) {
                if (work) { accumulateCost(work->candidateVisits, 1); }
                visit(other, writer, backward);
            }
            advance(backward, other);
        }
    }
}
struct GuardedStorageMode {
    uint32_t pipe;
    RegionExpressions::Id read, write;
};
// Every simultaneous envelope observes the same incoming state. The caller
// supplies an ordered list of original-occurrence groups, in either direction.
class GuardedBoundaryScan {
public:
    explicit GuardedBoundaryScan(RegionExpressions& expressions)
        : e(expressions), noWriter(e.boolean(true)) {}
    template <typename Range, typename Mode, typename Emit>
    void group(const Range& accesses, Mode mode, Emit emit)
    {
        auto writes = e.boolean(false);
        std::map<uint32_t, RegionExpressions::Id> reads;
        for (const auto& access : accesses) {
            const auto value = mode(access);
            auto unseen = noReader.try_emplace(value.pipe, e.boolean(true)).first->second;
            emit(access, e.land(noWriter, value.write),
                 e.land(noWriter, e.land(unseen, e.land(value.read, e.lnot(value.write)))));
            auto accumulated = reads.try_emplace(value.pipe, e.boolean(false)).first;
            accumulated->second = e.lor(accumulated->second, value.read);
            writes = e.lor(writes, value.write);
        }
        noWriter = e.land(noWriter, e.lnot(writes));
        for (auto [pipe, read] : reads) { noReader[pipe] = e.land(noReader[pipe], e.lnot(read)); }
    }
private:
    RegionExpressions& e;
    RegionExpressions::Id noWriter;
    std::map<uint32_t, RegionExpressions::Id> noReader;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
