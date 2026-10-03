// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exhaustive partition/profile oracle, independent of the ordered-line engine.
#include "PTO/Transforms/FrontierSynch/OneWayRepair.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>
namespace {
using Int = llvm::DynamicAPInt;
namespace fs = mlir::pto::frontiersynch;
struct Reference {
    std::optional<Int> excess;
    std::vector<std::size_t> cuts;
};
Int blockCost(llvm::ArrayRef<fs::OneWayCover> covers, const Int& nQ,
              std::size_t first, std::size_t last)
{
    Int result(0);
    for (std::size_t i = first; i <= last; ++i) {
        const auto next = i + 1 == covers.size() ? nQ + Int(1) : covers[i + 1].consumerRank;
        result += (next - covers[i].consumerRank) * (covers[last].sourceRank - covers[i].sourceRank);
    }
    return result;
}
Reference exhaustive(llvm::ArrayRef<fs::OneWayCover> covers, const Int& nQ, std::size_t blocks)
{
    Reference best;
    std::vector<std::size_t> cuts;
    std::function<void(std::size_t, std::size_t, Int)> visit =
        [&](std::size_t first, std::size_t left, Int cost) {
            if (left == 1) {
                cost += blockCost(covers, nQ, first, covers.size() - 1);
                const bool earlier = std::lexicographical_compare(cuts.rbegin(), cuts.rend(),
                    best.cuts.rbegin(), best.cuts.rend());
                if (!best.excess || cost < *best.excess || (cost == *best.excess && earlier)) {
                    best.excess = cost; best.cuts = cuts;
                }
                return;
            }
            for (std::size_t next = first + 1; next <= covers.size() - left + 1; ++next) {
                cuts.push_back(next);
                visit(next, left - 1, cost + blockCost(covers, nQ, first, next - 1));
                cuts.pop_back();
            }
        };
    visit(0, blocks, Int(0));
    return best;
}
bool checkProfile(llvm::ArrayRef<fs::OneWayCover> covers, const fs::OneWayRepair& actual, int64_t nQ)
{
    // Count prerequisites directly at every actual consumer, including gaps
    // and the suffix. No production DP weights, prefixes or envelope queries.
    Int excess(0);
    for (int64_t consumer = 1; consumer <= nQ; ++consumer) {
        Int required(0), acquired(0);
        for (const auto& cover : covers) {
            if (cover.consumerRank <= Int(consumer)) { required = cover.sourceRank; }
        }
        for (const auto& block : actual.blocks) {
            if (covers[block.first].consumerRank <= Int(consumer)) {
                acquired = covers[block.last].sourceRank;
            }
        }
        if (acquired < required) { return false; }
        excess += acquired - required;
    }
    return excess == actual.excess;
}
bool partitionCases(unsigned& cases)
{
    for (unsigned pattern = 0; pattern < 6; ++pattern) {
        for (unsigned m = 1; m <= 8; ++m) {
            llvm::SmallVector<fs::OneWayCover> covers;
            int64_t source = 0, consumer = 0;
            for (unsigned i = 0; i < m; ++i) {
                source += 1 + ((i * 3 + pattern) % (pattern + 1));
                consumer += 1 + ((i * 5 + pattern) % (pattern + 1));
                covers.push_back({Int(source), Int(consumer)});
            }
            const auto nP = source + 3, nQ = consumer + pattern;
            for (unsigned capacity = 1; capacity <= m + 1; ++capacity) {
                llvm::SmallVector<unsigned> eligible;
                for (unsigned i = 0; i < capacity; ++i) { eligible.push_back(2 + 3 * i); }
                auto actual = fs::repairOneWay(covers, Int(nP), Int(nQ), eligible);
                const auto t = std::min(capacity, m);
                auto reference = exhaustive(covers, Int(nQ), t);
                if (actual.status != fs::OneWayRepairStatus::Ready || actual.blocks.size() != t ||
                    !reference.excess || actual.excess != *reference.excess ||
                    !checkProfile(covers, actual, nQ)) { return false; }
                std::vector<std::size_t> cuts;
                std::size_t next = 0;
                for (auto [index, block] : llvm::enumerate(actual.blocks)) {
                    if (block.first != next || block.first > block.last || block.last >= covers.size() ||
                        block.id != eligible[index]) { return false; }
                    if (index) { cuts.push_back(block.first); }
                    next = block.last + 1;
                }
                if (next != covers.size() || cuts != reference.cuts) { return false; }
                ++cases;
            }
        }
    }
    return true;
}
bool boundaries()
{
    using Status = fs::OneWayRepairStatus;
    if (fs::repairOneWay({}, Int(0), Int(0), {}).status != Status::Ready ||
        fs::repairOneWay({}, Int(-1), Int(0), {}).status != Status::Unsupported ||
        fs::repairOneWay({{Int(1), Int(1)}}, Int(1), Int(1), {}).status != Status::NoCapacity ||
        fs::repairOneWay({{Int(1), Int(1)}}, Int(1), Int(1), {2, 2}).status != Status::Unsupported ||
        fs::repairOneWay({{Int(0), Int(1)}}, Int(1), Int(1), {2}).status != Status::Unsupported ||
        fs::repairOneWay({{Int(1), Int(2)}}, Int(1), Int(1), {2}).status != Status::Unsupported ||
        fs::repairOneWay({{Int(1), Int(1)}, {Int(1), Int(2)}}, Int(2), Int(2), {2}).status !=
            Status::Unsupported ||
        fs::repairOneWay({{Int(1), Int(2)}, {Int(2), Int(1)}}, Int(2), Int(2), {2}).status !=
            Status::Unsupported) { return false; }
    llvm::SmallVector<fs::OneWayCover> diagonal;
    llvm::SmallVector<unsigned> eligible;
    for (int64_t i = 1; i <= 100; ++i) { diagonal.push_back({Int(i), Int(i)}); }
    for (unsigned i = 0; i < 10; ++i) { eligible.push_back(i); }
    auto balanced = fs::repairOneWay(diagonal, Int(100), Int(100), eligible);
    if (balanced.status != Status::Ready || balanced.excess != Int(450) || balanced.blocks.size() != 10) {
        return false;
    }
    for (const auto& block : balanced.blocks) {
        if (block.last - block.first + 1 != 10) { return false; }
    }
    Int huge(1);
    for (unsigned bit = 0; bit < 200; ++bit) { huge *= Int(2); }
    llvm::SmallVector<fs::OneWayCover> encoded{{Int(1), Int(1)}, {huge, huge}};
    auto single = fs::repairOneWay(encoded, huge, huge + huge, {4});
    auto two = fs::repairOneWay(encoded, huge, huge + huge, {2, 4});
    return single.status == Status::Ready && single.excess == (huge - Int(1)) * (huge - Int(1)) &&
           single.blocks.size() == 1 && single.blocks[0].id == 4 &&
           two.status == Status::Ready && two.excess == Int(0) && two.blocks.size() == 2 &&
           two.blocks[0].id == 2 && two.blocks[1].id == 4;
}
} // namespace
int runOneWayRepairChecks(llvm::StringRef, mlir::MLIRContext&)
{
    unsigned cases = 0;
    if (!partitionCases(cases) || !boundaries()) {
        llvm::errs() << "one-way minimum-excess partition/profile oracle failed\n"; return 1;
    }
    llvm::outs() << "one-way exhaustive partitions, profiles and deterministic ties: " << cases
                 << " cases; boundary/holes/100-cover/huge-rank checks passed\n";
    return 0;
}
