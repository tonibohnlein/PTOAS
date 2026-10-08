// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Decode finite shared-pool evidence without changing the logical plan.
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/SharedHandoffAllocation.h"
#include "PTO/IR/PTO.h"
#include "llvm/ADT/DenseSet.h"
#include <set>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
std::optional<int64_t> number(DictionaryAttr d, StringRef key)
{
    auto a = d.getAs<IntegerAttr>(key);
    if (!a || !a.getType().isInteger(64)) { return std::nullopt; }
    return a.getInt();
}
bool pipe(int64_t p) { return p >= 0 && p <= int64_t(PIPE::PIPE_FIX) && p != int64_t(PIPE::PIPE_ALL); }
}
FailureOr<PhysicalAllocationPlan> decodeFiniteAllocation(
    func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds)
{
    std::set<int64_t> eligible(eligibleIds.begin(), eligibleIds.end());
    if (eligibleIds.size() > 6 || eligible.size() != eligibleIds.size() ||
        llvm::any_of(eligibleIds, [](int64_t id) { return id < 0 || id >= 6; })) {
        return function.emitError("invalid finite shared event-ID pool"), failure();
    }
    auto version = number(certificate, "version"), plan = number(certificate, "plan");
    auto mode = certificate.getAs<StringAttr>("mode");
    auto entries = certificate.getAs<ArrayAttr>("handoffs");
    if (!version || *version != 2 || !plan || *plan < 0 || !mode || !entries ||
        (mode.getValue() != "reuse-order" && mode.getValue() != "compatibility" &&
         mode.getValue() != "rank-query") || entries.size() > UINT32_MAX) {
        return function.emitError("malformed finite shared-pool allocation certificate"), failure();
    }
    const bool exact = mode.getValue() == "reuse-order", ranked = mode.getValue() == "rank-query";
    std::map<uint32_t, uint32_t> columns;
    if (ranked) {
        auto labels = certificate.getAs<DenseI64ArrayAttr>("pipe_labels");
        if (!labels) { return function.emitError("finite rank query lacks pipe labels"), failure(); }
        for (auto label : labels.asArrayRef()) {
            if (!pipe(label) || !columns.emplace(label, columns.size()).second) {
                return function.emitError("invalid finite rank-query pipe labels"), failure();
            }
        }
    }
    struct Rank { uint32_t source, target, targetRank, targetColumn; };
    std::vector<Rank> ranks;
    std::vector<SharedHandoff> handoffs;
    std::vector<std::vector<uint32_t>> evidence;
    SmallVector<int64_t> records;
    llvm::DenseSet<int64_t> seen;
    for (auto raw : entries) {
        auto d = dyn_cast<DictionaryAttr>(raw);
        if (!d) { return function.emitError("malformed finite shared handoff"), failure(); }
        auto p = number(d, "source"), q = number(d, "target"), r = number(d, "record");
        auto row = d.getAs<DenseI64ArrayAttr>("evidence");
        if (!p || !q || !r || !pipe(*p) || !pipe(*q) || *p == *q || *r < 0 || !row || !seen.insert(*r).second) {
            return function.emitError("invalid finite shared handoff"), failure();
        }
        std::vector<uint32_t> values;
        int64_t previous = -1;
        for (auto x : row.asArrayRef()) {
            if (ranked ? (x < 0 || uint64_t(x) > UINT32_MAX) :
                (x <= previous || x < 0 || uint64_t(x) >= (exact ? entries.size() : handoffs.size()))) {
                return function.emitError("invalid finite shared handoff evidence"), failure();
            }
            values.push_back(x); previous = x;
        }
        if (ranked) {
            auto source = number(d, "source_position"), target = number(d, "target_position");
            auto targetRank = number(d, "target_rank");
            if (!source || !target || !targetRank || *source < 0 || *source >= *target ||
                uint64_t(*target) >= UINT32_MAX || *targetRank <= 0 || *targetRank > *target + 1 ||
                (!ranks.empty() && uint64_t(*source) < ranks.back().source) ||
                !columns.count(*p) || !columns.count(*q) || values.size() != columns.size() ||
                llvm::any_of(values, [&](uint32_t rank) { return rank > uint64_t(*source); })) {
                return function.emitError("invalid finite handoff rank profile"), failure();
            }
            ranks.push_back({uint32_t(*source), uint32_t(*target), uint32_t(*targetRank), columns.at(*q)});
        }
        handoffs.push_back({uint32_t(*p), uint32_t(*q)});
        records.push_back(*r); evidence.push_back(std::move(values));
    }
    SharedHandoffAllocation allocation;
    if (ranked) {
        allocation = allocateSharedHandoffsByQuery(handoffs, [&](uint32_t from, uint32_t to) {
            const auto& a = ranks[from]; const auto& b = ranks[to];
            return handoffs[from].targetPipe == handoffs[to].sourcePipe ? a.target <= b.source :
                evidence[to][a.targetColumn] >= a.targetRank;
        }, eligibleIds.size());
        if (!allocation.error.empty()) { return function.emitError(allocation.error), failure(); }
    } else if (exact) {
        allocation = allocateSharedHandoffs(handoffs, evidence, eligibleIds.size());
        if (!allocation.error.empty()) { return function.emitError(allocation.error), failure(); }
    } else {
        // Mutually exclusive guards can permit sharing without one uniform
        // orientation. This is sufficient coloring, not a minimum-width claim.
        for (const auto& row : evidence) {
            SmallVector<bool> used(eligibleIds.size(), false);
            for (auto prior : row) { used[allocation.lanes[prior]] = true; }
            uint32_t color = 0;
            while (color < used.size() && used[color]) { ++color; }
            if (color == used.size()) {
                return function.emitError("finite guarded assignment not certified within shared capacity; "
                    "no minimum-capacity claim; scarcity repair not implemented yet"), failure();
            }
            allocation.lanes.push_back(color);
            allocation.budget = std::max(allocation.budget, uint64_t(color) + 1);
        }
    }
    if (allocation.budget > eligibleIds.size()) {
        return function.emitError("fixed finite handoff plan requires ") << allocation.budget <<
            " shared event IDs, only " << eligibleIds.size() <<
            " available; scarcity repair not implemented yet", failure();
    }
    PhysicalAllocationPlan result;
    result.planId = *plan;
    for (std::size_t i = 0; i < records.size(); ++i) {
        PhysicalRecordAllocation item{records[i], handoffs[i].sourcePipe, handoffs[i].targetPipe, 0, 0, {}};
        item.ids.push_back(eligibleIds[allocation.lanes[i]]);
        result.records.push_back(std::move(item));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
