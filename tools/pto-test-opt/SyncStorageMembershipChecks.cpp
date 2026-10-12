// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent byte-intersection oracle for compact cell memberships.
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>
#include <set>
#include <tuple>

using namespace mlir;
namespace {
bool checkRangeUnions()
{
    using namespace mlir::pto;
    SmallVector<SyncStorageEffect, 0> effects(5);
    effects[0].ranges = {{AddressSpace::VEC, 0, 12}, {AddressSpace::VEC, 0, 12},
        {AddressSpace::VEC, 2, 6}, {AddressSpace::VEC, 12, 16}, {AddressSpace::VEC, 20, 24}};
    effects[1].ranges = {{AddressSpace::VEC, 16, 20}, {AddressSpace::MAT, 0, 12}};
    effects[2].ranges = {{AddressSpace::VEC, 4, 8}, {AddressSpace::VEC, 22, 26}};
    effects[3].ranges = {{AddressSpace::VEC, 8, 8}, {AddressSpace::VEC, 10, 4}};
    effects[4].ranges = {{AddressSpace::VEC, UINT64_MAX - 1, UINT64_MAX}};
    auto check = [&](ArrayRef<SyncStorageCell> cells) {
        const std::vector<uint64_t> points{0, 2, 4, 6, 8, 12, 16, 20, 22, 24, 26};
        std::set<std::tuple<AddressSpace, uint64_t, uint64_t>> geometry;
        for (std::size_t i = 1; i < points.size(); ++i) {
            geometry.emplace(AddressSpace::VEC, points[i - 1], points[i]);
        }
        geometry.emplace(AddressSpace::VEC, UINT64_MAX - 1, UINT64_MAX);
        geometry.emplace(AddressSpace::MAT, 0, 12);
        for (const auto& cell : cells) {
            if (cell.base || !geometry.erase({cell.space, cell.begin, cell.end})) { return false; }
        }
        if (!geometry.empty()) { return false; }
        std::vector<std::vector<std::size_t>> expected(effects.size());
        for (auto [index, effect] : llvm::enumerate(effects)) {
            for (auto [id, cell] : llvm::enumerate(cells)) {
                if (llvm::any_of(effect.ranges, [&, cell = cell](const auto& range) {
                    return range.begin < range.end && sameStorageDomain(range, cell) &&
                           range.begin < cell.end && cell.begin < range.end;
                })) { expected[index].push_back(id); }
            }
            std::vector<std::size_t> actual(effect.cells.begin(), effect.cells.end());
            if (actual != expected[index] || effect.cells.size() != actual.size()) { return false; }
        }
        for (std::size_t a = 0; a < effects.size(); ++a) {
            for (std::size_t b = 0; b < effects.size(); ++b) {
                bool overlap = llvm::any_of(expected[a], [&](auto cell) {
                    return llvm::is_contained(expected[b], cell);
                });
                if (effects[a].cells.overlaps(effects[b].cells) != overlap) { return false; }
            }
        }
        return effects[0].cells.intervals().size() == 2 && effects[3].cells.empty();
    };
    auto cells = SyncStorageEffects::partitionRanges(effects);
    if (!check(cells)) { return false; }
    auto rebuilt = SyncStorageEffects::partitionRanges(effects);
    return cells.size() == rebuilt.size() && check(rebuilt);
}
} // namespace
LogicalResult checkSyncStorageMemberships(func::FuncOp function, const pto::SyncInput& input)
{
    if (!checkRangeUnions()) { return function.emitError("range-union membership oracle failed"); }
    const auto& storage = input.accesses();
    std::size_t runs = 0, incidences = 0;
    std::vector<std::size_t> previous;
    const pto::SyncCellMembership* prior = nullptr;
    for (const auto& effect : storage.effects()) {
        std::vector<std::size_t> expected;
        for (auto [id, cell] : llvm::enumerate(storage.cells())) {
            const bool intersects = llvm::any_of(effect.ranges, [&, cell = cell](const auto& range) {
                return pto::sameStorageDomain(range, cell) && range.begin < cell.end && cell.begin < range.end;
            });
            if (intersects) { expected.push_back(id); }
        }
        std::vector<std::size_t> actual(effect.cells.begin(), effect.cells.end());
        if (actual != expected || actual.size() != effect.cells.size() ||
            effect.cells.empty() != expected.empty() || effect.cells.intervals().size() > effect.ranges.size()) {
            return function.emitError("compact memberships disagree with independent byte intersections");
        }
        std::vector<std::size_t> intersection;
        std::set_intersection(previous.begin(), previous.end(), expected.begin(), expected.end(),
                              std::back_inserter(intersection));
        if (prior && prior->overlaps(effect.cells) != !intersection.empty()) {
            return function.emitError("compact overlap disagrees with explicit membership intersection");
        }
        if (effect.cells.overlaps(effect.cells) != !expected.empty()) { return failure(); }
        previous = std::move(expected); prior = &effect.cells;
        runs += effect.cells.intervals().size(); incidences += effect.cells.size();
    }
    if (auto wanted = function->getAttrOfType<DenseI64ArrayAttr>("test.membership_expect")) {
        std::vector<int64_t> actual{static_cast<int64_t>(storage.cells().size()),
            static_cast<int64_t>(storage.effects().size()), static_cast<int64_t>(runs),
            static_cast<int64_t>(incidences)};
        if (wanted.asArrayRef() != ArrayRef<int64_t>(actual)) {
            return function.emitError("compact membership scaling counts differ");
        }
        pto::SyncStorageEffects rebuilt;
        if (failed(rebuilt.build(input)) || rebuilt.effects().size() != storage.effects().size()) { return failure(); }
        for (auto [index, effect] : llvm::enumerate(rebuilt.effects())) {
            if (effect.cells.intervals() != storage.effects()[index].cells.intervals()) { return failure(); }
        }
        llvm::outs() << "memberships " << function.getSymName() << ": cells=" << storage.cells().size()
                     << " effects=" << storage.effects().size() << " runs=" << runs
                     << " incidences=" << incidences << "\n";
    }
    return success();
}
