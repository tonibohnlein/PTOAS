// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Structured/RotatingRegion.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalSelectorPartition.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <tuple>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Key = std::tuple<unsigned, uint32_t, uint64_t, std::vector<uint64_t>>;
std::optional<std::set<Key>> evaluate(const fs::RegionalAnalysis& region, uint64_t byte,
                                    fs::RegionExpressions::Substitution& values)
{
    auto& e = *region.expressions;
    auto selected = region.storageSelectors({pto::AddressSpace::LEFT, {}, e.constant(byte)});
    if (!selected) { return std::nullopt; }
    std::set<Key> result;
    auto append = [&](unsigned role, const auto& selectors) {
        for (const auto& selector : selectors) {
            auto present = e.constantValue(e.substitute(selector.present, values));
            if (!present) { return false; }
            if (!*present) { continue; }
            auto ordinal = e.constantValue(e.substitute(selector.event.ordinal, values));
            if (!ordinal) { return false; }
            std::vector<uint64_t> visits;
            for (auto coordinate : selector.event.visits) {
                auto value = e.constantValue(e.substitute(coordinate, values));
                if (!value) { return false; }
                visits.push_back(*value);
            }
            if (!result.emplace(role, selector.event.type, *ordinal, std::move(visits)).second) { return false; }
        }
        return true;
    };
    if (!append(0, selected->firstWriters) || !append(1, selected->lastWriters)) { return std::nullopt; }
    for (const auto& [pipe, readers] : selected->firstReaders) {
        if (!append(2 + 2 * pipe, readers)) { return std::nullopt; }
    }
    for (const auto& [pipe, readers] : selected->lastReaders) {
        if (!append(3 + 2 * pipe, readers)) { return std::nullopt; }
    }
    return result;
}
} // namespace
bool checkRotatingPartitions(const fs::RegionalAnalysis& region, func::FuncOp function)
{
    uint64_t work = 0;
    const pto::SyncStorageCell range{pto::AddressSpace::LEFT, 100, 1700};
    auto cuts = fs::uniformSelectorCuts(region, range, work);
    if (!cuts || *cuts != std::vector<uint64_t>({100, 512, 1024, 1536, 1700}) || work > 16) {
        llvm::errs() << "rotating partition lost bank boundaries or expanded physical bytes\n";
        return false;
    }
    auto& e = *region.expressions;
    for (uint64_t trips : {0, 1, 2, 4, 7}) {
        for (uint64_t inner : {0, 2}) {
            fs::RegionExpressions::Substitution values({
                {e.input(function.getArgument(0)), e.constant(trips)},
                {e.input(function.getArgument(1)), e.constant(inner)}});
            // Full supplied byte domain, not a representative-point proof.
            for (std::size_t atom = 1; atom < cuts->size(); ++atom) {
                auto expected = evaluate(region, (*cuts)[atom - 1], values);
                if (!expected) { return false; }
                for (auto byte = (*cuts)[atom - 1]; byte < (*cuts)[atom]; ++byte) {
                    if (evaluate(region, byte, values) != expected) { return false; }
                }
            }
        }
    }
    SmallVector<pto::SyncStorageCell> overlapping{
        {pto::AddressSpace::LEFT, 100, 512}, {pto::AddressSpace::LEFT, 512, 1024},
        {pto::AddressSpace::LEFT, 400, 512}, {pto::AddressSpace::LEFT, 512, 900}};
    if (!fs::normalizeSelectorSupport(overlapping) || overlapping.size() != 4) { return false; }
    const std::vector<uint64_t> expected{100, 400, 512, 900, 1024};
    for (std::size_t i = 0; i < overlapping.size(); ++i) {
        if (overlapping[i].begin != expected[i] || overlapping[i].end != expected[i + 1]) { return false; }
    }
    auto foreign = region;
    foreign.expressions = std::make_shared<fs::RegionExpressions>();
    auto alien = range;
    alien.base = function.getArgument(0);
    if (fs::uniformSelectorCuts(foreign, range, work) ||
        fs::selectorSupportRanges(foreign, work) ||
        fs::uniformSelectorCuts(region, alien, work) ||
        fs::uniformSelectorCuts(region, {pto::AddressSpace::LEFT, 512, 512}, work) ||
        fs::uniformSelectorCuts(region, {pto::AddressSpace::LEFT, 0, UINT64_MAX}, work)) { return false; }
    llvm::outs() << "rotating selector partitions: exhaustive bytes, empty visits, overlap and ownership passed\n";
    return true;
}
bool checkHugeRotatingPartition(const fs::RegionalAnalysis& region, uint64_t bank, uint64_t banks)
{
    uint64_t work = 0;
    const auto first = bank * 512;
    auto cuts = fs::uniformSelectorCuts(region, {pto::AddressSpace::LEFT, first, first + 512}, work);
    if (!cuts || *cuts != std::vector<uint64_t>({first, first + 512}) || work > 16) { return false; }
    auto support = fs::selectorSupportRanges(region, work);
    if (!support || work > 32) { return false; }
    for (const auto& range : *support) {
        if (range.space != pto::AddressSpace::LEFT) { continue; }
        if (range.begin != 0 || range.end != banks * 512) { return false; }
        // Complete materialization may exceed the existing interface limit;
        // refusing must not enumerate a trillion bank positions.
        return !fs::uniformSelectorCuts(region, range, work) && work <= 300;
    }
    return false;
}
bool checkGappedRotatingPartition(const fs::RegionalAnalysis& region, func::FuncOp function)
{
    uint64_t work = 0;
    auto cuts = fs::uniformSelectorCuts(region, {pto::AddressSpace::LEFT, 0, 6500}, work);
    const std::vector<uint64_t> expected{0, 512, 1024, 1536, 2048, 2560, 3072, 3584,
                                         4096, 4608, 5120, 5632, 6500};
    if (!cuts || *cuts != expected || work > 24) { return false; }
    auto& e = *region.expressions;
    for (uint64_t trips = 0; trips <= 8; ++trips) {
        fs::RegionExpressions::Substitution values({
            {e.input(function.getArgument(0)), e.constant(trips)},
            {e.input(function.getArgument(1)), e.constant(2)}});
        for (uint64_t bank = 0; bank < 6; ++bank) {
            auto selected = evaluate(region, bank * 1024, values);
            auto hole = evaluate(region, bank * 1024 + 512, values);
            if (!selected || !hole || !hole->empty()) { return false; }
            std::vector<uint64_t> owners;
            for (uint64_t visit = 0; visit < trips; ++visit) {
                if ((2 * visit) % 6 == bank) { owners.push_back(visit); }
            }
            std::set<Key> expectedSelectors;
            if (!owners.empty()) {
                expectedSelectors.emplace(0, 0, 0, std::vector<uint64_t>{owners.front()});
                expectedSelectors.emplace(1, 0, 0, std::vector<uint64_t>{owners.back()});
                const auto pipe = static_cast<uint32_t>(region.anchors[3].phase->kPipeValue);
                expectedSelectors.emplace(3 + 2 * pipe, 3, 1, std::vector<uint64_t>{owners.back()});
            }
            if (*selected != expectedSelectors) { return false; }
        }
    }
    return true;
}
