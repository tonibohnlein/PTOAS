// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent checks of shared-model extraction; no producer-equivalence claim.
#include "PTO/Transforms/FrontierSynch/CompactWriterReaderInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
bool checkBindings(scf::ForOp loop, const pto::SyncInput& input, const fs::PhaseIndex& index,
                   const fs::CompactWriterReaderInput& baseline)
{
    fs::CompactWriterReaderBindings bindings;
    bindings.accessBounds = [](std::size_t, std::size_t, fs::StorageHazard) {
        return std::optional<fs::OriginDistanceInterval>{{true, 4, 7}};
    };
    auto restricted = fs::buildCompactWriterReaderInput(loop, input, index, bindings);
    if (!restricted.error.empty() || restricted.queries.size() != baseline.queries.size()) { return false; }
    for (const auto& query : restricted.queries) {
        if (!query.bounds.reachable || query.bounds.minimum != 4 || query.bounds.maximum != 7) { return false; }
    }
    bindings.accessBounds = [](std::size_t, std::size_t, fs::StorageHazard) {
        return std::optional<fs::OriginDistanceInterval>{};
    };
    auto unknown = fs::buildCompactWriterReaderInput(loop, input, index, bindings);
    if (!unknown.error.empty() || unknown.queries.size() != baseline.queries.size()) { return false; }
    bindings.accessBounds = [](std::size_t, std::size_t, fs::StorageHazard) {
        return std::optional<fs::OriginDistanceInterval>{{false, 0, std::nullopt}};
    };
    auto empty = fs::buildCompactWriterReaderInput(loop, input, index, bindings);
    if (!empty.error.empty() || !empty.queries.empty() || empty.additional.size() != baseline.additional.size() ||
        empty.native.size() != baseline.native.size()) { return false; }
    bindings.accessBounds = [](std::size_t, std::size_t, fs::StorageHazard) {
        return std::optional<fs::OriginDistanceInterval>{{true, 7, 4}};
    };
    auto malformed = fs::buildCompactWriterReaderInput(loop, input, index, bindings);
    if (!baseline.queries.empty() && malformed.issue != fs::CompactInputIssue::InvalidBinding) { return false; }
    bindings.accessBounds = {};
    bindings.prerequisites = fs::CompactPrerequisiteBindings{};
    if (!baseline.payloads.empty()) {
        bindings.prerequisites->demands.push_back({0, 0, {true, 2, 2}});
        bindings.prerequisites->native.push_back({0, 0, 1});
    }
    auto supplied = fs::buildCompactWriterReaderInput(loop, input, index, bindings);
    if (!supplied.error.empty() || supplied.additional.size() != bindings.prerequisites->demands.size() ||
        supplied.native.size() != bindings.prerequisites->native.size()) { return false; }
    if (!supplied.additional.empty() && (supplied.additional[0].bounds.minimum != 2 ||
        supplied.native[0].displacement != 1)) { return false; }
    bindings.prerequisites->demands.push_back({UINT32_MAX, 0, {true, 0, 0}});
    return fs::buildCompactWriterReaderInput(loop, input, index, bindings).issue ==
           fs::CompactInputIssue::InvalidBinding;
}
bool checkInput(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    scf::ForOp loop;
    for (auto candidate : function.getOps<scf::ForOp>()) {
        if (loop) { return false; }
        loop = candidate;
    }
    if (!loop) { return false; }
    auto result = fs::buildCompactWriterReaderInput(loop, input, index);
    if (function->hasAttr("test.compact_nested")) {
        return result.issue == fs::CompactInputIssue::FixedSkeleton;
    }
    auto expected = function->getAttrOfType<IntegerAttr>("test.compact_classes");
    if (!expected || expected.getInt() < 0 || !result.error.empty() ||
        result.classEffects.size() != static_cast<uint64_t>(expected.getInt()) ||
        result.accesses.size() != result.accessEffects.size()) {
        llvm::errs() << "adapter error=" << result.error << " issue=" << static_cast<unsigned>(result.issue)
                     << " classes=" << result.classEffects.size() << " expected="
                     << (expected ? expected.getInt() : -1) << " accesses=" << result.accesses.size() << "\n";
        return false;
    }
    std::set<std::size_t> captured;
    for (const auto& group : result.classEffects) {
        if (group.empty()) { return false; }
        for (auto effect : group) { if (!captured.insert(effect).second) { return false; } }
    }
    std::set<std::size_t> original;
    for (auto* phase : result.phases) {
        for (auto effect : input.accesses().effectsFor(phase)) {
            const auto& record = input.accesses().effects()[effect];
            if (!(record.rangesMaterialized && record.ranges.empty())) { original.insert(effect); }
        }
    }
    if (captured != original) { return false; }
    bool rmw = false;
    for (const auto& access : result.accesses) {
        if (access.fullOverwrite) { return false; }
        rmw |= access.read && access.write;
    }
    if (function->hasAttr("test.compact_rmw") && !rmw) { return false; }
    if (function->hasAttr("test.compact_shifted")) {
        // These byte maps are disjoint in one visit but overlap across visits.
        // Shared overlap already rejects cancellation of loop-dependent symbols;
        // class extraction must preserve that conservative relationship.
        if (captured.size() != 2 || !input.accesses().mayOverlap(*captured.begin(), *captured.rbegin()) ||
            result.classEffects.size() != 1 || result.protectedPairs == 0) { return false; }
    }
    if (function->hasAttr("test.compact_unknown")) {
        bool unresolved = false;
        for (auto effect : captured) { unresolved |= !input.accesses().effects()[effect].rangesMaterialized; }
        if (!unresolved || result.queries.empty()) { return false; }
    }
    if (function->hasAttr("test.compact_prerequisite")) {
        const auto mapped = index.mapPrerequisites(result.phases);
        if (!mapped.error.empty() || mapped.demands.size() != result.additional.size() ||
            mapped.native.size() != result.native.size() || mapped.native.size() != 1 ||
            !mapped.demands.empty()) {
            llvm::errs() << "prerequisite classification mismatch: map error=" << mapped.error
                         << " demands=" << result.additional.size() << " native=" << result.native.size()
                         << " boundary=" << result.boundaryPrerequisites.size() << "\n";
            return false;
        }
        for (std::size_t id = 0; id < mapped.demands.size(); ++id) {
            const auto& actual = result.additional[id];
            if (actual.source != mapped.demands[id].source || actual.target != mapped.demands[id].target ||
                !actual.bounds.reachable || actual.bounds.minimum != 0 || actual.bounds.maximum != 0) {
                return false;
            }
        }
        for (std::size_t id = 0; id < mapped.native.size(); ++id) {
            const auto& actual = result.native[id];
            if (actual.source != mapped.native[id].source || actual.target != mapped.native[id].target ||
                actual.displacement != 0) { return false; }
        }
        // The fixture's scalar result is synchronously available: retain its
        // exact native 0->1 prerequisite, without inventing a software demand.
        if (result.native[0].source != 0 || result.native[0].target != 1) { return false; }
    }
    auto math = fs::analyzeCompactWriterReader(result.payloads, result.accesses, result.queries,
                                              result.additional, result.native);
    if (!math.error.empty()) { llvm::errs() << "compact math error=" << math.error << "\n"; return false; }
    if (!checkBindings(loop, input, index, result)) {
        llvm::errs() << "compact supplied-binding check failed: queries=" << result.queries.size()
                     << " additional=" << result.additional.size() << " native=" << result.native.size() << "\n";
        return false;
    }
    return true;
}
} // namespace
int runCompactWriterReaderInputChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!checkInput(function, input)) {
        llvm::errs() << "compact shared input check failed for " << function.getSymName() << "\n";
        return 1;
    }
    llvm::outs() << "compact shared input checks passed: " << function.getSymName() << "\n";
    return 0;
}
