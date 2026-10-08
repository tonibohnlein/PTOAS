// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ResolvedLifetimeStream.h"
#include "PTO/Transforms/FrontierSynch/GuardedRanks.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
namespace fs = mlir::pto::frontiersynch;
namespace {
bool oracle(unsigned seed)
{
    const std::vector<uint32_t> pipes{3, 7, 3, 9};
    constexpr uint32_t count = 32, span = 2;
    fs::RegionExpressions e;
    std::vector<fs::GuardedRankPayload> payloads;
    std::vector<fs::GuardedRankEdge> candidates, native;
    std::vector<std::vector<uint64_t>> incoming(count), fixed(count);
    for (uint32_t target = 0; target < count; ++target) {
        payloads.push_back({pipes[target % pipes.size()], e.boolean((target + seed) % 5 != 2)});
        for (uint32_t source = target; source; ) {
            --source;
            if (target / pipes.size() - source / pipes.size() > span) { continue; }
            const bool isNative = source + 1 == target && (target + seed) % 4 == 3;
            const bool active = (source + target + seed) % 3 != 0 &&
                (source + seed) % 5 != 2 && (target + seed) % 5 != 2;
            auto& edges = isNative ? native : candidates;
            edges.push_back({source, target, e.boolean(active)});
            if (active) { (isNative ? fixed[target] : incoming[target]).push_back(source); }
        }
    }
    const auto full = fs::reduceGuardedRanks(e, payloads, candidates, native);
    if (!full.error.empty()) { return false; }
    fs::ResolvedLifetimeStream stream(pipes, span, count / pipes.size());
    if (!stream.error().empty()) { return false; }
    for (uint32_t target = 0; target < count; ++target) {
        if (e.constantValue(payloads[target].present) == 0) { continue; }
        const auto step = stream.advance(target, true, incoming[target], fixed[target]);
        if (!step.error.empty() || step.rank != e.constantValue(full.ranks[target])) { return false; }
        std::set<uint64_t> expected;
        for (const auto& edge : full.retained) {
            if (edge.target == target && e.constantValue(edge.guard) == 1) { expected.insert(edge.source); }
        }
        if (std::set<uint64_t>(step.retained.begin(), step.retained.end()) != expected) { return false; }
        for (std::size_t k = 0; k < step.start.size(); ++k) {
            if (step.start[k] != e.constantValue(full.starts[target][k]) ||
                step.completion[k] != e.constantValue(full.completions[target][k])) { return false; }
        }
    }
    const auto& cost = stream.cost();
    return cost.rowEntries == stream.columns() * (cost.occurrences + cost.retained + cost.native);
}
bool boundaries()
{
    const std::vector<uint32_t> pipes{0, 1, 2};
    fs::ResolvedLifetimeStream stream(pipes, 2, 2000);
    if (!stream.error().empty() || stream.historySlots() != 9) { return false; }
    std::optional<uint64_t> previous;
    for (uint64_t id = 0; id < 6000; ++id) {
        if (id % 4 == 1) { continue; }
        std::vector<uint64_t> sources;
        if (previous) { sources.push_back(*previous); }
        const auto result = stream.advance(id, true, sources);
        if (!result.error.empty() || stream.historySlots() != 9) { return false; }
        previous = id;
    }
    if (stream.advance(6000, true).error.empty()) { return false; }
    fs::ResolvedLifetimeStream zero(pipes, 0, 0);
    if (!zero.error().empty() || zero.advance(0, true).error.empty()) { return false; }
    fs::ResolvedLifetimeStream overflow(pipes, 0, UINT64_MAX);
    if (overflow.error().empty()) { return false; }
    fs::ResolvedLifetimeStream invalid(pipes, 1, 10);
    if (!invalid.advance(0, true).error.empty() || !invalid.advance(1, false).error.empty()) { return false; }
    if (invalid.advance(2, true, {1}).error.empty()) { return false; }
    if (invalid.advance(2, true, {0, 0}).error.empty()) { return false; }
    if (!invalid.advance(2, true, {0}).error.empty()) { return false; }
    if (invalid.advance(6, true, {0}).error.empty()) { return false; }
    if (!invalid.advance(6, true).error.empty()) { return false; }
    return true;
}
} // namespace
int runResolvedLifetimeStreamChecks()
{
    for (unsigned seed = 0; seed < 8; ++seed) {
        if (!oracle(seed)) { llvm::errs() << "resolved lifetime oracle failed seed=" << seed << "\n"; return 1; }
    }
    if (!boundaries()) { llvm::errs() << "resolved lifetime register/domain checks failed\n"; return 1; }
    llvm::outs() << "resolved lifetime: forward oracle and bounded ring checks passed\n";
    return 0;
}
