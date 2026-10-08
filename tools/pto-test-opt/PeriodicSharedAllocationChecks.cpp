// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exhaustive permutation oracle; no assignment-solver helpers are reused.
#include "PTO/Transforms/FrontierSynch/PeriodicSharedAllocation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <numeric>

namespace fs = mlir::pto::frontiersynch;
namespace {
using Status = fs::PeriodicSharedAllocationStatus;
using Matrix = fs::PeriodicReuseMatrix;
std::optional<llvm::APInt> optimum(const Matrix& weights)
{
    std::vector<std::size_t> permutation(weights.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::optional<llvm::APInt> best;
    do {
        llvm::APInt total(128, 0);
        bool finite = true;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            auto weight = weights[i][permutation[i]];
            if (!weight) { finite = false; break; }
            total += llvm::APInt(128, *weight);
        }
        if (finite && (!best || total.ult(*best))) { best = total; }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}
bool layout(const Matrix& weights, const fs::PeriodicSharedAllocation& result)
{
    if (result.phases.size() != weights.size()) { return false; }
    std::vector<bool> seen(weights.size(), false);
    llvm::APInt total(128, 0);
    for (std::size_t c = 0; c < result.cycles.size(); ++c) {
        const auto& cycle = result.cycles[c];
        if (!cycle.laneCount || total.getActiveBits() > 64 || cycle.laneBegin != total.getZExtValue() ||
            cycle.firstPhase >= seen.size()) {
            return false;
        }
        llvm::APInt sum(128, 0);
        auto i = cycle.firstPhase;
        do {
            if (i >= seen.size() || seen[i]) { return false; }
            seen[i] = true;
            const auto& phase = result.phases[i];
            if (phase.cycle != c || phase.laneBegin != cycle.laneBegin || phase.laneCount != cycle.laneCount ||
                phase.offset >= cycle.laneCount || phase.successor >= seen.size() || !weights[i][phase.successor]) {
                return false;
            }
            const auto& next = result.phases[phase.successor];
            const llvm::APInt modulus(128, cycle.laneCount), weight(128, *weights[i][phase.successor]);
            for (auto ordinal : {uint64_t(0), uint64_t(1), uint64_t(7), uint64_t(UINT64_MAX)}) {
                auto a = (llvm::APInt(128, ordinal) + llvm::APInt(128, phase.offset)).urem(modulus);
                auto b = (llvm::APInt(128, ordinal) + weight + llvm::APInt(128, next.offset)).urem(modulus);
                if (a != b) { return false; }
            }
            sum += weight;
            i = phase.successor;
        } while (i != cycle.firstPhase);
        if (sum != llvm::APInt(128, cycle.laneCount)) { return false; }
        total += sum;
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }) &&
           total == llvm::APInt(128, result.budget);
}
bool compare(const Matrix& weights)
{
    const auto expected = optimum(weights);
    const auto result = fs::allocatePeriodicShared(weights);
    const auto status = !expected ? Status::NoFiniteCover :
        expected->getActiveBits() > 64 ? Status::Overflow : Status::Success;
    if (result.status != status) {
        llvm::errs() << "cycle-cover status disagrees with permutation oracle\n"; return false;
    }
    if (status != Status::Success) {
        return !result.error.empty() && result.phases.empty() && result.cycles.empty() && result.budget == 0;
    }
    if (!result.error.empty() || result.budget != expected->getZExtValue() || !layout(weights, result)) {
        llvm::errs() << "cycle-cover optimum or lane potentials disagree with oracle\n";
        return false;
    }
    return true;
}
bool exhaustive(uint64_t& checked)
{
    for (unsigned n = 0; n <= 3; ++n) {
        uint64_t cases = 1;
        for (unsigned cell = 0; cell < n * n; ++cell) { cases *= 3; }
        for (uint64_t pattern = 0; pattern < cases; ++pattern) {
            Matrix weights(n, std::vector<std::optional<uint64_t>>(n));
            auto digits = pattern;
            for (unsigned i = 0; i < n; ++i) {
                for (unsigned j = 0; j < n; ++j) {
                    const auto digit = digits % 3;
                    digits /= 3;
                    if (digit) { weights[i][j] = digit == 2 ? 5 : i < j ? 0 : 1; }
                }
            }
            if (!compare(weights)) { return false; }
            ++checked;
        }
    }
    // Larger matrices exercise longer alternating paths. Zero edges only go
    // forward, so every finite cycle has positive weight by construction.
    uint64_t random = 20261007;
    for (unsigned n = 4; n <= 6; ++n) {
        for (unsigned trial = 0; trial < 32; ++trial) {
            Matrix weights(n, std::vector<std::optional<uint64_t>>(n));
            for (unsigned i = 0; i < n; ++i) {
                for (unsigned j = 0; j < n; ++j) {
                    random ^= random << 13; random ^= random >> 7; random ^= random << 17;
                    if (random % 4) { weights[i][j] = random % 5 + (i >= j ? 1 : 0); }
                }
            }
            if (!compare(weights)) { return false; }
            ++checked;
        }
    }
    return true;
}
bool directedNamespaces()
{
    Matrix weights{{uint64_t(4), std::nullopt}, {std::nullopt, uint64_t(2)}};
    auto separate = fs::allocateDirectedPeriodic(weights, {{1, 2}, {2, 1}});
    auto same = fs::allocateDirectedPeriodic(weights, {{1, 2}, {1, 2}});
    if (separate.status != Status::Success || same.status != Status::Success || separate.budget != 4 || same.budget != 6 ||
        separate.phases[0].laneBegin != 0 || separate.phases[1].laneBegin != 0) { return false; }
    // Cross-direction edges cannot establish reuse of one notification state.
    weights[0][1] = 0; weights[1][0] = 1;
    return fs::allocateDirectedPeriodic(weights, {{1, 2}, {2, 1}}).budget == 4;
}
bool boundaries()
{
    const auto absent = std::nullopt;
    for (uint64_t b : {uint64_t(1), uint64_t(2), uint64_t(5), uint64_t(1) << 40, UINT64_MAX / 2 + 1}) {
        // Independent readiness/release cycles cost 2b. A certified zero-gap
        // readiness-to-release link and b-gap return share one b-wide cycle.
        Matrix shared{{b, uint64_t(0)}, {b, b}};
        if (!compare(shared) || fs::allocatePeriodicShared(shared).budget != b) { return false; }
        if (!compare(Matrix{{b, absent}, {absent, b}})) { return false; }
    }
    for (const auto& weights : std::vector<Matrix>{
             {{UINT64_MAX}}, {{UINT64_MAX, uint64_t(0)}, {UINT64_MAX, UINT64_MAX}},
             {{UINT64_MAX, absent}, {absent, uint64_t(1)}},
             {{uint64_t(2), absent}, {uint64_t(3), absent}},
             {{absent, uint64_t(0), absent}, {absent, absent, uint64_t(0)}, {uint64_t(7), absent, absent}}}) {
        if (!compare(weights)) { return false; }
    }
    for (const auto& weights : std::vector<Matrix>{
             {{uint64_t(0)}}, {{uint64_t(1), uint64_t(0)}, {uint64_t(0), uint64_t(1)}},
             {{absent, uint64_t(0), absent}, {absent, absent, uint64_t(0)}, {uint64_t(0), absent, absent}}}) {
        const auto result = fs::allocatePeriodicShared(weights);
        if (result.status != Status::ZeroWeightCycle || result.error.empty() || !result.phases.empty()) {
            return false;
        }
    }
    return fs::allocatePeriodicShared(Matrix{{uint64_t(1), uint64_t(2)}}).status == Status::InvalidInput;
}
} // namespace
int runPeriodicSharedAllocationChecks()
{
    uint64_t checked = 0;
    if (!exhaustive(checked) || !boundaries() || !directedNamespaces()) {
        llvm::errs() << "periodic shared allocation checks failed\n";
        return 1;
    }
    llvm::outs() << "periodic shared allocation: " << checked
                 << " permutation-oracle matrices plus capacity, zero-cycle and overflow boundaries passed\n";
    return 0;
}
