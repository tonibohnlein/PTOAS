// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent bounded witnesses and actual candidate counts for feasibility ranking.
#ifndef PTO_BOUNDED_FEASIBILITY_RANKING_CHECKS_H
#define PTO_BOUNDED_FEASIBILITY_RANKING_CHECKS_H
namespace {
bool boundedRankingTruthTables(uint64_t& checked)
{
    for (int64_t modulus : {4, 8, 16, 32}) {
        for (int64_t lower : {-3, -1, 0, 2}) {
            for (int64_t width : {0, 1, 3}) {
                for (int64_t residue : {int64_t(-2), int64_t(0), int64_t(1), modulus / 2 - 1}) {
                    auto source = System::create(2, {row({-1, 0}, -lower), row({1, 0}, lower + width)},
                        {congruence({1, 2}, residue, modulus), congruence({3, 0}, 3, 6)});
                    if (failed(source)) { return false; }
                    bool exists = false;
                    // All y occurrences repeat modulo lcm(modulus,6). The
                    // second atom constrains only x, so one modulus is exact.
                    for (int64_t x = lower; x <= lower + width; ++x) {
                        for (int64_t y = 0; y < modulus; ++y) {
                            exists |= (x + 2 * y - residue) % modulus == 0 && (3 * x - 3) % 6 == 0;
                            ++checked;
                        }
                    }
                    for (const auto& permutation : {std::vector<unsigned>{0, 1}, std::vector<unsigned>{1, 0}}) {
                        auto reordered = source->remap(2, permutation);
                        fs::IntegerFeasibilityCost cost;
                        if (failed(reordered) || reordered->isEmpty(&cost) == exists) { return false; }
                    }
                }
            }
        }
    }
    // Nonunit affine strips preserve residual divisibility, even when the
    // bounded unary interval and strip have equal or different candidate counts.
    for (int64_t lower : {-3, -1, 0}) {
        for (int64_t width : {0, 1, 2, 4}) {
            for (int64_t residue : {-1, 0, 1}) {
                auto source = System::create(2,
                    {row({-1, 0}, 2), row({1, 0}, 2), row({3, -2}, lower + width),
                     row({-3, 2}, -lower)}, {congruence({0, 1}, residue, 6)});
                if (failed(source)) { return false; }
                bool exists = false;
                for (int64_t x = -2; x <= 2; ++x) {
                    for (int64_t y = -8; y <= 8; ++y) {
                        exists |= lower <= 3 * x - 2 * y && 3 * x - 2 * y <= lower + width &&
                            (y - residue) % 6 == 0;
                        ++checked;
                    }
                }
                for (const auto& permutation : {std::vector<unsigned>{0, 1}, std::vector<unsigned>{1, 0}}) {
                    auto reordered = source->remap(2, permutation);
                    if (failed(reordered) || reordered->isEmpty() == exists) { return false; }
                }
            }
        }
    }
    return true;
}
bool boundedRankingWorkCounts()
{
    // Old period-only ordering prefers unbounded y (period M/2) over bounded
    // x (period M), although x has only two candidates. Exact elimination of
    // the old first coordinate supplies an independent small-M work reference.
    auto reference = System::create(2, {row({-1, 0}, 0), row({1, 0}, 1)}, {congruence({1, 2}, 127, 256)});
    if (failed(reference)) { return false; }
    auto oldOrder = reference->eliminateWithWitness(1);
    if (failed(oldOrder)) { return false; }
    uint64_t oldVisits = 0;
    for (const auto& candidate : *oldOrder) {
        ++oldVisits;
        if (!candidate.domain.isEmpty()) { break; }
    }
    if (oldVisits <= 2) { return false; }
    for (unsigned bits : {8U, 32U, 128U}) {
        Integer period(1);
        for (unsigned bit = 0; bit < bits; ++bit) { period *= 2; }
        auto source = System::create(2, {row({-1, 0}, 0), row({1, 0}, 1)},
            {{{Integer(1), Integer(2)}, period / 2 - 1, period}});
        if (failed(source)) { return false; }
        for (const auto& permutation : {std::vector<unsigned>{0, 1}, std::vector<unsigned>{1, 0}}) {
            auto reordered = source->remap(2, permutation);
            fs::IntegerFeasibilityCost cost;
            if (failed(reordered) || reordered->isEmpty(&cost) || cost.candidateSubstitutions != 2 ||
                cost.unaryDecisions != 1 || cost.calls != 3) { return false; }
        }
    }
    return true;
}
bool boundedFeasibilityRankingChecks(uint64_t& checked)
{
    return boundedRankingTruthTables(checked) && boundedRankingWorkCounts();
}
} // namespace
#endif
