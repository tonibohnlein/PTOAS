// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Differential machine-bit checks for proof-only ranges. The host references
// below do not use LLVM ConstantRange or the interval/threshold proof service.
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include <functional>
#include <algorithm>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
using Reference = std::function<bool(uint64_t, uint64_t)>;
bool signedLess(uint64_t a, uint64_t b)
{
    return (a ^ (uint64_t(1) << 63)) < (b ^ (uint64_t(1) << 63));
}
Id count(fs::RegionExpressions& e, Id x, uint64_t stride)
{
    auto zero = e.constant(0), one = e.constant(1), divisor = e.constant(stride);
    auto distance = e.select(e.slt(zero, x), x, zero);
    return e.add(e.div(distance, divisor), e.select(e.eq(e.rem(distance, divisor), zero), zero, one));
}
uint64_t countReference(uint64_t x, uint64_t stride)
{
    auto distance = signedLess(0, x) ? x : uint64_t(0);
    return distance / stride + uint64_t(distance % stride != 0);
}
bool verify(fs::RegionExpressions& e, Id premise, Id consequence, const Reference& p, const Reference& q)
{
    const auto size = e.size();
    const bool proved = e.impliesByIntervals(premise, consequence);
    if (e.size() != size || !e.constructionError().empty()) { return false; }
    const uint64_t samples[] = {0, 1, 2, 3, 4, 7, 8, 15, 16, uint64_t(INT64_MAX) - 1,
        uint64_t(INT64_MAX), uint64_t(1) << 63, (uint64_t(1) << 63) + 1, UINT64_MAX - 1, UINT64_MAX};
    for (auto a : samples) {
        for (auto b : samples) {
            if (proved && p(a, b) && !q(a, b)) { return false; }
        }
    }
    return true;
}
bool countedPhase(fs::RegionExpressions& e, Id x, uint64_t stride)
{
    auto zero = e.constant(0), one = e.constant(1), t = count(e, x, stride);
    if (!e.impliesByIntervals(e.boolean(true), e.sle(zero, t)) ||
        !e.impliesByIntervals(e.boolean(true), e.le(t, e.constant(INT64_MAX))) ||
        e.impliesByIntervals(e.boolean(true), e.le(t, e.constant(8)))) { return false; }
    auto begin = e.select(e.sle(t, zero), t, zero);
    auto end = e.select(e.sle(t, one), t, one);
    auto active = e.land(e.lt(begin, end), e.lt(end, t));
    auto positive = e.lt(zero, end);
    auto previous = e.sub(end, one);
    auto path = e.land(positive, e.lor(e.lt(begin, previous), e.eq(begin, previous)));
    auto selected = e.land(e.lt(end, t), e.select(path, positive, e.boolean(false)));
    auto absent = e.select(selected, e.lnot(selected), e.boolean(false));
    auto query = e.land(e.land(e.lt(end, t), path),
        e.lor(e.select(e.select(selected, e.lt(end, t), e.boolean(false)), path, e.boolean(false)),
              e.select(e.select(absent, e.lt(end, t), e.boolean(false)), path, e.boolean(false))));
    if (!e.impliesByIntervals(active, query) || !e.impliesByIntervals(absent, e.boolean(false))) { return false; }
    return verify(e, active, query,
        [stride](uint64_t a, uint64_t) { return countReference(a, stride) >= 2; },
        [stride](uint64_t a, uint64_t) { return countReference(a, stride) >= 2; });
}
bool differentialRanges(fs::RegionExpressions& e, Id x, Id y)
{
    struct Expression { Id id; std::function<uint64_t(uint64_t, uint64_t)> value; };
    auto zero = e.constant(0), one = e.constant(1);
    const Expression values[] = {
        {x, [](uint64_t a, uint64_t) { return a; }},
        {e.add(x, one), [](uint64_t a, uint64_t) { return a + uint64_t(1); }},
        {e.sub(x, one), [](uint64_t a, uint64_t) { return a - uint64_t(1); }},
        {count(e, x, 3), [](uint64_t a, uint64_t) { return countReference(a, 3); }},
        {e.select(e.lt(x, y), x, y), [](uint64_t a, uint64_t b) { return std::min(a, b); }}
    };
    for (const auto& value : values) {
        for (uint64_t bound : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(INT64_MAX),
                               uint64_t(1) << 63, UINT64_MAX}) {
            auto c = e.constant(bound);
            if (!verify(e, e.lt(value.id, c), e.sle(zero, value.id),
                [=](uint64_t a, uint64_t b) { return value.value(a, b) < bound; },
                [=](uint64_t a, uint64_t b) { return !signedLess(value.value(a, b), 0); })) { return false; }
            if (!verify(e, e.sle(value.id, c), e.le(value.id, c),
                [=](uint64_t a, uint64_t b) { return !signedLess(bound, value.value(a, b)); },
                [=](uint64_t a, uint64_t b) { return value.value(a, b) <= bound; })) { return false; }
        }
    }
    return true;
}
} // namespace
bool runRegionIntervalChecks(Value first)
{
    Block inputs;
    auto second = inputs.addArgument(first.getType(), first.getLoc());
    fs::RegionExpressions e;
    auto x = e.input(first), y = e.input(second), zero = e.constant(0), one = e.constant(1);
    if (!differentialRanges(e, x, y)) { return false; }
    for (uint64_t stride : {1U, 2U, 3U, 7U}) {
        if (!countedPhase(e, x, stride)) { return false; }
    }
    // No implicit nonnegative input or caller-bound facts. Wrapped increments
    // and independent counts cannot inherit another expression's phase facts.
    auto truth = e.boolean(true), nonnegative = e.sle(zero, x);
    auto wrapped = e.sle(zero, e.add(x, one));
    auto independent = e.eq(count(e, x, 2), count(e, y, 2));
    if (e.impliesByIntervals(truth, nonnegative) || e.impliesByIntervals(truth, wrapped) ||
        e.impliesByIntervals(truth, independent) ||
        e.impliesByIntervals(fs::RegionExpressions::invalid, truth)) { return false; }
    if (!verify(e, truth, wrapped, [](uint64_t, uint64_t) { return true; },
        [](uint64_t a, uint64_t) { return !signedLess(a + uint64_t(1), 0); })) { return false; }
    const auto size = e.size();
    {
        fs::RegionExpressions::Transaction transaction(e);
        if (!countedPhase(e, e.add(x, one), 3)) { return false; }
    }
    if (e.size() != size || !countedPhase(e, e.sub(x, one), 3)) { return false; }
    return true;
}
