// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using I = fs::BoundInteger;
using Id = fs::RegionExpressions::Id;
bool contains(const fs::IntegerSystem& system, ArrayRef<int64_t> point)
{
    for (const auto& row : system.constraints()) {
        I sum(0);
        for (auto [coefficient, value] : llvm::zip(row.coefficients, point)) { sum += coefficient * I(value); }
        if (sum > row.bound) { return false; }
    }
    for (const auto& row : system.congruences()) {
        I sum(0);
        for (auto [coefficient, value] : llvm::zip(row.coefficients, point)) { sum += coefficient * I(value); }
        if (mod(sum, row.modulus) != mod(row.residue, row.modulus)) { return false; }
    }
    return true;
}
fs::IntegerSystem box(ArrayRef<std::pair<int64_t, int64_t>> ranges)
{
    std::vector<fs::IntegerConstraint> rows;
    for (unsigned i = 0; i < ranges.size(); ++i) {
        fs::IntegerConstraint row;
        row.coefficients.resize(ranges.size()); row.coefficients[i] = I(1); row.bound = I(ranges[i].second);
        rows.push_back(row); row.coefficients[i] = I(-1); row.bound = -I(ranges[i].first); rows.push_back(row);
    }
    return *fs::IntegerSystem::create(ranges.size(), rows);
}
bool check(fs::RegionExpressions& arena, Id predicate, ArrayRef<Id> variables,
    const fs::IntegerSystem& domain, ArrayRef<std::vector<int64_t>> points,
    const std::function<bool(ArrayRef<int64_t>)>& expected, uint64_t& count)
{
    std::string error;
    fs::RegionExpressions::RelationCost cost;
    auto relation = arena.integerRelation(predicate, variables, domain, error, &cost);
    if (failed(relation)) { llvm::errs() << "expression relation: " << error << "\n"; return false; }
    for (const auto& point : points) {
        const bool actual = llvm::any_of(*relation, [&](const auto& piece) { return contains(piece, point); });
        const bool want = contains(domain, point) && expected(point);
        if (actual != want) { llvm::errs() << "expression relation differs at point\n"; return false; }
        ++count;
    }
    return true;
}
} // namespace
bool runExpressionRelationChecks(MLIRContext* context)
{
    Block block;
    auto location = UnknownLoc::get(context);
    auto xValue = block.addArgument(IndexType::get(context), location);
    auto yValue = block.addArgument(IndexType::get(context), location);
    auto pValue = block.addArgument(IntegerType::get(context, 1), location);
    auto byteValue = block.addArgument(IntegerType::get(context, 8), location);
    fs::RegionExpressions arena;
    auto x = arena.input(xValue), y = arena.input(yValue), p = arena.input(pValue), byte = arena.input(byteValue);
    const auto zero = arena.constant(0), one = arena.constant(1), three = arena.constant(3);
    std::vector<std::vector<int64_t>> points;
    for (int64_t a = -4; a <= 5; ++a) {
        for (int64_t b = -4; b <= 5; ++b) {
            for (int64_t c : {0, 1}) { points.push_back({a, b, c}); }
        }
    }
    auto domain = box({{-4, 5}, {-4, 5}, {0, 1}});
    const std::vector<Id> variables{x, y, p};
    uint64_t count = 0;
    auto test = [&](StringRef name, Id expression, auto expected) {
        llvm::outs() << "relation oracle: " << name << "\n"; llvm::outs().flush();
        return check(arena, expression, variables, domain, points, expected, count);
    };
    if (!test("wrap-add", arena.eq(arena.add(x, one), y), [](auto v) {
            return uint64_t(v[0]) + 1 == uint64_t(v[1]); }) ||
        !test("wrap-sub", arena.eq(arena.sub(x, one), y), [](auto v) {
            return uint64_t(v[0]) - 1 == uint64_t(v[1]); }) ||
        !test("unsigned-order", arena.lt(x, y), [](auto v) { return uint64_t(v[0]) < uint64_t(v[1]); }) ||
        !test("signed-order", arena.sle(x, y), [](auto v) { return v[0] <= v[1]; }) ||
        !test("unsigned-rem", arena.eq(arena.rem(x, three), y), [](auto v) {
            return uint64_t(v[0]) % 3 == uint64_t(v[1]); }) ||
        !test("unsigned-div", arena.eq(arena.div(x, three), y), [](auto v) {
            return uint64_t(v[0]) / 3 == uint64_t(v[1]); }) ||
        !test("selected-wrap", arena.eq(arena.select(p, arena.sub(x, one), arena.add(x, one)), y), [](auto v) {
            return (v[2] ? uint64_t(v[0]) - 1 : uint64_t(v[0]) + 1) == uint64_t(v[1]); }) ||
        !test("boolean", arena.lnot(arena.lor(p, arena.lt(x, y))), [](auto v) {
            return !v[2] && !(uint64_t(v[0]) < uint64_t(v[1])); })) { return false; }
    auto ordered = arena.land(arena.sle(zero, x), arena.slt(x, y));
    if (!test("mandatory-domain", arena.land(ordered, arena.eq(arena.minimum(x, y), x)), [](auto v) {
            return 0 <= v[0] && v[0] < v[1]; }) ||
        !test("optional-domain", arena.lor(ordered, arena.eq(x, y)), [](auto v) {
            return (0 <= v[0] && v[0] < v[1]) || v[0] == v[1]; }) ||
        !test("negated-domain", arena.lnot(arena.lor(arena.slt(x, zero), arena.sle(y, x))), [](auto v) {
            return 0 <= v[0] && v[0] < v[1]; }) ||
        !test("minimum-wrap", arena.eq(arena.minimum(x, arena.add(x, one)), y), [](auto v) {
            return std::min(uint64_t(v[0]), uint64_t(v[0]) + 1) == uint64_t(v[1]); }) ||
        !test("cancelled-difference", arena.eq(arena.sub(arena.add(x, y), x), y), [](auto) {
            return true; })) { return false; }
    const auto dx = getAffineDimExpr(0, context), dy = getAffineDimExpr(1, context);
    const auto mappedDomain = *fs::IntegerSystem::create(2,
        {{{I(1), I(0)}, I(3)}, {{I(-1), I(0)}, I(5)}, {{I(0), I(1)}, I(2)}});
    std::string mappedError;
    auto mapped = arena.integerMappedPredicate(mappedDomain, {x, y},
        {dx - dy * 2 + dy.floorDiv(3), dy.ceilDiv(2)}, 2, {1, 0}, mappedError);
    auto modulo = arena.integerMappedPredicate(mappedDomain, {x, y},
        {dx - dy % 3, dy.ceilDiv(2)}, 2, {1, 0}, mappedError);
    if (!mapped || !modulo) { return false; }
    auto floor = [](int64_t value, int64_t divisor) {
        return value / divisor - (value % divisor < 0);
    };
    auto expectedMap = [&](auto point, bool useModulo) {
        const auto first = useModulo ? point[0] - (point[1] - floor(point[1], 3) * 3) :
            point[0] - point[1] * 2 + floor(point[1], 3);
        const auto second = -floor(-point[1], 2);
        const auto quotient = floor(first, 2);
        return first - quotient * 2 == 1 && second % 2 == 0 && quotient >= -5 && quotient <= 3 &&
            floor(second, 2) <= 2;
    };
    if (!test("mapped-periodic", *mapped, [&](auto point) { return expectedMap(point, false); }) ||
        !test("mapped-periodic-complement", arena.lnot(*mapped), [&](auto point) {
            return !expectedMap(point, false); }) ||
        !test("mapped-modulo", *modulo, [&](auto point) { return expectedMap(point, true); })) { return false; }
    auto periodicCongruence = arena.integerPredicate(
        *fs::IntegerSystem::create(2, {}, {{{I(2), I(-3)}, I(2), I(5)}}), {x, y}, 3, {1, 2});
    auto expectedCongruence = [&](auto point) {
        const auto qx = floor(point[0], 3), qy = floor(point[1], 3);
        const auto sum = 2 * qx - 3 * qy;
        return point[0] - 3 * qx == 1 && point[1] - 3 * qy == 2 && sum - 5 * floor(sum, 5) == 2;
    };
    if (!test("periodic-congruence", periodicCongruence, expectedCongruence) ||
        !test("periodic-congruence-complement", arena.lnot(periodicCongruence), [&](auto point) {
            return !expectedCongruence(point); })) { return false; }
    auto strideDomain = *fs::IntegerSystem::create(2,
        {{{I(1), I(0)}, I(4)}, {{I(-1), I(0)}, I(0)}, {{I(1), I(2)}, I(-1)}});
    auto stridePredicate = arena.integerMappedPredicate(strideDomain, {x, y},
        {dx - dy.floorDiv(8) * 4096, dy}, 8, {0, 3}, mappedError);
    if (!stridePredicate) { return false; }
    std::vector<std::vector<int64_t>> stridePoints;
    for (int64_t parameter = -20; parameter <= 20; ++parameter) {
        for (int64_t address : {-8192, -4096, 0, 4096, 8192, 7}) {
            stridePoints.push_back({address, parameter});
        }
    }
    auto expectedStride = [&](auto point) {
        const auto parameter = floor(point[1], 8);
        const auto shifted = point[0] - 4096 * parameter;
        const auto address = floor(shifted, 8);
        return point[1] - 8 * parameter == 3 && shifted - 8 * address == 0 &&
            address >= 0 && address <= 4 && address + 2 * parameter <= -1;
    };
    const auto strideUniverse = box({{INT64_MIN, INT64_MAX}, {INT64_MIN, INT64_MAX}});
    if (!check(arena, *stridePredicate, {x, y}, strideUniverse, stridePoints, expectedStride, count) ||
        !check(arena, arena.lnot(*stridePredicate), {x, y}, strideUniverse, stridePoints,
            [&](auto point) { return !expectedStride(point); }, count)) { return false; }
    auto congruence = *fs::IntegerSystem::create(1, {}, {{{I(1)}, I(1), I(3)}});
    auto congruent = arena.integerPredicate(congruence, {x}, 1, {0});
    if (!test("negated-congruence", arena.lnot(congruent), [](auto v) { return (v[0] % 3 + 3) % 3 != 1; })) {
        return false;
    }
    auto witness = arena.integerWitness({{I(1)}, I(-2)}, I(3), {x}, 1, {0}, 0);
    if (!test("signed-floor-witness", arena.eq(witness, y), [](auto v) {
            const auto numerator = v[0] - 2;
            return numerator / 3 - (numerator % 3 < 0) == v[1]; })) { return false; }
    auto residue = arena.integerPredicate(*fs::IntegerSystem::create(1, {}), {x}, 3, {2});
    if (!test("negative-residue", residue, [](auto v) { return (v[0] % 3 + 3) % 3 == 2; })) { return false; }
    std::vector<std::vector<int64_t>> bytePoints{{-1}, {0}, {1}, {127}, {128}, {255}, {256}};
    if (!check(arena, arena.sle(byte, arena.constant(255)), {byte}, box({{-1, 256}}), bytePoints,
            [](auto v) { return 0 <= v[0] && v[0] <= 255; }, count)) { return false; }
    auto broad = box({{INT64_MIN, INT64_MAX}, {INT64_MIN, INT64_MAX}, {0, 1}});
    std::vector<std::vector<int64_t>> extremes;
    for (int64_t a : {INT64_MIN, INT64_MIN + 1, int64_t(-1), int64_t(0), int64_t(1), INT64_MAX}) {
        for (int64_t b : {INT64_MIN, INT64_MIN + 1, int64_t(-1), int64_t(0), int64_t(1), INT64_MAX}) {
            extremes.push_back({a, b, 0});
        }
    }
    if (!check(arena, arena.eq(arena.add(x, one), y), variables, broad, extremes,
            [](auto v) { return uint64_t(v[0]) + 1 == uint64_t(v[1]); }, count) ||
        !check(arena, arena.lt(x, y), variables, broad, extremes,
            [](auto v) { return uint64_t(v[0]) < uint64_t(v[1]); }, count)) { return false; }
    std::string error;
    fs::RegionExpressions::RelationCost unusedCost;
    auto unused = arena.integerRelation(arena.boolean(true), variables, broad, error, &unusedCost);
    if (failed(unused) || unusedCost.gates != 1 || unusedCost.projections != 0 || unused->size() != 1) {
        llvm::errs() << "unused relation inputs created projection work\n"; return false;
    }
    if (succeeded(arena.integerRelation(arena.eq(x, y), {x}, box({{0, 2}}), error)) || error.empty() ||
        succeeded(arena.integerRelation(arena.eq(x, y), {x, x}, box({{0, 2}, {0, 2}}), error)) ||
        succeeded(arena.integerRelation(zero, {}, *fs::IntegerSystem::create(0, {}), error))) { return false; }
    auto empty = box({{1, 0}});
    if (succeeded(arena.integerRelation(zero, {x}, empty, error)) ||
        succeeded(arena.integerRelation(arena.eq(x, y), {x}, empty, error))) { return false; }
    llvm::outs() << "regional expression relation checked " << count << " signed/unsigned points\n";
    return true;
}
