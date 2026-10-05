// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite integer witnesses for projection and Boolean subtraction.
#include "PTO/Transforms/FrontierSynch/IntegerRelations.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
namespace {
using Integer = fs::BoundInteger;
using System = fs::IntegerSystem;
using Constraint = fs::IntegerConstraint;
using Congruence = fs::IntegerCongruence;
using Point = std::vector<Integer>;
Constraint row(std::initializer_list<int64_t> coefficients, int64_t bound)
{
    Constraint result;
    for (auto value : coefficients) { result.coefficients.emplace_back(value); }
    result.bound = Integer(bound);
    return result;
}
Congruence congruence(std::initializer_list<int64_t> coefficients, int64_t residue, int64_t modulus)
{
    auto converted = row(coefficients, 0);
    return {std::move(converted.coefficients), Integer(residue), Integer(modulus)};
}
Integer evaluate(llvm::ArrayRef<Integer> coefficients, llvm::ArrayRef<Integer> point)
{
    Integer value(0);
    for (unsigned i = 0; i < coefficients.size(); ++i) { value += coefficients[i] * point[i]; }
    return value;
}
bool contains(const System& system, llvm::ArrayRef<Integer> point)
{
    if (system.dimensions() != point.size()) { return false; }
    for (const auto& atom : system.constraints()) {
        if (evaluate(atom.coefficients, point) > atom.bound) { return false; }
    }
    for (const auto& atom : system.congruences()) {
        auto remainder = (evaluate(atom.coefficients, point) - atom.residue) % atom.modulus;
        if (remainder != 0) { return false; }
    }
    return true;
}
bool contains(llvm::ArrayRef<System> systems, llvm::ArrayRef<Integer> point)
{
    for (const auto& system : systems) {
        if (contains(system, point)) { return true; }
    }
    return false;
}
bool boundedProjection(const System& source, llvm::ArrayRef<unsigned> keep,
                       const std::vector<std::pair<int64_t, int64_t>>& bounds, uint64_t& checked)
{
    auto projected = source.project(keep);
    if (failed(projected) || bounds.size() != source.dimensions()) { return false; }
    std::vector<bool> retained(source.dimensions(), false);
    for (auto column : keep) { retained[column] = true; }
    Point original(source.dimensions(), Integer(0));
    std::function<bool(unsigned)> exists = [&](unsigned coordinate) {
        if (coordinate == source.dimensions()) { return contains(source, original); }
        if (retained[coordinate]) { return exists(coordinate + 1); }
        for (auto value = bounds[coordinate].first; value <= bounds[coordinate].second; ++value) {
            original[coordinate] = Integer(value);
            if (exists(coordinate + 1)) { return true; }
        }
        return false;
    };
    std::function<bool(unsigned)> verify = [&](unsigned coordinate) {
        if (coordinate == source.dimensions()) {
            Point selected;
            for (auto column : keep) { selected.push_back(original[column]); }
            ++checked;
            return contains(*projected, selected) == exists(0);
        }
        if (!retained[coordinate]) { return verify(coordinate + 1); }
        for (auto value = bounds[coordinate].first - 1; value <= bounds[coordinate].second + 1; ++value) {
            original[coordinate] = Integer(value);
            if (!verify(coordinate + 1)) { return false; }
        }
        return true;
    };
    return verify(0);
}
bool projectionChecks(uint64_t& checked)
{
    // x+y<=N, including a negative N, with finite exact x/y ranges.
    auto sumBound = System::create(3, {row({1, 1, -1}, 0), row({-1, 0, 0}, 0), row({1, 0, 0}, 4),
        row({0, -1, 0}, 0), row({0, 1, 0}, 4), row({0, 0, -1}, 1), row({0, 0, 1}, 8)});
    if (failed(sumBound) || !boundedProjection(*sumBound, {2, 0}, {{0, 4}, {0, 4}, {-1, 8}}, checked)) {
        return false;
    }
    // u=7v+3w+c. Eliminate v while retaining u,w,c; congruence is essential.
    auto affine = System::create(4, {row({1, -7, -3, -1}, 0), row({-1, 7, 3, 1}, 0),
        row({1, 0, 0, 0}, 25), row({-1, 0, 0, 0}, 25), row({0, 1, 0, 0}, 2),
        row({0, -1, 0, 0}, 2), row({0, 0, 1, 0}, 2), row({0, 0, -1, 0}, 2),
        row({0, 0, 0, 1}, 2), row({0, 0, 0, -1}, 2)});
    if (failed(affine) || !boundedProjection(*affine, {0, 2, 3},
                                            {{-25, 25}, {-2, 2}, {-2, 2}, {-2, 2}}, checked)) { return false; }
    // A genuine signed octagon. Pairing two bounds can produce unary 2*y.
    auto signedRows = System::create(2, {row({1, 1}, 1), row({-1, 1}, 0), row({1, -1}, 2),
                                        row({-1, -1}, 2), row({1, 0}, 4), row({-1, 0}, 4)});
    if (failed(signedRows) || !boundedProjection(*signedRows, {1}, {{-4, 4}, {-4, 4}}, checked)) {
        return false;
    }
    auto divisibility = System::create(2, {row({1, 0}, 3), row({-1, 0}, 3),
                                         row({0, 1}, 3), row({0, -1}, 3)},
                                       {congruence({2, -1}, -1, 4), congruence({1, 1}, 1, 3)});
    return succeeded(divisibility) && boundedProjection(*divisibility, {1}, {{-3, 3}, {-3, 3}}, checked);
}
bool feasibilityAndWitnessChecks()
{
    auto half = System::create(1, {row({2}, 1), row({-2}, -1)});
    auto rounded = System::create(1, {row({2}, 3), row({-2}, -1)});
    auto parity = System::create(1, {}, {congruence({2}, 1, 4)});
    if (failed(half) || !half->isEmpty() || failed(rounded) || rounded->isEmpty() ||
        !contains(*rounded, {Integer(1)}) || contains(*rounded, {Integer(0)}) ||
        failed(parity) || !parity->isEmpty()) { return false; }
    // No lower bound: a witness must be found below a negative upper bound.
    auto negative = System::create(1, {row({1}, -5)}, {congruence({1}, 1, 3)});
    if (failed(negative) || negative->isEmpty()) { return false; }
    auto witnesses = negative->eliminateWithWitness(0);
    if (failed(witnesses)) { return false; }
    bool found = false;
    for (const auto& witness : *witnesses) {
        if (!contains(witness.domain, {})) { continue; }
        if (witness.numerator.constant % witness.denominator != 0) { return false; }
        auto value = witness.numerator.constant / witness.denominator;
        if (!contains(*negative, {value})) { return false; }
        found = true;
    }
    auto unbounded = System::create(1, {}, {congruence({3}, -1, 5)});
    return found && succeeded(unbounded) && !unbounded->isEmpty();
}
bool remapAndDifferenceChecks(uint64_t& checked)
{
    auto different = System::create(2, {row({1, -1}, -1)});
    if (failed(different)) { return false; }
    auto identified = different->remap(1, {0, 0});
    if (failed(identified) || !identified->isEmpty() || succeeded(different->project({0, 0}))) { return false; }
    auto interval = System::create(1, {row({1}, 8), row({-1}, 8)});
    auto evens = System::create(1, {}, {congruence({1}, 0, 2)});
    auto thirds = System::create(1, {}, {congruence({1}, 1, 3)});
    if (failed(interval) || failed(evens) || failed(thirds)) { return false; }
    auto difference = fs::subtractIntegerUnions(1, {*interval}, {*evens, *thirds});
    if (failed(difference)) { return false; }
    for (int64_t x = -10; x <= 10; ++x) {
        const Point point{Integer(x)};
        const bool expected = contains(*interval, point) && !contains(*evens, point) && !contains(*thirds, point);
        ++checked;
        if (contains(*difference, point) != expected) { return false; }
    }
    // RHS bit length must not cause enumeration of the magnitude of a bound.
    Integer huge(1);
    for (unsigned i = 0; i < 200; ++i) { huge *= 2; }
    auto singleton = System::create(1, {{{Integer(7)}, huge}, {{Integer(-7)}, -huge}});
    if (failed(singleton) || !singleton->isEmpty()) { return false; } // 7 does not divide 2^200.
    auto integer = System::create(1, {{{Integer(7)}, huge * 7}, {{Integer(-7)}, -huge * 7}});
    if (failed(integer) || integer->isEmpty()) { return false; }
    auto projected = integer->project({});
    return succeeded(projected) && contains(*projected, {});
}
} // namespace
int runIntegerRelationChecks()
{
    uint64_t checked = 0;
    if (!projectionChecks(checked)) { llvm::errs() << "integer projection oracle differs\n"; return 1; }
    if (!feasibilityAndWitnessChecks()) { llvm::errs() << "integer feasibility/witness check failed\n"; return 1; }
    if (!remapAndDifferenceChecks(checked)) { llvm::errs() << "integer remap/difference check failed\n"; return 1; }
    llvm::outs() << "integer relation exactness passed " << checked << " bounded points\n";
    return 0;
}
