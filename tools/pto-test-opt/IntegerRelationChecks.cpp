// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite integer witnesses for projection and Boolean subtraction.
#include "PTO/Transforms/FrontierSynch/Arithmetic/IntegerRelations.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <functional>
#include <optional>
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
bool permutedProjectionChecks(const System& source, llvm::ArrayRef<unsigned> keep,
    const std::vector<std::pair<int64_t, int64_t>>& bounds, uint64_t& checked)
{
    const auto dimensions = source.dimensions();
    for (unsigned shift = 0; shift < dimensions; ++shift) {
        for (bool reverse : {false, true}) {
            std::vector<unsigned> map(dimensions), selected;
            auto reorderedBounds = bounds;
            for (unsigned i = 0; i < dimensions; ++i) {
                map[i] = ((reverse ? dimensions - 1 - i : i) + shift) % dimensions;
                reorderedBounds[map[i]] = bounds[i];
            }
            for (auto column : keep) { selected.push_back(map[column]); }
            auto reordered = source.remap(dimensions, map);
            if (failed(reordered) || !boundedProjection(*reordered, selected, reorderedBounds, checked)) {
                return false;
            }
        }
    }
    return true;
}
bool projectionDefinitionChecks(uint64_t& checked)
{
    // x, y, sign, q, r: truncating division by three with an exact sign
    // branch, x=3*q+r and y=2*r+q. Enumerate witnesses independently of
    // elimination order; rotating/reflected columns also permute keep order.
    for (int64_t negative : {0, 1}) {
        auto source = System::create(5, {
            row({1, 0, 0, 0, 0}, negative ? -1 : 7),
            row({-1, 0, 0, 0, 0}, negative ? 7 : 0),
            row({0, 1, 0, 0, 0}, 8), row({0, -1, 0, 0, 0}, 8),
            row({0, 0, 1, 0, 0}, negative), row({0, 0, -1, 0, 0}, -negative),
            row({0, 0, 0, 1, 0}, 2), row({0, 0, 0, -1, 0}, 2),
            row({0, 0, 0, 0, 1}, 2), row({0, 0, 0, 0, -1}, 2),
            row({-1, 0, -2, 3, 0}, 0), row({1, 0, 2, -3, 0}, 2),
            row({1, 0, 0, -3, -1}, 0), row({-1, 0, 0, 3, 1}, 0),
            row({0, 1, 0, -1, -2}, 0), row({0, -1, 0, 1, 2}, 0)});
        if (failed(source) || !permutedProjectionChecks(*source, {1, 0},
            {{negative ? -7 : 0, negative ? -1 : 7}, {-8, 8}, {0, 1}, {-2, 2}, {-2, 2}}, checked)) {
            return false;
        }
    }
    // A signed four-bit wrap followed by a nonunit defining equality.
    // Odd wrapped values cannot satisfy y*2=r; projection must retain this
    // divisibility condition even after reordering auxiliary definitions.
    auto wrap = System::create(4, {
        row({1, 0, 0, 0}, 20), row({-1, 0, 0, 0}, 20),
        row({0, 1, 0, 0}, 4), row({0, -1, 0, 0}, 4),
        row({0, 0, 1, 0}, 7), row({0, 0, -1, 0}, 8),
        row({0, 0, 0, 1}, 2), row({0, 0, 0, -1}, 2),
        row({1, 0, -1, -16}, 0), row({-1, 0, 1, 16}, 0),
        row({0, 2, -1, 0}, 0), row({0, -2, 1, 0}, 0)});
    return succeeded(wrap) && permutedProjectionChecks(*wrap, {1, 0},
        {{-20, 20}, {-4, 4}, {-8, 7}, {-2, 2}}, checked);
}
bool congruenceCompatibilityChecks(uint64_t& checked)
{
    const std::vector<std::pair<bool, std::vector<Congruence>>> cases{
        {true, {congruence({1, 0}, 0, 8), congruence({1, 0}, 6, 64)}},
        {true, {congruence({1, 3}, 1, 4), congruence({5, 7}, 2, 8)}},
        {true, {congruence({-1, 3}, 1, 4), congruence({7, -1}, 2, 8)}},
        {true, {congruence({2, 6}, 2, 8), congruence({5, 7}, 6, 8)}},
        {true, {congruence({1, 1}, 0, 4), congruence({1, -1}, 1, 8)}},
        {false, {congruence({1, 3}, 1, 4), congruence({5, 7}, 5, 8)}},
        {false, {congruence({1, 0}, 0, 2), congruence({1, 0}, 1, 3)}},
        {false, {congruence({1, 1}, 0, 4), congruence({1, -1}, 2, 8)}}};
    for (const auto& [empty, atoms] : cases) {
        auto source = System::create(2, {row({1, 0}, 8), row({-1, 0}, 8),
                                        row({0, 1}, 8), row({0, -1}, 8)}, atoms);
        if (failed(source) || (empty && !source->isKnownEmpty())) { return false; }
        bool witnessed = false;
        for (int64_t x = -8; x <= 8; ++x) {
            for (int64_t y = -8; y <= 8; ++y) {
                const Point point{Integer(x), Integer(y)};
                bool expected = true;
                for (const auto& atom : atoms) {
                    expected &= (evaluate(atom.coefficients, point) - atom.residue) % atom.modulus == 0;
                }
                if (contains(*source, point) != expected) { return false; }
                witnessed |= expected;
                ++checked;
            }
        }
        if (witnessed == empty) { return false; }
    }
    Integer huge(1);
    for (unsigned bit = 0; bit < 130; ++bit) { huge *= 2; }
    const Congruence first{{Integer(1), huge}, Integer(7), Integer(2) * huge};
    const Congruence incompatible{{Integer(1), Integer(3) * huge}, Integer(8), Integer(4) * huge};
    const Congruence compatible{{Integer(1), Integer(3) * huge}, Integer(7), Integer(4) * huge};
    auto empty = System::create(2, {}, {first, incompatible});
    auto witnessed = System::create(2, {}, {first, compatible});
    checked += 2;
    return succeeded(empty) && empty->isKnownEmpty() && succeeded(witnessed) &&
        contains(*witnessed, {Integer(7), Integer(0)});
}
bool coupledCongruenceChecks(uint64_t& checked)
{
    const std::vector<std::pair<bool, std::vector<Congruence>>> cases{
        {true, {congruence({1, 0}, 0, 2), congruence({0, 1}, 0, 2), congruence({1, 1}, 1, 2)}},
        {false, {congruence({1, 0}, 0, 2), congruence({0, 1}, 0, 2), congruence({1, 1}, 0, 2)}},
        {true, {congruence({1, 0}, 1, 4), congruence({0, 1}, 2, 6), congruence({1, 1}, 4, 8)}},
        {false, {congruence({1, 0}, 1, 4), congruence({0, 1}, 2, 6), congruence({1, 1}, 3, 8)}},
        {true, {congruence({-1, 0}, -1, 4), congruence({0, -1}, -2, 6), congruence({-1, -1}, -4, 8)}}};
    for (const auto& [empty, atoms] : cases) {
        for (bool reverse : {false, true}) {
            auto ordered = atoms;
            if (reverse) { std::reverse(ordered.begin(), ordered.end()); }
            auto source = System::create(2, {row({1, 0}, 8), row({-1, 0}, 8),
                                            row({0, 1}, 8), row({0, -1}, 8)}, ordered);
            if (failed(source) || source->isEmpty() != empty ||
                !permutedProjectionChecks(*source, {1, 0}, {{-8, 8}, {-8, 8}}, checked)) { return false; }
        }
    }
    Integer huge(1);
    for (unsigned bit = 0; bit < 130; ++bit) { huge *= 2; }
    auto empty = System::create(2, {}, {{{Integer(1), Integer(0)}, Integer(0), huge},
        {{Integer(0), Integer(1)}, Integer(0), huge}, {{Integer(1), Integer(1)}, Integer(1), huge}});
    auto consistent = System::create(2, {}, {{{Integer(1), Integer(0)}, Integer(0), huge},
        {{Integer(0), Integer(1)}, Integer(0), huge}, {{Integer(1), Integer(1)}, Integer(0), huge}});
    // Consistent congruences must not hide an inequality contradiction.
    auto infeasible = System::create(2, {row({-1, 0}, 0), row({0, -1}, 0), row({1, 1}, -1)},
        {congruence({1, 0}, 0, 2), congruence({0, 1}, 0, 2), congruence({1, 1}, 0, 2)});
    auto zero = System::create(0, {}, {{{}, Integer(0), huge}});
    auto zeroEmpty = System::create(0, {}, {{{}, Integer(1), huge}});
    checked += 5;
    return succeeded(empty) && empty->isEmpty() && succeeded(consistent) && !consistent->isEmpty() &&
        succeeded(infeasible) && infeasible->isEmpty() && succeeded(zero) && !zero->isEmpty() &&
        succeeded(zeroEmpty) && zeroEmpty->isEmpty();
}
bool sharedSystemMemoChecks()
{
    auto source = System::create(2, {row({-1, 0}, 0), row({0, -1}, 0), row({1, 1}, -1)});
    if (failed(source) || source->isKnownEmpty()) { return false; }
    auto before = *source;
    auto solved = *source;
    if (!solved.isEmpty() || !solved.isKnownEmpty() || before.isKnownEmpty() || source->isKnownEmpty()) {
        return false;
    }
    auto moved = std::move(solved);
    solved = System{};
    return moved.isEmpty() && !solved.isEmpty() && solved.dimensions() == 0 &&
        before.isEmpty() && source->isEmpty();
}
std::optional<System> sharedSystemFixture()
{
    auto source = System::create(2, {row({1, 0}, 3), row({-1, 0}, 3),
        row({0, 1}, 3), row({0, -1}, 3)}, {congruence({1, 1}, 1, 2)});
    if (failed(source)) { return std::nullopt; }
    return *source;
}
bool sharedSystemOwnershipChecks(uint64_t& checked)
{
    auto retained = sharedSystemFixture();
    if (!retained) { return false; }
    // Grow and move the owning pool after the original creator has died.
    std::vector<System> pool;
    for (unsigned i = 0; i < 64; ++i) { pool.push_back(*retained); }
    retained.reset();
    auto swapped = pool.front().remap(2, {1, 0});
    auto restriction = System::create(2, {row({1, 0}, -4)});
    if (failed(restriction)) { return false; }
    auto restricted = pool.front().intersect(*restriction);
    auto projected = pool.front().project({0});
    if (failed(swapped) || failed(restricted) || !restricted->isEmpty() || failed(projected)) { return false; }
    for (int64_t x = -4; x <= 4; ++x) {
        for (int64_t y = -4; y <= 4; ++y) {
            const bool expected = x >= -3 && x <= 3 && y >= -3 && y <= 3 && (x + y) % 2 != 0;
            for (const auto& copy : pool) {
                if (contains(copy, {Integer(x), Integer(y)}) != expected || copy.isEmpty()) { return false; }
                ++checked;
            }
            if (contains(*swapped, {Integer(y), Integer(x)}) != expected) { return false; }
        }
        if (contains(*projected, {Integer(x)}) != (x >= -3 && x <= 3)) { return false; }
    }
    return sharedSystemMemoChecks();
}
bool equalityFeasibilityChecks(uint64_t& checked)
{
    Integer huge(1);
    for (unsigned bit = 0; bit < 100; ++bit) { huge *= 2; }
    // Enumerate signed points independently of elimination, including unit and
    // nonunit definitions, a floor remainder strip and coupled divisibility.
    for (const auto& factors : std::vector<Point>{{Integer(1), Integer(-3), Integer(2)},
             {Integer(2), Integer(3), Integer(-5)}, {huge, Integer(1), Integer(-2)}}) {
        for (int64_t constant : {-2, 0, 1}) {
            for (int64_t residue : {0, 1, 4}) {
                std::vector<Constraint> rows{row({1, 0, 0}, 3), row({-1, 0, 0}, 3),
                    row({0, 1, 0}, 3), row({0, -1, 0}, 3), row({0, 0, 1}, 3), row({0, 0, -1}, 3),
                    row({0, 2, -1}, 0), row({0, -2, 1}, 1)};
                rows.push_back({factors, Integer(constant)});
                auto opposite = factors;
                for (auto& factor : opposite) { factor = -factor; }
                rows.push_back({std::move(opposite), Integer(-constant)});
                const auto modular = congruence({1, 0, 2}, residue, 5);
                bool exists = false;
                for (int64_t x = -3; x <= 3; ++x) {
                    for (int64_t y = -3; y <= 3; ++y) {
                        for (int64_t z = -3; z <= 3; ++z) {
                            exists |= evaluate(factors, {Integer(x), Integer(y), Integer(z)}) == constant &&
                                2 * y <= z && z <= 2 * y + 1 && (x + 2 * z - residue) % 5 == 0 &&
                                (y + z - residue) % 3 == 0;
                            ++checked;
                        }
                    }
                }
                auto source = System::create(3, rows, {modular, congruence({0, 1, 1}, residue, 3)});
                if (failed(source)) { return false; }
                std::vector<unsigned> permutation{0, 1, 2};
                do {
                    auto reordered = source->remap(3, permutation);
                    fs::IntegerFeasibilityCost cost;
                    if (failed(reordered) || reordered->isEmpty(&cost) == exists || !cost.calls) { return false; }
                } while (std::next_permutation(permutation.begin(), permutation.end()));
            }
        }
    }
    // The inconsistent defining equality is resolved before any unrelated
    // coordinate, even with wide bounds. Count actual substitutions/recursion.
    for (unsigned dimensions : {4U, 16U, 64U}) {
        std::vector<Constraint> rows;
        for (unsigned column = 0; column < dimensions; ++column) {
            Point positive(dimensions, Integer(0));
            positive[column] = Integer(1);
            rows.push_back({positive, huge});
            positive[column] = Integer(-1);
            rows.push_back({positive, huge});
        }
        Point definition(dimensions, Integer(0));
        definition[dimensions - 2] = Integer(1);
        definition[dimensions - 1] = Integer(-2);
        rows.push_back({definition, Integer(0)});
        for (auto& factor : definition) { factor = -factor; }
        rows.push_back({definition, Integer(0)});
        Point parity(dimensions, Integer(0));
        parity[dimensions - 2] = Integer(1);
        Point other(dimensions, Integer(0));
        other[0] = Integer(1);
        auto source = System::create(dimensions, rows,
            {{parity, Integer(1), Integer(2)}, {other, Integer(0), Integer(3)}});
        fs::IntegerFeasibilityCost cost;
        if (failed(source) || !source->isEmpty(&cost) || cost.candidateSubstitutions != 1 ||
            cost.equalityEliminations != 1 || cost.calls != 2 || cost.octagonalEliminations ||
            cost.congruenceChecks || cost.congruenceMatrixColumns) { return false; }
        const auto hits = cost.cacheHits;
        if (!source->isEmpty(&cost) || cost.calls != 3 || cost.cacheHits != hits + 1 ||
            cost.candidateSubstitutions != 1) { return false; }
    }
    fs::IntegerFeasibilityCost zeroCost;
    auto empty = System::create(0, {row({}, -1)});
    return succeeded(empty) && empty->isEmpty(&zeroCost) && !System{}.isEmpty(&zeroCost) &&
        zeroCost.calls == 2 && !zeroCost.candidateSubstitutions;
}
bool compactCongruenceChecks(uint64_t& checked)
{
    // Two active coordinates retain the same residue lattice regardless of
    // interleaved unused or inequality-only coordinates. Enumerate its period.
    for (unsigned dimensions : {4U, 16U, 64U}) {
        const unsigned x = 1, y = dimensions - 2;
        for (int64_t sum : {0, 1}) {
            Point a(dimensions, Integer(0)), b = a, c = a, bound = a;
            a[x] = b[y] = c[x] = c[y] = bound[0] = Integer(1);
            std::vector<Constraint> rows;
            if (sum) { rows.push_back({bound, Integer(8)}); }
            auto source = System::create(dimensions, rows,
                {{a, Integer(0), Integer(2)}, {b, Integer(0), Integer(2)}, {c, Integer(sum), Integer(2)}});
            bool exists = false;
            for (int64_t first = 0; first < 2; ++first) {
                for (int64_t second = 0; second < 2; ++second) {
                    exists |= first % 2 == 0 && second % 2 == 0 && (first + second - sum) % 2 == 0;
                    ++checked;
                }
            }
            fs::IntegerFeasibilityCost cost;
            if (failed(source) || source->isEmpty(&cost) == exists || cost.congruenceChecks != 1 ||
                cost.congruenceMatrixColumns != 5) { return false; }
        }
    }
    // A nonoctagonal equality disappears first; the remaining three-way
    // contradiction needs HNF rather than pairwise residue normalization.
    auto residual = System::create(4, {row({1, 0, 0, -2}, 0), row({-1, 0, 0, 2}, 0)},
        {congruence({0, 1, 0, 0}, 0, 2), congruence({0, 0, 1, 0}, 0, 2),
         congruence({0, 1, 1, 0}, 1, 2)});
    fs::IntegerFeasibilityCost residualCost;
    if (failed(residual) || !residual->isEmpty(&residualCost) || residualCost.equalityEliminations != 1 ||
        residualCost.congruenceChecks != 1 || residualCost.congruenceMatrixColumns != 5) { return false; }
    // Consistent residues cannot override contradictory residual inequalities.
    auto inequalities = System::create(4,
        {row({1, 0, 0, -2}, 0), row({-1, 0, 0, 2}, 0), row({0, 0, 0, -1}, -1), row({1, 0, 0, 0}, 1)},
        {congruence({0, 1, 0, 0}, 0, 2), congruence({0, 0, 1, 0}, 1, 2)});
    return succeeded(inequalities) && inequalities->isEmpty();
}
bool unaryFeasibilityChecks(uint64_t& checked)
{
    // Enumerate signed bounded points independently, including negative
    // residues and nonunit coefficients. No candidate substitution is
    // needed, even when normalization alone cannot decide the intersection.
    for (int64_t lower = -5; lower <= 3; ++lower) {
        for (int64_t upper = lower; upper <= 5; ++upper) {
            for (int64_t residue = -3; residue <= 4; ++residue) {
                auto source = System::create(1, {row({-1}, -lower), row({1}, upper)},
                    {congruence({-3}, residue, 8), congruence({2}, 2, 6)});
                bool exists = false;
                for (int64_t x = lower; x <= upper; ++x) {
                    exists |= (-3 * x - residue) % 8 == 0 && (2 * x - 2) % 6 == 0;
                    ++checked;
                }
                fs::IntegerFeasibilityCost cost;
                if (failed(source) || source->isEmpty(&cost) == exists || cost.candidateSubstitutions ||
                    cost.congruenceChecks || cost.calls != 1) { return false; }
            }
        }
    }
    for (int64_t upper : {8, 9}) {
        auto source = System::create(1, {row({-1}, 0), row({1}, upper)},
            {congruence({1}, 1, 4), congruence({1}, 3, 6)});
        fs::IntegerFeasibilityCost cost;
        // The first nonnegative common residue is 9 modulo 12.
        if (failed(source) || source->isEmpty(&cost) != (upper == 8) || cost.unaryDecisions != 1 ||
            cost.candidateSubstitutions || cost.congruenceChecks) { return false; }
    }
    Integer huge(1);
    for (unsigned bit = 0; bit < 128; ++bit) { huge *= 2; }
    for (int64_t shift : {0, 1}) {
        const auto witness = -huge + 7;
        std::vector<Constraint> rows{{{Integer(-1)}, -witness}, {{Integer(1)}, witness + shift}};
        auto source = System::create(1, rows, {{{Integer(1)}, witness + 1, huge}});
        fs::IntegerFeasibilityCost cost;
        if (failed(source) || source->isEmpty(&cost) != (shift == 0) || cost.unaryDecisions != 1 ||
            cost.candidateSubstitutions || cost.congruenceChecks) { return false; }
    }
    for (bool lower : {false, true}) {
        auto source = System::create(1, {{{Integer(lower ? -1 : 1)}, -huge}},
            {{{Integer(1)}, Integer(7), huge}});
        fs::IntegerFeasibilityCost cost;
        if (failed(source) || source->isEmpty(&cost) || cost.unaryDecisions != 1 ||
            cost.candidateSubstitutions) { return false; }
    }
    return true;
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
bool latticeWitnessOracle(const System& source, int64_t lower, int64_t upper, uint64_t& checked)
{
    auto witnesses = source.eliminateWithWitness(0);
    if (failed(witnesses) || !boundedProjection(source, {1}, {{lower, upper}, {-8, 8}}, checked)) {
        return false;
    }
    for (int64_t y = -9; y <= 9; ++y) {
        bool expected = false, found = false;
        for (int64_t x = lower; x <= upper; ++x) {
            expected |= contains(source, {Integer(x), Integer(y)});
        }
        for (const auto& witness : *witnesses) {
            if (!contains(witness.domain, {Integer(y)})) { continue; }
            const auto numerator = evaluate(witness.numerator.coefficients, {Integer(y)}) +
                                   witness.numerator.constant;
            if (numerator % witness.denominator != 0 ||
                !contains(source, {numerator / witness.denominator, Integer(y)})) { return false; }
            found = true;
        }
        ++checked;
        if (found != expected) { return false; }
    }
    return true;
}
bool narrowStripChecks(uint64_t& checked)
{
    auto source = System::create(2,
        {row({1, 0}, 6), row({-1, 0}, 6), row({0, 1}, 20), row({0, -1}, 20),
         row({-2, 3}, -3), row({2, -3}, 5), row({-1, 2048}, 10000)},
        {congruence({1, 1}, -1, 5)});
    if (failed(source) || !boundedProjection(*source, {0}, {{-6, 6}, {-20, 20}}, checked)) {
        return false;
    }
    auto witnesses = source->eliminateWithWitness(1);
    if (failed(witnesses) || source->isEmpty()) { return false; }
    for (int64_t x = -6; x <= 6; ++x) {
        bool expected = false, actual = false;
        for (int64_t y = -20; y <= 20; ++y) { expected |= contains(*source, {Integer(x), Integer(y)}); }
        for (const auto& witness : *witnesses) {
            if (!contains(witness.domain, {Integer(x)})) { continue; }
            const auto numerator = evaluate(witness.numerator.coefficients, {Integer(x)}) +
                witness.numerator.constant;
            if (numerator % witness.denominator != 0 ||
                !contains(*source, {Integer(x), numerator / witness.denominator})) { return false; }
            actual = true;
        }
        auto fixed = System::create(2, {row({1, 0}, x), row({-1, 0}, -x)});
        auto restricted = source->intersect(*fixed);
        if (failed(restricted) || restricted->isEmpty() == expected || actual != expected) { return false; }
        ++checked;
    }
    return true;
}
bool unaryLatticeChecks(uint64_t& checked)
{
    const std::vector<std::vector<Congruence>> cases{
        {congruence({1, 0}, -2, 7)},
        {congruence({3, 0}, 2, 7)},
        {congruence({6, 0}, 4, 10)},
        {congruence({6, 0}, 3, 10)},
        {congruence({1, 0}, 1, 4), congruence({1, 0}, 3, 6)},
        {congruence({1, 0}, 0, 4), congruence({1, 0}, 1, 6)},
        {congruence({-3, 0}, -2, 7), congruence({1, 1}, 1, 3)},
        {congruence({1, 0}, 0, 1)}};
    for (const auto& congruences : cases) {
        auto source = System::create(2, {row({-1, 0}, 10), row({1, 0}, 8),
            row({1, -2}, 3), row({0, -1}, 8), row({0, 1}, 8)}, congruences);
        if (failed(source) || !latticeWitnessOracle(*source, -10, 8, checked)) { return false; }
    }
    // Actual normalization kernels use raw IVs in [0,4096) with steps 1024
    // and 2048. Projection must keep symbolic retained bounds and produce only
    // four/two candidate witnesses, rather than one candidate per byte step.
    for (int64_t step : {1024, 2048}) {
        auto source = System::create(2, {row({-1, 0}, 0), row({1, 0}, 4095),
            row({1, -1}, 0)}, {congruence({1, 0}, 0, step)});
        if (failed(source)) { return false; }
        auto witnesses = source->eliminateWithWitness(0);
        if (failed(witnesses) || witnesses->size() != static_cast<uint64_t>(4096 / step)) { return false; }
        for (std::size_t i = 0; i < witnesses->size(); ++i) {
            const auto& witness = (*witnesses)[i];
            if (witness.denominator != 1 || witness.numerator.coefficients != Point{Integer(0)} ||
                witness.numerator.constant != Integer(static_cast<int64_t>(i) * step)) { return false; }
        }
    }
    // A huge finite interval must retain the cheaper Cooper construction.
    Integer huge(1);
    for (unsigned i = 0; i < 200; ++i) { huge *= 2; }
    const auto origin = -huge + 7;
    auto shifted = System::create(1, {{{Integer(-1)}, -origin}, {{Integer(1)}, origin + 4095}},
                                  {{{Integer(1)}, origin, Integer(2048)}});
    if (failed(shifted)) { return false; }
    auto shiftedWitnesses = shifted->eliminateWithWitness(0);
    if (failed(shiftedWitnesses) || shiftedWitnesses->size() != 2 ||
        (*shiftedWitnesses)[0].numerator.constant != origin ||
        (*shiftedWitnesses)[1].numerator.constant != origin + 2048) { return false; }
    auto wide = System::create(1, {{{Integer(-1)}, huge}, {{Integer(1)}, huge}},
                               {congruence({1}, 1, 3)});
    if (failed(wide)) { return false; }
    auto witnesses = wide->eliminateWithWitness(0);
    return succeeded(witnesses) && witnesses->size() <= 3;
}
bool feasibilityEndpointChecks(uint64_t& checked)
{
    // Independent finite witnesses cover satisfiable and empty intersections,
    // signed affine endpoints, ties and modular restrictions.
    for (int64_t a = -2; a <= 2; ++a) {
        for (int64_t b = -2; b <= 2; ++b) {
            for (int64_t residue = 0; residue < 4; ++residue) {
                auto source = System::create(2, {row({-1, 0}, 6), row({1, 0}, 6),
                    row({0, -1}, 2), row({0, 1}, 2), row({-1, 2}, a), row({1, -3}, b)},
                    {congruence({1, 1}, residue, 4)});
                if (failed(source)) { return false; }
                bool exists = false;
                for (int64_t x = -6; x <= 6; ++x) {
                    for (int64_t y = -2; y <= 2; ++y) {
                        exists |= contains(*source, {Integer(x), Integer(y)}); ++checked;
                    }
                }
                if (source->isEmpty() == exists) { return false; }
            }
        }
    }
    return true;
}
bool wideFeasibilityEndpointChecks()
{
    // A loose machine bound precedes a much stronger affine endpoint. Both
    // signs check signed ranking arithmetic with an explicit feasible witness.
    Integer machine(1), parameter(1), wide(1);
    for (unsigned i = 0; i < 63; ++i) { machine *= 2; }
    for (unsigned i = 0; i < 31; ++i) { parameter *= 2; }
    for (unsigned i = 0; i < 72; ++i) { wide *= 2; }
    std::vector<Constraint> rows{row({-1, 0, 0}, 0), row({-1, 0, 256}, -1279),
        row({0, -1, 0}, 0), row({0, -1, 1}, 0), row({0, 0, -1}, 0),
        row({0, 0, 1}, 0), row({0, 1, -1}, -4), row({0, 1, 0}, 0),
        row({1, -256, 0}, 0), row({1, 0, -256}, 0), row({1, 0, 0}, 0)};
    rows[0].bound = machine; rows[10].bound = machine - 1;
    rows[2].bound = rows[4].bound = parameter;
    rows[5].bound = rows[7].bound = parameter - 1;
    rows[3].bound = machine * 2; rows[8].bound = rows[9].bound = wide - 1;
    std::vector<Congruence> congruences{congruence({0, 1, 3}, 0, 4),
                                      congruence({1, 0, 768}, 255, 1024)};
    for (int64_t sign : {1, -1}) {
        auto source = System::create(3, rows, congruences);
        if (failed(source) || !contains(*source, {Integer(sign * 1279), Integer(sign * -4), Integer(0)}) ||
            source->isEmpty()) { return false; }
        for (auto& atom : rows) { for (auto& coefficient : atom.coefficients) { coefficient = -coefficient; } }
        for (auto& atom : congruences) {
            for (auto& coefficient : atom.coefficients) { coefficient = -coefficient; }
        }
    }
    return true;
}
bool unaryResidueInclusionChecks()
{
    Integer huge(1);
    for (unsigned i = 0; i < 100; ++i) { huge *= 2; }
    auto point = System::create(2, {row({1, 0}, -7), row({-1, 0}, 7),
                                   row({0, 1}, 2), row({0, -1}, -2)});
    auto matching = System::create(2, {}, {{{Integer(-3), Integer(5)}, Integer(31), huge}});
    auto different = System::create(2, {}, {{{Integer(-3), Integer(5)}, Integer(32), huge}});
    if (failed(point) || failed(matching) || failed(different) ||
        !point->isSubsetOf(*matching) || point->isSubsetOf(*different)) { return false; }
    auto lattice = System::create(2, {}, {congruence({1, 0}, 1, 4096), congruence({0, 1}, 2, 2048)});
    auto sum = System::create(2, {}, {congruence({1, -1}, -1, 1024)});
    auto wrongSum = System::create(2, {}, {congruence({1, -1}, 0, 1024)});
    if (failed(lattice) || failed(sum) || failed(wrongSum) ||
        !lattice->isSubsetOf(*sum) || lattice->isSubsetOf(*wrongSum)) { return false; }
    // The bounds do not fix x, but their intersection with its grid does.
    auto singletonGrid = System::create(1, {row({-1}, 9), row({1}, -5)}, {congruence({1}, 1, 4)});
    auto singletonResidue = System::create(1, {}, {{{Integer(1)}, Integer(-7), huge}});
    if (failed(singletonGrid) || failed(singletonResidue) ||
        !singletonGrid->isSubsetOf(*singletonResidue)) { return false; }
    // Unary information cannot establish this coupled invariant: retain the
    // complete solver, including both its positive and negative answers.
    auto coupled = System::create(2, {row({1, -1}, 1), row({-1, 1}, -1)});
    auto coupledResidue = System::create(2, {}, {congruence({1, -1}, 1, 3)});
    auto coupledWrong = System::create(2, {}, {congruence({1, -1}, 0, 3)});
    auto empty = System::create(2, {row({0, 0}, -1)});
    return succeeded(coupled) && succeeded(coupledResidue) && succeeded(coupledWrong) && succeeded(empty) &&
           coupled->isSubsetOf(*coupledResidue) && !coupled->isSubsetOf(*coupledWrong) &&
           empty->isSubsetOf(*coupledWrong);
}
bool uncachedInclusionChecks()
{
    auto target = System::create(3, {}, {congruence({0, 0, 1}, 1, 2)});
    auto inequality = System::create(3, {row({1, 0, 0}, -10)});
    const bool targetsValid = succeeded(target) && succeeded(inequality);
    if (!targetsValid) { return false; }
    for (int64_t sum : {1, 2}) {
        auto source = System::create(3,
            {row({1, 1, 0}, sum), row({-1, -1, 0}, -sum), row({0, 0, 1}, 0), row({0, 0, -1}, 0)},
            {congruence({1, 0, 0}, 0, 2), congruence({0, 1, 0}, 0, 2)});
        const bool correctResidue = succeeded(source) && !source->isKnownEmpty() &&
            source->isSubsetOf(*target) == (sum == 1);
        if (!correctResidue) {
            return false;
        }
        auto fresh = System::create(3, source->constraints(), source->congruences());
        const bool correctInequality = succeeded(fresh) && !fresh->isKnownEmpty() &&
            fresh->isSubsetOf(*inequality) == (sum == 1);
        if (!correctInequality) {
            return false;
        }
    }
    auto source = System::create(1, {row({1}, 5)}, {congruence({1}, 1, 2)});
    auto weaker = System::create(1, {row({1}, 6)}, {congruence({1}, 1, 2)});
    return succeeded(source) && succeeded(weaker) && !source->isKnownEmpty() &&
        source->isSubsetOf(*source) && source->isSubsetOf(*weaker) && !source->isKnownEmpty();
}
bool seededDifferenceChecks(uint64_t& checked)
{
    auto seed = System::create(2, {row({1, 0}, 4), row({-1, 0}, 4), row({0, 1}, 4), row({0, -1}, 4)},
        {congruence({1, 0}, 1, 2)});
    auto restricted = System::create(2, {row({1, 0}, 5), row({0, 1}, 1)}, {congruence({1, 0}, 1, 2)});
    auto universal = System::create(2, {row({1, 0}, 5)}, {congruence({1, 0}, 1, 2)});
    const bool systemsValid = succeeded(seed) && succeeded(restricted) && succeeded(universal);
    if (!systemsValid) { return false; }
    for (const auto& blocker : {*restricted, *universal}) {
        auto difference = fs::subtractIntegerUnions(2, {*seed}, {blocker});
        if (failed(difference)) { return false; }
        for (int64_t x = -5; x <= 5; ++x) {
            for (int64_t y = -5; y <= 5; ++y) {
                const Point point{Integer(x), Integer(y)};
                ++checked;
                const bool expected = contains(*seed, point) && !contains(blocker, point);
                const bool matches = contains(*difference, point) == expected;
                if (!matches) {
                    return false;
                }
            }
        }
    }
    return true;
}
bool multiSeedDifferenceChecks(uint64_t& checked)
{
    auto left = System::create(2, {row({1, 0}, 1), row({-1, 0}, 4),
        row({0, 1}, 2), row({0, -1}, 2)}, {congruence({1, 0}, 1, 2)});
    auto right = System::create(2, {row({1, 0}, 4), row({-1, 0}, 1),
        row({0, 1}, 1), row({0, -1}, 3)}, {congruence({0, 1}, 0, 2)});
    auto empty = System::create(2, {row({0, 0}, -1)});
    auto diagonal = System::create(2, {row({1, 1}, 1)}, {congruence({0, 1}, 1, 2)});
    auto high = System::create(2, {row({-1, 0}, -3)});
    auto all = System::create(2, {});
    if (failed(left) || failed(right) || failed(empty) || failed(diagonal) || failed(high) || failed(all)) {
        return false;
    }
    // Distinct overlapping seeds, duplicate/empty seeds and seed-specific
    // congruences stress distribution and relative membership independently.
    const std::vector<System> sources{*left, *right, *empty, *left};
    for (const auto& blockers : std::vector<std::vector<System>>{
            {*diagonal, *high, *empty}, {*empty}, {*all}, {*diagonal, *diagonal}}) {
        auto difference = fs::subtractIntegerUnions(2, sources, blockers);
        auto reversedSources = sources;
        std::reverse(reversedSources.begin(), reversedSources.end());
        auto reversed = fs::subtractIntegerUnions(2, reversedSources, blockers);
        if (failed(difference) || failed(reversed)) { return false; }
        for (int64_t x = -6; x <= 6; ++x) {
            for (int64_t y = -6; y <= 6; ++y) {
                const Point point{Integer(x), Integer(y)};
                ++checked;
                const bool expected = contains(sources, point) && !contains(blockers, point);
                if (contains(*difference, point) != expected || contains(*reversed, point) != expected) {
                    return false;
                }
            }
        }
    }
    return true;
}
bool irrelevantArrangementAxes(uint64_t& checked)
{
    // A rejected conjunction contributes many middle axes; another live
    // conjunction still needs its final axis. The independent point oracle
    // must see the exact unsliced seed, including negative coordinates.
    auto seed = System::create(2, {row({1, 0}, -1), row({-1, 0}, 1),
        row({0, 1}, 2), row({0, -1}, 2)});
    auto last = System::create(2, {row({2, 1}, -2)});
    if (failed(seed) || failed(last)) { return false; }
    auto compare = [&](llvm::ArrayRef<System> sources, llvm::ArrayRef<System> blockers,
        const std::vector<System>& answer) {
        if (llvm::any_of(answer, [](const auto& piece) { return piece.isEmpty(); })) { return false; }
        for (int64_t x = -3; x <= 3; ++x) {
            for (int64_t y = -3; y <= 3; ++y) {
                const Point point{Integer(x), Integer(y)};
                ++checked;
                if (contains(answer, point) != (contains(sources, point) && !contains(blockers, point))) {
                    return false;
                }
            }
        }
        return true;
    };
    for (unsigned count : {8U, 32U, 128U}) {
        std::vector<Constraint> rows{row({-1, 0}, 0)};
        for (unsigned i = 1; i <= count; ++i) { rows.push_back(row({1, static_cast<int64_t>(i)}, 0)); }
        auto rejected = System::create(2, rows);
        if (failed(rejected)) { return false; }
        std::vector<System> blockers{*rejected, *last};
        for (unsigned order = 0; order < 2; ++order) {
            fs::IntegerDifferenceCost cost;
            auto answer = fs::subtractIntegerUnions(2, {*seed}, blockers, &cost);
            if (failed(answer) || !compare({*seed}, blockers, *answer) || cost.requests != 1 || cost.seeds != 1 ||
                cost.axes != count + 2 || cost.skippedAxes != count || cost.slices != 4 ||
                cost.outputPieces != 1 || !cost.sliceFeasibility.calls) { return false; }
            std::reverse(blockers.begin(), blockers.end());
        }
    }
    // The skipped residue axis may exceed machine width. No residue loop or
    // machine arithmetic is needed when its only conjunction is already false.
    Integer huge(1);
    for (unsigned bit = 0; bit < 130; ++bit) { huge *= 2; }
    auto rejected = System::create(2, {row({-1, 0}, 0)}, {{{Integer(1), Integer(1)}, Integer(0), huge}});
    auto modular = System::create(2, {}, {congruence({2, 1}, 1, 3)});
    if (failed(rejected) || failed(modular)) { return false; }
    fs::IntegerDifferenceCost cost;
    auto answer = fs::subtractIntegerUnions(2, {*seed}, {*rejected, *modular}, &cost);
    if (failed(answer) || !compare({*seed}, {*rejected, *modular}, *answer) || cost.skippedAxes != 1 ||
        cost.slices != 5 || cost.axes != 3) { return false; }
    // With both conjunctions live, a shared axis must still be partitioned.
    // Coupled facets and congruences share variables. Keep every axis needed
    // by a live clause regardless of the order of the disjunction.
    auto wide = System::create(2, {row({1, 0}, 2), row({-1, 0}, 2), row({0, 1}, 2), row({0, -1}, 2)});
    auto mixed = System::create(2, {row({-1, 0}, 0), row({1, 1}, 1)}, {congruence({1, -1}, -1, 3)});
    auto shared = System::create(2, {row({1, 1}, -1)}, {congruence({2, 1}, 1, 2)});
    if (failed(wide) || failed(mixed) || failed(shared)) { return false; }
    std::vector<System> blockers{*mixed, *shared, *last};
    for (unsigned order = 0; order < 3; ++order) {
        auto selected = fs::subtractIntegerUnions(2, {*wide, *seed}, blockers);
        if (failed(selected) || !compare({*wide, *seed}, blockers, *selected)) { return false; }
        std::rotate(blockers.begin(), blockers.begin() + 1, blockers.end());
    }
    return true;
}
bool arrangementMembershipChecks(uint64_t& checked)
{
    auto seed = System::create(2, {row({1, 0}, 5), row({-1, 0}, 5),
                                  row({0, 1}, 5), row({0, -1}, 5)});
    auto mixed = System::create(2, {row({1, 0}, 1), row({0, -1}, 2)},
        {congruence({1, 0}, 1, 2), congruence({1, 0}, 1, 3)});
    auto diagonal = System::create(2, {row({1, 1}, 2), row({-1, 0}, 1)},
        {congruence({-1, 0}, -2, 4), congruence({0, 1}, 1, 2)});
    auto low = System::create(2, {row({1, 0}, -3)});
    // This nonconstant contradictory conjunction must not contribute tests
    // after exact feasibility removes it from the active RHS axis list.
    auto empty = System::create(2, {row({1, -1}, -1), row({-1, 1}, -1)});
    auto all = System::create(2, {});
    if (failed(seed) || failed(mixed) || failed(diagonal) || failed(low) || failed(empty) || failed(all)) {
        return false;
    }
    for (auto blockers : std::vector<std::vector<System>>{
             {*mixed, *diagonal, *low, *empty}, {*mixed, *mixed}, {*empty}, {*all}, {}}) {
        for (unsigned order = 0; order < 2; ++order) {
            auto difference = fs::subtractIntegerUnions(2, {*seed}, blockers);
            if (failed(difference)) { return false; }
            for (int64_t x = -7; x <= 7; ++x) {
                for (int64_t y = -7; y <= 7; ++y) {
                    const Point point{Integer(x), Integer(y)};
                    ++checked;
                    const bool expected = contains(*seed, point) && !contains(blockers, point);
                    if (contains(*difference, point) != expected) { return false; }
                }
            }
            std::reverse(blockers.begin(), blockers.end());
        }
    }
    auto zero = System::create(0, {});
    auto zeroEmpty = System::create(0, {row({}, -1)});
    if (failed(zero) || failed(zeroEmpty)) { return false; }
    auto covered = fs::subtractIntegerUnions(0, {*zero}, {*zero});
    auto retained = fs::subtractIntegerUnions(0, {*zero}, {*zeroEmpty});
    return succeeded(covered) && covered->empty() && succeeded(retained) && contains(*retained, {});
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
#include "BoundedFeasibilityRankingChecks.h"
int runIntegerRelationChecks()
{
    uint64_t checked = 0;
    if (!boundedFeasibilityRankingChecks(checked)) {
        llvm::errs() << "integer bounded feasibility ranking failed\n"; return 1;
    }
    if (!projectionChecks(checked)) { llvm::errs() << "integer projection oracle differs\n"; return 1; }
    if (!projectionDefinitionChecks(checked)) {
        llvm::errs() << "integer projection definition ordering differs\n"; return 1;
    }
    if (!congruenceCompatibilityChecks(checked)) {
        llvm::errs() << "integer congruence compatibility differs\n"; return 1;
    }
    if (!coupledCongruenceChecks(checked)) {
        llvm::errs() << "integer coupled congruence consistency differs\n"; return 1;
    }
    if (!sharedSystemOwnershipChecks(checked)) {
        llvm::errs() << "integer immutable ownership or per-value memo isolation failed\n"; return 1;
    }
    if (!equalityFeasibilityChecks(checked)) {
        llvm::errs() << "integer equality feasibility ordering failed\n"; return 1;
    }
    if (!compactCongruenceChecks(checked)) {
        llvm::errs() << "integer compact congruence construction failed\n"; return 1;
    }
    if (!unaryFeasibilityChecks(checked) || !feasibilityAndWitnessChecks()) {
        llvm::errs() << "integer feasibility/witness check failed\n"; return 1;
    }
    if (!narrowStripChecks(checked)) { llvm::errs() << "integer narrow strip check failed\n"; return 1; }
    if (!unaryLatticeChecks(checked)) { llvm::errs() << "integer unary lattice check failed\n"; return 1; }
    if (!feasibilityEndpointChecks(checked) || !wideFeasibilityEndpointChecks()) {
        llvm::errs() << "integer feasibility endpoint ordering failed\n"; return 1;
    }
    if (!unaryResidueInclusionChecks()) { llvm::errs() << "integer unary residue inclusion failed\n"; return 1; }
    const bool uncached = uncachedInclusionChecks();
    if (!uncached) {
        llvm::errs() << "integer uncached inclusion failed\n"; return 1;
    }
    const bool seeded = seededDifferenceChecks(checked);
    if (!seeded) {
        llvm::errs() << "integer seeded difference failed\n"; return 1;
    }
    if (!multiSeedDifferenceChecks(checked)) { llvm::errs() << "integer multi-seed difference failed\n"; return 1; }
    if (!irrelevantArrangementAxes(checked)) { llvm::errs() << "irrelevant arrangement axes failed\n"; return 1; }
    if (!arrangementMembershipChecks(checked)) { llvm::errs() << "arrangement membership failed\n"; return 1; }
    if (!remapAndDifferenceChecks(checked)) { llvm::errs() << "integer remap/difference check failed\n"; return 1; }
    llvm::outs() << "integer relation exactness passed " << checked << " bounded points\n";
    return 0;
}
