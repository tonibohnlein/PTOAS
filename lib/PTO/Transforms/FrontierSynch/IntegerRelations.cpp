// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Standard integer endpoint pairing and Cooper-style bound-candidate elimination.
// The arithmetic-tractability appendix supplies the fixed-dimension/coefficient
// contract. Congruence periods are class constants, never distances or trips.
#include "PTO/Transforms/FrontierSynch/IntegerRelations.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
using Coefficients = std::vector<BoundInteger>;
BoundInteger magnitude(const BoundInteger& value) { return value < 0 ? -value : value; }
BoundInteger commonMultiple(const BoundInteger& a, const BoundInteger& b)
{
    return (a / gcd(a, b)) * b;
}
Coefficients without(llvm::ArrayRef<BoundInteger> values, unsigned coordinate)
{
    Coefficients result;
    result.reserve(values.size() - 1);
    for (unsigned i = 0; i < values.size(); ++i) {
        if (i != coordinate) { result.push_back(values[i]); }
    }
    return result;
}
IntegerSystem validSystem(unsigned dimensions, llvm::ArrayRef<IntegerConstraint> constraints,
                          llvm::ArrayRef<IntegerCongruence> congruences = {})
{
    auto result = IntegerSystem::create(dimensions, constraints, congruences);
    // Public create() rejects malformed inputs. These rows are constructed
    // internally from a validated schema and strictly positive moduli.
    if (failed(result)) {
        llvm::report_fatal_error("integer relation invariant: internally constructed schema is invalid");
    }
    return *result;
}
bool normalize(IntegerConstraint& atom)
{
    BoundInteger divisor(0);
    for (const auto& coefficient : atom.coefficients) { divisor = gcd(divisor, magnitude(coefficient)); }
    if (divisor == 0) { return atom.bound >= 0; }
    for (auto& coefficient : atom.coefficients) { coefficient /= divisor; }
    atom.bound = floorDiv(atom.bound, divisor);
    return true;
}
bool normalize(IntegerCongruence& atom)
{
    BoundInteger divisor = atom.modulus;
    for (auto& coefficient : atom.coefficients) {
        coefficient = mod(coefficient, atom.modulus);
        divisor = gcd(divisor, coefficient);
    }
    atom.residue = mod(atom.residue, atom.modulus);
    if (mod(atom.residue, divisor) != 0) { return false; }
    for (auto& coefficient : atom.coefficients) { coefficient /= divisor; }
    atom.residue /= divisor;
    atom.modulus /= divisor;
    return true;
}
}
FailureOr<IntegerSystem> IntegerSystem::create(unsigned dimensions,
    llvm::ArrayRef<IntegerConstraint> constraints, llvm::ArrayRef<IntegerCongruence> congruences)
{
    if (dimensions > Coefficients().max_size()) { return failure(); }
    IntegerSystem result;
    result.dimensionCount = dimensions;
    std::map<Coefficients, BoundInteger> bounds;
    std::map<std::pair<Coefficients, BoundInteger>, BoundInteger> residues;
    for (auto atom : constraints) {
        if (atom.coefficients.size() != dimensions) { return failure(); }
        if (!normalize(atom)) { result.contradiction = true; continue; }
        if (std::all_of(atom.coefficients.begin(), atom.coefficients.end(), [](const auto& c) { return c == 0; })) {
            continue;
        }
        auto found = bounds.emplace(atom.coefficients, atom.bound);
        if (!found.second && atom.bound < found.first->second) { found.first->second = atom.bound; }
    }
    for (auto atom : congruences) {
        if (atom.coefficients.size() != dimensions || atom.modulus <= 0) { return failure(); }
        if (!normalize(atom)) { result.contradiction = true; continue; }
        if (atom.modulus == 1) { continue; }
        auto found = residues.emplace(std::make_pair(atom.coefficients, atom.modulus), atom.residue);
        if (!found.second && found.first->second != atom.residue) { result.contradiction = true; }
    }
    // Opposite affine bounds can contradict before any variable elimination.
    // Coefficients and constants are arbitrary-precision mathematical integers.
    for (const auto& [coefficients, bound] : bounds) {
        auto opposite = coefficients;
        for (auto& coefficient : opposite) { coefficient = -coefficient; }
        auto found = bounds.find(opposite);
        if (found != bounds.end() && bound + found->second < 0) {
            result.contradiction = true;
            break;
        }
    }
    if (result.contradiction) {
        result.inequalities.push_back({Coefficients(dimensions), BoundInteger(-1)});
        result.emptyCache = true;
        return result;
    }
    for (auto& [coefficients, bound] : bounds) { result.inequalities.push_back({coefficients, bound}); }
    for (auto& [key, residue] : residues) { result.divisibilities.push_back({key.first, residue, key.second}); }
    if (!dimensions) { result.emptyCache = false; }
    return result;
}
FailureOr<IntegerSystem> IntegerSystem::intersect(const IntegerSystem& other) const
{
    if (dimensionCount != other.dimensionCount) { return failure(); }
    auto rows = inequalities;
    rows.insert(rows.end(), other.inequalities.begin(), other.inequalities.end());
    auto congruences = divisibilities;
    congruences.insert(congruences.end(), other.divisibilities.begin(), other.divisibilities.end());
    return create(dimensionCount, rows, congruences);
}
FailureOr<IntegerSystem> IntegerSystem::remap(unsigned newDimensions, llvm::ArrayRef<unsigned> oldToNew) const
{
    if (oldToNew.size() != dimensionCount || newDimensions > Coefficients().max_size() ||
        std::any_of(oldToNew.begin(), oldToNew.end(), [&](unsigned i) { return i >= newDimensions; })) {
        return failure();
    }
    auto map = [&](llvm::ArrayRef<BoundInteger> old) {
        Coefficients result(newDimensions);
        for (unsigned i = 0; i < dimensionCount; ++i) { result[oldToNew[i]] += old[i]; }
        return result;
    };
    std::vector<IntegerConstraint> rows;
    std::vector<IntegerCongruence> congruences;
    for (const auto& atom : inequalities) { rows.push_back({map(atom.coefficients), atom.bound}); }
    for (const auto& atom : divisibilities) {
        congruences.push_back({map(atom.coefficients), atom.residue, atom.modulus});
    }
    return create(newDimensions, rows, congruences);
}
bool IntegerSystem::isOctagonal() const
{
    if (!divisibilities.empty()) { return false; }
    for (const auto& atom : inequalities) {
        unsigned count = 0;
        for (const auto& coefficient : atom.coefficients) {
            if (coefficient != 0 && (++count > 2 || magnitude(coefficient) != 1)) { return false; }
        }
    }
    return true;
}
namespace {
struct Bounds {
    BoundInteger scale{1};
    std::vector<IntegerAffine> lower, upper;
};
Bounds endpointBounds(llvm::ArrayRef<IntegerConstraint> rows, unsigned coordinate)
{
    Bounds result;
    for (const auto& atom : rows) {
        const auto coefficient = magnitude(atom.coefficients[coordinate]);
        if (coefficient != 0) { result.scale = commonMultiple(result.scale, coefficient); }
    }
    for (const auto& atom : rows) {
        const auto& coefficient = atom.coefficients[coordinate];
        if (coefficient == 0) { continue; }
        const auto factor = result.scale / magnitude(coefficient);
        auto coefficients = without(atom.coefficients, coordinate);
        for (auto& value : coefficients) { value *= coefficient > 0 ? -factor : factor; }
        IntegerAffine endpoint{std::move(coefficients), coefficient > 0 ? factor * atom.bound : -factor * atom.bound};
        (coefficient > 0 ? result.upper : result.lower).push_back(std::move(endpoint));
    }
    return result;
}
IntegerSystem substitute(const IntegerSystem& system, unsigned coordinate,
                         const IntegerAffine& numerator, const BoundInteger& denominator)
{
    const unsigned dimensions = system.dimensions() - 1;
    auto replace = [&](llvm::ArrayRef<BoundInteger> old) {
        auto coefficients = without(old, coordinate);
        for (unsigned i = 0; i < dimensions; ++i) {
            coefficients[i] = denominator * coefficients[i] + old[coordinate] * numerator.coefficients[i];
        }
        return coefficients;
    };
    std::vector<IntegerConstraint> rows;
    std::vector<IntegerCongruence> congruences;
    for (const auto& atom : system.constraints()) {
        rows.push_back({replace(atom.coefficients), denominator * atom.bound -
                       atom.coefficients[coordinate] * numerator.constant});
    }
    for (const auto& atom : system.congruences()) {
        congruences.push_back({replace(atom.coefficients), denominator * atom.residue -
                              atom.coefficients[coordinate] * numerator.constant, denominator * atom.modulus});
    }
    congruences.push_back({numerator.coefficients, -numerator.constant, denominator});
    return validSystem(dimensions, rows, congruences);
}
}
namespace {
// Projection collects every candidate. Feasibility can stop at its first
// satisfying candidate without allocating the rest of the residue family.
template<class Visitor>
bool visitEliminationCandidates(const IntegerSystem& system, unsigned coordinate, Visitor&& visit)
{
    // Opposite tight bounds define an equality. Solve it once, retaining the
    // divisibility condition in substitute(), instead of enumerating residues
    // that this equality already fixes. create() sorts normalized coefficients.
    const auto& rows = system.constraints();
    for (const auto& row : rows) {
        const auto coefficient = row.coefficients[coordinate];
        if (coefficient == 0) { continue; }
        auto opposite = row.coefficients;
        for (auto& value : opposite) { value = -value; }
        auto found = std::lower_bound(rows.begin(), rows.end(), opposite,
            [](const IntegerConstraint& atom, const Coefficients& key) { return atom.coefficients < key; });
        if (found == rows.end() || found->coefficients != opposite || row.bound + found->bound != 0) { continue; }
        const BoundInteger sign(coefficient > 0 ? 1 : -1);
        IntegerAffine numerator{without(row.coefficients, coordinate), sign * row.bound};
        for (auto& value : numerator.coefficients) { value *= -sign; }
        const auto denominator = magnitude(coefficient);
        auto domain = substitute(system, coordinate, numerator, denominator);
        return visit(std::move(domain), std::move(numerator), denominator);
    }
    const auto bounds = endpointBounds(system.constraints(), coordinate);
    BoundInteger period = bounds.scale;
    for (const auto& atom : system.congruences()) {
        const auto modulus = bounds.scale * atom.modulus;
        const auto localPeriod = modulus / gcd(magnitude(atom.coefficients[coordinate]), modulus);
        period = commonMultiple(period, localPeriod);
    }
    const bool fromLower = !bounds.lower.empty();
    auto endpoints = fromLower ? bounds.lower : bounds.upper;
    if (endpoints.empty()) { endpoints.push_back({Coefficients(system.dimensions() - 1), BoundInteger(0)}); }
    // Put t=scale*y. All inequalities on t have unit coefficients, and all
    // divisibilities are periodic in t with this period. Any nonempty bounded
    // interval has a satisfying t within one period above its maximal lower
    // endpoint (or below its minimal upper endpoint). Substitution checks every
    // other bound, so enumerating endpoint candidates needs no guessed maximum.
    for (const auto& endpoint : endpoints) {
        for (BoundInteger offset(0); offset < period; ++offset) {
            auto numerator = endpoint;
            numerator.constant += fromLower ? offset : -offset;
            auto domain = substitute(system, coordinate, numerator, bounds.scale);
            if (!visit(std::move(domain), std::move(numerator), bounds.scale)) { return false; }
        }
    }
    return true;
}
} // namespace
FailureOr<std::vector<IntegerEliminationWitness>> IntegerSystem::eliminateWithWitness(unsigned coordinate) const
{
    if (coordinate >= dimensionCount) { return failure(); }
    std::vector<IntegerEliminationWitness> result;
    if (contradiction) { return result; }
    visitEliminationCandidates(*this, coordinate,
        [&](IntegerSystem domain, IntegerAffine numerator, const BoundInteger& denominator) {
            if (!domain.contradiction) {
                result.push_back({std::move(domain), std::move(numerator), denominator});
            }
            return true;
        });
    return result;
}
FailureOr<std::vector<IntegerSystem>> IntegerSystem::eliminate(unsigned coordinate) const
{
    if (coordinate >= dimensionCount) { return failure(); }
    std::vector<IntegerSystem> result;
    if (contradiction) { return result; }
    if (isOctagonal()) {
        const auto bounds = endpointBounds(inequalities, coordinate);
        std::vector<IntegerConstraint> rows;
        for (const auto& atom : inequalities) {
            if (atom.coefficients[coordinate] == 0) {
                rows.push_back({without(atom.coefficients, coordinate), atom.bound});
            }
        }
        // Integral lower/upper endpoints admit an integer between them iff
        // every pair is ordered. Repeated coordinates produce ±2*x bounds;
        // create() performs exact gcd/floor unary normalization, not a rational
        // relaxation. With a missing bound side there are no paired constraints.
        for (const auto& lower : bounds.lower) {
            for (const auto& upper : bounds.upper) {
                auto coefficients = lower.coefficients;
                for (unsigned i = 0; i < coefficients.size(); ++i) { coefficients[i] -= upper.coefficients[i]; }
                rows.push_back({std::move(coefficients), upper.constant - lower.constant});
            }
        }
        auto projected = validSystem(dimensionCount - 1, rows);
        if (!projected.contradiction) { result.push_back(std::move(projected)); }
    } else {
        auto witnesses = eliminateWithWitness(coordinate);
        if (failed(witnesses)) { return failure(); }
        for (auto& witness : *witnesses) { result.push_back(std::move(witness.domain)); }
    }
    return result;
}
bool IntegerSystem::isEmpty() const
{
    if (emptyCache) { return *emptyCache; }
    if (!dimensionCount) { return contradiction; }
    // Every recursive call removes one coordinate. Unlike project(), this
    // search returns at the first witness and never builds the full union.
    // Existential feasibility is independent of elimination order. Prefer a
    // coordinate with fewer endpoint/residue candidates; eliminating an outer
    // affine row index first can otherwise enumerate its large physical stride.
    unsigned selected = 0;
    std::optional<BoundInteger> leastCost;
    for (unsigned coordinate = 0; coordinate < dimensionCount; ++coordinate) {
        BoundInteger scale(1);
        unsigned lower = 0, upper = 0;
        for (const auto& atom : inequalities) {
            const auto& coefficient = atom.coefficients[coordinate];
            if (coefficient == 0) { continue; }
            scale = commonMultiple(scale, magnitude(coefficient));
            if (coefficient < 0) { ++lower; } else { ++upper; }
        }
        BoundInteger period = scale;
        for (const auto& atom : divisibilities) {
            const auto modulus = scale * atom.modulus;
            period = commonMultiple(period,
                modulus / gcd(magnitude(atom.coefficients[coordinate]), modulus));
        }
        const auto count = std::max(1U, lower ? lower : upper);
        const auto cost = period * BoundInteger(static_cast<int64_t>(count));
        if (!leastCost || cost < *leastCost) { leastCost = cost; selected = coordinate; }
    }
    if (!isOctagonal()) {
        emptyCache = visitEliminationCandidates(*this, selected,
            [](const IntegerSystem& domain, const IntegerAffine&, const BoundInteger&) {
                return domain.isEmpty();
            });
        return *emptyCache;
    }
    auto children = eliminate(selected);
    // The zero-dimensional case was handled above; this cannot be an invalid
    // caller-supplied coordinate. An internal error must not become an answer.
    if (failed(children)) {
        llvm::report_fatal_error(
            "integer relation invariant: nonconstant schema cannot eliminate the selected coordinate");
    }
    emptyCache = std::all_of(children->begin(), children->end(), [](const auto& child) { return child.isEmpty(); });
    return *emptyCache;
}
FailureOr<std::vector<IntegerSystem>> IntegerSystem::project(llvm::ArrayRef<unsigned> keep) const
{
    std::vector<bool> retained(dimensionCount, false);
    for (unsigned coordinate : keep) {
        if (coordinate >= dimensionCount || retained[coordinate]) { return failure(); }
        retained[coordinate] = true;
    }
    std::vector<IntegerSystem> pieces{*this};
    for (unsigned coordinate = dimensionCount; coordinate > 0; --coordinate) {
        if (retained[coordinate - 1]) { continue; }
        std::vector<IntegerSystem> next;
        for (const auto& piece : pieces) {
            auto projected = piece.eliminate(coordinate - 1);
            if (failed(projected)) { return failure(); }
            for (auto& part : *projected) { next.push_back(std::move(part)); }
        }
        pieces = std::move(next);
    }
    std::vector<unsigned> ordered;
    for (unsigned coordinate = 0; coordinate < dimensionCount; ++coordinate) {
        if (retained[coordinate]) { ordered.push_back(coordinate); }
    }
    std::vector<unsigned> map(ordered.size());
    for (unsigned i = 0; i < keep.size(); ++i) {
        const auto position = std::lower_bound(ordered.begin(), ordered.end(), keep[i]) - ordered.begin();
        map[position] = i;
    }
    std::vector<IntegerSystem> result;
    for (const auto& piece : pieces) {
        auto reordered = piece.remap(static_cast<unsigned>(keep.size()), map);
        if (failed(reordered)) { return failure(); }
        if (!reordered->isEmpty()) { result.push_back(std::move(*reordered)); }
    }
    return result;
}
bool IntegerSystem::isSubsetOf(const IntegerSystem& other) const
{
    if (dimensionCount != other.dimensionCount) { return false; }
    if (isEmpty()) { return true; }
    for (const auto& atom : other.inequalities) {
        if (std::any_of(inequalities.begin(), inequalities.end(), [&](const auto& known) {
                return known.coefficients == atom.coefficients && known.bound <= atom.bound;
            })) { continue; }
        auto coefficients = atom.coefficients;
        for (auto& coefficient : coefficients) { coefficient = -coefficient; }
        const auto complement = validSystem(dimensionCount, {{coefficients, -atom.bound - 1}});
        auto overlap = intersect(complement);
        if (failed(overlap)) {
            llvm::report_fatal_error("integer relation invariant: validated inclusion schemas disagree");
        }
        if (!overlap->isEmpty()) { return false; }
    }
    for (const auto& atom : other.divisibilities) {
        if (std::any_of(divisibilities.begin(), divisibilities.end(), [&](const auto& known) {
                return known.coefficients == atom.coefficients && known.modulus == atom.modulus &&
                       known.residue == atom.residue;
            })) { continue; }
        for (BoundInteger residue(0); residue < atom.modulus; ++residue) {
            if (residue == atom.residue) { continue; }
            const auto complement = validSystem(dimensionCount, {}, {{atom.coefficients, residue, atom.modulus}});
            auto overlap = intersect(complement);
            if (failed(overlap)) {
                llvm::report_fatal_error("integer relation invariant: validated inclusion schemas disagree");
            }
            if (!overlap->isEmpty()) { return false; }
        }
    }
    return true;
}
bool IntegerSystem::operator==(const IntegerSystem& other) const
{
    return dimensionCount == other.dimensionCount && isSubsetOf(other) && other.isSubsetOf(*this);
}
namespace {
struct Axis {
    Coefficients coefficients;
    std::vector<BoundInteger> thresholds;
    BoundInteger modulus{0}; // Zero denotes an inequality-threshold axis.
};
std::vector<Axis> arrangementAxes(llvm::ArrayRef<IntegerSystem> lhs, llvm::ArrayRef<IntegerSystem> rhs)
{
    std::map<Coefficients, std::vector<BoundInteger>> thresholds;
    std::map<Coefficients, BoundInteger> moduli;
    for (auto pieces : {lhs, rhs}) {
        for (const auto& piece : pieces) {
            if (piece.isEmpty()) { continue; }
            for (const auto& atom : piece.constraints()) { thresholds[atom.coefficients].push_back(atom.bound); }
            for (const auto& atom : piece.congruences()) {
                auto found = moduli.emplace(atom.coefficients, atom.modulus);
                if (!found.second) { found.first->second = commonMultiple(found.first->second, atom.modulus); }
            }
        }
    }
    std::vector<Axis> axes;
    for (auto& [coefficients, values] : thresholds) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        axes.push_back({coefficients, std::move(values), BoundInteger(0)});
    }
    for (const auto& [coefficients, modulus] : moduli) { axes.push_back({coefficients, {}, modulus}); }
    return axes;
}
bool contained(const IntegerSystem& region, llvm::ArrayRef<IntegerSystem> pieces)
{
    return std::any_of(pieces.begin(), pieces.end(), [&](const auto& piece) { return region.isSubsetOf(piece); });
}
bool useful(const IntegerSystem& region, llvm::ArrayRef<IntegerSystem> lhs, llvm::ArrayRef<IntegerSystem> rhs)
{
    if (contained(region, rhs)) { return false; }
    for (const auto& piece : lhs) {
        auto overlap = region.intersect(piece);
        if (failed(overlap)) {
            llvm::report_fatal_error("integer relation invariant: validated arrangement schemas disagree");
        }
        if (!overlap->isEmpty()) { return true; }
    }
    return false;
}
IntegerSystem thresholdSlice(unsigned dimensions, const Axis& axis, std::size_t interval)
{
    std::vector<IntegerConstraint> rows;
    if (interval < axis.thresholds.size()) { rows.push_back({axis.coefficients, axis.thresholds[interval]}); }
    if (interval) {
        auto negative = axis.coefficients;
        for (auto& coefficient : negative) { coefficient = -coefficient; }
        rows.push_back({std::move(negative), -axis.thresholds[interval - 1] - 1});
    }
    return validSystem(dimensions, rows);
}
std::vector<IntegerSystem> arrange(unsigned dimensions, llvm::ArrayRef<IntegerSystem> lhs,
                                  llvm::ArrayRef<IntegerSystem> rhs, llvm::ArrayRef<Axis> axes,
                                  const IntegerSystem* seed = nullptr)
{
    struct Frame {
        IntegerSystem region;
        std::size_t axis = 0, interval = 0;
        BoundInteger residue{0};
    };
    std::vector<Frame> stack{{seed ? *seed : validSystem(dimensions, {}), 0, 0, BoundInteger(0)}};
    std::vector<IntegerSystem> result;
    while (!stack.empty()) {
        auto& current = stack.back();
        if (current.axis == axes.size()) {
            if ((seed || contained(current.region, lhs)) && !contained(current.region, rhs)) {
                result.push_back(std::move(current.region));
            }
            stack.pop_back();
            continue;
        }
        const auto& axis = axes[current.axis];
        IntegerSystem slice;
        if (axis.modulus != 0) {
            if (current.residue >= axis.modulus) { stack.pop_back(); continue; }
            slice = validSystem(dimensions, {}, {{axis.coefficients, current.residue, axis.modulus}});
            ++current.residue;
        } else {
            if (current.interval > axis.thresholds.size()) { stack.pop_back(); continue; }
            slice = thresholdSlice(dimensions, axis, current.interval++);
        }
        auto next = current.region.intersect(slice);
        if (failed(next)) {
            llvm::report_fatal_error("integer relation invariant: validated arrangement slice has an invalid schema");
        }
        if (!next->isEmpty() && (seed ? !contained(*next, rhs) : useful(*next, lhs, rhs))) {
            stack.push_back({std::move(*next), current.axis + 1, 0, BoundInteger(0)});
        }
    }
    return result;
}
}
FailureOr<std::vector<IntegerSystem>> subtractIntegerUnions(unsigned dimensions,
    llvm::ArrayRef<IntegerSystem> lhs, llvm::ArrayRef<IntegerSystem> rhs)
{
    if (dimensions > Coefficients().max_size()) { return failure(); }
    for (auto pieces : {lhs, rhs}) {
        if (std::any_of(pieces.begin(), pieces.end(),
                        [&](const auto& piece) { return piece.dimensions() != dimensions; })) {
            return failure();
        }
    }
    if (std::all_of(lhs.begin(), lhs.end(), [](const auto& piece) { return piece.isEmpty(); })) {
        return std::vector<IntegerSystem>{};
    }
    if (std::all_of(rhs.begin(), rhs.end(), [](const auto& piece) { return piece.isEmpty(); })) {
        return std::vector<IntegerSystem>(lhs.begin(), lhs.end());
    }
    if (std::all_of(lhs.begin(), lhs.end(), [&](const auto& piece) { return contained(piece, rhs); })) {
        return std::vector<IntegerSystem>{};
    }
    if (lhs.size() == 1) {
        // Every descendant already belongs to this conjunction. Partition only
        // right-hand membership, retaining exact integer emptiness checks.
        // This avoids exploring the universe outside a selector's domain.
        const auto axes = arrangementAxes({}, rhs);
        return arrange(dimensions, lhs, rhs, axes, &lhs.front());
    }
    const auto axes = arrangementAxes(lhs, rhs);
    return arrange(dimensions, lhs, rhs, axes);
}
} // namespace mlir::pto::frontiersynch
