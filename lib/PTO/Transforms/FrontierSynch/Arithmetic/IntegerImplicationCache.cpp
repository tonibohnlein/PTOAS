// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "IntegerImplicationCache.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
#include <map>
#include <limits>
#include <set>
namespace mlir::pto::frontiersynch {
struct IntegerImplicationCache::State {
    explicit State(const std::vector<IntegerSystem>& systems) : uniqueSystems(systems) {}
    const std::vector<IntegerSystem>& uniqueSystems;
    Cost cost;
    std::map<std::size_t, std::unique_ptr<presburger::Simplex>> rationalSystems;
    using RationalBoundKey = std::pair<std::size_t, std::vector<BoundInteger>>;
    std::map<RationalBoundKey, presburger::MaybeOptimum<presburger::Fraction>> rationalBounds;
    static std::vector<BoundInteger> rationalRow(const IntegerConstraint& row)
    {
        std::vector<BoundInteger> coefficients;
        for (const auto& coefficient : row.coefficients) {
            coefficients.push_back(-coefficient);
        }
        coefficients.push_back(row.bound);
        return coefficients;
    }
    bool rationallySubsumed(std::size_t a, std::size_t b)
    {
        const auto& source = uniqueSystems[a];
        const auto& target = uniqueSystems[b];
        if (source.dimensions() != target.dimensions()) {
            return false;
        }
        // A rational relaxation cannot prove modular facts. Require those
        // separately before using rational implication for inequalities.
        for (const auto& row : target.congruences()) {
            if (!llvm::any_of(source.congruences(), [&](const auto& known) {
                    return known.coefficients == row.coefficients && known.modulus == row.modulus &&
                           known.residue == row.residue;
                })) {
                return false;
            }
        }
        auto& simplex = rationalSystems[a];
        if (!simplex) {
            simplex = std::make_unique<presburger::Simplex>(source.dimensions());
            for (const auto& row : source.constraints()) {
                simplex->addInequality(rationalRow(row));
            }
        }
        if (simplex->isEmpty()) {
            return true;
        }
        for (const auto& row : target.constraints()) {
            if (llvm::any_of(source.constraints(), [&](const auto& known) {
                    return known.coefficients == row.coefficients && known.bound <= row.bound;
                })) {
                continue;
            }
            const RationalBoundKey key{a, row.coefficients};
            auto found = rationalBounds.find(key);
            if (found == rationalBounds.end()) {
                auto objective = row.coefficients;
                objective.emplace_back(0);
                auto optimum = simplex->computeOptimum(presburger::Simplex::Direction::Up, objective);
                found = rationalBounds.emplace(key, std::move(optimum)).first;
            }
            const auto& optimum = found->second;
            // The target left side is integer-valued. A rational maximum
            // strictly below b+1 proves its integer bound b as well.
            if (!optimum.isBounded() || *optimum >= presburger::Fraction(row.bound + 1, BoundInteger(1))) {
                return false;
            }
        }
        return true;
    }
    struct LatticeRelaxation {
        std::vector<BoundInteger> shifts, scales;
        std::unique_ptr<presburger::Simplex> simplex;
    };
    std::map<std::size_t, LatticeRelaxation> latticeSystems;
    std::map<RationalBoundKey, presburger::MaybeOptimum<presburger::Fraction>> latticeBounds;
    static void increment(uint64_t& value)
    {
        if (value != std::numeric_limits<uint64_t>::max()) { ++value; }
    }
    LatticeRelaxation& latticeSystem(std::size_t id)
    {
        auto found = latticeSystems.find(id);
        if (found != latticeSystems.end()) { return found->second; }
        const auto& source = uniqueSystems[id];
        LatticeRelaxation relaxation;
        relaxation.shifts.assign(source.dimensions(), BoundInteger(0));
        relaxation.scales.assign(source.dimensions(), BoundInteger(1));
        bool selected = false;
        for (const auto& row : source.congruences()) {
            if (row.modulus <= 1) { continue; }
            std::optional<unsigned> coordinate;
            for (unsigned i = 0; i < source.dimensions(); ++i) {
                if (row.coefficients[i] == 0) { continue; }
                if (coordinate) { coordinate.reset(); break; }
                coordinate = i;
            }
            if (!coordinate || relaxation.scales[*coordinate] != 1) { continue; }
            const auto coefficient = row.coefficients[*coordinate];
            if (coefficient != 1 && coefficient != row.modulus - 1) { continue; }
            relaxation.scales[*coordinate] = row.modulus;
            relaxation.shifts[*coordinate] = mod(coefficient == 1 ? row.residue : -row.residue, row.modulus);
            selected = true;
        }
        if (selected) {
            increment(cost.latticeBuilds);
            relaxation.simplex = std::make_unique<presburger::Simplex>(source.dimensions());
            for (const auto& row : source.constraints()) {
                IntegerConstraint transformed{row.coefficients, row.bound};
                BoundInteger divisor(0);
                for (unsigned i = 0; i < source.dimensions(); ++i) {
                    transformed.bound -= row.coefficients[i] * relaxation.shifts[i];
                    transformed.coefficients[i] *= relaxation.scales[i];
                    const auto coefficient = transformed.coefficients[i];
                    divisor = gcd(divisor, coefficient < 0 ? -coefficient : coefficient);
                }
                // For integer q, c.q<=d is exactly equivalent after dividing
                // by gcd(c) and rounding d down, including negative bounds.
                // A zero row stays 0<=d and can prove the relaxation empty.
                if (divisor != 0) {
                    for (auto& coefficient : transformed.coefficients) { coefficient /= divisor; }
                    transformed.bound = floorDiv(transformed.bound, divisor);
                }
                relaxation.simplex->addInequality(rationalRow(transformed));
            }
        }
        return latticeSystems.emplace(id, std::move(relaxation)).first->second;
    }
    bool latticeSubsumed(std::size_t a, std::size_t b)
    {
        const auto& source = uniqueSystems[a];
        const auto& target = uniqueSystems[b];
        // The lattice LP proves inequalities only. Preserve the independent
        // proof obligation for every modular row required by the target.
        for (const auto& row : target.congruences()) {
            if (!llvm::any_of(source.congruences(), [&](const auto& known) {
                    return known.coefficients == row.coefficients && known.modulus == row.modulus &&
                           known.residue == row.residue;
                })) { return false; }
        }
        auto& relaxation = latticeSystem(a);
        if (!relaxation.simplex) { return false; }
        if (relaxation.simplex->isEmpty()) {
            increment(cost.latticeEmptyProofs); increment(cost.latticeProofs); return true;
        }
        for (const auto& row : target.constraints()) {
            const RationalBoundKey key{a, row.coefficients};
            increment(cost.latticeQueries);
            auto found = latticeBounds.find(key);
            if (found == latticeBounds.end()) {
                auto objective = row.coefficients;
                BoundInteger offset(0);
                for (unsigned i = 0; i < source.dimensions(); ++i) {
                    offset += row.coefficients[i] * relaxation.shifts[i];
                    objective[i] *= relaxation.scales[i];
                }
                objective.push_back(offset);
                auto optimum = relaxation.simplex->computeOptimum(presburger::Simplex::Direction::Up, objective);
                found = latticeBounds.emplace(key, std::move(optimum)).first;
            } else { increment(cost.latticeHits); }
            const auto& optimum = found->second;
            // This objective is the ORIGINAL integer expression t.x, with its
            // affine shift retained. Rational max < bound+1 is sufficient.
            if (!optimum.isBounded() || *optimum >= presburger::Fraction(row.bound + 1, BoundInteger(1))) {
                return false;
            }
        }
        increment(cost.latticeProofs); return true;
    }
    using IntegerPoint = std::vector<BoundInteger>;
    std::map<std::size_t, std::set<IntegerPoint>> integerSamples;
    static bool contains(const IntegerSystem& system, llvm::ArrayRef<BoundInteger> point)
    {
        if (point.size() != system.dimensions()) {
            return false;
        }
        auto dot = [&](llvm::ArrayRef<BoundInteger> coefficients) {
            BoundInteger value(0);
            for (unsigned i = 0; i < point.size(); ++i) {
                value += coefficients[i] * point[i];
            }
            return value;
        };
        for (const auto& row : system.constraints()) {
            if (dot(row.coefficients) > row.bound) {
                return false;
            }
        }
        for (const auto& row : system.congruences()) {
            if (mod(dot(row.coefficients) - row.residue, row.modulus) != 0) {
                return false;
            }
        }
        return true;
    }
    std::pair<BoundInteger, BoundInteger> latticeNeighbors(
        const IntegerSystem& source, unsigned coordinate, const presburger::Fraction& value)
    {
        BoundInteger residue(0), modulus(1);
        for (const auto& row : source.congruences()) {
            auto coefficient = row.coefficients[coordinate];
            if (coefficient != 1 && coefficient != row.modulus - 1) {
                continue;
            }
            bool unary = true;
            for (unsigned i = 0; i < source.dimensions(); ++i) {
                unary &= i == coordinate || row.coefficients[i] == 0;
            }
            if (!unary) {
                continue;
            }
            residue = coefficient == 1 ? row.residue : -row.residue;
            modulus = row.modulus;
            break;
        }
        const auto lower = floorDiv(value.num - residue * value.den, value.den * modulus) * modulus + residue;
        return {lower, lower * value.den == value.num ? lower : lower + modulus};
    }
    template <class Visitor>
    bool visitNearbyLatticePoints(
        const IntegerSystem& source, llvm::ArrayRef<presburger::Fraction> point, Visitor&& visit)
    {
        std::vector<std::pair<BoundInteger, BoundInteger>> choices;
        IntegerPoint candidate;
        for (unsigned i = 0; i < point.size(); ++i) {
            choices.push_back(latticeNeighbors(source, i, point[i]));
            candidate.push_back(choices.back().first);
        }
        if (visit(candidate)) {
            return true;
        }
        // Stream the finite combinations with O(dimensions) storage and stop
        // at the first verified counterexample. No shifted-size arithmetic.
        std::size_t depth = choices.size();
        while (depth) {
            --depth;
            if (candidate[depth] == choices[depth].second) {
                continue;
            }
            candidate[depth] = choices[depth].second;
            for (std::size_t i = depth + 1; i < choices.size(); ++i) {
                candidate[i] = choices[i].first;
            }
            if (visit(candidate)) {
                return true;
            }
            depth = choices.size();
        }
        return false;
    }
    bool counterexample(std::size_t a, std::size_t b, bool search)
    {
        const auto& source = uniqueSystems[a];
        const auto& target = uniqueSystems[b];
        auto& samples = integerSamples[a];
        for (const auto& point : samples) {
            if (!contains(target, point)) {
                return true;
            }
        }
        if (!search) {
            return false;
        }
        auto& simplex = rationalSystems[a];
        if (!simplex) {
            simplex = std::make_unique<presburger::Simplex>(source.dimensions());
            for (const auto& row : source.constraints()) {
                simplex->addInequality(rationalRow(row));
            }
        }
        if (simplex->isEmpty()) {
            return false;
        }
        auto verify = [&](presburger::Simplex& candidate) {
            if (candidate.isEmpty()) {
                return false;
            }
            auto point = candidate.getRationalSample();
            if (!point) {
                return false;
            }
            // These are guesses only. Every accepted point is checked against
            // the full original system, including all additional congruences.
            return visitNearbyLatticePoints(source, *point, [&](const IntegerPoint& guess) {
                if (!contains(source, guess)) {
                    return false;
                }
                samples.insert(guess);
                return !contains(target, guess);
            });
        };
        if (verify(*simplex)) {
            return true;
        }
        for (const auto& row : target.constraints()) {
            auto found = rationalBounds.find({a, row.coefficients});
            if (found == rationalBounds.end() ||
                (found->second.isBounded() && *found->second < presburger::Fraction(row.bound + 1, BoundInteger(1)))) {
                continue;
            }
            auto violation = row.coefficients;
            violation.push_back(-row.bound - 1);
            presburger::Simplex candidate(*simplex);
            candidate.addInequality(violation);
            if (verify(candidate)) {
                return true;
            }
        }
        return false;
    }
    llvm::DenseMap<std::pair<std::size_t, std::size_t>, bool> inclusions;
    llvm::DenseMap<std::pair<std::size_t, std::size_t>, bool> certificates;
    bool subsumed(std::size_t a, std::size_t b, bool certificateOnly = false)
    {
        if (uniqueSystems[a].dimensions() != uniqueSystems[b].dimensions()) {
            return false;
        }
        if (a == b) {
            return true;
        }
        const auto key = std::make_pair(a, b);
        // A negative certificate cache entry means unproved, while a negative
        // exact inclusion entry means disproved. Never mix these two contracts.
        if (certificateOnly) {
            if (auto found = inclusions.find(key); found != inclusions.end()) {
                ++cost.hits;
                return found->second;
            }
        }
        auto& answers = certificateOnly ? certificates : inclusions;
        if (auto found = answers.find(key); found != answers.end()) {
            ++cost.hits;
            return found->second;
        }
        if (!certificateOnly && counterexample(a, b, false)) {
            ++cost.witnessRefutations;
            inclusions.try_emplace(key, false);
            return false;
        }
        // Try sufficient row implication first, then cache the exact proof.
        // Keeping unions small avoids multiplying redundant projected pieces
        // across endpoint tags during later closure rounds.
        bool covered = true;
        {
            const auto& source = uniqueSystems[a];
            const auto& target = uniqueSystems[b];
            covered = source.dimensions() == target.dimensions();
            for (const auto& row : target.constraints()) {
                covered &= llvm::any_of(source.constraints(), [&](const auto& known) {
                    return known.coefficients == row.coefficients && known.bound <= row.bound;
                });
                if (!covered) {
                    break;
                }
            }
            for (const auto& row : target.congruences()) {
                covered &= llvm::any_of(source.congruences(), [&](const auto& known) {
                    return known.coefficients == row.coefficients && known.modulus == row.modulus &&
                           known.residue == row.residue;
                });
                if (!covered) {
                    break;
                }
            }
            if (!covered) {
                covered = rationallySubsumed(a, b);
                if (covered) {
                    ++cost.rationalProofs;
                }
                if (!covered) { covered = latticeSubsumed(a, b); }
                if (!covered && !certificateOnly) {
                    if (counterexample(a, b, true)) {
                        ++cost.witnessRefutations;
                    } else {
                        ++cost.exactChecks;
                        covered = source.isSubsetOf(target);
                    }
                }
            }
        }
        answers.try_emplace(key, covered);
        return covered;
    }
};
IntegerImplicationCache::IntegerImplicationCache(const std::vector<IntegerSystem>& systems)
    : state(std::make_unique<State>(systems))
{}
IntegerImplicationCache::~IntegerImplicationCache() = default;
bool IntegerImplicationCache::isSubsetOf(std::size_t source, std::size_t target)
{
    return state->subsumed(source, target);
}
bool IntegerImplicationCache::hasInclusionCertificate(std::size_t source, std::size_t target)
{
    return state->subsumed(source, target, true);
}
const IntegerImplicationCache::Cost& IntegerImplicationCache::cost() const { return state->cost; }
} // namespace mlir::pto::frontiersynch
