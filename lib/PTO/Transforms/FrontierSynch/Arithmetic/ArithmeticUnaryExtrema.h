// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_ARITHMETICUNARYEXTREMA_H
#define PTO_FRONTIER_ARITHMETICUNARYEXTREMA_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/AllocationProfile.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#include <map>
namespace mlir::pto::frontiersynch::detail {
namespace unaryExtrema {
struct Work {
    uint64_t inspected = 0, boundRows = 0, witnesses = 0, candidates = 0;
    uint64_t comparisons = 0, coveredPieces = 0, outputs = 0;
    uint64_t requests = 0, qualified = 0, selected = 0;
    uint64_t parameterDifferences = 0, parameterSlices = 0, parameterFeasibility = 0;
    void add(const Work& other)
    {
        accumulateCost(requests, other.requests); accumulateCost(qualified, other.qualified);
        accumulateCost(selected, other.selected); accumulateCost(inspected, other.inspected);
        accumulateCost(boundRows, other.boundRows); accumulateCost(witnesses, other.witnesses);
        accumulateCost(candidates, other.candidates); accumulateCost(comparisons, other.comparisons);
        accumulateCost(coveredPieces, other.coveredPieces); accumulateCost(outputs, other.outputs);
        accumulateCost(parameterDifferences, other.parameterDifferences);
        accumulateCost(parameterSlices, other.parameterSlices);
        accumulateCost(parameterFeasibility, other.parameterFeasibility);
    }
};
inline bool qualify(const GeneralArithmeticRelation& support, uint64_t period,
                    unsigned parameters, bool first, Work& work)
{
    if (!period || period > INT64_MAX || parameters == UINT_MAX) { return false; }
    std::map<std::vector<uint64_t>, std::size_t> sites;
    for (const auto& [key, pieces] : support) {
        if (key.source.residues.size() != 1 || !key.target.residues.empty() ||
            key.parameterResidues.size() != parameters || key.source.residues.front() >= period ||
            llvm::any_of(key.parameterResidues, [&](auto residue) { return residue >= period; })) { return false; }
        // Preserve order(support)'s mixed-site refusal, including empty keys.
        auto [entry, inserted] = sites.emplace(key.parameterResidues, key.source.site);
        if (!inserted && entry->second != key.source.site) { return false; }
        for (const auto& piece : pieces) {
            accumulateCost(work.inspected, 1);
            if (piece.dimensions() != parameters + 1) { return false; }
            if (piece.isKnownEmpty()) { continue; }
            // Column zero is exactly the endpoint quotient. Parameters are
            // never eliminated or mistaken for a bound on that endpoint.
            const bool bounded = llvm::any_of(piece.constraints(), [&](const auto& row) {
                accumulateCost(work.boundRows, 1);
                return first ? row.coefficients[0] < BoundInteger(0) : row.coefficients[0] > BoundInteger(0);
            });
            if (!bounded) { return false; }
        }
    }
    return true;
}
inline std::optional<GeneralArithmeticEndpointSelector> candidates(
    const GeneralArithmeticRelation& support, unsigned parameters, bool first,
    ArithmeticAnalysisCost& cost, AllocationProfile* profile, Work& work)
{
    GeneralArithmeticEndpointSelector pool;
    pool.parameterCount = parameters;
    std::vector<unsigned> permutation(parameters + 1);
    permutation[0] = parameters;
    for (unsigned i = 0; i < parameters; ++i) { permutation[i + 1] = i; }
    for (const auto& [key, pieces] : support) {
        for (const auto& piece : pieces) {
            if (piece.isKnownEmpty()) { continue; }
            auto domain = piece.remap(parameters + 1, permutation);
            if (failed(domain)) { return {}; }
            if (!first) {
                auto rows = domain->constraints();
                auto congruences = domain->congruences();
                for (auto& row : rows) { row.coefficients[parameters] = -row.coefficients[parameters]; }
                for (auto& row : congruences) { row.coefficients[parameters] = -row.coefficients[parameters]; }
                domain = IntegerSystem::create(parameters + 1, rows, congruences);
                if (failed(domain)) { return {}; }
            }
            // Existing elimination candidates contain the minimum for each
            // fixed parameter valuation: equality gives its unique witness;
            // finite unary/strip enumeration covers every possible value;
            // otherwise Cooper enumerates a full period above each lower
            // endpoint, including the least feasible value above the maximum.
            // Keep EVERY candidate, never treat stable first match as minimum.
            accumulateCost(work.witnesses, 1); accumulateCost(cost.projections, 1);
            AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticProjection);
            auto tuples = buildIntegerTupleWitnesses(*domain, parameters, key.source.residues);
            if (failed(tuples)) { return {}; }
            for (auto& tuple : *tuples) {
                if (tuple.outputs.size() != 1 || tuple.outputs.front().denominator <= BoundInteger(0)) { return {}; }
                auto& value = tuple.outputs.front();
                if (!first) {
                    value.numerator.constant = -value.numerator.constant;
                    for (auto& coefficient : value.numerator.coefficients) { coefficient = -coefficient; }
                }
                accumulateCost(work.candidates, 1);
                pool.pieces.push_back({std::move(tuple.domain), {}, key.parameterResidues,
                                      key.source.site, std::move(tuple.outputs)});
            }
        }
    }
    return pool;
}
inline FailureOr<IntegerSystem> strictBetter(const GeneralArithmeticSelectorOutput& a,
    const GeneralArithmeticSelectorOutput& b, const BoundInteger& scale, unsigned parameters, bool first)
{
    const auto divisor = a.denominator * b.denominator;
    const BoundInteger direction(first ? 1 : -1);
    // Both quotients are integral on the joined domain. Compare full
    // original coordinates P*N/d+r, including different residues:
    // direction*(value_b-value_a) <= -1. Positive denominators make
    // this cross multiplication exact, including negative values.
    IntegerConstraint row{std::vector<BoundInteger>(parameters), -divisor};
    row.bound -= direction * (scale * (b.numerator.constant * a.denominator -
        a.numerator.constant * b.denominator) +
        (BoundInteger(static_cast<int64_t>(b.residue)) -
         BoundInteger(static_cast<int64_t>(a.residue))) * divisor);
    for (unsigned i = 0; i < parameters; ++i) {
        row.coefficients[i] = direction * scale * (b.numerator.coefficients[i] * a.denominator -
            a.numerator.coefficients[i] * b.denominator);
    }
    return IntegerSystem::create(parameters, {row});
}
inline std::optional<GeneralArithmeticEndpointSelector> select(
    const GeneralArithmeticEndpointSelector& pool, uint64_t period, unsigned parameters, bool first,
    ArithmeticAnalysisCost& cost, AllocationProfile* profile, Work& work)
{
    GeneralArithmeticEndpointSelector result;
    result.parameterCount = parameters;
    const BoundInteger scale(static_cast<int64_t>(period));
    for (const auto& candidate : pool.pieces) {
        const auto& a = candidate.outputs.front();
        std::vector<IntegerSystem> better;
        for (const auto& competitor : pool.pieces) {
            if (&candidate == &competitor || candidate.parameterResidues != competitor.parameterResidues) { continue; }
            accumulateCost(work.comparisons, 1);
            const auto& b = competitor.outputs.front();
            auto strict = strictBetter(a, b, scale, parameters, first);
            if (failed(strict)) { return {}; }
            auto joined = competitor.domain.intersect(*strict);
            accumulateCost(cost.pieceJoins, 1);
            if (failed(joined)) { return {}; }
            if (!joined->isKnownEmpty()) {
                accumulateCost(work.coveredPieces, 1);
                better.push_back(std::move(*joined));
            }
        }
        AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticDifference);
        accumulateCost(cost.differences, 1);
        auto remaining = subtractIntegerUnions(parameters, {candidate.domain}, better, &cost.integerDifference);
        if (failed(remaining)) { return {}; }
        for (auto& domain : *remaining) {
            accumulateCost(work.outputs, 1);
            result.pieces.push_back({std::move(domain), {}, candidate.parameterResidues,
                                    candidate.outputSite, candidate.outputs});
        }
    }
    return result;
}
inline void report(const Work& work, bool first, AllocationProfile* profile)
{
    if (!profile || !profile->enabled()) { return; }
    profile->recordWork("arithmetic-unary-extrema", {{"first", first}, {"qualified", work.qualified},
        {"selected", work.selected}, {"inspected-pieces", work.inspected}, {"witness-requests", work.witnesses},
        {"bound-row-visits", work.boundRows}, {"candidate-witnesses", work.candidates},
        {"candidate-comparisons", work.comparisons}, {"dominating-pieces", work.coveredPieces},
        {"selected-pieces", work.outputs},
        {"parameter-differences", work.parameterDifferences}, {"parameter-slices", work.parameterSlices},
        {"parameter-feasibility-calls", work.parameterFeasibility}});
}
} // namespace unaryExtrema
// Optional exact construction, never an admission condition. The original
// order/project/difference route remains authoritative when this returns none.
inline std::optional<GeneralArithmeticEndpointSelector> unaryHandoffExtremum(
    const GeneralArithmeticRelation& support, uint64_t period, unsigned parameters, bool first,
    ArithmeticAnalysisCost& cost, AllocationProfile* profile = nullptr, unaryExtrema::Work* accumulated = nullptr)
{
    unaryExtrema::Work work;
    work.requests = 1;
    const auto initialDifference = cost.integerDifference;
    auto construct = [&]() -> std::optional<GeneralArithmeticEndpointSelector> {
        work.qualified = unaryExtrema::qualify(support, period, parameters, first, work);
        if (!work.qualified) { return {}; }
        auto pool = unaryExtrema::candidates(support, parameters, first, cost, profile, work);
        if (!pool) { return {}; }
        return unaryExtrema::select(*pool, period, parameters, first, cost, profile, work);
    };
    auto result = construct();
    work.selected = result.has_value();
    work.parameterDifferences = cost.integerDifference.requests - initialDifference.requests;
    work.parameterSlices = cost.integerDifference.slices - initialDifference.slices;
    work.parameterFeasibility =
        cost.integerDifference.sliceFeasibility.calls - initialDifference.sliceFeasibility.calls;
    if (accumulated) { accumulated->add(work); }
    else { unaryExtrema::report(work, first, profile); }
    return result;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
