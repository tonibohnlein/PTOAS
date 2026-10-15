// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Typed expression adapters share endpoint ownership without changing selector mathematics.
#ifndef PTO_FRONTIER_ARITHMETICENDPOINTEXPRESSIONS_H
#define PTO_FRONTIER_ARITHMETICENDPOINTEXPRESSIONS_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch::detail {
inline bool checkedDifferenceConstant(const BoundInteger& value, std::string& error)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << value;
    StringRef digits(text);
    digits.consume_front("-");
    APInt magnitude;
    if (digits.getAsInteger(10, magnitude) || magnitude.getActiveBits() > 119) {
        error = "arithmetic selector constant exceeds the checked i128 emission range";
        return false;
    }
    return true;
}
inline RegionExpressions::Id endpointPredicate(RegionExpressions& arena, const IntegerSystem& domain,
    ArrayRef<RegionExpressions::Id> inputs, uint64_t period, ArrayRef<uint64_t> residues, std::string&)
{
    return arena.integerPredicate(domain, inputs, period, residues);
}
inline RegionExpressions::Id endpointPredicate(RegionExpressions& arena, const DifferenceBoundSystem& domain,
    ArrayRef<RegionExpressions::Id> inputs, uint64_t period, ArrayRef<uint64_t> residues, std::string& error)
{
    if (domain.dimensions() != inputs.size()) { return RegionExpressions::invalid; }
    SmallVector<IntegerConstraint> constraints;
    for (const auto& atom : domain.constraints()) {
        if (atom.lhs > inputs.size() || atom.rhs > inputs.size() ||
            !checkedDifferenceConstant(atom.bound, error)) { return RegionExpressions::invalid; }
        IntegerConstraint row{std::vector<BoundInteger>(inputs.size(), BoundInteger(0)), atom.bound};
        if (atom.lhs) { row.coefficients[atom.lhs - 1] += BoundInteger(1); }
        if (atom.rhs) { row.coefficients[atom.rhs - 1] -= BoundInteger(1); }
        constraints.push_back(std::move(row));
    }
    auto system = IntegerSystem::create(domain.dimensions(), constraints);
    return failed(system) ? RegionExpressions::invalid : arena.integerPredicate(*system, inputs, period, residues);
}
inline RegionExpressions::Id endpointCoordinate(RegionExpressions& arena,
    const GeneralArithmeticSelectorOutput& coordinate, ArrayRef<RegionExpressions::Id> inputs,
    uint64_t period, ArrayRef<uint64_t> residues, std::string&)
{
    return arena.integerWitness(coordinate.numerator, coordinate.denominator,
        inputs, period, residues, coordinate.residue);
}
inline uint64_t differenceCoordinateBits(const BoundInteger& value)
{
    // Reduce the arbitrary-precision reconstruction constant before conversion.
    const BoundInteger modulus = BoundInteger(INT64_MAX) * BoundInteger(2) + BoundInteger(2);
    BoundInteger reduced = value % modulus;
    if (reduced < 0) { reduced += modulus; }
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << reduced;
    return APInt(64, StringRef(text), 10).getZExtValue();
}
inline RegionExpressions::Id endpointCoordinate(RegionExpressions& arena,
    const ArithmeticSelectorOutput& coordinate, ArrayRef<RegionExpressions::Id> inputs,
    uint64_t period, ArrayRef<uint64_t> residues, std::string& error)
{
    using Id = RegionExpressions::Id;
    if (!period || period > INT64_MAX || coordinate.residue >= period || inputs.size() != residues.size() ||
        coordinate.lowerBounds.empty()) { return RegionExpressions::invalid; }
    const BoundInteger scale(static_cast<int64_t>(period)), residue(static_cast<int64_t>(coordinate.residue));
    // The dividend is negative; truncation toward zero gives its exact ceiling.
    const BoundInteger minimumQuotient = (BoundInteger(INT64_MIN) - residue) / scale;
    const auto minimum = arena.constant(static_cast<uint64_t>(INT64_MIN));
    Id selected = RegionExpressions::invalid;
    for (const auto& term : coordinate.lowerBounds) {
        if (term.input > inputs.size() || !checkedDifferenceConstant(term.offset, error)) {
            return RegionExpressions::invalid;
        }
        Id fits, value;
        const BoundInteger translated = scale * term.offset + residue;
        if (!term.input) {
            fits = arena.boolean(term.offset >= minimumQuotient);
            value = arena.constant(differenceCoordinateBits(translated));
        } else {
            const unsigned index = term.input - 1;
            if (residues[index] >= period) { return RegionExpressions::invalid; }
            // q + offset >= ceil((INT64_MIN - outputResidue) / period).
            auto lower = IntegerSystem::create(1, {{{BoundInteger(-1)}, term.offset - minimumQuotient}});
            if (failed(lower)) { return RegionExpressions::invalid; }
            fits = arena.integerPredicate(*lower, {inputs[index]}, period, {residues[index]});
            auto original = arena.isBoolean(inputs[index]) ?
                arena.select(inputs[index], arena.constant(1), arena.constant(0)) : inputs[index];
            value = arena.add(original, arena.constant(differenceCoordinateBits(
                translated - BoundInteger(static_cast<int64_t>(residues[index])))));
        }
        if (fits == RegionExpressions::invalid || value == RegionExpressions::invalid) {
            return RegionExpressions::invalid;
        }
        // On the exact selector domain the attained maximum is an actual,
        // representable source coordinate. Every candidate is <= that maximum.
        // Saturating only candidates below INT64_MIN therefore preserves it.
        // Residue compatibility makes the wrapping reconstruction exact for
        // surviving candidates. Off-domain arithmetic is total and unpublished.
        value = arena.select(fits, value, minimum);
        selected = selected == RegionExpressions::invalid ? value :
            arena.select(arena.sle(selected, value), value, selected);
    }
    return selected;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
