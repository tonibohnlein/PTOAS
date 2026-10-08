// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact signed arithmetic adapters for relational regional queries.
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
struct RegionExpressions::IntegerRecipe {
    bool predicate = true;
    IntegerSystem system;
    IntegerAffine numerator;
    BoundInteger denominator{1};
    std::vector<Id> inputs;
    std::vector<uint64_t> residues;
    uint64_t period = 1, outputResidue = 0;
    std::string signature;
};
namespace {
BoundInteger powerOfTwo(unsigned bits)
{
    BoundInteger result(1);
    // Build in signed-representable chunks instead of one arbitrary-precision
    // multiplication per bit at every predicate/witness instantiation.
    while (bits != 0) {
        const auto chunk = std::min(bits, 62u);
        result *= BoundInteger(static_cast<int64_t>(uint64_t{1} << chunk));
        bits -= chunk;
    }
    return result;
}
BoundInteger affineMagnitude(const IntegerAffine& value)
{
    BoundInteger bound = llvm::abs(value.constant);
    const auto inputBound = powerOfTwo(63);
    for (const auto& coefficient : value.coefficients) { bound += llvm::abs(coefficient) * inputBound; }
    return bound;
}
APInt wideInteger(const BoundInteger& number)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << number;
    StringRef digits(text);
    bool negative = digits.consume_front("-");
    APInt value(128, digits, 10);
    return negative ? -value : value;
}
BoundInteger affineValue(const IntegerAffine& expression, ArrayRef<BoundInteger> inputs)
{
    auto result = expression.constant;
    for (auto [coefficient, value] : llvm::zip(expression.coefficients, inputs)) { result += coefficient * value; }
    return result;
}
} // namespace
bool RegionExpressions::equalInteger(const Node& a, const Node& b)
{
    if (!a.integer || !b.integer) { return a.integer == b.integer; }
    return a.integer->signature == b.integer->signature && a.integer->inputs == b.integer->inputs;
}
std::size_t RegionExpressions::hashInteger(const Node& node)
{
    if (!node.integer) { return 0; }
    return llvm::hash_combine(node.integer->signature,
        llvm::hash_combine_range(node.integer->inputs.begin(), node.integer->inputs.end()));
}
void RegionExpressions::appendOperands(const Node& node, SmallVectorImpl<Id>& operands) const
{
    for (Id operand : {node.a, node.b, node.c}) {
        if (operand != invalid) { operands.push_back(operand); }
    }
    if (node.integer) { llvm::append_range(operands, node.integer->inputs); }
}
std::optional<uint64_t> RegionExpressions::foldInteger(const IntegerRecipe& recipe) const
{
    SmallVector<BoundInteger> quotients;
    const BoundInteger period(static_cast<int64_t>(recipe.period));
    bool residueMatch = true;
    for (auto [input, residue] : llvm::zip(recipe.inputs, recipe.residues)) {
        auto bits = constantValue(input);
        if (!bits) { return std::nullopt; }
        BoundInteger original(APInt(64, *bits).getSExtValue());
        auto quotient = floorDiv(original, period);
        residueMatch &= original - quotient * period == BoundInteger(static_cast<int64_t>(residue));
        quotients.push_back(quotient);
    }
    if (!recipe.predicate) {
        auto quotient = floorDiv(affineValue(recipe.numerator, quotients), recipe.denominator);
        return wideInteger(quotient * period + BoundInteger(static_cast<int64_t>(recipe.outputResidue)))
            .trunc(64).getZExtValue();
    }
    if (!residueMatch) { return 0; }
    for (const auto& constraint : recipe.system.constraints()) {
        if (affineValue({constraint.coefficients, BoundInteger(0)}, quotients) > constraint.bound) { return 0; }
    }
    for (const auto& congruence : recipe.system.congruences()) {
        const auto value = affineValue({congruence.coefficients, BoundInteger(0)}, quotients);
        if (value - floorDiv(value, congruence.modulus) * congruence.modulus != congruence.residue) { return 0; }
    }
    return 1;
}
RegionExpressions::Id RegionExpressions::internInteger(std::shared_ptr<IntegerRecipe> recipe)
{
    if (auto folded = foldInteger(*recipe)) {
        return recipe->predicate ? boolean(*folded != 0) : constant(*folded);
    }
    Node node;
    node.kind = Kind::Integer;
    node.boolean = recipe->predicate;
    node.integer = std::move(recipe);
    return intern(std::move(node));
}
RegionExpressions::Id RegionExpressions::rebuildInteger(const Node& node, const llvm::DenseMap<Id, Id>& bindings)
{
    const auto& recipe = *node.integer;
    std::vector<Id> inputs(recipe.inputs);
    for (auto& input : inputs) { input = bindings.lookup(input); }
    if (inputs == recipe.inputs) { return intern(node); }
    return recipe.predicate ? integerPredicate(recipe.system, inputs, recipe.period, recipe.residues) :
        integerWitness(recipe.numerator, recipe.denominator, inputs,
                       recipe.period, recipe.residues, recipe.outputResidue);
}
RegionExpressions::Id RegionExpressions::integerPredicate(const IntegerSystem& system, ArrayRef<Id> inputs,
                                                         uint64_t period, ArrayRef<uint64_t> residues)
{
    if (!period || period > INT64_MAX || inputs.size() != residues.size() || system.dimensions() != inputs.size()) {
        return reject("integer predicate has inconsistent dimensions or period");
    }
    for (auto [input, residue] : llvm::zip(inputs, residues)) {
        if (!valid(input) || residue >= period) { return reject("integer predicate has invalid input or residue"); }
    }
    if (system.isKnownEmpty()) { return boolean(false); }
    if (period == 1 && system.constraints().empty() && system.congruences().empty()) { return boolean(true); }
    // Query composition can identify source and target coordinates. Preserve
    // that equality algebraically instead of emitting independent columns for
    // the same expression (including incompatible residue requirements).
    SmallVector<Id> unique;
    SmallVector<uint64_t> uniqueResidues;
    SmallVector<unsigned> columns;
    for (auto [input, residue] : llvm::zip(inputs, residues)) {
        auto found = llvm::find(unique, input);
        const unsigned column = std::distance(unique.begin(), found);
        if (found == unique.end()) { unique.push_back(input); uniqueResidues.push_back(residue); }
        else if (uniqueResidues[column] != residue) { return boolean(false); }
        columns.push_back(column);
    }
    if (unique.size() != inputs.size()) {
        auto identified = system.remap(unique.size(), columns);
        if (failed(identified)) { return reject("integer predicate coordinate identification failed"); }
        return integerPredicate(*identified, unique, period, uniqueResidues);
    }
    // Substitute constant original coordinates before constructing a circuit.
    // This is algebraic normalization, not projection or satisfiability search.
    std::vector<std::optional<BoundInteger>> constants(inputs.size());
    SmallVector<unsigned> kept;
    SmallVector<Id> remaining;
    SmallVector<uint64_t> remainingResidues;
    const BoundInteger modulus(static_cast<int64_t>(period));
    for (unsigned i = 0; i < inputs.size(); ++i) {
        if (auto bits = constantValue(inputs[i])) {
            BoundInteger original(APInt(64, *bits).getSExtValue());
            constants[i] = floorDiv(original, modulus);
            if (original - *constants[i] * modulus != BoundInteger(static_cast<int64_t>(residues[i]))) {
                return boolean(false);
            }
            continue;
        }
        bool used = period != 1;
        for (const auto& atom : system.constraints()) { used |= atom.coefficients[i] != 0; }
        for (const auto& atom : system.congruences()) { used |= atom.coefficients[i] != 0; }
        if (used) { kept.push_back(i); remaining.push_back(inputs[i]); remainingResidues.push_back(residues[i]); }
    }
    if (kept.size() != inputs.size()) {
        auto reduce = [&](const std::vector<BoundInteger>& coefficients, BoundInteger& right) {
            for (unsigned i = 0; i < constants.size(); ++i) {
                if (constants[i]) { right -= coefficients[i] * *constants[i]; }
            }
            std::vector<BoundInteger> reduced;
            for (auto i : kept) { reduced.push_back(coefficients[i]); }
            return reduced;
        };
        SmallVector<IntegerConstraint> constraints;
        SmallVector<IntegerCongruence> congruences;
        for (const auto& atom : system.constraints()) {
            auto bound = atom.bound;
            auto coefficients = reduce(atom.coefficients, bound);
            constraints.push_back({std::move(coefficients), bound});
        }
        for (const auto& atom : system.congruences()) {
            auto residue = atom.residue;
            auto coefficients = reduce(atom.coefficients, residue);
            congruences.push_back({std::move(coefficients), residue, atom.modulus});
        }
        auto reduced = IntegerSystem::create(kept.size(), constraints, congruences);
        if (failed(reduced)) { return reject("integer predicate constant substitution failed"); }
        return integerPredicate(*reduced, remaining, period, remainingResidues);
    }
    // Keep conjunctions visible to the shared Boolean DAG. Opaque whole-system
    // predicates otherwise hide identical tests in many boundary selectors.
    if (system.constraints().size() + system.congruences().size() > 1) {
        Id result = boolean(true);
        for (const auto& atom : system.constraints()) {
            auto single = IntegerSystem::create(inputs.size(), {atom});
            if (failed(single)) { return reject("integer predicate atom construction failed"); }
            result = land(result, integerPredicate(*single, inputs, period, residues));
        }
        for (const auto& atom : system.congruences()) {
            auto single = IntegerSystem::create(inputs.size(), {}, {atom});
            if (failed(single)) { return reject("integer congruence atom construction failed"); }
            result = land(result, integerPredicate(*single, inputs, period, residues));
        }
        return result;
    }
    // A scalar mathematical bound is a signed comparison, including bounds
    // outside the signed input domain. Do not form a potentially wrapping sum.
    if (period == 1 && inputs.size() == 1 && !isBoolean(inputs.front()) &&
        system.constraints().size() == 1 && system.congruences().empty()) {
        const auto& atom = system.constraints().front();
        const auto& coefficient = atom.coefficients.front();
        const BoundInteger minimum(INT64_MIN), maximum(INT64_MAX);
        if (coefficient > 0) {
            const auto upper = floorDiv(atom.bound, coefficient);
            if (upper < minimum) { return boolean(false); }
            if (upper >= maximum) { return boolean(true); }
            return sle(inputs.front(), constant(wideInteger(upper).trunc(64).getZExtValue()));
        }
        if (coefficient < 0) {
            const auto lower = -floorDiv(atom.bound, -coefficient);
            if (lower <= minimum) { return boolean(true); }
            if (lower > maximum) { return boolean(false); }
            return sle(constant(wideInteger(lower).trunc(64).getZExtValue()), inputs.front());
        }
    }
    const auto limit = powerOfTwo(126);
    for (const auto& atom : system.constraints()) {
        if (llvm::abs(atom.bound) >= limit || affineMagnitude({atom.coefficients, BoundInteger(0)}) >= limit) {
            return reject("integer predicate exceeds the proven i128 emission range");
        }
    }
    for (const auto& atom : system.congruences()) {
        if (atom.modulus <= 0 || atom.modulus >= limit || llvm::abs(atom.residue) >= limit ||
            affineMagnitude({atom.coefficients, BoundInteger(0)}) >= limit) {
            return reject("integer congruence exceeds the proven i128 emission range");
        }
    }
    auto recipe = std::make_shared<IntegerRecipe>();
    recipe->system = system;
    recipe->inputs.assign(inputs.begin(), inputs.end());
    recipe->residues.assign(residues.begin(), residues.end());
    recipe->period = period;
    {
        llvm::raw_string_ostream stream(recipe->signature);
        stream << "predicate:" << period << ':';
        for (auto residue : residues) { stream << residue << ','; }
        for (const auto& atom : system.constraints()) {
            stream << "[";
            for (const auto& coefficient : atom.coefficients) { stream << coefficient << ','; }
            stream << "<=" << atom.bound << ']';
        }
        for (const auto& atom : system.congruences()) {
            stream << "[";
            for (const auto& coefficient : atom.coefficients) { stream << coefficient << ','; }
            stream << "=" << atom.residue << '%' << atom.modulus << ']';
        }
    }
    return internInteger(std::move(recipe));
}
RegionExpressions::Id RegionExpressions::integerFloor(const IntegerAffine& numerator, const BoundInteger& denominator,
    ArrayRef<Id> inputs, uint64_t period, ArrayRef<uint64_t> residues)
{
    if (!period || period > INT64_MAX || inputs.size() != residues.size() ||
        numerator.coefficients.size() != inputs.size() || denominator <= 0) {
        return reject("integer floor has inconsistent dimensions, period or divisor");
    }
    SmallVector<Id> quotients;
    SmallVector<uint64_t> zeros(inputs.size(), 0);
    for (auto [input, residue] : llvm::zip(inputs, residues)) {
        if (!valid(input) || residue >= period) { return reject("integer floor has invalid input or residue"); }
        if (period == 1) { quotients.push_back(input); continue; }
        // Parameter quotient formation and final floor both use the existing
        // checked signed-i128 recipe. Output period one suppresses occurrence
        // reconstruction without changing any existing witness semantics.
        const IntegerAffine identity{{BoundInteger(1)}, BoundInteger(0)};
        quotients.push_back(integerWitness(identity, BoundInteger(static_cast<int64_t>(period)),
            {input}, 1, {0}, 0));
        if (quotients.back() == invalid) { return invalid; }
    }
    return integerWitness(numerator, denominator, quotients, 1, zeros, 0);
}
RegionExpressions::Id RegionExpressions::integerWitness(const IntegerAffine& numerator, const BoundInteger& denominator,
    ArrayRef<Id> inputs, uint64_t period, ArrayRef<uint64_t> residues, uint64_t outputResidue)
{
    if (!period || period > INT64_MAX || inputs.size() != residues.size() ||
        numerator.coefficients.size() != inputs.size() ||
        denominator <= 0 || outputResidue >= period) {
        return reject("integer witness has inconsistent dimensions, divisor or residues");
    }
    for (auto [input, residue] : llvm::zip(inputs, residues)) {
        if (!valid(input) || residue >= period) { return reject("integer witness has invalid input or residue"); }
    }
    IntegerAffine reduced{{}, numerator.constant};
    SmallVector<Id> remaining;
    SmallVector<uint64_t> remainingResidues;
    for (unsigned i = 0; i < inputs.size(); ++i) {
        const auto& coefficient = numerator.coefficients[i];
        if (coefficient == 0) { continue; }
        if (auto bits = constantValue(inputs[i])) {
            BoundInteger original(APInt(64, *bits).getSExtValue());
            reduced.constant += coefficient * floorDiv(original, BoundInteger(static_cast<int64_t>(period)));
        } else {
            reduced.coefficients.push_back(coefficient);
            remaining.push_back(inputs[i]); remainingResidues.push_back(residues[i]);
        }
    }
    if (remaining.size() != inputs.size()) {
        return integerWitness(reduced, denominator, remaining, period, remainingResidues, outputResidue);
    }
    // floor((g*a+c)/(g*d)) = floor((a+floor(c/g))/d).
    // This also merges different bytes selecting the same scalar occurrence.
    BoundInteger divisor = denominator;
    for (const auto& coefficient : numerator.coefficients) { divisor = gcd(divisor, llvm::abs(coefficient)); }
    if (divisor > 1) {
        IntegerAffine normalized = numerator;
        for (auto& coefficient : normalized.coefficients) { coefficient /= divisor; }
        normalized.constant = floorDiv(normalized.constant, divisor);
        return integerWitness(normalized, denominator/divisor, inputs, period, residues, outputResidue);
    }
    const auto magnitude = affineMagnitude(numerator), limit = powerOfTwo(126);
    // Floor division can increase absolute magnitude by less than one. This
    // bound covers every intermediate and final reconstruction, even outside
    // the domain where the witness denotes an actual occurrence.
    if (denominator >= limit || (magnitude + 1) * BoundInteger(static_cast<int64_t>(period)) +
        BoundInteger(static_cast<int64_t>(outputResidue)) >= limit) {
        return reject("integer witness exceeds the proven i128 emission range");
    }
    if (period == 1 && denominator == 1 && numerator.constant == 0) {
        Id identity = invalid;
        bool unit = true;
        for (auto [coefficient, input] : llvm::zip(numerator.coefficients, inputs)) {
            if (coefficient == 0) { continue; }
            if (coefficient != 1 || identity != invalid || isBoolean(input)) { unit = false; break; }
            identity = input;
        }
        if (unit && identity != invalid) { return identity; }
    }
    auto recipe = std::make_shared<IntegerRecipe>();
    recipe->predicate = false;
    recipe->numerator = numerator;
    recipe->denominator = denominator;
    recipe->inputs.assign(inputs.begin(), inputs.end());
    recipe->residues.assign(residues.begin(), residues.end());
    recipe->period = period;
    recipe->outputResidue = outputResidue;
    {
        llvm::raw_string_ostream stream(recipe->signature);
        stream << "witness:" << period << ':' << outputResidue << ':' << denominator << ':'
               << numerator.constant << ':';
        for (const auto& coefficient : numerator.coefficients) { stream << coefficient << ','; }
        for (auto residue : residues) { stream << residue << ','; }
    }
    return internInteger(std::move(recipe));
}
Value RegionExpressions::emitInteger(const Node& node, OpBuilder& builder, Location loc,
                                     const llvm::DenseMap<Id, Value>& memo) const
{
    const auto& recipe = *node.integer;
    Type wide = builder.getIntegerType(128);
    auto number = [&](const BoundInteger& value) -> Value {
        return builder.create<arith::ConstantOp>(loc, wide, IntegerAttr::get(wide, wideInteger(value)));
    };
    auto compare = [&](arith::CmpIPredicate predicate, Value a, Value b) -> Value {
        return builder.create<arith::CmpIOp>(loc, predicate, a, b);
    };
    Value period;
    if (recipe.period != 1) { period = number(BoundInteger(static_cast<int64_t>(recipe.period))); }
    Value guard;
    auto addGuard = [&](Value predicate) {
        guard = guard ? Value(builder.create<arith::AndIOp>(loc, guard, predicate)) : predicate;
    };
    SmallVector<Value> quotients;
    for (auto [id, residue] : llvm::zip(recipe.inputs, recipe.residues)) {
        Value input = memo.lookup(id);
        Value original = isBoolean(id) ? Value(builder.create<arith::ExtUIOp>(loc, wide, input)) :
            Value(builder.create<arith::IndexCastOp>(loc, wide, input));
        Value quotient = period ? Value(builder.create<arith::FloorDivSIOp>(loc, original, period)) : original;
        quotients.push_back(quotient);
        // Period one has only residue zero, checked at construction time.
        if (recipe.predicate && period) {
            Value multiple = builder.create<arith::MulIOp>(loc, quotient, period);
            Value remainder = builder.create<arith::SubIOp>(loc, original, multiple);
            addGuard(compare(arith::CmpIPredicate::eq, remainder, number(BoundInteger(static_cast<int64_t>(residue)))));
        }
    }
    auto affine = [&](const IntegerAffine& expression) -> Value {
        Value sum;
        if (expression.constant != 0) { sum = number(expression.constant); }
        for (auto [coefficient, input] : llvm::zip(expression.coefficients, quotients)) {
            if (coefficient == 0) { continue; }
            if (coefficient == -1) {
                if (!sum) { sum = number(BoundInteger(0)); }
                sum = builder.create<arith::SubIOp>(loc, sum, input);
                continue;
            }
            Value term = coefficient == 1 ? input :
                         Value(builder.create<arith::MulIOp>(loc, input, number(coefficient)));
            sum = sum ? Value(builder.create<arith::AddIOp>(loc, sum, term)) : term;
        }
        return sum ? sum : number(BoundInteger(0));
    };
    if (!recipe.predicate) {
        Value output = affine(recipe.numerator);
        if (recipe.denominator != 1) {
            output = builder.create<arith::FloorDivSIOp>(loc, output, number(recipe.denominator));
        }
        if (period) { output = builder.create<arith::MulIOp>(loc, output, period); }
        if (recipe.outputResidue != 0) {
            output = builder.create<arith::AddIOp>(loc, output,
                number(BoundInteger(static_cast<int64_t>(recipe.outputResidue))));
        }
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), output);
    }
    for (const auto& atom : recipe.system.constraints()) {
        Value lhs = affine({atom.coefficients, BoundInteger(0)});
        Value match = compare(arith::CmpIPredicate::sle, lhs, number(atom.bound));
        addGuard(match);
    }
    for (const auto& atom : recipe.system.congruences()) {
        Value lhs = affine({atom.coefficients, BoundInteger(0)});
        Value modulus = number(atom.modulus);
        Value quotient = builder.create<arith::FloorDivSIOp>(loc, lhs, modulus);
        Value multiple = builder.create<arith::MulIOp>(loc, quotient, modulus);
        Value remainder = builder.create<arith::SubIOp>(loc, lhs, multiple);
        Value match = compare(arith::CmpIPredicate::eq, remainder, number(atom.residue));
        addGuard(match);
    }
    return guard ? guard : Value(builder.create<arith::ConstantIntOp>(loc, 1, 1));
}
} // namespace mlir::pto::frontiersynch
