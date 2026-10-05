// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticSelectors.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using GroupKey = std::pair<std::size_t, uint32_t>;
using GroupIndex = std::map<GroupKey, std::size_t>;
struct Tuple {
    IntegerSystem domain;
    std::vector<GeneralArithmeticSelectorOutput> outputs;
};
BoundInteger natural(uint64_t value)
{
    return BoundInteger(static_cast<int64_t>(value >> 1)) * BoundInteger(2) +
           BoundInteger(static_cast<int64_t>(value & 1));
}
bool validResidues(llvm::ArrayRef<uint64_t> residues, uint64_t period)
{
    return llvm::all_of(residues, [&](uint64_t value) { return value < period; });
}
void normalize(GeneralArithmeticSelectorOutput& output)
{
    auto divisor = gcd(output.denominator, abs(output.numerator.constant));
    for (const auto& coefficient : output.numerator.coefficients) { divisor = gcd(divisor, abs(coefficient)); }
    output.denominator /= divisor;
    output.numerator.constant /= divisor;
    for (auto& coefficient : output.numerator.coefficients) { coefficient /= divisor; }
}
FailureOr<GeneralArithmeticSelectorOutput> substitute(
    const IntegerEliminationWitness& witness, llvm::ArrayRef<GeneralArithmeticSelectorOutput> earlier,
    unsigned inputs, uint64_t residue)
{
    if (witness.denominator <= BoundInteger(0) || witness.numerator.coefficients.size() != inputs + earlier.size()) {
        return failure();
    }
    GeneralArithmeticSelectorOutput result;
    result.numerator.constant = witness.numerator.constant;
    result.numerator.coefficients.assign(witness.numerator.coefficients.begin(),
                                         witness.numerator.coefficients.begin() + inputs);
    result.denominator = BoundInteger(1);
    result.residue = residue;
    for (auto [i, coordinate] : llvm::enumerate(earlier)) {
        if (coordinate.denominator <= BoundInteger(0) || coordinate.numerator.coefficients.size() != inputs) {
            return failure();
        }
        const auto weight = witness.numerator.coefficients[inputs + i];
        if (weight == BoundInteger(0)) { continue; }
        const auto scale = weight * result.denominator;
        result.numerator.constant = result.numerator.constant * coordinate.denominator +
                                    scale * coordinate.numerator.constant;
        for (unsigned j = 0; j < inputs; ++j) {
            result.numerator.coefficients[j] = result.numerator.coefficients[j] * coordinate.denominator +
                                               scale * coordinate.numerator.coefficients[j];
        }
        result.denominator *= coordinate.denominator;
    }
    result.denominator *= witness.denominator;
    normalize(result);
    return result;
}
FailureOr<std::vector<Tuple>> eliminateOutputs(const IntegerSystem& system, unsigned inputs,
                                               llvm::ArrayRef<uint64_t> residues)
{
    if (system.dimensions() != inputs + residues.size()) { return failure(); }
    if (residues.empty()) {
        if (system.isEmpty()) { return std::vector<Tuple>{}; }
        return std::vector<Tuple>{{system, {}}};
    }
    auto branches = system.eliminateWithWitness(system.dimensions() - 1);
    if (failed(branches)) { return failure(); }
    std::vector<Tuple> result;
    for (const auto& branch : *branches) {
        if (branch.domain.dimensions() + 1 != system.dimensions()) { return failure(); }
        auto earlier = eliminateOutputs(branch.domain, inputs, residues.drop_back());
        if (failed(earlier)) { return failure(); }
        for (auto& tuple : *earlier) {
            auto coordinate = substitute(branch, tuple.outputs, inputs, residues.back());
            if (failed(coordinate)) { return failure(); }
            tuple.outputs.push_back(std::move(*coordinate));
            result.push_back(std::move(tuple));
        }
    }
    return result;
}
const char* append(const ArithmeticRelationKey& key, const IntegerSystem& system, bool inverse,
                   llvm::ArrayRef<uint32_t> pipes, unsigned parameterCount,
                   GroupIndex& index, std::vector<GeneralArithmeticEndpointSelector>& selectors)
{
    const auto& input = inverse ? key.target : key.source;
    const auto& output = inverse ? key.source : key.target;
    const unsigned sourceDimensions = key.source.residues.size();
    const unsigned targetDimensions = key.target.residues.size();
    const unsigned inputs = input.residues.size() + parameterCount;
    std::vector<unsigned> map(system.dimensions());
    for (unsigned i = 0; i < sourceDimensions; ++i) { map[i] = inverse ? inputs + i : i; }
    for (unsigned i = 0; i < targetDimensions; ++i) { map[sourceDimensions + i] = inverse ? i : inputs + i; }
    for (unsigned i = 0; i < parameterCount; ++i) {
        map[sourceDimensions + targetDimensions + i] = input.residues.size() + i;
    }
    auto reordered = system.remap(system.dimensions(), map);
    if (failed(reordered)) { return "general selector column permutation failed"; }
    auto tuples = eliminateOutputs(*reordered, inputs, output.residues);
    if (failed(tuples)) { return "general selector exact output witness construction failed"; }
    if (tuples->empty()) { return nullptr; }
    const GroupKey group{input.site, pipes[output.site]};
    auto [position, inserted] = index.emplace(group, selectors.size());
    if (inserted) {
        selectors.push_back({input.site, group.second, static_cast<unsigned>(input.residues.size()),
                             parameterCount, {}});
    }
    auto& selector = selectors[position->second];
    if (selector.inputDimensions != input.residues.size()) {
        return "general selector site has inconsistent coordinate dimensions";
    }
    for (auto& tuple : *tuples) {
        selector.pieces.push_back({std::move(tuple.domain), input.residues, key.parameterResidues,
                                  output.site, std::move(tuple.outputs)});
    }
    return nullptr;
}
const char* construct(const GeneralArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> pipes,
                      GeneralArithmeticSelectors& result)
{
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.period) {
        return "general arithmetic selectors require certified exact minimum demands";
    }
    GroupIndex forward, inverse;
    for (const auto& [key, pieces] : analysis.minimumDemands) {
        const auto sourceDimensions = key.source.residues.size(), targetDimensions = key.target.residues.size();
        const auto maximum = std::numeric_limits<unsigned>::max();
        if (key.source.site >= pipes.size() || key.target.site >= pipes.size() ||
            key.source.event != ArithmeticEvent::Completion || key.target.event != ArithmeticEvent::Start ||
            sourceDimensions > maximum || targetDimensions > maximum - sourceDimensions ||
            analysis.parameterCount > maximum - sourceDimensions - targetDimensions ||
            key.parameterResidues.size() != analysis.parameterCount ||
            !validResidues(key.source.residues, analysis.period) ||
            !validResidues(key.target.residues, analysis.period) ||
            !validResidues(key.parameterResidues, analysis.period)) {
            return "general arithmetic selector has an invalid endpoint schema or residue tuple";
        }
        const auto dimensions = sourceDimensions + targetDimensions + analysis.parameterCount;
        for (const auto& system : pieces) {
            if (system.dimensions() != dimensions) { return "general arithmetic piece has inconsistent dimensions"; }
            if (system.isEmpty()) { continue; }
            if (const char* error = append(key, system, false, pipes, analysis.parameterCount,
                                           forward, result.forward)) {
                return error;
            }
            if (const char* error = append(key, system, true, pipes, analysis.parameterCount,
                                           inverse, result.inverse)) {
                return error;
            }
        }
    }
    return nullptr;
}
BoundInteger affine(const IntegerAffine& expression, llvm::ArrayRef<BoundInteger> values)
{
    auto result = expression.constant;
    for (auto [coefficient, value] : llvm::zip(expression.coefficients, values)) { result += coefficient * value; }
    return result;
}
FailureOr<bool> contains(const IntegerSystem& system, llvm::ArrayRef<BoundInteger> values)
{
    if (system.dimensions() != values.size()) { return failure(); }
    for (const auto& row : system.constraints()) {
        if (row.coefficients.size() != values.size()) { return failure(); }
        if (affine({row.coefficients, BoundInteger(0)}, values) > row.bound) { return false; }
    }
    for (const auto& row : system.congruences()) {
        if (row.coefficients.size() != values.size() || row.modulus <= BoundInteger(0)) { return failure(); }
        if (mod(affine({row.coefficients, BoundInteger(0)}, values) - row.residue, row.modulus) != BoundInteger(0)) {
            return false;
        }
    }
    return true;
}
bool matchesResidues(llvm::ArrayRef<BoundInteger> values, llvm::ArrayRef<uint64_t> residues,
                     const BoundInteger& period)
{
    for (auto [value, residue] : llvm::zip(values, residues)) {
        if (mod(value, period) != natural(residue)) { return false; }
    }
    return true;
}
} // namespace
GeneralArithmeticSelectors buildGeneralArithmeticSelectors(
    const GeneralArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> sitePipes)
{
    GeneralArithmeticSelectors result;
    result.period = analysis.period;
    result.parameterCount = analysis.parameterCount;
    if (const char* error = construct(analysis, sitePipes, result)) {
        result.error = error;
        result.forward.clear();
        result.inverse.clear();
    }
    return result;
}
FailureOr<std::optional<ArithmeticSelectedEndpoint>> evaluateGeneralArithmeticSelector(
    const GeneralArithmeticEndpointSelector& selector, uint64_t period,
    llvm::ArrayRef<BoundInteger> coordinates, llvm::ArrayRef<BoundInteger> parameters)
{
    if (!period || coordinates.size() != selector.inputDimensions || parameters.size() != selector.parameterCount) {
        return failure();
    }
    const auto divisor = natural(period);
    SmallVector<BoundInteger> inputs;
    for (const auto& coordinate : coordinates) { inputs.push_back(floorDiv(coordinate, divisor)); }
    for (const auto& parameter : parameters) { inputs.push_back(floorDiv(parameter, divisor)); }
    for (const auto& piece : selector.pieces) {
        if (piece.inputResidues.size() != coordinates.size() || piece.parameterResidues.size() != parameters.size() ||
            !validResidues(piece.inputResidues, period) || !validResidues(piece.parameterResidues, period)) {
            return failure();
        }
        if (!matchesResidues(coordinates, piece.inputResidues, divisor) ||
            !matchesResidues(parameters, piece.parameterResidues, divisor)) { continue; }
        auto enabled = contains(piece.domain, inputs);
        if (failed(enabled)) { return failure(); }
        if (!*enabled) { continue; }
        ArithmeticSelectedEndpoint output;
        output.site = piece.outputSite;
        for (const auto& coordinate : piece.outputs) {
            if (coordinate.numerator.coefficients.size() != inputs.size() ||
                coordinate.denominator <= BoundInteger(0) ||
                coordinate.residue >= period) { return failure(); }
            const auto numerator = affine(coordinate.numerator, inputs);
            if (mod(numerator, coordinate.denominator) != BoundInteger(0)) { return failure(); }
            output.coordinates.push_back(divisor * (numerator / coordinate.denominator) + natural(coordinate.residue));
        }
        return std::optional<ArithmeticSelectedEndpoint>(std::move(output));
    }
    return std::optional<ArithmeticSelectedEndpoint>();
}
} // namespace mlir::pto::frontiersynch
