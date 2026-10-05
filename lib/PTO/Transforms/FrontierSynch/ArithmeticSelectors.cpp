// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticSelectors.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using GroupKey = std::pair<std::size_t, uint32_t>;
using GroupIndex = std::map<GroupKey, std::size_t>;
BoundInteger natural(uint64_t value)
{
    // DynamicAPInt's small constructor is signed; this also handles uint64 max.
    return BoundInteger(static_cast<int64_t>(value >> 1)) * BoundInteger(2) +
           BoundInteger(static_cast<int64_t>(value & 1));
}
bool validResidues(llvm::ArrayRef<uint64_t> residues, uint64_t period)
{
    return llvm::all_of(residues, [&](uint64_t value) { return value < period; });
}
const char* append(const ArithmeticRelationKey& key, const DifferenceBoundSystem& system,
                   bool inverse, llvm::ArrayRef<uint32_t> pipes, unsigned parameterCount,
                   GroupIndex& index, std::vector<ArithmeticEndpointSelector>& selectors)
{
    const auto& input = inverse ? key.target : key.source;
    const auto& output = inverse ? key.source : key.target;
    const unsigned sourceDimensions = key.source.residues.size();
    const unsigned targetDimensions = key.target.residues.size();
    const unsigned inputDimensions = input.residues.size();
    const unsigned outputDimensions = output.residues.size();
    const unsigned inputBegin = inverse ? sourceDimensions : 0;
    const unsigned outputBegin = inverse ? 0 : sourceDimensions;
    SmallVector<unsigned> keep;
    for (unsigned i = 0; i < inputDimensions; ++i) { keep.push_back(inputBegin + i); }
    for (unsigned i = 0; i < parameterCount; ++i) {
        keep.push_back(sourceDimensions + targetDimensions + i);
    }
    auto domain = system.project(keep);
    if (failed(domain)) {
        return "arithmetic selector domain projection failed";
    }
    ArithmeticSelectorPiece piece{std::move(*domain), input.residues, key.parameterResidues, output.site, {}};
    for (unsigned coordinate = 0; coordinate < outputDimensions; ++coordinate) {
        const auto outputColumn = outputBegin + coordinate + 1;
        ArithmeticSelectorOutput selected;
        selected.residue = output.residues[coordinate];
        bool hasUpperBound = false;
        // Include the distinguished zero coordinate, then retained inputs.
        for (unsigned i = 0; i <= keep.size(); ++i) {
            const unsigned inputColumn = i == 0 ? 0 : keep[i - 1] + 1;
            if (auto lower = system.bound(inputColumn, outputColumn)) {
                selected.lowerBounds.push_back({i, -*lower});
            }
            hasUpperBound |= system.bound(outputColumn, inputColumn).has_value();
        }
        if (selected.lowerBounds.empty() || !hasUpperBound) {
            return "arithmetic selector output lacks finite input-relative bounds";
        }
        piece.outputs.push_back(std::move(selected));
    }
    const GroupKey group{input.site, pipes[output.site]};
    auto [position, inserted] = index.emplace(group, selectors.size());
    if (inserted) {
        selectors.push_back({input.site, group.second, inputDimensions, parameterCount, {}});
    }
    auto& selector = selectors[position->second];
    if (selector.inputDimensions != inputDimensions) {
        return "arithmetic selector site has inconsistent coordinate dimensions";
    }
    selector.pieces.push_back(std::move(piece));
    return nullptr;
}
const char* construct(const ArithmeticDemandAnalysis& analysis, llvm::ArrayRef<uint32_t> pipes,
                      ArithmeticSelectors& result)
{
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.period) {
        return "arithmetic selectors require a certified exact minimum-demand relation";
    }
    GroupIndex forward, inverse;
    for (const auto& [key, pieces] : analysis.minimumDemands) {
        const auto sourceDimensions = key.source.residues.size();
        const auto targetDimensions = key.target.residues.size();
        const auto maximum = std::numeric_limits<unsigned>::max();
        if (key.source.site >= pipes.size() || key.target.site >= pipes.size() ||
            key.source.event != ArithmeticEvent::Completion || key.target.event != ArithmeticEvent::Start ||
            sourceDimensions > maximum || targetDimensions > maximum - sourceDimensions ||
            analysis.parameterCount > maximum - sourceDimensions - targetDimensions ||
            key.parameterResidues.size() != analysis.parameterCount ||
            !validResidues(key.source.residues, analysis.period) ||
            !validResidues(key.target.residues, analysis.period) ||
            !validResidues(key.parameterResidues, analysis.period)) {
            return "arithmetic minimum-demand selector has an invalid schema or residue tuple";
        }
        const auto dimensions = sourceDimensions + targetDimensions + analysis.parameterCount;
        for (const auto& system : pieces) {
            if (system.dimensions() != dimensions) {
                return "arithmetic minimum-demand piece has inconsistent dimensions";
            }
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
bool matchesResidues(llvm::ArrayRef<BoundInteger> values, llvm::ArrayRef<uint64_t> residues,
                     const BoundInteger& period)
{
    if (values.size() != residues.size()) { return false; }
    for (auto [value, residue] : llvm::zip(values, residues)) {
        if (mod(value, period) != natural(residue)) { return false; }
    }
    return true;
}
bool contains(const DifferenceBoundSystem& system, llvm::ArrayRef<BoundInteger> matrixValues)
{
    if (system.isEmpty()) { return false; }
    for (const auto& constraint : system.constraints()) {
        if (matrixValues[constraint.lhs] - matrixValues[constraint.rhs] > constraint.bound) {
            return false;
        }
    }
    return true;
}
} // namespace
ArithmeticSelectors buildArithmeticSelectors(const ArithmeticDemandAnalysis& analysis,
                                             llvm::ArrayRef<uint32_t> sitePipes)
{
    ArithmeticSelectors result;
    result.period = analysis.period;
    result.parameterCount = analysis.parameterCount;
    if (const char* error = construct(analysis, sitePipes, result)) {
        result.error = error;
        result.forward.clear();
        result.inverse.clear();
    }
    return result;
}
FailureOr<std::optional<ArithmeticSelectedEndpoint>> evaluateArithmeticSelector(
    const ArithmeticEndpointSelector& selector, uint64_t period,
    llvm::ArrayRef<BoundInteger> coordinates, llvm::ArrayRef<BoundInteger> parameters)
{
    if (!period || coordinates.size() != selector.inputDimensions || parameters.size() != selector.parameterCount) {
        return failure();
    }
    const auto divisor = natural(period);
    SmallVector<BoundInteger> inputs{BoundInteger(0)};
    for (const auto& coordinate : coordinates) { inputs.push_back(floorDiv(coordinate, divisor)); }
    for (const auto& parameter : parameters) { inputs.push_back(floorDiv(parameter, divisor)); }
    for (const auto& piece : selector.pieces) {
        if (piece.domain.dimensions() != inputs.size() - 1 ||
            piece.inputResidues.size() != coordinates.size() ||
            piece.parameterResidues.size() != parameters.size() ||
            !validResidues(piece.inputResidues, period) || !validResidues(piece.parameterResidues, period)) {
            return failure();
        }
        if (!matchesResidues(coordinates, piece.inputResidues, divisor) ||
            !matchesResidues(parameters, piece.parameterResidues, divisor) || !contains(piece.domain, inputs)) {
            continue;
        }
        ArithmeticSelectedEndpoint output;
        output.site = piece.outputSite;
        for (const auto& coordinate : piece.outputs) {
            if (coordinate.lowerBounds.empty() || coordinate.residue >= period) {
                return failure();
            }
            std::optional<BoundInteger> selected;
            for (const auto& term : coordinate.lowerBounds) {
                if (term.input >= inputs.size()) { return failure(); }
                const auto candidate = inputs[term.input] + term.offset;
                if (!selected || candidate > *selected) { selected = candidate; }
            }
            output.coordinates.push_back(divisor * *selected + natural(coordinate.residue));
        }
        return std::optional<ArithmeticSelectedEndpoint>(std::move(output));
    }
    return std::optional<ArithmeticSelectedEndpoint>();
}
} // namespace mlir::pto::frontiersynch
