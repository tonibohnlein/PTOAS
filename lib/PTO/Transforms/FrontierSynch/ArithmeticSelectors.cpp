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
// Prove that one affine term equals the selected maximum on a piece.
bool equalsMaximum(const ArithmeticSelectorTerm& term, const ArithmeticSelectorOutput& output,
                   const DifferenceBoundSystem& domain)
{
    bool attained = false;
    for (const auto& other : output.lowerBounds) {
        auto upper = domain.bound(other.input, term.input);
        if (!upper || *upper + other.offset > term.offset) { return false; }
        auto lower = domain.bound(term.input, other.input);
        attained |= lower && *lower + term.offset <= other.offset;
    }
    return attained;
}
// The DBM hull is exact iff no point violates an atom of each operand.
// Check pairs of complemented atoms using integer b+1, without distributing a
// general union complement or enumerating values of parameters/iterations.
std::optional<DifferenceBoundSystem> exactHull(const DifferenceBoundSystem& a,
                                               const DifferenceBoundSystem& b)
{
    if (a.isSubsetOf(b)) { return b; }
    if (b.isSubsetOf(a)) { return a; }
    auto atoms = a.constraints();
    const auto otherAtoms = b.constraints();
    std::vector<DifferenceBoundConstraint> common;
    for (const auto& atom : atoms) {
        if (auto bound = b.bound(atom.lhs, atom.rhs)) {
            common.push_back({atom.lhs, atom.rhs, std::max(atom.bound, *bound)});
        }
    }
    auto hull = DifferenceBoundSystem::create(a.dimensions(), common);
    if (failed(hull)) { return std::nullopt; }
    for (const auto& x : atoms) {
        auto hx = hull->bound(x.lhs, x.rhs);
        if (hx && *hx <= x.bound) { continue; }
        for (const auto& y : otherAtoms) {
            auto hy = hull->bound(y.lhs, y.rhs);
            if (hy && *hy <= y.bound) { continue; }
            auto trial = common;
            trial.push_back({x.rhs, x.lhs, -x.bound - BoundInteger(1)});
            trial.push_back({y.rhs, y.lhs, -y.bound - BoundInteger(1)});
            auto outside = DifferenceBoundSystem::create(a.dimensions(), trial);
            if (failed(outside) || !outside->isEmpty()) { return std::nullopt; }
        }
    }
    return *hull;
}
bool mergePieces(ArithmeticSelectorPiece& a, const ArithmeticSelectorPiece& b)
{
    if (a.outputSite != b.outputSite || a.inputResidues != b.inputResidues ||
        a.parameterResidues != b.parameterResidues || a.outputs.size() != b.outputs.size()) { return false; }
    std::vector<ArithmeticSelectorOutput> outputs;
    for (unsigned i = 0; i < a.outputs.size(); ++i) {
        if (a.outputs[i].residue != b.outputs[i].residue) { return false; }
        auto candidates = a.outputs[i].lowerBounds;
        llvm::append_range(candidates, b.outputs[i].lowerBounds);
        auto found = llvm::find_if(candidates, [&](const auto& term) {
            return equalsMaximum(term, a.outputs[i], a.domain) && equalsMaximum(term, b.outputs[i], b.domain);
        });
        if (found == candidates.end()) { return false; }
        outputs.push_back({{*found}, a.outputs[i].residue});
    }
    auto hull = exactHull(a.domain, b.domain);
    if (!hull) { return false; }
    a.domain = std::move(*hull);
    a.outputs = std::move(outputs);
    return true;
}
void simplify(ArithmeticEndpointSelector& selector)
{
    // One greedy sweep bounds the number of pair trials quadratically. A
    // rejected pair need not be retried: maximal coalescing is not required.
    // Only equivalent tagged maps merge, preserving first-match semantics.
    for (std::size_t i = 0; i < selector.pieces.size(); ++i) {
        for (std::size_t j = i + 1; j < selector.pieces.size();) {
            if (mergePieces(selector.pieces[i], selector.pieces[j])) {
                selector.pieces.erase(selector.pieces.begin() + j);
            } else { ++j; }
        }
    }
    for (auto& piece : selector.pieces) {
        for (auto& selected : piece.outputs) {
            // Remove a max term only when another remaining term dominates it
            // everywhere on this exact domain. Sequential removal retains a witness
            // even for equal terms; prefer nonconstant coordinates on ties.
            for (std::size_t i = 0; i < selected.lowerBounds.size();) {
                bool redundant = false;
                const auto& term = selected.lowerBounds[i];
                for (std::size_t j = 0; j < selected.lowerBounds.size(); ++j) {
                    if (i == j) { continue; }
                    const auto& other = selected.lowerBounds[j];
                    auto bound = piece.domain.bound(term.input, other.input);
                    if (bound && *bound + term.offset <= other.offset) {
                        redundant = true;
                        break;
                    }
                }
                if (redundant) { selected.lowerBounds.erase(selected.lowerBounds.begin() + i); }
                else { ++i; }
            }
        }
    }
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
    if (result.error.empty()) {
        for (auto& selector : result.forward) { simplify(selector); }
        for (auto& selector : result.inverse) { simplify(selector); }
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
