// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalRelationsInternal.h"
#include <numeric>
namespace mlir::pto::frontiersynch {
namespace {
BoundInteger natural(uint64_t value)
{
    return BoundInteger(static_cast<int64_t>(value >> 1)) * BoundInteger(2) +
        BoundInteger(static_cast<int64_t>(value & 1));
}
FailureOr<IntegerSystem> rawSystem(const IntegerSystem& system, uint64_t period,
    ArrayRef<uint64_t> residues, ArrayRef<unsigned> mapping, unsigned dimensions)
{
    if (!period || residues.size() != system.dimensions() || mapping.size() != residues.size()) { return failure(); }
    std::vector<IntegerConstraint> rows;
    std::vector<IntegerCongruence> mods;
    for (auto row : system.constraints()) {
        row.bound *= natural(period);
        for (unsigned i = 0; i < residues.size(); ++i) { row.bound += row.coefficients[i] * natural(residues[i]); }
        rows.push_back(std::move(row));
    }
    for (auto row : system.congruences()) {
        row.residue *= natural(period); row.modulus *= natural(period);
        for (unsigned i = 0; i < residues.size(); ++i) { row.residue += row.coefficients[i] * natural(residues[i]); }
        mods.push_back(std::move(row));
    }
    if (period != 1) {
        for (unsigned i = 0; i < residues.size(); ++i) {
            IntegerCongruence row; row.coefficients.resize(residues.size()); row.coefficients[i] = BoundInteger(1);
            row.residue = natural(residues[i]); row.modulus = natural(period); mods.push_back(std::move(row));
        }
    }
    auto result = IntegerSystem::create(system.dimensions(), rows, mods);
    return failed(result) ? FailureOr<IntegerSystem>(failure()) : result->remap(dimensions, mapping);
}
} // namespace
LogicalResult normalizeRegionalRelationData(RegionalRelationData& data,
    ArrayRef<Value> parameters, ArrayRef<RegionExpressions::Id> bindings, std::string& error)
{
    const unsigned oldP = data.analysis.parameterCount, newP = parameters.size();
    if (oldP != data.parameterValues.size() || oldP != data.parameters.size() || newP != bindings.size()) {
        error = "regional relation parameter schema is malformed"; return failure();
    }
    std::vector<unsigned> parameterMap;
    for (unsigned i = 0; i < oldP; ++i) {
        auto found = llvm::find(parameters, data.parameterValues[i]);
        if (found == parameters.end() || bindings[found - parameters.begin()] != data.parameters[i]) {
            error = "regional relation parameter bindings differ"; return failure();
        }
        parameterMap.push_back(found - parameters.begin());
    }
    auto mapFor = [&](unsigned count) {
        std::vector<unsigned> map(count); std::iota(map.begin(), map.end(), 0);
        for (auto parameter : parameterMap) { map.push_back(count + parameter); }
        return map;
    };
    const uint64_t period = data.analysis.period;
    for (auto* relation : {&data.analysis.nativeOrder, &data.analysis.requiredOrder,
                           &data.analysis.minimumDemands, &data.analysis.generators}) {
        GeneralArithmeticRelation translated;
        for (const auto& [key, pieces] : *relation) {
            auto residues = key.source.residues; llvm::append_range(residues, key.target.residues);
            const unsigned coordinates = residues.size(); llvm::append_range(residues, key.parameterResidues);
            auto next = key;
            std::fill(next.source.residues.begin(), next.source.residues.end(), 0);
            std::fill(next.target.residues.begin(), next.target.residues.end(), 0);
            next.parameterResidues.assign(newP, 0);
            for (const auto& piece : pieces) {
                auto lifted = rawSystem(piece, period, residues, mapFor(coordinates), coordinates + newP);
                if (failed(lifted)) { error = "regional relation residue lifting failed"; return failure(); }
                translated[next].push_back(std::move(*lifted));
            }
        }
        *relation = std::move(translated);
    }
    for (auto& occurrence : data.occurrences) {
        auto residues = occurrence.residues; llvm::append_range(residues, occurrence.parameterResidues);
        const unsigned d = occurrence.residues.size();
        auto lifted = rawSystem(occurrence.system, period, residues, mapFor(d), d + newP);
        if (failed(lifted)) { return failure(); }
        occurrence.system = std::move(*lifted); occurrence.residues.assign(d, 0);
        occurrence.parameterResidues.assign(newP, 0);
    }
    for (auto& support : data.selectors.support) {
        std::vector<uint64_t> residues{support.byteResidue}; llvm::append_range(residues, support.parameterResidues);
        auto lifted = rawSystem(support.domain, period, residues, mapFor(1), 1 + newP);
        if (failed(lifted)) { return failure(); }
        support.domain = std::move(*lifted); support.byteResidue = 0; support.parameterResidues.assign(newP, 0);
    }
    for (auto& boundary : data.selectors.boundaries) {
        const unsigned d = boundary.selector.inputDimensions;
        boundary.selector.parameterCount = newP;
        for (auto& piece : boundary.selector.pieces) {
            auto residues = piece.inputResidues; llvm::append_range(residues, piece.parameterResidues);
            const auto map = mapFor(d);
            auto lifted = rawSystem(piece.domain, period, residues, map, d + newP);
            if (failed(lifted)) { return failure(); }
            for (auto& output : piece.outputs) {
                auto old = output.numerator;
                output.numerator.coefficients.assign(d + newP, BoundInteger(0));
                output.numerator.constant = natural(period) * old.constant +
                    output.denominator * natural(output.residue);
                if (old.coefficients.size() != map.size()) { return failure(); }
                for (unsigned i = 0; i < map.size(); ++i) {
                    output.numerator.coefficients[map[i]] += old.coefficients[i];
                    output.numerator.constant -= old.coefficients[i] * natural(residues[i]);
                }
                output.residue = 0;
            }
            piece.domain = std::move(*lifted); piece.inputResidues.assign(d, 0);
            piece.parameterResidues.assign(newP, 0);
        }
    }
    data.analysis.period = 1; data.selectors.period = 1;
    data.analysis.parameterCount = newP; data.selectors.parameterCount = newP;
    data.parameterValues.assign(parameters.begin(), parameters.end());
    data.parameters.assign(bindings.begin(), bindings.end());
    return success();
}
} // namespace mlir::pto::frontiersynch
