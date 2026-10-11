// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check the P=1 existential-local certificate against concrete congruences.
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRecognition.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
fs::ArithmeticPrimitives primitives(const fs::IntegerSystem& system)
{
    fs::ArithmeticPrimitives input;
    input.pipeCount = 1;
    for (auto kind : {fs::PrimitiveKind::Context, fs::PrimitiveKind::Occurrences, fs::PrimitiveKind::Order,
             fs::PrimitiveKind::Native, fs::PrimitiveKind::Reads, fs::PrimitiveKind::Writes,
             fs::PrimitiveKind::Prerequisites}) {
        fs::PrimitiveRelation relation;
        relation.kind = kind;
        relation.dimensions = 1;
        relation.coordinates.push_back({"x", fs::CoordinateKind::Occurrence});
        if (kind == fs::PrimitiveKind::Occurrences) {
            fs::ResiduePiece piece;
            piece.residues.push_back(0);
            piece.machineSystem = std::make_shared<const fs::IntegerSystem>(system);
            relation.pieces.push_back(std::move(piece));
        }
        input.relations.push_back(std::move(relation));
    }
    return input;
}
bool contains(const fs::IntegerSystem& system, ArrayRef<int64_t> point)
{
    auto dot = [&](const auto& coefficients) {
        fs::BoundInteger value(0);
        for (auto [coefficient, coordinate] : llvm::zip(coefficients, point)) {
            value += coefficient * fs::BoundInteger(coordinate);
        }
        return value;
    };
    return !system.isKnownEmpty() && llvm::all_of(system.constraints(), [&](const auto& row) {
        return dot(row.coefficients) <= row.bound;
    }) && llvm::all_of(system.congruences(), [&](const auto& row) {
        return mod(dot(row.coefficients) - row.residue, row.modulus) == fs::BoundInteger(0);
    });
}
bool equivalence(const fs::IntegerSystem& system, unsigned dimensions, uint64_t coefficient)
{
    auto recognition = fs::recognizeArithmetic(primitives(system), {1, dimensions, 1, coefficient});
    if (recognition.state != fs::RecognitionState::Applicable || recognition.pieces.size() != 1) { return false; }
    const auto& piece = recognition.pieces.front();
    if (piece.empty) { return system.isEmpty(); }
    if (!piece.machineSystem || !piece.machineSystem->congruences().empty() ||
        recognition.arithmeticClass != fs::ArithmeticClass::BoundedCoefficients ||
        recognition.observedDimensions != dimensions) { return false; }
    auto projection = piece.machineSystem->project({0});
    if (failed(projection)) { return false; }
    for (int64_t x = -12; x <= 12; ++x) {
        bool expected = false;
        if (system.dimensions() == 1) { expected = contains(system, {x}); }
        else {
            for (int64_t z = -1; z <= 1; ++z) { expected |= contains(system, {x, z}); }
        }
        const bool actual = llvm::any_of(*projection, [&](const auto& projected) { return contains(projected, {x}); });
        if (actual != expected) { return false; }
    }
    return true;
}
bool coefficientContract(const fs::ArithmeticPrimitives& input, const fs::ArithmeticRecognition& recognition,
                         uint64_t coefficient)
{
    fs::ArithmeticProgram arithmetic;
    arithmetic.extraction.state = fs::RecognitionState::Applicable;
    arithmetic.primitives = input;
    arithmetic.recognition = recognition;
    fs::ProgramRecognition program;
    fs::recordArithmeticContractAttempt(program, {1, 2, 1, coefficient}, arithmetic);
    if (program.arithmeticContracts.size() != 3) { return false; }
    return llvm::all_of(program.arithmeticContracts, [](const auto& candidate) {
        const auto expected = candidate.kind == fs::ContractClass::BoundedCoefficients ?
            fs::ContractStatus::Established : fs::ContractStatus::Violated;
        return candidate.membership == expected;
    });
}
bool coefficientBoundary(const fs::BoundInteger& signedValue, uint64_t coefficient)
{
    using B = fs::BoundInteger;
    auto system = fs::IntegerSystem::create(2, {{{signedValue, B(1)}, B(0)}});
    if (failed(system)) { return false; }
    const auto input = primitives(*system);
    auto exact = fs::recognizeArithmetic(input, {1, 2, 1, coefficient});
    auto maximum = fs::recognizeArithmetic(input, {1, 2, 1, UINT64_MAX});
    auto below = fs::recognizeArithmetic(input, {1, 2, 1, coefficient - 1});
    if (exact.state != fs::RecognitionState::Applicable || maximum.state != fs::RecognitionState::Applicable ||
        exact.observedCoefficient != coefficient || maximum.observedCoefficient != coefficient ||
        below.state != fs::RecognitionState::NotApplicable ||
        !llvm::any_of(below.diagnostics, [](const auto& diagnostic) {
            return diagnostic.issue == fs::ArithmeticIssue::CoefficientLimit;
        })) {
        llvm::errs() << "wide coefficient recognition differs: coefficient=" << coefficient << "\n";
        return false;
    }
    return coefficientContract(input, exact, coefficient);
}
bool unsignedCoefficientLimits()
{
    using B = fs::BoundInteger;
    const uint64_t highBit = uint64_t(1) << 63;
    for (uint64_t coefficient : {highBit - 1, highBit, UINT64_MAX}) {
        // Independent 32-bit limbs avoid the production half-value conversion.
        // The second row coefficient prevents gcd normalization shrinking it.
        const B wide = B(static_cast<int64_t>(coefficient >> 32)) * B(int64_t(1) << 32) +
            B(static_cast<int64_t>(coefficient & UINT32_MAX));
        if (!coefficientBoundary(wide, coefficient) || !coefficientBoundary(-wide, coefficient)) { return false; }
    }
    return true;
}
} // namespace
bool runMachineCertificateChecks()
{
    if (!unsignedCoefficientLimits()) { return false; }
    auto single = fs::IntegerSystem::create(1, {},
        {{{fs::BoundInteger(1)}, fs::BoundInteger(-1), fs::BoundInteger(4)}});
    if (failed(single) || !equivalence(*single, 2, 4)) { return false; }
    std::vector<fs::IntegerConstraint> bounds{
        {{fs::BoundInteger(0), fs::BoundInteger(1)}, fs::BoundInteger(1)},
        {{fs::BoundInteger(0), fs::BoundInteger(-1)}, fs::BoundInteger(1)}};
    std::vector<fs::IntegerCongruence> congruences{
        {{fs::BoundInteger(1), fs::BoundInteger(1)}, fs::BoundInteger(-1), fs::BoundInteger(4)},
        {{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(1), fs::BoundInteger(6)}};
    auto existing = fs::IntegerSystem::create(2, bounds, congruences);
    if (failed(existing) || !equivalence(*existing, 4, 6)) { return false; }
    const auto input = primitives(*existing);
    auto tooSmall = fs::recognizeArithmetic(input, {1, 3, 1, 6});
    auto tooNarrow = fs::recognizeArithmetic(input, {1, 4, 1, 5});
    if (tooSmall.state != fs::RecognitionState::MissingPremise || !tooSmall.pieces.empty() ||
        tooNarrow.state != fs::RecognitionState::NotApplicable || !tooNarrow.pieces.empty()) { return false; }
    auto contradiction = fs::IntegerSystem::create(1, {}, {
        {{fs::BoundInteger(1)}, fs::BoundInteger(0), fs::BoundInteger(4)},
        {{fs::BoundInteger(1)}, fs::BoundInteger(1), fs::BoundInteger(4)}});
    if (failed(contradiction) || !equivalence(*contradiction, 1, 4)) { return false; }
    llvm::outs() << "machine certificates: exact congruences, auxiliaries and class boundaries passed\n";
    return true;
}
