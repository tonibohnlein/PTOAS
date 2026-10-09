// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Recognize every primitive piece against one configured arithmetic class.
// Relation completeness/equivalence comes from the producer, never from syntax.
#include "PTO/Transforms/FrontierSynch/ArithmeticRecognition.h"
#include "ArithmeticRows.h"
#include "llvm/ADT/StringSet.h"
#include <array>

namespace mlir::pto::frontiersynch {
FailureOr<ArithmeticLimits> parseArithmeticProfile(StringRef text)
{
    SmallVector<StringRef> fields;
    text.split(fields, ':');
    ArithmeticLimits result{};
    const bool invalid = fields.size() != 4 || fields[0].getAsInteger(10, result.pipes) ||
        fields[1].getAsInteger(10, result.dimensions) || fields[2].getAsInteger(10, result.period) ||
        fields[3].getAsInteger(10, result.coefficient) || !result.pipes || !result.dimensions ||
        !result.period || !result.coefficient || result.period > static_cast<uint64_t>(INT64_MAX);
    if (invalid) {
        return failure();
    }
    return result;
}
namespace {
void reject(ArithmeticRecognition& output, ArithmeticIssue issue, std::size_t relation = 0, std::size_t piece = 0,
            bool missing = false)
{
    output.diagnostics.push_back({issue, !missing, relation, piece});
    if (!missing || output.state == RecognitionState::NotApplicable) {
        output.state = RecognitionState::NotApplicable;
    } else {
        output.state = RecognitionState::MissingPremise;
    }
}
bool tags(const PrimitiveRelation& relation)
{
    if (relation.sourceDimensions > relation.dimensions ||
        relation.targetDimensions > relation.dimensions - relation.sourceDimensions ||
        (!relation.sourceSite && relation.sourceDimensions) || (!relation.targetSite && relation.targetDimensions)) {
        return false;
    }
    const auto eventValid = [](ArithmeticEvent event) {
        return event == ArithmeticEvent::Payload || event == ArithmeticEvent::Start ||
               event == ArithmeticEvent::Completion;
    };
    return eventValid(relation.sourceEvent) && eventValid(relation.targetEvent);
}
bool schema(const PrimitiveRelation& relation, const ArithmeticPrimitives& input, const ArithmeticLimits& limits)
{
    if (!tags(relation) || relation.dimensions > relation.coordinates.size() ||
        relation.coordinates.size() > limits.dimensions) {
        return false;
    }
    llvm::StringSet<> names, parameters;
    for (const auto& coordinate : relation.coordinates) {
        if (coordinate.name.empty() || !names.insert(coordinate.name).second) {
            return false;
        }
        switch (coordinate.kind) {
        case CoordinateKind::Parameter: parameters.insert(coordinate.name); break;
        case CoordinateKind::Occurrence:
        case CoordinateKind::Storage:
        case CoordinateKind::Auxiliary: break;
        default: return false;
        }
    }
    if (parameters.size() != input.parameters.size()) {
        return false;
    }
    return llvm::all_of(input.parameters, [&](const auto& parameter) { return parameters.contains(parameter); });
}
bool validPiece(const ResiduePiece& piece, const PrimitiveRelation& relation, uint64_t period)
{
    auto system = piece.system;
    return system && system.getNumDims() == relation.dimensions &&
           system.getNumInputs() >= relation.coordinates.size() &&
           piece.residues.size() == relation.coordinates.size() &&
           llvm::all_of(piece.residues, [&](uint64_t residue) { return residue < period; });
}
void inspectPiece(const ResiduePiece& source, const ArithmeticLimits& limits, NormalizedPiece& normalized,
                  ArithmeticRecognition& output)
{
    auto system = source.system;
    for (unsigned rowId = 0; rowId < system.getNumConstraints(); ++rowId) {
        LinearRow row;
        row.equality = system.isEq(rowId);
        if (auto issue = detail::collectRow(system.getConstraint(rowId), system.getNumDims(),
                                            system.getNumSymbols(), row)) {
            reject(output, *issue, normalized.relation, normalized.piece,
                   *issue != ArithmeticIssue::UnsupportedExpression);
            continue;
        }
        const bool contradiction = detail::normalizeRow(row);
        normalized.empty |= contradiction;
        // A known false row contributes no unsupported coordinate direction.
        if (contradiction) {
            continue;
        }
        for (auto coefficient : row.coefficients) {
            const auto value = detail::magnitude(coefficient);
            output.observedCoefficient = std::max(output.observedCoefficient, value);
            if (value > limits.coefficient) {
                reject(output, ArithmeticIssue::CoefficientLimit, normalized.relation, normalized.piece);
            }
        }
        output.arithmeticClass = std::max(output.arithmeticClass, detail::classifyRow(row));
        normalized.rows.push_back(std::move(row));
    }
    if (normalized.empty) {
        normalized.rows.clear();
    }
}
} // namespace

ArithmeticRecognition recognizeArithmetic(const ArithmeticPrimitives& input, const ArithmeticLimits& limits)
{
    ArithmeticRecognition output;
    if (!limits.pipes || !limits.period || !limits.coefficient) {
        reject(output, ArithmeticIssue::InvalidConfiguration, 0, 0, true);
        return output;
    }
    if (input.relations.empty()) {
        reject(output, ArithmeticIssue::MissingPrimitives, 0, 0, true);
        return output;
    }
    if (input.pipeCount > limits.pipes) {
        reject(output, ArithmeticIssue::PipeLimit);
    }
    if (!input.period || input.period != limits.period) {
        reject(output, ArithmeticIssue::PeriodMismatch, 0, 0, !input.period);
    }
    llvm::StringSet<> parameterNames;
    for (const auto& parameter : input.parameters) {
        if (parameter.empty() || !parameterNames.insert(parameter).second) {
            reject(output, ArithmeticIssue::InvalidCoordinate, 0, 0, true);
        }
    }
    std::array<bool, 7> roles{};
    for (auto [id, relation] : llvm::enumerate(input.relations)) {
        const auto role = static_cast<unsigned>(relation.kind);
        if (role >= roles.size()) {
            reject(output, ArithmeticIssue::MissingRole, id, 0, true);
        } else {
            roles[role] = true;
        }
        if (relation.coordinates.size() > limits.dimensions) {
            reject(output, ArithmeticIssue::DimensionLimit, id);
            continue;
        }
        if (!schema(relation, input, limits)) {
            reject(output, ArithmeticIssue::InvalidCoordinate, id, 0, true);
            continue;
        }
        output.observedDimensions = std::max(output.observedDimensions,
                                             static_cast<unsigned>(relation.coordinates.size()));
        for (auto [pieceId, piece] : llvm::enumerate(relation.pieces)) {
            if (!validPiece(piece, relation, input.period)) {
                reject(output, ArithmeticIssue::InvalidResidue, id, pieceId, true);
                continue;
            }
            if (piece.system.getNumInputs() > limits.dimensions) {
                reject(output, ArithmeticIssue::DimensionLimit, id, pieceId);
                continue;
            }
            output.observedDimensions = std::max(output.observedDimensions, piece.system.getNumInputs());
            // General integer projection retains congruences introduced by
            // quotient elimination. Do not hand these lifted pieces to a
            // cheaper backend whose import assumes no existential locals.
            if (piece.system.getNumInputs() > relation.coordinates.size()) {
                output.arithmeticClass = ArithmeticClass::BoundedCoefficients;
            }
            NormalizedPiece normalized{id, pieceId, false, {}};
            inspectPiece(piece, limits, normalized, output);
            output.pieces.push_back(std::move(normalized));
        }
    }
    if (llvm::is_contained(roles, false)) {
        reject(output, ArithmeticIssue::MissingRole, 0, 0, true);
    }
    // Partial normalized output is diagnostic only; never publish it as input
    // to an exact backend when any part of the contract failed.
    if (output.state != RecognitionState::Applicable) {
        output.pieces.clear();
    }
    return output;
}
StringRef recognitionName(ArithmeticClass kind)
{
    switch (kind) {
    case ArithmeticClass::Differences: return "differences";
    case ArithmeticClass::Octagons: return "octagons";
    case ArithmeticClass::BoundedCoefficients: return "bounded-coefficients";
    default: return "invalid";
    }
}
StringRef recognitionName(ArithmeticIssue issue)
{
    switch (issue) {
    case ArithmeticIssue::MissingPrimitives: return "missing-primitives";
    case ArithmeticIssue::MissingRole: return "missing-role";
    case ArithmeticIssue::InvalidCoordinate: return "invalid-coordinate";
    case ArithmeticIssue::DimensionLimit: return "dimension-limit";
    case ArithmeticIssue::PipeLimit: return "pipe-limit";
    case ArithmeticIssue::PeriodMismatch: return "period-mismatch";
    case ArithmeticIssue::InvalidResidue: return "invalid-residue";
    case ArithmeticIssue::UnsupportedExpression: return "unsupported-expression";
    case ArithmeticIssue::CoefficientOverflow: return "coefficient-overflow";
    case ArithmeticIssue::CoefficientLimit: return "coefficient-limit";
    case ArithmeticIssue::InvalidConfiguration: return "invalid-configuration";
    default: return "invalid";
    }
}
} // namespace mlir::pto::frontiersynch
