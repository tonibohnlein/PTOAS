// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check supplied exact primitive relations, not the syntax of buffer indices.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICRECOGNITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICRECOGNITION_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Value.h"
#include <string>

namespace mlir::pto::frontiersynch {
enum class PrimitiveKind { Context, Occurrences, Order, Native, Reads, Writes, Prerequisites };
enum class CoordinateKind { Occurrence, Parameter, Storage, Auxiliary };
enum class ArithmeticClass { Differences, Octagons, BoundedCoefficients };
struct ArithmeticCoordinate {
    std::string name;
    CoordinateKind kind = CoordinateKind::Occurrence;
};
struct ResiduePiece {
    // Dimensions and leading symbols follow the declared coordinate schema.
    // Additional trailing symbols are existential quotient locals, never
    // occurrence coordinates or independent parameters. Importers project
    // them with exact integer elimination before exposing a relation.
    IntegerSet system;
    // system variables are quotient coordinates q, in dimensions-then-symbols
    // order; the original coordinate is period*q + residue. Locals have no residue.
    SmallVector<uint64_t> residues;
};
enum class ArithmeticEvent { Payload, Start, Completion };
struct PrimitiveRelation {
    PrimitiveKind kind = PrimitiveKind::Context;
    // Optional endpoint identities into the producer's site table. Native
    // relations may encode the strict native closure rather than adjacent edges.
    std::optional<std::size_t> sourceSite;
    std::optional<std::size_t> targetSite;
    ArithmeticEvent sourceEvent = ArithmeticEvent::Payload;
    ArithmeticEvent targetEvent = ArithmeticEvent::Payload;
    std::optional<AddressSpace> storageSpace;
    // Physical identity is (storageSpace, storageBase, byte). A null base means
    // an absolute address; a nonnull base is a canonical GM entry pointer and
    // the byte coordinate is relative to it, never its numeric pointer value.
    // Distinct base tags require a supplied disjointness policy/certificate.
    Value storageBase;
    // Leading dimensions belong to source, followed by target dimensions;
    // access relations then have one physical-byte dimension.
    unsigned sourceDimensions = 0;
    unsigned targetDimensions = 0;
    SmallVector<ArithmeticCoordinate> coordinates;
    unsigned dimensions = 0; // Remaining coordinates are IntegerSet symbols.
    SmallVector<ResiduePiece> pieces; // Empty union denotes false, not missing.
};
struct ArithmeticPrimitives {
    unsigned pipeCount = 0;
    uint64_t period = 1;
    SmallVector<std::string> parameters;
    SmallVector<PrimitiveRelation> relations;
};
struct ArithmeticLimits {
    unsigned pipes;
    unsigned dimensions;
    uint64_t period;
    uint64_t coefficient;
};
enum class ArithmeticIssue {
    MissingPrimitives, MissingRole, InvalidCoordinate, DimensionLimit,
    PipeLimit, PeriodMismatch, InvalidResidue, UnsupportedExpression,
    CoefficientOverflow, CoefficientLimit, InvalidConfiguration
};
struct ArithmeticDiagnostic {
    ArithmeticIssue issue;
    std::size_t relation = 0;
    std::size_t piece = 0;
};
struct LinearRow {
    SmallVector<int64_t> coefficients;
    int64_t constant = 0;
    bool equality = false; // sum(coefficients*q) + constant == 0 or >= 0.
};
// Indices refer to the supplied relation schema and residue metadata. Retain
// that input when consuming the normalized rows.
struct NormalizedPiece {
    std::size_t relation = 0;
    std::size_t piece = 0;
    bool empty = false;
    SmallVector<LinearRow> rows;
};
struct ArithmeticRecognition {
    RecognitionState state = RecognitionState::Applicable;
    ArithmeticClass arithmeticClass = ArithmeticClass::Differences;
    unsigned observedDimensions = 0;
    uint64_t observedCoefficient = 0;
    SmallVector<ArithmeticDiagnostic> diagnostics;
    SmallVector<NormalizedPiece> pieces;
};
// The producer supplies exact, complete relations, the coordinate meanings,
// finite admitted executions and the core-native/control contract. This checks
// their representation only. Configuration is fixed across an input class;
// never infer it by taking the maxima of the current input. No backend runs.
ArithmeticRecognition recognizeArithmetic(const ArithmeticPrimitives& input, const ArithmeticLimits& limits);
StringRef recognitionName(ArithmeticClass kind);
StringRef recognitionName(ArithmeticIssue issue);
} // namespace mlir::pto::frontiersynch
#endif
