// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICDEMANDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICDEMANDANALYSIS_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/Transforms/FrontierSynch/DifferenceBoundRelations.h"
#include "PTO/Transforms/FrontierSynch/IntegerRelations.h"
#include <map>
#include <vector>
namespace mlir::pto::frontiersynch {
struct ArithmeticEventKey {
    std::size_t site = 0;
    ArithmeticEvent event = ArithmeticEvent::Payload;
    std::vector<uint64_t> residues;
    bool operator<(const ArithmeticEventKey& other) const;
    bool operator==(const ArithmeticEventKey& other) const;
};
struct ArithmeticRelationKey {
    ArithmeticEventKey source, target;
    std::vector<uint64_t> parameterResidues;
    bool operator<(const ArithmeticRelationKey& other) const;
};
// Columns: source quotient coordinates, target quotient coordinates, one shared
// parameter quotient tuple. Original coordinates are period*q + their residue.
// Every piece denotes exact integer points; an empty union is false. The DBM
// specialization is closed; general integer pieces retain normalized congruences.
template<class System>
using TypedArithmeticRelation = std::map<ArithmeticRelationKey, std::vector<System>>;
using ArithmeticRelation = TypedArithmeticRelation<DifferenceBoundSystem>;
using GeneralArithmeticRelation = TypedArithmeticRelation<IntegerSystem>;
struct ArithmeticAnalysisCost {
    uint64_t primitivePieces = 0;
    uint64_t pieceJoins = 0;
    uint64_t projections = 0;
    uint64_t relationCompositions = 0;
    uint64_t differences = 0;
    uint64_t outputPieces = 0;
};
template<class System>
struct TypedArithmeticDemandAnalysis {
    std::string error;
    uint64_t period = 1;
    unsigned parameterCount = 0, pipeCount = 0;
    TypedArithmeticRelation<System> generators;
    TypedArithmeticRelation<System> nativeOrder; // Reflexive N, restricted to present events.
    TypedArithmeticRelation<System> requiredOrder; // Strict H, all payload event pairs.
    TypedArithmeticRelation<System> minimumDemands; // F*: completion-to-start covers.
    ArithmeticAnalysisCost cost;
    // This certifies the relational construction only. Executable selectors,
    // cut availability and same-pipe adjacency remain consumer obligations.
    bool exactMinimum = false;
    bool adjacentLocalDemands = false; // No executed same-pipe payload lies between local endpoints.
};
using ArithmeticDemandAnalysis = TypedArithmeticDemandAnalysis<DifferenceBoundSystem>;
using GeneralArithmeticDemandAnalysis = TypedArithmeticDemandAnalysis<IntegerSystem>;
// The DBM entry point accepts the difference-bound subclass. Input must supply
// complete physical primitives and the core-native model. Every join retains
// one parameter context and matches endpoint residue tuples exactly. No loop
// unfolding, IR mutation, hardware-protection inference or endpoint synthesis.
ArithmeticDemandAnalysis analyzeArithmeticDemands(const ArithmeticProgram& program);
// Exact integer projection extends the same relational engine to recognized
// octagons and bounded coefficients. Projected unions retain their congruences;
// generated coefficients are not rechecked against the primitive-input bound.
GeneralArithmeticDemandAnalysis analyzeGeneralArithmeticDemands(const ArithmeticProgram& program);
} // namespace mlir::pto::frontiersynch
#endif
