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
#include <memory>
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
    // This certifies the relational construction only. Executable selectors and
    // cut availability remain consumer obligations.
    bool exactMinimum = false;
    bool adjacentLocalDemands = false; // Order-equality fact, not an insertion requirement.
};
using ArithmeticDemandAnalysis = TypedArithmeticDemandAnalysis<DifferenceBoundSystem>;
using GeneralArithmeticDemandAnalysis = TypedArithmeticDemandAnalysis<IntegerSystem>;
class StructuredProtection;
// Paused exact generator construction. The original program/shared effects are
// borrowed during creation; continuation uses only the imported relations.
// Moving into completion consumes the stage, so fallback never regenerates its
// accesses, projected conflicts or prerequisite relations.
template<class System>
struct TypedArithmeticOccurrenceDomain {
    std::size_t site = 0;
    std::vector<uint64_t> residues, parameterResidues;
    System system;
};
using ArithmeticOccurrenceDomain = TypedArithmeticOccurrenceDomain<IntegerSystem>;
using DifferenceArithmeticOccurrenceDomain = TypedArithmeticOccurrenceDomain<DifferenceBoundSystem>;
class GeneralArithmeticGeneratorStage {
public:
    using System = IntegerSystem;
    GeneralArithmeticGeneratorStage();
    ~GeneralArithmeticGeneratorStage();
    GeneralArithmeticGeneratorStage(GeneralArithmeticGeneratorStage&&) noexcept;
    GeneralArithmeticGeneratorStage& operator=(GeneralArithmeticGeneratorStage&&) noexcept;
    GeneralArithmeticGeneratorStage(const GeneralArithmeticGeneratorStage&) = delete;
    GeneralArithmeticGeneratorStage& operator=(const GeneralArithmeticGeneratorStage&) = delete;
    const GeneralArithmeticDemandAnalysis& analysis() const;
    bool belongsTo(const ArithmeticProgram& program) const;
    const std::vector<ArithmeticOccurrenceDomain>& occurrences() const;
private:
    struct State;
    std::unique_ptr<State> state;
    friend GeneralArithmeticGeneratorStage analyzeGeneralArithmeticGenerators(
        const ArithmeticProgram&, const StructuredProtection*);
    friend GeneralArithmeticDemandAnalysis completeGeneralArithmeticDemands(GeneralArithmeticGeneratorStage&&);
    friend std::shared_ptr<GeneralArithmeticDemandAnalysis> reduceGeneralArithmeticDemands(
        GeneralArithmeticGeneratorStage&);
};
GeneralArithmeticGeneratorStage analyzeGeneralArithmeticGenerators(
    const ArithmeticProgram& program, const StructuredProtection* protection = nullptr);
GeneralArithmeticDemandAnalysis completeGeneralArithmeticDemands(GeneralArithmeticGeneratorStage&& stage);
// Non-consuming completion retains the immutable generator/native-order view
// for a later adapter attempt and caches the reduced result independently.
std::shared_ptr<GeneralArithmeticDemandAnalysis> reduceGeneralArithmeticDemands(
    GeneralArithmeticGeneratorStage& stage);
class DifferenceArithmeticGeneratorStage {
public:
    using System = DifferenceBoundSystem;
    DifferenceArithmeticGeneratorStage();
    ~DifferenceArithmeticGeneratorStage();
    DifferenceArithmeticGeneratorStage(DifferenceArithmeticGeneratorStage&&) noexcept;
    DifferenceArithmeticGeneratorStage& operator=(DifferenceArithmeticGeneratorStage&&) noexcept;
    DifferenceArithmeticGeneratorStage(const DifferenceArithmeticGeneratorStage&) = delete;
    DifferenceArithmeticGeneratorStage& operator=(const DifferenceArithmeticGeneratorStage&) = delete;
    const ArithmeticDemandAnalysis& analysis() const;
    bool belongsTo(const ArithmeticProgram& program) const;
    const std::vector<DifferenceArithmeticOccurrenceDomain>& occurrences() const;
private:
    struct State;
    std::unique_ptr<State> state;
    friend DifferenceArithmeticGeneratorStage analyzeDifferenceArithmeticGenerators(
        const ArithmeticProgram&, const StructuredProtection*);
    friend ArithmeticDemandAnalysis completeDifferenceArithmeticDemands(DifferenceArithmeticGeneratorStage&&);
    friend std::shared_ptr<ArithmeticDemandAnalysis> reduceDifferenceArithmeticDemands(
        DifferenceArithmeticGeneratorStage&);
};
DifferenceArithmeticGeneratorStage analyzeDifferenceArithmeticGenerators(
    const ArithmeticProgram& program, const StructuredProtection* protection = nullptr);
ArithmeticDemandAnalysis completeDifferenceArithmeticDemands(DifferenceArithmeticGeneratorStage&& stage);
// Non-consuming completion retains the immutable generator/native-order view
// for a later adapter attempt and caches the reduced result independently.
std::shared_ptr<ArithmeticDemandAnalysis> reduceDifferenceArithmeticDemands(
    DifferenceArithmeticGeneratorStage& stage);
// The DBM entry point accepts the difference-bound subclass. Input must supply
// complete physical primitives and the core-native model. Every join retains
// one parameter context and matches endpoint residue tuples exactly. No loop
// unfolding, IR mutation, hardware-protection inference or endpoint synthesis.
class StructuredProtection;
ArithmeticDemandAnalysis analyzeArithmeticDemandsWithProtection(const ArithmeticProgram& program,
                                                               const StructuredProtection& protection);
GeneralArithmeticDemandAnalysis analyzeGeneralArithmeticDemandsWithProtection(const ArithmeticProgram& program,
                                                                             const StructuredProtection& protection);
ArithmeticDemandAnalysis analyzeArithmeticDemands(const ArithmeticProgram& program);
// Exact integer projection extends the same relational engine to recognized
// octagons and bounded coefficients. Projected unions retain their congruences;
// generated coefficients are not rechecked against the primitive-input bound.
// Supplied core native relation is reflexive and closed. Additional C->I
// prerequisites are reference-forward under the same pipe-chain shortcut
// bound used by exact demand reduction. Reuses that native closure engine.
FailureOr<GeneralArithmeticRelation> completeGeneralArithmeticNativeOrder(
    GeneralArithmeticRelation native, unsigned pipeCount, unsigned parameterCount,
    ArithmeticAnalysisCost& cost, std::string& error);
GeneralArithmeticDemandAnalysis analyzeGeneralArithmeticDemands(const ArithmeticProgram& program);
} // namespace mlir::pto::frontiersynch
#endif
