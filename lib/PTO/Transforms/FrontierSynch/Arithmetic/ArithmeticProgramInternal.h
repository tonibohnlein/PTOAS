// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Internal producer state, residue splitting, and physical access extraction.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICPROGRAMINTERNAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICPROGRAMINTERNAL_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticProgram.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch { struct FiniteExpansionPlan; }
namespace mlir::pto::frontiersynch::detail {
struct ArithmeticValuePiece {
    AffineExpr value;
    SmallVector<AffineExpr> rows;
};
struct ProgramBuilder {
    ArithmeticProgram& output;
    const ArithmeticLimits& limits;
    MLIRContext* context;
    const PhaseIndex& index;
    DenseMap<Value, unsigned> parameterIds;
    ArithmeticEntryConstant entryConstant;
    // Values already admitted as exact guarded alternatives by prepareBound.
    // Domain construction must not replace these with machine placeholders.
    DenseSet<Value> guardedSelections = DenseSet<Value>();
    mutable DenseMap<std::pair<Value, unsigned>, bool> entryInputs = DenseMap<std::pair<Value, unsigned>, bool>();
    struct MachineBinding {
        Value value;
        ArithmeticSite site;
        unsigned offset;
        unsigned indexBits;
    };
    mutable std::vector<MachineBinding> machineBindings;
    using MachineKey = std::tuple<unsigned, unsigned, unsigned,
        std::vector<const void*>, std::vector<Operation*>>;
    std::map<MachineKey, std::vector<ResiduePiece>> machinePrimitives;
    bool expandEntryArithmetic = true;
    bool preferGuardedSelections = false;
    bool prepareMachine(Value input, const ArithmeticSite& site);
    std::optional<std::size_t> machineBinding(AffineExpr expression) const;
    AffineExpr machineValue(Value input, const ArithmeticSite& site, unsigned offset) const;
    bool emitMachine(PrimitiveRelation& target, ArrayRef<AffineExpr> rows);
    std::optional<int64_t> constant(Value value) const;
    bool appendPrimitive(PrimitiveRelation relation);
    bool constructionReady() const;
    PrimitiveRelation relation(PrimitiveKind kind, unsigned dimensions) const;
    bool staticallyEmpty(const ArithmeticSite& site) const;
    bool entryParameter(Value value, bool regionalLeaves = false) const;
    // Keep supported pure entry definitions coupled to their original leaves.
    // Cast boundaries and opaque values retain their SSA parameter identities.
    bool expandEntryDefinition(Value value) const;
    AffineExpr registerParameter(Value value);
    // Control domains cannot assume the occurrence whose presence they define.
    enum class ScalarProofContext { Occurrence, Definition };
    bool prepareValue(Value input, const ArithmeticSite& site,
                      ScalarProofContext proof = ScalarProofContext::Occurrence);
    bool prepareGuard(Value condition, const ArithmeticSite& site);
    bool prepareBound(Value input, const ArithmeticSite& site,
                      ScalarProofContext proof = ScalarProofContext::Occurrence);
    // Exact guarded choices for an origin with no local dimensions. Repeated
    // SSA symbols share one choice; unsupported branches are never discarded.
    std::optional<SmallVector<ArithmeticValuePiece>> piecewiseMap(
        AffineExpr origin, ArrayRef<Value> symbols, const ArithmeticSite& site, unsigned offset);
    // Conjoin both endpoint guard domains at their respective coordinate offsets.
    void emitForSites(PrimitiveRelation& relation, ArrayRef<AffineExpr> rows,
                      ArrayRef<std::pair<const ArithmeticSite*, unsigned>> endpoints,
                      SmallVector<SmallVector<AffineExpr>>* recipes = nullptr);
    AffineExpr value(Value input, const ArithmeticSite& site, unsigned offset,
                     ScalarProofContext proof = ScalarProofContext::Occurrence) const;
    SmallVector<AffineExpr> domain(const ArithmeticSite& site, unsigned offset) const;
    // All rows are >= 0. Filter optionally restricts one original coordinate
    // to a modular residue; its modulus must divide the configured period.
    void emit(PrimitiveRelation& relation, ArrayRef<AffineExpr> rows,
              std::optional<std::pair<unsigned, uint64_t>> filter = std::nullopt,
              uint64_t modulus = 1);
};
void collectExpanded(ProgramBuilder& builder, const FiniteExpansionLimits& limits,
    const ::mlir::pto::frontiersynch::FiniteExpansionPlan* plan = nullptr);
// Byte-coordinate translation; original physical primitives stay unchanged.
// Applied translations are published only after successful construction.
// An unavailable or unnecessary adapter returns no alternative.
std::optional<ArithmeticProgram> normalizeFiniteDemandAccesses(
    const ArithmeticProgram& program, uint64_t* attemptedFragments = nullptr,
    llvm::MapVector<std::pair<AddressSpace, Value>, AffineExpr>* appliedTranslations = nullptr);
void extractAccesses(ProgramBuilder& builder, const SyncInput& input, const SyncStorageEffects& effects);
} // namespace mlir::pto::frontiersynch::detail
#endif
