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
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
namespace mlir::pto::frontiersynch::detail {
struct ProgramBuilder {
    ArithmeticProgram& output;
    const ArithmeticLimits& limits;
    MLIRContext* context;
    const PhaseIndex& index;
    DenseMap<Value, unsigned> parameterIds;
    ArithmeticEntryConstant entryConstant;
    mutable DenseMap<std::pair<Value, unsigned>, bool> entryInputs = DenseMap<std::pair<Value, unsigned>, bool>();
    std::optional<int64_t> constant(Value value) const;
    PrimitiveRelation relation(PrimitiveKind kind, unsigned dimensions) const;
    bool staticallyEmpty(const ArithmeticSite& site) const;
    bool entryParameter(Value value, bool regionalLeaves = false) const;
    AffineExpr registerParameter(Value value);
    bool prepareValue(Value input, const ArithmeticSite& site);
    bool prepareGuard(Value condition, const ArithmeticSite& site);
    bool prepareBound(Value input, const ArithmeticSite& site);
    // Conjoin both endpoint guard domains at their respective coordinate offsets.
    void emitForSites(PrimitiveRelation& relation, ArrayRef<AffineExpr> rows,
                      ArrayRef<std::pair<const ArithmeticSite*, unsigned>> endpoints,
                      SmallVector<SmallVector<AffineExpr>>* recipes = nullptr);
    AffineExpr value(Value input, const ArithmeticSite& site, unsigned offset) const;
    SmallVector<AffineExpr> domain(const ArithmeticSite& site, unsigned offset) const;
    // All rows are >= 0. Filter optionally restricts one original coordinate
    // to a modular residue; its modulus must divide the configured period.
    void emit(PrimitiveRelation& relation, ArrayRef<AffineExpr> rows,
              std::optional<std::pair<unsigned, uint64_t>> filter = std::nullopt,
              uint64_t modulus = 1);
};
void collectExpanded(ProgramBuilder& builder, const FiniteExpansionLimits& limits);
// Demand-only byte-coordinate translation; original physical primitives stay
// unchanged. An unavailable or unnecessary adapter returns no alternative.
std::optional<ArithmeticProgram> normalizeFiniteDemandAccesses(
    const ArithmeticProgram& program, uint64_t* attemptedFragments = nullptr);
void extractAccesses(ProgramBuilder& builder, const SyncInput& input, const SyncStorageEffects& effects);
} // namespace mlir::pto::frontiersynch::detail
#endif
