// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact typed composition over an existing schema; normalization stays with callers.
#ifndef PTO_FRONTIERSYNCH_SYMBOLIC_COMPOSITION_H
#define PTO_FRONTIERSYNCH_SYMBOLIC_COMPOSITION_H
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
namespace mlir::pto::frontiersynch {
// Exact unit-equality local substitution, ordinary simplification and removal
// of integer-empty disjuncts; nonunit/division locals remain represented.
// This changes presentation only; it never rejects the source model.
presburger::PresburgerRelation normalizeSymbolicRelation(presburger::PresburgerRelation relation);
// Finite endpoint tags only: this set contains the actual endpoint projection
// while ignoring coordinates/parameters. Unknown tags return universe, never
// a source admission failure. Use only to restrict outer candidate support.
presburger::PresburgerSet symbolicEndpointSupport(
    const presburger::PresburgerRelation& relation, SymbolicSchemaHandle schema,
    SymbolicTuple endpoint, bool domain, bool bound = false);
// Operands are aligned to source->intermediate and intermediate->target in the
// same original schema/context. Fixed occurrence/event tags prune impossible
// joins; other tuples and nonliteral tags use ordinary Presburger composition.
presburger::PresburgerRelation composeSymbolicRelations(
    presburger::PresburgerRelation first, const presburger::PresburgerRelation& second,
    SymbolicSchemaHandle schema, SymbolicTuple source, SymbolicTuple intermediate,
    SymbolicTuple target, bool bound = false);
// Exact difference; normalize only RHS disjuncts that can share finite outer
// endpoint identities with the left relation. Unknown tags use exact generic
// QE; they do not create a new source admissibility requirement.
presburger::PresburgerRelation subtractSymbolicRelations(
    const presburger::PresburgerRelation& first, const presburger::PresburgerRelation& second,
    SymbolicSchemaHandle schema, SymbolicTuple source, SymbolicTuple target, bool bound = false);
} // namespace mlir::pto::frontiersynch
#endif
