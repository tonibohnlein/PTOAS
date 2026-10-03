// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite guarded potential occurrences, not a program/control IR. Predicate
// evaluation selects a trace; no static site is assumed to execute unconditionally.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDANALYSIS_H
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include <cstdint>

namespace mlir::pto::frontiersynch {
using Predicate = std::size_t;
enum class PredicateKind { False, True, Atom, Not, And, Or };
// Supplied finite evaluation identity, independent of payload position. Several
// payloads under one evaluated branch share it; repeated evaluations need not.
struct GuardEvaluationId {
    std::uint64_t nameSpace = 0;
    std::uint64_t evaluation = 0;
};
struct PredicateNode {
    PredicateKind kind = PredicateKind::False;
    Predicate first = 0;
    Predicate second = 0;
    Value condition;
    std::optional<GuardEvaluationId> evaluation;
};
class PredicateArena {
public:
    PredicateArena();
    // Atom values must be live i1 SSA values. No scalar feasibility is inferred.
    // The unqualified overload explicitly asserts one shared truth value (the
    // legacy loop-free importer, or supplied invocation-immutable sharing).
    FailureOr<Predicate> atom(Value condition);
    FailureOr<Predicate> atom(Value condition, GuardEvaluationId evaluation);
    // Operands must belong to this arena. Construction interns binary nodes,
    // preserving sharing; it never expands Boolean normal forms.
    Predicate negate(Predicate operand);
    Predicate conjunction(Predicate first, Predicate second);
    Predicate disjunction(Predicate first, Predicate second);
    ArrayRef<PredicateNode> nodes() const { return expressions; }
    bool valid(Predicate id) const { return id < expressions.size(); }

private:
    Predicate intern(PredicateKind kind, Predicate first, Predicate second);
    Predicate combine(PredicateKind kind, Predicate first, Predicate second);
    SmallVector<PredicateNode> expressions;
    DenseMap<Value, Predicate> atoms;
    DenseMap<std::pair<Value, std::pair<std::uint64_t, std::uint64_t>>, Predicate> evaluatedAtoms;
    DenseMap<std::pair<unsigned, std::pair<Predicate, Predicate>>, Predicate> interned;
};
struct GuardedWitness {
    StorageWitness storage;
    Predicate predicate = 0;
};
struct GuardedDemand {
    std::size_t source = 0;
    std::size_t consumer = 0;
    Predicate predicate = 0;
    SmallVector<GuardedWitness> witnesses;
};
struct GuardedLocalDemand {
    std::size_t demand = 0;
    Predicate nonadjacent = 0;
};
class GuardedDemandAnalysis {
public:
    // Both MODELED pairwise may-alias builds are atomic. This result owns its arena and borrows phases,
    // conditions and memory records from the unchanged source/SyncInput.
    LogicalResult build(func::FuncOp source, const SyncInput& input, const StorageAnalysis& storage);
    // Original loop-free subtree, under its enclosing if outcomes. The supplied
    // index and phase list borrow the same unchanged source/input as storage.
    LogicalResult build(Operation* scope, const PhaseIndex& index,
                        ArrayRef<const CompoundInstanceElement*> phases, const StorageAnalysis& storage);
    // Analysis-boundary entry point, also used by bounded synthetic tests.
    // Extra prerequisites have no storage witnesses, are forward, and are
    // restricted by endpoint occurrence.
    LogicalResult build(
        ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<Predicate> occurrences,
        ArrayRef<StorageFootprint> footprints, ArrayRef<StorageAlias> aliases, PredicateArena predicates,
        ArrayRef<GuardedDemand> additional = {});
    // Strengthen only surviving local covers to the actual executed native
    // predecessor and run this reducer again. Atomic conservative transform;
    // occurrence guards and the original Boolean evaluation identities stay.
    LogicalResult adjacentLocalUpper();
    const PredicateArena& predicates() const { return conditions; }
    ArrayRef<const CompoundInstanceElement*> phases() const { return sites; }
    ArrayRef<Predicate> occurrences() const { return presence; }
    ArrayRef<GuardedDemand> generators() const { return demands; }
    ArrayRef<GuardedDemand> retained() const { return covers; }
    ArrayRef<GuardedLocalDemand> localDemands() const { return locals; }
    // Empty means invalid endpoints, not an absent requirement. Returned IDs
    // refer to this result's arena. Numerical ranks depend on executed sites.
    std::optional<Predicate> completionBeforeStart(std::size_t source, std::size_t consumer) const;

private:
    // Shared reducer input after modeled or exact generator extraction. This is
    // not a public physical-exactness claim for arbitrary supplied witnesses.
    LogicalResult buildPrepared(
        ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<Predicate> occurrences, PredicateArena predicates,
        ArrayRef<GuardedDemand> generators);
    void initializeNative();
    void closeRequired();
    void reduce();
    void recordLocal(const GuardedDemand& edge);
    PredicateArena conditions;
    SmallVector<const CompoundInstanceElement*> sites;
    SmallVector<Predicate> presence;
    SmallVector<GuardedDemand> demands;
    SmallVector<GuardedDemand> covers;
    SmallVector<GuardedLocalDemand> locals;
    SmallVector<SmallVector<Predicate>> reach;
};
} // namespace mlir::pto::frontiersynch
#endif
