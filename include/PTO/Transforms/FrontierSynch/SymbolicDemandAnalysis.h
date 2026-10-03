// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Lazy exact Presburger analysis over supplied occurrence/effect relations.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_SYMBOLICDEMANDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_SYMBOLICDEMANDANALYSIS_H
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <array>
#include <memory>
#include <vector>

namespace mlir::pto::frontiersynch {
struct ImmutableParameterBinding {
    Value value;
    IntegerAttr result;
};
enum class SymbolicTuple { Unit, Occurrence, Event, Cell };
enum class RequirementModel { ExactPhysical, SharedModeled };
struct SymbolicSite {
    const CompoundInstanceElement* phase = nullptr;
    // Actual enclosing coordinate SSA identities, in the supplied canonical order.
    // Coordinates are mathematical integers, including negatives when admitted.
    SmallVector<Value> coordinates;
};
class SymbolicSchema;
using SymbolicSchemaHandle = std::shared_ptr<const SymbolicSchema>;
class SymbolicSchema {
public:
    static FailureOr<SymbolicSchemaHandle> create(
        ArrayRef<SymbolicSite> sites, ArrayRef<Value> parameters, ArrayRef<Type> cellAxes,
        std::uint64_t physicalPartition);
    SymbolicSchema(const SymbolicSchema&) = delete;
    SymbolicSchema& operator=(const SymbolicSchema&) = delete;
    ArrayRef<SymbolicSite> sites() const { return phaseSites; }
    ArrayRef<Value> parameters() const { return symbols; }
    ArrayRef<Type> cellTypes() const { return cellAxes; }
    std::uint64_t partition() const { return physicalPartition; }
    unsigned coordinateDepth() const { return depth; }
    std::size_t pipeCount() const { return pipes; }
    FailureOr<unsigned> arity(SymbolicTuple tuple) const;
    // Stable schema-owned axis IDs distinguish source/range variable copies.
    // Per-site SSA coordinate meaning remains in sites(), not a column's raw ID.
    FailureOr<presburger::PresburgerSpace> space(SymbolicTuple domain, SymbolicTuple range, bool bound = false) const;

private:
    SymbolicSchema() = default;
    struct Axis {};
    SmallVector<SymbolicSite> phaseSites;
    SmallVector<Value> symbols;
    SmallVector<Type> cellAxes;
    std::uint64_t physicalPartition = 0;
    unsigned depth = 0;
    std::size_t pipes = 0;
    // Identifier requires mutable opaque pointers; these tokens carry no state.
    mutable std::array<std::vector<Axis>, 4> domainAxes, rangeAxes;
};

class SymbolicPrimitive {
public:
    // Explicit schema, tuple roles, canonical axis ordering and ordered SSA
    // symbols declare the meaning of unidentified columns. Existing conflicting
    // IDs are rejected, including disjunct IDs. Arity alone never establishes it.
    static FailureOr<SymbolicPrimitive> import(
        SymbolicSchemaHandle schema, SymbolicTuple domain, SymbolicTuple range, ArrayRef<Value> orderedSymbols,
        const presburger::PresburgerRelation& relation);
    bool valid() const { return bool(owner); }
    SymbolicSchemaHandle schema() const { return owner; }
    SymbolicTuple domain() const { return source; }
    SymbolicTuple range() const { return target; }
    const presburger::PresburgerRelation& relation() const { return value; }

private:
    SymbolicSchemaHandle owner;
    SymbolicTuple source = SymbolicTuple::Unit, target = SymbolicTuple::Unit;
    presburger::PresburgerRelation value =
        presburger::PresburgerRelation::getEmpty(presburger::PresburgerSpace::getRelationSpace());
};
struct SymbolicInputs {
    SymbolicPrimitive context;   // Unit -> Unit: admitted invocation symbols.
    SymbolicPrimitive present;   // Unit -> Occurrence.
    SymbolicPrimitive reference; // Occurrence -> Occurrence: exact strict order.
    SymbolicPrimitive reads;     // Occurrence -> Cell: ALL exact physical reads.
    SymbolicPrimitive writes;    // Occurrence -> Cell: ALL exact physical writes.
    SymbolicPrimitive extras;    // Occurrence -> Occurrence: required prerequisites.
    // Optional smaller exact forward generator, replacing reads/writes extraction.
    // Its native closure must cover every required physical conflict.
    std::optional<SymbolicPrimitive> generators;
    // Exactness of reduction is relative to these supplied requirements.
    // SharedModeled preserves production may-alias conflicts; it does not claim
    // exact physical effects or physical leastness.
    RequirementModel model = RequirementModel::ExactPhysical;
};
enum class SymbolicRelationOp { Input, Union, Intersect, Difference, Compose, Inverse, Domain, Range, Context };
struct SymbolicRelationNode {
    SymbolicRelationOp operation = SymbolicRelationOp::Input;
    SymbolicTuple domain = SymbolicTuple::Unit, range = SymbolicTuple::Unit;
    std::size_t first = 0, second = 0;
    std::optional<SymbolicPrimitive> primitive;
};
struct SymbolicArenaIdentity {};
class SymbolicDemandAnalysis;
class SymbolicEvaluator;
class SymbolicRoot {
public:
    std::size_t index() const { return id; }

private:
    friend class SymbolicDemandAnalysis;
    friend class SymbolicEvaluator;
    std::shared_ptr<const SymbolicArenaIdentity> arena;
    std::size_t id = 0;
};
using SymbolicAnalysisHandle = std::shared_ptr<const SymbolicDemandAnalysis>;
class SymbolicDemandAnalysis {
public:
    // Structural validation only: no solving, normalization, emptiness or subset
    // checks. Supplier establishes exact physical effects/partition, a finite
    // occurrence domain per admitted parameter valuation, strict reference order
    // and fixed native pipe chains. The set of admitted valuations may be infinite.
    // No may-alias recovery or copied compiler control/effects model is implied.
    static FailureOr<SymbolicAnalysisHandle> build(SymbolicSchemaHandle schema, const SymbolicInputs& inputs);
    SymbolicSchemaHandle schema() const { return owner; }
    ArrayRef<SymbolicRelationNode> nodes() const { return graph; }
    std::size_t preparationNodes() const { return prepared; }
    // Exactly k+6 nodes beyond prepared N/G/Id, with one shared H subgraph.
    SymbolicRoot identity() const { return root(identityId); }
    SymbolicRoot native() const { return root(nativeId); }
    SymbolicRoot generators() const { return root(generatorId); }
    SymbolicRoot reachability() const { return root(reachId); }
    SymbolicRoot strictReachability() const { return root(strictId); }
    SymbolicRoot minimum() const { return root(minimumId); }
    SymbolicRoot presence() const { return root(presentId); }

private:
    friend class SymbolicEvaluator;
    SymbolicRoot root(std::size_t id) const;
    SymbolicSchemaHandle owner;
    std::shared_ptr<const SymbolicArenaIdentity> arena;
    SmallVector<SymbolicRelationNode, 0> graph;
    SymbolicPrimitive admitted;
    std::size_t prepared = 0, identityId = 0, nativeId = 0, generatorId = 0;
    std::size_t reachId = 0, strictId = 0, minimumId = 0, presentId = 0;
};
struct SymbolicContext {
    SmallVector<ImmutableParameterBinding> bindings;
    SmallVector<llvm::DynamicAPInt> values;
    bool admitted = false;
};
// Pure typed conversion shared by exact backends; no solver or admission test.
// The caller sets admitted only after checking its owned exact context relation.
FailureOr<SymbolicContext> decodeSymbolicBindings(
    SymbolicSchemaHandle schema, ArrayRef<ImmutableParameterBinding> bindings);
// Deliberately distinct from SymbolicPrimitive: specialization cannot silently
// re-enter an unbound arena. Even with zero backend symbols, binding meaning stays
// owned here. Uniform values have no context; bound values keep their full key.
class MaterializedSymbolicRelation {
public:
    SymbolicSchemaHandle schema() const { return owner; }
    SymbolicTuple domain() const { return source; }
    SymbolicTuple range() const { return target; }
    std::shared_ptr<const SymbolicContext> boundContext() const { return context; }
    const presburger::PresburgerRelation& relation() const { return value; }

private:
    friend class SymbolicEvaluator;
    SymbolicSchemaHandle owner;
    SymbolicTuple source = SymbolicTuple::Unit, target = SymbolicTuple::Unit;
    SymbolicRoot root;
    std::shared_ptr<const SymbolicContext> context;
    presburger::PresburgerRelation value =
        presburger::PresburgerRelation::getEmpty(presburger::PresburgerSpace::getRelationSpace());
};

struct SymbolicEvent {
    std::size_t site = 0;
    SmallVector<llvm::DynamicAPInt> coordinates;
    PeriodicEventKind kind = PeriodicEventKind::Start;
};
class SymbolicEvaluator {
public:
    static FailureOr<SymbolicEvaluator> uniform(SymbolicAnalysisHandle analysis);
    // All symbols require exactly one same-type IntegerAttr. i1 is Boolean0/1;
    // unsigned integers use unsigned interpretation, index/signless/signed use
    // signed interpretation. Generated relation math remains unbounded/exact.
    static FailureOr<SymbolicEvaluator> bind(
        SymbolicAnalysisHandle analysis, ArrayRef<ImmutableParameterBinding> bindings);
    bool admitted() const { return context && context->admitted; }
    std::shared_ptr<const SymbolicContext> boundContext() const { return context; }
    // Explicit potentially expensive exact normalization, with one owned cache
    // per immutable analysis/binding. Roots from another arena fail even when
    // schemas/node indices coincide. Inadmissible bound contexts fail.
    FailureOr<MaterializedSymbolicRelation> materialize(const SymbolicRoot& root);
    // Point queries require a complete admitted binding. Signed coordinates are
    // legal if present. Bad sites/kinds/coordinate arities fail. contains returns
    // false for absent events; reaches/minimumDemand require BOTH present.
    FailureOr<bool> contains(const SymbolicEvent& event);
    FailureOr<bool> reaches(const SymbolicEvent& source, const SymbolicEvent& consumer);
    FailureOr<bool> minimumDemand(const SymbolicEvent& source, const SymbolicEvent& consumer);
    // Consume an Event->Event artifact only in its ORIGINAL arena/binding cache.
    // Uniform artifacts and different bindings are rejected by point queries.
    FailureOr<bool> containsPair(
        const MaterializedSymbolicRelation& relation, const SymbolicEvent& source, const SymbolicEvent& consumer);
    std::size_t cachedNodes() const;

private:
    FailureOr<const presburger::PresburgerRelation*> evaluate(std::size_t id);
    FailureOr<SmallVector<llvm::DynamicAPInt>> point(const SymbolicEvent& event) const;
    FailureOr<bool> query(const SymbolicRoot& root, const SymbolicEvent& source, const SymbolicEvent& consumer);
    SymbolicAnalysisHandle analysis;
    std::shared_ptr<const SymbolicContext> context;
    std::vector<std::optional<presburger::PresburgerRelation>> cache;
};
// All results own schema/relations/bindings. Borrowed MLIR contexts, SSA values,
// phase records and coordinate identities must remain alive and unchanged.
} // namespace mlir::pto::frontiersynch
#endif
