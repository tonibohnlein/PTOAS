// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Construct the factored bounded-witness expression without normalizing it.
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using Tuple = SymbolicTuple;
using Op = SymbolicRelationOp;
using Relation = presburger::PresburgerRelation;
using IntegerRelation = presburger::IntegerRelation;
using llvm::DynamicAPInt;
bool matches(const SymbolicPrimitive& primitive, SymbolicSchemaHandle schema, Tuple domain, Tuple range)
{
    return primitive.valid() && primitive.schema() == schema && primitive.domain() == domain &&
           primitive.range() == range;
}
class Builder {
public:
    explicit Builder(SymbolicSchemaHandle schema) : schema(std::move(schema)) {}
    std::size_t input(const SymbolicPrimitive& primitive)
    {
        if (primitive.schema() != schema) {
            valid = false;
            return 0;
        }
        nodes.push_back({Op::Input, primitive.domain(), primitive.range(), 0, 0, primitive});
        return nodes.size() - 1;
    }
    std::size_t primitive(Tuple domain, Tuple range, const Relation& relation)
    {
        auto value = SymbolicPrimitive::import(schema, domain, range, schema->parameters(), relation);
        if (failed(value)) {
            valid = false;
            return 0;
        }
        return input(*value);
    }
    std::size_t inverse(std::size_t id)
    {
        nodes.push_back({Op::Inverse, nodes[id].range, nodes[id].domain, id, 0, std::nullopt});
        return nodes.size() - 1;
    }
    std::size_t operation(Op op, std::size_t left, std::size_t right)
    {
        auto domain = nodes[left].domain, range = nodes[left].range;
        const auto a = nodes[right].domain, b = nodes[right].range;
        bool compatible = false;
        switch (op) {
            case Op::Union:
            case Op::Intersect:
            case Op::Difference:
                compatible = domain == a && range == b;
                break;
            case Op::Compose:
                compatible = range == a;
                range = b;
                break;
            case Op::Domain:
                compatible = a == Tuple::Unit && b == domain;
                break;
            case Op::Range:
                compatible = a == Tuple::Unit && b == range;
                break;
            case Op::Context:
                compatible = a == Tuple::Unit && b == Tuple::Unit;
                break;
            default:
                break;
        }
        if (!compatible) {
            valid = false;
            return 0;
        }
        nodes.push_back({op, domain, range, left, right, std::nullopt});
        return nodes.size() - 1;
    }
    std::size_t restrict(std::size_t id, std::size_t context, std::size_t present, bool range)
    {
        id = operation(Op::Context, id, context);
        id = operation(Op::Domain, id, present);
        return range ? operation(Op::Range, id, present) : id;
    }
    std::size_t lift(std::size_t pairs, std::size_t source, std::size_t target)
    {
        return operation(Op::Compose, operation(Op::Compose, inverse(source), pairs), target);
    }
    SmallVector<SymbolicRelationNode, 0> nodes;
    bool valid = true;

private:
    SymbolicSchemaHandle schema;
};
void equate(IntegerRelation& relation, unsigned first, unsigned second)
{
    SmallVector<DynamicAPInt> row(relation.getNumCols(), DynamicAPInt(0));
    row[first] = DynamicAPInt(1);
    row[second] = DynamicAPInt(-1);
    relation.addEquality(row);
}
Relation tupleIdentity(SymbolicSchemaHandle schema, Tuple tuple)
{
    IntegerRelation relation(*schema->space(tuple, tuple));
    const auto count = *schema->arity(tuple);
    for (unsigned i = 0; i < count; ++i) {
        equate(relation, i, count + i);
    }
    return Relation(relation);
}
Relation shape(SymbolicSchemaHandle schema)
{
    const auto space = *schema->space(Tuple::Unit, Tuple::Occurrence);
    auto result = Relation::getEmpty(space);
    for (auto [id, site] : llvm::enumerate(schema->sites())) {
        IntegerRelation piece(space);
        piece.addBound(presburger::BoundType::EQ, 0, DynamicAPInt(static_cast<std::int64_t>(id)));
        for (unsigned coordinate = site.coordinates.size(); coordinate < schema->coordinateDepth(); ++coordinate) {
            piece.addBound(presburger::BoundType::EQ, coordinate + 1, DynamicAPInt(0));
        }
        result.unionInPlace(piece);
    }
    return result;
}
Relation embedding(SymbolicSchemaHandle schema, std::int64_t kind)
{
    IntegerRelation relation(*schema->space(Tuple::Occurrence, Tuple::Event));
    const auto count = *schema->arity(Tuple::Occurrence);
    for (unsigned i = 0; i < count; ++i) {
        equate(relation, i, count + i);
    }
    relation.addBound(presburger::BoundType::EQ, 2 * count, DynamicAPInt(kind));
    return Relation(relation);
}
Relation samePipe(SymbolicSchemaHandle schema)
{
    const auto space = *schema->space(Tuple::Occurrence, Tuple::Occurrence);
    auto result = Relation::getEmpty(space);
    for (auto [source, a] : llvm::enumerate(schema->sites())) {
        for (auto [target, b] : llvm::enumerate(schema->sites())) {
            if (a.phase->kPipeValue != b.phase->kPipeValue) {
                continue;
            }
            IntegerRelation piece(space);
            piece.addBound(presburger::BoundType::EQ, 0, DynamicAPInt(static_cast<std::int64_t>(source)));
            piece.addBound(
                presburger::BoundType::EQ, *schema->arity(Tuple::Occurrence),
                DynamicAPInt(static_cast<std::int64_t>(target)));
            result.unionInPlace(piece);
        }
    }
    return result;
}
} // namespace

SymbolicRoot SymbolicDemandAnalysis::root(std::size_t id) const
{
    SymbolicRoot result;
    result.arena = arena;
    result.id = id;
    return result;
}
FailureOr<SymbolicAnalysisHandle> SymbolicDemandAnalysis::build(
    SymbolicSchemaHandle schema, const SymbolicInputs& inputs)
{
    if (!schema || !matches(inputs.context, schema, Tuple::Unit, Tuple::Unit) ||
        !matches(inputs.present, schema, Tuple::Unit, Tuple::Occurrence) ||
        !matches(inputs.reference, schema, Tuple::Occurrence, Tuple::Occurrence) ||
        !matches(inputs.extras, schema, Tuple::Occurrence, Tuple::Occurrence)) {
        return failure();
    }
    if (inputs.generators) {
        if (!matches(*inputs.generators, schema, Tuple::Occurrence, Tuple::Occurrence)) {
            return failure();
        }
    } else if (
        !matches(inputs.reads, schema, Tuple::Occurrence, Tuple::Cell) ||
        !matches(inputs.writes, schema, Tuple::Occurrence, Tuple::Cell)) {
        return failure();
    }
    Builder builder(schema);
    const auto context = builder.input(inputs.context);
    auto present = builder.operation(
        Op::Intersect, builder.input(inputs.present), builder.primitive(Tuple::Unit, Tuple::Occurrence, shape(schema)));
    present = builder.operation(Op::Context, present, context);
    auto reference = builder.restrict(builder.input(inputs.reference), context, present, true);
    const auto occurrenceIdentity = builder.restrict(
        builder.primitive(Tuple::Occurrence, Tuple::Occurrence, tupleIdentity(schema, Tuple::Occurrence)), context,
        present, true);
    const auto start = builder.operation(
        Op::Domain, builder.primitive(Tuple::Occurrence, Tuple::Event, embedding(schema, 0)), present);
    const auto completion = builder.operation(
        Op::Domain, builder.primitive(Tuple::Occurrence, Tuple::Event, embedding(schema, 1)), present);
    const auto eventPresent = builder.operation(Op::Compose, present, builder.operation(Op::Union, start, completion));
    const auto eventIdentity = builder.restrict(
        builder.primitive(Tuple::Event, Tuple::Event, tupleIdentity(schema, Tuple::Event)), context, eventPresent,
        true);
    const auto ordered = builder.operation(
        Op::Intersect, builder.operation(Op::Union, reference, occurrenceIdentity),
        builder.primitive(Tuple::Occurrence, Tuple::Occurrence, samePipe(schema)));
    const auto native = builder.operation(
        Op::Union,
        builder.operation(
            Op::Union, builder.lift(ordered, start, start), builder.lift(ordered, completion, completion)),
        builder.lift(ordered, start, completion));
    std::size_t memory = 0;
    if (inputs.generators) {
        memory = builder.restrict(builder.input(*inputs.generators), context, present, true);
    } else {
        const auto reads = builder.restrict(builder.input(inputs.reads), context, present, false);
        const auto writes = builder.restrict(builder.input(inputs.writes), context, present, false);
        memory = builder.operation(
            Op::Union,
            builder.operation(Op::Compose, writes, builder.inverse(builder.operation(Op::Union, reads, writes))),
            builder.operation(Op::Compose, reads, builder.inverse(writes)));
    }
    const auto extras = builder.restrict(builder.input(inputs.extras), context, present, true);
    const auto demands = builder.operation(Op::Intersect, builder.operation(Op::Union, memory, extras), reference);
    const auto generators = builder.lift(demands, completion, start);
    const auto prepared = builder.nodes.size();
    const auto step = builder.operation(Op::Union, eventIdentity, builder.operation(Op::Compose, generators, native));
    auto reach = native;
    for (std::size_t pipe = 0; pipe < schema->pipeCount(); ++pipe) {
        reach = builder.operation(Op::Compose, reach, step);
    }
    const auto strict = builder.operation(Op::Difference, reach, eventIdentity);
    const auto alternatives = builder.operation(Op::Union, native, builder.operation(Op::Compose, strict, strict));
    const auto minimum = builder.operation(Op::Difference, generators, alternatives);
    if (!builder.valid) {
        return failure();
    }
    auto result = std::make_shared<SymbolicDemandAnalysis>();
    result->owner = std::move(schema);
    result->arena = std::make_shared<SymbolicArenaIdentity>();
    result->admitted = inputs.context;
    result->graph = std::move(builder.nodes);
    result->prepared = prepared;
    result->presentId = eventPresent;
    result->identityId = eventIdentity;
    result->nativeId = native;
    result->generatorId = generators;
    result->reachId = reach;
    result->strictId = strict;
    result->minimumId = minimum;
    return SymbolicAnalysisHandle(result);
}
} // namespace mlir::pto::frontiersynch
