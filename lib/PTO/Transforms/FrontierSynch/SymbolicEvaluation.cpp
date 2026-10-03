// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact materialization is explicit and separately charged from DAG creation.
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
#include "llvm/ADT/SmallString.h"
namespace mlir::pto::frontiersynch {
namespace {
using Relation = presburger::PresburgerRelation;
using Op = SymbolicRelationOp;
using llvm::DynamicAPInt;
DynamicAPInt integer(IntegerAttr attribute)
{
    const auto type = dyn_cast<IntegerType>(attribute.getType());
    const bool signedValue = !type || (!type.isUnsigned() && type.getWidth() != 1);
    llvm::SmallString<64> digits;
    attribute.getValue().toString(digits, 10, signedValue);
    llvm::StringRef text(digits);
    const bool negative = text.consume_front("-");
    DynamicAPInt result(0);
    for (char digit : text) {
        result = result * 10 + (digit - '0');
    }
    return negative ? -result : result;
}
Relation specialize(const SymbolicPrimitive& primitive, const SymbolicContext& context)
{
    auto space = *primitive.schema()->space(primitive.domain(), primitive.range(), true);
    auto result = Relation::getEmpty(space);
    for (auto piece : primitive.relation().getAllDisjuncts()) {
        piece.setAndEliminate(piece.getNumDimVars(), context.values);
        piece.setSpaceExceptLocals(space);
        result.unionInPlace(piece);
    }
    return result;
}
FailureOr<Relation> operation(const SymbolicRelationNode& node, const Relation& left, const Relation& right)
{
    switch (node.operation) {
        case Op::Union:
            return left.unionSet(right);
        case Op::Intersect:
            return left.intersect(right);
        case Op::Difference: {
            // Stable public contract: normalize non-division existential RHS
            // locals exactly, never via IntegerRelation::projectOut.
            const auto normalized = right.hasOnlyDivLocals() ? right : right.computeReprWithOnlyDivLocals();
            return left.subtract(normalized);
        }
        case Op::Compose: {
            auto result = left;
            result.compose(right); // MLIR copies operands and introduces fresh locals.
            return result;
        }
        case Op::Domain: {
            // LLVM 19 inverse() leaves an empty relation's top-level space unchanged.
            // Lift the set directly, preserving empty-domain typing as well.
            auto domain = right;
            domain.convertVarKind(
                presburger::VarKind::Range, 0, right.getNumRangeVars(), presburger::VarKind::Domain, 0);
            domain.insertVarInPlace(presburger::VarKind::Range, 0, left.getNumRangeVars());
            return left.intersect(domain);
        }
        case Op::Range:
            return left.intersectRange(presburger::PresburgerSet(right));
        case Op::Context: {
            auto context = right;
            context.insertVarInPlace(presburger::VarKind::Domain, 0, left.getNumDomainVars());
            context.insertVarInPlace(presburger::VarKind::Range, 0, left.getNumRangeVars());
            return left.intersect(context);
        }
        default:
            return failure();
    }
}
} // namespace

FailureOr<SymbolicContext> decodeSymbolicBindings(
    SymbolicSchemaHandle schema, ArrayRef<ImmutableParameterBinding> bindings)
{
    if (!schema || bindings.size() != schema->parameters().size()) {
        return failure();
    }
    DenseMap<Value, IntegerAttr> supplied;
    for (const auto& binding : bindings) {
        if (!binding.value || !binding.result || binding.value.getType() != binding.result.getType() ||
            !llvm::is_contained(schema->parameters(), binding.value) ||
            !supplied.try_emplace(binding.value, binding.result).second) {
            return failure();
        }
    }
    SymbolicContext context;
    for (auto value : schema->parameters()) {
        auto bound = supplied.lookup(value);
        if (!bound) {
            return failure();
        }
        context.bindings.push_back({value, bound});
        context.values.push_back(integer(bound));
    }
    return context;
}

FailureOr<SymbolicEvaluator> SymbolicEvaluator::uniform(SymbolicAnalysisHandle analysis)
{
    if (!analysis || !analysis->owner ||
        !analysis->arena || analysis->graph.empty()) {
        return failure();
    }
    SymbolicEvaluator result;
    result.analysis = std::move(analysis);
    result.cache.resize(result.analysis->graph.size());
    return result;
}
FailureOr<SymbolicEvaluator> SymbolicEvaluator::bind(
    SymbolicAnalysisHandle analysis, ArrayRef<ImmutableParameterBinding> bindings)
{
    auto result = uniform(std::move(analysis));
    if (failed(result) || bindings.size() != result->analysis->owner->parameters().size()) {
        return failure();
    }
    auto decoded = decodeSymbolicBindings(result->analysis->owner, bindings);
    if (failed(decoded)) {
        return failure();
    }
    auto context = std::make_shared<SymbolicContext>(std::move(*decoded));
    context->admitted = result->analysis->admitted.relation().containsPoint(context->values);
    result->context = std::move(context);
    return result;
}
FailureOr<const Relation*> SymbolicEvaluator::evaluate(std::size_t id)
{
    if (id >= cache.size()) {
        return failure();
    }
    if (cache[id]) {
        return &*cache[id];
    }
    const auto& node = analysis->graph[id];
    std::optional<Relation> value;
    if (node.operation == Op::Input) {
        value = context ? specialize(*node.primitive, *context) : node.primitive->relation();
    } else {
        auto left = evaluate(node.first);
        if (failed(left)) {
            return failure();
        }
        if (node.operation == Op::Inverse) {
            value = **left;
            value->inverse();
        } else {
            auto right = evaluate(node.second);
            if (failed(right)) {
                return failure();
            }
            auto result = operation(node, **left, **right);
            if (failed(result)) {
                return failure();
            }
            value = std::move(*result);
        }
    }
    // Simplification belongs to explicit evaluation, never expression creation.
    // Its cost and possible normalized growth are not bounded by the k+6 claim.
    value = value->simplify();
    value->setSpace(*analysis->owner->space(node.domain, node.range, bool(context)));
    cache[id] = std::move(value);
    return &*cache[id];
}
FailureOr<MaterializedSymbolicRelation> SymbolicEvaluator::materialize(const SymbolicRoot& root)
{
    if (!analysis || root.arena != analysis->arena || root.id >= analysis->graph.size() ||
        (context && !context->admitted)) {
        return failure();
    }
    auto value = evaluate(root.id);
    if (failed(value)) {
        return failure();
    }
    const auto& node = analysis->graph[root.id];
    MaterializedSymbolicRelation result;
    result.owner = analysis->owner;
    result.source = node.domain;
    result.target = node.range;
    result.root = root;
    result.context = context;
    result.value = **value;
    return result;
}
FailureOr<SmallVector<DynamicAPInt>> SymbolicEvaluator::point(const SymbolicEvent& event) const
{
    if (!analysis || !admitted() || event.site >= analysis->owner->sites().size() ||
        event.coordinates.size() != analysis->owner->sites()[event.site].coordinates.size() ||
        (event.kind != PeriodicEventKind::Start && event.kind != PeriodicEventKind::Completion)) {
        return failure();
    }
    SmallVector<DynamicAPInt> result;
    result.push_back(DynamicAPInt(static_cast<std::int64_t>(event.site)));
    result.append(event.coordinates.begin(), event.coordinates.end());
    result.resize(analysis->owner->coordinateDepth() + 1, DynamicAPInt(0));
    result.push_back(DynamicAPInt(event.kind == PeriodicEventKind::Start ? 0 : 1));
    return result;
}
FailureOr<bool> SymbolicEvaluator::contains(const SymbolicEvent& event)
{
    auto coordinates = point(event);
    if (failed(coordinates)) {
        return failure();
    }
    auto present = evaluate(analysis->presentId);
    if (failed(present)) {
        return failure();
    }
    return (*present)->containsPoint(*coordinates);
}
FailureOr<bool> SymbolicEvaluator::containsPair(
    const MaterializedSymbolicRelation& relation, const SymbolicEvent& source, const SymbolicEvent& consumer)
{
    if (!analysis || !admitted() || relation.root.arena != analysis->arena || relation.owner != analysis->owner ||
        relation.context != context || relation.source != SymbolicTuple::Event ||
        relation.target != SymbolicTuple::Event) {
        return failure();
    }
    auto a = contains(source), b = contains(consumer);
    if (failed(a) || failed(b) || !*a || !*b) {
        return failure();
    }
    auto coordinates = point(source), target = point(consumer);
    coordinates->append(target->begin(), target->end());
    return relation.value.containsPoint(*coordinates);
}
FailureOr<bool> SymbolicEvaluator::query(
    const SymbolicRoot& root, const SymbolicEvent& source, const SymbolicEvent& consumer)
{
    // Validate presence first, including failed/inadmissible contexts and empty executions.
    auto a = contains(source), b = contains(consumer);
    if (failed(a) || failed(b) || !*a || !*b) {
        return failure();
    }
    if (!analysis || root.arena != analysis->arena || root.id >= cache.size()) {
        return failure();
    }
    auto relation = evaluate(root.id);
    if (failed(relation)) {
        return failure();
    }
    auto coordinates = point(source), target = point(consumer);
    coordinates->append(target->begin(), target->end());
    return (*relation)->containsPoint(*coordinates);
}
FailureOr<bool> SymbolicEvaluator::reaches(const SymbolicEvent& source, const SymbolicEvent& consumer)
{
    if (!analysis) {
        return failure();
    }
    return query(analysis->reachability(), source, consumer);
}
FailureOr<bool> SymbolicEvaluator::minimumDemand(const SymbolicEvent& source, const SymbolicEvent& consumer)
{
    if (!analysis) {
        return failure();
    }
    return query(analysis->minimum(), source, consumer);
}
std::size_t SymbolicEvaluator::cachedNodes() const
{
    return llvm::count_if(cache, [](const auto& value) { return value.has_value(); });
}
} // namespace mlir::pto::frontiersynch
