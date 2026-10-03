// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact materialization is explicit and separately charged from DAG creation.
// Typed finite-tag joins avoid unrelated arithmetic products; normalization
// and bound-context handling remain in the evaluator's original cache path.
#include "SymbolicComposition.h"
#include "llvm/ADT/SmallString.h"
#include <map>
#include <set>
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using Relation = presburger::PresburgerRelation;
using Op = SymbolicRelationOp;
using llvm::DynamicAPInt;
using Tag = std::pair<int64_t, int64_t>;
SmallVector<std::optional<DynamicAPInt>> equalityConstants(const presburger::IntegerRelation& poly)
{
    SmallVector<std::optional<DynamicAPInt>> constants(poly.getNumVars());
    // Exact propagation through equality rows only. Each successful step
    // derives one previously unknown constant; there are at most numVars steps.
    // Inequalities and unresolved affine rows never authorize a tag split.
    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned index = 0; index < poly.getNumEqualities(); ++index) {
            auto row = poly.getEquality(index);
            DynamicAPInt known = row.back();
            std::optional<unsigned> unknown;
            bool multiple = false;
            for (unsigned column = 0; column < poly.getNumVars(); ++column) {
                if (row[column] == DynamicAPInt(0)) { continue; }
                if (constants[column]) { known += row[column] * *constants[column]; }
                else if (unknown) { multiple = true; break; }
                else { unknown = column; }
            }
            if (multiple || !unknown || (-known) % row[*unknown] != DynamicAPInt(0)) { continue; }
            constants[*unknown] = (-known) / row[*unknown];
            changed = true;
        }
    }
    return constants;
}
std::optional<int64_t> literalConstant(ArrayRef<std::optional<DynamicAPInt>> constants, unsigned position)
{
    if (position >= constants.size() || !constants[position]) { return std::nullopt; }
    const auto& value = *constants[position];
    if (value < DynamicAPInt(std::numeric_limits<int64_t>::min()) ||
        value > DynamicAPInt(std::numeric_limits<int64_t>::max())) { return std::nullopt; }
    return static_cast<int64_t>(value);
}
std::optional<Tag> finiteTag(ArrayRef<std::optional<DynamicAPInt>> constants, unsigned start,
                           SymbolicTuple tuple, SymbolicSchemaHandle schema)
{
    if (tuple != SymbolicTuple::Occurrence && tuple != SymbolicTuple::Event) { return std::nullopt; }
    auto site = literalConstant(constants, start);
    std::optional<int64_t> kind = int64_t(0);
    if (tuple == SymbolicTuple::Event) {
        kind = literalConstant(constants, start + schema->coordinateDepth() + 1);
    }
    if (!site || !kind) { return std::nullopt; }
    return Tag{*site, *kind};
}
Relation nonemptyIntegerDisjuncts(const Relation& relation)
{
    auto result = Relation::getEmpty(relation.getSpace());
    for (const auto& piece : relation.getAllDisjuncts()) {
        // Exact integer emptiness, including parity contradictions. Rational
        // feasibility is insufficient for LLVM19's complement construction.
        if (!piece.isIntegerEmpty()) { result.unionInPlace(piece); }
    }
    return result;
}
void eliminateUnitLocals(presburger::IntegerRelation& piece)
{
    // A local with coefficient +/-1 is uniquely an integer affine expression
    // of the remaining variables. Substitution preserves every constraint and
    // existential solution exactly; nonunit/division locals stay represented.
    while (true) {
        std::optional<std::pair<unsigned, unsigned>> pivot;
        unsigned localStart = piece.getVarKindOffset(presburger::VarKind::Local);
        for (unsigned equality = 0; equality < piece.getNumEqualities() && !pivot; ++equality) {
            for (unsigned local = localStart; local < piece.getNumVars(); ++local) {
                auto coefficient = piece.atEq(equality, local);
                if (coefficient == DynamicAPInt(1) || coefficient == DynamicAPInt(-1)) {
                    pivot = std::make_pair(equality, local);
                    break;
                }
            }
        }
        if (!pivot) { return; }
        auto [equality, local] = *pivot;
        SmallVector<DynamicAPInt> defining(piece.getEquality(equality));
        for (unsigned row = 0; row < piece.getNumEqualities(); ++row) {
            if (row == equality) { continue; }
            auto factor = piece.atEq(row, local) / defining[local];
            if (factor == DynamicAPInt(0)) { continue; }
            for (unsigned column = 0; column < piece.getNumCols(); ++column) {
                piece.atEq(row, column) -= factor * defining[column];
            }
        }
        for (unsigned row = 0; row < piece.getNumInequalities(); ++row) {
            auto factor = piece.atIneq(row, local) / defining[local];
            if (factor == DynamicAPInt(0)) { continue; }
            for (unsigned column = 0; column < piece.getNumCols(); ++column) {
                piece.atIneq(row, column) -= factor * defining[column];
            }
        }
        piece.removeEquality(equality);
        piece.removeVar(local);
    }
}
Relation normalizedSubtrahend(const Relation& relation)
{
    // Discard empty clauses before QE and also any empty clauses produced by
    // its disjunctive representation. This preserves the set exactly and avoids
    // LLVM19 subtraction treating a rationally feasible integer-empty clause
    // as a redundant-constraint witness that can erase valid left points.
    auto nonempty = normalizeSymbolicRelation(relation);
    if (nonempty.hasOnlyDivLocals()) { return nonempty; }
    return normalizeSymbolicRelation(nonempty.computeReprWithOnlyDivLocals());
}
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
FailureOr<Relation> operation(const SymbolicRelationNode& node, const Relation& left, const Relation& right,
    SymbolicSchemaHandle schema, SymbolicTuple intermediate, bool bound)
{
    switch (node.operation) {
        case Op::Union:
            return left.unionSet(right);
        case Op::Intersect:
            return left.intersect(right);
        case Op::Difference: {
            return subtractSymbolicRelations(left, right, std::move(schema), node.domain, node.range, bound);
        }
        case Op::Compose: {
            return composeSymbolicRelations(left, right, std::move(schema),
                node.domain, intermediate, node.range, bound);
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

presburger::PresburgerRelation normalizeSymbolicRelation(presburger::PresburgerRelation relation)
{
    auto simplified = Relation::getEmpty(relation.getSpace());
    for (auto piece : relation.getAllDisjuncts()) {
        eliminateUnitLocals(piece);
        piece.simplify();
        simplified.unionInPlace(piece);
    }
    return nonemptyIntegerDisjuncts(simplified);
}

presburger::PresburgerSet symbolicEndpointSupport(
    const presburger::PresburgerRelation& relation, SymbolicSchemaHandle schema,
    SymbolicTuple endpoint, bool domain, bool bound)
{
    auto space = *schema->space(SymbolicTuple::Unit, endpoint, bound);
    auto result = Relation::getEmpty(space);
    std::set<Tag> tags;
    for (const auto& poly : relation.getAllDisjuncts()) {
        auto constants = equalityConstants(poly);
        auto tag = finiteTag(constants, domain ? 0 : poly.getNumDomainVars(), endpoint, schema);
        if (!tag) { return presburger::PresburgerSet(Relation::getUniverse(space)); }
        tags.insert(*tag);
    }
    for (const auto& tag : tags) {
        presburger::IntegerRelation poly(space);
        poly.addBound(presburger::BoundType::EQ, 0, DynamicAPInt(tag.first));
        if (endpoint == SymbolicTuple::Event) {
            poly.addBound(presburger::BoundType::EQ, schema->coordinateDepth() + 1, DynamicAPInt(tag.second));
        }
        result.unionInPlace(poly);
    }
    return presburger::PresburgerSet(result);
}

presburger::PresburgerRelation composeSymbolicRelations(
    presburger::PresburgerRelation first, const presburger::PresburgerRelation& second,
    SymbolicSchemaHandle schema, SymbolicTuple source, SymbolicTuple intermediate,
    SymbolicTuple target, bool bound)
{
    auto outputSpace = *schema->space(source, target, bound);
    if (first.getNumDisjuncts() == 0 || second.getNumDisjuncts() == 0) {
        return Relation::getEmpty(outputSpace);
    }
    auto generic = [&]() {
        first.compose(second); // Backend introduces fresh existential locals.
        first.setSpace(outputSpace);
        return std::move(first);
    };
    if (intermediate != SymbolicTuple::Occurrence && intermediate != SymbolicTuple::Event) {
        return generic();
    }
    // Union distributes over composition. Equality at the intermediate tuple
    // includes site, and for events kind; preserve all other original axes.
    using Buckets = std::map<Tag, Relation>;
    auto partition = [&](const Relation& relation, bool range, Buckets& buckets) {
        for (const auto& poly : relation.getAllDisjuncts()) {
            unsigned start = range ? poly.getNumDomainVars() : 0;
            auto constants = equalityConstants(poly);
            auto tag = finiteTag(constants, start, intermediate, schema);
            if (!tag) { return false; }
            auto entry = buckets.try_emplace(*tag, Relation::getEmpty(relation.getSpace())).first;
            entry->second.unionInPlace(poly);
        }
        return true;
    };
    Buckets left, right;
    if (!partition(first, true, left) || !partition(second, false, right)) { return generic(); }
    auto result = Relation::getEmpty(outputSpace);
    for (auto& [tag, relation] : left) {
        auto found = right.find(tag);
        if (found == right.end()) { continue; }
        relation.compose(found->second);
        relation.setSpace(outputSpace);
        result.unionInPlace(relation);
    }
    return result;
}

presburger::PresburgerRelation subtractSymbolicRelations(
    const presburger::PresburgerRelation& first, const presburger::PresburgerRelation& second,
    SymbolicSchemaHandle schema, SymbolicTuple source, SymbolicTuple target, bool bound)
{
    auto outputSpace = *schema->space(source, target, bound);
    auto generic = [&]() {
        // Preserve exact integer projection of arbitrary existential RHS locals.
        const auto normalized = normalizedSubtrahend(second);
        auto result = first.subtract(normalized);
        result.setSpace(outputSpace);
        return result;
    };
    if (first.getNumDisjuncts() == 0) { return Relation::getEmpty(outputSpace); }
    if (second.getNumDisjuncts() == 0) {
        auto result = first;
        result.setSpace(outputSpace);
        return result;
    }
    if ((source != SymbolicTuple::Occurrence && source != SymbolicTuple::Event) ||
        (target != SymbolicTuple::Occurrence && target != SymbolicTuple::Event)) { return generic(); }
    using Endpoints = std::pair<Tag, Tag>;
    using Buckets = std::map<Endpoints, Relation>;
    auto partition = [&](const Relation& relation, Buckets& buckets) {
        for (const auto& poly : relation.getAllDisjuncts()) {
            auto constants = equalityConstants(poly);
            auto a = finiteTag(constants, 0, source, schema);
            auto b = finiteTag(constants, poly.getNumDomainVars(), target, schema);
            if (!a || !b) { return false; }
            auto entry = buckets.try_emplace(Endpoints{*a, *b}, Relation::getEmpty(relation.getSpace())).first;
            entry->second.unionInPlace(poly);
        }
        return true;
    };
    Buckets left, right;
    if (!partition(first, left) || !partition(second, right)) { return generic(); }
    auto result = Relation::getEmpty(outputSpace);
    // Distinct fixed endpoint tuples are disjoint. Normalize only matching
    // RHS buckets, rather than projecting unrelated native/cross-pipe paths.
    for (auto& [endpoints, relation] : left) {
        auto found = right.find(endpoints);
        if (found != right.end()) {
            const auto& rhs = found->second;
            const auto normalized = normalizedSubtrahend(rhs);
            relation = relation.subtract(normalized);
        }
        relation.setSpace(outputSpace);
        result.unionInPlace(relation);
    }
    return result;
}

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
            auto result = operation(node, **left, **right, analysis->owner,
                analysis->graph[node.first].range, bool(context));
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
