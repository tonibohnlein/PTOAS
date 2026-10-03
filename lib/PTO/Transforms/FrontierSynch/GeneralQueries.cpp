// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact uniform Presburger query/selector adapters over the shared source model.
#include "GeneralQueries.h"
#include "StructuredInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>
namespace mlir::pto::frontiersynch {
LogicalResult qualifyCountedReaderPattern(SelectedAnalysis&, SymbolicSchemaHandle, const SyncInput&,
                                         bool, bool, CostLedger&, std::string&);
namespace {
using R = presburger::PresburgerRelation;
using I = presburger::IntegerRelation;
using Int = llvm::DynamicAPInt;
using Tuple = SymbolicTuple;
FailureOr<Int> coefficientImpl(Value value, scf::ForOp outer, DenseMap<Value, std::optional<Int>>& cache);
FailureOr<Int> coefficient(Value value, scf::ForOp outer, DenseMap<Value, std::optional<Int>>& cache)
{
    auto found = cache.find(value);
    if (found != cache.end()) {
        if (found->second) { return *found->second; }
        return failure();
    }
    auto result = coefficientImpl(value, outer, cache);
    cache[value] = succeeded(result) ? std::optional<Int>(*result) : std::nullopt;
    return result;
}
FailureOr<Int> coefficientImpl(Value value, scf::ForOp outer, DenseMap<Value, std::optional<Int>>& cache)
{
    if (value == outer.getInductionVar()) { return Int(1); }
    auto* definition = value.getDefiningOp();
    if (!definition) {
        auto argument = dyn_cast<BlockArgument>(value);
        return argument && argument.getOwner()->getParentOp() != outer &&
            !outer->isProperAncestor(argument.getOwner()->getParentOp()) ?
            FailureOr<Int>(Int(0)) : FailureOr<Int>(failure());
    }
    if (!outer->isProperAncestor(definition)) { return Int(0); }
    if (auto add = dyn_cast<arith::AddIOp>(definition)) {
        auto a = coefficient(add.getLhs(), outer, cache), b = coefficient(add.getRhs(), outer, cache);
        if (succeeded(a) && succeeded(b)) { return *a + *b; }
    }
    if (auto sub = dyn_cast<arith::SubIOp>(definition)) {
        auto a = coefficient(sub.getLhs(), outer, cache), b = coefficient(sub.getRhs(), outer, cache);
        if (succeeded(a) && succeeded(b)) { return *a - *b; }
    }
    if (auto mul = dyn_cast<arith::MulIOp>(definition)) {
        APInt number;
        Value variable;
        if (matchPattern(mul.getLhs(), m_ConstantInt(&number))) { variable = mul.getRhs(); }
        else if (matchPattern(mul.getRhs(), m_ConstantInt(&number))) { variable = mul.getLhs(); }
        if (variable) {
            auto a = coefficient(variable, outer, cache);
            if (succeeded(a)) { return *a * structured::integer(number); }
        }
    }
    return failure();
}
R canonical(R relation, SymbolicSchemaHandle schema)
{
    relation = relation.simplify();
    relation.setSpace(*schema->space(Tuple::Event, Tuple::Event));
    return relation;
}
R compose(R a, const R& b, SymbolicSchemaHandle schema)
{
    a.compose(b); return canonical(std::move(a), schema);
}
R subtract(const R& a, const R& b, SymbolicSchemaHandle schema)
{
    auto right = b.hasOnlyDivLocals() ? b : b.computeReprWithOnlyDivLocals();
    return canonical(a.subtract(right), schema);
}
FailureOr<R> materialize(SymbolicEvaluator& evaluator, const SymbolicRoot& root)
{
    auto value = evaluator.materialize(root);
    if (failed(value) || value->boundContext()) { return failure(); }
    return value->relation();
}
FailureOr<SymbolicAnalysisHandle> sourcePremises(StructuredInputHandle source)
{
    return SymbolicDemandAnalysis::build(source->schema(), source->relations());
}
R empty(SymbolicSchemaHandle schema)
{
    return R::getEmpty(*schema->space(Tuple::Event, Tuple::Event));
}
FailureOr<R> crossing(const R& relation, ArrayRef<std::size_t> owners)
{
    R result = R::getEmpty(relation.getSpace());
    for (const auto& poly : relation.getAllDisjuncts()) {
        auto a = poly.getConstantBound64(presburger::BoundType::EQ, 0);
        auto b = poly.getConstantBound64(presburger::BoundType::EQ, poly.getNumDomainVars());
        if (!a || !b || *a < 0 || *b < 0 || static_cast<std::size_t>(*a) >= owners.size() ||
            static_cast<std::size_t>(*b) >= owners.size()) { return failure(); }
        if (owners[*a] == 0 && owners[*b] == 1) { result.unionInPlace(poly); }
    }
    return result;
}
} // namespace
AnalysisContract generalAnalysisContract()
{
    AnalysisContract result;
    result.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
        interfaceBit(DemandInterface::RegionQueries) | interfaceBit(DemandInterface::UniformMembership);
    return result;
}
LogicalResult buildGeneralCounted(SelectedAnalysis& selected, const SyncInput& input,
                                 CostLedger& costs, std::string& reason)
{
    auto source = selected.structured;
    if (!source || selected.sites.size() != 2 || !selected.loop) {
        reason = "general counted source structure not established"; return failure();
    }
    auto schema = source->schema();
    auto sites = schema->sites();
    auto reader = selected.sites[1];
    if (reader >= sites.size() || sites[reader].coordinates.size() != 2) {
        reason = "general counted reader coordinates not established"; return failure();
    }
    auto argument = dyn_cast<BlockArgument>(sites[reader].coordinates[1]);
    auto inner = argument ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
    DenseMap<Value, std::optional<Int>> coefficients;
    auto slope = inner ? coefficient(inner.getUpperBound(), selected.loop, coefficients) : FailureOr<Int>(failure());
    if (failed(slope) || *slope < Int(0)) {
        reason = "general counted row bound is not original nondecreasing affine a*i+c"; return failure();
    }
    // SCF's zero-origin domain executes max(0,a*i+c) readers. No c>=0
    // assumption is inserted: negative original parameters retain their empty
    // prefix rows. The appendix's all-event/cover rules hold for these actual
    // nonnegative row lengths; monotone clipping preserves first/last selection.
    if (failed(qualifyCountedReaderPattern(selected, schema, input,
        source->relations().extras.relation().isIntegerEmpty(), true, costs, reason))) { return failure(); }
    CostScope backend(costs, CostStage::Backend);
    auto premises = sourcePremises(source);
    auto evaluator = succeeded(premises) ? SymbolicEvaluator::uniform(*premises) :
                                          FailureOr<SymbolicEvaluator>(failure());
    if (failed(evaluator)) { reason = "general counted Presburger backend unavailable"; return failure(); }
    auto present = materialize(*evaluator, (*premises)->presence());
    auto native = materialize(*evaluator, (*premises)->native());
    if (failed(present) || failed(native)) {
        reason = "general counted original premise materialization failed"; return failure();
    }
    R reach = empty(schema);
    auto relationSpace = *schema->space(Tuple::Event, Tuple::Event);
    unsigned width = relationSpace.getNumDomainVars();
    unsigned depth = schema->coordinateDepth();
    auto make = [&](std::size_t a, std::size_t b, unsigned s, unsigned t) {
        I poly(relationSpace);
        poly.addBound(presburger::BoundType::EQ, 0, Int(static_cast<int64_t>(a)));
        poly.addBound(presburger::BoundType::EQ, width, Int(static_cast<int64_t>(b)));
        poly.addBound(presburger::BoundType::EQ, depth + 1, Int(s));
        poly.addBound(presburger::BoundType::EQ, width + depth + 1, Int(t));
        for (unsigned i = sites[a].coordinates.size(); i < depth; ++i) {
            poly.addBound(presburger::BoundType::EQ, i + 1, Int(0));
        }
        for (unsigned i = sites[b].coordinates.size(); i < depth; ++i) {
            poly.addBound(presburger::BoundType::EQ, width + i + 1, Int(0));
        }
        return poly;
    };
    auto order = [&](I& poly, unsigned axis, bool strict, bool equal) {
        SmallVector<Int> row(poly.getNumCols(), Int(0));
        row[width + axis + 1] = Int(1); row[axis + 1] = Int(-1);
        row.back() = Int(strict ? -1 : 0);
        if (equal) { poly.addEquality(row); } else { poly.addInequality(row); }
    };
    auto writer = selected.sites[0];
    for (auto a : {writer, reader}) {
        for (auto b : {writer, reader}) {
            for (unsigned s : {0U, 1U}) {
                for (unsigned t : {0U, 1U}) {
                    auto poly = make(a, b, s, t);
                    bool acquire = s == 1 && t == 0;
                    if (a == reader && b == reader && !acquire) {
                        order(poly, 0, true, false); reach.unionInPlace(poly);
                        poly = make(a, b, s, t); order(poly, 0, false, true); order(poly, 1, false, false);
                    } else { order(poly, 0, (a == reader && b == writer) || (a == b && acquire), false); }
                    reach.unionInPlace(poly);
                }
            }
        }
    }
    reach = reach.intersectDomain(present->getRangeSet()).intersectRange(present->getRangeSet());
    reach = canonical(std::move(reach), schema);
    // The qualified complete signature admits precisely three cover families.
    // Keep original presence (including clipped empty rows) on both ends.
    auto executed = [&](I poly) {
        return canonical(R(poly).intersectDomain(present->getRangeSet())
            .intersectRange(present->getRangeSet()), schema);
    };
    auto successor = [&](I& poly, unsigned axis) {
        SmallVector<Int> row(poly.getNumCols(), Int(0));
        row[width + axis + 1] = Int(1); row[axis + 1] = Int(-1); row.back() = Int(-1);
        poly.addEquality(row);
    };
    auto firstPoly = make(writer, reader, 1, 0);
    order(firstPoly, 0, false, true);
    firstPoly.addBound(presburger::BoundType::EQ, width + 2, Int(0));
    auto first = executed(firstPoly);
    auto nextReader = make(reader, reader, 1, 0);
    order(nextReader, 0, false, true); successor(nextReader, 1);
    auto laterReader = executed(nextReader).getDomainSet();
    auto lastPoly = make(reader, writer, 1, 0); successor(lastPoly, 0);
    auto last = executed(lastPoly);
    last = subtract(last, last.intersectDomain(laterReader), schema);
    auto bypassPoly = make(writer, writer, 1, 0); successor(bypassPoly, 0);
    auto bypass = executed(bypassPoly);
    bypass = subtract(bypass, bypass.intersectDomain(first.getDomainSet()), schema);
    auto minimum = canonical(first.unionSet(last).unionSet(bypass), schema);
    auto queries = std::make_shared<GeneralQueries>(schema, source->relations().context.relation(),
                                                   *present, *native, minimum, reach);
    queries->premises = *premises;
    selected.general = queries; selected.kind = SelectedAnalysis::Kind::General;
    selected.route = "regional-affine-counted-readers";
    selected.contract = generalAnalysisContract();
    return success();
}
FailureOr<std::shared_ptr<const GeneralQueries>> generalQueries(const SelectedAnalysis& selected, std::string& reason)
{
    if (selected.general) { return selected.general; }
    if (selected.generalInterchange) { return selected.generalInterchange; }
    if (!selected.minimum || !selected.reachability || !selected.structured) {
        reason = "child did not supply common-schema arithmetic event queries"; return failure();
    }
    auto source = selected.structured;
    auto premises = sourcePremises(source);
    auto evaluator = succeeded(premises) ? SymbolicEvaluator::uniform(*premises) :
                                          FailureOr<SymbolicEvaluator>(failure());
    if (failed(evaluator)) { reason = "child Presburger backend unavailable"; return failure(); }
    auto present = materialize(*evaluator, (*premises)->presence());
    auto native = materialize(*evaluator, (*premises)->native());
    auto minimum = selected.minimum->toSymbolic(), reach = selected.reachability->toSymbolic();
    if (failed(present) || failed(native) || failed(minimum) || failed(reach) ||
        minimum->schema() != source->schema() || reach->schema() != source->schema()) {
        reason = "child arithmetic interchange identity mismatch"; return failure();
    }
    auto queries = std::make_shared<GeneralQueries>(source->schema(), source->relations().context.relation(),
        *present, *native, minimum->relation(), reach->relation());
    queries->premises = *premises;
    selected.generalInterchange = queries;
    return selected.generalInterchange;
}
FailureOr<SelectedAnalysisHandle> composeGeneralRegions(func::FuncOp function, StructuredInputHandle input,
    ArrayRef<SelectedAnalysisHandle> children, CostLedger& costs, std::string& reason)
{
    CostScope mergeScope(costs, CostStage::Merge);
    auto schema = input->schema();
    SmallVector<std::size_t> owners(schema->sites().size(), children.size());
    for (auto [owner, child] : llvm::enumerate(children)) {
        for (auto id : child->sites) {
            if (id >= owners.size() || owners[id] != children.size()) {
                reason = "general merge child ownership mismatch"; return failure();
            }
            owners[id] = owner;
        }
    }
    if (!function || (input->sites().empty() && !children.empty())) {
        reason = "general merge original scoped source is missing"; return failure();
    }
    DenseMap<const CompoundInstanceElement*, std::size_t> tags;
    for (auto [id, site] : llvm::enumerate(schema->sites())) { tags.try_emplace(site.phase, id); }
    SmallVector<std::size_t> active;
    for (const auto& site : input->sites()) {
        auto found = tags.find(site.phase);
        if (found == tags.end()) {
            reason = "general merge original phase identity missing"; return failure();
        }
        auto id = found->second;
        if (owners[id] == children.size()) {
            reason = "general merge children do not cover original scoped sites"; return failure();
        }
        active.push_back(id);
    }
    auto premises = sourcePremises(input);
    auto evaluator = succeeded(premises) ? SymbolicEvaluator::uniform(*premises) :
                                          FailureOr<SymbolicEvaluator>(failure());
    if (failed(evaluator)) { reason = "general merge Presburger backend unavailable"; return failure(); }
    auto native = materialize(*evaluator, (*premises)->native());
    auto identity = materialize(*evaluator, (*premises)->identity());
    auto generators = materialize(*evaluator, (*premises)->generators());
    auto present = materialize(*evaluator, (*premises)->presence());
    if (failed(native) || failed(identity) || failed(generators) || failed(present)) {
        reason = "general merge primitive normalization failed"; return failure();
    }
    SmallVector<std::shared_ptr<const GeneralQueries>> queries;
    for (const auto& child : children) {
        auto query = generalQueries(*child, reason);
        if (failed(query) || (*query)->schema != schema ||
            !(*query)->context.isEqual(input->relations().context.relation())) {
            reason = "general child query uses a different original schema/context"; return failure();
        }
        queries.push_back(*query);
    }
    // Balanced sequence composition, never a whole-parent closure iteration.
    // Forward crossings cannot make an internal child cover redundant.
    std::function<FailureOr<std::pair<R, R>>(std::size_t, std::size_t)> merge;
    merge = [&](std::size_t begin, std::size_t end) -> FailureOr<std::pair<R, R>> {
        if (end - begin == 1) { return std::make_pair(queries[begin]->reachability, queries[begin]->minimum); }
        auto middle = begin + (end - begin) / 2;
        auto left = merge(begin, middle), right = merge(middle, end);
        if (failed(left) || failed(right)) { return failure(); }
        SmallVector<std::size_t> sides(owners.size(), 2);
        for (auto id : active) {
            if (owners[id] >= begin && owners[id] < middle) { sides[id] = 0; }
            if (owners[id] >= middle && owners[id] < end) { sides[id] = 1; }
        }
        auto g = crossing(*generators, sides), n = crossing(*native, sides);
        if (failed(g) || failed(n)) { return failure(); }
        auto edges = g->unionSet(*n);
        auto bridge = compose(compose(left->first, edges, schema), right->first, schema);
        auto reach = canonical(left->first.unionSet(right->first).unionSet(bridge), schema);
        auto strict = subtract(reach, *identity, schema);
        // Only paths with endpoints in a crossing generator can remove it.
        auto fromCrossing = strict.intersectDomain(g->getDomainSet());
        auto toCrossing = strict.intersectRange(g->getRangeSet());
        auto alternatives = compose(fromCrossing, toCrossing, schema);
        auto retained = subtract(*g, native->unionSet(alternatives), schema);
        auto minimum = canonical(left->second.unionSet(right->second).unionSet(retained), schema);
        return std::make_pair(std::move(reach), std::move(minimum));
    };
    if (children.empty()) { reason = "general merge has no child queries"; return failure(); }
    auto merged = merge(0, children.size());
    if (failed(merged)) { reason = "general crossing finite tags not established"; return failure(); }
    auto result = std::make_shared<SelectedAnalysis>();
    result->kind = SelectedAnalysis::Kind::General; result->route = "presburger-region-composition";
    result->structured = input;
    result->general = std::make_shared<GeneralQueries>(schema, input->relations().context.relation(),
        *present, *native, merged->second, merged->first);
    result->sites = std::move(active);
    llvm::append_range(result->regionalChildren, children);
    result->contract = generalAnalysisContract();
    if (llvm::any_of(children, [](const auto& child) {
        return child->contract.closure == SelectedClosure::SoundUpper;
    })) { result->contract.closure = SelectedClosure::SoundUpper; }
    return SelectedAnalysisHandle(result);
}
} // namespace mlir::pto::frontiersynch
