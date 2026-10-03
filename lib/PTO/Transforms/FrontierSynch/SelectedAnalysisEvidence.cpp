// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Durable typed semantics, independent of diagnostic formatting. Periodic
// closure is kept as its compact weighted graph, never an eager threshold dump.
#include "DirectEmissionInternal.h"
#include "GeneralQueries.h"
#include "FixedBodyUpper.h"
#include "PeriodicNativeGraph.h"
#include "SingleStreamLoop.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
namespace {
std::string decimal(const llvm::DynamicAPInt& value)
{
    std::string text;
    llvm::raw_string_ostream os(text);
    os << value;
    return text;
}
ArrayAttr integers(Builder& builder, ArrayRef<std::size_t> values)
{
    SmallVector<Attribute> attrs;
    for (auto value : values) {
        attrs.push_back(builder.getI64IntegerAttr(value));
    }
    return builder.getArrayAttr(attrs);
}
ArrayAttr relation(Builder& builder, SignedRelationHandle value)
{
    SmallVector<Attribute> pieces;
    for (const auto& piece : value->pieces()) {
        SmallVector<Attribute> rows, residues;
        for (const auto& residue : piece.residues) {
            residues.push_back(builder.getStringAttr(decimal(residue)));
        }
        for (const auto& atom : piece.atoms) {
            SmallVector<Attribute> terms;
            for (const auto& term : atom.terms) {
                terms.push_back(builder.getArrayAttr(
                    {builder.getI64IntegerAttr(static_cast<int64_t>(term.axis.role)),
                     builder.getI64IntegerAttr(term.axis.index), builder.getI64IntegerAttr(term.sign)}));
            }
            rows.push_back(builder.getDictionaryAttr(
                {builder.getNamedAttr("terms", builder.getArrayAttr(terms)),
                 builder.getNamedAttr("bound", builder.getStringAttr(decimal(atom.bound)))}));
        }
        auto tag = [&](const SignedTag& tag) {
            return builder.getArrayAttr(
                {builder.getI64IntegerAttr(tag.site ? static_cast<int64_t>(*tag.site) : -1),
                 builder.getI64IntegerAttr(tag.kind ? static_cast<int64_t>(*tag.kind) : -1)});
        };
        pieces.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("source", tag(piece.domain)), builder.getNamedAttr("target", tag(piece.range)),
             builder.getNamedAttr("residues", builder.getArrayAttr(residues)),
             builder.getNamedAttr("rows", builder.getArrayAttr(rows))}));
    }
    return builder.getArrayAttr(pieces);
}
ArrayAttr relation(Builder& builder, const presburger::PresburgerRelation& value)
{
    SmallVector<Attribute> pieces;
    for (const auto& poly : value.getAllDisjuncts()) {
        auto rows = [&](bool equalities) {
            SmallVector<Attribute> result;
            unsigned count = equalities ? poly.getNumEqualities() : poly.getNumInequalities();
            for (unsigned i = 0; i < count; ++i) {
                SmallVector<Attribute> row;
                for (const auto& coefficient : equalities ? poly.getEquality(i) : poly.getInequality(i)) {
                    row.push_back(builder.getStringAttr(decimal(coefficient)));
                }
                result.push_back(builder.getArrayAttr(row));
            }
            return builder.getArrayAttr(result);
        };
        pieces.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("domain", builder.getI64IntegerAttr(poly.getNumDomainVars())),
            builder.getNamedAttr("range", builder.getI64IntegerAttr(poly.getNumRangeVars())),
            builder.getNamedAttr("symbols", builder.getI64IntegerAttr(poly.getNumSymbolVars())),
            builder.getNamedAttr("locals", builder.getI64IntegerAttr(poly.getNumLocalVars())),
            builder.getNamedAttr("equalities", rows(true)), builder.getNamedAttr("inequalities", rows(false))}));
    }
    return builder.getArrayAttr(pieces);
}
void explicitEvidence(Builder& builder, const SelectedAnalysis& selected, NamedAttrList& attrs)
{
    SmallVector<Attribute> demands, summaries, pipes;
    for (auto id : selected.explicitReduction.retained()) {
        const auto& edge = selected.generators[id];
        demands.push_back(integers(builder, {selected.sites[edge.source], selected.sites[edge.consumer]}));
    }
    for (const auto& summary : selected.explicitReduction.summaries()) {
        summaries.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr(
                 "pipe",
                 builder.getI64IntegerAttr(static_cast<int64_t>(selected.explicitReduction.pipes()[summary.pipe]))),
             builder.getNamedAttr("pipe_column", builder.getI64IntegerAttr(summary.pipe)),
             builder.getNamedAttr("rank", builder.getI64IntegerAttr(summary.rank)),
             builder.getNamedAttr("S", integers(builder, summary.S)),
             builder.getNamedAttr("T", integers(builder, summary.T))}));
    }
    for (auto pipe : selected.explicitReduction.pipes()) {
        pipes.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(pipe)));
    }
    attrs.set("representation", builder.getStringAttr("explicit-ranks"));
    attrs.set("pipes", builder.getArrayAttr(pipes));
    attrs.set("summaries", builder.getArrayAttr(summaries));
    attrs.set("demands", builder.getArrayAttr(demands));
}
void periodicEvidence(Builder& builder, const SelectedAnalysis& selected, NamedAttrList& attrs)
{
    SmallVector<Attribute> demands, edges;
    auto& periodic = selected.periodic;
    detail::QuotientGraph native(periodic.vertexCount());
    detail::addNative(native, periodic.sites());
    auto edgeAttr = [&](std::size_t a, std::size_t b, const llvm::DynamicAPInt& distance) {
        return builder.getArrayAttr(
            {builder.getI64IntegerAttr(a), builder.getI64IntegerAttr(b), builder.getStringAttr(decimal(distance))});
    };
    for (const auto& edge : native.edges) {
        edges.push_back(edgeAttr(edge.source, edge.target, edge.weight));
    }
    for (const auto& record : periodic.generators()) {
        edges.push_back(edgeAttr(2 * record.edge.source + 1, 2 * record.edge.consumer, record.edge.distance));
    }
    for (auto id : periodic.retained()) {
        const auto& edge = periodic.generators()[id].edge;
        demands.push_back(edgeAttr(selected.sites[edge.source], selected.sites[edge.consumer], edge.distance));
    }
    attrs.set("representation", builder.getStringAttr("weighted-periodic-quotient"));
    attrs.set("distance_unit", builder.getStringAttr("iterations"));
    attrs.set("event_encoding", builder.getStringAttr("start=2*site; completion=2*site+1"));
    attrs.set("vertices", builder.getI64IntegerAttr(periodic.vertexCount()));
    attrs.set("edges", builder.getI64IntegerAttr(periodic.edgeCount()));
    attrs.set("native_edge_count", builder.getI64IntegerAttr(native.edges.size()));
    attrs.set("closure_graph", builder.getArrayAttr(edges));
    attrs.set("demands", builder.getArrayAttr(demands));
}
void signedEvidence(Builder& builder, const SelectedAnalysis& selected, NamedAttrList& attrs)
{
    SmallVector<Attribute> depths, parameters;
    for (const auto& site : selected.structured->schema()->sites()) {
        depths.push_back(builder.getI64IntegerAttr(site.coordinates.size()));
    }
    for (Value parameter : selected.structured->schema()->parameters()) {
        parameters.push_back(builder.getI64IntegerAttr(cast<BlockArgument>(parameter).getArgNumber()));
    }
    attrs.set("representation", builder.getStringAttr("signed-piece-union"));
    attrs.set("period", builder.getStringAttr(decimal(selected.minimum->space()->period())));
    attrs.set("context", relation(builder, selected.context));
    attrs.set("native", relation(builder, selected.native));
    attrs.set("reachability", relation(builder, selected.reachability));
    attrs.set("demands", relation(builder, selected.minimum));
    attrs.set("coordinate_depths", builder.getArrayAttr(depths));
    attrs.set("parameters", builder.getArrayAttr(parameters));
}
void guardedEvidence(
    Builder& builder, const SelectedAnalysis& selected, const TraceDemandAnalysis& source,
    const DenseMap<Operation*, int64_t>& ids, NamedAttrList& attrs)
{
    SmallVector<Attribute> predicates, demands;
    for (const auto& node : selected.guarded.predicates().nodes()) {
        NamedAttrList predicate;
        predicate.set("kind", builder.getI64IntegerAttr(static_cast<int64_t>(node.kind)));
        predicate.set("first", builder.getI64IntegerAttr(node.first));
        predicate.set("second", builder.getI64IntegerAttr(node.second));
        if (node.kind == PredicateKind::Atom) {
            if (auto argument = dyn_cast<BlockArgument>(node.condition)) {
                predicate.set("argument", builder.getI64IntegerAttr(argument.getArgNumber()));
            } else {
                auto value = cast<OpResult>(node.condition);
                predicate.set("value_anchor", builder.getI64IntegerAttr(ids.lookup(value.getOwner())));
                predicate.set("result", builder.getI64IntegerAttr(value.getResultNumber()));
            }
        }
        predicates.push_back(predicate.getDictionary(builder.getContext()));
    }
    DenseMap<const CompoundInstanceElement*, std::size_t> sites;
    for (auto [id, site] : llvm::enumerate(source.sites())) {
        sites[site.phase] = id;
    }
    for (const auto& edge : selected.guarded.retained()) {
        demands.push_back(integers(
            builder, {sites.lookup(selected.guarded.phases()[edge.source]),
                      sites.lookup(selected.guarded.phases()[edge.consumer]), edge.predicate}));
    }
    attrs.set("representation", builder.getStringAttr("finite-guarded-circuit"));
    attrs.set("predicates", builder.getArrayAttr(predicates));
    attrs.set("demands", builder.getArrayAttr(demands));
}
} // namespace
Attribute retainSelectedAnalysis(
    func::FuncOp function, const TraceDemandAnalysis& source, const DirectEmissionResult& result)
{
    Builder builder(function.getContext());
    NamedAttrList attrs;
    const auto& selected = *result.selected;
    attrs.set("requirements", builder.getStringAttr("shared-modeled"));
    bool upper = selected.contract.closure == SelectedClosure::SoundUpper;
    attrs.set("selected_closure", builder.getStringAttr(upper ? "sound-upper" : "modeled-requirements"));
    attrs.set("demand_name", builder.getStringAttr(upper ? "F_hat" : "F_star"));
    if (upper) {
        attrs.set("lower_baseline", builder.getStringAttr("native-only; no inferred physical conflict facts"));
        attrs.set("upper_certificate", builder.getStringAttr("covers original modeled closure"));
        if (selected.upper) {
            SmallVector<Attribute> terms;
            for (const auto& term : selected.upper->excessTerms()) {
                terms.push_back(builder.getDictionaryAttr({
                    builder.getNamedAttr("consumer", builder.getI64IntegerAttr(selected.sites[term.consumer])),
                    builder.getNamedAttr("pipe", builder.getI64IntegerAttr(
                        static_cast<int64_t>(selected.periodic.pipes()[term.pipe]))),
                    builder.getNamedAttr("count", builder.getStringAttr(decimal(term.count))),
                    builder.getNamedAttr("offset", builder.getStringAttr(decimal(term.offset))),
                    builder.getNamedAttr("startup", builder.getStringAttr(decimal(term.startup)))}));
            }
            attrs.set("excess_terms", builder.getArrayAttr(terms));
            attrs.set("excess_formula", builder.getStringAttr("sum max(0,count*j+offset), 0<=j<T"));
            attrs.set("quadratic_numerator", builder.getStringAttr(decimal(selected.upper->quadraticNumerator())));
            attrs.set("quadratic_denominator", builder.getI64IntegerAttr(2));
        } else {
            attrs.set("excess_formula", builder.getStringAttr(
                "cardinality of selected upper C-to-I relation; parent numeric rank counting not materialized"));
        }
    }
    attrs.set("status", builder.getStringAttr("prepared-logical"));
    attrs.set("quality", builder.getStringAttr("selected-order-covers"));
    attrs.set("route", builder.getStringAttr(selected.route));
    attrs.set("logical_order_equality", builder.getStringAttr(
        realizedOrderStatement(result)));
    attrs.set("sites", integers(builder, selected.sites));
    DenseMap<Operation*, int64_t> ids;
    int64_t next = 0;
    function.walk([&](Operation* operation) { ids[operation] = next++; });
    SmallVector<Attribute> occurrences;
    for (auto [id, site] : llvm::enumerate(source.sites())) {
        SmallVector<Attribute> loops;
        for (Operation* loop : site.loops) {
            loops.push_back(builder.getI64IntegerAttr(ids.lookup(loop)));
        }
        occurrences.push_back(builder.getDictionaryAttr(
            {builder.getNamedAttr("site", builder.getI64IntegerAttr(id)),
             builder.getNamedAttr("loops", builder.getArrayAttr(loops))}));
    }
    attrs.set("occurrences", builder.getArrayAttr(occurrences));
    switch (selected.kind) {
        case SelectedAnalysis::Kind::BoundaryLoop:
            attrs.set("representation", builder.getStringAttr("single-stream-affine-boundaries"));
            attrs.set("single_stream", selected.boundaryLoop->evidence(function.getContext()));
            break;
        case SelectedAnalysis::Kind::Explicit:
            explicitEvidence(builder, selected, attrs);
            break;
        case SelectedAnalysis::Kind::Periodic:
            periodicEvidence(builder, selected, attrs);
            break;
        case SelectedAnalysis::Kind::General: {
            attrs.set("representation", builder.getStringAttr("uniform-presburger"));
            attrs.set("requirements", builder.getStringAttr("shared-modeled"));
            attrs.set("quality", builder.getStringAttr("selected-order-covers"));
            const auto& query = *selected.general;
            attrs.set("context", relation(builder, query.context));
            attrs.set("presence", relation(builder, query.present));
            attrs.set("minimum", relation(builder, query.minimum));
            attrs.set("native", relation(builder, query.native));
            attrs.set("reachability", relation(builder, query.reachability));
            SmallVector<Attribute> depths, parameters;
            for (const auto& site : query.schema->sites()) {
                depths.push_back(builder.getI64IntegerAttr(site.coordinates.size()));
            }
            for (Value parameter : query.schema->parameters()) {
                parameters.push_back(builder.getI64IntegerAttr(cast<BlockArgument>(parameter).getArgNumber()));
            }
            attrs.set("coordinate_depths", builder.getArrayAttr(depths));
            attrs.set("parameters", builder.getArrayAttr(parameters));
            break;
        }
        case SelectedAnalysis::Kind::Signed:
            signedEvidence(builder, selected, attrs);
            break;
        case SelectedAnalysis::Kind::Guarded:
            guardedEvidence(builder, selected, source, ids, attrs);
            break;
    }
    return attrs.getDictionary(function.getContext());
}
} // namespace mlir::pto::frontiersynch
