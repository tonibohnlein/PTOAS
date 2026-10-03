// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Uniform reuse over the selected logical relation, then bounded scalar counters.
// Source effects and control remain shared SyncInput records and original SCF.
#include "ExplicitPhysicalEmission.h"
#include "DirectEmissionInternal.h"
#include "StructuredEventCounters.h"
#include "SignedInternal.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Relation = SignedRelationHandle;
using Kind = PeriodicEventKind;
using Tuple = SymbolicTuple;
using PoolKey = std::tuple<SyncPhysicalCore, PIPE, PIPE>;
struct Relations { SignedSpaceHandle space; Relation minimum, native, readiness; };
struct Pool {
    SyncEventPool target;
    Relation demand, successor;
};
Relation take(SignedResult<Relation> result, std::string& reason)
{
    if (!result.succeeded()) { reason = signedDiagnostic(result.status).str(); return {}; }
    return result.value;
}
bool differenceBound(const Relation& relation)
{
    if (!relation) { return false; }
    for (const auto& piece : relation->pieces()) {
        for (const auto& atom : piece.atoms) {
            if (atom.terms.size() == 2 && !(atom.terms[0].axis == atom.terms[1].axis) &&
                atom.terms[0].sign == atom.terms[1].sign) { return false; }
        }
    }
    return true;
}
// Finite Boolean circuits are translated exactly, with one shared valuation
// axis per original SSA outcome. This is not a test for predicate!=False.
FailureOr<Relations> guardedRelations(const TraceDemandAnalysis& trace, const SelectedAnalysis& selected,
                                     std::string& reason)
{
    const auto& guarded = selected.guarded;
    auto nodes = guarded.predicates().nodes();
    SmallVector<Value> parameters;
    DenseMap<Value, unsigned> axes;
    for (const auto& node : nodes) {
        if (node.kind != PredicateKind::Atom) { continue; }
        if (node.evaluation) {
            reason = "finite Boolean adapter needs once-evaluated outcome identities"; return failure();
        }
        if (axes.try_emplace(node.condition, parameters.size()).second) { parameters.push_back(node.condition); }
    }
    SmallVector<SymbolicSite> sites;
    DenseMap<const CompoundInstanceElement*, std::size_t> ids;
    for (auto [id, site] : llvm::enumerate(trace.sites())) {
        sites.push_back({site.phase, {}}); ids[site.phase] = id;
    }
    auto schema = SymbolicSchema::create(sites, parameters, {}, 0);
    if (failed(schema)) { reason = "finite Boolean relation schema is unavailable"; return failure(); }
    auto created = SignedSpace::create(*schema, llvm::DynamicAPInt(1));
    if (!created.succeeded()) { reason = signedDiagnostic(created.status).str(); return failure(); }
    auto space = created.value;
    SignedPiece context;
    context.residues.assign(parameters.size(), llvm::DynamicAPInt(0));
    for (unsigned axis = 0; axis < parameters.size(); ++axis) {
        context.atoms.push_back({{{{SignedAxisRole::Parameter, axis}, 1}}, llvm::DynamicAPInt(1)});
        context.atoms.push_back({{{{SignedAxisRole::Parameter, axis}, -1}}, llvm::DynamicAPInt(0)});
    }
    auto universe = take(SignedRelation::import(space, Tuple::Unit, Tuple::Unit, {context}), reason);
    if (!universe) { return failure(); }
    SmallVector<Relation> predicates;
    for (const auto& node : nodes) {
        Relation value;
        switch (node.kind) {
            case PredicateKind::False:
                value = take(SignedRelation::import(space, Tuple::Unit, Tuple::Unit, {}), reason); break;
            case PredicateKind::True: value = universe; break;
            case PredicateKind::Atom: {
                auto piece = context;
                piece.atoms.push_back({{{{SignedAxisRole::Parameter, axes.lookup(node.condition)}, -1}},
                                       llvm::DynamicAPInt(-1)});
                value = take(SignedRelation::import(space, Tuple::Unit, Tuple::Unit, {piece}), reason); break;
            }
            case PredicateKind::Not: value = take(universe->subtract(predicates[node.first]), reason); break;
            case PredicateKind::And:
                value = take(predicates[node.first]->intersect(predicates[node.second]), reason); break;
            case PredicateKind::Or:
                value = take(predicates[node.first]->unite(predicates[node.second]), reason); break;
        }
        if (!value) { return failure(); }
        predicates.push_back(std::move(value));
    }
    SmallVector<SignedPiece> minimum, native, readiness;
    auto append = [&](SmallVectorImpl<SignedPiece>& output, Relation predicate,
                      std::size_t a, std::size_t b, Kind first, Kind second) {
        for (auto piece : predicate->pieces()) {
            piece.domain = {ids.lookup(guarded.phases()[a]), first};
            piece.range = {ids.lookup(guarded.phases()[b]), second};
            output.push_back(std::move(piece));
        }
    };
    for (const auto& edge : guarded.retained()) {
        append(minimum, predicates[edge.predicate], edge.source, edge.consumer, Kind::Completion, Kind::Start);
    }
    for (std::size_t a = 0; a < guarded.phases().size(); ++a) {
        for (std::size_t b = a; b < guarded.phases().size(); ++b) {
            if (guarded.phases()[a]->kPipeValue == guarded.phases()[b]->kPipeValue) {
                auto present = take(predicates[guarded.occurrences()[a]]->intersect(
                    predicates[guarded.occurrences()[b]]), reason);
                if (!present) { return failure(); }
                append(native, present, a, b, Kind::Start, Kind::Start);
                append(native, present, a, b, Kind::Completion, Kind::Completion);
                append(native, present, a, b, Kind::Start, Kind::Completion);
            }
            auto ready = guarded.completionBeforeStart(a, b);
            if (ready) { append(readiness, predicates[*ready], a, b, Kind::Completion, Kind::Start); }
        }
    }
    Relations result{space, take(SignedRelation::import(space, Tuple::Event, Tuple::Event, minimum), reason),
        take(SignedRelation::import(space, Tuple::Event, Tuple::Event, native), reason),
        take(SignedRelation::import(space, Tuple::Event, Tuple::Event, readiness), reason)};
    if (!result.minimum || !result.native || !result.readiness) { return failure(); }
    return result;
}
Relation strictOrder(const Relations& relations, PipelineType pipe, Kind kind, std::string& reason)
{
    SmallVector<SignedPiece> pieces;
    const auto sites = relations.space->schema()->sites();
    for (const auto& piece : relations.native->pieces()) {
        if (piece.domain.kind == kind && piece.range.kind == kind &&
            sites[*piece.domain.site].phase->kPipeValue == pipe &&
            sites[*piece.range.site].phase->kPipeValue == pipe) { pieces.push_back(piece); }
    }
    auto order = take(SignedRelation::import(relations.space, Tuple::Event, Tuple::Event, pieces), reason);
    if (!order) { return {}; }
    auto present = take(order->domainSet(), reason);
    auto identity = present ? take(signed_detail::identity(present), reason) : Relation{};
    return identity ? take(order->subtract(identity), reason) : Relation{};
}
LogicalResult certifyPool(const Relations& relations, Pool& pool, std::size_t capacity, std::string& reason)
{
    auto source = static_cast<PipelineType>(pool.target.source);
    auto target = static_cast<PipelineType>(pool.target.target);
    auto sources = take(pool.demand->domainSet(), reason);
    auto order = strictOrder(relations, source, Kind::Completion, reason);
    if (order && sources) { order = take(order->restrictDomain(sources), reason); }
    if (order && sources) { order = take(order->restrictRange(sources), reason); }
    auto inverse = take(pool.demand->inverse(), reason);
    auto targets = strictOrder(relations, target, Kind::Start, reason);
    auto image = inverse && order ? take(inverse->compose(order), reason) : Relation{};
    if (image) { image = take(image->compose(pool.demand), reason); }
    auto unordered = image && targets ? take(image->subtract(targets), reason) : Relation{};
    if (!unordered) { return failure(); }
    if (!unordered->empty()) { reason = "selected symbolic handoffs lack strict ordered matching"; return failure(); }
    if (capacity == 0) {
        if (pool.demand->empty()) { return success(); }
        reason = "nonempty selected symbolic pool has zero eligible IDs"; return failure();
    }
    auto intervening = take(order->compose(order), reason);
    pool.successor = intervening ? take(order->subtract(intervening), reason) : Relation{};
    auto later = pool.successor;
    for (std::size_t count = 1; later && count < capacity; ++count) {
        later = take(later->compose(pool.successor), reason);
    }
    auto obligations = later ? take(inverse->compose(later), reason) : Relation{};
    if (!obligations) { return failure(); }
    SmallVector<SignedPiece> pieces(obligations->pieces().begin(), obligations->pieces().end());
    for (auto& piece : pieces) { piece.domain.kind = Kind::Completion; piece.range.kind = Kind::Start; }
    auto reuse = take(SignedRelation::import(relations.space, Tuple::Event, Tuple::Event, pieces), reason);
    auto counterexample = reuse ? take(reuse->subtract(relations.readiness), reason) : Relation{};
    if (!counterexample) { return failure(); }
    if (!counterexample->empty()) {
        auto primitive = counterexample->toSymbolic();
        if (failed(primitive)) {
            reason = "unsupported: symbolic reuse witness cannot be exported"; return failure();
        }
        auto relation = primitive->relation();
        SmallVector<llvm::DynamicAPInt> sample;
        if (!relation.findIntegerSample(sample)) {
            reason = "unsupported: symbolic reuse witness has no recovered integer execution"; return failure();
        }
        llvm::raw_string_ostream message(reason);
        message << "counterexample: consumption-before-republication fails in directed pool "
                << static_cast<unsigned>(pool.target.source) << "->" << static_cast<unsigned>(pool.target.target)
                << " at capacity " << capacity << "; consumer event, next source event, parameters = [";
        llvm::interleaveComma(sample, message);
        message << "]";
        return failure();
    }
    return success();
}
} // namespace
LogicalResult certifySymbolicReuse(SignedRelationHandle demand, SignedRelationHandle native,
    SignedRelationHandle readiness, PipelineType source, PipelineType target,
    std::size_t capacity, std::string& reason)
{
    reason.clear();
    if (!demand || !native || !readiness || demand->space() != native->space() ||
        demand->space() != readiness->space() || !differenceBound(demand) ||
        !differenceBound(native) || !differenceBound(readiness)) {
        reason = "unsupported: uniform reuse requires exact difference-bound relations in one shared space";
        return failure();
    }
    Relations relations{demand->space(), demand, native, readiness};
    Pool pool;
    pool.target.source = static_cast<PIPE>(source);
    pool.target.target = static_cast<PIPE>(target);
    pool.demand = std::move(demand);
    return certifyPool(relations, pool, capacity, reason);
}
LogicalResult emitSymbolicPhysical(const SyncInput& input, const TraceDemandAnalysis& trace,
                                  DirectEmissionResult& result, CostLedger& costs)
{
    CostScope allocation(costs, CostStage::Allocation);
    if (!result.emitted || !result.selected || !result.pending || !pendingPlanUnchanged(result) ||
        result.selected->contract.reduction != SelectedReduction::SelectedOrderCovers) {
        result.reason = "symbolic physical assignment requires the sealed selected cover plan";
        return failure();
    }
    const auto& selected = *result.selected;
    Relations relations;
    if (selected.kind == SelectedAnalysis::Kind::Guarded) {
        auto converted = guardedRelations(trace, selected, result.reason);
        if (failed(converted)) { return failure(); }
        relations = std::move(*converted);
    } else {
        if (!selected.minimum || !selected.native || !selected.reachability) {
            result.reason = "selected symbolic plan has no shared occurrence relations"; return failure();
        }
        relations = {selected.minimum->space(), selected.minimum, selected.native, selected.reachability};
    }
    if (!differenceBound(relations.minimum) || !differenceBound(relations.native) ||
        !differenceBound(relations.readiness)) {
        result.reason = "symbolic physical query is outside the default difference-bound relation interface";
        return failure();
    }
    if (failed(preflightStructuredCounters(*result.pending, result.reason))) { return failure(); }
    const auto sites = relations.space->schema()->sites();
    DenseMap<const CompoundInstanceElement*, SyncPhysicalCore> cores;
    for (const auto& phase : input.target().phases()) { cores[phase.phase] = phase.context.core; }
    std::map<PoolKey, Pool> pools;
    std::map<int64_t, PoolKey> directions;
    std::map<PoolKey, SmallVector<SignedPiece>> demands;
    for (const auto& piece : relations.minimum->pieces()) {
        if (!piece.domain.site || !piece.range.site) {
            result.reason = "selected demand lost its occurrence endpoints"; return failure();
        }
        const auto* source = sites[*piece.domain.site].phase;
        const auto* target = sites[*piece.range.site].phase;
        if (source->kPipeValue == target->kPipeValue) { continue; }
        auto core = cores.lookup(source);
        if (core != cores.lookup(target)) {
            result.reason = "selected symbolic handoff crosses physical cores"; return failure();
        }
        PoolKey key{core, static_cast<PIPE>(source->kPipeValue), static_cast<PIPE>(target->kPipeValue)};
        auto [found, added] = pools.try_emplace(key);
        if (added) {
            auto pool = input.target().eventPool(core, std::get<1>(key), std::get<2>(key), result.reason);
            if (failed(pool)) { return failure(); }
            found->second.target = std::move(*pool);
        }
        demands[key].push_back(piece);
        int64_t identity = *piece.domain.site * sites.size() + *piece.range.site;
        directions.emplace(identity, key);
    }
    SmallVector<StructuredEventPool> counterPools;
    std::map<PoolKey, std::size_t> poolIndices;
    SmallVector<Attribute> evidence;
    Builder builder(result.pending->getContext());
    for (auto& [key, pool] : pools) {
        pool.demand = take(SignedRelation::import(relations.space, Tuple::Event, Tuple::Event, demands[key]),
                           result.reason);
        if (!pool.demand || failed(certifyPool(relations, pool, pool.target.eligibleIds.size(), result.reason))) {
            if (selected.kind == SelectedAnalysis::Kind::Guarded) {
                result.reason += "; context: shared modeled Boolean outcomes; scalar feasibility is not inferred";
            }
            return failure();
        }
        poolIndices[key] = counterPools.size();
        counterPools.push_back({pool.target.source, pool.target.target, pool.target.eligibleIds});
        SmallVector<int64_t> eligible(pool.target.eligibleIds.begin(), pool.target.eligibleIds.end());
        SmallVector<int64_t> reserved(pool.target.reservedIds.begin(), pool.target.reservedIds.end());
        evidence.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source", PipeAttr::get(builder.getContext(), pool.target.source)),
            builder.getNamedAttr("target", PipeAttr::get(builder.getContext(), pool.target.target)),
            builder.getNamedAttr("core", builder.getI64IntegerAttr(static_cast<int64_t>(pool.target.core))),
            builder.getNamedAttr("eligible", builder.getDenseI64ArrayAttr(eligible)),
            builder.getNamedAttr("reserved", builder.getDenseI64ArrayAttr(reserved)),
            builder.getNamedAttr("namespace", builder.getStringAttr(pool.target.namespaceSource))}));
    }
    DenseMap<Operation*, std::size_t> endpoints;
    std::map<int64_t, std::pair<unsigned, unsigned>> inventory;
    bool invalid = false;
    result.pending->walk([&](Operation* operation) {
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp,
                BarrierSyncOp>(operation)) { invalid = true; }
        if (!isa<LogicalSetOp, LogicalWaitOp>(operation)) { return; }
        auto key = operation->getAttrOfType<IntegerAttr>("key");
        auto found = key ? directions.find(key.getInt()) : directions.end();
        if (found == directions.end()) { invalid = true; return; }
        endpoints[operation] = poolIndices.at(found->second);
        auto& counts = inventory[key.getInt()];
        if (isa<LogicalSetOp>(operation)) { ++counts.first; } else { ++counts.second; }
    });
    for (const auto& [key, direction] : directions) {
        auto found = inventory.find(key);
        if (found == inventory.end() || !found->second.first || !found->second.second) { invalid = true; }
    }
    if (invalid || endpoints.size() != result.sets + result.waits) {
        result.reason = "symbolic physical endpoint inventory differs from the sealed selected plan";
        return failure();
    }
    // All exact queries have succeeded. Only the detached plan is changed;
    // endpoint participation controls advance the corresponding bounded state.
    if (failed(lowerStructuredCounters(*result.pending, counterPools, endpoints, result.reason)) ||
        failed(verify(*result.pending))) {
        if (result.reason.empty()) { result.reason = "symbolic physical lowering produced invalid IR"; }
        return failure();
    }
    result.pending.get()->setAttr("pto.frontier.physical_status",
        builder.getStringAttr(selected.kind == SelectedAnalysis::Kind::Guarded ?
            "certified-finite-guarded-pools" : "certified-symbolic-pools"));
    result.pending.get()->setAttr("pto.frontier.physical", builder.getDictionaryAttr({
        builder.getNamedAttr("allocator", builder.getStringAttr("exact-fixed-capacity-successor")),
        builder.getNamedAttr("pools", builder.getArrayAttr(evidence)),
        builder.getNamedAttr("order", builder.getStringAttr("equal to recorded selected closure")),
        builder.getNamedAttr("context", builder.getStringAttr(selected.kind == SelectedAnalysis::Kind::Guarded ?
            "shared modeled Boolean outcomes; scalar feasibility not inferred" : "selected symbolic context")),
        builder.getNamedAttr("reuse", builder.getStringAttr("consumer completion before E-th successor start")),
        builder.getNamedAttr("counter", builder.getStringAttr("invocation-wide separate SET/WAIT modulo state")),
        builder.getNamedAttr("repair", builder.getStringAttr("none"))}));
    result.privateSelectors = false;
    return success();
}
} // namespace mlir::pto::frontiersynch
