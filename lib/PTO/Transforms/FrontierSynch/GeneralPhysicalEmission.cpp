// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Optional exact Presburger reuse adapter. It consumes the selected relation,
// then reuses the existing bounded SCF counter lowering on the detached clone.
// No source effect reconstruction, trace expansion or capacity repair occurs.
#include "GeneralPhysicalEmission.h"
#include "GeneralQueries.h"
#include "StructuredEventCounters.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using R = presburger::PresburgerRelation;
using I = presburger::IntegerRelation;
using Int = llvm::DynamicAPInt;
using Kind = PeriodicEventKind;
using PoolKey = std::tuple<SyncPhysicalCore, PIPE, PIPE>;
struct Pool {
    SyncEventPool target;
    R demand;
    Pool(SyncEventPool target, R demand) : target(std::move(target)), demand(std::move(demand)) {}
};
R canonical(R value, SymbolicSchemaHandle schema)
{
    value = value.simplify();
    value.setSpace(*schema->space(SymbolicTuple::Event, SymbolicTuple::Event));
    return value;
}
R compose(R first, const R& second, SymbolicSchemaHandle schema)
{
    first.compose(second);
    return canonical(std::move(first), schema);
}
R subtract(const R& first, const R& second, SymbolicSchemaHandle schema)
{
    auto represented = second.hasOnlyDivLocals() ? second : second.computeReprWithOnlyDivLocals();
    return canonical(first.subtract(represented), schema);
}
FailureOr<std::pair<std::size_t, std::size_t>> tags(const I& piece, SymbolicSchemaHandle schema)
{
    auto source = piece.getConstantBound64(presburger::BoundType::EQ, 0);
    auto target = piece.getConstantBound64(presburger::BoundType::EQ, piece.getNumDomainVars());
    if (!source || !target || *source < 0 || *target < 0 ||
        static_cast<std::size_t>(*source) >= schema->sites().size() ||
        static_cast<std::size_t>(*target) >= schema->sites().size()) { return failure(); }
    return std::pair{static_cast<std::size_t>(*source), static_cast<std::size_t>(*target)};
}
FailureOr<R> strictOrder(const R& native, SymbolicSchemaHandle schema, PipelineType pipe, Kind kind)
{
    R order = R::getEmpty(native.getSpace()), identity = R::getEmpty(native.getSpace());
    unsigned width = native.getNumDomainVars(), kindAxis = schema->coordinateDepth() + 1;
    for (auto piece : native.getAllDisjuncts()) {
        auto endpoints = tags(piece, schema);
        if (failed(endpoints)) { return failure(); }
        if (schema->sites()[endpoints->first].phase->kPipeValue != pipe ||
            schema->sites()[endpoints->second].phase->kPipeValue != pipe) { continue; }
        piece.addBound(presburger::BoundType::EQ, kindAxis, Int(static_cast<int64_t>(kind)));
        piece.addBound(presburger::BoundType::EQ, width + kindAxis, Int(static_cast<int64_t>(kind)));
        order.unionInPlace(piece);
        for (unsigned axis = 0; axis < width; ++axis) {
            SmallVector<Int> row(piece.getNumCols(), Int(0));
            row[axis] = Int(1);
            row[width + axis] = Int(-1);
            piece.addEquality(row);
        }
        identity.unionInPlace(piece);
    }
    return subtract(order, identity, schema);
}
R rearmKinds(const R& obligations, SymbolicSchemaHandle schema)
{
    R result = R::getEmpty(obligations.getSpace());
    const unsigned kind = schema->coordinateDepth() + 1;
    for (auto piece : obligations.getAllDisjuncts()) {
        // Earlier consumer Start becomes Completion (+1); later producer
        // Completion becomes Start (-1). Substitute old coordinates in rows.
        const unsigned targetKind = piece.getNumDomainVars() + kind, constant = piece.getNumCols() - 1;
        for (unsigned row = 0; row < piece.getNumEqualities(); ++row) {
            piece.atEq(row, constant) += -piece.atEq(row, kind) + piece.atEq(row, targetKind);
        }
        for (unsigned row = 0; row < piece.getNumInequalities(); ++row) {
            piece.atIneq(row, constant) += -piece.atIneq(row, kind) + piece.atIneq(row, targetKind);
        }
        result.unionInPlace(piece);
    }
    return canonical(std::move(result), schema);
}
LogicalResult reportCounterexample(const R& relation, PipelineType source, PipelineType target,
                                  std::size_t capacity, std::string& reason)
{
    for (const auto& piece : relation.getAllDisjuncts()) {
        auto sample = piece.findIntegerSample();
        if (!sample) { continue; }
        llvm::raw_string_ostream stream(reason);
        stream << "counterexample: general consumption-before-republication fails in directed pool "
               << static_cast<unsigned>(source) << "->" << static_cast<unsigned>(target)
               << " at capacity " << capacity << "; consumer/producer events, parameters, locals = [";
        llvm::interleaveComma(*sample, stream);
        stream << "]";
        return failure();
    }
    reason = "unsupported: general reuse counterexample has no recovered integer witness";
    return failure();
}
DictionaryAttr poolEvidence(Builder& builder, const SyncEventPool& pool)
{
    SmallVector<int64_t> eligible(pool.eligibleIds.begin(), pool.eligibleIds.end());
    SmallVector<int64_t> reserved(pool.reservedIds.begin(), pool.reservedIds.end());
    return builder.getDictionaryAttr({
        builder.getNamedAttr("source", PipeAttr::get(builder.getContext(), pool.source)),
        builder.getNamedAttr("target", PipeAttr::get(builder.getContext(), pool.target)),
        builder.getNamedAttr("core", builder.getI64IntegerAttr(static_cast<int64_t>(pool.core))),
        builder.getNamedAttr("eligible", builder.getDenseI64ArrayAttr(eligible)),
        builder.getNamedAttr("reserved", builder.getDenseI64ArrayAttr(reserved)),
        builder.getNamedAttr("namespace", builder.getStringAttr(pool.namespaceSource))});
}
} // namespace
LogicalResult certifyGeneralReuse(SymbolicSchemaHandle schema, const R& demand, const R& native,
    const R& readiness, PipelineType source, PipelineType target, std::size_t capacity, std::string& reason)
{
    reason.clear();
    auto expected = schema ? schema->space(SymbolicTuple::Event, SymbolicTuple::Event) :
                            FailureOr<presburger::PresburgerSpace>(failure());
    if (!schema || failed(expected) || !demand.getSpace().isAligned(*expected) ||
        !native.getSpace().isAligned(*expected) || !readiness.getSpace().isAligned(*expected)) {
        reason = "unsupported: general reuse requires one original Event relation schema";
        return failure();
    }
    auto order = strictOrder(native, schema, source, Kind::Completion);
    auto targets = strictOrder(native, schema, target, Kind::Start);
    if (failed(order) || failed(targets)) {
        reason = "unsupported: general native order does not preserve literal source-site tags";
        return failure();
    }
    auto sources = demand.getDomainSet();
    R selectedOrder = canonical(order->intersectDomain(sources).intersectRange(sources), schema);
    R inverse = demand;
    inverse.inverse();
    R image = compose(compose(inverse, selectedOrder, schema), demand, schema);
    if (!subtract(image, *targets, schema).isIntegerEmpty()) {
        reason = "selected general handoffs lack strict ordered matching";
        return failure();
    }
    if (capacity == 0) {
        if (demand.isIntegerEmpty()) { return success(); }
        reason = "counterexample: nonempty selected general pool has zero eligible IDs";
        return failure();
    }
    R successor = subtract(selectedOrder, compose(selectedOrder, selectedOrder, schema), schema);
    R later = successor;
    for (std::size_t count = 1; count < capacity; ++count) { later = compose(later, successor, schema); }
    R obligations = rearmKinds(compose(inverse, later, schema), schema);
    R counterexample = subtract(obligations, readiness, schema);
    return counterexample.isIntegerEmpty() ? success() :
        reportCounterexample(counterexample, source, target, capacity, reason);
}
LogicalResult emitGeneralPhysical(const SyncInput& input, const TraceDemandAnalysis& trace,
                                 DirectEmissionResult& result, CostLedger& costs)
{
    CostScope allocation(costs, CostStage::Allocation);
    if (!result.emitted || !result.selected || !result.pending || !pendingPlanUnchanged(result) ||
        result.selected->kind != SelectedAnalysis::Kind::General ||
        result.selected->contract.reduction != SelectedReduction::SelectedOrderCovers) {
        result.reason = "general physical assignment requires the sealed selected cover plan";
        return failure();
    }
    auto queries = generalQueries(*result.selected, result.reason);
    if (failed(queries) || failed(preflightStructuredCounters(*result.pending, result.reason))) { return failure(); }
    const auto& relations = **queries;
    auto schema = relations.schema;
    const auto sites = schema->sites();
    if (sites.size() != trace.sites().size() ||
        (sites.size() && sites.size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) / sites.size())) {
        result.reason = "general physical site identity extent is unsupported";
        return failure();
    }
    for (auto [id, site] : llvm::enumerate(sites)) {
        if (site.phase != trace.sites()[id].phase) {
            result.reason = "general physical schema differs from the shared original phase identities";
            return failure();
        }
    }
    DenseMap<const CompoundInstanceElement*, SyncPhysicalCore> cores;
    for (const auto& phase : input.target().phases()) { cores[phase.phase] = phase.context.core; }
    std::map<PoolKey, Pool> pools;
    std::map<int64_t, PoolKey> directions;
    for (const auto& piece : relations.minimum.getAllDisjuncts()) {
        auto endpoints = tags(piece, schema);
        if (failed(endpoints)) { result.reason = "general demand lost literal occurrence endpoints"; return failure(); }
        auto* source = sites[endpoints->first].phase;
        auto* target = sites[endpoints->second].phase;
        if (source->kPipeValue == target->kPipeValue) { continue; }
        auto core = cores.lookup(source);
        if (core != cores.lookup(target)) {
            result.reason = "general handoff crosses physical cores"; return failure();
        }
        PoolKey key{core, static_cast<PIPE>(source->kPipeValue), static_cast<PIPE>(target->kPipeValue)};
        auto found = pools.find(key);
        if (found == pools.end()) {
            auto pool = input.target().eventPool(core, std::get<1>(key), std::get<2>(key), result.reason);
            if (failed(pool)) { return failure(); }
            found = pools.emplace(key, Pool(std::move(*pool), R::getEmpty(relations.minimum.getSpace()))).first;
        }
        found->second.demand.unionInPlace(piece);
        directions.emplace(static_cast<int64_t>(endpoints->first * sites.size() + endpoints->second), key);
    }
    SmallVector<StructuredEventPool> counterPools;
    SmallVector<Attribute> evidence;
    std::map<PoolKey, std::size_t> indices;
    Builder builder(result.pending->getContext());
    for (const auto& [key, pool] : pools) {
        if (failed(certifyGeneralReuse(schema, pool.demand, relations.native, relations.reachability,
            static_cast<PipelineType>(pool.target.source), static_cast<PipelineType>(pool.target.target),
            pool.target.eligibleIds.size(), result.reason))) { return failure(); }
        indices[key] = counterPools.size();
        counterPools.push_back({pool.target.source, pool.target.target, pool.target.eligibleIds});
        evidence.push_back(poolEvidence(builder, pool.target));
    }
    DenseMap<Operation*, std::size_t> endpoints;
    std::map<int64_t, std::pair<unsigned, unsigned>> inventory;
    bool invalid = false;
    result.pending->walk([&](Operation* operation) {
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp,
                BarrierSyncOp>(operation)) { invalid = true; }
        if (!isa<LogicalSetOp, LogicalWaitOp>(operation)) { return; }
        auto identity = operation->getAttrOfType<IntegerAttr>("key");
        auto direction = identity ? directions.find(identity.getInt()) : directions.end();
        if (direction == directions.end()) { invalid = true; return; }
        auto src = operation->getAttrOfType<PipeAttr>("src_pipe");
        auto dst = operation->getAttrOfType<PipeAttr>("dst_pipe");
        if (!src || !dst || src.getPipe() != std::get<1>(direction->second) ||
            dst.getPipe() != std::get<2>(direction->second)) { invalid = true; return; }
        endpoints[operation] = indices.at(direction->second);
        auto& counts = inventory[identity.getInt()];
        if (isa<LogicalSetOp>(operation)) { ++counts.first; } else { ++counts.second; }
    });
    for (const auto& [identity, direction] : directions) {
        auto found = inventory.find(identity);
        if (found == inventory.end() || !found->second.first || !found->second.second) { invalid = true; }
    }
    if (invalid || endpoints.size() != result.sets + result.waits) {
        result.reason = "general physical endpoint inventory differs from the sealed selected plan";
        return failure();
    }
    if (failed(lowerStructuredCounters(*result.pending, counterPools, endpoints, result.reason)) ||
        failed(verify(*result.pending))) {
        if (result.reason.empty()) { result.reason = "general physical lowering produced invalid IR"; }
        return failure();
    }
    result.pending.get()->setAttr("pto.frontier.physical_status", builder.getStringAttr("certified-general-pools"));
    result.pending.get()->setAttr("pto.frontier.physical", builder.getDictionaryAttr({
        builder.getNamedAttr("allocator", builder.getStringAttr("exact-presburger-fixed-capacity-successor")),
        builder.getNamedAttr("pools", builder.getArrayAttr(evidence)),
        builder.getNamedAttr("order", builder.getStringAttr("equal to recorded selected closure")),
        builder.getNamedAttr("reuse", builder.getStringAttr("consumer completion before E-th successor start")),
        builder.getNamedAttr("counter", builder.getStringAttr("invocation-wide separate SET/WAIT modulo state")),
        builder.getNamedAttr("repair", builder.getStringAttr("none"))}));
    result.privateSelectors = false;
    return success();
}
} // namespace mlir::pto::frontiersynch
