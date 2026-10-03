// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Physical lowering of numerical repeating-prefix plans. The selected quotient
// supplies uniform reuse; guards and logical identities stay unchanged. No
// iteration unfolding, source-pattern recognizer or runtime dependence scan.
#include "ExplicitPhysicalEmission.h"
#include "DirectEmissionInternal.h"
#include "PTO/Transforms/FrontierSynch/PeriodicEventAssignment.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using PoolKey = std::pair<PIPE, PIPE>;
struct Pool {
    SyncEventPool target;
    SmallVector<std::size_t> records;
    PeriodicEventAssignment assignment;
};
struct Endpoint {
    Operation* operation;
    PoolKey pool;
    std::size_t phase;
};
std::string decimal(const llvm::DynamicAPInt& value)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << value;
    return text;
}
LogicalResult repeatingPrefix(const SyncInput& input, const TraceDemandAnalysis& trace,
                              const DirectEmissionResult& result, std::string& reason)
{
    const auto& selected = *result.selected;
    auto loop = selected.loop;
    auto sites = selected.periodic.sites();
    if (!loop || !isa<func::FuncOp>(loop->getParentOp()) || sites.empty() ||
        input.instructions().size() != sites.size() || selected.sites.size() != sites.size() ||
        !llvm::equal(input.instructions(), sites) ||
        selected.contract.reduction != SelectedReduction::SelectedOrderCovers) {
        reason = "numerical physical allocation needs a complete invocation-owned repeating prefix";
        return failure();
    }
    APInt step;
    if (!isa<IndexType>(loop.getInductionVar().getType()) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isStrictlyPositive() ||
        DataLayout::closest(loop).getTypeSizeInBits(loop.getInductionVar().getType()).getFixedValue() > 64) {
        reason = "numerical physical ordinal lowering needs Index coordinates and positive constant progression";
        return failure();
    }
    for (auto [id, phase] : llvm::enumerate(sites)) {
        if (id >= selected.sites.size() || selected.sites[id] >= trace.sites().size() ||
            trace.sites()[selected.sites[id]].phase != phase || phase->elementOp->getParentOp() != loop) {
            reason = "numerical physical allocation has exceptional or independently guarded payload occurrences";
            return failure();
        }
    }
    return success();
}
LogicalResult preparePools(const SyncInput& input, const DirectEmissionResult& result,
                           std::map<PoolKey, Pool>& pools, std::string& reason)
{
    auto& reduction = result.selected->periodic;
    std::optional<llvm::DynamicAPInt> prefix;
    APInt lower, upper, step;
    auto loop = result.selected->loop;
    if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
        matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) &&
        matchPattern(loop.getStep(), m_ConstantInt(&step))) {
        auto distance = llvm::DynamicAPInt(upper.getSExtValue()) - llvm::DynamicAPInt(lower.getSExtValue());
        auto stride = llvm::DynamicAPInt(step.getSExtValue());
        auto trips = distance <= 0 ? llvm::DynamicAPInt(0) : (distance + stride - 1) / stride;
        prefix = trips * llvm::DynamicAPInt(static_cast<int64_t>(reduction.sites().size()));
    }
    DenseMap<const CompoundInstanceElement*, SyncPhysicalCore> cores;
    for (const auto& phase : input.target().phases()) { cores[phase.phase] = phase.context.core; }
    for (auto id : reduction.retained()) {
        const auto& edge = reduction.generators()[id].edge;
        auto* source = reduction.sites()[edge.source];
        auto* target = reduction.sites()[edge.consumer];
        if (source->kPipeValue == target->kPipeValue) { continue; }
        auto core = cores.lookup(source);
        if (core == SyncPhysicalCore::Unknown || core == SyncPhysicalCore::Conflict || core != cores.lookup(target)) {
            reason = "numerical handoff endpoints have incompatible physical cores";
            return failure();
        }
        PoolKey key{static_cast<PIPE>(source->kPipeValue), static_cast<PIPE>(target->kPipeValue)};
        auto [found, added] = pools.try_emplace(key);
        if (added) {
            auto targetPool = input.target().eventPool(core, key.first, key.second, reason);
            if (failed(targetPool)) { return failure(); }
            found->second.target = std::move(*targetPool);
        } else if (found->second.target.core != core) {
            reason = "numerical directed quotient spans distinct physical-core namespaces";
            return failure();
        }
        found->second.records.push_back(id);
    }
    for (auto& entry : pools) {
        auto& pool = entry.second;
        pool.assignment = assignPeriodicEvents(reduction, pool.records, pool.target.eligibleIds, prefix);
        if (pool.assignment.status != PeriodicAssignmentStatus::Certified) {
            reason = pool.assignment.reason;
            return failure();
        }
        // Budget computation is independent of the supplied capacity. Use its
        // minimum eligible prefix without repeating any quotient queries.
        auto required = static_cast<int64_t>(*pool.assignment.required);
        pool.assignment.eligibleIds.resize(static_cast<std::size_t>(required));
    }
    return success();
}
LogicalResult collectEndpoints(const TraceDemandAnalysis& trace, DirectEmissionResult& result,
                              const std::map<PoolKey, Pool>& pools, SmallVectorImpl<Endpoint>& endpoints)
{
    std::map<std::pair<int64_t, bool>, Endpoint> expected;
    for (const auto& entry : pools) {
        for (auto [phase, edge] : llvm::enumerate(entry.second.assignment.phases)) {
            auto key = static_cast<int64_t>(result.selected->sites[edge.source] * trace.sites().size() +
                                           result.selected->sites[edge.consumer]);
            for (bool set : {false, true}) {
                expected.emplace(std::pair{key, set}, Endpoint{nullptr, entry.first, phase});
            }
        }
    }
    bool invalid = false;
    unsigned barriers = 0, terminals = 0;
    result.pending->walk([&](Operation* operation) {
        if (auto barrier = dyn_cast<BarrierOp>(operation)) {
            ++barriers;
            terminals += barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
        }
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp,
                BarrierSyncOp>(operation)) {
            invalid = true;
        }
        if (!isa<LogicalSetOp, LogicalWaitOp>(operation)) { return; }
        auto key = operation->getAttrOfType<IntegerAttr>("key");
        auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
        auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
        auto found = key ? expected.find({key.getInt(), isa<LogicalSetOp>(operation)}) : expected.end();
        if (found == expected.end() || !source || !target || operation->getNumOperands() != 1 ||
            PoolKey{source.getPipe(), target.getPipe()} != found->second.pool || found->second.operation) {
            invalid = true;
            return;
        }
        found->second.operation = operation;
        endpoints.push_back(found->second);
    });
    if (invalid || endpoints.size() != expected.size() || endpoints.size() != result.sets + result.waits ||
        barriers != result.barriers + 1 || terminals != 1) {
        result.reason = "numerical physical command inventory differs from the sealed direct plan";
        return failure();
    }
    return success();
}
} // namespace
Value lowerPeriodicPhysicalId(OpBuilder& builder, Location loc, Value coordinate, Value lower, int64_t step,
                              const PeriodicEventAssignment& assignment, std::size_t phase)
{
    auto constant = [&](int64_t value) -> Value { return builder.create<arith::ConstantIntOp>(loc, value, 64); };
    Value source = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), coordinate);
    Value origin = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), lower);
    // Every active source satisfies origin<=source in signed Index order.
    // Unsigned subtraction/division represents its full nonnegative distance,
    // including spans larger than INT64_MAX, without signed overflow.
    Value distance = builder.create<arith::SubIOp>(loc, source, origin);
    Value period = builder.create<arith::DivUIOp>(loc, distance, constant(step));
    auto capacity = static_cast<int64_t>(assignment.eligibleIds.size());
    Value modulus = constant(capacity);
    Value residue = builder.create<arith::RemUIOp>(loc, period, modulus);
    Value scaled = builder.create<arith::MulIOp>(loc, residue,
        constant(static_cast<int64_t>(assignment.phases.size() % capacity)));
    Value ordinal = builder.create<arith::AddIOp>(loc, scaled, constant(static_cast<int64_t>(phase % capacity)));
    Value index = builder.create<arith::RemUIOp>(loc, ordinal, modulus);
    Value id = constant(assignment.eligibleIds.front());
    for (std::size_t slot = 1; slot < assignment.eligibleIds.size(); ++slot) {
        Value chosen = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, index, constant(slot));
        id = builder.create<arith::SelectOp>(loc, chosen, constant(assignment.eligibleIds[slot]), id);
    }
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), id);
}
namespace {
ArrayAttr poolEvidence(Builder& builder, const std::map<PoolKey, Pool>& pools)
{
    SmallVector<Attribute> entries;
    for (const auto& entry : pools) {
        auto& pool = entry.second;
        SmallVector<int64_t> eligible(pool.target.eligibleIds.begin(), pool.target.eligibleIds.end());
        SmallVector<int64_t> reserved(pool.target.reservedIds.begin(), pool.target.reservedIds.end());
        SmallVector<int64_t> assigned(pool.assignment.eligibleIds.begin(), pool.assignment.eligibleIds.end());
        SmallVector<Attribute> separations;
        for (const auto& distance : pool.assignment.firstReusableSeparations) {
            separations.push_back(builder.getStringAttr(distance ? decimal(*distance) : "infinite"));
        }
        entries.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source", PipeAttr::get(builder.getContext(), entry.first.first)),
            builder.getNamedAttr("target", PipeAttr::get(builder.getContext(), entry.first.second)),
            builder.getNamedAttr("core", builder.getI64IntegerAttr(static_cast<int64_t>(pool.target.core))),
            builder.getNamedAttr("eligible", builder.getDenseI64ArrayAttr(eligible)),
            builder.getNamedAttr("reserved", builder.getDenseI64ArrayAttr(reserved)),
            builder.getNamedAttr("assigned", builder.getDenseI64ArrayAttr(assigned)),
            builder.getNamedAttr("required", builder.getStringAttr(decimal(*pool.assignment.required))),
            builder.getNamedAttr("uniform_required", builder.getStringAttr(pool.assignment.uniformRequired ?
                decimal(*pool.assignment.uniformRequired) : "infinite")),
            builder.getNamedAttr("reusable_separations", builder.getArrayAttr(separations)),
            builder.getNamedAttr("namespace", builder.getStringAttr(pool.target.namespaceSource))}));
    }
    return builder.getArrayAttr(entries);
}
} // namespace
LogicalResult emitPeriodicPhysical(const SyncInput& input, const TraceDemandAnalysis& trace,
                                  DirectEmissionResult& result, CostLedger& costs)
{
    CostScope allocation(costs, CostStage::Allocation);
    if (!result.emitted || !result.selected || !result.pending || !pendingPlanUnchanged(result)) {
        result.reason = "numerical physical plan changed after endpoint placement";
        return failure();
    }
    if (failed(repeatingPrefix(input, trace, result, result.reason))) { return failure(); }
    std::map<PoolKey, Pool> pools;
    SmallVector<Endpoint> endpoints;
    if (failed(preparePools(input, result, pools, result.reason)) ||
        failed(collectEndpoints(trace, result, pools, endpoints))) { return failure(); }
    APInt step;
    auto originalLoop = result.selected->loop;
    matchPattern(originalLoop.getStep(), m_ConstantInt(&step));
    // Recover this loop by its saved source identity. Unrelated scalar loops
    // may coexist without changing the selected payload occurrence domain.
    auto sourceId = input.target().sourceIdentity(originalLoop);
    scf::ForOp loop;
    result.pending->walk([&](scf::ForOp candidate) {
        auto identity = candidate->getAttrOfType<IntegerAttr>("pto.frontier.source");
        if (sourceId && identity && identity.getInt() == *sourceId) { loop = candidate; }
    });
    if (!loop) { result.reason = "numerical pending plan lost its counted loop"; return failure(); }
    for (auto endpoint : endpoints) {
        auto* operation = endpoint.operation;
        auto& pool = pools.find(endpoint.pool)->second;
        OpBuilder builder(operation);
        auto src = PipeAttr::get(builder.getContext(), endpoint.pool.first);
        auto dst = PipeAttr::get(builder.getContext(), endpoint.pool.second);
        if (pool.assignment.eligibleIds.empty()) {
            // A certified constant zero prefix executes no endpoint. Removing
            // its unreachable templates needs neither an ID nor a new guard.
            if (isa<LogicalSetOp>(operation)) { --result.sets; }
            else { --result.waits; }
        } else if (pool.assignment.eligibleIds.size() == 1) {
            auto id = EventAttr::get(builder.getContext(), static_cast<EVENT>(pool.assignment.eligibleIds.front()));
            if (isa<LogicalSetOp>(operation)) { builder.create<SetFlagOp>(operation->getLoc(), src, dst, id); }
            else { builder.create<WaitFlagOp>(operation->getLoc(), src, dst, id); }
        } else {
            auto id = lowerPeriodicPhysicalId(builder, operation->getLoc(), operation->getOperand(0),
                loop.getLowerBound(), step.getSExtValue(), pool.assignment, endpoint.phase);
            if (isa<LogicalSetOp>(operation)) { builder.create<SetFlagDynOp>(operation->getLoc(), src, dst, id); }
            else { builder.create<WaitFlagDynOp>(operation->getLoc(), src, dst, id); }
        }
        operation->erase();
    }
    if (failed(verify(*result.pending))) {
        result.reason = "numerical physical lowering produced invalid IR";
        return failure();
    }
    Builder builder(result.pending->getContext());
    bool fixedPrefix = !pools.empty() && pools.begin()->second.assignment.handoffCount.has_value();
    result.pending.get()->setAttr("pto.frontier.physical_status", builder.getStringAttr("certified-periodic-pools"));
    result.pending.get()->setAttr("pto.frontier.physical", builder.getDictionaryAttr({
        builder.getNamedAttr("allocator", builder.getStringAttr("uniform-numerical-periodic-budget")),
        builder.getNamedAttr("pools", poolEvidence(builder, pools)),
        builder.getNamedAttr("order", builder.getStringAttr("equal to recorded selected closure")),
        builder.getNamedAttr("scope", builder.getStringAttr(fixedPrefix ?
            "certified constant prefix; no trip expansion" : "all finite prefixes; zero/short executions included")),
        builder.getNamedAttr("counter", builder.getStringAttr("source ordinal modulo eligible pool")),
        builder.getNamedAttr("repair", builder.getStringAttr("none"))}));
    result.privateSelectors = false;
    return success();
}
} // namespace mlir::pto::frontiersynch
