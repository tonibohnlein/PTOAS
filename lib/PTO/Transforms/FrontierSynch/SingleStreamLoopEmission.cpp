// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Uniform single-generation boundary protocol. The certificate below records
// the actually constructed pending operations, not a trusted attribute. Exact
// unchanged-plan validation prevents mutable helper callers bypassing its cuts.
#include "SingleStreamLoop.h"
#include "DirectEmissionInternal.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/PeriodicEventAssignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Matchers.h"
#include <array>
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
struct SingleStreamPending {
    SmallVector<Operation*> commands;
};
namespace {
void logical(OpBuilder& builder, Location location, bool set, PIPE source, PIPE target, int64_t key,
             ValueRange coordinates = {})
{
    auto src = PipeAttr::get(builder.getContext(), source);
    auto dst = PipeAttr::get(builder.getContext(), target);
    auto identity = builder.getI64IntegerAttr(key);
    if (set) { builder.create<LogicalSetOp>(location, src, dst, identity, coordinates); }
    else { builder.create<LogicalWaitOp>(location, src, dst, identity, coordinates); }
}
LogicalResult emitSequentialBoundaries(const SingleStreamLoop& plan, const TraceDemandAnalysis& trace,
    IRMapping& mapping, DirectEmissionResult& result)
{
    if (plan.children.size() != 2 || plan.children[0]->bodies.empty() || plan.children[1]->bodies.empty()) {
        return failure();
    }
    const bool exclusive = plan.protocol == SingleStreamProtocol::ExclusiveArmBoundaries;
    auto a = cast<scf::ForOp>(mapping.lookup(plan.children[0]->loop));
    auto b = cast<scf::ForOp>(mapping.lookup(plan.children[1]->loop));
    auto* cut = mapping.lookupOrNull(plan.publicationCut);
    auto* exit = mapping.lookup(trace.sites()[plan.exit].anchor);
    if (exclusive ? cut != mapping.lookupOrNull(plan.participation) :
        (cut != a.getOperation() || a->getBlock() != b->getBlock())) { return failure(); }
    const auto count = trace.sites().size();
    const auto entryA = plan.entry * count + plan.children[0]->first();
    const auto entryB = plan.entry * count + plan.children[1]->first();
    const auto exitA = plan.children[0]->last() * count + plan.exit;
    const auto exitB = plan.children[1]->last() * count + plan.exit;
    OpBuilder publication(cut); auto location = a.getLoc();
    auto zero = publication.create<arith::ConstantIndexOp>(location, 0);
    auto positiveA = publication.create<arith::CmpIOp>(location, arith::CmpIPredicate::sgt, a.getUpperBound(), zero);
    auto positiveB = publication.create<arith::CmpIOp>(location, arith::CmpIPredicate::sgt, b.getUpperBound(), zero);
    // Mutually exclusive ACTUAL logical pairs, not a relabelled destination.
    if (exclusive) {
        for (unsigned child = 0; child < 2; ++child) {
            OpBuilder armPublication(child ? b.getOperation() : a.getOperation());
            auto positive = child ? positiveB : positiveA;
            auto guard = armPublication.create<scf::IfOp>(location, positive, false);
            auto armBuilder = guard.getThenBodyBuilder();
            logical(armBuilder, location, true, PIPE::PIPE_MTE2, PIPE::PIPE_V, child ? entryB : entryA);
        }
    } else {
        auto firstChoice = publication.create<scf::IfOp>(location, positiveA, true);
        auto firstA = firstChoice.getThenBodyBuilder();
        logical(firstA, location, true, PIPE::PIPE_MTE2, PIPE::PIPE_V, entryA);
        auto fallback = firstChoice.getElseBodyBuilder();
        auto secondChoice = fallback.create<scf::IfOp>(location, positiveB, false);
        auto firstB = secondChoice.getThenBodyBuilder();
        logical(firstB, location, true, PIPE::PIPE_MTE2, PIPE::PIPE_V, entryB);
    }
    for (unsigned child = 0; child < 2; ++child) {
        auto loop = child ? b : a; auto coordinate = loop.getInductionVar();
        auto* payload = mapping.lookup(trace.sites()[plan.children[child]->first()].anchor);
        OpBuilder incoming(payload);
        auto later = incoming.create<arith::CmpIOp>(location, arith::CmpIPredicate::sgt, coordinate, zero);
        Value local = later;
        if (child && !exclusive) { local = incoming.create<arith::OrIOp>(location, later, positiveA); }
        auto localGuard = incoming.create<scf::IfOp>(location, local, false);
        auto localBuilder = localGuard.getThenBodyBuilder();
        localBuilder.create<BarrierOp>(location, PipeAttr::get(loop.getContext(), PIPE::PIPE_V));
        auto first = incoming.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, coordinate, zero);
        auto incomingGuard = incoming.create<scf::IfOp>(location, first, false);
        auto firstBuilder = incomingGuard.getThenBodyBuilder();
        if (child && !exclusive) {
            auto absentA = firstBuilder.create<scf::IfOp>(location, positiveA, true);
            auto absentBuilder = absentA.getElseBodyBuilder();
            logical(absentBuilder, location, false, PIPE::PIPE_MTE2, PIPE::PIPE_V, entryB);
        } else { logical(firstBuilder, location, false, PIPE::PIPE_MTE2, PIPE::PIPE_V, child ? entryB : entryA); }
        for (auto site : ArrayRef<std::size_t>(plan.children[child]->bodies).drop_front()) {
            OpBuilder lexical(mapping.lookup(trace.sites()[site].anchor));
            lexical.create<BarrierOp>(location, PipeAttr::get(loop.getContext(), PIPE::PIPE_V));
        }
        auto* lastPayload = mapping.lookup(trace.sites()[plan.children[child]->last()].anchor);
        OpBuilder outgoing(lastPayload); outgoing.setInsertionPointAfter(lastPayload);
        auto one = outgoing.create<arith::ConstantIndexOp>(location, 1);
        // On an executed iteration i+1<=its positive bound: no signed overflow.
        auto next = outgoing.create<arith::AddIOp>(location, coordinate, one);
        auto last = outgoing.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, next, loop.getUpperBound());
        auto lastGuard = outgoing.create<scf::IfOp>(location, last, false);
        auto lastBuilder = lastGuard.getThenBodyBuilder();
        if (!child && !exclusive) {
            auto absentB = lastBuilder.create<scf::IfOp>(location, positiveB, true);
            auto absentBuilder = absentB.getElseBodyBuilder();
            logical(absentBuilder, location, true, PIPE::PIPE_V, PIPE::PIPE_MTE3, exitA, ValueRange{coordinate});
        } else { logical(lastBuilder, location, true, PIPE::PIPE_V, PIPE::PIPE_MTE3,
                         child ? exitB : exitA, ValueRange{coordinate}); }
    }
    OpBuilder acquisition(exit);
    auto finalChoice = acquisition.create<scf::IfOp>(location,
        exclusive ? mapping.lookup(plan.activation) : Value(positiveB), true);
    auto waitB = finalChoice.getThenBodyBuilder();
    auto oneB = waitB.create<arith::ConstantIndexOp>(location, 1);
    auto lastB = waitB.create<arith::SubIOp>(location, exclusive ? a.getUpperBound() : b.getUpperBound(), oneB);
    logical(waitB, location, false, PIPE::PIPE_V, PIPE::PIPE_MTE3, exclusive ? exitA : exitB, ValueRange{lastB});
    auto waitA = finalChoice.getElseBodyBuilder();
    auto oneA = waitA.create<arith::ConstantIndexOp>(location, 1);
    auto lastA = waitA.create<arith::SubIOp>(location, exclusive ? b.getUpperBound() : a.getUpperBound(), oneA);
    logical(waitA, location, false, PIPE::PIPE_V, PIPE::PIPE_MTE3, exclusive ? exitB : exitA, ValueRange{lastA});
    result.sets = 4; result.waits = 4; result.barriers = plan.bodies.size(); result.privateSelectors = false;
    return success();
}
LogicalResult sealSingleStreamPlan(DirectEmissionResult& result);
} // namespace
LogicalResult emitSingleStreamLoop(const SingleStreamLoop& plan, const SyncInput& input,
                                  const TraceDemandAnalysis& trace,
                                  IRMapping& mapping, DirectEmissionResult& result)
{
    if (plan.regionOnly || !result.selected || result.selected->boundaryLoop.get() != &plan ||
        trace.sites().size() != plan.exit + 1) {
        result.reason = "single-stream result provenance missing"; return failure();
    }
    if (plan.repeatedPair() || plan.boundaryAlternatives()) {
        auto emitted = plan.repeatedPair() ? emitRepeatedPairLoop(plan, trace, mapping, result) :
                                           emitSequentialBoundaries(plan, trace, mapping, result);
        if (failed(emitted)) { return failure(); }
        if (failed(input.target().preflightMechanisms(result.pending.get(), mapping, result.reason))) {
            return failure();
        }
        result.pending->walk([&](func::ReturnOp terminal) {
            OpBuilder drain(terminal);
            drain.create<BarrierOp>(terminal.getLoc(), PipeAttr::get(terminal.getContext(), PIPE::PIPE_ALL));
        });
        return sealSingleStreamPlan(result);
    }
    auto loop = cast<scf::ForOp>(mapping.lookup(plan.loop));
    SmallVector<scf::ForOp> frames;
    if (plan.frames.empty()) { frames.push_back(loop); }
    else {
        for (auto frame : plan.frames) { frames.push_back(cast<scf::ForOp>(mapping.lookup(frame))); }
    }
    auto outer = frames.front();
    auto guard = cast<scf::IfOp>(mapping.lookup(plan.exitGuard));
    auto* entry = mapping.lookup(trace.sites()[plan.entry].anchor);
    auto* body = mapping.lookup(trace.sites()[plan.first()].anchor);
    auto* exit = mapping.lookup(trace.sites()[plan.exit].anchor);
    auto* lastBody = mapping.lookup(trace.sites()[plan.last()].anchor);
    const auto count = trace.sites().size();
    auto entryKey = plan.entry * count + plan.first();
    auto exitKey = plan.last() * count + plan.exit;
    auto arithmetic = loop.getInductionVar().getType();
    auto location = loop.getLoc();
    // An entry publication participates iff the loop executes. It is paired
    // with precisely iteration zero's acquisition, never a cleanup WAIT.
    auto* publicationCut = plan.publicationCut ? mapping.lookupOrNull(plan.publicationCut) : nullptr;
    if (plan.participation && (!publicationCut || publicationCut != outer.getOperation() ||
        outer->getBlock()->getParentOp() != mapping.lookupOrNull(plan.participation))) {
        result.reason = "guarded publication source cut not preserved"; return failure();
    }
    OpBuilder before(publicationCut ? publicationCut : entry);
    if (!publicationCut) { before.setInsertionPointAfter(entry); }
    auto lower = before.create<arith::ConstantOp>(location, arithmetic, before.getZeroAttr(arithmetic));
    Value nonempty;
    for (auto [index, frame] : llvm::enumerate(frames)) {
        Value boundAtEntry = frame.getUpperBound();
        auto original = plan.frames.empty() ? plan.loop : plan.frames[index];
        Attribute constantBound;
        if (matchPattern(original.getUpperBound(), m_Constant(&constantBound))) {
            boundAtEntry = before.create<arith::ConstantOp>(location, arithmetic, cast<TypedAttr>(constantBound));
        }
        auto positive = before.create<arith::CmpIOp>(location, arith::CmpIPredicate::slt, lower, boundAtEntry);
        nonempty = nonempty ? Value(before.create<arith::AndIOp>(location, nonempty, positive)) : Value(positive);
    }
    auto entryGuard = before.create<scf::IfOp>(location, nonempty, false);
    auto entryBuilder = entryGuard.getThenBodyBuilder();
    logical(entryBuilder, location, true, PIPE::PIPE_MTE2, PIPE::PIPE_V, entryKey);
    OpBuilder incoming(body);
    auto zero = incoming.create<arith::ConstantOp>(location, arithmetic, incoming.getZeroAttr(arithmetic));
    Value first, local;
    for (auto frame : frames) {
        auto atFirst = incoming.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                                     frame.getInductionVar(), zero);
        auto later = incoming.create<arith::CmpIOp>(location, arith::CmpIPredicate::sgt,
                                                   frame.getInductionVar(), zero);
        first = first ? Value(incoming.create<arith::AndIOp>(location, first, atFirst)) : Value(atFirst);
        local = local ? Value(incoming.create<arith::OrIOp>(location, local, later)) : Value(later);
    }
    // Local effects precede incoming waits at the same cut.
    auto localGuard = incoming.create<scf::IfOp>(location, local, false);
    auto localBuilder = localGuard.getThenBodyBuilder();
    localBuilder.create<BarrierOp>(location, PipeAttr::get(loop.getContext(), PIPE::PIPE_V));
    auto firstGuard = incoming.create<scf::IfOp>(location, first, false);
    auto firstBuilder = firstGuard.getThenBodyBuilder();
    logical(firstBuilder, location, false, PIPE::PIPE_MTE2, PIPE::PIPE_V, entryKey);
    for (auto site : ArrayRef<std::size_t>(plan.bodies).drop_front()) {
        OpBuilder localCut(mapping.lookup(trace.sites()[site].anchor));
        localCut.create<BarrierOp>(location, PipeAttr::get(loop.getContext(), PIPE::PIPE_V));
    }
    OpBuilder outgoing(lastBody);
    outgoing.setInsertionPointAfter(lastBody);
    // On every executed iteration 0<=i<n, so i+1<=n. Original signed/index
    // arithmetic cannot overflow, including n at the representable maximum.
    auto one = outgoing.create<arith::ConstantOp>(location, arithmetic, outgoing.getIntegerAttr(arithmetic, 1));
    Value last;
    for (auto frame : frames) {
        // Each executed coordinate satisfies i+1<=its own invariant bound.
        // No product, negation or arithmetic on an absent frame is emitted.
        auto successor = outgoing.create<arith::AddIOp>(location, frame.getInductionVar(), one);
        auto atLast = outgoing.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                                    successor, frame.getUpperBound());
        last = last ? Value(outgoing.create<arith::AndIOp>(location, last, atLast)) : Value(atLast);
    }
    auto lastGuard = outgoing.create<scf::IfOp>(location, last, false);
    auto lastBuilder = lastGuard.getThenBodyBuilder();
    logical(lastBuilder, location, true, PIPE::PIPE_V, PIPE::PIPE_MTE3, exitKey);
    OpBuilder acquisition(exit);
    logical(acquisition, guard.getLoc(), false, PIPE::PIPE_V, PIPE::PIPE_MTE3, exitKey);
    result.sets = 2; result.waits = 2; result.barriers = plan.bodies.size();
    result.privateSelectors = false;
    // As for the other emitters, the invocation drain is not an internal
    // mechanism. Qualify all internal cuts before adding and sealing it.
    if (failed(input.target().preflightMechanisms(result.pending.get(), mapping, result.reason))) {
        return failure();
    }
    result.pending->walk([&](func::ReturnOp terminal) {
        OpBuilder drain(terminal);
        drain.create<BarrierOp>(terminal.getLoc(), PipeAttr::get(loop.getContext(), PIPE::PIPE_ALL));
    });
    return sealSingleStreamPlan(result);
}
LogicalResult emitRepeatedPairLoop(const SingleStreamLoop& plan, const TraceDemandAnalysis& trace,
                                   IRMapping& mapping, DirectEmissionResult& result)
{
    auto loop = cast<scf::ForOp>(mapping.lookup(plan.loop));
    auto* producer = mapping.lookup(trace.sites()[plan.first()].anchor);
    auto* consumer = mapping.lookup(trace.sites()[plan.firstCompute()].anchor);
    auto* lastConsumer = mapping.lookup(trace.sites()[plan.last()].anchor);
    auto* exit = mapping.lookup(trace.sites()[plan.exit].anchor);
    auto* original = mapping.lookup(plan.originalBarrier);
    if (original != producer->getPrevNode()) {
        result.reason = "original native barrier cut changed"; return failure();
    }
    const auto size = trace.sites().size();
    const auto ready = plan.first() * size + plan.firstCompute();
    const auto release = plan.last() * size + plan.first();
    const auto final = plan.last() * size + plan.exit;
    SmallVector<scf::ForOp> frames;
    if (plan.frames.empty()) { frames.push_back(loop); }
    else {
        for (auto frame : plan.frames) { frames.push_back(cast<scf::ForOp>(mapping.lookup(frame))); }
    }
    SmallVector<Value> coordinates;
    for (auto frame : frames) { coordinates.push_back(frame.getInductionVar()); }
    auto location = loop.getLoc();
    OpBuilder before(producer);
    auto zero = before.create<arith::ConstantIndexOp>(location, 0);
    auto one = before.create<arith::ConstantIndexOp>(location, 1);
    Value later;
    for (auto coordinate : coordinates) {
        auto positive = before.create<arith::CmpIOp>(location, arith::CmpIPredicate::sgt, coordinate, zero);
        later = later ? before.create<arith::OrIOp>(location, later, positive).getResult() : positive.getResult();
    }
    auto incoming = before.create<scf::IfOp>(location, later, false);
    auto acquire = incoming.getThenBodyBuilder();
    SmallVector<Value> previous(coordinates.size());
    Value deeperZero = acquire.create<arith::ConstantIntOp>(location, 1, 1);
    // Lexicographic predecessor in the ORIGINAL rectangular frame tuple.
    // All bounds are positive here because this producer occurrence executes.
    for (std::size_t axis = frames.size(); axis-- > 0;) {
        auto coordinate = coordinates[axis];
        auto positive = acquire.create<arith::CmpIOp>(location, arith::CmpIPredicate::sgt, coordinate, zero);
        auto decrement = acquire.create<arith::SubIOp>(location, coordinate, one);
        auto reset = acquire.create<arith::SubIOp>(location, frames[axis].getUpperBound(), one);
        auto borrowed = acquire.create<arith::SelectOp>(location, positive, decrement, reset);
        previous[axis] = frames.size() == 1 ? decrement.getResult() :
            acquire.create<arith::SelectOp>(location, deeperZero, borrowed, coordinate).getResult();
        auto isZero = acquire.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, coordinate, zero);
        deeperZero = acquire.create<arith::AndIOp>(location, deeperZero, isZero);
    }
    logical(acquire, location, false, PIPE::PIPE_V, PIPE::PIPE_MTE2, release, previous);
    OpBuilder publication(producer); publication.setInsertionPointAfter(producer);
    logical(publication, location, true, PIPE::PIPE_MTE2, PIPE::PIPE_V, ready, coordinates);
    OpBuilder acquisition(consumer);
    logical(acquisition, location, false, PIPE::PIPE_MTE2, PIPE::PIPE_V, ready, coordinates);
    for (auto site : ArrayRef<std::size_t>(plan.bodies).drop_front(2)) {
        OpBuilder local(mapping.lookup(trace.sites()[site].anchor));
        local.create<BarrierOp>(location, PipeAttr::get(loop.getContext(), PIPE::PIPE_V));
    }
    OpBuilder after(lastConsumer); after.setInsertionPointAfter(lastConsumer);
    Value more;
    for (std::size_t axis = 0; axis < frames.size(); ++axis) {
        // Executed coordinates satisfy 0<=i<bound, hence i+1 cannot overflow.
        auto next = after.create<arith::AddIOp>(location, coordinates[axis], one);
        auto remaining = after.create<arith::CmpIOp>(location, arith::CmpIPredicate::slt,
                                                   next, frames[axis].getUpperBound());
        more = more ? after.create<arith::OrIOp>(location, more, remaining).getResult() : remaining.getResult();
    }
    auto outgoing = after.create<scf::IfOp>(location, more, true);
    auto publish = outgoing.getThenBodyBuilder();
    logical(publish, location, true, PIPE::PIPE_V, PIPE::PIPE_MTE2, release, coordinates);
    auto last = outgoing.getElseBodyBuilder();
    logical(last, location, true, PIPE::PIPE_V, PIPE::PIPE_MTE3, final);
    OpBuilder exitAcquire(exit);
    logical(exitAcquire, location, false, PIPE::PIPE_V, PIPE::PIPE_MTE3, final);
    result.sets = 3; result.waits = 3; result.barriers = plan.bodies.size() - 2; result.privateSelectors = false;
    return success();
}
namespace {
LogicalResult sealSingleStreamPlan(DirectEmissionResult& result)
{
    if (!result.pending || !result.selected ||
        result.selected->kind != SelectedAnalysis::Kind::BoundaryLoop) { return failure(); }
    auto sealed = std::make_shared<SingleStreamPending>();
    unsigned barriers = 0;
    bool unsupported = false;
    result.pending->walk<WalkOrder::PreOrder>([&](Operation* operation) {
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp,
                RecordEventOp, WaitEventOp, BarrierSyncOp>(operation)) {
            unsupported = true;
        }
        if (isa<BarrierOp>(operation)) { ++barriers; }
        if (isa<OpPipeInterface>(operation) && !operation->hasAttr("pto.frontier.site")) { unsupported = true; }
        if (isa<LogicalSetOp, LogicalWaitOp>(operation)) { sealed->commands.push_back(operation); }
    });
    const auto& plan = *result.selected->boundaryLoop;
    const auto barrierTemplates = plan.repeatedPair() ? plan.bodies.size() : plan.bodies.size() + 1;
    if (unsupported || barriers != barrierTemplates ||
        sealed->commands.size() != (plan.repeatedPair() ? 6 :
            plan.boundaryAlternatives() ? 8 : 4)) { return failure(); }
    result.singleStreamPending = std::move(sealed);
    return success();
}
} // namespace
LogicalResult assignSingleStreamPhysical(const SyncInput& input, const TraceDemandAnalysis& trace,
                                        DirectEmissionResult& result, CostLedger& costs)
{
    CostScope allocation(costs, CostStage::Allocation);
    if (!result.emitted || !result.pending || !result.singleStreamPending ||
        !result.selected || !result.selected->boundaryLoop) {
        result.reason = "single-generation boundary protocol premises missing"; return failure();
    }
    const auto& plan = *result.selected->boundaryLoop;
    if (plan.regionOnly || trace.sites().size() != plan.exit + 1 || plan.bodies.empty()) {
        result.reason = "single-generation body/source identity changed"; return failure();
    }
    const auto& sealed = *result.singleStreamPending;
    if (!pendingPlanUnchanged(result)) {
        result.reason = "single-generation boundary plan changed after placement qualification"; return failure();
    }
    Builder builder(result.pending->getContext());
    SmallVector<Attribute> pools;
    SmallVector<unsigned> ids;
    // The boundary-only protocol publishes/consumes once in each pool.
    // n<=0 executes neither endpoint. For n>0: entry SET precedes first WAIT;
    // last SET precedes guarded exit WAIT. No within-invocation republication
    // exists, and all publications are consumed before invocation exit/reentry.
    DenseMap<const CompoundInstanceElement*, SyncPhysicalCore> cores;
    for (const auto& phase : input.target().phases()) { cores[phase.phase] = phase.context.core; }
    SmallVector<std::pair<std::size_t, std::size_t>> endpoints;
    if (plan.repeatedPair()) {
        endpoints = {{plan.first(), plan.firstCompute()}, {plan.last(), plan.first()}, {plan.last(), plan.exit}};
    }
    else if (plan.boundaryAlternatives()) {
        endpoints = {{plan.entry, plan.children[0]->first()}, {plan.entry, plan.children[1]->first()},
                     {plan.children[0]->last(), plan.exit}, {plan.children[1]->last(), plan.exit}};
    } else { endpoints = {{plan.entry, plan.first()}, {plan.last(), plan.exit}}; }
    const StringRef allocator = plan.repeatedPair() ? "numerical-periodic-budget-and-uniform-causal-rearm" :
                                                    (plan.boundaryAlternatives() ?
        "exclusive-single-generation-boundaries" : "single-generation-boundary-protocol");
    // For the repeated pair, actual WAIT_ready(i) gates Q(i), whose release
    // SET gates WAIT_release(i+1), P(i+1), then SET_ready(i+1). The dual
    // release chain passes through READY(i+1). Thus each previous consumption
    // causally precedes the next publication in its directed pool for every n.
    // Last release is absent; all READY and exit generations are consumed.

    DenseSet<std::pair<unsigned, unsigned>> reportedPools;
    for (auto [sourceSite, targetSite] : endpoints) {
        auto direction = std::pair{static_cast<PIPE>(trace.sites()[sourceSite].phase->kPipeValue),
                                   static_cast<PIPE>(trace.sites()[targetSite].phase->kPipeValue)};
        auto core = cores.lookup(trace.sites()[sourceSite].phase);
        if (core != cores.lookup(trace.sites()[targetSite].phase) || core == SyncPhysicalCore::Unknown ||
            core == SyncPhysicalCore::Conflict) {
            result.reason = "single-generation endpoints use incompatible physical cores"; return failure();
        }
        auto pool = input.target().eventPool(core, direction.first, direction.second, result.reason);
        if (failed(pool)) { return failure(); }
        if (pool->eligibleIds.empty()) {
            result.reason = "selected single-generation realization requires 1 eligible ID; available 0";
            return failure();
        }
        unsigned physicalId = pool->eligibleIds.front();
        if (plan.repeatedPair() && targetSite != plan.exit) {
            // The saved body quotient has no exceptional boundary flags in
            // these directions. Source/native/cut/matching and original-barrier
            // equivalence were qualified by the trusted builder and sealed above.
            const auto& index = plan.bodyIndex();
            SmallVector<std::size_t> records;
            for (auto id : index.retained()) {
                const auto& edge = index.generators()[id].edge;
                if (static_cast<PIPE>(index.sites()[edge.source]->kPipeValue) == direction.first &&
                    static_cast<PIPE>(index.sites()[edge.consumer]->kPipeValue) == direction.second) {
                    records.push_back(id);
                }
            }
            auto assignment = assignPeriodicEvents(index, records, pool->eligibleIds);
            if (assignment.status != PeriodicAssignmentStatus::Certified || !assignment.required) {
                result.reason = assignment.reason; return failure();
            }
            if (*assignment.required != 1 || assignment.phases.size() != 1) {
                result.reason = "structured immediate-ID adapter requires certified single-phase budget one";
                return failure();
            }
            // Use precisely the minimum eligible prefix. A mathematical modulo
            // over all available IDs would require dynamic target dispatch.
            assignment = assignPeriodicEvents(index, records, ArrayRef<unsigned>(pool->eligibleIds).take_front(1));
            auto sourceId = assignment.sourceId(0, llvm::DynamicAPInt(0));
            auto consumerId = assignment.consumerId(0, assignment.phases.front().distance);
            if (failed(sourceId) || failed(consumerId) || *sourceId != *consumerId) {
                result.reason = "periodic publication/acquisition ordinal reconstruction failed"; return failure();
            }
            physicalId = *sourceId;
        }
        ids.push_back(physicalId);
        // Both alternatives in a directed pool use its same eligible ID. The
        // trusted builder proved their original positive/absent guards exclusive;
        // the unchanged seal binds those guards and actual logical pair keys.
        auto poolKey = std::pair{static_cast<unsigned>(direction.first), static_cast<unsigned>(direction.second)};
        if (!reportedPools.insert(poolKey).second) {
            continue;
        }
        SmallVector<int64_t> reserved(pool->reservedIds.begin(), pool->reservedIds.end());
        SmallVector<Attribute> eligible;
        for (auto id : pool->eligibleIds) { eligible.push_back(builder.getI64IntegerAttr(id)); }
        pools.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("allocator", builder.getStringAttr(allocator)),
            builder.getNamedAttr("source", builder.getI64IntegerAttr(static_cast<unsigned>(direction.first))),
            builder.getNamedAttr("target", builder.getI64IntegerAttr(static_cast<unsigned>(direction.second))),
            builder.getNamedAttr("eligible", builder.getArrayAttr(eligible)),
            builder.getNamedAttr("required", builder.getI64IntegerAttr(1)),
            builder.getNamedAttr("src_pipe", PipeAttr::get(builder.getContext(), direction.first)),
            builder.getNamedAttr("dst_pipe", PipeAttr::get(builder.getContext(), direction.second)),
            builder.getNamedAttr("id", builder.getI64IntegerAttr(ids.back())),
            builder.getNamedAttr("physical_core", builder.getI64IntegerAttr(static_cast<int64_t>(core))),
            builder.getNamedAttr("namespace", builder.getStringAttr("physical-core/directed-pipe-pair/numeric-id")),
            builder.getNamedAttr("reserved_ids", builder.getDenseI64ArrayAttr(reserved)),
            builder.getNamedAttr("namespace_source", builder.getStringAttr(pool->namespaceSource)),
            builder.getNamedAttr("required_ids", builder.getI64IntegerAttr(1))}));
    }
    for (auto* operation : sealed.commands) {
        auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
        auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
        auto key = operation->getAttrOfType<IntegerAttr>("key").getInt();
        auto match = llvm::find_if(endpoints, [&](const auto& endpoint) {
            return key == static_cast<int64_t>(endpoint.first * trace.sites().size() + endpoint.second);
        });
        if (match == endpoints.end()) { result.reason = "unqualified protocol endpoint key"; return failure(); }
        auto id = ids[std::distance(endpoints.begin(), match)];
        auto event = EventAttr::get(builder.getContext(), static_cast<EVENT>(id));
        OpBuilder insertion(operation);
        if (isa<LogicalSetOp>(operation)) {
            insertion.create<SetFlagOp>(operation->getLoc(), source, target, event);
        } else {
            insertion.create<WaitFlagOp>(operation->getLoc(), source, target, event);
        }
        operation->erase();
    }
    result.singleStreamPending.reset();
    if (failed(verify(*result.pending))) {
        result.reason = "single-generation physical pending IR verification failed"; return failure();
    }
    SmallVector<Attribute> matching;
    for (unsigned direction = 0; direction < ids.size(); ++direction) {
        matching.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source_site", builder.getI64IntegerAttr(endpoints[direction].first)),
            builder.getNamedAttr("consumer_site", builder.getI64IntegerAttr(endpoints[direction].second)),
            builder.getNamedAttr("id", builder.getI64IntegerAttr(ids[direction]))}));
    }
    result.pending->getOperation()->setAttr("pto.frontier.physical", builder.getDictionaryAttr({
        builder.getNamedAttr("allocator", builder.getStringAttr(allocator)),
        builder.getNamedAttr("pools", builder.getArrayAttr(pools)),
        builder.getNamedAttr("matching", builder.getArrayAttr(matching)),
        builder.getNamedAttr("boundary_summary", result.selected->boundaryLoop->evidence(builder.getContext())),
        builder.getNamedAttr("entry_abi", builder.getStringAttr("eligible directed flags initially clear")),
        builder.getNamedAttr("exit_abi", builder.getStringAttr("all publications consumed; invocation drain")),
        builder.getNamedAttr("external_binding", builder.getStringAttr("normative non-auto PTO-ISA ABI required")),
        builder.getNamedAttr("empirical", builder.getStringAttr("M12 pending"))}));
    result.pending->getOperation()->setAttr("pto.frontier.physical_status",
        builder.getStringAttr(plan.repeatedPair() ? "certified-repeated-ready-release" :
                                                  "certified-single-generation-boundaries"));
    return success();
}
} // namespace mlir::pto::frontiersynch
