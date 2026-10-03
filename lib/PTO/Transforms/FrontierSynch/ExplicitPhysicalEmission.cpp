// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate matching, placement, directed eligibility and causal rearm before
// replacing any logical command. Source function publication remains separate.
#include "ExplicitPhysicalEmission.h"
#include "DirectEmissionInternal.h"
#include "SingleStreamLoop.h"
#include "PTO/Transforms/FrontierSynch/FiniteEventAssignment.h"
#include "PTO/Transforms/FrontierSynch/OneWayRepair.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <chrono>
#include "llvm/ADT/DenseSet.h"
#include <limits>
#include <optional>
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
namespace {
struct Match {
    LogicalSetOp set;
    LogicalWaitOp wait;
    std::size_t source = 0, consumer = 0;
    PIPE src, dst;
    unsigned id = 0;
};
} // namespace
namespace {
LogicalResult certifyPeriodicLocals(const SyncInput& input, DirectEmissionResult& result, CostLedger& costs)
{
    CostScope allocation(costs, CostStage::Allocation);
    if (!pendingPlanUnchanged(result) || !result.emitted || !result.selected ||
        result.selected->kind != SelectedAnalysis::Kind::Periodic || !result.selected->loop) {
        result.reason = "periodic local plan lacks unchanged original/placement provenance";
        return failure();
    }
    const auto& selected = *result.selected;
    const auto phases = selected.periodic.sites();
    auto loop = selected.loop;
    if (!isa<func::FuncOp>(loop->getParentOp())) {
        result.reason = "periodic local physical interface requires an invocation-owned fixed body; "
                        "nested/re-entered execution domains are unmet";
        return failure();
    }
    if (input.instructions().size() != phases.size() || !llvm::equal(input.instructions(), phases)) {
        result.reason = "periodic local physical interface requires complete original invocation phase coverage";
        return failure();
    }
    if (!isa<IndexType>(loop.getInductionVar().getType())) {
        result.reason = "periodic local target lowering requires original Index coordinates";
        return failure();
    }
    if (phases.empty() || llvm::any_of(phases, [&](const auto* phase) {
            return phase->kPipeValue != phases.front()->kPipeValue;
        })) {
        result.reason = "unmet periodic local physical interface: one physical pipe required";
        return failure();
    }
    for (auto id : selected.periodic.retained()) {
        const auto& edge = selected.periodic.generators()[id].edge;
        if (edge.source >= phases.size() || edge.consumer >= phases.size() ||
            (edge.distance != 0 && edge.distance != 1)) {
            result.reason = "periodic local cover identity/distance is unsupported";
            return failure();
        }
    }
    bool foreign = false;
    unsigned locals = 0, terminals = 0;
    result.pending->walk([&](Operation* operation) {
        if (isa<LogicalSetOp, LogicalWaitOp, SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp,
                RecordEventOp, WaitEventOp, BarrierSyncOp>(operation)) { foreign = true; }
        if (auto barrier = dyn_cast<BarrierOp>(operation)) {
            if (barrier.getPipe().getPipe() == PIPE::PIPE_ALL) { ++terminals; }
            else if (barrier.getPipe().getPipe() == static_cast<PIPE>(phases.front()->kPipeValue)) { ++locals; }
            else { foreign = true; }
        }
    });
    if (foreign || locals != result.barriers || terminals != 1 || result.sets || result.waits ||
        failed(verify(*result.pending))) {
        result.reason = "periodic local command inventory/matching is not certified";
        return failure();
    }
    // These guards are ordinary Index comparisons, not private selectors.
    result.privateSelectors = false;
    Builder builder(result.pending->getContext());
    result.pending->getOperation()->setAttr(
        "pto.frontier.physical_status", builder.getStringAttr("certified-periodic-locals"));
    result.pending->getOperation()->setAttr("pto.frontier.physical", builder.getDictionaryAttr({
        builder.getNamedAttr("pools", builder.getArrayAttr({})),
        builder.getNamedAttr("matching", builder.getArrayAttr({})),
        builder.getNamedAttr("allocator", builder.getStringAttr("empty-directed-pools; consumer-local covers")),
        builder.getNamedAttr("order", builder.getStringAttr("equal to recorded selected closure")),
        builder.getNamedAttr("repair", builder.getStringAttr("none; zero event capacity required")),
        builder.getNamedAttr("local_templates", builder.getI64IntegerAttr(locals)),
        builder.getNamedAttr("exit_abi", builder.getStringAttr("no publications; fixed invocation drain")),
        builder.getNamedAttr("external_binding", builder.getStringAttr("recorded normative source ABI required"))}));
    return success();
}
} // namespace
namespace {
LogicalResult qualifyOneWayBaseline(const TraceDemandAnalysis& trace,
    DirectEmissionResult& result, PIPE sourcePipe)
{
    // Original source drains are fixed H0 prerequisites, not repair choices.
    // They impose no target->source return and preserve staircase prefix cost.
    for (auto id : result.selected->explicitReduction.retained()) {
        const auto& demand = result.selected->generators[id];
        auto a = result.selected->sites[demand.source], b = result.selected->sites[demand.consumer];
        auto src = static_cast<PIPE>(trace.sites()[a].phase->kPipeValue);
        auto dst = static_cast<PIPE>(trace.sites()[b].phase->kPipeValue);
        if (src == dst && (!demand.originalBarrier || src != sourcePipe)) {
            result.reason = "one-way restricted theorem excludes variable local readiness covers";
            return failure();
        }
    }
    for (auto barrier : result.pending->getBody().front().getOps<BarrierOp>()) {
        if (barrier.getPipe().getPipe() != PIPE::PIPE_ALL && barrier.getPipe().getPipe() != sourcePipe) {
            result.reason = "one-way restricted family excludes target/global/internal observer mechanisms";
            return failure();
        }
    }
    return success();
}
struct OneWayCommandEvidence {
    SmallVector<Attribute> blocks, matching;
};
FailureOr<OneWayCommandEvidence> replaceOneWayEndpoints(Builder& builder, DirectEmissionResult& result,
    ArrayRef<Match*> matches, const OneWayRepair& plan, const DenseMap<std::size_t, Operation*>& anchors,
    std::size_t count)
{
    const auto sourcePipe = matches.front()->src, targetPipe = matches.front()->dst;
    for (const auto& block : plan.blocks) {
        if (block.first >= matches.size() || block.last >= matches.size() ||
            !anchors.lookup(matches[block.last]->source) || !anchors.lookup(matches[block.first]->consumer)) {
            return failure();
        }
    }
    OneWayCommandEvidence evidence;
    // Every selected obligation was qualified before editing the pending clone.
    for (auto* match : matches) { match->set.erase(); match->wait.erase(); }
    for (const auto& block : plan.blocks) {
        const auto source = matches[block.last]->source, consumer = matches[block.first]->consumer;
        auto* sourceAnchor = anchors.lookup(source);
        auto* consumerAnchor = anchors.lookup(consumer);
        OpBuilder publication(sourceAnchor), acquisition(consumerAnchor);
        publication.setInsertionPointAfter(sourceAnchor);
        auto src = PipeAttr::get(builder.getContext(), sourcePipe);
        auto dst = PipeAttr::get(builder.getContext(), targetPipe);
        auto event = EventAttr::get(builder.getContext(), static_cast<EVENT>(block.id));
        auto set = publication.create<SetFlagOp>(sourceAnchor->getLoc(), src, dst, event);
        auto wait = acquisition.create<WaitFlagOp>(consumerAnchor->getLoc(), src, dst, event);
        auto key = builder.getI64IntegerAttr(source * count + consumer);
        set->setAttr("pto.frontier.key", key); wait->setAttr("pto.frontier.key", key);
        evidence.matching.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source_site", builder.getI64IntegerAttr(source)),
            builder.getNamedAttr("consumer_site", builder.getI64IntegerAttr(consumer)),
            builder.getNamedAttr("id", builder.getI64IntegerAttr(block.id))}));
        evidence.blocks.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("first", builder.getI64IntegerAttr(block.first)),
            builder.getNamedAttr("last", builder.getI64IntegerAttr(block.last)),
            builder.getNamedAttr("source_site", builder.getI64IntegerAttr(source)),
            builder.getNamedAttr("consumer_site", builder.getI64IntegerAttr(consumer)),
            builder.getNamedAttr("id", builder.getI64IntegerAttr(block.id))}));
    }
    return evidence;
}
DictionaryAttr oneWayCertificate(Builder& builder, const SyncEventPool& pool, const OneWayRepair& plan,
    std::size_t fixedRequired, ArrayRef<Attribute> requiredCovers, const OneWayCommandEvidence& commands)
{
    std::string repairExcess;
    llvm::raw_string_ostream excessStream(repairExcess);
    excessStream << plan.excess;
    SmallVector<Attribute> eligibleEvidence, reservedEvidence;
    for (auto id : pool.eligibleIds) { eligibleEvidence.push_back(builder.getI64IntegerAttr(id)); }
    for (auto id : pool.reservedIds) { reservedEvidence.push_back(builder.getI64IntegerAttr(id)); }
    return builder.getDictionaryAttr({
        builder.getNamedAttr("allocator", builder.getStringAttr("minimum-excess contiguous one-way partition")),
        builder.getNamedAttr("permitted_family", builder.getStringAttr("explicit finite-one-way restriction")),
        builder.getNamedAttr("excluded_repairs", builder.getStringAttr("reverse/helper/internal-global/coordinator")),
        builder.getNamedAttr("guarantee",
            builder.getStringAttr("restricted minimality and minimum total repair excess")),
        builder.getNamedAttr("order", builder.getStringAttr("covers selected closure; repair adds prerequisites")),
        builder.getNamedAttr("repair_excess", builder.getStringAttr(repairExcess)),
        builder.getNamedAttr("matching", builder.getArrayAttr(commands.matching)),
        builder.getNamedAttr("pools", builder.getArrayAttr({builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(static_cast<unsigned>(pool.source))),
            builder.getNamedAttr("target", builder.getI64IntegerAttr(static_cast<unsigned>(pool.target))),
            builder.getNamedAttr("core", builder.getI64IntegerAttr(static_cast<unsigned>(pool.core))),
            builder.getNamedAttr("allocator", builder.getStringAttr("minimum-excess one-way partition")),
            builder.getNamedAttr("eligible", builder.getArrayAttr(eligibleEvidence)),
            builder.getNamedAttr("reserved", builder.getArrayAttr(reservedEvidence)),
            builder.getNamedAttr("required", builder.getI64IntegerAttr(plan.blocks.size()))})})),
        builder.getNamedAttr("fixed_required", builder.getI64IntegerAttr(fixedRequired)),
        builder.getNamedAttr("required_covers", builder.getArrayAttr(requiredCovers)),
        builder.getNamedAttr("blocks", builder.getArrayAttr(commands.blocks)),
        builder.getNamedAttr("namespace", builder.getStringAttr("physical-core/directed-template/id")),
        builder.getNamedAttr("pool_qualification", builder.getStringAttr(pool.namespaceSource)),
        builder.getNamedAttr("exit_abi",
            builder.getStringAttr("distinct single-generation IDs consumed; invocation drain"))});
}
LogicalResult emitOneWayRepair(const SyncInput& input, const TraceDemandAnalysis& trace,
    DirectEmissionResult& result, ArrayRef<Match*> matches, ArrayRef<unsigned> eligible,
    const DenseMap<std::size_t, Operation*>& anchors,
    const DenseMap<const CompoundInstanceElement*, const CompletionSummary*>& summaries,
    std::size_t fixedRequired)
{
    if (matches.empty()) { result.reason = "one-way repair has no selected cross covers"; return failure(); }
    const auto sourcePipe = matches.front()->src, targetPipe = matches.front()->dst;
    auto core = SyncPhysicalCore::Unknown;
    for (const auto& phase : input.target().phases()) {
        if (phase.phase == trace.sites()[matches.front()->source].phase) { core = phase.context.core; }
    }
    SmallVector<OneWayCover> covers;
    llvm::DynamicAPInt sourceCount(0), targetCount(0);
    for (const auto& site : trace.sites()) {
        auto pipe = static_cast<PIPE>(site.phase->kPipeValue);
        auto* summary = summaries.lookup(site.phase);
        if (!summary || (pipe != sourcePipe && pipe != targetPipe)) {
            result.reason = "one-way repair requires exactly two complete physical streams";
            return failure();
        }
        auto rank = llvm::DynamicAPInt(static_cast<int64_t>(summary->rank));
        if (pipe == sourcePipe) {
            sourceCount = std::max(sourceCount, rank);
        } else { targetCount = std::max(targetCount, rank); }
    }
    for (auto* match : matches) {
        auto* source = summaries.lookup(trace.sites()[match->source].phase);
        auto* consumer = summaries.lookup(trace.sites()[match->consumer].phase);
        if (match->src != sourcePipe || match->dst != targetPipe || !source || !consumer ||
            consumer->pipe >= source->S.size() || source->S[consumer->pipe] != 0) {
            result.reason = "one-way repair lacks ordered no-return selected-cover premises";
            return failure();
        }
        covers.push_back({llvm::DynamicAPInt(static_cast<int64_t>(source->rank)),
                          llvm::DynamicAPInt(static_cast<int64_t>(consumer->rank))});
    }
    if (failed(qualifyOneWayBaseline(trace, result, sourcePipe))) { return failure(); }
    auto plan = repairOneWay(covers, sourceCount, targetCount, eligible);
    if (plan.status != OneWayRepairStatus::Ready) { result.reason = plan.reason; return failure(); }
    auto pool = input.target().eventPool(core, sourcePipe, targetPipe, result.reason);
    if (failed(pool)) { return failure(); }
    Builder builder(result.pending->getContext());
    SmallVector<Attribute> requiredCovers;
    for (auto* match : matches) {
        requiredCovers.push_back(builder.getArrayAttr({builder.getI64IntegerAttr(match->source),
                                                      builder.getI64IntegerAttr(match->consumer)}));
    }
    auto commands = replaceOneWayEndpoints(builder, result, matches, plan, anchors, trace.sites().size());
    if (failed(commands)) {
        result.reason = "one-way block endpoint identity is invalid";
        return failure();
    }
    if (failed(verify(*result.pending))) {
        result.reason = "one-way physical pending verification failed";
        return failure();
    }
    result.sets = plan.blocks.size(); result.waits = plan.blocks.size(); result.privateSelectors = false;
    result.pending->getOperation()->setAttr("pto.frontier.physical_status",
        builder.getStringAttr("certified-finite-one-way-repair"));
    result.pending->getOperation()->setAttr("pto.frontier.physical",
        oneWayCertificate(builder, *pool, plan, fixedRequired, requiredCovers, *commands));
    return success();
}
} // namespace
LogicalResult emitExplicitPhysical(const SyncInput& input, const TraceDemandAnalysis& trace,
                                  DirectEmissionResult& result, CostLedger& costs, bool finiteOneWayFamily)
{
    if (result.selected && result.selected->kind == SelectedAnalysis::Kind::BoundaryLoop) {
        return assignSingleStreamPhysical(input, trace, result, costs);
    }
    if (result.selected && result.selected->kind == SelectedAnalysis::Kind::Periodic) {
        return certifyPeriodicLocals(input, result, costs);
    }
    CostScope allocation(costs, CostStage::Allocation);
    if (!pendingPlanUnchanged(result)) {
        result.reason = "finite pending/source plan changed after placement qualification";
        return failure();
    }
    if (!result.emitted || !result.pending || !result.selected ||
        result.selected->kind != SelectedAnalysis::Kind::Explicit || !trace.sequence()) {
        result.reason = "unmet finite physical interface: explicit once-executed plan required"; return failure();
    }
    if (result.pending->getBody().empty()) {
        result.reason = "finite pending function has no executable body"; return failure();
    }
    auto* body = &result.pending->getBody().front();
    bool unsupported = false;
    result.pending->walk([&](Operation* op) {
        if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp,
                RecordEventOp, WaitEventOp, BarrierSyncOp>(op)) { unsupported = true; }
        if (isa<LogicalSetOp, LogicalWaitOp, BarrierOp>(op) && op->getBlock() != body) {
            unsupported = true;
        }
        if (isa<OpPipeInterface>(op) &&
            (op->getBlock() != body || !op->hasAttr("pto.frontier.site"))) { unsupported = true; }
    });
    if (unsupported) {
        result.reason = "unsupported unmodeled command or payload in finite pending plan"; return failure();
    }
    const auto count = trace.sites().size();
    const auto& sequence = *trace.sequence();
    if (sequence.size() != count || result.selected->sites.size() != count) {
        result.reason = "explicit physical phase coverage mismatch"; return failure();
    }
    DenseMap<std::size_t, Operation*> anchors;
    for (auto& op : result.pending->getBody().front()) {
        if (auto site = op.getAttrOfType<IntegerAttr>("pto.frontier.site")) {
            if (site.getInt() < 0 || static_cast<std::size_t>(site.getInt()) >= count ||
                !anchors.try_emplace(site.getInt(), &op).second) {
                result.reason = "ambiguous original physical site identity"; return failure();
            }
        }
    }
    if (anchors.size() != count) { result.reason = "original physical anchors missing"; return failure(); }
    DenseMap<int64_t, Match> keyed;
    bool malformed = false;
    result.pending->walk([&](Operation* op) {
        if (!isa<LogicalSetOp, LogicalWaitOp>(op)) { return; }
        auto key = op->getAttrOfType<IntegerAttr>("key").getInt();
        if (!count || key < 0 || static_cast<uint64_t>(key) / count >= count ||
            op->getNumOperands() || op->getBlock() != &result.pending->getBody().front()) {
            malformed = true; return;
        }
        auto& match = keyed[key];
        match.source = static_cast<std::size_t>(key) / count;
        match.consumer = static_cast<std::size_t>(key) % count;
        auto src = op->getAttrOfType<PipeAttr>("src_pipe").getPipe();
        auto dst = op->getAttrOfType<PipeAttr>("dst_pipe").getPipe();
        if (src != static_cast<PIPE>(trace.sites()[match.source].phase->kPipeValue) ||
            dst != static_cast<PIPE>(trace.sites()[match.consumer].phase->kPipeValue)) { malformed = true; return; }
        match.src = src; match.dst = dst;
        if (auto set = dyn_cast<LogicalSetOp>(op)) {
            if (match.set) { malformed = true; } else { match.set = set; }
        } else {
            if (match.wait) { malformed = true; } else { match.wait = cast<LogicalWaitOp>(op); }
        }
    });
    if (malformed) { result.reason = "invalid finite logical matching or generation"; return failure(); }
    DenseMap<Operation*, std::size_t> sourceSites;
    for (const auto& entry : anchors) { sourceSites[entry.second] = entry.first; }
    DenseMap<Operation*, Operation*> beforePayload, afterPayload;
    Operation* priorPayload = nullptr;
    for (auto& op : result.pending->getBody().front()) {
        beforePayload[&op] = priorPayload;
        if (sourceSites.count(&op)) { priorPayload = &op; }
    }
    Operation* nextPayload = nullptr;
    for (auto& op : llvm::reverse(result.pending->getBody().front())) {
        afterPayload[&op] = nextPayload;
        if (sourceSites.count(&op)) { nextPayload = &op; }
    }
    SmallVector<Match*> matches;
    for (auto& [key, match] : keyed) {
        if (!match.set || !match.wait) { result.reason = "unmatched finite logical endpoint"; return failure(); }
        auto* before = beforePayload.lookup(match.set);
        auto* after = afterPayload.lookup(match.wait);
        if (before != anchors.lookup(match.source) || after != anchors.lookup(match.consumer)) {
            result.reason = "logical endpoint moved outside its original cut"; return failure();
        }
    }
    // Original SET order gives the source order without sorting global IDs.
    for (auto set : result.pending->getBody().front().getOps<LogicalSetOp>()) {
        matches.push_back(&keyed[set->getAttrOfType<IntegerAttr>("key").getInt()]);
    }
    SmallVector<BarrierOp> barriers;
    for (auto barrier : result.pending->getBody().front().getOps<BarrierOp>()) { barriers.push_back(barrier); }
    Operation* lastInternal = nullptr;
    unsigned universal = 0;
    for (auto& op : *body) {
        if (sourceSites.count(&op) || isa<LogicalSetOp, LogicalWaitOp, BarrierOp>(op)) { lastInternal = &op; }
        if (auto barrier = dyn_cast<BarrierOp>(op)) {
            if (barrier.getPipe().getPipe() == PIPE::PIPE_ALL) { ++universal; }
        }
    }
    auto terminalDrain = dyn_cast_or_null<BarrierOp>(lastInternal);
    if (count ? (universal != 1 || !terminalDrain || terminalDrain.getPipe().getPipe() != PIPE::PIPE_ALL) :
        (universal != 0 || lastInternal)) {
        result.reason = "finite pending plan lacks its single closed terminal PIPE_ALL drain"; return failure();
    }
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (count > limit / 2 || matches.size() > (limit - 2 * count) / 2 ||
        barriers.size() > limit - 2 * count - 2 * matches.size()) {
        result.reason = "finite command graph size not representable"; return failure();
    }
    DenseSet<std::pair<std::size_t, std::size_t>> crossCovers;
    DenseMap<std::size_t, PIPE> localCovers;
    bool canonical = result.selected->contract.reduction == SelectedReduction::SelectedOrderCovers &&
                     result.selected->explicitReduction.nonadjacentLocal().empty();
    for (auto id : result.selected->explicitReduction.retained()) {
        const auto& edge = result.selected->generators[id];
        auto a = result.selected->sites[edge.source], b = result.selected->sites[edge.consumer];
        auto pipe = static_cast<PIPE>(trace.sites()[a].phase->kPipeValue);
        if (pipe == static_cast<PIPE>(trace.sites()[b].phase->kPipeValue)) { localCovers[b] = pipe; }
        else { crossCovers.insert({a, b}); }
    }
    canonical &= crossCovers.size() == matches.size() && barriers.size() == localCovers.size() + (count ? 1 : 0);
    for (auto* match : matches) { canonical &= crossCovers.count({match->source, match->consumer}) != 0; }
    unsigned cutCategory = 0, terminal = 0;
    std::size_t payloadPosition = 0;
    DenseSet<std::size_t> localConsumers;
    for (auto& op : result.pending->getBody().front()) {
        if (sourceSites.count(&op)) {
            canonical &= payloadPosition < sequence.size() && sequence[payloadPosition] == sourceSites.lookup(&op);
            ++payloadPosition;
            cutCategory = 0;
            continue;
        }
        unsigned category = 0;
        if (isa<LogicalSetOp>(op)) { category = 0; }
        else if (isa<LogicalWaitOp>(op)) { category = 2; }
        else if (auto barrier = dyn_cast<BarrierOp>(op)) {
            category = 1;
            auto* after = afterPayload.lookup(&op);
            auto pipe = barrier.getPipe().getPipe();
            if (pipe == PIPE::PIPE_ALL) { ++terminal; canonical &= !after; }
            else if (after) {
                auto consumer = sourceSites.lookup(after);
                canonical &= localCovers.count(consumer) && localCovers.lookup(consumer) == pipe &&
                             localConsumers.insert(consumer).second;
            } else { canonical = false; }
        } else { continue; }
        canonical &= category >= cutCategory;
        cutCategory = category;
    }
    canonical &= terminal == (count ? 1U : 0U) && localConsumers.size() == localCovers.size();
    auto nodes = 2 * count + 2 * matches.size() + barriers.size();
    SmallVector<std::pair<std::size_t, std::size_t>> basis;
    auto edge = [&](std::size_t source, std::size_t target) { basis.push_back({source, target}); };
    DenseMap<PipelineType, std::size_t> lastPayload;
    for (auto site : sequence) {
        auto pipe = trace.sites()[site].phase->kPipeValue;
        auto previous = lastPayload.find(pipe);
        if (previous != lastPayload.end()) { edge(2 * previous->second + 1, 2 * site + 1); }
        lastPayload[pipe] = site;
    }
    struct Command { std::size_t node; PIPE pipe; unsigned kind; };
    // Payloads retain I/C. Each SET/WAIT/barrier has one command event:
    // publication, consumption or barrier completion (draft859b3c8).
    DenseMap<Operation*, Command> commands;
    for (const auto& entry : anchors) {
        commands[entry.second] = {2 * entry.first, static_cast<PIPE>(trace.sites()[entry.first].phase->kPipeValue), 0};
    }
    std::map<std::pair<PIPE, PIPE>, SmallVector<std::size_t>> pools;
    for (auto [i, match] : llvm::enumerate(matches)) {
        auto publication = 2 * count + 2 * i, acquisition = publication + 1;
        commands[match->set] = {publication, match->src, 1};
        commands[match->wait] = {acquisition, match->dst, 2};
        edge(publication, acquisition);
        pools[{match->src, match->dst}].push_back(i);
    }
    for (auto [i, barrier] : llvm::enumerate(barriers)) {
        commands[barrier] = {2 * count + 2 * matches.size() + i, barrier.getPipe().getPipe(), 3};
    }
    SmallVector<Command> issued;
    for (auto& op : result.pending->getBody().front()) {
        auto found = commands.find(&op);
        if (found != commands.end()) { issued.push_back(found->second); }
    }
    struct Prefix {
        std::optional<std::size_t> payloadStart, gate, observer;
        SmallVector<std::size_t> pending;
    };
    std::map<PIPE, Prefix> prefixes;
    std::optional<std::size_t> globalGate;
    for (auto command : issued) {
        auto& prefix = prefixes[command.pipe];
        auto completion = command.kind ? command.node : command.node + 1;
        if (!command.kind) { edge(command.node, completion); }
        if (prefix.payloadStart) { edge(*prefix.payloadStart, command.node); }
        if (prefix.gate) { edge(*prefix.gate, command.node); }
        if (globalGate) { edge(*globalGate, command.node); }
        auto observe = [&](Prefix& observed) {
            if (observed.observer) { edge(*observed.observer, completion); }
            for (auto earlier : observed.pending) { edge(earlier, completion); }
            observed.pending.clear();
            observed.observer = completion;
        };
        if (command.kind == 3 && command.pipe == PIPE::PIPE_ALL) {
            for (auto& item : prefixes) {
                if (item.second.payloadStart) { edge(*item.second.payloadStart, completion); }
                observe(item.second);
            }
            globalGate = completion;
        } else if (command.kind == 1 || command.kind == 3) { observe(prefix); }
        else { prefix.pending.push_back(completion); }
        if (command.kind == 2 || command.kind == 3) { prefix.gate = completion; }
        if (!command.kind) { prefix.payloadStart = command.node; }
    }
    Builder evidence(result.pending->getContext());
    SmallVector<Attribute> causalEdges;
    for (auto [source, target] : basis) {
        causalEdges.push_back(evidence.getArrayAttr(
            {evidence.getI64IntegerAttr(source), evidence.getI64IntegerAttr(target)}));
    }
    SmallVector<llvm::BitVector> reach;
    if (!canonical) {
        reach.assign(nodes, llvm::BitVector(nodes));
        for (auto [source, target] : basis) { reach[source].set(target); }
        SmallVector<llvm::BitVector> expected(2 * count, llvm::BitVector(2 * count));
        DenseMap<PipelineType, std::size_t> previous;
        for (auto site : sequence) {
            expected[2 * site].set(2 * site + 1);
            auto pipe = trace.sites()[site].phase->kPipeValue;
            auto prior = previous.find(pipe);
            if (prior != previous.end()) {
                expected[2 * prior->second].set(2 * site);
                expected[2 * prior->second + 1].set(2 * site + 1);
            }
            previous[pipe] = site;
        }
        for (auto id : result.selected->explicitReduction.retained()) {
            const auto& demand = result.selected->generators[id];
            auto source = result.selected->sites[demand.source], target = result.selected->sites[demand.consumer];
            expected[2 * source + 1].set(2 * target);
        }
        for (std::size_t k = 0; k < nodes; ++k) {
            for (std::size_t i = 0; i < nodes; ++i) { if (reach[i].test(k)) { reach[i] |= reach[k]; } }
        }
        for (std::size_t k = 0; k < 2 * count; ++k) {
            for (std::size_t i = 0; i < 2 * count; ++i) { if (expected[i].test(k)) { expected[i] |= expected[k]; } }
        }
        for (std::size_t source = 0; source < 2 * count; ++source) {
            for (std::size_t target = 0; target < 2 * count; ++target) {
                if (expected[source].test(target) != reach[source].test(target)) {
                    result.reason = "expanded command plan does not preserve selected modeled payload order";
                    return failure();
                }
            }
        }
    }
    SmallVector<Attribute> poolEvidence;
    DenseMap<const CompoundInstanceElement*, SyncPhysicalCore> cores;
    for (const auto& phase : input.target().phases()) { cores[phase.phase] = phase.context.core; }
    DenseMap<const CompoundInstanceElement*, const CompletionSummary*> summaries;
    for (const auto& summary : result.selected->explicitReduction.summaries()) {
        summaries[summary.phase] = &summary;
    }
    for (const auto& [direction, indices] : pools) {
        SmallVector<FiniteHandoff> handoffs;
        auto core = cores.lookup(trace.sites()[matches[indices.front()]->source].phase);
        for (auto i : indices) {
            auto& match = *matches[i];
            if (cores.lookup(trace.sites()[match.source].phase) != core ||
                cores.lookup(trace.sites()[match.consumer].phase) != core) {
                result.reason = "directed finite handoff crosses physical cores"; return failure();
            }
            handoffs.push_back({2 * count + 2 * i, 2 * count + 2 * i + 1});
        }
        auto pool = input.target().eventPool(core, direction.first, direction.second, result.reason);
        if (failed(pool)) { return failure(); }
        const auto& eligible = pool->eligibleIds;
        SmallVector<OrderedHandoffSummary> ordered;
        if (canonical) {
            for (auto i : indices) {
                auto* source = summaries.lookup(trace.sites()[matches[i]->source].phase);
                auto* consumer = summaries.lookup(trace.sites()[matches[i]->consumer].phase);
                if (!source || !consumer || consumer->pipe >= source->S.size()) {
                    result.reason = "saved completion summary identity mismatch"; return failure();
                }
                ordered.push_back({source->rank, consumer->rank, source->S[consumer->pipe]});
            }
        }
        auto assignmentStarted = std::chrono::steady_clock::now();
        auto assigned = canonical ? assignOrderedFiniteEvents(ordered, eligible) :
                                    assignFiniteEvents(handoffs, eligible, reach);
        if (!assigned.certified) {
            auto assignmentElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - assignmentStarted).count();
            result.attempts.push_back({nullptr, "physical/fixed-handoffs",
                assigned.required > eligible.size() ? "CapacityInsufficient" : "Unmet", assigned.reason,
                assignmentElapsed});
            if (finiteOneWayFamily && canonical && pools.size() == 1 && assigned.required > eligible.size()) {
                auto started = std::chrono::steady_clock::now();
                auto repaired = emitOneWayRepair(input, trace, result, matches, eligible, anchors, summaries,
                                                assigned.required);
                auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count();
                result.attempts.push_back({nullptr, "physical/finite-one-way",
                    succeeded(repaired) ? "Ready" : "Unmet", succeeded(repaired) ? "" : result.reason, elapsed});
                if (failed(repaired)) {
                    result.attempts.push_back({nullptr, "physical/supplied-closed", "NotSupplied", "", 0});
                    result.attempts.push_back({nullptr, "physical/supplied-coordinator", "NotSupplied", "", 0});
                    result.reason += "; supplied closed decomposition/coordinator unavailable in recorded policy";
                }
                return repaired;
            }
            result.reason = assigned.reason;
            if (finiteOneWayFamily && (!canonical || pools.size() != 1)) {
                result.reason += "; one-way repair unmet: canonical single directed pool required";
            }
            result.attempts.push_back({nullptr, "physical/finite-one-way",
                finiteOneWayFamily ? "Unmet" : "NotEnabled", "", 0});
            result.attempts.push_back({nullptr, "physical/supplied-closed", "NotSupplied", "", 0});
            result.attempts.push_back({nullptr, "physical/supplied-coordinator", "NotSupplied", "", 0});
            result.reason += "; supplied closed decomposition/coordinator unavailable in recorded policy";
            return failure();
        }
        for (auto [local, index] : llvm::enumerate(indices)) { matches[index]->id = assigned.ids[local]; }
        SmallVector<Attribute> reserved;
        for (auto id : pool->reservedIds) { reserved.push_back(evidence.getI64IntegerAttr(id)); }
        SmallVector<Attribute> ids;
        for (auto id : eligible) { ids.push_back(evidence.getI64IntegerAttr(id)); }
        SmallVector<Attribute> thresholds;
        for (auto threshold : assigned.thresholds) { thresholds.push_back(evidence.getI64IntegerAttr(threshold)); }
        poolEvidence.push_back(evidence.getDictionaryAttr({
            evidence.getNamedAttr("core", evidence.getI64IntegerAttr(static_cast<unsigned>(pool->core))),
            evidence.getNamedAttr("namespace", evidence.getStringAttr("physical-core/directed-template/id")),
            evidence.getNamedAttr("allocator",
                evidence.getStringAttr(canonical ? "saved-summary-cyclic" : "causal-chain")),
            evidence.getNamedAttr("thresholds", evidence.getArrayAttr(thresholds)),
            evidence.getNamedAttr("source", evidence.getI64IntegerAttr(static_cast<unsigned>(direction.first))),
            evidence.getNamedAttr("target", evidence.getI64IntegerAttr(static_cast<unsigned>(direction.second))),
            evidence.getNamedAttr("eligible", evidence.getArrayAttr(ids)),
            evidence.getNamedAttr("reserved", evidence.getArrayAttr(reserved)),
            evidence.getNamedAttr("qualification", evidence.getStringAttr(pool->namespaceSource)),
            evidence.getNamedAttr("required", evidence.getI64IntegerAttr(assigned.required))}));
    }
    // All matching, graph, pool and reuse queries succeeded. Mutate only the
    // pending clone; original IR stays untouched even if verification fails.
    SmallVector<Attribute> matching;
    for (auto [i, match] : llvm::enumerate(matches)) {
        matching.push_back(evidence.getDictionaryAttr({
            evidence.getNamedAttr("source_site", evidence.getI64IntegerAttr(match->source)),
            evidence.getNamedAttr("consumer_site", evidence.getI64IntegerAttr(match->consumer)),
            evidence.getNamedAttr("publication_event", evidence.getI64IntegerAttr(2 * count + 2 * i)),
            evidence.getNamedAttr("acquisition_event", evidence.getI64IntegerAttr(2 * count + 2 * i + 1)),
            evidence.getNamedAttr("id", evidence.getI64IntegerAttr(match->id))}));
        auto event = EventAttr::get(result.pending->getContext(), static_cast<EVENT>(match->id));
        OpBuilder setBuilder(match->set), waitBuilder(match->wait);
        auto set = setBuilder.create<SetFlagOp>(match->set.getLoc(), match->set.getSrcPipe(),
                                               match->set.getDstPipe(), event);
        auto wait = waitBuilder.create<WaitFlagOp>(match->wait.getLoc(), match->wait.getSrcPipe(),
                                                  match->wait.getDstPipe(), event);
        auto key = match->set->getAttr("key");
        set->setAttr("pto.frontier.key", key); wait->setAttr("pto.frontier.key", key);
        match->set.erase(); match->wait.erase();
    }
    if (failed(verify(*result.pending))) {
        result.reason = "physical pending IR verification failed";
        return failure();
    }
    result.pending->getOperation()->setAttr("pto.frontier.physical", evidence.getDictionaryAttr({
        evidence.getNamedAttr("nodes", evidence.getI64IntegerAttr(nodes)),
        evidence.getNamedAttr("causal_edges", evidence.getArrayAttr(causalEdges)),
        evidence.getNamedAttr("matching", evidence.getArrayAttr(matching)),
        evidence.getNamedAttr("pools", evidence.getArrayAttr(poolEvidence)),
        evidence.getNamedAttr("order", evidence.getStringAttr("equal to selected shared-modeled closure")),
        evidence.getNamedAttr("profile", evidence.getStringAttr("classic A2/A3 static flags; R1/R2; IDs0-5")),
        evidence.getNamedAttr("entry_abi", evidence.getStringAttr("precondition: used directed flags clear")),
        evidence.getNamedAttr("exit_abi",
            evidence.getStringAttr(count ? "all generated publications consumed; PIPE_ALL drain" :
                                              "empty invocation; no outstanding command or payload")),
        evidence.getNamedAttr("external_binding", evidence.getStringAttr("compatible PTO-ISA/CANN header required")),
        evidence.getNamedAttr("empirical", evidence.getStringAttr("M12 pending; not an analytical premise"))}));
    result.pending->getOperation()->setAttr("pto.frontier.physical_status",
        StringAttr::get(result.pending->getContext(), "certified-finite"));
    return success();
}
} // namespace mlir::pto::frontiersynch
