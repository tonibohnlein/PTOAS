// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/HandoffPlanning.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::insert_sync_frontier;
namespace {
bool event(Operation* op) { return isa<SetFlagOp, WaitFlagOp>(op); }
bool sync(Operation* op) { return event(op) || isa<BarrierOp>(op); }
using Key = std::tuple<unsigned, unsigned, unsigned>;
Key key(Operation* op) {
    if (auto s = dyn_cast<SetFlagOp>(op))
        return {unsigned(s.getSrcPipe().getPipe()), unsigned(s.getDstPipe().getPipe()), unsigned(s.getEventId().getEvent())};
    auto w = cast<WaitFlagOp>(op);
    return {unsigned(w.getSrcPipe().getPipe()), unsigned(w.getDstPipe().getPipe()), unsigned(w.getEventId().getEvent())};
}
struct View {
    StorageFrontierSnapshot facts;
    std::vector<Operation*> phases;
    bool proved = false;
    bool internalError = false;
    std::string reason;
};
View read(func::FuncOp function, bool useMmad, Budget& budget) {
    View v;
    auto contract = resolveInsertSyncGMAlias(function, "");
    if (failed(contract)) { v.internalError = true; v.reason = "invalid GM contract"; return v; }
    bool helpers = false;
    function.walk([&](func::CallOp) { helpers = true; });
    if (helpers) { v.reason = "helper effects/reservations not imported"; return v; }
    bool dynamicEvents = false;
    function.walk([&](Operation* op) { dynamicEvents |= isa<SetFlagDynOp, WaitFlagDynOp>(op); });
    if (dynamicEvents) {
        v.reason = "dynamic event selectors/reservations not imported"; return v;
    }
    MemoryDependentAnalyzer memory;
    memory.setGMContract(function, *contract);
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
    translator.enableGenerationFlow(); translator.Build();
    // Effect coverage describes payload translation. Static local events have
    // a separate resource semantics checked by NativeGraph/completion below;
    // they are not memory operands in that pre-insertion coverage API.
    SmallVector<Operation*> synchronization;
    function.walk([&](Operation* op) { if (sync(op)) synchronization.push_back(op); });
    SmallVector<std::tuple<Operation*, Operation*, Block*>> detached;
    for (auto* op : synchronization) { detached.push_back({op, op->getNextNode(), op->getBlock()}); op->remove(); }
    auto coverage = inspectInsertSyncEffectCoverage(function, ir, false);
    for (auto [op, next, block] : llvm::reverse(detached))
        block->getOperations().insert(next ? Block::iterator(next) : block->end(), op);
    if (failed(coverage)) { v.internalError = true; v.reason = "physical effect inspection failed"; return v; }
    if (!*coverage) { v.reason = "incomplete physical effect coverage"; return v; }
    v.facts = analyzeInsertSyncStorageFrontiers(function, ir, useMmad, budget, true);
    if (v.facts.status != StorageFrontierSnapshot::Status::Complete) {
        v.internalError = v.facts.status == StorageFrontierSnapshot::Status::InternalError;
        v.reason = v.facts.reason; return v;
    }
    v.phases.resize(v.facts.program.phaseLane.size());
    for (unsigned n = 0; n < v.facts.program.nodes.size(); ++n) {
        const auto& node = v.facts.program.nodes[n];
        if (node.kind == Node::Kind::Issue) v.phases[node.phase] = v.facts.anchors[n];
    }
    // Read/read ACC resource obligations are distinct from ordinary memory
    // hazards. Conservatively retain them for every possibly overlapping
    // cross-lane pair, even though the reference's memory relation omits RAR.
    for (const auto& a : v.facts.accesses) for (const auto& b : v.facts.accesses) {
        if (!budget.spend()) { v.reason = "ACC resource requirement budget"; return v; }
        if (a.write || b.write || a.space != AddressSpace::ACC || b.space != AddressSpace::ACC ||
            v.facts.program.phaseLane[a.phase] == v.facts.program.phaseLane[b.phase]) continue;
        bool overlap = a.addresses.empty() || b.addresses.empty();
        for (uint64_t x : a.addresses) for (uint64_t y : b.addresses)
            overlap |= __uint128_t(x) < __uint128_t(y) + b.upperBoundBytes &&
                       __uint128_t(y) < __uint128_t(x) + a.upperBoundBytes;
        if (overlap) v.facts.requirements.push_back({a.phase, b.phase, Requirement::Kind::Conservative});
    }
    auto coverageProof = covers(v.facts.program, v.facts.supply, v.facts.requirements);
    v.proved = coverageProof.proved && completionAtExits(v.facts.program, v.facts.supply).proved;
    v.reason = v.proved ? "complete guarded requirements, event recurrence and exit proved" :
        "seed/trial requirement or event proof unavailable: " + std::to_string(coverageProof.source) +
        " -> " + std::to_string(coverageProof.target);
    return v;
}
// This compares guarded program-point identity, not an invented flattened
// execution. Abstract first/middle/last nodes remain universally quantified by
// the existing completion engine, with its loop backedges intact.
bool samePayload(const View& a, const View& b) {
    if (a.phases != b.phases || a.facts.program.phaseLane != b.facts.program.phaseLane ||
        a.facts.guardDomains.size() != b.facts.guardDomains.size()) return false;
    for (unsigned g = 0; g < a.facts.guardDomains.size(); ++g) {
        const auto& x = a.facts.guardDomains[g]; const auto& y = b.facts.guardDomains[g];
        if (x.expression != y.expression || x.invocationScope != y.invocationScope ||
            x.tripShapeOf != y.tripShapeOf || x.possibleValues != y.possibleValues) return false;
    }
    // Partition discovery order can change when a synchronization node is
    // removed. Compare stable payload/guard identities, never BFS indices.
    using Occurrence = std::pair<unsigned, GuardEnvironment>;
    std::map<Occurrence, unsigned> left, right;
    for (unsigned n = 0; n < a.facts.program.nodes.size(); ++n)
        if (a.facts.program.nodes[n].kind == Node::Kind::Issue)
            ++left[{a.facts.program.nodes[n].phase, a.facts.guards[n]}];
    for (unsigned n = 0; n < b.facts.program.nodes.size(); ++n)
        if (b.facts.program.nodes[n].kind == Node::Kind::Issue)
            ++right[{b.facts.program.nodes[n].phase, b.facts.guards[n]}];
    if (left.size() != right.size()) return false;
    auto x = left.begin(), y = right.begin();
    for (; x != left.end(); ++x, ++y)
        if (x->first < y->first || y->first < x->first || x->second != y->second) return false;
    return true;
}
bool accepts(const View& seed, const View& trial) {
    return trial.proved && samePayload(seed, trial) &&
        covers(trial.facts.program, trial.facts.supply, seed.facts.requirements).proved;
}
bool clearInterval(Operation* a, Operation* b) {
    if (!a || !b || a->getBlock() != b->getBlock() || !a->isBeforeInBlock(b)) return false;
    for (auto* op = a->getNextNode(); op != b; op = op->getNextNode())
        if (!op || sync(op) || op->getNumRegions()) return false;
    return true;
}
struct Rewrite {
    enum Kind { Advance, Delay, Split } kind;
    Operation *signal, *wait, *producer, *consumer;
};
// Backward first-demand deadlines and forward source-prefix maxima. These are
// proposals within one original block invocation, including a loop/branch
// body. They are NOT proof that a static source belongs to a guessed iteration.
// The complete guarded trial must establish every retained requirement.
SmallVector<Rewrite> frontiers(const View& view, const llvm::SmallPtrSetImpl<Operation*>& owned, Budget& budget) {
    SmallVector<Rewrite> result;
    SmallVector<Block*> blocks;
    for (auto* phase : view.phases) if (phase && !llvm::is_contained(blocks, phase->getBlock())) blocks.push_back(phase->getBlock());
    for (Block* block : blocks) {
        std::map<Key, Operation*> pending;
        llvm::DenseMap<Operation*, unsigned> position;
        unsigned rank = 0;
        for (Operation& op : *block) position[&op] = rank++;
        for (Operation& op : *block) {
            if (op.getNumRegions()) { pending.clear(); continue; }
            if (!event(&op)) continue;
            auto k = key(&op);
            if (isa<SetFlagOp>(op)) { pending[k] = &op; continue; }
            auto found = pending.find(k);
            if (found == pending.end()) continue;
            Operation* signal = found->second; pending.erase(found);
            if (!owned.contains(signal) || !owned.contains(&op)) continue;
            unsigned sourceLane = kInvalid, targetLane = kInvalid;
            for (unsigned n = 0; n < view.facts.anchors.size(); ++n)
                if (view.facts.anchors[n] == &op) {
                    const auto& domain = view.facts.program.keys[view.facts.program.nodes[n].key];
                    sourceLane = domain.source; targetLane = domain.target; break;
                }
            std::map<unsigned, unsigned> deadlines;
            for (const auto& need : view.facts.requirements) {
                if (!budget.spend()) return result;
                Operation* source = view.phases[need.source]; Operation* target = view.phases[need.target];
                if (!source || !target || source->getBlock() != block || target->getBlock() != block ||
                    view.facts.program.phaseLane[need.source] != sourceLane ||
                    view.facts.program.phaseLane[need.target] != targetLane ||
                    !source->isBeforeInBlock(signal) || !op.isBeforeInBlock(target)) continue;
                // Only a fact actually exported by this handoff belongs to its
                // first-demand staircase; another handoff may already supply it.
                bool serves = false;
                for (unsigned n = 0; n < view.facts.anchors.size(); ++n) {
                    if (!budget.spend()) return result;
                    if (view.facts.anchors[n] != &op || !view.facts.supply.before[n]) continue;
                    const auto& fact = *view.facts.supply.before[n];
                    unsigned token = view.facts.program.nodes[n].key;
                    serves |= !fact.known[targetLane].test(need.source) && fact.published[token].test(need.source);
                }
                if (serves) deadlines[position[target]] = std::max(deadlines[position[target]], position[source]);
            }
            if (deadlines.empty()) continue;
            std::vector<Operation*> operations;
            for (Operation& original : *block) operations.push_back(&original);
            unsigned latest = 0;
            for (auto [target, source] : deadlines) latest = std::max(latest, source);
            auto* producer = operations[latest]; auto* consumer = operations[deadlines.begin()->first];
            if (producer->getNextNode() != signal && clearInterval(producer, signal))
                result.push_back({Rewrite::Advance, signal, &op, producer, nullptr});
            // A later demand may already have another supplier. Ask the whole
            // trial whether the earliest-demand prefix alone suffices before
            // introducing a second stream for that later demand.
            auto* firstProducer = operations[deadlines.begin()->second];
            if (firstProducer != producer && clearInterval(firstProducer, signal))
                result.push_back({Rewrite::Advance, signal, &op, firstProducer, nullptr});
            if (op.getNextNode() != consumer && clearInterval(&op, consumer))
                result.push_back({Rewrite::Delay, signal, &op, nullptr, consumer});
            unsigned prefix = deadlines.begin()->second;
            auto next = std::next(deadlines.begin());
            while (next != deadlines.end() && next->second <= prefix) ++next;
            if (next != deadlines.end() && clearInterval(operations[prefix], signal) &&
                clearInterval(&op, operations[next->first]))
                result.push_back({Rewrite::Split, signal, &op, operations[prefix], operations[next->first]});
        }
    }
    return result;
}
// Whole-scope fresh-key assignment preserves the selected frontiers. It is a
// bounded allocation query; failure never inserts a drain/ack or widens a cut.
std::optional<unsigned> freshKey(func::FuncOp f, Operation* signal) {
    auto domain = key(signal); std::set<unsigned> used;
    f.walk([&](Operation* op) { if (event(op)) {
        auto other = key(op);
        if (std::get<0>(domain) == std::get<0>(other) && std::get<1>(domain) == std::get<1>(other))
            used.insert(std::get<2>(other));
    }});
    for (unsigned id = 0; id < kTotalEventIdNum; ++id) if (!used.count(id)) return id;
    return std::nullopt;
}
struct Transaction {
    SmallVector<std::tuple<Operation*, Operation*, Block*>> detached;
    SmallVector<Operation*> added;
    Operation *moved = nullptr, *next = nullptr;
    void remove(Operation* op) { detached.push_back({op, op->getNextNode(), op->getBlock()}); op->remove(); }
    void rollback() {
        if (moved) moved->moveBefore(next);
        for (auto* op : added) op->erase();
        for (auto [op, anchor, block] : llvm::reverse(detached))
            block->getOperations().insert(anchor ? Block::iterator(anchor) : block->end(), op);
    }
    void commit(llvm::SmallPtrSetImpl<Operation*>& owned) {
        for (auto [op, next, block] : detached) { owned.erase(op); op->destroy(); }
        for (auto* op : added) owned.insert(op);
    }
};
}

HandoffPlanningResult mlir::pto::planInsertSyncHandoffs(func::FuncOp function,
    ArrayRef<Operation*> ownedEvents, ArrayRef<Operation*> ownedBarriers, bool useMmad, Budget& budget) {
    HandoffPlanningResult result;
    const uint64_t start = budget.left;
    OwningOpRef<ModuleOp> stage(ModuleOp::create(function.getLoc()));
    SmallVector<ModuleOp> ancestors;
    for (Operation* op = function->getParentOp(); op; op = op->getParentOp())
        if (auto module = dyn_cast<ModuleOp>(op)) ancestors.push_back(module);
    ModuleOp parent = *stage;
    for (auto module : llvm::reverse(ancestors)) {
        auto child = ModuleOp::create(module.getLoc()); child->setAttrs(module->getAttrs());
        parent.getBody()->push_back(child); parent = child;
    }
    IRMapping mapping;
    auto working = cast<func::FuncOp>(function->clone(mapping)); parent.getBody()->push_back(working);
    llvm::SmallPtrSet<Operation*, 32> owned;
    for (Operation* op : ownedEvents) if (auto* clone = mapping.lookupOrNull(op)) owned.insert(clone);
    for (Operation* op : ownedBarriers) if (auto* clone = mapping.lookupOrNull(op)) owned.insert(clone);
    auto seed = read(working, useMmad, budget);
    if (!seed.proved) { result.internalError = seed.internalError; result.reason = seed.reason; result.work = start - budget.left; return result; }
    auto current = seed;
    std::string rejected;
    bool changed = true;
    while (changed && budget.left && result.attempts < 64) {
        changed = false;
        // Requirement-driven native construction comes first. One complete
        // replacement per iteration invalidates all plan-dependent supply.
        for (const auto& proposal : frontiers(current, owned, budget)) {
            if (!budget.left || result.attempts >= 64) break;
            Transaction transaction;
            if (proposal.kind == Rewrite::Split) {
                auto free = freshKey(working, proposal.signal);
                if (!free) continue;
                auto set = cast<SetFlagOp>(proposal.signal); auto wait = cast<WaitFlagOp>(proposal.wait);
                auto id = EventAttr::get(working.getContext(), static_cast<EVENT>(*free));
                OpBuilder builder(proposal.producer); builder.setInsertionPointAfter(proposal.producer);
                transaction.added.push_back(builder.create<SetFlagOp>(set.getLoc(), set.getSrcPipe(), set.getDstPipe(), id));
                builder.setInsertionPoint(wait);
                transaction.added.push_back(builder.create<WaitFlagOp>(wait.getLoc(), wait.getSrcPipe(), wait.getDstPipe(), id));
                transaction.moved = wait; transaction.next = wait->getNextNode(); wait->moveBefore(proposal.consumer);
            } else {
                transaction.moved = proposal.kind == Rewrite::Advance ? proposal.signal : proposal.wait;
                transaction.next = transaction.moved->getNextNode();
                if (proposal.kind == Rewrite::Advance) transaction.moved->moveAfter(proposal.producer);
                else transaction.moved->moveBefore(proposal.consumer);
            }
            ++result.attempts;
            auto trial = read(working, useMmad, budget);
            if (trial.internalError) { transaction.rollback(); result.internalError = true; result.reason = trial.reason; break; }
            if (!accepts(seed, trial)) { rejected = trial.reason; transaction.rollback(); continue; }
            if (failed(verify(working))) { transaction.rollback(); result.internalError = true; result.reason = "invalid native handoff construction"; break; }
            // Publication moves earlier / acquisition later inside clear blocks;
            // split uses a strict subset of the old ordering rectangle. No old
            // event is crossed, no payload reordered, and a fresh domain key
            // cannot alter another stream's token ownership. Full recurrence
            // was checked, including zero/skipped invocations, on emitted IR.
            transaction.commit(owned); current = std::move(trial); changed = true;
            result.advanced += proposal.kind == Rewrite::Advance;
            result.delayed += proposal.kind == Rewrite::Delay;
            result.split += proposal.kind == Rewrite::Split;
            break;
        }
        if (result.internalError) break;
        if (changed) continue;
        // Logical units are operation identities, not numeric event IDs. Close
        // over all participants of their concrete key before family deletion.
        std::map<Key, SmallVector<Operation*>> families;
        working.walk([&](Operation* op) { if (event(op)) families[key(op)].push_back(op); });
        SmallVector<SmallVector<Operation*>> units;
        for (const auto& [key, actions] : families)
            if (llvm::all_of(actions, [&](Operation* op) { return owned.contains(op); })) units.push_back(actions);
        // A key may implement an entry handoff plus several recurring ones.
        // A consecutive set/wait in one uninterrupted original block is a
        // complete logical episode, even when its key has other participants.
        // Removing both endpoints leaves the ordered remaining key word
        // unchanged. Never infer such a pair across a branch or loop boundary.
        SmallVector<Block*> blocks;
        working.walk([&](Operation* op) {
            if (event(op) && !llvm::is_contained(blocks, op->getBlock())) blocks.push_back(op->getBlock());
        });
        for (auto* block : blocks) {
            std::map<Key, Operation*> pending;
            for (Operation& op : *block) {
                if (op.getNumRegions()) { pending.clear(); continue; }
                if (!event(&op)) continue;
                auto k = key(&op);
                if (isa<SetFlagOp>(op)) { pending[k] = &op; continue; }
                auto found = pending.find(k);
                if (found == pending.end()) continue;
                if (families[k].size() > 2 && llvm::all_of(families[k], [&](Operation* a) { return owned.contains(a); }))
                    units.push_back({found->second, &op});
                pending.erase(found);
            }
        }
        working.walk([&](BarrierOp op) {
            if (owned.contains(op) && (op.getPipe().getPipe() != PIPE::PIPE_ALL || op->hasAttr("pto.auto_sync_tail_barrier")))
                units.push_back({op});
        });
        for (const auto& actions : units) {
            if (!budget.left || result.attempts >= 64) break;
            auto trialProgram = current.facts.program;
            for (unsigned n = 0; n < current.facts.anchors.size(); ++n)
                if (llvm::is_contained(actions, current.facts.anchors[n])) trialProgram.nodes[n].kind = Node::Kind::Pass;
            ++result.attempts;
            auto supply = completion(trialProgram, Bits(trialProgram.nodes.size()), budget);
            auto missing = covers(trialProgram, supply, seed.facts.requirements);
            if (!missing.proved || !completionAtExits(trialProgram, supply).proved) {
                rejected = "unproved requirement " + std::to_string(missing.source) + " -> " + std::to_string(missing.target) +
                    "; event node " + std::to_string(supply.unprovedEventNode);
                continue;
            }
            Transaction transaction;
            unsigned sets = 0, waits = 0, barriers = 0;
            for (auto* op : actions) { sets += isa<SetFlagOp>(op); waits += isa<WaitFlagOp>(op); barriers += isa<BarrierOp>(op); transaction.remove(op); }
            auto trial = read(working, useMmad, budget);
            if (trial.internalError) { transaction.rollback(); result.internalError = true; result.reason = trial.reason; break; }
            if (!accepts(seed, trial)) {
                rejected = trial.reason;
                transaction.rollback(); continue;
            }
            if (failed(verify(working))) { transaction.rollback(); result.internalError = true; result.reason = "invalid native handoff deletion"; break; }
            // Remaining commands and keys stay fixed; deleting a closed family
            // introduces no new ordering and cannot swap remaining token owners.
            transaction.commit(owned); current = std::move(trial); changed = true;
            result.setsRemoved += sets; result.waitsRemoved += waits; result.barriersRemoved += barriers;
            result.unitsRemoved += bool(sets || waits); break;
        }
    }
    if (!result.internalError && (result.advanced || result.delayed || result.split || result.unitsRemoved || result.barriersRemoved))
        function.getBody().takeBody(working.getBody());
    result.work = start - budget.left;
    if (!result.internalError && !rejected.empty()) result.reason = "retained unproved candidate: " + rejected;
    if (result.reason.empty()) result.reason = budget.left ?
        "guarded prefix construction, combined supply and complete realization checked" : "budget exhausted; retained last proved plan";
    return result;
}
