// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_BUFFERGENERATIONANALYSIS_H
#define PTO_TRANSFORMS_INSERTSYNC_BUFFERGENERATIONANALYSIS_H

#include "PTO/Transforms/InsertSync/LifecycleProtocol.h"
#include "PTO/Transforms/InsertSync/StorageFrontierQueries.h"

namespace mlir::pto::insert_sync_frontier {

// One defining site denotes a family of dynamic generations, not all previous
// executions of that site. A whole-slot write kills the reaching definitions.
// Joins retain alternatives; loop backedges converge without a trip horizon.
struct BufferGenerationRead {
    unsigned node = 0, member = 0;
    Bits reachingWrites; // phase identities; phase-count is the live-in sentinel
    bool update = false;
};
// A parameterized transfer, not an observed global boundary snapshot. The
// phase-count bit denotes arbitrary incoming definitions and is substituted at
// every invocation. Local and incoming definitions survive branches/loops
// independently of protocol selection. Reader/reuse obligations are retained
// separately in the shared flow frontiers.
struct BufferRegionTransfer {
    unsigned entry = 0, exit = 0, parameter = 0;
    std::map<unsigned, Bits> before;
    Bits afterExit;
    Bits apply(const Bits& expression, const Bits& incoming) const {
        Bits result = expression;
        bool substitutes = result.test(parameter);
        result.reset(parameter);
        if (substitutes) result.unite(incoming);
        return result;
    }
};
inline std::optional<BufferRegionTransfer> summarizeBufferRegion(
    const Program& p, const RegionScope& scope, const Atom& atom,
    const Bits& definiteWrites, Budget& budget)
{
    const unsigned phases = p.phaseLane.size();
    Bits members(p.nodes.size());
    if (scope.members.empty()) {
        if (scope.entry > scope.exit || scope.exit >= p.nodes.size()) return std::nullopt;
        for (unsigned n = scope.entry; n <= scope.exit; ++n) members.set(n);
    } else {
        for (unsigned n : scope.members) {
            if (n >= p.nodes.size()) return std::nullopt;
            members.set(n);
        }
    }
    if (!members.test(scope.entry) || !members.test(scope.exit)) return std::nullopt;
    for (unsigned n = 0; n < p.nodes.size(); ++n) {
        if (!budget.spend(1 + p.nodes[n].next.size())) return std::nullopt;
        for (unsigned next : p.nodes[n].next) {
            if (!members.test(n) && members.test(next) && next != scope.entry) return std::nullopt;
            if (members.test(n) && !members.test(next) && n != scope.exit) return std::nullopt;
        }
    }
    BufferRegionTransfer result;
    result.entry = scope.entry; result.exit = scope.exit; result.parameter = phases;
    result.afterExit = Bits(phases + 1);
    result.before.emplace(scope.entry, Bits(phases + 1)); result.before[scope.entry].set(phases);
    std::deque<unsigned> queue{scope.entry};
    Bits queued(p.nodes.size()); queued.set(scope.entry);
    while (!queue.empty()) {
        unsigned n = queue.front(); queue.pop_front(); queued.reset(n);
        if (!budget.spend(1 + p.nodes[n].next.size())) return std::nullopt;
        auto outgoing = result.before.at(n);
        const auto& node = p.nodes[n];
        if (node.kind == Node::Kind::Issue && atom.writes.test(node.phase)) {
            if (definiteWrites.test(node.phase)) outgoing.clear();
            outgoing.set(node.phase);
        }
        if (n == scope.exit) { result.afterExit = outgoing; continue; }
        for (unsigned next : node.next) {
            auto [at, fresh] = result.before.emplace(next, Bits(phases + 1));
            bool change = fresh || !at->second.contains(outgoing);
            at->second.unite(outgoing);
            if (change && !queued.test(next)) { queue.push_back(next); queued.set(next); }
        }
    }
    if (result.afterExit.empty()) return std::nullopt;
    return result;
}

// Immutable storage facts, independent of lane recipes, token closure and keys.
// Atom::writes is a MAY effect. Only explicitly qualified definite writes kill
// reaching content; ordering frontiers are conservative for both kinds.
struct BufferGenerationFlow {
    enum class Status { Complete, InvalidInput, AnalysisLimit };
    Status status = Status::InvalidInput;
    std::vector<BufferGenerationRead> reads;
    std::vector<std::vector<unsigned>> firstAccess, nextAccess;
    std::vector<Bits> finalReaders;
    std::vector<Bits> orderedPhases;
    std::vector<Bits> precedingOnLane; // phase-count sentinel means no earlier issue
    GenerationFrontiers frontiers;
    LifecycleResult ordering;
    unsigned regionTransfersUsed = 0;
};
inline BufferGenerationFlow analyzeBufferGenerationFlow(
    const Program& program, const std::vector<Atom>& atoms,
    const std::vector<Bits>& definiteWrites, Budget& budget)
{
    BufferGenerationFlow result;
    const unsigned count = program.nodes.size(), phases = program.phaseLane.size();
    if (!valid(program) || atoms.size() > 256 || definiteWrites.size() != atoms.size())
        return result;
    for (unsigned m = 0; m < atoms.size(); ++m)
        if (atoms[m].reads.size() != phases || atoms[m].writes.size() != phases ||
            definiteWrites[m].size() != phases || !atoms[m].writes.contains(definiteWrites[m]))
            return result;
    auto limit = [&]() {
        result.status = BufferGenerationFlow::Status::AnalysisLimit;
        return result;
    };
    result.ordering = lifecycles(program, atoms, budget);
    result.frontiers = generationFrontiers(program, atoms, budget);
    if (!result.ordering.complete || result.frontiers.status != GenerationFrontiers::Status::Complete)
        return limit();
    // Ordered occurrence reachability includes loop backedges and the native
    // guarded product. Absence is usable even when no closed protocol fits.
    result.orderedPhases.assign(phases, Bits(phases));
    result.precedingOnLane.assign(phases, Bits(phases + 1));
    for (unsigned source = 0; source < phases; ++source) {
        std::deque<unsigned> queue;
        Bits seen(count);
        for (const auto& node : program.nodes)
            if (node.kind == Node::Kind::Issue && node.phase == source)
                for (unsigned next : node.next) queue.push_back(next);
        while (!queue.empty()) {
            unsigned n = queue.front(); queue.pop_front();
            if (seen.test(n)) continue;
            if (!budget.spend(1 + program.nodes[n].next.size())) return limit();
            seen.set(n);
            if (program.nodes[n].kind == Node::Kind::Issue)
                result.orderedPhases[source].set(program.nodes[n].phase);
            for (unsigned next : program.nodes[n].next) queue.push_back(next);
        }
    }
    // Immediate issue predecessors, preserving guard/loop alternatives. This
    // is a sequencing fact; consumers still need a target-specific order rule.
    for (unsigned lane = 0; lane < program.lanes; ++lane) {
        std::vector<Bits> before(count, Bits(phases + 1));
        Bits reached(count), queued(count);
        std::deque<unsigned> queue{0};
        reached.set(0); queued.set(0); before[0].set(phases);
        while (!queue.empty()) {
            unsigned n = queue.front(); queue.pop_front(); queued.reset(n);
            if (!budget.spend(1 + program.nodes[n].next.size())) return limit();
            auto nextState = before[n];
            const auto& node = program.nodes[n];
            if (node.kind == Node::Kind::Issue && node.lane == lane) {
                nextState.clear(); nextState.set(node.phase);
            }
            for (unsigned next : node.next) {
                bool change = !reached.test(next) || !before[next].contains(nextState);
                before[next].unite(nextState); reached.set(next);
                if (change && !queued.test(next)) { queue.push_back(next); queued.set(next); }
            }
        }
        for (unsigned n = 0; n < count; ++n) {
            const auto& node = program.nodes[n];
            if (reached.test(n) && node.kind == Node::Kind::Issue && node.lane == lane)
                result.precedingOnLane[node.phase].unite(before[n]);
        }
    }
    for (unsigned m = 0; m < atoms.size(); ++m) {
        const auto& atom = atoms[m];
        std::map<unsigned, BufferRegionTransfer> summaries;
        Bits summarized(count);
        for (const auto& region : program.regions) {
            if (region.kind != RegionScope::Kind::Loop && region.kind != RegionScope::Kind::Choice) continue;
            if (summarized.test(region.entry)) continue;
            auto summary = summarizeBufferRegion(program, region, atom, definiteWrites[m], budget);
            if (!budget.left) return limit();
            if (!summary) continue;
            bool overlap = false;
            for (const auto& [n, expression] : summary->before) overlap |= summarized.test(n);
            if (overlap) continue;
            for (const auto& [n, expression] : summary->before) summarized.set(n);
            summaries.emplace(region.entry, std::move(*summary));
            ++result.regionTransfersUsed;
        }
        std::vector<Bits> before(count, Bits(phases + 1));
        Bits reached(count), queued(count);
        std::deque<unsigned> queue{0};
        reached.set(0); queued.set(0); before[0].set(phases);
        while (!queue.empty()) {
            unsigned n = queue.front(); queue.pop_front(); queued.reset(n);
            if (!budget.spend(1 + program.nodes[n].next.size())) return limit();
            auto outgoing = before[n];
            const auto& node = program.nodes[n];
            unsigned last = n;
            auto summary = summaries.find(n);
            if (summary != summaries.end()) {
                const auto incoming = before[n];
                for (const auto& [inside, expression] : summary->second.before) {
                    before[inside].unite(summary->second.apply(expression, incoming));
                    reached.set(inside);
                }
                outgoing = summary->second.apply(summary->second.afterExit, incoming);
                last = summary->second.exit;
            } else if (node.kind == Node::Kind::Issue && atom.writes.test(node.phase)) {
                if (definiteWrites[m].test(node.phase)) outgoing.clear();
                outgoing.set(node.phase);
            }
            for (unsigned next : program.nodes[last].next) {
                bool change = !reached.test(next) || !before[next].contains(outgoing);
                before[next].unite(outgoing); reached.set(next);
                if (change && !queued.test(next)) { queue.push_back(next); queued.set(next); }
            }
        }
        for (unsigned n = 0; n < count; ++n) {
            const auto& node = program.nodes[n];
            if (reached.test(n) && node.kind == Node::Kind::Issue && atom.reads.test(node.phase))
                result.reads.push_back({n, m, before[n], atom.writes.test(node.phase)});
        }
        std::vector<unsigned> first(count), after(count);
        bool change = true;
        while (change) {
            change = false;
            for (unsigned n = count; n-- > 0;) {
                if (!budget.spend(1 + program.nodes[n].next.size())) return limit();
                const auto& node = program.nodes[n];
                unsigned next = 0;
                for (unsigned successor : node.next) next |= first[successor];
                if (node.kind == Node::Kind::Exit || node.next.empty()) next |= 4;
                bool read = node.kind == Node::Kind::Issue && atom.reads.test(node.phase);
                bool write = node.kind == Node::Kind::Issue && atom.writes.test(node.phase);
                unsigned here = read && write ? 8 : write ? 1 : read ? 2 : next;
                if (first[n] != here || after[n] != next) {
                    first[n] = here; after[n] = next; change = true;
                }
            }
        }
        Bits last(count);
        for (const auto& generation : result.frontiers.generations)
            if (generation.atom == m)
                for (unsigned n : generation.finalReaders) last.set(n);
        result.firstAccess.push_back(std::move(first));
        result.nextAccess.push_back(std::move(after));
        result.finalReaders.push_back(std::move(last));
    }
    result.status = BufferGenerationFlow::Status::Complete;
    return result;
}

struct BufferGenerationAnalysis {
    LifecycleCertificate certificate;
    std::vector<BufferGenerationRead> reads;
    Bits possibleFinalReaders;
    // Per-node continuation summary: overwrite=1, reader=2, exit=4, update=8.
    // These summaries stop at the next access to THIS buffer, not another slot.
    std::vector<unsigned> nextAccess;
    unsigned readerlessReturns = 0;
    unsigned unprovedNode = kInvalid;
};

inline BufferGenerationAnalysis analyzeBufferGenerations(
    const Program& program, const LifecycleSpec& spec, const Bits& boundaries, Budget& budget)
{
    BufferGenerationAnalysis result;
    auto& cert = result.certificate;
    using Status = LifecycleCertificate::Status;
    auto fail = [&](Status status, const char* reason) {
        cert.status = status;
        cert.reason = reason;
        cert.roles.clear();
        cert.nodeRoles.clear();
        cert.uniformRoles.clear();
        return result;
    };
    const unsigned count = program.nodes.size(), phases = program.phaseLane.size();
    if (!valid(program) || !spec.members || spec.members > 4 || spec.phases.size() != phases ||
        boundaries.size() != count || spec.producerLane >= program.lanes || spec.consumerLane >= program.lanes ||
        spec.producerLane == spec.consumerLane) {
        return fail(Status::InvalidInput, "invalid buffer-generation domain");
    }
    const unsigned full = (1u << spec.members) - 1;
    for (unsigned p = 0; p < phases; ++p) {
        const auto& t = spec.phases[p];
        if ((t.writes | t.reads | t.updates) & ~full || (t.updates & ~t.writes)) {
            return fail(Status::InvalidInput, "invalid generation member mask");
        }
        if ((t.writes && t.reads) || (t.reads && t.reads != full) ||
            (t.updates && (t.updates != full || t.writes != full)) ||
            (t.writes && program.phaseLane[p] != spec.producerLane) ||
            (t.reads && program.phaseLane[p] != spec.consumerLane)) {
            return fail(Status::Unsupported, "partial generation or incompatible pipeline participants");
        }
    }
    auto touch = [&](unsigned n) {
        const auto& node = program.nodes[n];
        return node.kind == Node::Kind::Issue ? spec.phases[node.phase] : LifecycleTouch{};
    };
    std::vector<Atom> atoms(spec.members, Atom{Bits(phases), Bits(phases)});
    std::vector<Bits> definite(spec.members, Bits(phases));
    for (unsigned m = 0; m < spec.members; ++m)
        for (unsigned p = 0; p < phases; ++p) {
            if ((spec.phases[p].reads | spec.phases[p].updates) & (1u << m)) atoms[m].reads.set(p);
            if (spec.phases[p].writes & (1u << m)) { atoms[m].writes.set(p); definite[m].set(p); }
        }
    auto flow = analyzeBufferGenerationFlow(program, atoms, definite, budget);
    if (flow.status != BufferGenerationFlow::Status::Complete)
        return fail(Status::AnalysisLimit, "generation storage-flow budget");
    result.reads = flow.reads;
    for (const auto& read : flow.reads)
        if (read.reachingWrites.test(phases) || read.reachingWrites.empty())
            return fail(Status::Unsupported, "read can reach an uninitialized buffer generation");
    // The bundle boundary is the next access to any member. The same storage
    // flow transfer computes it; member reaching definitions stay independent.
    Atom bundle{Bits(phases), Bits(phases)};
    for (const auto& atom : atoms) { bundle.reads.unite(atom.reads); bundle.writes.unite(atom.writes); }
    auto boundaryFlow = spec.members == 1 ? flow :
        analyzeBufferGenerationFlow(program, {bundle}, {Bits(phases)}, budget);
    if (boundaryFlow.status != BufferGenerationFlow::Status::Complete)
        return fail(Status::AnalysisLimit, "generation bundle-flow budget");
    auto first = boundaryFlow.firstAccess[0];
    result.nextAccess = boundaryFlow.nextAccess[0];
    result.possibleFinalReaders = boundaryFlow.finalReaders[0];

    // Synthesize a ready/free round trip from the dataflow boundaries. Filling
    // includes producer-side read/modify/write chains. Ready can survive loops
    // and branches; an unused publication is consumed before overwrite or exit.
    // 0 free, 1 filling, 2 published, 3 acquired by the consumer.
    using State = std::tuple<unsigned, unsigned, unsigned, bool>;
    std::set<State> seen;
    std::deque<State> work{{0, 0, 0, false}};
    std::vector<std::optional<LifecycleRole>> roles(count);
    bool exitSeen = false, consumerSeen = false;
    while (!work.empty()) {
        auto [n, stage, written, reused] = work.front();
        work.pop_front();
        if (!budget.spend())
            return fail(Status::AnalysisLimit, "generation protocol-transfer budget");
        if (!seen.emplace(n, stage, written, reused).second)
            continue;
        const auto t = touch(n);
        const bool exit = program.nodes[n].kind == Node::Kind::Exit || program.nodes[n].next.empty();
        const unsigned after = result.nextAccess[n];
        LifecycleRole role;
        // If the previous read has a conditional successor, release on the
        // first available boundary proving that no further reader can execute.
        if (stage == 3 &&
            ((t.writes && !t.updates) || exit || (boundaries.test(n) && first[n] && !(first[n] & (2u | 8u))))) {
            role.releaseBefore = true;
            stage = 0;
            written = 0;
            reused = true;
        }
        if ((t.writes && !t.updates) || exit || (boundaries.test(n) && first[n] && !(first[n] & (2u | 8u)))) {
            if (stage == 1 && written == full) {
                role.publishBefore = true;
                stage = 2;
            }
            if (stage == 2) {
                role.bypassReady = true;
                stage = 0;
                written = 0;
                reused = true;
            }
        }
        if (t.updates) {
            if (stage != 1 || written != full)
                return fail(Status::Unsupported, "update is outside its unpublished produced generation");
        } else if (t.writes) {
            if (stage == 0) {
                role.acquireFree = true;
                cert.mayReuse |= reused;
                stage = 1;
                written = 0;
            }
            if (stage != 1 || (written & t.writes))
                return fail(Status::Unsupported, "overwrite overlaps an unfinished generation bundle");
            written |= t.writes;
        } else if (t.reads) {
            if (stage == 1 && written == full) {
                role.publishBefore = true;
                stage = 2;
            }
            if (stage == 2) {
                role.acquireReady = true;
                stage = 3;
            }
            if (stage != 3 || written != full)
                return fail(Status::Unsupported, "reader has no acquired complete generation");
            consumerSeen = true;
            if (after && !(after & (2u | 8u))) {
                role.publishFree = true;
                stage = 0;
                written = 0;
                reused = true;
            }
        }
        // Publish at the last producer whenever its continuation proves this.
        // If update/read alternatives remain, defer to a qualified boundary.
        if (t.writes && stage == 1 && written == full && after && !(after & 8u)) {
            role.publishReady = true;
            stage = 2;
        }
        if (roles[n] && !(*roles[n] == role)) {
            result.unprovedNode = n;
            return fail(Status::Unsupported, "generation boundary depends on unavailable path history");
        }
        roles[n] = role;
        if (exit) {
            exitSeen = true;
            if (stage != 0)
                return fail(Status::Unsupported, "incomplete generation escapes its physical scope");
        }
        for (unsigned next : program.nodes[n].next)
            work.emplace_back(next, stage, written, reused);
    }
    if (!exitSeen || !consumerSeen)
        return fail(Status::Unsupported, "no complete consumed buffer generation");
    cert.nodeRoles.resize(count);
    cert.roles.resize(phases);
    cert.uniformRoles.assign(phases, true);
    std::vector<bool> seenPhase(phases);
    for (unsigned n = 0; n < count; ++n) {
        if (!roles[n])
            continue;
        cert.nodeRoles[n] = *roles[n];
        result.readerlessReturns += roles[n]->bypassReady;
        if (program.nodes[n].kind != Node::Kind::Issue)
            continue;
        unsigned p = program.nodes[n].phase;
        if (seenPhase[p] && !(cert.roles[p] == *roles[n]))
            cert.uniformRoles[p] = false;
        else if (!seenPhase[p])
            cert.roles[p] = *roles[n];
        seenPhase[p] = true;
    }
    cert.status = Status::Complete;
    cert.reason = "reaching generations, final readers, and readerless returns proved";
    return result;
}
} // namespace mlir::pto::insert_sync_frontier
#endif
