// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERQUERIES_H
#define PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERQUERIES_H

#include "PTO/Transforms/InsertSync/StorageFrontierRelations.h"
#include <set>
#include <array>

namespace mlir::pto::insert_sync_frontier {

struct LaneProjection {
    unsigned lane = 0;
    std::vector<unsigned> issues, publications, acquisitions;
};
inline std::vector<LaneProjection> projectLanes(const Program& p)
{
    std::vector<LaneProjection> lanes(p.lanes);
    for (
        unsigned l = 0; l < p.lanes; ++l) {
        lanes[l].lane = l;
    }
    for (
        unsigned n = 0; n < p.nodes.size(); ++n) {
        const auto& v = p.nodes[n];
        if (
            v.kind == Node::Kind::Issue) {
            lanes[v.lane].issues.push_back(n);
        }
        if (
            v.kind == Node::Kind::Signal) {
            lanes[p.keys[v.key].source].publications.push_back(n);
        }
        if (
            v.kind == Node::Kind::Wait) {
            lanes[p.keys[v.key].target].acquisitions.push_back(n);
        }
    }
    // Node order is discovery order, NOT a total dynamic completion order.
    return lanes;
}

struct CompletionWitness {
    enum class Status { Proved, NotProved, InvalidInput, AnalysisLimit };
    Status status = Status::InvalidInput;
    unsigned sourcePhase = kInvalid, targetNode = kInvalid, lane = kInvalid;
    // A checked invariant, not a newly inserted ordering edge or a device proof.
    enum class Claim { AllEarlierOccurrencesComplete };
    Claim claim = Claim::AllEarlierOccurrencesComplete;
    std::vector<unsigned> supportingSyncNodes;
    bool includesLoopInduction = false;
};

// First experimental demand transfer: one finite, linear execution, unique
// static phase occurrences and one publication/acquisition per logical key.
// Keep requirement IDs and their original first-demand targets while following
// transitive supply backwards. These are proposal facts, never proof witnesses.
struct LinearHandoffNeeds {
    bool supported = false;
    std::vector<unsigned> order;
    std::vector<Bits> served; // requirement IDs at each publication/acquisition
};
inline LinearHandoffNeeds backwardHandoffNeeds(
    const Program& p, const CompletionResult& supply,
    const std::vector<Requirement>& requirements, Budget& budget)
{
    LinearHandoffNeeds result;
    if (!valid(p) || !covers(p, supply, requirements).proved) return result;
    Bits visited(p.nodes.size()), phases(p.phaseLane.size());
    std::vector<unsigned> signals(p.keys.size()), waits(p.keys.size());
    for (unsigned n = 0;;) {
        if (!budget.spend() || visited.test(n)) return result;
        visited.set(n);
        result.order.push_back(n);
        const auto& node = p.nodes[n];
        if (node.kind == Node::Kind::Issue) {
            if (phases.test(node.phase)) return result;
            phases.set(node.phase);
        }
        if (node.kind == Node::Kind::Signal && ++signals[node.key] != 1) return result;
        if (node.kind == Node::Kind::Wait && ++waits[node.key] != 1) return result;
        if (node.next.empty()) break;
        if (node.next.size() != 1) return result;
        n = node.next.front();
    }
    if (result.order.size() != p.nodes.size() || signals != waits ||
        phases != Bits(p.phaseLane.size(), true)) return result;
    result.served.assign(p.nodes.size(), Bits(requirements.size()));
    std::vector<Bits> needs(p.lanes, Bits(requirements.size()));
    std::vector<Bits> messages(p.keys.size(), Bits(requirements.size()));
    for (auto position = result.order.rbegin(); position != result.order.rend(); ++position) {
        unsigned n = *position;
        if (!budget.spend(1 + requirements.size() * p.lanes)) return result;
        const auto& node = p.nodes[n];
        const auto& before = *supply.before[n];
        for (unsigned r = 0; r < requirements.size(); ++r) {
            const auto& requirement = requirements[r];
            if (node.kind == Node::Kind::Issue && node.phase == requirement.target)
                needs[node.lane].set(r);
            if (node.kind == Node::Kind::All) {
                for (auto& lane : needs) lane.reset(r);
            } else if (node.kind == Node::Kind::Barrier && p.phaseLane[requirement.source] == node.lane) {
                needs[node.lane].reset(r);
            } else if (node.kind == Node::Kind::Wait) {
                unsigned target = p.keys[node.key].target;
                if (needs[target].test(r) && !before.known[target].test(requirement.source) &&
                    before.published[node.key].test(requirement.source)) {
                    result.served[n].set(r);
                    messages[node.key].set(r);
                    needs[target].reset(r);
                }
            } else if (node.kind == Node::Kind::Signal && messages[node.key].test(r)) {
                unsigned source = p.keys[node.key].source;
                result.served[n].set(r);
                // A source-prefix publication completes its own lane. Facts
                // imported from another lane retain their earlier supplier.
                if (p.phaseLane[requirement.source] != source) needs[source].set(r);
                messages[node.key].reset(r);
            }
        }
    }
    result.supported = true;
    return result;
}

struct ExitCompletion {
    bool proved = false;
    unsigned node = kInvalid, phase = kInvalid;
};
// The queried program must exclude the tail barrier being considered. Known
// bits start true for unissued phases and are invalidated on every reissue.
// A completed wait on any lane can establish global retirement at return.
inline ExitCompletion completionAtExits(const Program& p, const CompletionResult& facts)
{
    if (facts.status != CompletionResult::Status::Complete || !facts.eventsProved ||
        facts.before.size() != p.nodes.size()) return {};
    bool exit = false;
    for (unsigned n = 0; n < p.nodes.size(); ++n) {
        if (!facts.before[n] || p.nodes[n].kind != Node::Kind::Exit) continue;
        exit = true;
        Bits retired(p.phaseLane.size());
        for (const auto& lane : facts.before[n]->known) retired.unite(lane);
        for (unsigned phase = 0; phase < p.phaseLane.size(); ++phase)
            if (!retired.test(phase)) return {false, n, phase};
    }
    return {exit, kInvalid, kInvalid};
}

// A removed barrier's full prefix must still be established before the next
// operation that observes that lane, including a publication to another lane.
// Acquisitions can supply that prefix; they do not themselves publish it.
inline bool preservesBarrierCompletion(const Program& p, unsigned barrier,
                                      const CompletionResult& facts, Budget& budget)
{
    if (barrier >= p.nodes.size() || facts.status != CompletionResult::Status::Complete ||
        !facts.eventsProved || facts.before.size() != p.nodes.size()) return false;
    unsigned lane = p.nodes[barrier].lane;
    Bits required(p.phaseLane.size());
    for (unsigned phase = 0; phase < p.phaseLane.size(); ++phase)
        if (p.phaseLane[phase] == lane) required.set(phase);
    Bits seen(p.nodes.size());
    std::deque<unsigned> work(p.nodes[barrier].next.begin(), p.nodes[barrier].next.end());
    while (!work.empty()) {
        unsigned n = work.front(); work.pop_front();
        if (seen.test(n) || !facts.before[n]) continue;
        if (!budget.spend(1 + p.nodes[n].next.size())) return false;
        seen.set(n);
        const auto& node = p.nodes[n];
        if (node.kind == Node::Kind::All || (node.kind == Node::Kind::Barrier && node.lane == lane)) continue;
        bool observes = (node.kind == Node::Kind::Issue && node.lane == lane) ||
                        (node.kind == Node::Kind::Signal && p.keys[node.key].source == lane);
        if (observes) {
            if (!facts.before[n]->known[lane].contains(required)) return false;
            continue;
        }
        if (node.kind == Node::Kind::Exit) {
            Bits retired(p.phaseLane.size());
            for (const auto& known : facts.before[n]->known) retired.unite(known);
            if (!retired.contains(required)) return false;
            continue;
        }
        for (unsigned next : node.next) work.push_back(next);
    }
    return true;
}
inline CompletionWitness completionBefore(
    const Program& p, const CompletionResult& facts, unsigned source, unsigned node, Budget& budget)
{
    CompletionWitness w;
    w.sourcePhase = source;
    w.targetNode = node;
    if (
        !valid(p) || source >= p.phaseLane.size() || node >= p.nodes.size() ||
        facts.status != CompletionResult::Status::Complete || facts.before.size() != p.nodes.size()) {
        return w;
    }
    w.lane = p.nodes[node].lane;
    if (
        p.nodes[node].kind != Node::Kind::Issue || !facts.eventsProved) {
        return w;
    }
    if (
        !facts.before[node] || facts.before[node]->known[w.lane].test(source)) {
        w.status = CompletionWitness::Status::Proved;
    } else {
        w.status = CompletionWitness::Status::NotProved;
        return w;
    }
    // Retain the complete relevant predecessor cone as an explanation. Acceptance
    // comes from the independently recomputed must fixed point, not this cone and
    // not existence of one path. This intentionally does not pretend to be minimal.
    std::vector<std::vector<unsigned>> previous(p.nodes.size());
    for (
        unsigned n = 0; n < p.nodes.size(); ++n) {
        for (
            unsigned t : p.nodes[n].next) {
            previous[t].push_back(n);
        }
    }
    std::vector<bool> seen(p.nodes.size(), false);
    std::deque<unsigned> work{node};
    seen[node] = true;
    while (
        !work.empty()) {
        if (
            !budget.spend()) {
            w.status = CompletionWitness::Status::AnalysisLimit;
            w.supportingSyncNodes.clear();
            return w;
        }
        unsigned n = work.front();
        work.pop_front();
        for (
            unsigned from : previous[n]) {
            if (
                from >= n) {
                w.includesLoopInduction = true;
            }
            if (
                seen[from] || !facts.before[from]) {
                continue;
            }
            seen[from] = true;
            auto k = p.nodes[from].kind;
            if (
                k == Node::Kind::Signal || k == Node::Kind::Wait || k == Node::Kind::Barrier || k == Node::Kind::All) {
                w.supportingSyncNodes.push_back(from);
            }
            work.push_back(from);
        }
    }
    std::sort(w.supportingSyncNodes.begin(), w.supportingSyncNodes.end());
    return w;
}

// A generation class is an ordering epoch, not a definite-byte overwrite claim.
// May-content definitions in LifecycleResult remain independent. Node identities
// retain guarded alternatives; repeated executions are symbolic occurrences.
struct GenerationFrontier {
    unsigned atom = 0, producerNode = kInvalid, producerPhase = kInvalid;
    std::vector<unsigned> readers, firstReaders, finalReaders, nextOverwrites;
    // First/final use is per consumer lane. A later read on a different pipe
    // does not make an earlier reader disappear from the reclamation frontier.
    std::vector<std::vector<unsigned>> firstByLane, finalByLane;
    std::vector<unsigned> crossedRegionExits;
    bool canHaveNoReader = false;
    bool reachesFunctionExit = false;
};
struct GenerationFrontiers {
    enum class Status { Complete, InvalidInput, AnalysisLimit };
    Status status = Status::InvalidInput;
    std::vector<GenerationFrontier> generations;
};
inline GenerationFrontiers generationFrontiers(const Program& p, const std::vector<Atom>& atoms, Budget& budget)
{
    GenerationFrontiers result;
    if (
        !valid(p) || atoms.size() > 256) {
        return result;
    }
    for (
        const auto& a : atoms) {
        if (
            a.reads.size() != p.phaseLane.size() || a.writes.size() != p.phaseLane.size()) {
            return result;
        }
    }
    for (
        unsigned ai = 0; ai < atoms.size(); ++ai) {
        const auto& atom = atoms[ai];
        std::vector<unsigned> producers{kInvalid};
        for (
            unsigned n = 0; n < p.nodes.size(); ++n) {
            if (
                p.nodes[n].kind == Node::Kind::Issue && atom.writes.test(p.nodes[n].phase)) {
                producers.push_back(n);
            }
        }
        for (
            unsigned producer : producers) {
            GenerationFrontier f;
            f.atom = ai;
            f.producerNode = producer;
            if (
                producer != kInvalid) {
                f.producerPhase = p.nodes[producer].phase;
            }
            std::vector<unsigned> starts = producer == kInvalid ? std::vector<unsigned>{0} : p.nodes[producer].next;
            // State includes whether this path has already read the generation. Never
            // join first-use information merely because another branch has a reader.
            std::vector<std::set<unsigned>> seen(p.nodes.size());
            std::deque<std::pair<unsigned, unsigned>> work;
            for (
                unsigned n : starts) {
                seen[n].insert(0);
                work.push_back({n, 0});
            }
            std::set<unsigned> readers, first, final, nextWrites, exits;
            while (
                !work.empty()) {
                if (
                    !budget.spend()) {
                    result.generations.clear();
                    result.status = GenerationFrontiers::Status::AnalysisLimit;
                    return result;
                }
                auto [n, readLanes] = work.front();
                work.pop_front();
                const auto& node = p.nodes[n];
                bool read = node.kind == Node::Kind::Issue && atom.reads.test(node.phase);
                bool write = node.kind == Node::Kind::Issue && atom.writes.test(node.phase);
                if (
                    read) {
                    readers.insert(n);
                    if (
                        !(readLanes & (1u << node.lane))) {
                        first.insert(n);
                    }
                    readLanes |= 1u << node.lane;
                }
                if (
                    write) {
                    nextWrites.insert(n);
                    if (
                        !readLanes) {
                        f.canHaveNoReader = true;
                    }
                    continue;
                }
                if (
                    node.kind == Node::Kind::Exit) {
                    f.reachesFunctionExit = true;
                    if (
                        !readLanes) {
                        f.canHaveNoReader = true;
                    }
                    continue;
                }
                for (
                    unsigned r = 0; r < p.regions.size(); ++r) {
                    if (
                        p.regions[r].exit == n) {
                        exits.insert(r);
                    }
                }
                for (
                    unsigned t : node.next) {
                    if (
                        seen[t].insert(readLanes).second) {
                        work.push_back({t, readLanes});
                    }
                }
            }
            // A possible final reader has a path to the next overwrite/function exit
            // with no intervening same-lane read. This is a frontier SET, not a chosen global
            // last operation. An infinite path alone does not prove a final reader.
            for (
                unsigned reader : readers) {
                std::vector<bool> visited(p.nodes.size(), false);
                std::deque<unsigned> scan;
                for (
                    unsigned n : p.nodes[reader].next) {
                    visited[n] = true;
                    scan.push_back(n);
                }
                bool reaches = false;
                // A read-write phase consumes the old generation and starts the next.
                if (
                    atom.writes.test(p.nodes[reader].phase)) {
                    reaches = true;
                }
                while (
                    !scan.empty() && !reaches) {
                    if (
                        !budget.spend()) {
                        result.generations.clear();
                        result.status = GenerationFrontiers::Status::AnalysisLimit;
                        return result;
                    }
                    unsigned n = scan.front();
                    scan.pop_front();
                    const auto& v = p.nodes[n];
                    if (
                        v.kind == Node::Kind::Exit) {
                        reaches = true;
                        break;
                    }
                    if (
                        v.kind == Node::Kind::Issue) {
                        if (
                            atom.reads.test(v.phase) && v.lane == p.nodes[reader].lane) {
                            continue;
                        }
                        if (
                            atom.writes.test(v.phase)) {
                            reaches = true;
                            break;
                        }
                    }
                    for (
                        unsigned t : v.next) {
                        if (
                            !visited[t]) {
                            visited[t] = true;
                            scan.push_back(t);
                        }
                    }
                }
                if (
                    reaches) {
                    final.insert(reader);
                }
            }
            f.readers.assign(readers.begin(), readers.end());
            f.firstReaders.assign(first.begin(), first.end());
            f.finalReaders.assign(final.begin(), final.end());
            f.nextOverwrites.assign(nextWrites.begin(), nextWrites.end());
            f.firstByLane.resize(p.lanes);
            f.finalByLane.resize(p.lanes);
            for (
                unsigned n : first) {
                f.firstByLane[p.nodes[n].lane].push_back(n);
            }
            for (
                unsigned n : final) {
                f.finalByLane[p.nodes[n].lane].push_back(n);
            }
            f.crossedRegionExits.assign(exits.begin(), exits.end());
            result.generations.push_back(std::move(f));
        }
    }
    result.status = GenerationFrontiers::Status::Complete;
    return result;
}

struct EventKeySharing {
    std::vector<unsigned> representative;
    unsigned merged = 0;
};
// Check token identity as well as balance: every wait must still consume a
// publication from its original logical stream after physical key sharing.
inline bool preservesEventOwners(const Program& original, const Program& assigned, Budget& budget)
{
    const unsigned keys = original.keys.size(), count = original.nodes.size();
    using Owners = std::vector<Bits>;
    std::vector<std::optional<Owners>> before(count);
    before[0] = Owners(keys, Bits(keys + 1));
    for (auto& owner : *before[0]) owner.set(keys); // empty
    std::deque<unsigned> queue{0}; Bits queued(count); queued.set(0);
    while (!queue.empty()) {
        unsigned n = queue.front(); queue.pop_front(); queued.reset(n);
        if (!budget.spend(1 + keys * (1 + assigned.nodes[n].next.size()))) return false;
        auto outgoing = *before[n];
        const auto& node = assigned.nodes[n];
        if (node.kind == Node::Kind::Signal || node.kind == Node::Kind::Wait) {
            outgoing[node.key].clear();
            outgoing[node.key].set(node.kind == Node::Kind::Signal ? original.nodes[n].key : keys);
        }
        for (unsigned next : node.next) {
            bool change = !before[next];
            if (!before[next]) before[next] = outgoing;
            else for (unsigned k = 0; k < keys; ++k) {
                change |= !(*before[next])[k].contains(outgoing[k]);
                (*before[next])[k].unite(outgoing[k]);
            }
            if (change && !queued.test(next)) { queue.push_back(next); queued.set(next); }
        }
    }
    for (unsigned n = 0; n < count; ++n)
        if (before[n] && assigned.nodes[n].kind == Node::Kind::Wait) {
            Bits expected(keys + 1); expected.set(original.nodes[n].key);
            if ((*before[n])[assigned.nodes[n].key] != expected) return false;
        }
    return true;
}
inline EventKeySharing shareEventKeys(const Program& p, const Bits& eligible, Budget& budget)
{
    EventKeySharing result;
    for (unsigned k = 0; k < p.keys.size(); ++k) result.representative.push_back(k);
    if (!valid(p) || eligible.size() != p.keys.size()) return result;
    for (unsigned key = 0; key < p.keys.size(); ++key) {
        if (!eligible.test(key)) continue;
        for (unsigned other = 0; other < key; ++other) {
            if (!eligible.test(other) || result.representative[other] != other ||
                p.keys[key].source != p.keys[other].source || p.keys[key].target != p.keys[other].target) continue;
            auto mapping = result.representative; mapping[key] = other;
            auto trial = p;
            for (auto& node : trial.nodes)
                if (node.kind == Node::Kind::Signal || node.kind == Node::Kind::Wait) node.key = mapping[node.key];
            auto proof = completion(trial, Bits(trial.nodes.size()), budget);
            if (proof.status == CompletionResult::Status::Complete && proof.eventsProved &&
                preservesEventOwners(p, trial, budget)) {
                result.representative = std::move(mapping); ++result.merged; break;
            }
            if (!budget.left) return result;
        }
    }
    return result;
}

struct BarrierGroup {
    std::vector<unsigned> nodes;
};
inline Refinement refineGroups(
    const Program& p, const std::vector<Requirement>& requirements, const std::vector<BarrierGroup>& groups,
    Budget& budget)
{
    Refinement result;
    result.omitted = Bits(p.nodes.size());
    if (
        !valid(p)) {
        return result;
    }
    for (
        const auto& g : groups) {
        for (
            unsigned n : g.nodes) {
            if (
                n >= p.nodes.size() || p.nodes[n].kind != Node::Kind::Barrier) {
                return result;
            }
        }
    }
    auto baseline = completion(p, result.omitted, budget);
    if (
        baseline.status == CompletionResult::Status::LimitExceeded) {
        result.status = Refinement::Status::AnalysisLimit;
        return result;
    }
    result.failure = covers(p, baseline, requirements);
    if (
        !result.failure.proved) {
        result.status = Refinement::Status::UnprovedBaseline;
        return result;
    }
    for (
        const auto& group : groups) {
        ++result.attempts;
        Bits previous = result.omitted;
        for (
            unsigned n : group.nodes) {
            result.omitted.set(n);
        }
        auto trial = completion(p, result.omitted, budget);
        if (
            trial.status == CompletionResult::Status::LimitExceeded) {
            result.omitted.clear();
            result.status = Refinement::Status::AnalysisLimit;
            return result;
        }
        if (
            !covers(p, trial, requirements).proved) {
            result.omitted = std::move(previous);
        }
    }
    result.status = Refinement::Status::Complete;
    return result;
}
} // namespace mlir::pto::insert_sync_frontier
#endif
