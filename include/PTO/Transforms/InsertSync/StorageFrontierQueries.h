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
