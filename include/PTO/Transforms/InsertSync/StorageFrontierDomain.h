// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERDOMAIN_H
#define PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERDOMAIN_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::pto::insert_sync_frontier {

// No MLIR or target dependencies. The native adapter is responsible for exact
// execution lanes, conservative footprints, event domains and instruction rules.
// No numeric iteration horizon is used by either fixed point below.
class Bits {
public:
    explicit Bits(unsigned size = 0, bool fill = false) : size_(size), words_((size + 63) / 64, fill ? ~uint64_t(0) : 0)
    {
        if (
            fill && size % 64 && !words_.empty()) {
            words_.back() &= (uint64_t(1) << (size % 64)) - 1;
        }
    }
    unsigned size() const { return size_; }
    bool test(unsigned bit) const { return bit < size_ && (words_[bit / 64] & (uint64_t(1) << (bit % 64))); }
    void set(unsigned bit)
    {
        if (
            bit < size_) {
            words_[bit / 64] |= uint64_t(1) << (bit % 64);
        }
    }
    void reset(unsigned bit)
    {
        if (
            bit < size_) {
            words_[bit / 64] &= ~(uint64_t(1) << (bit % 64));
        }
    }
    void clear() { std::fill(words_.begin(), words_.end(), 0); }
    void unite(const Bits& other)
    {
        for (
            unsigned i = 0; i < words_.size(); ++i) {
            words_[i] |= other.words_[i];
        }
    }
    void intersect(const Bits& other)
    {
        for (
            unsigned i = 0; i < words_.size(); ++i) {
            words_[i] &= other.words_[i];
        }
    }
    bool contains(const Bits& other) const
    {
        for (
            unsigned i = 0; i < words_.size(); ++i) {
            if (
                other.words_[i] & ~words_[i]) {
                return false;
            }
        }
        return true;
    }
    bool empty() const
    {
        return std::all_of(words_.begin(), words_.end(), [](uint64_t w) { return w == 0; });
    }
    bool operator==(const Bits& other) const { return size_ == other.size_ && words_ == other.words_; }
    bool operator!=(const Bits& other) const { return !(*this == other); }

private:
    unsigned size_;
    std::vector<uint64_t> words_;
};

inline constexpr unsigned kInvalid = std::numeric_limits<unsigned>::max();
struct EventKey {
    unsigned source = 0, target = 0;
};
struct Node {
    enum class Kind { Pass, Issue, Signal, Wait, Barrier, All, Exit };
    Kind kind = Kind::Pass;
    unsigned lane = 0;
    unsigned phase = kInvalid;
    unsigned key = kInvalid;
    std::vector<unsigned> next;
};
struct RegionScope {
    unsigned entry = 0, exit = 0;
    enum class Kind { Sequence, Choice, Loop, Function };
    Kind kind = Kind::Sequence;
    unsigned parent = kInvalid;
};
struct Program {
    unsigned lanes = 0;
    std::vector<unsigned> phaseLane;
    std::vector<EventKey> keys;
    std::vector<Node> nodes;
    std::vector<RegionScope> regions;
    bool allowUnrepresentedPhases = false; // only for a proved guard partition
};
struct Budget {
    uint64_t left = 8000000;
    bool spend(uint64_t amount = 1)
    {
        if (
            amount > left) {
            left = 0;
            return false;
        }
        left -= amount;
        return true;
    }
};

inline bool valid(const Program& p)
{
    if (
        !p.lanes || p.lanes > 8 || p.nodes.empty() || p.nodes.size() > 8192 || p.phaseLane.size() > 256 ||
        p.keys.size() > 64) {
        return false;
    }
    for (
        const auto& r : p.regions) {
        if (
            r.entry >= p.nodes.size() || r.exit >= p.nodes.size()) {
            return false;
        }
    }
    std::vector<unsigned> definitions(p.phaseLane.size(), 0);
    for (
        unsigned lane : p.phaseLane) {
        if (
            lane >= p.lanes) {
            return false;
        }
    }
    for (
        const auto& key : p.keys) {
        if (
            key.source >= p.lanes || key.target >= p.lanes || key.source == key.target) {
            return false;
        }
    }
    for (
        const auto& n : p.nodes) {
        if (
            n.lane >= p.lanes) {
            return false;
        }
        for (
            unsigned next : n.next) {
            if (
                next >= p.nodes.size()) {
                return false;
            }
        }
        if (
            n.kind == Node::Kind::Issue) {
            if (
                n.phase >= p.phaseLane.size() || p.phaseLane[n.phase] != n.lane) {
                return false;
            }
            ++definitions[n.phase];
        }
        if (
            (n.kind == Node::Kind::Signal || n.kind == Node::Kind::Wait) && n.key >= p.keys.size()) {
            return false;
        }
    }
    return p.allowUnrepresentedPhases ||
           std::all_of(definitions.begin(), definitions.end(), [](unsigned n) { return n >= 1; });
}

struct CompletionState {
    std::vector<Bits> known;     // complete effects, at each destination lane
    std::vector<Bits> published; // snapshot held by an event generation
    std::vector<Bits> causal;    // last wait consumption known at each lane
    std::vector<Bits> eventCause;
    std::vector<uint8_t> marking; // may states: bit 0 empty, bit 1 live

    CompletionState(unsigned lanes = 0, unsigned phases = 0, unsigned keys = 0)
        : known(lanes, Bits(phases, true)),
          published(keys, Bits(phases)),
          causal(lanes, Bits(keys, true)),
          eventCause(keys, Bits(keys)),
          marking(keys, 1)
    {}
    bool operator==(const CompletionState& s) const
    {
        return known == s.known && published == s.published && causal == s.causal && eventCause == s.eventCause &&
               marking == s.marking;
    }
    void meet(const CompletionState& s)
    {
        for (
            unsigned i = 0; i < known.size(); ++i) {
            known[i].intersect(s.known[i]);
            causal[i].intersect(s.causal[i]);
        }
        for (
            unsigned i = 0; i < published.size(); ++i) {
            published[i].intersect(s.published[i]);
            eventCause[i].intersect(s.eventCause[i]);
            marking[i] |= s.marking[i];
        }
    }
};

struct CompletionResult {
    enum class Status { Complete, InvalidInput, LimitExceeded };
    Status status = Status::InvalidInput;
    std::vector<std::optional<CompletionState>> before;
    bool eventsProved = false;
    unsigned unprovedEventNode = kInvalid;
    uint64_t transfers = 0;
};

// Bit s means ALL earlier dynamic occurrences of static phase s are completed.
// Issuing s invalidates that bit everywhere, including saved snapshots. This is
// essential: an old token must never certify a new generation of the same site.
// It is conservative for overlapping generations, not an implicit reuse fence.
inline CompletionState transfer(
    const Program& p, const Node& n, CompletionState s, bool omitBarrier, const std::vector<Bits>& lanePhases)
{
    switch (n.kind) {
        case Node::Kind::Issue:
            for (
                auto& known : s.known) {
                known.reset(n.phase);
            }
            for (
                auto& published : s.published) {
                published.reset(n.phase);
            }
            break;
        case Node::Kind::Signal: {
            unsigned source = p.keys[n.key].source;
            s.published[n.key] = s.known[source];
            s.published[n.key].unite(lanePhases[source]);
            s.eventCause[n.key] = s.causal[source];
            s.marking[n.key] = 2;
            // Signal submission is not completion acquired on the source lane.
            break;
        }
        case Node::Kind::Wait: {
            unsigned target = p.keys[n.key].target;
            s.known[target].unite(s.published[n.key]);
            // Latest consumption is a new causal generation, separate from payload.
            for (
                auto& clock : s.causal) {
                clock.reset(n.key);
            }
            for (
                auto& clock : s.eventCause) {
                clock.reset(n.key);
            }
            s.causal[target].unite(s.eventCause[n.key]);
            s.causal[target].set(n.key);
            s.published[n.key].clear();
            s.eventCause[n.key].clear();
            s.marking[n.key] = 1;
            break;
        }
        case Node::Kind::Barrier:
            if (
                !omitBarrier) {
                s.known[n.lane].unite(lanePhases[n.lane]);
            }
            break;
        case Node::Kind::All:
            // Never selectable for deletion. All lanes in this Program have one owner.
            for (
                auto& known : s.known) {
                known = Bits(p.phaseLane.size(), true);
            }
            break;
        default:
            break;
    }
    return s;
}

inline CompletionResult completion(const Program& p, const Bits& omitted, Budget& budget)
{
    CompletionResult r;
    if (
        !valid(p) || omitted.size() != p.nodes.size()) {
        return r;
    }
    const unsigned phases = p.phaseLane.size(), keys = p.keys.size();
    std::vector<Bits> lanePhases(p.lanes, Bits(phases));
    for (
        unsigned s = 0; s < phases; ++s) {
        lanePhases[p.phaseLane[s]].set(s);
    }
    r.before.resize(p.nodes.size());
    std::vector<std::optional<CompletionState>> after(p.nodes.size());
    std::vector<bool> queued(p.nodes.size(), false);
    std::deque<unsigned> work{0};
    queued[0] = true;
    r.before[0].emplace(p.lanes, phases, keys);
    while (
        !work.empty()) {
        if (
            !budget.spend(1 + (p.lanes + keys) * ((phases + keys + 63) / 64))) {
            r.before.clear();
            r.status = CompletionResult::Status::LimitExceeded;
            return r;
        }
        unsigned id = work.front();
        work.pop_front();
        queued[id] = false;
        ++r.transfers;
        CompletionState next = transfer(p, p.nodes[id], *r.before[id], omitted.test(id), lanePhases);
        if (
            after[id] && *after[id] == next) {
            continue;
        }
        after[id] = next;
        for (
            unsigned succ : p.nodes[id].next) {
            CompletionState merged = r.before[succ] ? *r.before[succ] : next;
            if (
                r.before[succ]) {
                merged.meet(next);
            }
            if (
                r.before[succ] && *r.before[succ] == merged) {
                continue;
            }
            r.before[succ] = std::move(merged);
            if (
                !queued[succ]) {
                queued[succ] = true;
                work.push_back(succ);
            }
        }
    }
    r.eventsProved = true;
    for (
        unsigned id = 0; id < p.nodes.size(); ++id) {
        if (
            !r.before[id]) {
            continue;
        }
        const auto& n = p.nodes[id];
        const auto& state = *r.before[id];
        bool validEvent = true;
        if (
            n.kind == Node::Kind::Signal) {
            validEvent = state.marking[n.key] == 1 && state.causal[p.keys[n.key].source].test(n.key);
        } else if (n.kind == Node::Kind::Wait) {
            validEvent = state.marking[n.key] == 2;
        } else if (n.kind == Node::Kind::Exit) {
            validEvent = std::all_of(state.marking.begin(), state.marking.end(), [](uint8_t m) { return m == 1; });
        }
        if (
            !validEvent) {
            r.eventsProved = false;
            r.unprovedEventNode = id;
            break;
        }
    }
    r.status = CompletionResult::Status::Complete;
    return r;
}

struct Requirement {
    enum class Kind { Availability, Reclamation, WriteOrder, Conservative };
    unsigned source = 0, target = 0;
    Kind kind = Kind::Conservative;
    unsigned storageAtom = kInvalid;
};
struct Coverage {
    bool proved = false;
    unsigned source = kInvalid, target = kInvalid;
};
inline Coverage covers(const Program& p, const CompletionResult& s, const std::vector<Requirement>& requirements)
{
    if (
        s.status != CompletionResult::Status::Complete || !s.eventsProved) {
        return {};
    }
    // One original phase can occur at several guard-partitioned CFG nodes.
    // Every reachable representation of the target must satisfy the requirement.
    for (
        const auto& req : requirements) {
        if (
            req.source >= p.phaseLane.size() || req.target >= p.phaseLane.size()) {
            return {};
        }
        bool represented = false;
        for (
            unsigned node = 0; node < p.nodes.size(); ++node) {
            if (
                p.nodes[node].kind != Node::Kind::Issue || p.nodes[node].phase != req.target) {
                continue;
            }
            represented = true;
            if (
                s.before[node] && !s.before[node]->known[p.phaseLane[req.target]].test(req.source)) {
                return {false, req.source, req.target};
            }
        }
        if (
            !represented && !p.allowUnrepresentedPhases) {
            return {};
        }
    }
    return {true, kInvalid, kInvalid};
}

// One physical atom, already partitioned by the native adapter. Read/Write masks
// describe may effects. A write retires ordering frontiers only by recording ALL
// their mandatory requirements; it never makes a semantic content-generation
// kill or a completion claim.
struct Atom {
    Bits reads, writes;
};
struct AtomState {
    Bits writers, readers, mayDefinitions;
    bool mayHaveNoPriorWrite = true;
    explicit AtomState(unsigned phases = 0) : writers(phases), readers(phases), mayDefinitions(phases) {}
    void join(const AtomState& s)
    {
        writers.unite(s.writers);
        readers.unite(s.readers);
        mayDefinitions.unite(s.mayDefinitions);
        mayHaveNoPriorWrite |= s.mayHaveNoPriorWrite;
    }
    bool operator==(const AtomState& s) const
    {
        return writers == s.writers && readers == s.readers && mayDefinitions == s.mayDefinitions &&
               mayHaveNoPriorWrite == s.mayHaveNoPriorWrite;
    }
};
struct RegionAtomBoundary {
    unsigned region = 0, atom = 0;
    AtomState incoming, outgoing;
};
struct LifecycleResult {
    bool complete = false;
    std::vector<Requirement> requirements;
    std::vector<std::vector<std::optional<AtomState>>> before;
    std::vector<Bits> firstAccesses;
    std::vector<RegionAtomBoundary> boundaries;
    unsigned atoms = 0;
};
inline LifecycleResult lifecycles(const Program& p, const std::vector<Atom>& atoms, Budget& budget)
{
    LifecycleResult r;
    if (
        !valid(p) || atoms.size() > 256) {
        return r;
    }
    const unsigned count = p.phaseLane.size();
    for (
        const auto& a : atoms) {
        if (
            a.reads.size() != count || a.writes.size() != count) {
            return r;
        }
    }
    r.atoms = atoms.size();
    for (
        unsigned atom = 0; atom < atoms.size(); ++atom) {
        const auto& a = atoms[atom];
        std::vector<std::optional<AtomState>> in(p.nodes.size()), out(p.nodes.size());
        std::vector<bool> queued(p.nodes.size(), false);
        std::deque<unsigned> work{0};
        queued[0] = true;
        in[0].emplace(count);
        while (
            !work.empty()) {
            if (
                !budget.spend(1 + (count + 63) / 64)) {
                return {};
            }
            unsigned n = work.front();
            work.pop_front();
            queued[n] = false;
            auto next = *in[n];
            const auto& node = p.nodes[n];
            if (
                node.kind == Node::Kind::Issue) {
                unsigned phase = node.phase;
                if (
                    a.writes.test(phase)) {
                    next.writers.clear();
                    next.writers.set(phase);
                    next.readers.clear();
                    next.mayDefinitions.set(phase);
                    next.mayHaveNoPriorWrite = false;
                } else if (a.reads.test(phase)) {
                    next.readers.set(phase);
                }
            }
            if (
                out[n] && *out[n] == next) {
                continue;
            }
            out[n] = next;
            for (
                unsigned succ : node.next) {
                auto merged = in[succ] ? *in[succ] : next;
                if (
                    in[succ]) {
                    merged.join(next);
                }
                if (
                    in[succ] && *in[succ] == merged) {
                    continue;
                }
                in[succ] = std::move(merged);
                if (
                    !queued[succ]) {
                    queued[succ] = true;
                    work.push_back(succ);
                }
            }
        }
        Bits first(count);
        for (
            unsigned n = 0; n < p.nodes.size(); ++n) {
            if (
                !in[n] || p.nodes[n].kind != Node::Kind::Issue) {
                continue;
            }
            unsigned t = p.nodes[n].phase;
            bool reads = a.reads.test(t), writes = a.writes.test(t);
            if (
                !reads && !writes) {
                continue;
            }
            if (
                in[n]->mayHaveNoPriorWrite) {
                first.set(t);
            }
            for (
                unsigned s = 0; s < count; ++s) {
                if (
                    in[n]->writers.test(s)) {
                    r.requirements.push_back(
                        {s, t, writes ? Requirement::Kind::WriteOrder : Requirement::Kind::Availability, atom});
                }
                if (
                    writes && in[n]->readers.test(s)) {
                    r.requirements.push_back({s, t, Requirement::Kind::Reclamation, atom});
                }
            }
        }
        for (
            unsigned region = 0; region < p.regions.size(); ++region) {
            const auto& scope = p.regions[region];
            if (
                in[scope.entry] && in[scope.exit]) {
                r.boundaries.push_back({region, atom, *in[scope.entry], *in[scope.exit]});
            }
        }
        r.before.push_back(std::move(in));
        r.firstAccesses.push_back(std::move(first));
    }
    r.complete = true;
    return r;
}

// Proof of a bounded affine occurrence partition [stride*i+bias, ...+extent).
// Conditions refer to the ORIGINAL loop bound, not a benchmark-specific trip
// count. Unknown bounds require one invariant runtime predicate to select the
// fast synchronization plan; the conservative plan is retained otherwise.
struct AffineSlice {
    uint64_t stride = 0, bias = 0, extent = 0;
    uint64_t safeUpper = 0;
};
inline std::optional<uint64_t> partitionLimit(
    uint64_t stride, uint64_t bias, uint64_t extent, uint64_t arithmeticMax = 2147483647)
{
    if (!stride || !extent || bias >= stride || extent > stride - bias || bias > arithmeticMax ||
        extent > arithmeticMax - bias) {
        return std::nullopt;
    }
    return (arithmeticMax - bias - extent) / stride + 1;
}
inline bool disjointOccurrencePartition(
    const AffineSlice& a, const AffineSlice& b, bool sameSite, bool exclusiveWithinIteration)
{
    if (!a.safeUpper || !b.safeUpper || a.stride != b.stride || !a.stride || a.bias >= a.stride || b.bias >= b.stride ||
        !a.extent || !b.extent || a.extent > a.stride - a.bias || b.extent > b.stride - b.bias) {
        return false;
    }
    bool disjointWithin = a.bias + a.extent <= b.bias || b.bias + b.extent <= a.bias;
    return disjointWithin || sameSite || exclusiveWithinIteration;
}

struct Refinement {
    enum class Status { Complete, InvalidInput, UnprovedBaseline, AnalysisLimit };
    Status status = Status::InvalidInput;
    Bits omitted;
    unsigned attempts = 0;
    Coverage failure;
};
inline Refinement refine(
    const Program& p, const std::vector<Requirement>& requirements, const Bits& candidates, Budget& budget)
{
    Refinement r;
    r.omitted = Bits(p.nodes.size());
    if (
        !valid(p) || candidates.size() != p.nodes.size()) {
        return r;
    }
    auto base = completion(p, r.omitted, budget);
    if (
        base.status == CompletionResult::Status::LimitExceeded) {
        r.status = Refinement::Status::AnalysisLimit;
        return r;
    }
    r.failure = covers(p, base, requirements);
    if (
        !r.failure.proved) {
        r.status = Refinement::Status::UnprovedBaseline;
        return r;
    }
    // Complete barrier deletions, never moving or merging event handoffs.
    for (
        unsigned id = 0; id < p.nodes.size(); ++id) {
        if (
            !candidates.test(id) || p.nodes[id].kind != Node::Kind::Barrier) {
            continue;
        }
        ++r.attempts;
        r.omitted.set(id);
        auto trial = completion(p, r.omitted, budget);
        if (
            trial.status == CompletionResult::Status::LimitExceeded) {
            r.omitted.clear(); // Transactional budget outcome: no partial proposal.
            r.status = Refinement::Status::AnalysisLimit;
            return r;
        }
        if (
            !covers(p, trial, requirements).proved) {
            r.omitted.reset(id);
        }
    }
    r.status = Refinement::Status::Complete;
    return r;
}

} // namespace mlir::pto::insert_sync_frontier
#endif
