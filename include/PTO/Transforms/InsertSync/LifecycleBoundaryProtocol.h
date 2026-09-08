// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LIFECYCLEBOUNDARYPROTOCOL_H
#define PTO_TRANSFORMS_INSERTSYNC_LIFECYCLEBOUNDARYPROTOCOL_H

#include "PTO/Transforms/InsertSync/LifecycleProtocol.h"

namespace mlir::pto::insert_sync_frontier {

// The adapter proves that an empty marker is the bypass of a read-only consumer
// region. A flush is its enclosing definition-scope boundary, AFTER independent
// suffix work. It is not a whole-loop completion fact. Ordinary access nodes and
// all their control edges remain in the Program; the marker has no payload effect.
struct LifecycleBoundaryFacts {
    Bits emptyConsumers;
    Bits flushReaderless;
};

// R6's W*R+ recognition extended to node-specific first/final roles and a checked
// W*(empty-consumer-region) episode. Repetition is a finite state fixed point,
// not a bounded trip sample. An episode with no consumers is not immediately
// declared free: its Ready is consumed before a deferred Free publication.
inline LifecycleCertificate recognizeBoundaryLifecycle(
    const Program& program, const LifecycleSpec& spec, const LifecycleBoundaryFacts& boundaries, Budget& budget)
{
    LifecycleCertificate result;
    using Status = LifecycleCertificate::Status;
    auto stop = [&](Status status, const char* reason) {
        result.status = status;
        result.reason = reason;
        result.roles.clear();
        result.nodeRoles.clear();
        result.uniformRoles.clear();
        return result;
    };
    if (
        !valid(program) || spec.producerLane >= program.lanes || spec.consumerLane >= program.lanes ||
        spec.producerLane == spec.consumerLane || !spec.members || spec.members > 4 ||
        spec.phases.size() != program.phaseLane.size() || boundaries.emptyConsumers.size() != program.nodes.size() ||
        boundaries.flushReaderless.size() != program.nodes.size()) {
        return stop(Status::InvalidInput, "invalid boundary lifecycle domain");
    }
    const unsigned full = (1u << spec.members) - 1, reading = full + 1;
    for (unsigned p = 0; p < spec.phases.size(); ++p) {
        auto touch = spec.phases[p];
        if (
            (touch.writes | touch.reads) & ~full) {
            return stop(Status::InvalidInput, "invalid lifecycle member mask");
        }
        if (
            touch.updates || (touch.writes && touch.reads) || (touch.reads && touch.reads != full) ||
            (touch.writes && program.phaseLane[p] != spec.producerLane) ||
            (touch.reads && program.phaseLane[p] != spec.consumerLane)) {
            return stop(Status::Unsupported, "incomplete bundle or incompatible participant");
        }
    }
    std::vector<unsigned> first(program.nodes.size()), after(program.nodes.size());
    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned n = program.nodes.size(); n-- > 0;) {
            if (!budget.spend(1 + program.nodes[n].next.size())) {
                return stop(Status::AnalysisLimit, "boundary next-use fixed-point budget");
            }
            const auto& node = program.nodes[n];
            unsigned following = 0;
            for (unsigned next : node.next) {
                following |= first[next];
            }
            unsigned here = following;
            if (node.kind == Node::Kind::Exit || node.next.empty()) {
                here |= 4;
            }
            if (node.kind == Node::Kind::Issue) {
                auto touch = spec.phases[node.phase];
                if (touch.writes) {
                    here = 1;
                } else if (touch.reads) {
                    here = 2;
                }
            }
            if (first[n] != here || after[n] != following) {
                first[n] = here;
                after[n] = following;
                changed = true;
            }
        }
    }
    using State = std::tuple<unsigned, unsigned, bool, bool>;
    std::set<State> seen;
    std::deque<State> work{{0, 0, false, false}};
    std::vector<std::optional<LifecycleRole>> roles(program.nodes.size());
    bool consumerSeen = false, bypassSeen = false, exitSeen = false;
    while (!work.empty()) {
        auto [n, state, reused, emptySeen] = work.front();
        work.pop_front();
        if (!budget.spend()) {
            return stop(Status::AnalysisLimit, "guarded generation-transfer budget");
        }
        if (!seen.emplace(n, state, reused, emptySeen).second) {
            continue;
        }
        const auto& node = program.nodes[n];
        LifecycleRole role;
        if (
            boundaries.emptyConsumers.test(n) && state == full) {
            emptySeen = true;
        }
        const bool exit = node.kind == Node::Kind::Exit || node.next.empty();
        const auto touch = node.kind == Node::Kind::Issue ? spec.phases[node.phase] : LifecycleTouch{};
        if (
            (touch.writes || boundaries.flushReaderless.test(n) || exit) && state == full && emptySeen) {
            // No physical read is invented. This pair completes the producer and
            // returns its reuse credit at the next overwrite/scope boundary.
            role.bypassReady = true;
            state = 0;
            emptySeen = false;
            reused = true;
            bypassSeen = true;
        }
        if (touch.writes) {
            if (state > full || (state & touch.writes)) {
                return stop(Status::Unsupported, "overwrite before generation final use or qualified bypass");
            }
            role.acquireFree = state == 0;
            result.mayReuse |= role.acquireFree && reused;
            state |= touch.writes;
            role.publishReady = state == full;
            emptySeen = false;
        } else if (touch.reads) {
            if (state != full && state != reading) {
                return stop(Status::Unsupported, "reader lacks a complete produced generation");
            }
            consumerSeen = true;
            role.acquireReady = state == full;
            emptySeen = false;
            const unsigned following = after[n] | (node.next.empty() ? 4u : 0u);
            if (
                !(following & 2u) && following) {
                role.publishFree = true;
                state = 0;
                reused = true;
            } else if (following == 2u) {
                state = reading;
            } else {
                return stop(Status::Unsupported, "conditional final reader not separated by existing control facts");
            }
        }
        if (roles[n] && !(*roles[n] == role)) {
            return stop(Status::Unsupported, "history-dependent frontier is not determined by available guard state");
        }
        roles[n] = role;
        if (exit) {
            exitSeen = true;
            if (state != 0) {
                return stop(Status::Unsupported, "live generation escapes without a proved readerless transition");
            }
        }
        for (unsigned next : node.next) {
            work.emplace_back(next, state, reused, emptySeen);
        }
    }
    if (
        (!consumerSeen && !bypassSeen) || !exitSeen) {
        return stop(Status::Unsupported, "no complete reachable boundary lifecycle");
    }
    result.roles.resize(spec.phases.size());
    result.nodeRoles.resize(program.nodes.size());
    result.uniformRoles.assign(spec.phases.size(), true);
    std::vector<bool> seenPhase(spec.phases.size(), false);
    for (unsigned n = 0; n < roles.size(); ++n) {
        if (!roles[n]) {
            continue;
        }
        result.nodeRoles[n] = *roles[n];
        const auto& node = program.nodes[n];
        if (node.kind != Node::Kind::Issue) {
            continue;
        }
        if (seenPhase[node.phase] && !(result.roles[node.phase] == *roles[n])) {
            result.uniformRoles[node.phase] = false;
        } else if (!seenPhase[node.phase]) {
            result.roles[node.phase] = *roles[n];
        }
        seenPhase[node.phase] = true;
    }
    result.status = Status::Complete;
    result.reason = "guarded first/final roles and deferred readerless credit return";
    return result;
}

// APInt's signed extraction of i1 true is -1, whereas guard domains use
// boolean values 0 and 1. Normalize before recording equality constraints.
inline int64_t canonicalGuardConstant(int64_t signedValue, unsigned bitWidth)
{
    return bitWidth == 1 ? (signedValue != 0 ? 1 : 0) : signedValue;
}

// Sufficient 32-bit-safe qualification for an existing (iv + step < upper)
// test. On an executed iteration 0 <= iv <= upper-1. No new no-wrap promise.
inline bool safeNextIterationPredicate(int64_t maximumUpper, int64_t step)
{
    return maximumUpper > 0 && step > 0 && __int128(maximumUpper) - 1 + step <= std::numeric_limits<int32_t>::max();
}

// Three-valued facts at one static insertion point. Unknown must never be used
// to exclude a failing path. These are compiler-derived conditions, not input
// annotations. The native adapter checks dominance and actual expression shape.
using RoleFacts = std::vector<int8_t>; // -1 unknown, 0 false, 1 true
struct RoleCase {
    RoleFacts facts;
    bool performs = false;
};
struct GuardLiteral {
    unsigned test = 0;
    bool truth = true;
    bool operator==(const GuardLiteral& x) const { return test == x.test && truth == x.truth; }
};
using GuardConjunction = std::vector<GuardLiteral>;
struct GuardedRole {
    enum class Status { Complete, Unexpressible, AnalysisLimit, InvalidInput };
    Status status = Status::InvalidInput;
    std::vector<GuardConjunction> alternatives; // empty OR is false; empty AND is true
};
inline bool excludes(const GuardConjunction& term, const RoleFacts& facts)
{
    for (const auto& literal : term) {
        if (
            facts[literal.test] >= 0 && bool(facts[literal.test]) != literal.truth) {
            return true;
        }
    }
    return false;
}
inline bool subsumes(const GuardConjunction& a, const GuardConjunction& b)
{
    return std::all_of(
        a.begin(), a.end(), [&](const GuardLiteral& x) { return std::find(b.begin(), b.end(), x) != b.end(); });
}
inline GuardedRole synthesizeRoleGuard(const std::vector<RoleCase>& cases, Budget& budget, unsigned maximumTerms = 32)
{
    GuardedRole result;
    if (cases.empty()) {
        result.status = GuardedRole::Status::Complete;
        return result;
    }
    const unsigned tests = cases.front().facts.size();
    if (tests > 128) {
        result.status = GuardedRole::Status::AnalysisLimit;
        return result;
    }
    for (const auto& c : cases) {
        if (
            c.facts.size() != tests ||
            std::any_of(c.facts.begin(), c.facts.end(), [](int8_t v) { return v < -1 || v > 1; })) {
            return result;
        }
    }
    auto safe = [&](const GuardConjunction& term) {
        for (const auto& c : cases) {
            if (!budget.spend(1 + term.size())) {
                return false;
            }
            if (!c.performs && !excludes(term, c.facts)) {
                return false;
            }
        }
        return true;
    };
    for (const auto& c : cases) {
        if (!c.performs) {
            continue;
        }
        GuardConjunction term;
        for (unsigned i = 0; i < tests; ++i) {
            if (c.facts[i] >= 0) {
                term.push_back({i, bool(c.facts[i])});
            }
        }
        if (!safe(term)) {
            result.status = budget.left ? GuardedRole::Status::Unexpressible : GuardedRole::Status::AnalysisLimit;
            result.alternatives.clear();
            return result;
        }
        // Deterministic literal deletion. Removing a literal must still exclude
        // EVERY false case; positive unknown assignments must remain covered.
        for (size_t i = term.size(); i-- > 0;) {
            auto trial = term;
            trial.erase(trial.begin() + i);
            if (safe(trial)) {
                term = std::move(trial);
            }
            if (!budget.left) {
                result.status = GuardedRole::Status::AnalysisLimit;
                result.alternatives.clear();
                return result;
            }
        }
        if (std::any_of(result.alternatives.begin(), result.alternatives.end(), [&](const auto& old) {
                return subsumes(old, term);
            })) {
            continue;
        }
        result.alternatives.erase(
            std::remove_if(
                result.alternatives.begin(), result.alternatives.end(),
                [&](const auto& old) { return subsumes(term, old); }),
            result.alternatives.end());
        result.alternatives.push_back(std::move(term));
        if (
            result.alternatives.size() > maximumTerms) {
            result.status = GuardedRole::Status::AnalysisLimit;
            result.alternatives.clear();
            return result;
        }
    }
    result.status = GuardedRole::Status::Complete;
    return result;
}

// An optional failure retries from the original unsynchronized function. This
// helper chooses ONLY an identity to omit; it never edits a partly repaired plan.
inline std::optional<unsigned> chooseLifecycleRetry(
    const std::vector<LogicalLifecycle>& plans, unsigned source = kInvalid, unsigned target = kInvalid)
{
    for (auto i = plans.rbegin(); i != plans.rend(); ++i) {
        const auto& s = i->spec;
        if (
            source == kInvalid || (s.producerLane == source && s.consumerLane == target) ||
            (s.consumerLane == source && s.producerLane == target)) {
            return i->identity;
        }
    }
    return std::nullopt;
}
} // namespace mlir::pto::insert_sync_frontier
#endif
