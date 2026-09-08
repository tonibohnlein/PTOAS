// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LIFECYCLEPROTOCOL_H
#define PTO_TRANSFORMS_INSERTSYNC_LIFECYCLEPROTOCOL_H

#include "PTO/Transforms/InsertSync/StorageFrontierDomain.h"
#include <array>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::insert_sync_frontier {

// Reuses R5's immutable structured Program. This is NOT a demand-cover graph.
// Each channel denotes ONE exact physical slot or operand bundle. Successive
// traversals of its write/read protocol denote different dynamic generations;
// another channel's issue never invalidates this channel's current generation.
struct LifecycleTouch {
    unsigned writes = 0;
    unsigned reads = 0;
    unsigned updates = 0; // producer-side read/modify/write; also present in writes
};
struct LifecycleSpec {
    unsigned producerLane = 0;
    unsigned consumerLane = 0;
    unsigned members = 1;
    std::vector<LifecycleTouch> phases;
};
struct LifecycleRole {
    bool acquireFree = false;
    bool publishReady = false;
    bool acquireReady = false;
    bool publishFree = false;
    // Before overwrite/exit: consume an unused Ready and return Free.
    // This is allowed only after a proved empty consumer-region invocation.
    bool bypassReady = false;
    bool publishBefore = false;
    bool releaseBefore = false;
    bool operator==(const LifecycleRole& other) const
    {
        return acquireFree == other.acquireFree && publishReady == other.publishReady &&
               acquireReady == other.acquireReady && publishFree == other.publishFree &&
               bypassReady == other.bypassReady && publishBefore == other.publishBefore &&
               releaseBefore == other.releaseBefore;
    }
};
struct LifecycleCertificate {
    enum class Status { Complete, Unsupported, AnalysisLimit, InvalidInput };
    Status status = Status::Unsupported;
    std::string reason;
    std::vector<LifecycleRole> roles;
    // R7 keeps roles per guarded graph occurrence. Static roles are populated
    // only where uniform; the native adapter must synthesize checked guards.
    std::vector<LifecycleRole> nodeRoles;
    std::vector<bool> uniformRoles;
    bool mayReuse = false; // A second production can follow a completed episode.
    // Only a complete certificate can answer a dependency query. This is a
    // storage/occurrence relationship, NEVER a whole-pipe completion fact.
    enum class Supply { None, Availability, Reclamation, WriteOrder };
    Supply supplies(const LifecycleSpec& spec, unsigned source, unsigned target, unsigned member) const
    {
        if (
            status != Status::Complete || source >= spec.phases.size() || target >= spec.phases.size() ||
            member >= spec.members) {
            return Supply::None;
        }
        unsigned bit = 1u << member;
        const auto a = spec.phases[source], b = spec.phases[target];
        if (
            (a.writes & bit) && (b.reads & bit)) {
            return Supply::Availability;
        }
        if (
            (a.reads & bit) && (b.writes & bit)) {
            return Supply::Reclamation;
        }
        if (
            (a.writes & bit) && (b.writes & bit) && !a.updates && !b.updates) {
            return Supply::WriteOrder;
        }
        return Supply::None;
    }
};

// Recognize the language W_1 ... W_k R+ over an exact physical bundle.
// Every member is written exactly once before the first consumer. Every
// consumer reads the complete bundle. No overwrite may occur until final use.
// Optional CLOSED episodes and arbitrarily many nested repetitions are valid;
// a write with a missing consumer on even one represented exit is not valid.
// Frontiers must be uniform at each static site, otherwise a guarded boundary
// constructor is needed (not guessed here). All paths are checked, not a finite
// sample of loop iterations.
inline LifecycleCertificate recognizeLifecycle(const Program& program, const LifecycleSpec& spec, Budget& budget)
{
    LifecycleCertificate result;
    auto stop = [&](LifecycleCertificate::Status status, const char* reason) {
        result.status = status;
        result.reason = reason;
        result.roles.clear();
        return result;
    };
    using Status = LifecycleCertificate::Status;
    if (
        !valid(program) || spec.producerLane >= program.lanes || spec.consumerLane >= program.lanes ||
        spec.producerLane == spec.consumerLane || !spec.members || spec.members > 4 ||
        spec.phases.size() != program.phaseLane.size()) {
        return stop(Status::InvalidInput, "invalid lifecycle domain");
    }
    const unsigned full = (1u << spec.members) - 1, reading = full + 1;
    for (unsigned p = 0; p < spec.phases.size(); ++p) {
        auto touch = spec.phases[p];
        if (
            (touch.writes | touch.reads) & ~full) {
            return stop(Status::InvalidInput, "invalid member mask");
        }
        if (
            touch.updates || (touch.writes && touch.reads) || (touch.reads && touch.reads != full) ||
            (touch.writes && program.phaseLane[p] != spec.producerLane) ||
            (touch.reads && program.phaseLane[p] != spec.consumerLane)) {
            return stop(Status::Unsupported, "incomplete bundle or incompatible participant");
        }
    }
    // Next relevant access along every represented continuation. Bits:
    // writer=1, reader=2, exit=4. This determines whether a read is certainly
    // final, not lexically final. A conditional last use needs another recipe.
    std::vector<unsigned> first(program.nodes.size(), 0), after(program.nodes.size(), 0);
    bool changed = true;
    while (
        changed) {
        changed = false;
        for (unsigned n = program.nodes.size(); n-- > 0;) {
            if (
                !budget.spend(1 + program.nodes[n].next.size())) {
                return stop(Status::AnalysisLimit, "next-use fixed-point budget");
            }
            const auto& node = program.nodes[n];
            unsigned following = 0;
            for (unsigned next : node.next) {
                following |= first[next];
            }
            unsigned here = following;
            if (
                node.kind == Node::Kind::Exit || node.next.empty()) {
                here |= 4;
            }
            if (
                node.kind == Node::Kind::Issue) {
                const auto touch = spec.phases[node.phase];
                if (
                    touch.writes) {
                    here = 1;
                } else if (touch.reads) {
                    here = 2;
                }
            }
            if (
                first[n] != here || after[n] != following) {
                first[n] = here;
                after[n] = following;
                changed = true;
            }
        }
    }
    // Product: (structured program point, current generation's write/read
    // state). The dynamic generation number is renamed after release. We do
    // not replace it by a static-phase completion bit or assume a reuse distance.
    std::vector<std::array<uint32_t, 2>> seen(program.nodes.size());
    std::deque<std::tuple<unsigned, unsigned, bool>> work{{0, 0, false}};
    std::vector<std::optional<LifecycleRole>> uniform(spec.phases.size());
    bool consumer = false, exitSeen = false;
    while (
        !work.empty()) {
        auto [n, state, releasedBefore] = work.front();
        work.pop_front();
        if (
            !budget.spend()) {
            return stop(Status::AnalysisLimit, "generation-transfer budget");
        }
        if (
            seen[n][releasedBefore] & (1u << state)) {
            continue;
        }
        seen[n][releasedBefore] |= 1u << state;
        const auto& node = program.nodes[n];
        unsigned outgoing = state;
        if (
            node.kind == Node::Kind::Issue) {
            const auto touch = spec.phases[node.phase];
            LifecycleRole role;
            if (
                touch.writes) {
                if (
                    state > full || (state & touch.writes)) {
                    return stop(Status::Unsupported, "overwrite before generation's final use");
                }
                role.acquireFree = state == 0;
                result.mayReuse |= role.acquireFree && releasedBefore;
                outgoing = state | touch.writes;
                role.publishReady = outgoing == full;
            } else if (touch.reads) {
                if (
                    state != full && state != reading) {
                    return stop(Status::Unsupported, "consumer lacks a complete produced generation");
                }
                consumer = true;
                role.acquireReady = state == full;
                const unsigned following = after[n] | (node.next.empty() ? 4u : 0u);
                if (
                    !(following & 2u) && following != 0) {
                    role.publishFree = true;
                    outgoing = 0;
                    releasedBefore = true;
                } else if (following == 2u) {
                    outgoing = reading;
                } else {
                    return stop(Status::Unsupported, "conditional final use needs a guarded recipe");
                }
            }
            if (
                touch.writes || touch.reads) {
                auto& old = uniform[node.phase];
                if (
                    old && !(*old == role)) {
                    return stop(Status::Unsupported, "nonuniform first/final frontier at static site");
                }
                old = role;
            }
        }
        if (
            node.kind == Node::Kind::Exit || node.next.empty()) {
            exitSeen = true;
            if (
                outgoing != 0) {
                return stop(Status::Unsupported, "live generation escapes lifecycle scope");
            }
        }
        for (unsigned next : node.next) {
            work.emplace_back(next, outgoing, releasedBefore);
        }
    }
    if (
        !consumer || !exitSeen) {
        return stop(Status::Unsupported, "no complete reachable lifecycle");
    }
    result.roles.resize(spec.phases.size());
    for (unsigned p = 0; p < uniform.size(); ++p) {
        if (
            uniform[p]) {
            result.roles[p] = *uniform[p];
        }
    }
    result.status = Status::Complete;
    result.reason = "exact slot W*R+ transfer; ready/current and free/previous generations kept distinct";
    return result;
}

struct LogicalLifecycle {
    unsigned identity = 0;
    LifecycleSpec spec;
    LifecycleCertificate certificate;
    // No hardware keys in a semantic plan. Ready(g) and Free(previous(g))
    // are distinct logical occurrence families. Finite realization follows.
};
struct LifecycleAllocation {
    enum class Status { Complete, ResourceUnresolved, InvalidInput };
    Status status = Status::InvalidInput;
    std::vector<unsigned> ready, free;
    std::string reason;
    unsigned failedIdentity = kInvalid;
    unsigned failedSource = kInvalid, failedTarget = kInvalid;
    uint32_t conflictingKeyMask = 0;
};
using ReservedLifecycleKeys = std::map<std::pair<unsigned, unsigned>, uint32_t>;

// Conservative, nonserializing allocation: a selected lifecycle gets dedicated
// keys not used by any residual stream. No endpoint movement, lifetime widening,
// implicit ring contraction or PIPE_ALL recovery. Failure is NOT a proof of
// silicon scarcity; the unchanged ordinary plan remains the fallback.
inline LifecycleAllocation allocateLifecycles(
    const std::vector<LogicalLifecycle>& plans, ReservedLifecycleKeys used, unsigned availableIds = 6)
{
    LifecycleAllocation result;
    if (
        !availableIds || availableIds > 8) {
        return result;
    }
    for (const auto& plan : plans) {
        if (
            plan.certificate.status != LifecycleCertificate::Status::Complete) {
            return result;
        }
        auto choose = [&](unsigned p, unsigned q) -> std::optional<unsigned> {
            auto& mask = used[{p, q}];
            for (unsigned id = 0; id < availableIds; ++id) {
                if (
                    !(mask & (1u << id))) {
                    mask |= 1u << id;
                    return id;
                }
            }
            return std::nullopt;
        };
        auto ready = choose(plan.spec.producerLane, plan.spec.consumerLane);
        auto free = ready ? choose(plan.spec.consumerLane, plan.spec.producerLane) : std::nullopt;
        if (
            !ready || !free) {
            result.failedIdentity = plan.identity;
            result.failedSource = ready ? plan.spec.consumerLane : plan.spec.producerLane;
            result.failedTarget = ready ? plan.spec.producerLane : plan.spec.consumerLane;
            result.conflictingKeyMask = used[{result.failedSource, result.failedTarget}];
            result.ready.clear();
            result.free.clear();
            result.status = LifecycleAllocation::Status::ResourceUnresolved;
            result.reason = "candidate " + std::to_string(plan.identity) + " needs a key in domain " +
                std::to_string(result.failedSource) + "->" + std::to_string(result.failedTarget) +
                "; occupied mask=" + std::to_string(result.conflictingKeyMask) +
                "; dedicated assignment unresolved; no serialization was introduced";
            return result;
        }
        result.ready.push_back(*ready);
        result.free.push_back(*free);
    }
    result.status = LifecycleAllocation::Status::Complete;
    result.reason = "dedicated directed keys; existing residual key assignments unchanged";
    return result;
}

// Independent protocol-reconstruction alphabet. Native code reconstructs these
// actions from the emitted concrete keys, not from candidate coverage bits.
enum class LifecycleAction {
    None,
    Prime,
    AcquireFree,
    Write,
    Update,
    PublishReady,
    AcquireReady,
    Read,
    PublishFree,
    Drain,
    FreeSignal,
    FreeWait
};
struct ReconstructedLifecycleNode {
    LifecycleAction action = LifecycleAction::None;
    unsigned members = 0;
    std::vector<unsigned> next;
};
inline bool verifyReconstructedLifecycle(
    const std::vector<ReconstructedLifecycleNode>& nodes, unsigned members, Budget& budget)
{
    if (
        nodes.empty() || !members || members > 4) {
        return false;
    }
    const unsigned full = (1u << members) - 1;
    // 0 unprimed, 1 free token, 2 filling, 3 ready token, 4 reading,
    // 5 drained. The write mask is separate. This proves participation and
    // per-slot generation transitions; signal-prefix contracts supply hardware
    // completion. The full event graph is checked independently by R5 as well.
    using State = std::tuple<unsigned, unsigned, unsigned>;
    std::set<State> seen;
    std::deque<State> work{{0, 0, 0}};
    while (
        !work.empty()) {
        auto [n, stage, written] = work.front();
        work.pop_front();
        if (
            !budget.spend() || n >= nodes.size()) {
            return false;
        }
        if (
            !seen.emplace(n, stage, written).second) {
            continue;
        }
        const auto& node = nodes[n];
        switch (node.action) {
            case LifecycleAction::None:
                break;
            case LifecycleAction::Prime:
                if (
                    stage != 0) {
                    return false;
                }
                stage = 1;
                break;
            case LifecycleAction::AcquireFree:
                if (
                    stage != 1) {
                    return false;
                }
                stage = 2;
                written = 0;
                break;
        case LifecycleAction::Write:
            if (
                stage == 6) {
                stage = 2;
            }
            if (
                stage != 2 || !node.members || (node.members & ~full) || (written & node.members)) {
                    return false;
                }
                written |= node.members;
                break;
            case LifecycleAction::Update:
                if (stage != 2 || !node.members || (written & node.members) != node.members)
                    return false;
                break;
            case LifecycleAction::PublishReady:
                if (
                    stage != 2 || written != full) {
                    return false;
                }
                stage = 3;
                break;
            case LifecycleAction::AcquireReady:
                if (
                    stage != 3) {
                    return false;
                }
                stage = 4;
                break;
            case LifecycleAction::Read:
                if (
                    stage != 4 || node.members != full) {
                    return false;
                }
                break;
            case LifecycleAction::PublishFree:
                if (
                    stage != 4) {
                    return false;
                }
                stage = 1;
                written = 0;
                break;
        case LifecycleAction::Drain:
            if (
                stage != 1) {
                return false;
            }
            stage = 5;
            break;
        case LifecycleAction::FreeSignal:
            // Reconstruct from the reverse key and actual protocol state, not
            // the planner's prime/release tags or a lexical-region heuristic.
            if (
                stage != 0 && stage != 4) {
                return false;
            }
            stage = 1;
            written = 0;
            break;
        case LifecycleAction::FreeWait:
            if (
                stage != 1) {
                return false;
            }
            stage = 6;
            written = 0;
            break;
        }
        if (
            node.next.empty() && stage != 5 && !(stage == 6 && written == 0)) {
            return false;
        }
        for (unsigned next : node.next) {
            work.emplace_back(next, stage, written);
        }
    }
    return true;
}

} // namespace mlir::pto::insert_sync_frontier
#endif
