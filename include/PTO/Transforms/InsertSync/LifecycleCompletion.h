// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LIFECYCLECOMPLETION_H
#define PTO_TRANSFORMS_INSERTSYNC_LIFECYCLECOMPLETION_H

#include "PTO/Transforms/InsertSync/LifecycleProtocol.h"

namespace mlir::pto::insert_sync_frontier {

// One local, forward residual handoff at unchanged physical-phase endpoints.
// Native admission additionally requires the two endpoints in the same block,
// no backedge/compensation/dynamic selector, and the actual before/after lists.
// Orders describe those lists, NOT the order of syncOperations storage.
struct ResidualCompletionHandoff {
    unsigned source = kInvalid, target = kInvalid;
    unsigned setOrder = 0, waitOrder = 0;
};

struct LifecycleCompletionProjection {
    enum class Status { Complete, InvalidInput, AnalysisLimit };
    Status status = Status::InvalidInput;
    Program program;
    // Original guarded graph node -> physical node in the projection.
    std::vector<unsigned> physicalNode;
    unsigned protocolKeys = 0, residualKeys = 0;
    std::string reason;
};

// Add a READ-ONLY overlay of logical events to the existing guarded control
// graph. No payload or IR is changed. Keys are logical identities, unrelated to
// hardware capacity. Every generated named barrier is absent from this view:
// it is impossible for a barrier candidate to prove its own redundancy here.
inline LifecycleCompletionProjection projectLifecycleCompletion(
    const Program& base, const std::vector<LogicalLifecycle>& channels,
    const std::vector<ResidualCompletionHandoff>& residuals, Budget& budget, bool includeConcreteResiduals = false)
{
    LifecycleCompletionProjection result;
    using Status = LifecycleCompletionProjection::Status;
    auto fail = [&](Status status, const char* why) {
        result.status = status;
        result.reason = why;
        result.program = Program{};
        result.physicalNode.clear();
        return result;
    };
    if (
        !valid(base) || channels.empty() || (!includeConcreteResiduals && !base.keys.empty())) {
        return fail(Status::InvalidInput, "expected an unsynchronized structured program and selected lifecycles");
    }
    for (const Node& node : base.nodes) {
        if (!includeConcreteResiduals && (node.kind == Node::Kind::Signal || node.kind == Node::Kind::Wait ||
            node.kind == Node::Kind::Barrier || node.kind == Node::Kind::All)) {
            return fail(Status::InvalidInput, "fixed or generated synchronization is not structural input");
        }
    }
    if (
        channels.size() > 32 || 2 * channels.size() + residuals.size() + base.keys.size() > 64) {
        return fail(Status::AnalysisLimit, "logical completion-key analysis limit (not hardware scarcity)");
    }

    Program& p = result.program;
    p.lanes = base.lanes;
    p.phaseLane = base.phaseLane;
    p.allowUnrepresentedPhases = base.allowUnrepresentedPhases;
    for (const auto& channel : channels) {
        const auto& spec = channel.spec;
        const auto& cert = channel.certificate;
        if (
            cert.status != LifecycleCertificate::Status::Complete || spec.phases.size() != base.phaseLane.size() ||
            spec.producerLane >= base.lanes || spec.consumerLane >= base.lanes ||
            spec.producerLane == spec.consumerLane ||
            (!cert.nodeRoles.empty() && cert.nodeRoles.size() != base.nodes.size()) ||
            (cert.nodeRoles.empty() && cert.roles.size() != spec.phases.size())) {
            return fail(Status::InvalidInput, "inconsistent certified lifecycle projection");
        }
        p.keys.push_back({spec.producerLane, spec.consumerLane}); // Ready
        p.keys.push_back({spec.consumerLane, spec.producerLane}); // Free
    }
    result.protocolKeys = p.keys.size();
    std::vector<std::vector<unsigned>> beforeResidual(base.phaseLane.size());
    std::vector<std::vector<unsigned>> afterResidual(base.phaseLane.size());
    for (unsigned i = 0; i < residuals.size(); ++i) {
        const auto& r = residuals[i];
        if (
            r.source >= base.phaseLane.size() || r.target >= base.phaseLane.size() ||
            base.phaseLane[r.source] == base.phaseLane[r.target]) {
            return fail(Status::InvalidInput, "invalid residual completion endpoints");
        }
        p.keys.push_back({base.phaseLane[r.source], base.phaseLane[r.target]});
        beforeResidual[r.target].push_back(i);
        afterResidual[r.source].push_back(i);
    }
    result.residualKeys = residuals.size();
    const unsigned concreteOffset = p.keys.size();
    if (includeConcreteResiduals) {
        p.keys.insert(p.keys.end(), base.keys.begin(), base.keys.end());
        result.residualKeys += base.keys.size();
    }
    for (auto& ids : beforeResidual) {
        std::stable_sort(ids.begin(), ids.end(), [&](unsigned a, unsigned b) {
            return residuals[a].waitOrder < residuals[b].waitOrder;
        });
    }
    for (auto& ids : afterResidual) {
        std::stable_sort(ids.begin(), ids.end(), [&](unsigned a, unsigned b) {
            return residuals[a].setOrder < residuals[b].setOrder;
        });
    }

    bool limit = false;
    auto append = [&](Node node) -> unsigned {
        if (
            limit || !budget.spend() || p.nodes.size() >= 8192) {
            limit = true;
            return kInvalid;
        }
        unsigned id = p.nodes.size();
        p.nodes.push_back(std::move(node));
        return id;
    };
    auto connect = [&](unsigned a, unsigned b) {
        if (!limit && a != kInvalid && b != kInvalid) {
            p.nodes[a].next.push_back(b);
        }
    };
    auto action = [&](unsigned& tail, Node::Kind kind, unsigned key) {
        Node node;
        node.kind = kind;
        node.key = key;
        node.lane = kind == Node::Kind::Signal ? p.keys[key].source : p.keys[key].target;
        unsigned id = append(node);
        connect(tail, id);
        tail = id;
    };

    unsigned entry = append(Node{}), primeTail = entry;
    for (unsigned c = 0; c < channels.size(); ++c) {
        action(primeTail, Node::Kind::Signal, 2 * c + 1);
    }
    result.physicalNode.resize(base.nodes.size(), kInvalid);
    std::vector<unsigned> start(base.nodes.size()), end(base.nodes.size());
    const bool hasExit =
        std::any_of(base.nodes.begin(), base.nodes.end(), [](const Node& n) { return n.kind == Node::Kind::Exit; });
    for (unsigned n = 0; n < base.nodes.size(); ++n) {
        const Node& old = base.nodes[n];
        unsigned tail = append(Node{});
        start[n] = tail;
        if (old.kind == Node::Kind::Issue) {
            for (unsigned i : beforeResidual[old.phase]) {
                action(tail, Node::Kind::Wait, result.protocolKeys + i);
            }
        }
        // Codegen emits residuals first. Protocol materialization then inserts
        // its waits immediately before the original phase (after residual waits),
        // and its signals immediately after it (before residual signals).
        for (unsigned c = 0; c < channels.size(); ++c) {
            if (!budget.spend()) {
                limit = true;
            }
            const auto& cert = channels[c].certificate;
            LifecycleRole role = !cert.nodeRoles.empty() ?
                                     cert.nodeRoles[n] :
                                     (old.kind == Node::Kind::Issue ? cert.roles[old.phase] : LifecycleRole{});
            if (role.publishBefore) action(tail, Node::Kind::Signal, 2 * c);
            if (role.releaseBefore) action(tail, Node::Kind::Signal, 2 * c + 1);
            if (role.bypassReady) {
                action(tail, Node::Kind::Wait, 2 * c);
                action(tail, Node::Kind::Signal, 2 * c + 1);
            }
            if (role.acquireFree) {
                action(tail, Node::Kind::Wait, 2 * c + 1);
            }
            if (role.acquireReady) {
                action(tail, Node::Kind::Wait, 2 * c);
            }
        }
        // NativeGraph may have a trailing bookkeeping Pass after an Exit.
        // Do not drain once at Exit and a second time at that leaf.
        const bool closes = old.kind == Node::Kind::Exit || (!hasExit && old.next.empty());
        if (closes) {
            for (unsigned c = 0; c < channels.size(); ++c) {
                action(tail, Node::Kind::Wait, 2 * c + 1);
            }
        }
        Node physical = old;
        if (physical.kind == Node::Kind::Signal || physical.kind == Node::Kind::Wait)
            physical.key += concreteOffset;
        physical.next.clear();
        unsigned id = append(std::move(physical));
        connect(tail, id);
        tail = id;
        result.physicalNode[n] = id;
        for (unsigned c = 0; c < channels.size(); ++c) {
            if (!budget.spend()) {
                limit = true;
            }
            const auto& cert = channels[c].certificate;
            LifecycleRole role = !cert.nodeRoles.empty() ?
                                     cert.nodeRoles[n] :
                                     (old.kind == Node::Kind::Issue ? cert.roles[old.phase] : LifecycleRole{});
            if (role.publishReady) {
                action(tail, Node::Kind::Signal, 2 * c);
            }
            if (role.publishFree) {
                action(tail, Node::Kind::Signal, 2 * c + 1);
            }
        }
        if (old.kind == Node::Kind::Issue) {
            for (unsigned i : afterResidual[old.phase]) {
                action(tail, Node::Kind::Signal, result.protocolKeys + i);
            }
        }
        end[n] = tail;
        if (limit) {
            break;
        }
    }
    if (limit) {
        return fail(Status::AnalysisLimit, "logical completion projection work/size limit");
    }
    connect(primeTail, start[0]);
    for (unsigned n = 0; n < base.nodes.size(); ++n) {
        for (unsigned successor : base.nodes[n].next) {
            connect(end[n], start[successor]);
        }
    }
    for (const auto& region : base.regions) {
        auto copy = region;
        copy.entry = start[region.entry];
        copy.exit = end[region.exit];
        p.regions.push_back(copy);
    }
    if (!valid(p)) {
        return fail(Status::InvalidInput, "malformed logical completion projection");
    }
    result.status = Status::Complete;
    result.reason = "unchanged physical control plus selected protocol and qualified residual actions";
    return result;
}

struct LifecycleCompletionSupply {
    enum class Status { Complete, InvalidInput, AnalysisLimit, UnprovedEvents };
    Status status = Status::InvalidInput;
    // Index [target][source]. This is full completion of ALL earlier executed
    // source-site occurrences, at EVERY reachable occurrence of target. It is
    // stronger than an exact-slot access certificate and can safely cover an
    // unrelated memory effect performed by that same physical source phase.
    std::vector<Bits> beforePhase;
    std::vector<bool> represented;
    unsigned logicalKeys = 0, residualHandoffs = 0;
    uint64_t transfers = 0;
    std::string reason;
    bool proves(unsigned source, unsigned target) const
    {
        return status == Status::Complete && target < beforePhase.size() && source < beforePhase.size() &&
               represented[target] && beforePhase[target].test(source);
    }
};

inline LifecycleCompletionSupply analyzeLifecycleCompletion(
    const Program& base, const std::vector<LogicalLifecycle>& channels,
    const std::vector<ResidualCompletionHandoff>& residuals, Budget& budget)
{
    LifecycleCompletionSupply result;
    using Status = LifecycleCompletionSupply::Status;
    auto projection = projectLifecycleCompletion(base, channels, residuals, budget);
    if (projection.status != LifecycleCompletionProjection::Status::Complete) {
        result.status = projection.status == LifecycleCompletionProjection::Status::AnalysisLimit ?
                            Status::AnalysisLimit :
                            Status::InvalidInput;
        result.reason = projection.reason;
        return result;
    }
    const Program& program = projection.program;
    auto state = completion(program, Bits(program.nodes.size()), budget);
    result.transfers = state.transfers;
    if (state.status != CompletionResult::Status::Complete || !state.eventsProved) {
        result.status = state.status == CompletionResult::Status::LimitExceeded ?
                            Status::AnalysisLimit :
                            (state.status == CompletionResult::Status::InvalidInput ? Status::InvalidInput :
                                                                                      Status::UnprovedEvents);
        result.reason = "logical event participation/rearm is unproved at projected node " +
                        std::to_string(state.unprovedEventNode);
        return result;
    }
    const unsigned phases = base.phaseLane.size();
    result.beforePhase.assign(phases, Bits(phases, true));
    result.represented.assign(phases, false);
    for (unsigned n = 0; n < program.nodes.size(); ++n) {
        if (
            !budget.spend(1 + (phases + 63) / 64)) {
            result.status = Status::AnalysisLimit;
            result.reason = "completion summary budget";
            result.beforePhase.clear();
            result.represented.clear();
            return result;
        }
        const auto& node = program.nodes[n];
        if (node.kind != Node::Kind::Issue || !state.before[n]) {
            continue;
        }
        result.represented[node.phase] = true;
        result.beforePhase[node.phase].intersect(state.before[n]->known[node.lane]);
    }
    result.logicalKeys = program.keys.size();
    result.residualHandoffs = residuals.size();
    result.status = Status::Complete;
    result.reason = "all-path full completion from the selected combination; not singleton coverage";
    return result;
}

} // namespace mlir::pto::insert_sync_frontier
#endif
