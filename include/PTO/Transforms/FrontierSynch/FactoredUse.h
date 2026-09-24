// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FACTOREDUSE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FACTOREDUSE_H

// The expression transfer is independent of MLIR. FactoredProvenance supplies
// its qualified original-control/effect projection; no second effect importer
// or selected-completion state is defined here.
#include "PTO/Transforms/FrontierSynch/OriginalProgramPoints.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::pto::frontiersynch {
inline constexpr std::size_t NoFactoredId = std::numeric_limits<std::size_t>::max();

// value identifies the defining original SSA value, not a test site or its
// spelling. scope identifies the repeated scope in which that value is defined.
// The native adapter uses borrowed Value identity, valid for one original IR
// version. Qualification of availability at a *different* endpoint is separate.
struct FactoredGuardIdentity {
    std::uintptr_t value = 0;
    std::size_t scope = NoFactoredId;
    bool operator<(const FactoredGuardIdentity& other) const
    {
        return std::tie(value, scope) < std::tie(other.value, other.scope);
    }
    bool operator==(const FactoredGuardIdentity& other) const { return value == other.value && scope == other.scope; }
};

// A local body visit is not an invocation, a backedge, or a while-before visit.
// Changing this interpretation requires a new arena or a qualified transport;
// merely passing roots from an arena with another frame is rejected.
struct FactoredUseFrame {
    enum class Kind { Invocation, ForBody, WhileBefore, WhileAfter } kind = Kind::Invocation;
    std::uintptr_t original = 0;
    std::size_t version = 0, cell = NoFactoredId, owner = NoFactoredId;
    OriginalProgramVersion snapshot;
    bool operator==(const FactoredUseFrame& other) const
    {
        return std::tie(kind, original, version, cell, owner, snapshot) ==
               std::tie(other.kind, other.original, other.version, other.cell, other.owner, other.snapshot);
    }
};

struct FactoredUseAccess {
    std::size_t operation = NoFactoredId;
    bool read = false, write = false, definiteWrite = false;
    // Indices in this original operation's translated access vector. A retained
    // access node owns these immutable lists, even across composed projections.
    std::vector<std::size_t> readIncidences, writeIncidences;
};

struct FactoredUseNode {
    enum class Kind { Empty, True, Test, Incoming, Access, Both, Choose, Demand, Unresolved } kind = Kind::Empty;
    enum class Sort { Origins, Demands, Condition } sort = Sort::Origins;
    enum class Hazard { RAW, WAR, WAW } hazard = Hazard::RAW;
    enum class Role { Writer, Reader } role = Role::Writer;
    enum class Boundary { Entry, Exit } boundary = Boundary::Entry;
    std::size_t left = 0, right = 0, condition = 0;
    std::size_t owner = NoFactoredId, operation = NoFactoredId;
    FactoredGuardIdentity guard;
    std::shared_ptr<const FactoredUseAccess> access;
    bool containsIncoming = false, containsUnresolved = false;
};

class FactoredUseArena {
public:
    explicit FactoredUseArena(FactoredUseFrame frame) : interpretation(frame)
    {
        storage.push_back({}); // Polymorphic empty set / false.
        FactoredUseNode one;
        one.kind = FactoredUseNode::Kind::True;
        one.sort = FactoredUseNode::Sort::Condition;
        storage.push_back(one);
    }
    const FactoredUseFrame& frame() const { return interpretation; }
    const std::vector<FactoredUseNode>& nodes() const { return storage; }
    const FactoredUseNode& operator[](std::size_t id) const { return storage.at(id); }
    std::size_t attempts() const { return constructionAttempts; }
    bool hasSort(std::size_t id, FactoredUseNode::Sort sort) const
    {
        return id < storage.size() && (!id || storage[id].sort == sort);
    }

    // Guard interning belongs to qualified projection preparation. The transfer
    // receives these handles and only uses constant-arity constructors below.
    std::size_t test(FactoredGuardIdentity guard)
    {
        const auto found = tests.find(guard);
        if (found != tests.end()) {
            return found->second;
        }
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Test;
        node.sort = FactoredUseNode::Sort::Condition;
        node.guard = guard;
        const auto id = append(std::move(node));
        tests.emplace(guard, id);
        return id;
    }
    std::size_t incoming(std::size_t owner, FactoredUseNode::Boundary boundary, FactoredUseNode::Role role)
    {
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Incoming;
        node.owner = owner;
        node.boundary = boundary;
        node.role = role;
        node.containsIncoming = true;
        return append(std::move(node));
    }
    std::size_t origin(std::shared_ptr<const FactoredUseAccess> access)
    {
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Access;
        node.owner = interpretation.owner;
        node.operation = access->operation;
        node.access = std::move(access);
        return append(std::move(node));
    }
    std::size_t unresolved(
        std::size_t owner, FactoredUseNode::Sort sort, FactoredUseNode::Role role = FactoredUseNode::Role::Writer,
        std::size_t left = 0, std::size_t right = 0)
    {
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Unresolved;
        node.sort = sort;
        node.owner = owner;
        node.role = role;
        // A repeated demand keeps its incoming state, rather than erasing it when
        // the repeat transfer is not qualified. Origins/conditions use no operands.
        node.left = left;
        node.right = right;
        node.containsUnresolved = true;
        node.containsIncoming = storage.at(left).containsIncoming || storage.at(right).containsIncoming;
        return append(std::move(node));
    }
    std::size_t both(std::size_t a, std::size_t b, FactoredUseNode::Sort sort)
    {
        ++constructionAttempts;
        assert(sort != FactoredUseNode::Sort::Condition && hasSort(a, sort) && hasSort(b, sort));
        if (!a || a == b) {
            return b;
        }
        if (!b) {
            return a;
        }
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Both;
        node.sort = sort;
        node.left = a;
        node.right = b;
        inherit(node, a, b);
        return append(std::move(node));
    }
    std::size_t choose(std::size_t condition, std::size_t yes, std::size_t no, FactoredUseNode::Sort sort)
    {
        ++constructionAttempts;
        assert(hasSort(condition, FactoredUseNode::Sort::Condition) && hasSort(yes, sort) && hasSort(no, sort));
        if (!condition) {
            return no;
        }
        if (condition == 1 || yes == no) {
            return yes;
        }
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Choose;
        node.sort = sort;
        node.condition = condition;
        node.left = yes;
        node.right = no;
        inherit(node, yes, no);
        node.containsUnresolved |= storage[condition].containsUnresolved;
        return append(std::move(node));
    }
    // Evaluation of a conjunction consults b only when a holds. This matters for
    // a nested original test which is not defined on the bypass path.
    std::size_t conjunction(std::size_t a, std::size_t b) { return choose(a, b, 0, FactoredUseNode::Sort::Condition); }
    std::size_t negate(std::size_t condition) { return choose(condition, 0, 1, FactoredUseNode::Sort::Condition); }
    std::size_t demand(
        std::size_t sources, std::shared_ptr<const FactoredUseAccess> target, FactoredUseNode::Hazard hazard)
    {
        ++constructionAttempts;
        assert(hasSort(sources, FactoredUseNode::Sort::Origins));
        if (!sources) {
            return 0;
        }
        FactoredUseNode node;
        node.kind = FactoredUseNode::Kind::Demand;
        node.sort = FactoredUseNode::Sort::Demands;
        node.hazard = hazard;
        node.left = sources;
        node.operation = target->operation;
        node.owner = interpretation.owner;
        node.access = std::move(target);
        inherit(node, sources, 0);
        return append(std::move(node));
    }

private:
    std::size_t append(FactoredUseNode node)
    {
        ++constructionAttempts;
        const auto id = storage.size();
        storage.push_back(std::move(node));
        return id;
    }
    void inherit(FactoredUseNode& node, std::size_t a, std::size_t b) const
    {
        node.containsIncoming = storage[a].containsIncoming || storage[b].containsIncoming;
        node.containsUnresolved = storage[a].containsUnresolved || storage[b].containsUnresolved;
    }
    FactoredUseFrame interpretation;
    std::vector<FactoredUseNode> storage;
    std::map<FactoredGuardIdentity, std::size_t> tests;
    std::size_t constructionAttempts = 0;
};

struct FactoredUseState {
    std::size_t writers = 0, readers = 0;
};
struct FactoredUseInterface {
    std::shared_ptr<FactoredUseArena> arena;
    FactoredUseState incoming, following;
};

// A finite, already qualified projection of the original syntax. OpaqueRepeat
// retains its original identity and may effects, never a one-iteration model.
// The native adapter obtains the flags from every represented descendant.
struct FactoredUseRegion {
    enum class Kind { Sequence, Choice, Access, OpaqueRepeat, Unresolved } kind = Kind::Sequence;
    std::vector<FactoredUseRegion> children;
    std::size_t access = NoFactoredId, condition = NoFactoredId, owner = NoFactoredId;
    bool mayRead = false, mayWrite = false;
};
struct FactoredUseProjection {
    FactoredUseFrame frame;
    FactoredUseRegion body;
    std::vector<std::shared_ptr<const FactoredUseAccess>> accesses;
};
struct FactoredUseSite {
    std::shared_ptr<const FactoredUseAccess> access;
    std::size_t origin = 0, applicability = 0;
    std::size_t priorWriters = 0, priorReaders = 0, nextWriters = 0, nextReaders = 0;
    std::array<std::size_t, 3> demands{};
    bool visited = false;
};
struct FactoredRepeatedInterface {
    std::size_t owner = NoFactoredId;
    FactoredUseState incoming, outgoing;
    bool backward = false;
    std::string missingPremise = "qualified repeated-region succession and demand summary";
};
struct FactoredFormationStats {
    std::size_t inputNodes = 0, forwardSteps = 0, backwardSteps = 0;
    std::size_t constructorCalls = 0, addedNodes = 0, weakWrites = 0;
};
struct FactoredDemandView {
    std::shared_ptr<const FactoredUseArena> arena;
    std::shared_ptr<const FactoredUseAccess> target;
    FactoredUseNode::Hazard hazard = FactoredUseNode::Hazard::RAW;
    std::size_t sources = 0, applicability = 0, demands = 0;
    bool valid = false;
    bool hasUnresolved() const
    {
        return !valid || (*arena)[demands].containsUnresolved || (*arena)[applicability].containsUnresolved;
    }
    const std::vector<std::size_t>& targetEffects() const
    {
        static const std::vector<std::size_t> empty;
        if (!target) {
            return empty;
        }
        return hazard == FactoredUseNode::Hazard::RAW ? target->readIncidences : target->writeIncidences;
    }
};
struct FactoredUseResult {
    // Formed fixed-use transfer, possibly conditional on explicit boundary
    // parameters. This is NOT full-cell, endpoint, or recurrence qualification.
    bool complete = false;
    std::size_t cell = NoFactoredId;
    FactoredUseFrame frame;
    std::string reason;
    std::shared_ptr<FactoredUseArena> arena;
    std::size_t demands = 0, finalWriters = 0, finalReaders = 0;
    std::size_t entryNextWriters = 0, entryNextReaders = 0;
    std::vector<FactoredUseSite> sites;
    std::unordered_map<std::size_t, std::size_t> siteIndex;
    std::vector<FactoredRepeatedInterface> repeatedInterfaces;
    FactoredFormationStats work;
    const std::vector<FactoredUseNode>& nodes() const
    {
        static const std::vector<FactoredUseNode> empty;
        return arena ? arena->nodes() : empty;
    }
    const FactoredUseSite* site(std::size_t operation) const
    {
        const auto found = siteIndex.find(operation);
        return found == siteIndex.end() ? nullptr : &sites[found->second];
    }
    FactoredDemandView requirement(std::size_t operation, FactoredUseNode::Hazard hazard) const
    {
        FactoredDemandView view;
        const auto* use = site(operation);
        if (!use || !use->visited ||
            (hazard == FactoredUseNode::Hazard::RAW ? !use->access->read : !use->access->write)) {
            return view;
        }
        view.valid = true;
        view.arena = arena;
        view.target = use->access;
        view.hazard = hazard;
        view.sources = hazard == FactoredUseNode::Hazard::WAR ? use->priorReaders : use->priorWriters;
        view.applicability = use->applicability;
        view.demands = use->demands[static_cast<std::size_t>(hazard)];
        return view;
    }
};

// v0.44 I.1/I.6 transfers, including the backward read half of RMW. Formation
// only appends constant-arity references. Neither origins nor guard valuations
// are enumerated, and supplied incoming/following DAGs are not copied.
class FactoredUseBuilder {
public:
    FactoredUseBuilder(FactoredUseProjection projection, FactoredUseInterface boundary)
        : result{}, arena(boundary.arena)
    {
        result.frame = projection.frame;
        result.cell = projection.frame.cell;
        result.arena = arena;
        if (!arena || !(arena->frame() == projection.frame) ||
            !arena->hasSort(boundary.incoming.writers, FactoredUseNode::Sort::Origins) ||
            !arena->hasSort(boundary.incoming.readers, FactoredUseNode::Sort::Origins) ||
            !arena->hasSort(boundary.following.writers, FactoredUseNode::Sort::Origins) ||
            !arena->hasSort(boundary.following.readers, FactoredUseNode::Sort::Origins)) {
            result.reason = "incompatible factored boundary arena, frame, or expression sort";
            return;
        }
        result.complete = true;
        result.work.inputNodes = arena->nodes().size();
        const auto initialAttempts = arena->attempts();
        result.sites.reserve(projection.accesses.size());
        for (auto& access : projection.accesses) {
            if (!access || access->operation == NoFactoredId ||
                !result.siteIndex.emplace(access->operation, result.sites.size()).second) {
                obstruct("invalid or duplicate original operation in fixed-use projection");
                return;
            }
            FactoredUseSite site;
            site.access = std::move(access);
            site.origin = arena->origin(site.access);
            result.sites.push_back(std::move(site));
        }
        State forward{boundary.incoming.writers, boundary.incoming.readers, 0};
        forward = transfer(projection.body, forward, false, 1);
        for (const auto& site : result.sites) {
            if (!site.visited) {
                obstruct("original access is absent from the fixed-use syntax");
            }
        }
        result.demands = forward.demands;
        result.finalWriters = forward.writers;
        result.finalReaders = forward.readers;
        const auto backward =
            transfer(projection.body, {boundary.following.writers, boundary.following.readers, 0}, true, 1);
        result.entryNextWriters = backward.writers;
        result.entryNextReaders = backward.readers;
        result.work.constructorCalls = arena->attempts() - initialAttempts;
        result.work.addedNodes = arena->nodes().size() - result.work.inputNodes;
    }
    const FactoredUseResult& get() const { return result; }
    FactoredUseResult take() { return std::move(result); }

private:
    using Node = FactoredUseNode;
    struct State {
        std::size_t writers = 0, readers = 0, demands = 0;
    };
    void obstruct(const std::string& reason)
    {
        result.complete = false;
        if (result.reason.empty()) {
            result.reason = reason;
        }
    }
    State opaque(const FactoredUseRegion& region, State state, bool backward)
    {
        if (region.kind == FactoredUseRegion::Kind::OpaqueRepeat && !region.mayRead && !region.mayWrite) {
            return state; // An all-access exclusion, not a selected-state identity.
        }
        obstruct(
            region.kind == FactoredUseRegion::Kind::OpaqueRepeat ?
                "qualified repeated-region summary is missing; fixed-use fragments are retained" :
                "unresolved original control or effect projection");
        FactoredRepeatedInterface interface;
        interface.owner = region.owner;
        interface.incoming = {state.writers, state.readers};
        interface.backward = backward;
        if (region.kind == FactoredUseRegion::Kind::Unresolved) {
            interface.missingPremise = "qualified original control or effect projection";
        }
        if (!backward) {
            state.demands = arena->both(
                state.demands,
                arena->unresolved(region.owner, Node::Sort::Demands, Node::Role::Writer, state.writers, state.readers),
                Node::Sort::Demands);
        }
        if (region.mayWrite || region.kind == FactoredUseRegion::Kind::Unresolved) {
            state.writers = arena->both(
                state.writers, arena->unresolved(region.owner, Node::Sort::Origins, Node::Role::Writer),
                Node::Sort::Origins);
        }
        if (region.mayRead || region.kind == FactoredUseRegion::Kind::Unresolved) {
            state.readers = arena->both(
                state.readers, arena->unresolved(region.owner, Node::Sort::Origins, Node::Role::Reader),
                Node::Sort::Origins);
        }
        interface.outgoing = {state.writers, state.readers};
        result.repeatedInterfaces.push_back(std::move(interface));
        return state;
    }
    State transfer(const FactoredUseRegion& region, State state, bool backward, std::size_t activation)
    {
        if (backward) {
            ++result.work.backwardSteps;
        } else {
            ++result.work.forwardSteps;
        }
        if (region.kind == FactoredUseRegion::Kind::OpaqueRepeat ||
            region.kind == FactoredUseRegion::Kind::Unresolved) {
            return opaque(region, state, backward);
        }
        if (region.kind == FactoredUseRegion::Kind::Access) {
            if (region.access >= result.sites.size()) {
                FactoredUseRegion unknown;
                unknown.kind = FactoredUseRegion::Kind::Unresolved;
                return opaque(unknown, state, backward);
            }
            auto& site = result.sites[region.access];
            const auto& effect = *site.access;
            if (!backward) {
                if (site.visited) {
                    FactoredUseRegion unknown;
                    unknown.kind = FactoredUseRegion::Kind::Unresolved;
                    return opaque(unknown, state, backward);
                }
                site.visited = true;
                site.applicability = activation;
                site.priorWriters = state.writers;
                site.priorReaders = state.readers;
                auto addDemand = [&](std::size_t sources, Node::Hazard hazard) {
                    const auto raw = arena->demand(sources, site.access, hazard);
                    site.demands[static_cast<std::size_t>(hazard)] =
                        arena->choose(activation, raw, 0, Node::Sort::Demands);
                    state.demands = arena->both(state.demands, raw, Node::Sort::Demands);
                };
                if (effect.read) {
                    addDemand(state.writers, Node::Hazard::RAW);
                }
                if (effect.write) {
                    addDemand(state.writers, Node::Hazard::WAW);
                    addDemand(state.readers, Node::Hazard::WAR);
                    result.work.weakWrites += !effect.definiteWrite;
                }
            } else {
                site.nextWriters = state.writers;
                site.nextReaders = state.readers;
            }
            // Demands above refer to the OLD writer/reader roots. A full write may
            // replace provenance, never those roots or the previously generated DAG.
            if (effect.write && effect.definiteWrite) {
                state.writers = site.origin;
                state.readers = backward && effect.read ? site.origin : 0;
            } else {
                if (effect.write) {
                    state.writers = arena->both(state.writers, site.origin, Node::Sort::Origins);
                }
                if (effect.read) {
                    state.readers = arena->both(state.readers, site.origin, Node::Sort::Origins);
                }
            }
            return state;
        }
        if (region.kind == FactoredUseRegion::Kind::Choice) {
            if (region.children.size() != 2) {
                FactoredUseRegion unknown;
                unknown.kind = FactoredUseRegion::Kind::Unresolved;
                unknown.owner = region.owner;
                return opaque(unknown, state, backward);
            }
            auto condition = region.condition;
            if (!arena->hasSort(condition, Node::Sort::Condition)) {
                obstruct("original guard identity or occurrence scope is unqualified");
                condition = arena->unresolved(region.owner, Node::Sort::Condition);
            }
            const auto yesActivation = arena->conjunction(activation, condition);
            const auto noActivation = arena->conjunction(activation, arena->negate(condition));
            const auto yes = transfer(region.children[0], state, backward, yesActivation);
            const auto no = transfer(region.children[1], state, backward, noActivation);
            return {
                arena->choose(condition, yes.writers, no.writers, Node::Sort::Origins),
                arena->choose(condition, yes.readers, no.readers, Node::Sort::Origins),
                arena->choose(condition, yes.demands, no.demands, Node::Sort::Demands)};
        }
        if (!backward) {
            for (const auto& child : region.children) {
                state = transfer(child, state, false, activation);
            }
        } else {
            for (auto child = region.children.rbegin(); child != region.children.rend(); ++child) {
                state = transfer(*child, state, true, activation);
            }
        }
        return state;
    }
    FactoredUseResult result;
    std::shared_ptr<FactoredUseArena> arena;
};
} // namespace mlir::pto::frontiersynch
#endif
