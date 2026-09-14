// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite control-flow verification for protocols that remain live across
// region boundaries. Included by StructuredSyncComposition.cpp to share its
// physical prefix, visibility, macro, and hardware transfers.
namespace {
struct OpenProtocolState : PrefixState {
    // Per key: MAY occupancy (empty=1, full=2), and lanes that MUST know the
    // most recent consumption. Initial empty keys need no acknowledgment.
    std::vector<uint8_t> occupancy, known;
    std::vector<std::vector<uint8_t>> acknowledgment;
    OpenProtocolState(unsigned cells, unsigned keys)
        : PrefixState(cells),
          occupancy(keys, 1),
          known(keys, (1u << c::LaneCount) - 1),
          acknowledgment(keys, std::vector<uint8_t>(keys))
    {}
    void join(const OpenProtocolState& other)
    {
        PrefixState::join(other);
        for (unsigned k = 0; k < occupancy.size(); ++k) {
            occupancy[k] |= other.occupancy[k];
            known[k] &= other.known[k];
            for (unsigned j = 0; j < occupancy.size(); ++j)
                acknowledgment[k][j] &= other.acknowledgment[k][j];
        }
    }
    void acquire(unsigned source, unsigned observer)
    {
        State::acquire(source, observer);
        for (auto& lanes : known)
            if (lanes & (1u << source))
                lanes |= uint8_t(1u << observer);
    }
    bool same(const OpenProtocolState& other) const
    {
        auto effectsEqual = [](const c::Effects& a, const c::Effects& b) {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](auto x, auto y) {
                       return x.readers == y.readers && x.writers == y.writers;
                   });
        };
        for (unsigned lane = 0; lane < c::LaneCount; ++lane)
            if (!effectsEqual(pending[lane], other.pending[lane]))
                return false;
        if (receipts.size() != other.receipts.size())
            return false;
        auto a = receipts.begin(), b = other.receipts.begin();
        for (; a != receipts.end(); ++a, ++b)
            if (a->first != b->first || !effectsEqual(a->second, b->second))
                return false;
        return written == other.written && cleanedScalar == other.cleanedScalar &&
               fencedNonScalar == other.fencedNonScalar && remotePending == other.remotePending &&
               remoteScalarStale == other.remoteScalarStale && accumulatorCell == other.accumulatorCell &&
               accumulatorChain == other.accumulatorChain && occupancy == other.occupancy && known == other.known &&
               acknowledgment == other.acknowledgment;
    }
};

c::Result verifyOpenProtocol(const c::Program& p, const Commands& actual, uint64_t limit, Commands* residual)
{
    using namespace c;
    c::Result result;
    using Key = PrefixState::EventKey;
    std::map<Key, unsigned> keys;
    uint64_t work = 0;
    auto charge = [&](uint64_t count) {
        if (count > limit - work) {
            result.reason = "open protocol verification exhausted its bounded allowance";
            return false;
        }
        work += count;
        result.selectedAccountingWork = work;
        return true;
    };
    if (p.cells > MaxCells || p.nodes.empty() || actual.size() != p.nodes.size() ||
        !charge(uint64_t(p.nodes.size()) * (1 + uint64_t(p.cells) * 4)))
        return result;
    for (const auto& node : p.nodes)
        if (!charge(node.children.size() + node.macroTransfers.size() +
                    uint64_t(node.macroPhases.size()) * (1 + uint64_t(p.cells) * 2)))
            return result;
    for (const auto& fixed : p.fixedBefore)
        if (!charge(uint64_t(fixed.size()) * (1 + uint64_t(p.cells))))
            return result;
    std::vector<Effects> effects;
    if (!summarize(p, effects, result.reason))
        return result;
    for (const auto& site : actual)
        for (const auto& m : site) {
            if (!charge(1 + p.target.compilerKeys.size() + p.target.reservations.size()))
                return result;
            if (m.kind != Mechanism::Publish && m.kind != Mechanism::Acquire) {
                if (!validMechanism(p, m)) {
                    result.reason = "invalid open protocol mechanism";
                    return result;
                }
            } else if (
                m.participation != Mechanism::Every || m.loop != NoCut || m.word != NoCut || m.first >= LaneCount ||
                m.second >= LaneCount || !p.target.event(lane(p, m.first), lane(p, m.second)) ||
                std::find(p.target.compilerKeys.begin(), p.target.compilerKeys.end(), m.forwardKey) ==
                    p.target.compilerKeys.end() ||
                !p.target.available(lane(p, m.first), lane(p, m.second), m.forwardKey)) {
                result.reason = "invalid open protocol key or participation";
                return result;
            }
            if (m.kind == Mechanism::Publish || m.kind == Mechanism::Acquire || m.kind == Mechanism::Rendezvous)
                keys.emplace(Key{m.first, m.second, m.forwardKey}, keys.size());
            if (m.kind == Mechanism::Rendezvous)
                keys.emplace(Key{m.second, m.first, m.reverseKey}, keys.size());
        }
    // State copies, joins and products are charged before allocation. The
    // target bounds key count; this allowance is separate from fallback work.
    uint64_t width =
        1 + uint64_t(keys.size()) * (keys.size() + 3) + uint64_t(p.cells) * (8 * LaneCount + 4 * keys.size() + 8);
    if (!charge(width))
        return result;
    auto endpoint = [&](OpenProtocolState& state, const Mechanism& m, bool validate, unsigned site) {
        unsigned k = keys.at({m.first, m.second, m.forwardKey});
        if (m.kind == Mechanism::Publish) {
            if (validate && (state.occupancy[k] != 1 || !(state.known[k] & (1u << m.first)))) {
                result.reason = "open publication can rearm before fresh consumption at node " +
                    std::to_string(site) + " for " + std::to_string(m.first) + "->" +
                    std::to_string(m.second) + " key " + std::to_string(m.forwardKey) +
                    " occupancy " + std::to_string(state.occupancy[k]);
                return false;
            }
            for (unsigned j = 0; j < keys.size(); ++j)
                state.acknowledgment[k][j] = bool(state.known[j] & (1u << m.first));
            state.occupancy[k] = 2;
            state.publish(m);
        } else {
            if (validate && (state.occupancy[k] != 2 || !state.receipts.count({m.first, m.second, m.forwardKey}))) {
                result.reason = "open acquisition has no must-established publication at node " +
                    std::to_string(site) + " for " + std::to_string(m.first) + "->" +
                    std::to_string(m.second) + " key " + std::to_string(m.forwardKey) +
                    " occupancy " + std::to_string(state.occupancy[k]);
                return false;
            }
            state.acquirePrefix(m);
            for (unsigned j = 0; j < keys.size(); ++j)
                if (state.acknowledgment[k][j])
                    state.known[j] |= uint8_t(1u << m.second);
            // Fresh consumption invalidates every stale generation, including
            // snapshots still carried by other live event tokens.
            state.known[k] = uint8_t(1u << m.second);
            for (auto& snapshot : state.acknowledgment)
                snapshot[k] = 0;
            state.occupancy[k] = 1;
        }
        return true;
    };
    std::function<bool(unsigned, OpenProtocolState&, bool, unsigned, unsigned, int)> visit;
    visit = [&](unsigned id, OpenProtocolState& state, bool validate, unsigned activeOwner,
                unsigned residue, int more) {
        uint64_t transfers = actual[id].size() * 4 + p.nodes[id].macroPhases.size() * (1 + p.nodes[id].macroTransfers.size());
        if (!p.fixedBefore.empty())
            transfers += p.fixedBefore[id].size();
        if (transfers >= (limit - work) / width) {
            result.reason = "open protocol transfer exhausted its bounded allowance";
            return false;
        }
        if (!charge((1 + transfers) * width))
            return false;
        ++result.nodeVisits;
        result.cellVisits += p.cells * LaneCount;
        if (!applyFixed(p, id, state, result.reason))
            return false;
        auto visibilityState = state;
        for (const auto& m : actual[id]) {
            if (m.kind == Mechanism::Publish || m.kind == Mechanism::Acquire) {
                if (!endpoint(state, m, validate, id))
                    return false;
            } else if (m.kind == Mechanism::Rendezvous) {
                for (auto e :
                     {event(Mechanism::Publish, m.first, m.second, m.forwardKey),
                      event(Mechanism::Acquire, m.first, m.second, m.forwardKey),
                      event(Mechanism::Publish, m.second, m.first, m.reverseKey),
                      event(Mechanism::Acquire, m.second, m.first, m.reverseKey)})
                    if (!endpoint(state, e, validate, id))
                        return false;
            } else
                apply(p, state, m);
        }
        const auto& node = p.nodes[id];
        if (node.kind == Node::Operation) {
            if (validate && residual) {
                auto& commands = (*residual)[id];
                if (!realizeVisibility(p, node, state, commands, result.reason, result.visibilityRequirements))
                    return false;
                for (unsigned source = 0; source < LaneCount; ++source) {
                    if (!(operationDemands(p, node, state) & (1u << source))) continue;
                    ++result.acquisitions;
                    std::vector<Mechanism> generated;
                    if (!acquire(p, source, node.lane, state, generated)) {
                        result.reason = "target cannot realize residual physical obligation";
                        return false;
                    }
                    // Same-cut exchanges transport consumption knowledge too.
                    // These are proposal effects only; the next complete pass
                    // reconstructs their real SET/WAIT occupancy and receipts.
                    for (const auto& command : generated)
                        if (command.kind == Mechanism::Rendezvous)
                            for (auto& known : state.known) {
                                if (known & (1u << command.first)) known |= 1u << command.second;
                                if (known & (1u << command.second)) known |= 1u << command.first;
                            }
                    for (const auto& command : generated)
                        if (std::find(commands.begin(), commands.end(), command) == commands.end())
                            commands.push_back(command);
                }
            }
            if (validate && !residual &&
                (!visibilityCovered(p, node, visibilityState, actual[id]) || operationDemands(p, node, state))) {
                result.reason = "uncovered open-protocol physical or visibility demand at node " + std::to_string(id);
                return false;
            }
            seedOperation(p, node, state);
            return true;
        }
        if (node.kind == Node::Macro)
            return executeMacro(p, node, state, result.reason, validate);
        if (node.kind == Node::Choice) {
            if (node.nextIterationOwner == activeOwner && more >= 0)
                return visit(node.children[more ? 0 : 1], state, validate, activeOwner, residue, more);
            if (node.periodicOwner == activeOwner && node.periodicPeriod && residue < node.periodicPeriod) {
                unsigned arm = bool(node.periodicResidues & (uint32_t(1) << residue)) ? 0 : 1;
                return visit(node.children[arm], state, validate, activeOwner, residue, more);
            }
            if (!charge(2 * width))
                return false;
            auto other = state;
            if (!visit(node.children[0], state, validate, activeOwner, residue, more) ||
                !visit(node.children[1], other, validate, activeOwner, residue, more))
                return false;
            state.join(other);
            return true;
        }
        if (node.kind == Node::For || node.kind == Node::While) {
            if (!charge(2 * width))
                return false;
            const auto entry = state;
            unsigned period = 0;
            if (node.kind == Node::For && activeOwner == NoCut) {
                if (node.nonEmpty)
                    period = 1;
                std::function<void(unsigned)> findPeriod = [&](unsigned child) {
                    const auto& nested = p.nodes[child];
                    if (nested.periodicOwner == id && nested.periodicPeriod)
                        period = period > 1 && period != nested.periodicPeriod ? 33 : nested.periodicPeriod;
                    if (nested.nextIterationOwner == id && !period)
                        period = 1;
                    if (nested.kind != Node::For && nested.kind != Node::While)
                        for (unsigned next : nested.children)
                            findPeriod(next);
                };
                for (unsigned child : node.children)
                    findPeriod(child);
            }
            if (period && period <= 32 && node.periodicLower >= 0) {
                if (!charge(uint64_t(period) * 8 * width))
                    return false;
                std::vector<std::optional<OpenProtocolState>> starts(period);
                unsigned first = unsigned(node.periodicLower) % period;
                starts[first] = entry;
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (unsigned current = 0; current < period; ++current) {
                        if (!starts[current])
                            continue;
                        auto backedge = *starts[current];
                        for (unsigned child : node.children)
                            if (!visit(child, backedge, false, id, current, 1))
                                return false;
                        unsigned next = (current + 1) % period;
                        if (!starts[next]) {
                            starts[next] = std::move(backedge);
                            changed = true;
                        } else {
                            auto joined = *starts[next];
                            joined.join(backedge);
                            if (!joined.same(*starts[next])) {
                                starts[next] = std::move(joined);
                                changed = true;
                            }
                        }
                    }
                }
                std::optional<OpenProtocolState> exit;
                if (!node.nonEmpty)
                    exit = entry; // original zero-trip successor
                for (unsigned current = 0; current < period; ++current) {
                    if (!starts[current])
                        continue;
                    auto continuing = *starts[current];
                    for (unsigned child : node.children)
                        if (!visit(child, continuing, validate, id, current, 1))
                            return false;
                    auto last = *starts[current];
                    for (unsigned child : node.children)
                        if (!visit(child, last, validate, id, current, 0))
                            return false;
                    if (!exit)
                        exit = std::move(last);
                    else
                        exit->join(last);
                }
                if (!exit) {
                    result.reason = "qualified nonempty loop has no reachable exit state";
                    return false;
                }
                state = std::move(*exit);
                return true;
            }
            auto invariant = entry;
            for (;;) {
                if (!charge(3 * width))
                    return false;
                auto backedge = invariant;
                for (unsigned child : node.children)
                    if (!visit(child, backedge, false, NoCut, 0, -1))
                        return false;
                backedge.join(entry);
                // Widen monotonically even when an optional must witness is
                // reestablished by a transfer. This is finite lattice dataflow,
                // independent of dynamic trip counts and branch decisions.
                backedge.join(invariant);
                if (backedge.same(invariant))
                    break;
                invariant = std::move(backedge);
            }
            state = invariant;
            for (unsigned child : node.children)
                if (!visit(child, state, validate, NoCut, 0, -1))
                    return false;
            if (node.kind == Node::For)
                state.join(entry); // Preserve the original zero-trip successor.
            else {
                // A while exits after its before-region, not its after-region.
                state = invariant;
                if (!visit(node.children[0], state, validate, NoCut, 0, -1))
                    return false;
            }
            return true;
        }
        for (unsigned child : node.children)
            if (!visit(child, state, validate, activeOwner, residue, more))
                return false;
        return true;
    };
    OpenProtocolState initial(p.cells, keys.size()), invariant = initial;
    // Repeated invocations share event generations. Only actual modeled
    // commands may retire physical work; no implicit drain is introduced.
    for (;;) {
        if (!charge(3 * width))
            return result;
        auto next = invariant;
        if (!visit(p.nodes.size() - 1, next, false, NoCut, 0, -1))
            return result;
        next.join(initial);
        next.join(invariant);
        if (next.same(invariant))
            break;
        invariant = std::move(next);
    }
    if (!visit(p.nodes.size() - 1, invariant, true, NoCut, 0, -1))
        return result;
    if (std::any_of(invariant.occupancy.begin(), invariant.occupancy.end(), [](auto bits) { return bits != 1; })) {
        result.reason = "open protocol exports an unconsumed lifetime credit";
        return result;
    }
    result.fixedActions = fixedActionCount(p);
    result.success = true;
    return result;
}
} // namespace

c::Result c::testing::verifyOpenDemands(
    const Program& p, const std::vector<std::vector<Mechanism>>& actual, uint64_t limit)
{
    return verifyOpenProtocol(p, actual, limit);
}
