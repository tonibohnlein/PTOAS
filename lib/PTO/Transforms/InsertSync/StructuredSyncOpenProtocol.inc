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

c::Result verifyOpenProtocol(const c::Program& p, const Commands& actual, uint64_t limit, Commands* residual,
                            ProtocolResiduals* protocolResidual, ProviderCompletions* completions,
                            BoundaryObservations* boundaries)
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
    // Correlate guarded entry endpoints with the same loop decision. Split
    // at their common owner, not independently at each SET/WAIT. First visits
    // are peeled below so an acquired prefix cannot consume a token twice or
    // erase a new write on a subsequent visit.
    std::vector<unsigned> parent(p.nodes.size(), NoCut);
    for (unsigned id = 0; id < p.nodes.size(); ++id)
        for (unsigned child : p.nodes[id].children) parent[child] = id;
    std::map<unsigned, unsigned> guardedOwners;
    uint64_t guardCommandCount = 0;
    auto commonOwner = [&](unsigned a, unsigned b) {
        std::set<unsigned> ancestors;
        for (unsigned at = a; at != NoCut; at = parent[at]) {
            if (!charge(1)) return NoCut;
            ancestors.insert(at);
        }
        for (unsigned at = b; at != NoCut; at = parent[at]) {
            if (!charge(1)) return NoCut;
            if (ancestors.count(at)) return at;
        }
        return NoCut;
    };
    for (unsigned site = 0; site < actual.size(); ++site)
        for (const auto& m : actual[site]) {
            if (!charge(1)) return result;
            ++guardCommandCount;
            if (m.participation == Mechanism::Every) continue;
            if ((m.participation != Mechanism::First && m.participation != Mechanism::NonEmpty) ||
                m.loop >= p.nodes.size() || p.nodes[m.loop].kind != Node::For ||
                p.nodes[m.loop].entryGuardStart == NoCut || m.word != NoCut ||
                (m.kind != Mechanism::Publish && m.kind != Mechanism::Acquire)) {
                result.reason = "unsupported guarded open provider";
                return result;
            }
            if (m.participation == Mechanism::First) {
                unsigned owner = commonOwner(site, m.loop);
                if (!result.reason.empty()) return result;
                if (owner != m.loop) {
                    result.reason = "first acquisition is outside its owning loop";
                    return result;
                }
            }
            auto inserted = guardedOwners.emplace(m.loop, m.loop);
            inserted.first->second = commonOwner(inserted.first->second, site);
            if (inserted.first->second == NoCut) return result;
        }
    if (!guardedOwners.empty()) {
        // Re-extract the original first consumer and its storage witness.
        // Correct token flow alone must not admit an earlier unrelated cut.
        if (!charge(DemandAnalysis::MaxIncomingEntryWork + DemandAnalysis::MaxChoiceIncomingWork +
                    uint64_t(p.nodes.size()) * (16 + uint64_t(p.cells) * LaneCount))) return result;
        DemandAnalysis guardedAnalysis;
        if (!guardedAnalysis.build(p, result.reason)) return result;
        for (const auto& [loop, owner] : guardedOwners) {
            (void)owner;
            if (!charge(uint64_t(p.nodes.size()) * (1 + guardCommandCount))) return result;
            std::vector<Mechanism> word;
            for (unsigned child : p.nodes[loop].children)
                if (!projectEntryWord(p, actual, child, loop, word, result.reason)) return result;
        }
        for (unsigned site = 0; site < actual.size(); ++site)
            for (const auto& m : actual[site]) {
                if (!charge(1)) return result;
                if (m.participation != Mechanism::First) continue;
                if (!charge(1 + guardedAnalysis.entries[m.loop].size() * (2 + MaxAlternatives))) return result;
                if (m.kind != Mechanism::Acquire ||
                    (!guardedAnalysis.entryDemand(m.loop, site, m.first, m.second) &&
                     !guardedAnalysis.lateEntryDemand(m.loop, site, m.first, m.second))) {
                    result.reason = "guarded acquisition lost its original first-consumer witness";
                    return result;
                }
            }
    }
    if (p.terminalRetirementCut != NoCut) {
        const auto& root = p.nodes.back();
        unsigned cut = p.terminalRetirementCut;
        if (cut >= p.nodes.size() - 1 || root.kind != Node::Sequence || root.children.empty() ||
            root.children.back() != cut || p.nodes[cut].kind != Node::Sequence ||
            !p.nodes[cut].children.empty() ||
            std::any_of(effects[cut].begin(), effects[cut].end(),
                        [](auto effect) { return effect.readers || effect.writers; })) {
            result.reason = "terminal retirement requires an empty final root cut";
            return result;
        }
    }
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
                (m.participation == Mechanism::Every && m.loop != NoCut) || m.word != NoCut || m.first >= LaneCount ||
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
    auto endpoint = [&](OpenProtocolState& state, const Mechanism& m, bool validate, unsigned site, unsigned position) {
        unsigned k = keys.at({m.first, m.second, m.forwardKey});
        if (m.kind == Mechanism::Publish) {
            if (validate && protocolResidual && state.occupancy[k] == 1 &&
                !(state.known[k] & (1u << m.first)) && (state.known[k] & (1u << m.second))) {
                // The prior generation has been consumed, but its producer
                // still needs the consumption receipt before rearming. Propose
                // an actual return at this precise command boundary. Its own
                // rearm still has to be proved on the next complete replay.
                if (!charge(1 + protocolResidual->size() + width +
                            LaneCount * LaneCount * p.target.compilerKeys.size() *
                                (1 + p.target.compilerKeys.size() + p.target.reservations.size()))) return false;
                std::vector<Mechanism> acknowledgment;
                if (auto receiptKey = key(p, m.second, m.first)) {
                    acknowledgment.push_back(event(Mechanism::Publish, m.second, m.first, *receiptKey));
                    acknowledgment.push_back(event(Mechanism::Acquire, m.second, m.first, *receiptKey));
                    state.acquire(m.second, m.first);
                } else if (!acquire(p, m.second, m.first, state, acknowledgment)) {
                    result.reason = "target cannot realize residual consumption acknowledgment";
                    return false;
                }
                auto& commands = (*protocolResidual)[{site, position}];
                for (const auto& command : acknowledgment) {
                    if (!charge(1 + commands.size())) return false;
                    if (std::find(commands.begin(), commands.end(), command) == commands.end())
                        commands.push_back(command);
                    if (command.kind == Mechanism::Rendezvous)
                        for (auto& known : state.known) {
                            if (known & (1u << command.first)) known |= 1u << command.second;
                            if (known & (1u << command.second)) known |= 1u << command.first;
                        }
                }
                // This is only proposal state. The next complete replay
                // reconstructs the inserted packet's occupancy and snapshots.
            }
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
            Effects before;
            if (validate && completions) {
                if (!charge(2 * uint64_t(p.cells) + completions->size() + 1)) return false;
                result.cellVisits += 2 * uint64_t(p.cells);
                before = state.pending[m.second];
            }
            state.acquirePrefix(m);
            if (validate && completions) {
                auto identity = std::make_tuple(m.first, m.second, m.forwardKey);
                if (!completions->count(identity)) {
                    if (!charge(p.cells)) return false;
                    result.cellVisits += p.cells;
                }
                auto& supplied = completions->try_emplace(identity, p.cells).first->second;
                for (unsigned cell = 0; cell < p.cells; ++cell) {
                    supplied[cell].readers |= before[cell].readers & ~state.pending[m.second][cell].readers;
                    supplied[cell].writers |= before[cell].writers & ~state.pending[m.second][cell].writers;
                }
            }
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
    std::map<unsigned, bool> nonempty, firstVisit;
    visit = [&](unsigned id, OpenProtocolState& state, bool validate, unsigned activeOwner,
                unsigned residue, int more) {
        for (auto [loop, owner] : guardedOwners) {
            if (!charge(1)) return false;
            if (owner != id || nonempty.count(loop)) continue;
            if (!charge(2 * width)) return false;
            auto empty = state;
            nonempty[loop] = true;
            bool ok = visit(id, state, validate, activeOwner, residue, more);
            if (ok && !p.nodes[loop].nonEmpty) {
                nonempty[loop] = false;
                ok = visit(id, empty, validate, activeOwner, residue, more);
                if (ok) state.join(empty);
            }
            nonempty.erase(loop);
            return ok;
        }
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
        for (unsigned position = 0; position < actual[id].size(); ++position) {
            const auto& m = actual[id][position];
            if (m.participation != Mechanism::Every) {
                auto decision = m.participation == Mechanism::First ? firstVisit.find(m.loop) : nonempty.find(m.loop);
                const auto& decisions = m.participation == Mechanism::First ? firstVisit : nonempty;
                if (decision == decisions.end()) {
                    result.reason = "guarded endpoint has no structural participation decision";
                    return false;
                }
                if (!decision->second) continue;
            }
            if (m.kind == Mechanism::Publish || m.kind == Mechanism::Acquire) {
                if (!endpoint(state, m, validate, id, position))
                    return false;
            } else if (m.kind == Mechanism::Rendezvous) {
                for (auto e :
                     {event(Mechanism::Publish, m.first, m.second, m.forwardKey),
                      event(Mechanism::Acquire, m.first, m.second, m.forwardKey),
                      event(Mechanism::Publish, m.second, m.first, m.reverseKey),
                      event(Mechanism::Acquire, m.second, m.first, m.reverseKey)})
                    if (!endpoint(state, e, validate, id, position))
                        return false;
            } else
                apply(p, state, m);
        }
        const auto& node = p.nodes[id];
        // Proposal transfers must also satisfy the authored notify during
        // invariant discovery. Pure emitted verification has no residual sink
        // and therefore cannot invent this prerequisite.
        if (residual && node.notificationPrerequisite) {
            if (!charge((2 + LaneCount) * width + uint64_t(p.cells) * (LaneCount + 2))) return false;
            std::vector<Mechanism> prerequisites;
            if (!realizeNotificationPrerequisite(p, id, state, prerequisites, result.reason,
                                                result.visibilityRequirements, result.cellVisits)) return false;
            auto& commands = (*residual)[id];
            for (const auto& command : prerequisites) {
                if (!charge(1 + commands.size())) return false;
                if (std::find(commands.begin(), commands.end(), command) == commands.end())
                    commands.push_back(command);
            }
        }
        if (node.kind == Node::Operation) {
            if (validate && residual) {
                auto& commands = (*residual)[id];
                std::vector<Mechanism> visibility;
                if (!realizeVisibility(p, node, state, visibility, result.reason, result.visibilityRequirements, &result, id))
                    return false;
                // A structural site may be visited for several residues and
                // for both continuing and final iterations. Its residual
                // recipe is emitted once, regardless of abstract visit count.
                for (const auto& command : visibility) {
                    if (!charge(1 + commands.size())) return false;
                    if (std::find(commands.begin(), commands.end(), command) == commands.end())
                        commands.push_back(command);
                }
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
                    for (const auto& command : generated) {
                        if (!charge(1 + commands.size())) return false;
                        if (std::find(commands.begin(), commands.end(), command) == commands.end())
                            commands.push_back(command);
                    }
                }
            }
            if (validate && !residual &&
                (!visibilityCovered(p, node, visibilityState, actual[id]) || operationDemands(p, node, state))) {
                result.reason = "uncovered open-protocol physical or visibility demand at node " + std::to_string(id);
                return false;
            }
            if (validate && boundaries) {
                if (!charge(1 + 3 * uint64_t(p.cells))) return false;
                auto inserted = boundaries->try_emplace(id);
                auto& observation = inserted.first->second;
                if (inserted.second) {
                    observation.leastPending = observation.mostPending = state.pending[node.lane];
                } else for (unsigned cell = 0; cell < p.cells; ++cell) {
                    const auto& pending = state.pending[node.lane][cell];
                    observation.leastPending[cell].readers &= pending.readers;
                    observation.leastPending[cell].writers &= pending.writers;
                    observation.mostPending[cell].readers |= pending.readers;
                    observation.mostPending[cell].writers |= pending.writers;
                }
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
            const bool guarded = guardedOwners.count(id);
            if (guarded && !nonempty.at(id)) return true;
            if (!charge(2 * width))
                return false;
            const auto entry = state;
            unsigned period = 0;
            if (node.kind == Node::For && (activeOwner == NoCut || guarded)) {
                if (node.nonEmpty || guarded)
                    period = 1;
                std::function<void(unsigned)> findPeriod = [&](unsigned child) {
                    const auto& nested = p.nodes[child];
                    if (!result.reason.empty() || !charge(1 + nested.children.size())) return;
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
                if (!result.reason.empty()) return false;
            }
            if (period && period <= 32 && node.periodicLower >= 0) {
                if (!charge(uint64_t(period) * 8 * width))
                    return false;
                std::vector<std::optional<OpenProtocolState>> starts(period);
                unsigned first = unsigned(node.periodicLower) % period;
                std::optional<OpenProtocolState> firstExit;
                if (guarded) {
                    if (!charge(3 * width)) return false;
                    firstVisit[id] = true;
                    auto continuing = entry;
                    firstExit = entry;
                    for (unsigned child : node.children)
                        if (!visit(child, continuing, validate, id, first, 1)) return false;
                    for (unsigned child : node.children)
                        if (!visit(child, *firstExit, validate, id, first, 0)) return false;
                    firstVisit[id] = false;
                    starts[(first + 1) % period] = std::move(continuing);
                } else starts[first] = entry;
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (unsigned current = 0; current < period; ++current) {
                        if (!starts[current])
                            continue;
                        if (!charge(6 * width)) return false;
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
                std::optional<OpenProtocolState> exit = std::move(firstExit);
                if (!node.nonEmpty && !guarded)
                    exit = entry; // original zero-trip successor
                for (unsigned current = 0; current < period; ++current) {
                    if (!starts[current])
                        continue;
                    if (!charge(6 * width)) return false;
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
                if (guarded) firstVisit.erase(id);
                return true;
            }
            if (guarded) {
                result.reason = "guarded open provider requires a checked loop ordinal";
                return false;
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
    // Repeated invocations share event generations. An explicit terminal ALL
    // orders only already-consumed tokens before the next invocation. It must
    // not conceal a missing WAIT, provide an in-body receipt, or publish GM.
    auto retire = [&](OpenProtocolState& state) {
        if (p.terminalRetirementCut == NoCut) return;
        if (std::any_of(state.occupancy.begin(), state.occupancy.end(), [](auto bits) { return bits != 1; }))
            return;
        state.barrierAll();
        std::fill(state.known.begin(), state.known.end(), (1u << LaneCount) - 1);
    };
    for (;;) {
        if (!charge(3 * width))
            return result;
        auto next = invariant;
        if (!visit(p.nodes.size() - 1, next, false, NoCut, 0, -1))
            return result;
        if (p.terminalRetirementCut != NoCut && !charge(width))
            return result;
        retire(next);
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
