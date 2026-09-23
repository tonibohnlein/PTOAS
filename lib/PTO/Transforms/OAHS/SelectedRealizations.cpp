// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <deque>

namespace mlir::pto::oahs::selected {
const RecurringCertificate& Constructor::recurringCertificate(Id id, bool normal)
{
    ++result.work.recurringInterfaceQueries;
    auto found = recurringCertificates.find({id, normal});
    if (found != recurringCertificates.end()) { return found->second; }
    auto& proof = recurringCertificates[{id, normal}];
    const auto& family = recurringFrontiers.families[id];
    proof.reason = "finite role interface is not a paired exchange";
    // A reusable bounded lemma, not a dense virtual-key machine for a closure.
    // More roles retain their independent facts and use the transitional path.
    const bool paired = family.roles.size() == 2;
    if (!paired) { return proof; }
    const auto& first = recurringFrontiers.roles[family.roles[0]];
    const auto& second = recurringFrontiers.roles[family.roles[1]];
    const bool reverse = first.source == second.observer && first.observer == second.source;
    if (!reverse) { return proof; }
    if (!projectionIndexed) {
        for (Id op = 0; op < program.operations.size(); ++op) {
            const auto& accesses = program.operations[op].accesses;
            for (Id index = 0; index < accesses.size(); ++index) {
                projectionAccesses[accesses[index].cell].push_back({op, index});
                ++result.work.recurringInterfaceAccesses;
            }
        }
        projectionIndexed = true;
    }
    // Share original control. Only relevant access incidences enter this small
    // model; unrelated payload is an identity transfer, never invented support.
    Program slice;
    slice.target = program.target;
    slice.nativeAccumulatorClasses = program.nativeAccumulatorClasses;
    for (auto& row : slice.target.keys) {
        for (auto& keys : row) { keys.clear(); }
    }
    std::map<unsigned, unsigned> cells;
    std::map<Id, std::vector<Access>> operations;
    for (auto cell : family.cells) {
        cells.emplace(cell, slice.cells.size());
        slice.cells.push_back(program.cells[cell]);
        const auto at = projectionAccesses.find(cell);
        if (at == projectionAccesses.end()) { continue; }
        for (const auto& incidence : at->second) {
            auto access = program.operations[incidence.first].accesses[incidence.second];
            access.cell = cells.at(cell);
            operations[incidence.first].push_back(access);
            ++result.work.recurringInterfaceAccesses;
        }
    }
    std::map<Id, Id> localOperations;
    for (auto& item : operations) {
        const auto original = item.first;
        const auto& source = program.operations[original];
        Operation op;
        op.pipe = source.pipe;
        op.original = source.original;
        op.phase = source.phase;
        op.complete = source.complete;
        op.nativeMmadAccumulate = source.nativeMmadAccumulate;
        op.resources = source.resources;
        op.visibility = source.visibility;
        op.authoredEvents = source.authoredEvents;
        op.internalTransfers = source.internalTransfers;
        op.finalBlock = source.finalBlock;
        op.accesses = std::move(item.second);
        const auto local = slice.operations.size();
        localOperations.emplace(original, local);
        slice.operations.push_back(std::move(op));
        Region region;
        region.kind = Region::Operation;
        region.operation = local;
        slice.body.children.push_back(std::move(region));
    }
    std::map<Cut, std::vector<Command>> words;
    for (auto role : family.roles) {
        const auto& request = recurringFrontiers.roles[role];
        slice.target.keys[unsigned(request.source)][unsigned(request.observer)] = {0};
        for (auto cut : request.publications) {
            words[cut].push_back({Command::Publish, request.source, request.observer, 0});
            proof.words[cut].push_back({role, Command::Publish});
        }
        for (auto cut : request.acquisitions) {
            words[cut].push_back({Command::Acquire, request.source, request.observer, 0});
            proof.words[cut].push_back({role, Command::Acquire});
        }
    }
    if (normal) {
        for (auto& word : words) {
            std::stable_sort(word.second.begin(), word.second.end(), [](const auto& a, const auto& b) {
                return a.kind == Command::Publish && b.kind != Command::Publish;
            });
        }
        for (auto& word : proof.words) {
            std::stable_sort(word.second.begin(), word.second.end(), [](const auto& a, const auto& b) {
                return a.second == Command::Publish && b.second != Command::Publish;
            });
        }
    }
    CausalFrontier projected(std::move(slice));
    if (!projected.complete()) { proof.reason = projected.reason(); return proof; }
    // The worklist follows ALL original guarded paths, including bypass,
    // re-entry and exits. It neither samples trips nor multiplies predicates.
    std::vector<FrontierState> incoming(control.graph.sites.size()), before(incoming.size());
    std::vector<bool> queued(incoming.size());
    std::deque<Cut> todo{control.graph.entry};
    incoming[control.graph.entry] = projected.initial();
    queued[control.graph.entry] = true;
    std::map<std::pair<Cut, Id>, Id> endpointIdentities;
    while (!todo.empty()) {
        const auto site = todo.front(); todo.pop_front(); queued[site] = false;
        ++result.work.recurringInterfaceSites;
        auto state = incoming[site];
        const auto word = words.find(control.canonicalCut[site]);
        if (word != words.end()) {
            Id offset = 0;
            for (const auto& command : word->second) {
                // A synthetic legal word names opaque identities only. Original
                // guarded occurrence/order comes from the shared graph above.
                const auto identity = endpointIdentities.emplace(
                    std::make_pair(site, offset++), endpointIdentities.size()).first->second;
                const auto next = projected.command(state, command, {0, identity});
                if (!next.applied) { proof.reason = next.reason; return proof; }
                state = next.state;
            }
        }
        before[site] = state;
        const auto op = localOperations.find(control.graph.operations[site]);
        if (op != localOperations.end()) {
            const auto next = projected.pendingIssue(state, op->second);
            if (!next.applied) { proof.reason = next.reason; return proof; }
            state = next.state;
        }
        for (auto next : control.graph.sites[site].successors) {
            const auto joined = projected.join(incoming[next], state);
            if (!joined.applied) { proof.reason = joined.reason; return proof; }
            if (joined.state == incoming[next]) { continue; }
            incoming[next] = joined.state;
            if (!queued[next]) { todo.push_back(next); queued[next] = true; }
        }
    }
    const auto exit = projected.exit(before[control.graph.exit]);
    if (!exit.applied) { proof.reason = exit.reason; return proof; }
    for (Cut site = 0; site < before.size(); ++site) {
        const auto operation = control.graph.operations[site];
        const bool relevant = localOperations.count(operation) && before[site].reachable();
        if (!relevant) { continue; }
        const auto observer = unsigned(program.operations[operation].pipe);
        // Each retained row quantifies over ALL original incidences of that
        // access class, including older/outside accesses and partial writes.
        for (const auto& cell : cells) {
            for (unsigned source = 0; source < PipeCount; ++source) {
                for (unsigned write = 0; write != 2; ++write) {
                    const auto local = (Id(cell.second) * PipeCount + source) * 2 + write;
                    const auto* history = before[site].facts()->history.find(local);
                    if (history && frontierContains(*history, observer)) {
                        proof.guaranteed[site].insert((Id(cell.first) * PipeCount + source) * 2 + write);
                    }
                }
            }
        }
    }
    proof.complete = true;
    proof.reason.clear();
    return proof;
}

bool Constructor::qualifyRecurringInterface(
    const std::vector<Id>& families, RecurringPacket& proposal, const ProducerSupportScope& scope)
{
    const auto view = ledger.packetView(proposal.packet.prepared);
    if (!view || !proposal.packet.restoredWaits.empty()) { return false; }
    // Retirement is not an ordinary monotone insertion boundary. The core
    // permits it only at invocation exit; no newly embedded endpoint may follow.
    bool retired = false;
    for (auto endpoint : view->word(control.graph.exit)) {
        ++result.work.recurringInterfaceEmbedding;
        if (retired) { return false; }
        retired = view->endpoint(endpoint).command.kind == Command::BarrierAll;
    }
    std::map<Id, EventIdentity> bindings;
    using RoleKey = std::tuple<Pipe, Pipe, std::vector<Cut>, std::vector<Cut>>;
    std::map<RoleKey, Id> roles;
    std::set<Id> indexedRoles;
    for (auto family : families) {
        for (auto id : recurringFrontiers.families[family].roles) {
            if (!indexedRoles.insert(id).second) { continue; }
            const auto& role = recurringFrontiers.roles[id];
            roles.emplace(RoleKey{role.source, role.observer, role.publications, role.acquisitions}, id);
            const auto active = activeRoles.find(id);
            if (active == activeRoles.end()) { continue; }
            const auto& channel = result.channels[active->second];
            bindings.emplace(id, EventIdentity{channel.source, channel.observer, channel.key, false});
        }
    }
    for (Id index = 0; index < proposal.requests.size(); ++index) {
        const auto& request = proposal.requests[index];
        const auto role = roles.find({request.source, request.observer, request.publications, request.acquisitions});
        if (role == roles.end()) { return false; }
        bindings.emplace(role->second, frontier.keys()[proposal.keys[index]]);
    }
    std::set<Id> checkedUses;
    std::set<std::vector<Id>> checkedWords;
    for (auto family : families) {
        const auto& proof = recurringCertificate(family, proposal.normalWords);
        if (!proof.complete) { return false; }
        for (auto role : recurringFrontiers.families[family].roles) {
            if (!checkedUses.insert(role).second) { continue; }
            const auto binding = bindings.find(role);
            if (binding == bindings.end()) { return false; }
            for (auto endpoint : ledger.eventUses(binding->second)) {
                ++result.work.recurringInterfaceEmbedding;
                if (!ledger.active(endpoint)) { continue; }
                const auto& existing = ledger.endpoint(endpoint);
                const auto word = proof.words.find(existing.cut);
                if (word == proof.words.end()) { return false; }
                const auto expected = std::make_pair(role, existing.command.kind);
                const bool represented =
                    std::find(word->second.begin(), word->second.end(), expected) != word->second.end();
                if (!represented) {
                    return false;
                }
            }
        }
        const bool newPair = checkedWords.insert(recurringFrontiers.families[family].roles).second;
        if (newPair) {
            for (const auto& word : proof.words) {
                std::vector<std::pair<Id, Command::Kind>> actual;
                for (auto endpoint : view->word(word.first)) {
                    const auto& command = view->endpoint(endpoint).command;
                    ++result.work.recurringInterfaceEmbedding;
                    if (command.kind != Command::Publish && command.kind != Command::Acquire) { continue; }
                    for (auto role : recurringFrontiers.families[family].roles) {
                        const auto at = bindings.find(role);
                        if (at == bindings.end()) { return false; }
                        const auto& key = at->second;
                        const bool sameKey = key.source == command.source && key.observer == command.observer &&
                            key.key == command.key;
                        if (sameKey) {
                            actual.push_back({role, command.kind});
                        }
                    }
                }
                if (actual != word.second) { return false; }
            }
        }
        for (const auto& at : proof.guaranteed) {
            proposal.guaranteed[at.first].insert(at.second.begin(), at.second.end());
        }
    }
    // Keep the repair-relocation guard. This certificate can discharge its
    // actual obligations; a discovery match alone cannot grant that coverage.
    for (auto site : scope.consumers) {
        const auto operation = control.graph.operations[site];
        const auto pipe = unsigned(program.operations[operation].pipe);
        const auto missing = frontier.inspect(cache.cuts[site].before.causal, operation).residuals;
        for (const auto& requirement : missing) {
            const auto access = accessClass(requirement);
            const bool unsupported = scope.classes[pipe].count(access) && !proposal.guaranteed[site].count(access);
            if (unsupported) { return false; }
        }
    }
    proposal.packet.qualified = true;
    proposal.localCertificate = true;
    return true;
}
} // namespace mlir::pto::oahs::selected
