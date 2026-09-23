// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Persistent source signatures share symbolic expressions, not payload-order
// closures or causal states. A delta follows original successors until its
// hardware-sized interface is unchanged. Exact equality is sufficient only;
// unqualified cyclic interfaces remain Unknown. No signature grants credit.
#include "SelectedInternal.h"
#include <algorithm>

namespace mlir::pto::oahs::selected {
PublicationSupport::PublicationSupport(const Program& p, const Control& c, const CausalFrontier& f)
    : program(p), control(c), frontier(f), outgoing(c.graph.sites.size())
{
}
Id PublicationSupport::expression(Kind kind, Id identity, std::vector<Id> operands)
{
    const bool unknown = std::find(operands.begin(), operands.end(), NoAnalysisId) != operands.end();
    if (unknown) { return NoAnalysisId; }
    const auto next = expressions.size() + 1;
    const auto entry = expressions.emplace(Node{kind, identity, std::move(operands)}, next);
    nodes += entry.second;
    return entry.first->second;
}
Id PublicationSupport::unite(Id a, Id b)
{
    if (a == NoAnalysisId || b == NoAnalysisId) { return NoAnalysisId; }
    if (!a || a == b) { return b; }
    if (!b) { return a; }
    return expression(Kind::Union, 0, {std::min(a, b), std::max(a, b)});
}
std::vector<Id> PublicationSupport::incoming(Cut site, const Probe& probe)
{
    const auto width = 2 * PipeCount + frontier.keys().size();
    std::vector<Id> state(width, 0);
    bool first = true;
    for (auto predecessor : control.predecessors[site]) {
        if (!control.reachable[predecessor]) { continue; }
        const auto changed = probe.states.find(predecessor);
        const auto& prior = changed == probe.states.end() ? outgoing[predecessor] : changed->second;
        const bool knownWidth = prior.size() == width;
        if (!knownWidth) { return std::vector<Id>(width, NoAnalysisId); }
        if (first) { state = prior; first = false; continue; }
        for (Id port = 0; port < width; ++port) {
            if (state[port] != prior[port]) {
                // Preserve the original join and predecessor order. This is a
                // conditional structural expression, never conjunctive credit.
                state[port] = expression(Kind::Choice, site, {port, state[port], prior[port]});
            }
        }
    }
    return state;
}
void PublicationSupport::command(std::vector<Id>& state, const SelectedEndpoint& endpoint, Cut site, Probe& probe)
{
    ++commands;
    const auto& c = endpoint.command;
    const auto pipe = unsigned(c.kind == Command::Acquire ? c.observer : c.source);
    const auto prefix = PipeCount + pipe;
    if (c.kind == Command::Publish) {
        const auto root = unite(state[pipe], state[prefix]);
        probe.roots[{endpoint.id, site}] = root;
        const auto found = byOccurrence.find({endpoint.id, site});
        if (found != byOccurrence.end()) {
            ++checks;
            const auto& old = contracts[found->second];
            probe.preserved &= root != NoAnalysisId && root == old.admittedRoot;
        }
        const auto key = keyIndex(frontier, c);
        if (key != NoAnalysisId) { state[2 * PipeCount + key] = expression(Kind::Publication, endpoint.id, {root}); }
        state[prefix] = root;
    } else if (c.kind == Command::Acquire) {
        const auto key = keyIndex(frontier, c);
        const auto source = key == NoAnalysisId ? NoAnalysisId : state[2 * PipeCount + key];
        state[pipe] = source == 0 ? NoAnalysisId : expression(Kind::Receipt, endpoint.id, {unite(state[pipe], source)});
        state[prefix] = unite(state[prefix], state[pipe]);
        if (key != NoAnalysisId) { state[2 * PipeCount + key] = 0; }
    } else if (c.kind == Command::Barrier) {
        state[pipe] = unite(state[pipe], state[prefix]);
        state[prefix] = state[pipe];
    } else {
        // Retirement and unsupported command interfaces cannot certify a
        // source exported across them. Causal legality is checked elsewhere.
        std::fill(state.begin(), state.end(), NoAnalysisId);
    }
}
void PublicationSupport::payload(std::vector<Id>& state, Cut site)
{
    const auto operation = control.graph.operations[site];
    if (operation == NoAnalysisId) { return; }
    const auto pipe = unsigned(program.operations[operation].pipe);
    const auto issue = expression(Kind::Issue, site, {state[pipe]});
    const auto complete = expression(Kind::Completion, site, {issue});
    state[PipeCount + pipe] = unite(state[PipeCount + pipe], complete);
    state[pipe] = program.target.synchronous[pipe] ? complete : issue;
}
PublicationSupport::Probe PublicationSupport::inspect(
    const Ledger& ledger, const PacketView* view, const std::vector<Cut>& changed)
{
    Probe probe;
    probe.version = view ? view->version() : ledger.version();
    if (view && (!initialized || version != ledger.version())) {
        probe.preserved = false;
        return probe;
    }
    // Component order is topological. A dirty batch visits a site once; unchanged
    // interfaces stop propagation, including changes confined to other engines.
    std::set<std::pair<Id, Cut>> pending;
    const auto enqueue = [&](Cut site) {
        const bool reachable = site < control.reachable.size() && control.reachable[site];
        if (reachable) {
            pending.emplace(control.component[site], site);
        }
    };
    if (!initialized) {
        for (Cut site = 0; site < control.reachable.size(); ++site) { enqueue(site); }
    } else {
        for (auto word : changed) {
            for (auto site : control.wordOccurrences[control.canonicalCut[word]]) { enqueue(site); }
        }
    }
    while (!pending.empty()) {
        const auto site = pending.begin()->second;
        pending.erase(pending.begin());
        ++sites;
        const bool cyclic = control.components[control.component[site]].cyclic;
        // An unchanged Unknown interface cannot establish preservation of
        // affected recurring publications. Do not unroll the SCC to hide this.
        if (cyclic) { probe.preserved = false; }
        auto state = cyclic ? std::vector<Id>(2 * PipeCount + frontier.keys().size(), NoAnalysisId)
                            : incoming(site, probe);
        const auto& word = view ? view->word(site) : ledger.word(site);
        for (auto id : word) {
            const auto& endpoint = view ? view->endpoint(id) : ledger.endpoint(id);
            command(state, endpoint, site, probe);
        }
        payload(state, site);
        const bool different = state != outgoing[site];
        probe.states[site] = std::move(state);
        if (different) {
            for (auto next : control.graph.sites[site].successors) {
                if (control.component[next] != control.component[site]) { enqueue(next); }
            }
        }
    }
    return probe;
}
void PublicationSupport::accept(const Ledger& ledger, Probe probe)
{
    for (auto& [site, state] : probe.states) { outgoing[site] = std::move(state); }
    for (const auto& [key, root] : probe.roots) {
        const auto found = byOccurrence.emplace(key, contracts.size());
        if (found.second) {
            contracts.push_back({key.first, key.second, ledger.version(), ledger.version(),
                                 root, root != NoAnalysisId});
        } else {
            auto& contract = contracts[found.first->second];
            contract.checkedVersion = ledger.version();
            contract.preserved = root != NoAnalysisId && root == contract.admittedRoot;
        }
    }
    version = ledger.version();
    initialized = true;
}
void PublicationSupport::refresh(const Ledger& ledger)
{
    if (!initialized || version != ledger.version()) { accept(ledger, inspect(ledger, nullptr, ledger.changes())); }
}
} // namespace mlir::pto::oahs::selected
