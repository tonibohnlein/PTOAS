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
namespace {
bool same(const State& a, const State& b)
{
    return a.causal == b.causal && a.latest == b.latest && a.consumptions == b.consumptions;
}
Id keyIndex(const CausalFrontier& frontier, const Command& command)
{
    const auto& keys = frontier.keys();
    for (Id key = 0; key < keys.size(); ++key) {
        const auto& identity = keys[key];
        if (identity.source == command.source && identity.observer == command.observer &&
            identity.key == command.key) {
            return key;
        }
    }
    return NoAnalysisId;
}
bool refused(Replay& replay, Cut site, const FrontierStep& step)
{
    replay.success = false;
    replay.failureCut = site;
    replay.reason = step.reason;
    return false;
}
} // namespace

State Constructor::initial() const
{
    State out;
    out.causal = frontier.initial();
    out.latest.assign(program.cells.size() * PipeCount * 2, NoAnalysisId);
    out.consumptions.resize(frontier.keys().size());
    return out;
}
bool Constructor::join(State& target, const State& input)
{
    if (!input.causal.reachable()) {
        return true;
    }
    if (!target.causal.reachable()) {
        target = input;
        return true;
    }
    auto merged = frontier.join(target.causal, input.causal);
    if (!merged.applied) {
        return false;
    }
    for (Id i = 0; i < target.latest.size(); ++i) {
        const auto& a = target.causal.facts()->history[i];
        const auto& b = input.causal.facts()->history[i];
        if (!a) {
            target.latest[i] = input.latest[i];
        } else if (b && target.latest[i] != input.latest[i]) {
            target.latest[i] = NoAnalysisId;
        }
    }
    for (Id i = 0; i < target.consumptions.size(); ++i) {
        target.consumptions[i] = unionIds(target.consumptions[i], input.consumptions[i]);
    }
    target.causal = std::move(merged.state);
    return true;
}
bool Constructor::word(State& state, Cut site, Replay& replay)
{
    Id offset = 0;
    for (auto id : ledger.word(site)) {
        const auto& endpoint = ledger.endpoint(id);
        auto step = frontier.command(state.causal, endpoint.command, {site, offset++});
        if (!step.applied) {
            return refused(replay, site, step);
        }
        state.causal = std::move(step.state);
        if (endpoint.command.kind == Command::Acquire) {
            const auto key = keyIndex(frontier, endpoint.command);
            if (key == NoAnalysisId) {
                replay.success = false;
                replay.reason = "selected acquisition lost its eligible key";
                replay.failureCut = site;
                return false;
            }
            state.consumptions[key] = {id};
        }
        if (!join(replay.afterEndpoint[id], state)) {
            replay.success = false;
            replay.reason = "incompatible selected endpoint snapshots";
            replay.failureCut = site;
            return false;
        }
    }
    return true;
}
bool Constructor::payload(State& state, Cut site, Replay& replay)
{
    const auto operation = control.graph.operations[site];
    if (operation == NoAnalysisId) {
        return true;
    }
    const auto step = frontier.issue(state.causal, operation);
    if (!step.applied) {
        return refused(replay, site, step);
    }
    state.causal = step.state;
    const auto& op = program.operations[operation];
    for (const auto& access : op.accesses) {
        const auto index = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
        if (access.read) {
            state.latest[index] = site;
        }
        if (access.write) {
            state.latest[index + 1] = site;
        }
    }
    return true;
}
bool Constructor::fixedComponent(
    Id index, const std::vector<State>& boundary, Replay& replay, std::vector<State>& outgoing)
{
    const auto& block = control.components[index];
    if (!block.cyclic) {
        const auto site = block.sites.front();
        auto state = boundary[site];
        replay.cuts[site].incoming = state;
        if (!state.causal.reachable()) {
            return true;
        }
        ++replay.evaluations;
        if (!word(state, site, replay)) {
            return false;
        }
        replay.cuts[site].before = state;
        if (!payload(state, site, replay)) {
            return false;
        }
        replay.cuts[site].outgoing = state;
        for (auto next : control.graph.sites[site].successors) {
            if (!join(outgoing[next], state)) {
                return false;
            }
        }
        return true;
    }
    std::vector<State> incoming(control.graph.sites.size());
    std::deque<Id> queue;
    std::vector<bool> queued(incoming.size());
    for (auto site : block.sites) {
        incoming[site] = boundary[site];
        if (incoming[site].causal.reachable()) {
            queue.push_back(site);
            queued[site] = true;
        }
    }
    while (!queue.empty()) {
        const auto site = queue.front();
        queue.pop_front();
        queued[site] = false;
        ++replay.evaluations;
        auto state = incoming[site];
        replay.cuts[site].incoming = state;
        if (!word(state, site, replay)) {
            return false;
        }
        replay.cuts[site].before = state;
        if (!payload(state, site, replay)) {
            return false;
        }
        replay.cuts[site].outgoing = state;
        for (auto next : control.graph.sites[site].successors) {
            if (control.component[next] != index) {
                continue;
            }
            const auto previous = incoming[next];
            if (!join(incoming[next], state)) {
                return false;
            }
            if (!same(previous, incoming[next]) && !queued[next]) {
                queue.push_back(next);
                queued[next] = true;
            }
        }
    }
    // Export only the stabilized selected-body interfaces, never a fresh state.
    for (auto site : block.sites) {
        for (auto next : control.graph.sites[site].successors) {
            if (control.component[next] != index && !join(outgoing[next], replay.cuts[site].outgoing)) {
                return false;
            }
        }
    }
    return true;
}
bool Constructor::partialComponent(Id index, const std::vector<State>& boundary, Replay& replay)
{
    const auto& block = control.components[index];
    auto incoming = boundary;
    if (block.cyclic) {
        std::vector<Id> operations;
        for (auto site : block.sites) {
            if (control.graph.operations[site] != NoAnalysisId) {
                operations.push_back(control.graph.operations[site]);
            }
        }
        for (auto entry : block.entries) {
            if (!incoming[entry].causal.reachable()) {
                continue;
            }
            auto seed = frontier.assumePreviousAccesses(incoming[entry].causal, operations);
            if (!seed.applied) {
                return refused(replay, entry, seed);
            }
            incoming[entry].causal = std::move(seed.state);
            for (auto operation : operations) {
                const auto& op = program.operations[operation];
                for (const auto& access : op.accesses) {
                    const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                    if (access.read) {
                        incoming[entry].latest[base] = NoAnalysisId;
                    }
                    if (access.write) {
                        incoming[entry].latest[base + 1] = NoAnalysisId;
                    }
                }
            }
        }
    }
    for (Id offset = 0; offset <= activeOffset; ++offset) {
        const auto site = block.order[offset];
        auto state = incoming[site];
        if (state.causal.reachable() && !control.headerAccesses[site].empty()) {
            auto seed = frontier.assumePreviousAccesses(state.causal, control.headerAccesses[site]);
            if (!seed.applied) {
                return refused(replay, site, seed);
            }
            state.causal = std::move(seed.state);
            for (auto operation : control.headerAccesses[site]) {
                const auto& op = program.operations[operation];
                for (const auto& access : op.accesses) {
                    const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                    if (access.read) {
                        state.latest[base] = NoAnalysisId;
                    }
                    if (access.write) {
                        state.latest[base + 1] = NoAnalysisId;
                    }
                }
            }
        }
        replay.cuts[site].incoming = state;
        if (!state.causal.reachable()) {
            continue;
        }
        ++replay.evaluations;
        if (!word(state, site, replay)) {
            return false;
        }
        replay.cuts[site].before = state;
        if (site != current && finalized[site] && !payload(state, site, replay)) {
            return false;
        }
        replay.cuts[site].outgoing = state;
        if (site == current) {
            continue;
        }
        for (auto next : control.constructionEdges[site]) {
            if (control.component[next] == index) {
                if (!join(incoming[next], state)) {
                    return false;
                }
            }
        }
    }
    return true;
}
bool Constructor::replay()
{
    Replay fresh;
    fresh.version = ledger.version();
    fresh.cuts.resize(control.graph.sites.size());
    std::vector<State> boundary(control.graph.sites.size());
    // The component order is a topological order of the condensation, so a
    // component before the earliest changed word has unchanged equations and
    // unchanged inputs: its previous least solution is the same and is reused
    // verbatim. No changed or cyclic region is seeded with old facts, and a
    // cyclic component is reused only from its actual fixed point, never from
    // the hypothesis-seeded construction traversal of the active component.
    Id resume = 0;
    if (cache.success && cache.cuts.size() == control.graph.sites.size()) {
        resume = activeComponent;
        // Ledger changes name canonical words, not every original occurrence.
        // Numeric canonical-cut order need not agree with component order.
        for (auto cut : ledger.changes()) {
            for (Cut site = 0; site < control.component.size(); ++site) {
                if (control.component[site] != NoAnalysisId && canonicalCommandCut(program, site) == cut) {
                    resume = std::min(resume, control.component[site]);
                }
            }
        }
        for (Id index = 0; index < resume; ++index) {
            if (control.components[index].cyclic && index >= cache.fixedComponents) {
                resume = index;
                break;
            }
        }
        // afterEndpoint joins all occurrences of an endpoint. Do not reuse an
        // aggregate containing contributions from the region being recomputed.
        // Keep each shared nonempty word wholly on one side of the boundary.
        bool widened;
        do {
            widened = false;
            std::map<Cut, std::pair<Id, Id>> spans;
            for (Cut site = 0; site < control.component.size(); ++site) {
                const auto component = control.component[site];
                if (component == NoAnalysisId || ledger.word(site).empty()) {
                    continue;
                }
                const auto word = canonicalCommandCut(program, site);
                auto found = spans.emplace(word, std::make_pair(component, component)).first;
                found->second.first = std::min(found->second.first, component);
                found->second.second = std::max(found->second.second, component);
            }
            for (const auto& span : spans) {
                if (span.second.first < resume && span.second.second >= resume) {
                    resume = span.second.first;
                    widened = true;
                }
            }
        } while (widened);
    }
    if (control.component[control.graph.entry] >= resume) {
        boundary[control.graph.entry] = initial();
    }
    for (Id index = 0; fresh.success && index < resume; ++index) {
        for (auto site : control.components[index].sites) {
            fresh.cuts[site] = cache.cuts[site];
            for (auto id : ledger.word(site)) {
                const auto found = cache.afterEndpoint.find(id);
                if (found != cache.afterEndpoint.end()) {
                    fresh.afterEndpoint.emplace(id, found->second);
                }
            }
            for (auto next : control.graph.sites[site].successors) {
                if (!control.reachable[next] || control.component[next] < resume) {
                    continue;
                }
                if (!join(boundary[next], fresh.cuts[site].outgoing)) {
                    fresh.success = false;
                    fresh.reason = "incompatible reused predecessor state";
                    fresh.failureCut = next;
                    break;
                }
            }
        }
    }
    fresh.reusedComponents = resume;
    for (Id index = resume; fresh.success && index < activeComponent; ++index) {
        if (!fixedComponent(index, boundary, fresh, boundary)) {
            fresh.success = false;
            break;
        }
    }
    if (fresh.success && activeComponent < control.components.size()) {
        fresh.success = partialComponent(activeComponent, boundary, fresh);
    }
    fresh.fixedComponents = activeComponent;
    result.work.replaySiteEvaluations += fresh.evaluations;
    cache = std::move(fresh);
    if (!cache.success) {
        return fail(SelectedFailure::SelectedUpdate, cache.reason, cache.failureCut);
    }
    refreshSources();
    return true;
}
void Constructor::refreshSources()
{
    for (auto& source : result.sources) {
        if (source.cut < cache.cuts.size()) {
            source.snapshot = cache.cuts[source.cut].before.causal;
            source.version = cache.version;
        }
    }
}
bool Constructor::advance()
{
    bool reusablePrefix = cache.version == ledger.version() && cache.cuts.size() == control.graph.sites.size() &&
                          !control.components[activeComponent].cyclic;
    for (auto predecessor : control.predecessors[current]) {
        reusablePrefix &= !control.reachable[predecessor] ||
            (finalized[predecessor] && !control.components[control.component[predecessor]].cyclic);
    }
    if (!reusablePrefix) {
        return replay();
    }
    State state;
    if (current == control.graph.entry) {
        state = initial();
    }
    for (auto predecessor : control.predecessors[current]) {
        if (control.reachable[predecessor] && !join(state, cache.cuts[predecessor].outgoing)) {
            return fail(SelectedFailure::SelectedUpdate, "incompatible forward predecessor state", current);
        }
    }
    cache.cuts[current].incoming = state;
    if (state.causal.reachable() && !word(state, current, cache)) {
        return fail(SelectedFailure::SelectedUpdate, cache.reason, cache.failureCut);
    }
    cache.cuts[current].before = state;
    cache.cuts[current].outgoing = state;
    ++result.work.forwardSiteEvaluations;
    refreshSources();
    return true;
}
bool Constructor::update()
{
    // Recompute the affected suffix from bottom. Reuse only predecessor-closed
    // unchanged components; no old facts seed a changed prefix or loop.
    const auto before = result.work.replaySiteEvaluations;
    if (!replay()) {
        return false;
    }
    SelectedUpdate record;
    record.version = ledger.version();
    record.siteEvaluations = result.work.replaySiteEvaluations - before;
    record.finalizedQueries = std::count(finalized.begin(), finalized.end(), true);
    record.changedCuts = ledger.changes();
    result.updates.push_back(std::move(record));
    ledger.clearChanges();
    ++result.work.selectedUpdates;
    return true;
}
State& Constructor::currentState() { return cache.cuts.at(current).before; }
std::vector<FrontierRequirement> Constructor::residual() const
{
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId || !cache.cuts[current].before.causal.reachable()) {
        return {};
    }
    return frontier.inspect(cache.cuts[current].before.causal, operation).residuals;
}
} // namespace mlir::pto::oahs::selected
