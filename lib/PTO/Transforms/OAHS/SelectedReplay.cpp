// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <chrono>
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
    // A class represented in neither operand has no origin on either side, so
    // the rule below is a no-op there: it either copies an absent value or
    // clears an already absent one. Visiting the union is therefore exact.
    std::vector<Id> classes;
    classes.reserve(target.latest.entries().size() + input.latest.entries().size());
    for (const auto& entry : target.latest.entries()) {
        classes.push_back(entry.first);
    }
    for (const auto& entry : input.latest.entries()) {
        classes.push_back(entry.first);
    }
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
    const auto& history = target.causal.facts()->history;
    const auto& other = input.causal.facts()->history;
    for (auto i : classes) {
        const auto* a = history.find(i);
        const auto* b = other.find(i);
        if (!a) {
            target.latest.set(i, input.latest.get(i));
        } else if (b && target.latest.get(i) != input.latest.get(i)) {
            target.latest.set(i, NoAnalysisId);
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
            replay.failureEndpoint = id;
            replay.failure = step.failure;
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
bool Constructor::payload(State& state, Cut site, Replay& replay, bool pending)
{
    const auto operation = control.graph.operations[site];
    if (operation == NoAnalysisId) {
        return true;
    }
    const auto step = pending ? frontier.pendingIssue(state.causal, operation) : frontier.issue(state.causal, operation);
    if (!step.applied) {
        return refused(replay, site, step);
    }
    state.causal = step.state;
    const auto& op = program.operations[operation];
    for (const auto& access : op.accesses) {
        const auto index = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
        if (access.read) {
            state.latest.set(index, site);
        }
        if (access.write) {
            state.latest.set(index + 1, site);
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
                        incoming[entry].latest.set(base, NoAnalysisId);
                    }
                    if (access.write) {
                        incoming[entry].latest.set(base + 1, NoAnalysisId);
                    }
                }
            }
        }
    }
    for (Id offset = 0; offset <= activeOffset; ++offset) {
        if (!partialSite(index, block.order[offset], incoming, replay)) {
            return false;
        }
    }
    replay.partialComponent = index;
    replay.partialOffset = activeOffset;
    replay.partialIncoming = std::move(incoming);
    return true;
}
bool Constructor::partialSite(Id index, Cut site, std::vector<State>& incoming, Replay& replay)
{
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
                    state.latest.set(base, NoAnalysisId);
                }
                if (access.write) {
                    state.latest.set(base + 1, NoAnalysisId);
                }
            }
        }
    }
    replay.cuts[site].incoming = state;
    if (!state.causal.reachable()) {
        return true;
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
        return true;
    }
    for (auto next : control.constructionEdges[site]) {
        if (control.component[next] == index) {
            if (!join(incoming[next], state)) {
                return false;
            }
        }
    }
    return true;
}
bool Constructor::contextualReplay()
{
    // Solve the actual selected word on every original edge. Unfinished payloads
    // contribute pending effects, never their desired conflict edges. In
    // particular a child region does not reset events or erase incoming work.
    Replay fresh;
    SelectedReplayTrace trace;
    std::vector<bool> visited;
    const auto traceStart = options.traceReplay ? std::chrono::steady_clock::now() :
                                                std::chrono::steady_clock::time_point{};
    if (options.traceReplay) {
        trace.version = ledger.version();
        trace.current = current;
        trace.activeComponent = activeComponent;
        trace.changedCuts = ledger.changes();
        trace.components.resize(control.components.size());
        visited.resize(control.graph.sites.size());
        for (Id i = 0; i < control.components.size(); ++i) {
            trace.components[i].sites = control.components[i].sites.size();
            trace.components[i].cyclic = control.components[i].cyclic;
            auto& successors = trace.components[i].successors;
            for (auto site : control.components[i].sites)
                for (auto next : control.graph.sites[site].successors)
                    if (control.reachable[next] && control.component[next] != i)
                        successors.push_back(control.component[next]);
            std::sort(successors.begin(), successors.end());
            successors.erase(std::unique(successors.begin(), successors.end()), successors.end());
        }
    }
    fresh.version = ledger.version();
    ++result.work.contextualReplays;
    fresh.cuts.resize(control.graph.sites.size());
    std::vector<State> incoming(control.graph.sites.size());
    std::deque<Id> queue;
    std::vector<bool> queued(incoming.size());
    auto enqueue = [&](Id site) {
        if (!queued[site]) {
            queue.push_back(site);
            queued[site] = true;
        }
    };
    // Reuse only predecessor-closed components with unchanged equations, and
    // keep every nonempty shared word entirely on one side of invalidation.
    // Thus both cut states and complete endpoint aggregates remain valid.
    const bool siblingReuse = options.siblingReplayReuse && cache.contextualFixedPoint;
    // The old prefix is only a comparison statistic on the sibling path. Its
    // repeated shared-word widening must not be part of ordinary invalidation.
    const bool comparePrefix = !siblingReuse || options.traceReplay;
    const auto resume = comparePrefix ? reusablePrefix(options.traceReplay ? &trace : nullptr) : 0;
    auto reusable = std::vector<bool>(control.components.size());
    if (siblingReuse)
        reusable = reusableComponents(options.traceReplay ? &trace : nullptr);
    else
        std::fill(reusable.begin(), reusable.begin() + resume, true);
    for (Id index = 0; fresh.success && index < reusable.size(); ++index) {
        if (!reusable[index]) continue;
        ++fresh.reusedComponents;
        if (comparePrefix && index >= resume) {
            ++result.work.siblingReusedComponents;
            if (options.traceReplay) ++trace.siblingComponents;
        }
        for (auto site : control.components[index].sites) {
            fresh.cuts[site] = cache.cuts[site];
            if (options.traceReplay) ++trace.reusedSites;
            for (auto id : ledger.word(site)) {
                const auto found = cache.afterEndpoint.find(id);
                if (found != cache.afterEndpoint.end()) {
                    fresh.afterEndpoint.emplace(id, found->second);
                }
            }
            for (auto next : control.graph.sites[site].successors) {
                if (!control.reachable[next] || reusable[control.component[next]]) {
                    continue;
                }
                if (!join(incoming[next], fresh.cuts[site].outgoing)) {
                    fresh.success = false;
                    fresh.reason = "incompatible reused predecessor state";
                    fresh.failureCut = next;
                    break;
                }
                enqueue(next);
            }
        }
    }
    fresh.fixedComponents = control.components.size();
    fresh.contextualFixedPoint = true;
    if (!reusable[control.component[control.graph.entry]]) {
        incoming[control.graph.entry] = initial();
        enqueue(control.graph.entry);
    }
    while (!queue.empty() && fresh.success) {
        const auto site = queue.front();
        queue.pop_front();
        queued[site] = false;
        auto state = incoming[site];
        ++fresh.evaluations;
        if (options.traceReplay) {
            auto& component = trace.components[control.component[site]];
            ++component.evaluations;
            if (!visited[site]) {
                visited[site] = true;
                ++component.uniqueSites;
                ++trace.uniqueSites;
            }
        }
        fresh.cuts[site].incoming = state;
        if (!word(state, site, fresh)) break;
        fresh.cuts[site].before = state;
        if (!payload(state, site, fresh, true)) break;
        fresh.cuts[site].outgoing = state;
        for (auto next : control.graph.sites[site].successors) {
            if (reusable[control.component[next]]) {
                continue;
            }
            const auto old = incoming[next];
            if (options.traceReplay) ++trace.successorJoins;
            if (!join(incoming[next], state)) {
                fresh.success = false;
                fresh.reason = "incompatible contextual loop state";
                fresh.failureCut = next;
                break;
            }
            if (!same(old, incoming[next])) {
                if (options.traceReplay) ++trace.changedJoins;
                enqueue(next);
            }
        }
    }
    // Earlier selected requirements must still hold after every ledger edit.
    // Test the stabilized states, not an intermediate worklist approximation.
    for (Cut site = 0; fresh.success && site < finalized.size(); ++site) {
        const auto operation = control.graph.operations[site];
        if (!finalized[site] || operation == NoAnalysisId ||
            !fresh.cuts[site].before.causal.reachable()) continue;
        const auto checked = frontier.inspect(fresh.cuts[site].before.causal, operation);
        if (options.traceReplay) ++trace.finalizedQueries;
        if (!checked.applied) refused(fresh, site, checked);
    }
    result.work.replaySiteEvaluations += fresh.evaluations;
    if (options.traceReplay) {
        trace.success = fresh.success;
        trace.evaluations = fresh.evaluations;
        trace.microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - traceStart).count();
        result.replayTraces.push_back(std::move(trace));
    }
    cache = std::move(fresh);
    if (!cache.success) return fail(SelectedFailure::SelectedUpdate, cache.reason, cache.failureCut);
    refreshSources();
    return true;
}
std::vector<bool> Constructor::reusableComponents(SelectedReplayTrace* trace)
{
    std::vector<bool> reusable(control.components.size(), false);
    if (!cache.success || !cache.contextualFixedPoint || cache.cuts.size() != control.graph.sites.size() ||
        cache.fixedComponents != control.components.size() ||
        (cache.version != ledger.version() && ledger.changes().empty())) return reusable;

    // Invalidation is closed under original control successors and under every
    // occurrence of an occupied command word. The complement has unchanged
    // equations and unchanged predecessor inputs, and no endpoint aggregate
    // mixes a reused occurrence with a recomputed occurrence. Pending payload
    // semantics are independent of the construction cursor/finalized flags.
    std::vector<bool> dirty(control.components.size()), visitedWords(control.wordOccurrences.size());
    std::deque<Id> pending;
    auto invalidate = [&](Id component) {
        if (component != NoAnalysisId && !dirty[component]) {
            dirty[component] = true;
            pending.push_back(component);
        }
    };
    uint64_t sites = 0, edges = 0, occurrences = 0;
    for (auto word : ledger.changes()) {
        if (word >= control.wordOccurrences.size()) return reusable;
        for (auto site : control.wordOccurrences[word]) {
            ++occurrences;
            invalidate(control.component[site]);
        }
    }
    while (!pending.empty()) {
        const auto component = pending.front();
        pending.pop_front();
        for (auto site : control.components[component].sites) {
            ++sites;
            for (auto next : control.graph.sites[site].successors) {
                ++edges;
                invalidate(control.component[next]);
            }
            const auto word = control.canonicalCut[site];
            if (visitedWords[word] || ledger.word(word).empty()) continue;
            visitedWords[word] = true;
            for (auto occurrence : control.wordOccurrences[word]) {
                ++occurrences;
                invalidate(control.component[occurrence]);
            }
        }
    }
    result.work.replayInvalidationSites += sites;
    result.work.replayInvalidationEdges += edges;
    result.work.replaySharedWordOccurrences += occurrences;
    if (trace) {
        trace->invalidationSites = sites;
        trace->invalidationEdges = edges;
        trace->sharedWordOccurrences = occurrences;
    }
    for (Id i = 0; i < reusable.size(); ++i) reusable[i] = !dirty[i];
    return reusable;
}
Id Constructor::reusablePrefix(SelectedReplayTrace* trace)
{
    ++result.work.replayPrefixQueries;
    if (!cache.success || cache.cuts.size() != control.graph.sites.size()) {
        return 0;
    }
    // The component order is a topological order of the condensation, so a
    // component before the earliest changed word has unchanged equations and
    // unchanged inputs: its previous least solution is the same and is reused
    // verbatim. No changed or cyclic region is seeded with old facts.
    Id resume = activeComponent;
    // Ledger changes name canonical words, not every original occurrence.
    // Numeric canonical-cut order need not agree with component order, and an
    // occurrence may be unreachable or later than its canonical site.
    for (auto cut : ledger.changes()) {
        if (cut >= control.wordOccurrences.size()) {
            return 0;
        }
        for (auto site : control.wordOccurrences[cut]) {
            if (control.component[site] != NoAnalysisId) {
                resume = std::min(resume, control.component[site]);
                if (trace) trace->changedComponents.push_back(control.component[site]);
            }
        }
    }
    if (trace) {
        trace->changedBoundary = resume;
        auto& changed = trace->changedComponents;
        std::sort(changed.begin(), changed.end());
        changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    }
    // A cyclic component is reused only from its actual fixed point, never from
    // the hypothesis-seeded construction traversal of the active component.
    for (Id index = 0; index < resume; ++index) {
        if (control.components[index].cyclic && index >= cache.fixedComponents) {
            resume = index;
            break;
        }
    }
    if (trace) trace->fixedBoundary = resume;
    // afterEndpoint joins all occurrences of an endpoint. Do not reuse an
    // aggregate containing contributions from the region being recomputed.
    // Keep each shared nonempty word wholly on one side of the boundary. The
    // span of a word is a control fact; only its emptiness depends on the
    // ledger, so the widening reads the precomputed spans.
    bool widened = true;
    while (widened) {
        widened = false;
        for (Cut word = 0; word < control.wordSpan.size(); ++word) {
            ++result.work.replayPrefixSpanExaminations;
            const auto& span = control.wordSpan[word];
            if (span.first == NoAnalysisId || span.first >= resume || span.second < resume ||
                ledger.word(word).empty()) {
                continue;
            }
            resume = span.first;
            if (trace) ++trace->sharedWordLowerings;
            widened = true;
        }
    }
    if (trace) trace->resume = resume;
    return resume;
}
bool Constructor::replay()
{
    if (needsContextualReplay) return contextualReplay();
    Replay fresh;
    fresh.version = ledger.version();
    fresh.cuts.resize(control.graph.sites.size());
    std::vector<State> boundary(control.graph.sites.size());
    const auto resume = reusablePrefix();
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
void Constructor::refreshSources(Cut only)
{
    auto refresh = [&](SelectedSource& source) {
        if (source.cut < cache.cuts.size()) {
            source.snapshot = cache.cuts[source.cut].before.causal;
            source.postOrigin = cache.cuts[source.cut].incoming.causal;
            source.version = cache.version;
        }
    };
    if (only != NoAnalysisId) {
        const auto found = sourcesAtCut.find(only);
        if (found != sourcesAtCut.end()) {
            for (auto id : found->second) {
                refresh(result.sources[id]);
            }
        }
    } else {
        for (auto& source : result.sources) {
            refresh(source);
        }
    }
}
bool Constructor::advance()
{
    if (needsContextualReplay) {
        if (cache.version == ledger.version() && !cache.cuts.empty()) {
            return true;
        }
        return contextualReplay();
    }
    // Continue the hypothesis-seeded construction DAG, not a previous cyclic
    // fixed point. The prior cursor had not yet propagated its payload. Its
    // now-finalized outgoing state is the only new boundary contribution.
    if (cache.success && cache.version == ledger.version() &&
        cache.partialComponent == activeComponent &&
        cache.partialOffset != NoAnalysisId && cache.partialOffset + 1 == activeOffset &&
        cache.partialIncoming.size() == control.graph.sites.size()) {
        const auto previous = control.components[activeComponent].order[cache.partialOffset];
        if (finalized[previous]) {
            for (auto next : control.constructionEdges[previous]) {
                if (control.component[next] == activeComponent &&
                    !join(cache.partialIncoming[next], cache.cuts[previous].outgoing)) {
                    return fail(SelectedFailure::SelectedUpdate, "incompatible construction boundary", next);
                }
            }
            const auto before = cache.evaluations;
            if (!partialSite(activeComponent, current, cache.partialIncoming, cache)) {
                return fail(SelectedFailure::SelectedUpdate, cache.reason, cache.failureCut);
            }
            result.work.forwardSiteEvaluations += cache.evaluations - before;
            cache.partialOffset = activeOffset;
            refreshSources(current);
            return true;
        }
    }
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
    refreshSources(current);
    return true;
}
bool Constructor::update()
{
    // Recompute the affected suffix from bottom. Reuse only predecessor-closed
    // unchanged components; no old facts seed a changed prefix or loop.
    const auto before = result.work.replaySiteEvaluations;
    while (!replay()) {
        if (cache.failure != FrontierFailure::ConsumptionNotEstablished || cache.failureEndpoint == NoAnalysisId)
            return false;
        const auto key = keyIndex(frontier, ledger.endpoint(cache.failureEndpoint).command);
        if (key == NoAnalysisId || !restoreReturns(key)) return false;
        // A new selected endpoint exposed an earlier rearming deadline on a
        // different original path. Restore only that key's pending helpers at
        // their unchanged prefixes. Each helper is permanently required after
        // this transition, so the repair cannot cycle or enumerate subsets.
        result.failure = SelectedFailure::None;
        result.reason.clear();
        result.cut = NoAnalysisId;
    }
    SelectedUpdate record;
    record.version = ledger.version();
    record.siteEvaluations = result.work.replaySiteEvaluations - before;
    record.finalizedQueries = std::count(finalized.begin(), finalized.end(), true);
    record.reusedComponents = cache.reusedComponents;
    record.contextual = needsContextualReplay;
    record.changedCuts = ledger.changes();
    result.work.unreusedUpdates += record.reusedComponents == 0;
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
