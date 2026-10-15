// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "SequenceIncomingQuery.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "WriterPairIndex.h"
namespace mlir::pto::frontiersynch {
namespace {
SequenceAnalysisState::EventKey eventKey(const SequenceEvent& value)
{
    return {value.child, value.type, value.ordinal, value.kind, value.visits};
}
const SequenceAnalysisState::PortChoice* portChoice(const SequenceAnalysisState& state, const SequenceEvent& value)
{
    auto portId = state.portIds.find({value.child, value.type, value.ordinal, value.visits});
    if (portId == state.portIds.end()) { return nullptr; }
    auto found = state.portChoices.find(portId->second);
    return found == state.portChoices.end() ? nullptr : &found->second;
}
std::optional<Expr> selectedPresence(SequenceAnalysisState& state, SequenceEvent root)
{
    // Reflexive reachability is exactly presence. Use the existing owned cache
    // rather than constructing a product of source and target choice DAGs.
    // Choices refer to lower port IDs; an explicit stack also handles deep
    // selection chains without consuming the host call stack.
    auto selfKey = [](const SequenceEvent& value) {
        const auto key = eventKey(value);
        return std::make_pair(key, key);
    };
    auto& memo = state.reachabilityCache;
    std::vector<SequenceEvent> pending{root};
    while (!pending.empty()) {
        const auto value = pending.back();
        const auto key = selfKey(value);
        if (memo.count(key)) { pending.pop_back(); continue; }
        if (const auto* choice = portChoice(state, value)) {
            auto yes = state.event(2 * static_cast<std::size_t>(choice->yes));
            auto no = state.event(2 * static_cast<std::size_t>(choice->no));
            yes.kind = no.kind = value.kind;
            auto yesValue = memo.find(selfKey(yes)), noValue = memo.find(selfKey(no));
            if (yesValue == memo.end() || noValue == memo.end()) {
                if (yesValue == memo.end()) { pending.push_back(std::move(yes)); }
                if (noValue == memo.end()) { pending.push_back(std::move(no)); }
                continue;
            }
            memo.emplace(key, state.expressions.select(choice->choose, yesValue->second, noValue->second));
        } else {
            const auto& region = state.children[value.child].regional;
            auto present = regionalPresence(region, {value.type, value.ordinal, value.kind, value.visits});
            if (!present) { return std::nullopt; }
            memo.emplace(key, *present);
        }
        pending.pop_back();
    }
    return memo.at(selfKey(root));
}
} // namespace
// A child replaces the complete preceding group only after its links have
// been constructed. Keep genuine conditional alternatives, including distinct
// simultaneous envelopes; discard only presence proved false by the arena.
void SequenceAnalysisState::maskFrontier(std::vector<Selected>& values, Expr replaced, SequenceBridgeWork& work)
{
    if (expressions.constantValue(replaced) == 1) { values.clear(); return; }
    if (expressions.constantValue(replaced) == 0) { return; }
    const auto keep = negate(replaced);
    llvm::erase_if(values, [&](Selected& value) {
        ++work.remasked;
        value.present = both(value.present, keep);
        return expressions.constantValue(value.present) == 0;
    });
}
void SequenceAnalysisState::bridges()
{
    bridgeWork = {};
    const SyncStorageEffects* model = nullptr;
    SmallVector<const CompoundInstanceElement*> phases;
    bool residual = false;
    for (const auto& child : children) {
        if (child.regional.accessModel) { model = child.regional.accessModel; }
        for (const auto& anchor : child.regional.anchors) { phases.push_back(anchor.phase); }
        for (const auto& access : child.regional.accessBoundary) { residual |= !access.representedByCells; }
    }
    residual |= model && model->hasUniformRelationships(phases);
    const auto protection = model ? structuredProtection(*model) : StructuredProtection{};
    auto storageCrossing = [&](uint32_t cell, Selected source, Selected target) {
        ++bridgeWork.candidates;
        if (expressions.constantValue(both(source.present, target.present)) == 0) {
            return;
        }
        if (ptoStorageProtection().protectsScalar(pipe(source.port), pipe(target.port))) {
            return;
        }
        const auto& a = ports[source.port];
        const auto& b = ports[target.port];
        auto x = children[a.child].anchors[a.type].phase;
        auto y = children[b.child].anchors[b.type].phase;
        if (cells[cell].space == AddressSpace::ACC &&
            hardwareProtectsConflict(
                pipe(source.port), protection.within(x, requiredOuterLoops), pipe(target.port),
                protection.within(y, requiredOuterLoops))) {
            return;
        }
        const auto before = crossings.size();
        crossing(source, target);
        bridgeWork.retained += crossings.size() - before;
    };
    if (residual) {
        struct BoundaryRoles {
            WriterPairIndex active, deferred;
        };
        std::vector<BoundaryRoles> roles;
        bool valid = true;
        for (const auto& child : children) {
            const auto& region = child.regional;
            auto index = [&](ArrayRef<RegionalAccessBoundary> accesses) {
                return WriterPairIndex(accesses.size(), [&](std::size_t i) {
                    if (!region.accessModel) {
                        return true;
                    }
                    if (region.accessModel != model || accesses[i].effect >= model->effects().size()) {
                        valid = false;
                        return false;
                    }
                    return model->effects()[accesses[i].effect].mode == SyncAccessMode::Read;
                });
            };
            roles.push_back({index(region.accessBoundary), index(region.deferredAccessBoundary)});
        }
        if (!valid) {
            fail("residual boundary has a foreign or invalid original effect identity");
            return;
        }
        // The child index also skips read/read child products. An all-reader
        // sequence does not perform a quadratic empty boundary traversal.
        WriterPairIndex pairs(children.size(), [&](std::size_t i) {
            return !roles[i].active.hasWriters() && !roles[i].deferred.hasWriters();
        });
        pairs.pairs([&](std::size_t first, std::size_t second) {
            const auto a = static_cast<uint32_t>(first), b = static_cast<uint32_t>(second);
            const auto& left = children[a].regional;
            const auto& right = children[b].regional;
            if (!left.accessModel || !right.accessModel) {
                return true;
            }
            auto pair = [&](const auto& xs, const auto& ys, const WriterPairIndex& xroles,
                            const WriterPairIndex& yroles, bool activeEffects) {
                return xroles.cross(yroles, [&](std::size_t i, std::size_t j) {
                    ++bridgeWork.residualCandidates;
                    const auto& x = xs[i];
                    const auto& y = ys[j];
                    const auto& model = *left.accessModel;
                    const auto& source = model.effects()[x.effect];
                    const auto& target = model.effects()[y.effect];
                    const auto sourcePipe = static_cast<uint32_t>(source.phase->kPipeValue);
                    const auto targetPipe = static_cast<uint32_t>(target.phase->kPipeValue);
                    if (ptoStorageProtection().protectsScalar(sourcePipe, targetPipe)) {
                        return true;
                    }
                    const bool accumulator = source.memory && target.memory &&
                                             source.memory->scope == AddressSpace::ACC &&
                                             target.memory->scope == AddressSpace::ACC;
                    if (accumulator && hardwareProtectsConflict(
                                           sourcePipe, protection.within(source.phase, requiredOuterLoops), targetPipe,
                                           protection.within(target.phase, requiredOuterLoops))) {
                        return true;
                    }
                    bool conflict = model.uniformConflict(x.effect, y.effect);
                    const bool residualPair =
                        !conflict && activeEffects && !finiteCrossingPairs.count({a, x.effect, b, y.effect}) &&
                        (!x.representedByCells || !y.representedByCells) && model.residualConflict(x.effect, y.effect);
                    if (residualPair) {
                        if (left.occurrenceLoops[x.last.event.type] || right.occurrenceLoops[y.first.event.type] ||
                            !x.last.event.visits.empty() || !y.first.event.visits.empty()) {
                            fail("crossing symbolic access predicate needs an occurrence adapter");
                            return false;
                        }
                        conflict = true;
                    }
                    if (conflict) {
                        crossing({port(a, x.last.event), x.last.present}, {port(b, y.first.event), y.first.present});
                    }
                    return error.empty();
                });
            };
            return pair(left.accessBoundary, right.accessBoundary, roles[a].active, roles[b].active, true) &&
                   pair(left.accessBoundary, right.deferredAccessBoundary, roles[a].active, roles[b].deferred, false) &&
                   pair(left.deferredAccessBoundary, right.accessBoundary, roles[a].deferred, roles[b].active, false) &&
                   pair(
                       left.deferredAccessBoundary, right.deferredAccessBoundary, roles[a].deferred, roles[b].deferred,
                       false);
        });
        if (!error.empty()) {
            return;
        }
    }
    auto appendLive = [&](std::vector<Selected>& frontier, const std::vector<Selected>& values) {
        for (auto value : values) {
            if (expressions.constantValue(value.present) != 0) {
                frontier.push_back(value);
            }
        }
    };
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        std::vector<Selected> writers;
        std::map<uint32_t, std::vector<Selected>> readers;
        for (const auto& [child, summary] : boundaries[cell]) {
            Expr overwritten = no();
            for (auto first : summary.firstWriters) {
                overwritten = either(overwritten, first.present);
                for (auto old : writers) {
                    storageCrossing(cell, old, first);
                }
                for (const auto& [p, oldReaders] : readers) {
                    for (auto old : oldReaders) { storageCrossing(cell, old, first); }
                }
            }
            for (const auto& [p, firstReaders] : summary.firstReaders) {
                for (auto first : firstReaders) {
                    for (auto old : writers) { storageCrossing(cell, old, first); }
                }
            }
            maskFrontier(writers, overwritten, bridgeWork);
            appendLive(writers, summary.lastWriters);
            for (auto& [p, old] : readers) {
                Expr replaced = overwritten;
                auto found = summary.lastReaders.find(p);
                if (found != summary.lastReaders.end()) {
                    for (auto current : found->second) { replaced = either(replaced, current.present); }
                }
                maskFrontier(old, replaced, bridgeWork);
            }
            for (const auto& [p, current] : summary.lastReaders) { appendLive(readers[p], current); }
        }
    }
}
void SequenceAnalysisState::canonicalizeCrossings()
{
    // Only records with matching child and site identities can denote the
    // same actual pair. Avoid materializing an all-storage-port matrix.
    using Identity = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
    std::map<Identity, std::vector<Crossing>> prior;
    for (auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) == 0) { continue; }
        const auto& source = ports[edge.source];
        const auto& target = ports[edge.target];
        auto& candidates = prior[{source.child, source.type, target.child, target.type}];
        Expr earlier = no();
        for (const auto& previous : candidates) {
            auto identical = both(same(edge.source, previous.source), same(edge.target, previous.target));
            earlier = either(earlier, expressions.select(previous.guard, identical, no()));
        }
        candidates.push_back(edge);
        edge.guard = expressions.select(edge.guard, negate(earlier), no());
    }
}
uint32_t SequenceAnalysisState::selectPort(Expr choose, uint32_t yesPort, uint32_t noPort)
{
    if (yesPort == noPort) { return yesPort; }
    const auto candidate = ports[yesPort], previous = ports[noPort];
    if (candidate.child != previous.child || candidate.type != previous.type ||
        candidate.visits.size() != previous.visits.size()) {
        fail("selected boundary alternatives require one occurrence frame"); return noPort;
    }
    auto ordinal = expressions.select(choose, candidate.ordinal, previous.ordinal);
    std::vector<Expr> visits;
    for (std::size_t i = 0; i < candidate.visits.size(); ++i) {
        visits.push_back(expressions.select(choose, candidate.visits[i], previous.visits[i]));
    }
    const auto previousCount = ports.size();
    auto combined = port(candidate.child, candidate.type, ordinal, visits);
    if (combined == previousCount) { portChoices.emplace(combined, PortChoice{choose, yesPort, noPort}); }
    return combined;
}
void SequenceAnalysisState::normalizeSelectorAlternatives(std::vector<Selected>& values)
{
    // Merge guarded descriptions of the same boundary occurrence before bridge
    // products. Different types, including simultaneous macro pipe envelopes,
    // remain distinct; only coordinate alternatives of one type are selected.
    // Every neighboring region reuses the resulting selected-coordinate maps.
    std::map<uint32_t, Expr> byPort;
    for (const auto& selected : values) {
        if (expressions.constantValue(selected.present) == 0) { continue; }
        auto [found, added] = byPort.emplace(selected.port, selected.present);
        if (!added) { found->second = either(found->second, selected.present); }
    }
    SelectorKey key(byPort.begin(), byPort.end());
    if (auto found = selectorAlternatives.find(key); found != selectorAlternatives.end()) {
        values = found->second; return;
    }
    using Group = std::tuple<uint32_t, uint32_t, std::size_t>;
    std::map<Group, std::size_t> groups;
    std::vector<Selected> folded;
    for (const auto& [portId, presence] : key) {
        const auto selected = ports[portId];
        auto [found, added] = groups.emplace(
            Group{selected.child, selected.type, selected.visits.size()}, folded.size());
        if (added) { folded.push_back({portId, presence}); continue; }
        auto& prior = folded[found->second];
        prior.port = selectPort(presence, portId, prior.port);
        prior.present = either(prior.present, presence);
    }
    selectorAlternatives.emplace(std::move(key), folded);
    values = std::move(folded);
}
void SequenceAnalysisState::foldCrossingEndpoints(bool incomingSources)
{
    // A fixed opposite endpoint needs only the latest active completion on
    // one source site, or the earliest active start on one consumer site.
    // Preserve the closure through native endpoint chains. This folds the candidate
    // description into one selected occurrence, without enumerating guard
    // valuations or building one exclusion predicate per physical byte.
    using Group = std::tuple<uint32_t, uint32_t, uint32_t, std::size_t>;
    std::map<Group, std::size_t> groups;
    std::vector<Crossing> folded;
    for (const auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) == 0) { continue; }
        auto selected = incomingSources ? edge.source : edge.target;
        const auto candidate = ports[selected];
        auto fixed = incomingSources ? edge.target : edge.source;
        auto [found, added] = groups.emplace(
            Group{fixed, candidate.child, candidate.type, candidate.visits.size()}, folded.size());
        if (added) { folded.push_back(edge); continue; }
        auto& accumulated = folded[found->second];
        auto prior = incomingSources ? accumulated.source : accumulated.target;
        auto stronger = incomingSources ? before(prior, selected) : before(selected, prior);
        // The candidate/prior coordinates may use values evaluated only
        // when their respective occurrence guards hold. Preserve lazy masking
        // through selection instead of evaluating an inactive comparison.
        auto choose = expressions.select(edge.guard,
            expressions.select(accumulated.guard, stronger, yes()), no());
        auto combined = selectPort(choose, selected, prior);
        if (incomingSources) { accumulated.source = combined; }
        else { accumulated.target = combined; }
        accumulated.guard = either(accumulated.guard, edge.guard);
    }
    crossings = std::move(folded);
}
void SequenceAnalysisState::consolidateCrossings(bool incomingSources)
{
    // Incoming records keep the latest active source on each pipe. Dually,
    // outgoing records keep the earliest active consumer on each pipe. Native
    // completion/start chains imply the eliminated records. The consolidated
    // snapshot stays immutable while cover deletion changes retained guards.
    auto selectedPort = [&](const Crossing& edge) {
        return incomingSources ? edge.source : edge.target;
    };
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::size_t>> groups;
    for (std::size_t id = 0; id < crossings.size(); ++id) {
        const auto& edge = crossings[id];
        groups[{incomingSources ? edge.target : edge.source, pipe(selectedPort(edge))}].push_back(id);
    }
    for (auto& [key, ids] : groups) {
        ++crossingOrderWork.groups;
        using Order = std::tuple<uint32_t, std::vector<uint64_t>, uint32_t>;
        std::vector<std::pair<Order, std::size_t>> ordered;
        std::map<uint32_t, unsigned> perChild;
        for (auto id : ids) { ++perChild[ports[selectedPort(crossings[id])].child]; }
        bool dynamic = false;
        for (auto id : ids) {
            const auto& source = ports[selectedPort(crossings[id])];
            std::vector<uint64_t> coordinates;
            auto values = source.visits;
            values.push_back(source.ordinal);
            bool constant = true;
            for (auto value : values) {
                auto number = expressions.constantValue(value);
                if (!number) { constant = false; break; }
                coordinates.push_back(*number);
            }
            // Distinct children have certified reference order independently
            // of their occurrence coordinates. Within a child retain the
            // existing constant-coordinate sort and exact adjacent proof.
            if (!constant && perChild.at(source.child) != 1) { break; }
            if (!constant) { coordinates.clear(); dynamic = true; }
            ordered.push_back({{source.child, std::move(coordinates), source.type}, id});
        }
        bool chain = ordered.size() == ids.size();
        if (chain) {
            llvm::sort(ordered);
            for (std::size_t i = 1; i < ordered.size(); ++i) {
                ++crossingOrderWork.orderQueries;
                chain &= expressions.constantValue(before(selectedPort(crossings[ordered[i-1].second]),
                    selectedPort(crossings[ordered[i].second]))) == 1;
            }
        }
        if (chain) {
            ++crossingOrderWork.qualified;
            if (dynamic && ids.size() > 1) { ++crossingOrderWork.dynamicQualified; }
            // A verified ordering supports a suffix union for sources and a
            // prefix union for consumers, including differing presence guards.
            if (incomingSources) { std::reverse(ordered.begin(), ordered.end()); }
            auto stronger = no();
            for (const auto& entry : ordered) {
                auto& edge = crossings[entry.second];
                auto original = edge.guard;
                edge.guard = both(original, negate(stronger));
                stronger = either(stronger, original);
            }
            continue;
        }
        ++crossingOrderWork.residual;
        std::vector<Crossing> original;
        for (auto id : ids) { original.push_back(crossings[id]); }
        for (auto id : ids) {
            auto& edge = crossings[id];
            auto covered = no();
            for (const auto& other : original) {
                if (selectedPort(other) == selectedPort(edge)) { continue; }
                ++crossingOrderWork.orderQueries;
                auto order = incomingSources ? before(edge.source, other.source) : before(other.target, edge.target);
                covered = either(covered, expressions.select(other.guard, order, no()));
            }
            edge.guard = expressions.select(edge.guard, negate(covered), no());
        }
    }
}
SequenceEvent SequenceAnalysisState::event(std::size_t id) const
{
    const auto& selected = ports[id / 2];
    return {selected.child, selected.type, selected.ordinal,
        id % 2 ? PeriodicEventKind::Completion : PeriodicEventKind::Start, selected.visits};
}
std::optional<Expr> SequenceAnalysisState::eventReachability(std::size_t source, std::size_t target)
{
    if (source / 2 >= ports.size() || target / 2 >= ports.size()) { return std::nullopt; }
    if (numerical) {
        return numericalReachability(event(source), event(target), numericalQueryCost);
    }
    return eventReachability(event(source), event(target));
}
std::optional<Expr> SequenceAnalysisState::eventReachability(SequenceEvent source, SequenceEvent target)
{
    if (numerical) {
        auto answer = numericalReachability(source, target, numericalQueryCost);
        if (answer) { return answer; }
    }
    auto pair = std::make_pair(eventKey(source), eventKey(target));
    auto cached = reachabilityCache.find(pair);
    if (cached != reachabilityCache.end()) { return cached->second; }
    if (source.child >= children.size() || target.child >= children.size()) { return std::nullopt; }
    if (source.child > target.child) { return no(); }
    // Reference order places every occurrence of an earlier child before
    // every occurrence of a later child. On one pipe, native start and
    // completion chains therefore decide all event-kind pairs except C->I,
    // without traversing storage crossings or the intervening children.
    const auto& left = children[source.child];
    const auto& right = children[target.child];
    const bool nativeKinds = source.kind != PeriodicEventKind::Completion || target.kind != PeriodicEventKind::Start;
    if (source.child < target.child && nativeKinds &&
        source.type < left.anchors.size() && target.type < right.anchors.size() &&
        left.anchors[source.type].phase && right.anchors[target.type].phase &&
        left.anchors[source.type].phase->kPipeValue == right.anchors[target.type].phase->kPipeValue) {
        auto pa = selectedPresence(*this, source);
        auto pb = selectedPresence(*this, target);
        if (!pa || !pb) { return std::nullopt; }
        auto answer = both(*pa, *pb);
        reachabilityCache.emplace(std::move(pair), answer);
        return answer;
    }
    // A selected occurrence denotes exactly one original occurrence, including
    // its presence. Expand the alias before any child query. Memoization is by
    // endpoint pair, so shared selection DAGs do not enumerate choice paths.
    const auto* sourceChoice = portChoice(*this, source);
    const auto* targetChoice = portChoice(*this, target);
    if (sourceChoice && targetChoice && sourceChoice->choose == targetChoice->choose) {
        // The same original selector cannot choose opposite arms. This is the
        // existing nested-select identity, applied before asking the children
        // to construct unreachable pairs. Keep both unrestricted diagonal
        // queries and the outer select, including its undefined-arm masking.
        const auto a = *sourceChoice, b = *targetChoice;
        auto original = [&](uint32_t port, PeriodicEventKind kind) {
            auto value = event(2 * static_cast<std::size_t>(port));
            value.kind = kind;
            return value;
        };
        accumulateCost(crossingQueryWork.correlatedChoices, 1);
        auto yes = eventReachability(original(a.yes, source.kind), original(b.yes, target.kind));
        auto no = eventReachability(original(a.no, source.kind), original(b.no, target.kind));
        if (!yes || !no) { return std::nullopt; }
        auto answer = expressions.select(a.choose, *yes, *no);
        reachabilityCache.emplace(std::move(pair), answer);
        return answer;
    }
    if (sourceChoice || targetChoice) {
        const auto selected = sourceChoice ? *sourceChoice : *targetChoice;
        const auto kind = sourceChoice ? source.kind : target.kind;
        auto original = [&](uint32_t portId) {
            auto value = event(2 * static_cast<std::size_t>(portId));
            value.kind = kind;
            return value;
        };
        auto yes = sourceChoice ? eventReachability(original(selected.yes), target) :
                                  eventReachability(source, original(selected.yes));
        auto no = sourceChoice ? eventReachability(original(selected.no), target) :
                                 eventReachability(source, original(selected.no));
        if (!yes || !no) { return std::nullopt; }
        auto answer = expressions.select(selected.choose, *yes, *no);
        reachabilityCache.emplace(std::move(pair), answer);
        return answer;
    }
    auto local = [&](SequenceEvent a, SequenceEvent b) {
        const auto& region = children[a.child].regional;
        RegionalEvent left{a.type, a.ordinal, a.kind, a.visits};
        RegionalEvent right{b.type, b.ordinal, b.kind, b.visits};
        // Every modeled edge is forward in occurrence order, regardless of
        // its endpoint event kinds. Do not ask a compact backend to expand a
        // query whose target is provably an earlier occurrence.
        if (region.capabilities.exactQueries) {
            auto reverse = regionalReferenceBefore(region, right, left);
            if (reverse && expressions.constantValue(*reverse) == 1) {
                return std::optional<Expr>{no()};
            }
        }
        auto reach = regionalReachability(region, left, right);
        auto leftPresent = regionalPresence(region, left), rightPresent = regionalPresence(region, right);
        if (!reach || !leftPresent || !rightPresent) { return std::optional<Expr>{}; }
        return std::optional<Expr>{both(*reach, both(*leftPresent, *rightPresent))};
    };
    if (source.child == target.child) {
        auto answer = local(source, target);
        if (answer) { reachabilityCache.emplace(std::move(pair), *answer); }
        return answer;
    }
    // Every path has a last cross-child link. Its prefix ends in an earlier
    // child, so this memoized recurrence is acyclic. Only queried pairs are
    // materialized; an exported storage selector need not be a graph vertex.
    auto answer = no();
    auto found = incoming.find(target.child);
    if (found != incoming.end()) {
        for (const auto& link : found->second) {
            if (ports[link.source / 2].child < source.child || expressions.constantValue(link.guard) == 0) {
                continue;
            }
            auto suffix = eventReachability(event(link.target), target);
            if (!suffix) { return std::nullopt; }
            auto tail = expressions.select(link.guard, *suffix, no());
            if (expressions.constantValue(tail) == 0) { continue; }
            auto prefix = eventReachability(source, event(link.source));
            if (!prefix) { return std::nullopt; }
            answer = either(answer, expressions.select(tail, *prefix, no()));
        }
    }
    reachabilityCache.emplace(std::move(pair), answer);
    return answer;
}
std::optional<uint64_t> SequenceIncomingQuery::fact(Expr activity, Expr condition)
{
    const auto key = std::make_pair(activity, condition);
    if (auto found = facts.find(key); found != facts.end()) { return found->second; }
    std::optional<uint64_t> value;
    {
        RegionExpressions::Transaction proof(state.expressions);
        value = state.expressions.constantUnder(activity, condition, &state.crossingQueryWork.incomingProofVisits);
    }
    if (value) { accumulateCost(state.crossingQueryWork.incomingFacts, 1); }
    facts.emplace(key, value);
    return value;
}
Expr SequenceIncomingQuery::coordinate(Expr activity, Expr root)
{
    const auto key = std::make_pair(activity, root);
    if (auto found = coordinates.find(key); found != coordinates.end()) { return found->second; }
    auto& e = state.expressions;
    SmallVector<Expr> conditions;
    e.visitReferenced({root}, [&](Expr id) {
        if (e.isBoolean(id) && !e.constantValue(id)) { conditions.push_back(id); }
    });
    SmallVector<std::pair<Expr, Expr>> bindings;
    for (auto condition : conditions) {
        if (auto value = fact(activity, condition)) { bindings.emplace_back(condition, e.boolean(*value != 0)); }
    }
    Expr answer = root;
    if (!bindings.empty()) {
        RegionExpressions::Transaction specialization(e);
        RegionExpressions::Substitution substitution(bindings);
        const auto selected = e.substitute(root, substitution);
        if (selected != RegionExpressions::invalid && e.constructionError().empty()) {
            answer = selected;
            specialization.commit();
        }
        // An unsupported rebuild retains the original exact query. No child
        // callback has run, and no partially rebuilt expression may escape.
    }
    if (answer != root) { accumulateCost(state.crossingQueryWork.incomingCoordinates, 1); }
    coordinates.emplace(key, answer);
    return answer;
}
SequenceEvent SequenceIncomingQuery::occurrence(Expr activity, SequenceEvent value)
{
    while (const auto* choice = portChoice(state, value)) {
        const auto selected = fact(activity, choice->choose);
        // An unresolved alias retains its complete original choice/frame query.
        // Do not turn its folded coordinates into an invented primitive event.
        if (!selected) { return value; }
        const auto kind = value.kind;
        value = state.event(2 * static_cast<std::size_t>(*selected ? choice->yes : choice->no));
        value.kind = kind;
        accumulateCost(state.crossingQueryWork.incomingArms, 1);
    }
    // Optional exact callbacks need not accept a different representation of
    // the same coordinate. Never speculate then recover from a stateful refusal.
    if (!state.children[value.child].regional.capabilities.substitutionStableQueries) { return value; }
    value.ordinal = coordinate(activity, value.ordinal);
    for (auto& visit : value.visits) { visit = coordinate(activity, visit); }
    return value;
}
std::optional<Expr> SequenceIncomingQuery::query(std::size_t source, std::size_t target, Expr activity)
{
    if (source / 2 >= state.ports.size() || target / 2 >= state.ports.size()) { return std::nullopt; }
    return query(state.event(source), state.event(target), activity);
}
std::optional<Expr> SequenceIncomingQuery::query(SequenceEvent source, SequenceEvent target, Expr activity)
{
    auto& e = state.expressions;
    auto valid = [&](const SequenceEvent& event) {
        auto coordinate = [&](Expr id) { return id < e.size() && !e.isBoolean(id); };
        return event.child < state.children.size() && event.type < state.children[event.child].anchors.size() &&
            coordinate(event.ordinal) && llvm::all_of(event.visits, coordinate);
    };
    if (activity >= e.size() || !e.isBoolean(activity) || !valid(source) || !valid(target)) { return std::nullopt; }
    accumulateCost(state.crossingQueryWork.incomingRequests, 1);
    if (e.constantValue(activity) == 0) { return e.boolean(false); }
    const Key key{activity, eventKey(source), eventKey(target)};
    if (auto found = answers.find(key); found != answers.end()) { return found->second; }
    const auto a = occurrence(activity, source), b = occurrence(activity, target);
    if (eventKey(a) == eventKey(source) && eventKey(b) == eventKey(target)) {
        accumulateCost(state.crossingQueryWork.incomingResidual, 1);
    }
    // These are ordinary exact queries of the qualified coordinates. Only the
    // caller's activity restricts their use; captured executed ranks and original
    // occurrence recipes are untouched. Missing residual queries still refuse.
    auto answer = state.eventReachability(a, b);
    if (!answer) { return std::nullopt; }
    const auto masked = e.select(activity, *answer, e.boolean(false));
    answers.emplace(key, masked);
    return masked;
}
bool SequenceAnalysisState::valueBridges()
{
    for (uint32_t b = 0; b < children.size(); ++b) {
        for (uint32_t target = 0; target < children[b].anchors.size(); ++target) {
            for (const auto& edge : index.prerequisitesFor(children[b].anchors[target].phase->elementOp)) {
                for (uint32_t a = 0; a < b; ++a) {
                    for (uint32_t source = 0; source < children[a].anchors.size(); ++source) {
                        if (children[a].anchors[source].phase != edge.producer) { continue; }
                        const auto& left = children[a].regional;
                        const auto& right = children[b].regional;
                        const auto targetFirsts = right.firstSitePayloads.find(target);
                        if ((!left.outerLoops.empty() && !left.outerLoops[source].empty()) ||
                            (targetFirsts == right.firstSitePayloads.end() &&
                             !right.outerLoops.empty() && !right.outerLoops[target].empty())) {
                            return fail("nested crossing value prerequisite requires a coordinate map");
                        }
                        auto sourceLoop = children[a].regional.occurrenceLoops[source];
                        auto targetLoop = children[b].regional.occurrenceLoops[target];
                        // Slices of one original loop have disjoint ordinals.
                        // SSA edges within its body connect the same iteration,
                        // and have already been handled by each local analysis.
                        if (sourceLoop && sourceLoop == targetLoop) { continue; }
                        if (sourceLoop) {
                            return fail("crossing value producer requires a last-occurrence selector");
                        }
                        const auto x = port(a, source, c(0));
                        auto connect = [&](RegionalSelector first) {
                            const auto y = port(b, first.event);
                            const auto active = both(first.present, present(y));
                            const auto guard = both(present(x), active);
                            if (edge.native) { nativeValueCrossings.push_back({x, y, guard}); }
                            else { crossing({x, present(x)}, {y, active}); }
                        };
                        if (targetFirsts != right.firstSitePayloads.end()) {
                            // The first executed occurrence need not have ordinal
                            // zero: a guard may skip an arbitrary initial prefix.
                            // Native start order carries this prerequisite to
                            // every later occurrence of the same payload site.
                            for (auto first : targetFirsts->second) { connect(first); }
                        } else {
                            connect({{target, right.firstOrdinal.value_or(c(0)), PeriodicEventKind::Start}, yes()});
                        }
                    }
                }
            }
        }
    }
    return error.empty();
}
void SequenceAnalysisState::buildNativeLinks()
{
    nativeFrontierWork = {};
    auto add = [&](std::size_t source, std::size_t target, Expr guard) {
        if (expressions.constantValue(guard) == 0) { return; }
        incoming[ports[target / 2].child].push_back({source, target, guard});
        ++nativeFrontierWork.retained;
    };
    std::map<uint32_t, std::vector<Selected>> preceding;
    for (uint32_t childId = 0; childId < children.size(); ++childId) {
        for (const auto& [p, firsts] : nativeFirst[childId]) {
            auto& frontier = preceding[p];
            auto nonempty = no();
            for (auto selected : firsts) {
                if (expressions.constantValue(selected.present) == 0) { continue; }
                nonempty = either(nonempty, selected.present);
                auto first = selected.port;
                for (auto old : frontier) {
                    ++nativeFrontierWork.candidates;
                    auto guard = both(old.present, selected.present);
                    add(2*old.port, 2*first, guard);
                    add(2*old.port+1, 2*first+1, guard);
                }
            }
            maskFrontier(frontier, nonempty, nativeFrontierWork);
            auto found = nativeLast[childId].find(p);
            if (found != nativeLast[childId].end()) {
                for (auto selected : found->second) {
                    if (expressions.constantValue(selected.present) != 0) { frontier.push_back(selected); }
                }
            }
        }
    }
}
bool SequenceAnalysisState::closure()
{
    crossingPresence.reset();
    if (!error.empty()) { return false; }
    if (nativeFirst.size() != children.size() || nativeLast.size() != children.size()) {
        return fail("sequence native boundary selectors were not imported");
    }
    incoming.clear();
    reachabilityCache.clear();
    auto add = [&](std::size_t source, std::size_t target, Expr guard) {
        if (expressions.constantValue(guard) != 0) {
            incoming[ports[target / 2].child].push_back({source, target, guard});
        }
    };
    buildNativeLinks();
    for (const auto& edge : nativeValueCrossings) { add(2*edge.source+1, 2*edge.target, edge.guard); }
    // Estimate before leaf callbacks or symbolic crossing circuits are built.
    // Each attempt receives the same native and full crossing graph.
    const auto nativeLinks = incoming;
    for (const auto& edge : crossings) { add(2 * edge.source + 1, 2 * edge.target, edge.guard); }
    const bool numericalFirst = preferNumericalCrossings();
    const auto allLinks = incoming;
    for (unsigned position = 0; position < 2; ++position) {
        const bool numeric = position == 0 ? numericalFirst : !numericalFirst;
        auto& method = crossingMethods[numeric ? 0 : 1];
        ++method.attemptConstructions;
        // Normal adapter failures may append shared callback circuits. Keep
        // their IDs alive: child caches can own them across this retry.
        const auto savedPorts = ports;
        const auto savedIds = portIds;
        const auto savedChoices = portChoices;
        const auto savedCrossings = crossings;
        const auto savedCrossingIds = crossingIds;
        const auto savedNative = nativeValueCrossings;
        incoming = numeric ? allLinks : nativeLinks;
        const bool accepted = numeric ? numericalCrossingReduction() : symbolicCrossingReduction();
        if (accepted && error.empty() && expressions.constructionError().empty()) {
            sealCrossingPresence();
            return true;
        }
        crossingObligations.push_back(method.method + ": " +
            (error.empty() ? "required representation or query unavailable" : error));
        if (!expressions.constructionError().empty()) { return false; }
        ports = savedPorts; portIds = savedIds; portChoices = savedChoices;
        crossings = savedCrossings; crossingIds = savedCrossingIds; nativeValueCrossings = savedNative;
        incoming = allLinks;
        reachabilityCache.clear();
        numericalTree.reset(); numerical.reset(); numericalChainKeys.clear();
        numericalEvents.clear(); numericalProjectedPorts = 0; numericalBoundary.reset();
        error.clear();
    }
    std::string diagnostic = "neither implemented crossing reducer supplied its required interfaces";
    for (const auto& obligation : crossingObligations) { diagnostic += "; " + obligation; }
    return fail(diagnostic);
}
bool SequenceAnalysisState::symbolicCrossingReduction()
{
    crossingQueryWork = {};
    auto reductionLinks = incoming;
    auto add = [&](std::size_t source, std::size_t target, Expr guard) {
        if (expressions.constantValue(guard) != 0) {
            incoming[ports[target / 2].child].push_back({source, target, guard});
        }
    };
    for (auto& edge : crossings) {
        auto native = no();
        for (const auto& fixed : nativeValueCrossings) {
            auto identical = both(same(fixed.source, edge.source), same(fixed.target, edge.target));
            native = either(native, both(fixed.guard, identical));
        }
        add(2*edge.source+1, 2*edge.target, edge.guard);
        edge.guard = both(edge.guard, negate(native));
    }
    foldCrossingEndpoints(true);
    consolidateCrossings(true);
    foldCrossingEndpoints(false);
    consolidateCrossings(false);
    // Native order and the consolidated generators have the same closure as
    // the original links. Snapshot every candidate before testing deletion;
    // deleting candidates in place must not change another candidate's test.
    // Both exported queries and last-entry tests use the smaller graph. The
    // immutable snapshot must precede cover deletion, whose retained guards
    // can depend on those very queries.
    for (const auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) != 0) {
            reductionLinks[ports[edge.target].child].push_back({2*edge.source+1, 2*edge.target, edge.guard});
        }
    }
    incoming = reductionLinks;
    reachabilityCache.clear();
    // Consolidation can make candidate descriptions mutually exclusive. Prove
    // those exclusions before constructing child reachability circuits. The
    // cache keys are immutable predicate IDs from the predeletion snapshot;
    // no retained guard is used to justify another candidate's deletion.
    SequenceIncomingQuery incomingQueries(*this);
    std::map<std::pair<Expr, Expr>, bool> incompatibilities;
    auto incompatible = [&](Expr a, Expr b) {
        const auto av = expressions.constantValue(a), bv = expressions.constantValue(b);
        if (av == 0 || bv == 0) { return true; }
        if (a == b || av == 1 || bv == 1) { return false; }
        if (b < a) { std::swap(a, b); }
        const auto key = std::make_pair(a, b);
        auto found = incompatibilities.find(key);
        if (found != incompatibilities.end()) { return found->second; }
        const bool answer = expressions.implies(a, negate(b));
        incompatibilities.emplace(key, answer);
        return answer;
    };
    // An alternative path has one last link entering the consumer's child.
    // Its prefix ends earlier and cannot use the tested edge. The suffix is
    // local. Exclude every alias of the tested completion-to-start pair.
    for (auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) == 0) { continue; }
        const auto candidateGuard = edge.guard;
        Expr alternate = no();
        for (const auto& entry : reductionLinks[ports[edge.target].child]) {
            ++crossingQueryWork.entries;
            if (incompatible(candidateGuard, entry.guard)) { continue; }
            auto distinct = yes();
            if (entry.source % 2 == 1 && entry.target % 2 == 0) {
                distinct = negate(both(same(entry.source / 2, edge.source), same(entry.target / 2, edge.target)));
            }
            if (expressions.constantValue(distinct) == 0) { ++crossingQueryWork.aliases; continue; }
            if (ports[entry.source / 2].child < ports[edge.source].child) { continue; }
            // Every modeled path advances original occurrence order. A
            // completion cannot reach its own start. This exact certificate
            // needs no child callback and avoids an unused cross-child prefix.
            // Other tails retain prefix-first evaluation: unavailable child
            // queries can record errors, even when a zero prefix would skip
            // them, so speculative suffix callbacks are not interchangeable.
            if (entry.target == 2 * edge.target + 1) { ++crossingQueryWork.zeroTails; continue; }
            // Only this exact activity consumes either path query. Qualify
            // selected coordinates before constructing an unrestricted child DAG.
            const auto activity = expressions.select(candidateGuard,
                expressions.select(entry.guard, distinct, no()), no());
            ++crossingQueryWork.prefixQueries;
            auto prefix = incomingQueries.query(2*edge.source+1, entry.source, activity);
            if (!prefix) { return fail("child all-event query unavailable"); }
            if (expressions.constantValue(*prefix) == 0) { continue; }
            ++crossingQueryWork.suffixQueries;
            auto suffix = incomingQueries.query(entry.target, 2*edge.target, activity);
            if (!suffix) { return fail("child all-event query unavailable"); }
            auto tail = expressions.select(entry.guard, both(distinct, *suffix), no());
            if (expressions.constantValue(tail) == 0) { continue; }
            alternate = either(alternate, expressions.select(tail, *prefix, no()));
        }
        edge.guard = expressions.select(edge.guard, negate(alternate), no());
    }
    canonicalizeCrossings();
    return error.empty();
}

} // namespace mlir::pto::frontiersynch
