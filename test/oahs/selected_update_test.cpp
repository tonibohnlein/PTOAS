// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
#include <numeric>
using namespace selected_test;
namespace mlir::pto::oahs::selected {
// Inspect intermediate construction checkpoints, not the already-cold final
// certificate. Keep the production API free of a second replay/planner mode.
struct ReplayTestAccess {
    // The complete semantic checkpoint, not just acceptance: causal facts, the
    // original-occurrence record and consumption evidence at every cut, plus
    // every endpoint aggregate.
    static void identical(const Replay& reused, const Replay& cold) {
        auto sameState = [](const State& a, const State& b) {
            require(a.causal == b.causal && a.latest == b.latest && a.consumptions == b.consumptions,
                    "incremental selected checkpoint differs from cold replay");
        };
        require(reused.cuts.size() == cold.cuts.size(), "checkpoint population");
        for (Cut site = 0; site < reused.cuts.size(); ++site) {
            sameState(reused.cuts[site].incoming, cold.cuts[site].incoming);
            sameState(reused.cuts[site].before, cold.cuts[site].before);
            sameState(reused.cuts[site].outgoing, cold.cuts[site].outgoing);
        }
        require(reused.afterEndpoint.size() == cold.afterEndpoint.size(), "endpoint population");
        for (const auto& entry : reused.afterEndpoint) {
            sameState(entry.second, cold.afterEndpoint.at(entry.first));
        }
    }
    static void compareAdvances(const Program& program, Pipe pipe, bool barriers = true, bool mixed = false, bool contextual = false) {
        Constructor c(program);
        c.needsContextualReplay = contextual;
        require(c.control.complete && c.frontier.complete(),
                "advance fixture import: " + c.control.reason + " / " + c.frontier.reason());
        Commands fixed(commandCutCount(program));
        for (Cut cut = 0; cut < fixed.size(); ++cut) {
            if (barriers && legalCommandCut(program, cut)) {
                fixed[cut] = {{Command::Barrier, pipe, pipe, 0}};
            }
        }
        std::string reason;
        require(c.ledger.initialize(fixed, reason), reason);
        uint64_t incrementalWork = 0, coldWork = 0;
        bool edited = false;
        unsigned resumed = 0, coveredQueries = 0, positiveCoverage = 0;
        auto compareCold = [&] {
            auto reused = c.cache;
            const auto sources = c.result.sources;
            auto required = c.residual();
            // Query release coverage even in the no-write fixture, where no
            // current payload needs a transfer. These queries must carry real
            // outstanding reader histories, not just compare empty residuals.
            for (unsigned cell = 0; cell < program.cells.size(); ++cell)
                required.push_back({cell, pipe, false, NoAnalysisId, false, true});
            std::vector<std::set<Id>> coverage;
            for (const auto& source : sources)
                coverage.push_back(c.coverage(source.cut, source.pipe, required));
            c.cache = {};
            require(c.replay(), c.cache.reason);
            coldWork += c.cache.evaluations;
            identical(reused, c.cache);
            for (Id i = 0; i < sources.size(); ++i) {
                const auto& actual = sources[i];
                const auto& expected = c.result.sources[i];
                require(actual.snapshot == expected.snapshot && actual.postOrigin == expected.postOrigin && actual.version == expected.version,
                        "indexed source refresh differs from cold replay");
                require(coverage[i] == c.coverage(actual.cut, actual.pipe, required),
                        "source coverage differs after cold replay");
                coveredQueries += actual.snapshot.reachable();
                positiveCoverage += !coverage[i].empty();
            }
            c.cache = std::move(reused);
            c.result.sources = sources;
        };
        for (c.activeComponent = 0; c.activeComponent < c.control.components.size(); ++c.activeComponent) {
            const auto& block = c.control.components[c.activeComponent];
            for (c.activeOffset = 0; c.activeOffset < block.order.size(); ++c.activeOffset) {
                c.current = block.order[c.activeOffset];
                auto before = c.result.work.replaySiteEvaluations + c.result.work.forwardSiteEvaluations;
                require(c.advance(), c.cache.reason);
                incrementalWork += c.result.work.replaySiteEvaluations + c.result.work.forwardSiteEvaluations - before;
                compareCold();
                if (edited) ++resumed;
                // Insert a real, rearmed event exchange after several cached
                // advances. The early SET forces rebuilding the current cyclic
                // prefix; subsequent advances must resume from that new ledger.
                if (mixed && !edited && c.activeOffset >= 6 && !c.result.sources.empty()) {
                    Cut source = NoAnalysisId;
                    for (const auto& saved : c.result.sources) {
                        if (saved.cut != c.current && c.control.straight(saved.cut, c.current) &&
                            c.control.position[saved.cut] < c.control.position[c.current] &&
                            legalCommandCut(program, saved.cut)) source = saved.cut;
                    }
                    if (source != NoAnalysisId && legalCommandCut(program, c.current)) {
                        const auto target = Pipe::V;
                        c.ledger.append(source, {Command::Publish, pipe, target, 0}, EndpointPurpose::Fixed);
                        c.ledger.append(c.current, {Command::Acquire, pipe, target, 0}, EndpointPurpose::Fixed);
                        c.ledger.append(c.current, {Command::Publish, target, pipe, 0}, EndpointPurpose::Fixed);
                        c.ledger.append(c.current, {Command::Acquire, target, pipe, 0}, EndpointPurpose::Fixed);
                        require(c.update(), c.cache.reason);
                        compareCold();
                        edited = true;
                    }
                }
                require(c.consume(), c.result.reason);
                compareCold();
                auto outgoing = c.currentState();
                if (outgoing.causal.reachable()) {
                    require(c.payload(outgoing, c.current, c.cache), c.cache.reason);
                }
                c.cache.cuts[c.current].outgoing = std::move(outgoing);
                c.finalized[c.current] = true;
                const auto operation = c.control.graph.operations[c.current];
                const auto after = c.control.after(c.current);
                c.registerSource();
                if (operation != NoAnalysisId && after != NoAnalysisId) {
                    if (c.needsContextualReplay) {
                        const auto& source = c.result.sources.back();
                        require(source.snapshot.reachable() && source.version == c.ledger.version() &&
                                source.snapshot == c.cache.cuts[after].before.causal,
                                "new contextual source must be usable without a later edit");
                    }
                }
            }
        }
        require(!mixed || (edited && resumed >= 3), "mixed edit did not resume cached advancement");
        require(coveredQueries != 0, "source snapshots were never inspected");
        require(barriers || positiveCoverage != 0, "no outstanding reader coverage was queried");
        require(contextual || c.result.work.forwardSiteEvaluations > 0,
                "no construction prefix was continued");
        require(incrementalWork <= 8 * c.control.graph.sites.size(),
                "unchanged traversal revisits growing cyclic prefixes");
        require(coldWork > incrementalWork, "test did not exercise saved replay work");
    }
    static void prefixComparisonCost(unsigned words) {
        const auto P = Pipe::MTE2;
        auto p = base(1);
        p.operations = {op(P, {{0, true, false}})};
        ObservedControl graph;
        graph.qualification = "chained shared-word prefix comparison";
        graph.entry = 0; graph.exit = 2 * words + 1;
        for (Id site = 0; site <= graph.exit; ++site) {
            graph.observations.push_back({site, {}, true});
            graph.sites.push_back({site == 1 ? 0 : NoControlId, site, {}, {}, 0});
            if (site < graph.exit) graph.sites.back().successors = {site + 1};
        }
        for (Id word = 0; word < words; ++word)
            graph.sites[2 * word + 3].observation = 2 * word;
        p.observed = std::move(graph);
        Constructor c(p);
        require(c.control.complete && c.frontier.complete(), c.control.reason + " / " + c.frontier.reason());
        c.needsContextualReplay = true;
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        for (Id word = 0; word < words; ++word)
            c.ledger.append(2 * word, {Command::Barrier, P, P, 0}, EndpointPurpose::Fixed);
        c.current = c.control.graph.exit;
        c.activeComponent = c.control.component[c.current];
        require(c.replay(), c.cache.reason);
        c.ledger.clearChanges();
        c.ledger.append(2 * words, {Command::Barrier, P, P, 0}, EndpointPurpose::Fixed);
        const auto original = c.cache;
        const auto before = c.result.work;
        require(c.replay(), c.cache.reason);
        const auto reused = c.cache;
        require(c.result.work.replayPrefixQueries == before.replayPrefixQueries &&
                c.result.work.replayPrefixSpanExaminations == before.replayPrefixSpanExaminations,
                "normal sibling replay still computes the legacy comparison prefix");
        const auto visits = c.result.work.replayInvalidationSites - before.replayInvalidationSites;
        require(visits <= c.control.graph.sites.size(), "invalidation revisited a shared-word site");
        c.cache = original;
        c.options.traceReplay = true;
        require(c.replay(), c.cache.reason);
        identical(reused, c.cache);
        const auto scans = c.result.work.replayPrefixSpanExaminations - before.replayPrefixSpanExaminations;
        require(scans >= uint64_t(words) * words,
                "comparison fixture did not exercise chained quadratic widening");
        c.cache = original;
        c.options.traceReplay = false;
        c.options.siblingReplayReuse = false;
        require(c.replay(), c.cache.reason);
        identical(reused, c.cache);
        c.cache = {};
        require(c.replay(), c.cache.reason);
        identical(reused, c.cache);
        std::cout << "shared_words=" << words << " comparison_spans=" << scans
                  << " invalidation_sites=" << visits << '\n';
    }
    static void dischargeRestore()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE3;
        auto p = base(4, 2);
        p.operations = {op(P, {{0, true, false}}), op(Q, {{1, true, false}}),
                        op(R, {{2, true, false}}), op(P, {{3, true, false}})};
        Constructor c(p);
        c.needsContextualReplay = true;
        std::string reason;
        require(c.ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        auto append = [&](Cut cut, decltype(Command::Publish) kind, Pipe from, Pipe to, unsigned key,
                          EndpointPurpose purpose, Id ack = NoAnalysisId) {
            return c.ledger.append(cut, {kind, from, to, key}, purpose, NoAnalysisId, ack);
        };
        append(1, Command::Publish, P, Q, 0, EndpointPurpose::Completion);
        const auto forward = append(1, Command::Acquire, P, Q, 0, EndpointPurpose::Completion);
        const auto helperSet = append(1, Command::Publish, Q, P, 0, EndpointPurpose::ConsumptionAcknowledgment, forward);
        const auto helperWait = append(1, Command::Acquire, Q, P, 0, EndpointPurpose::ConsumptionAcknowledgment, forward);
        c.rememberReturn(helperSet, helperWait);
        c.result.work.acknowledgments = 1;
        append(3, Command::Publish, Q, P, 1, EndpointPurpose::Completion);
        const auto actual = append(3, Command::Acquire, Q, P, 1, EndpointPurpose::Completion);
        const auto originalWord = c.ledger.word(1);
        c.current = c.control.graph.exit;
        c.activeComponent = c.control.component[c.current];
        c.activeOffset = 0;
        std::fill(c.finalized.begin(), c.finalized.end(), true);
        require(c.update(), c.cache.reason);
        for (Cut at = 0; at < c.control.graph.sites.size(); ++at) {
            c.current = at;
            c.registerSource();
        }
        c.current = c.control.graph.exit;
        auto compare = [&] {
            const auto reused = c.cache;
            const auto sources = c.result.sources;
            c.cache = {};
            require(c.replay(), c.cache.reason);
            identical(reused, c.cache);
            require(!sources.empty(), "erase/restore has no saved sources");
            for (Id i = 0; i < sources.size(); ++i)
                require(sources[i].snapshot == c.result.sources[i].snapshot &&
                        sources[i].version == c.result.sources[i].version,
                        "erase/restore source snapshot differs from cold replay");
            c.cache = reused;
            c.result.sources = sources;
        };
        compare();
        SelectedDecision decision;
        decision.endpoints = {actual};
        require(c.settleRearming(decision), c.cache.reason);
        require(!c.ledger.active(helperSet) && !c.ledger.active(helperWait) &&
                c.result.work.rearmingDischarged == 1, "fixture never discharged its helper");
        compare();
        // New earlier republication needs the removed helper BEFORE the later
        // actual return. update() must restore original IDs and word positions.
        append(2, Command::Publish, P, Q, 0, EndpointPurpose::Completion);
        append(2, Command::Acquire, P, Q, 0, EndpointPurpose::Completion);
        require(c.update(), c.cache.reason);
        require(c.result.work.rearmingRestored == 1 && c.result.work.rearmingDischarged == 0 &&
                c.requiredReturns.count(helperWait) && c.ledger.word(1) == originalWord,
                "earlier deadline did not restore and pin original helper identities/positions");
        compare();
        require(checkCausalFrontier(p, c.ledger.commands()).accepted, "restored plan not accepted");
        c.current = 3;
        c.activeComponent = c.control.component[c.current];
        require(c.advance(), c.cache.reason);
        compare();
        decision.endpoints.clear();
        decision.endpoints.push_back(helperWait);
        require(c.settleRearming(decision) && c.ledger.word(1) == originalWord,
                "required return was removed again");
    }
    // One edit, replayed with the reusable prefix kept and then entirely cold.
    // Returns how many components the incremental run actually kept.
    // Returns the components the incremental run kept, and whether the edit was
    // admissible at all.
    static std::pair<std::size_t, bool> compareEdit(const Program& program, Cut edit, Pipe pipe, bool contextual) {
        Constructor c(program);
        auto plan = c.run({});
        require(plan.success, plan.reason);
        require(c.needsContextualReplay == contextual, "unexpected replay path for this program");
        c.current = c.control.graph.exit;
        c.activeComponent = c.control.component[c.current];
        c.activeOffset = 0;
        c.ledger.append(edit, {Command::Barrier, pipe, pipe, 0}, EndpointPurpose::Fixed);
        const bool incremental = c.replay();
        auto reused = c.cache;
        c.cache = {};
        const bool cold = c.replay();
        // An inadmissible edit must be refused identically. Whether a candidate
        // is valid may not depend on how much of the prefix was kept.
        require(incremental == cold, "prefix reuse changed whether the edit is admissible");
        if (!cold) {
            require(reused.reason == c.cache.reason && reused.failureCut == c.cache.failureCut,
                    "incremental and cold replay disagree about the refusal");
            return {reused.reusedComponents, false};
        }
        identical(reused, c.cache);
        return {reused.reusedComponents, true};
    }
    static void compare(const Program& program, Cut edit, Pipe pipe) {
        require(compareEdit(program, edit, pipe, false).second, "edit was expected to hold");
    }
    static void siblingReuse(unsigned length, bool sequential, bool sharedWord,
                             unsigned siblings = 2, unsigned prefixLength = 0) {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        auto p = base(2);
        std::vector<Region> left, right;
        for (unsigned i = 0; i < 2 * length; ++i) {
            p.operations.push_back(op(P, {{i < length ? 0u : 1u, true, false}}));
            (i < length ? left : right).push_back(leaf(i));
        }
        Region a{Region::For, {{Region::Sequence, left}}, 0, true};
        Region b{Region::For, {{Region::Sequence, right}}, 1, true};
        p.body = sequential ? seq({a, b}) : Region{Region::Choice, {a, b}};
        for (unsigned sibling = 2; sibling < siblings; ++sibling) {
            std::vector<Region> child;
            for (unsigned i = 0; i < length; ++i) {
                child.push_back(leaf(p.operations.size()));
                p.operations.push_back(op(P, {{1, true, false}}));
            }
            p.body = {Region::Choice, {p.body, {Region::For, {{Region::Sequence, child}}, sibling, true}}};
        }
        std::vector<Region> prefix;
        for (unsigned i = 0; i < prefixLength; ++i) {
            prefix.push_back(leaf(p.operations.size()));
            p.operations.push_back(op(Q, {{0, true, false}}));
        }
        if (!prefix.empty()) p.body = seq({{Region::Sequence, prefix}, p.body});
        auto input = addStructuredBoundaryCuts(p);
        require(input.success, input.reason);
        p = std::move(input.program);
        Cut first = NoAnalysisId, second = NoAnalysisId;
        for (Cut i = 0; i < p.observed->sites.size(); ++i) {
            if (p.observed->sites[i].operation == 0) first = i;
            if (p.observed->sites[i].operation == length) second = i;
        }
        require(first != NoAnalysisId && second != NoAnalysisId, "sibling fixture lost payloads");
        if (sharedWord)
            p.observed->sites[second].observation = p.observed->sites[first].observation;
        SelectedOptions options;
        options.traceReplay = true;
        Constructor c(p, options);
        c.needsContextualReplay = true;
        require(c.run({}).success, "sibling fixture construction");
        c.current = c.control.graph.exit;
        c.activeComponent = c.control.component[c.current];
        c.ledger.clearChanges();

        auto compare = [&](bool expected, bool saves) {
            const auto original = c.cache;
            c.options.siblingReplayReuse = true;
            c.options.traceReplay = false;
            const auto prefixQueries = c.result.work.replayPrefixQueries;
            const bool untraced = c.replay();
            const auto normal = c.cache;
            if (original.contextualFixedPoint)
                require(c.result.work.replayPrefixQueries == prefixQueries,
                        "normal sibling update ran the comparison prefix");
            c.cache = original;
            c.options.traceReplay = true;
            const bool incremental = c.replay();
            const auto reused = c.cache;
            require(untraced == incremental, "trace comparison changed replay acceptance");
            if (incremental) identical(normal, reused);
            else require(normal.failureCut == reused.failureCut && normal.reason == reused.reason,
                         "trace comparison changed replay refusal");
            const auto trace = c.result.replayTraces.back();
            c.options.siblingReplayReuse = false;
            c.cache = original;
            const bool prefix = c.replay();
            const auto prefixWork = c.cache.evaluations;
            c.cache = {};
            const bool cold = c.replay();
            require(incremental == expected && prefix == cold && cold == incremental,
                    "sibling cache changed acceptance");
            if (cold) {
                identical(reused, c.cache);
                require(reused.evaluations <= prefixWork && (!saves || reused.evaluations < prefixWork),
                        "unchanged alternative did not save replay work");
                if (saves) {
                    const auto other = c.control.component[second];
                    require(trace.components[other].evaluations == 0 && trace.siblingComponents != 0,
                            "alternative component was still evaluated");
                }
            } else {
                require(reused.failureCut == c.cache.failureCut && reused.reason == c.cache.reason,
                        "sibling cache changed refusal evidence");
            }
            c.options.siblingReplayReuse = true;
            c.cache = reused;
            c.ledger.clearChanges();
        };
        const auto endpoint = c.ledger.append(first, {Command::Barrier, P, P, 0}, EndpointPurpose::Fixed);
        compare(true, !sequential && !sharedWord);
        c.ledger.erase(endpoint);
        compare(true, !sequential && !sharedWord);
        // Real matched event generations, including consumption/republication
        // evidence, rather than only payload/fence histories.
        for (const auto& command : std::vector<Command>{
                 {Command::Publish, P, Q, 0}, {Command::Acquire, P, Q, 0},
                 {Command::Publish, Q, P, 0}, {Command::Acquire, Q, P, 0}})
            c.ledger.append(first, command, EndpointPurpose::Fixed);
        compare(true, !sequential && !sharedWord);
        // A shared word that was NOT edited must also couple invalidation.
        if (sharedWord && !sequential) {
            Cut inside = NoAnalysisId;
            for (Cut i = 0; i < p.observed->sites.size(); ++i)
                if (p.observed->sites[i].operation == 1) inside = i;
            c.ledger.append(inside, {Command::Barrier, P, P, 0}, EndpointPurpose::Fixed);
            compare(true, false);
            require(c.result.replayTraces[c.result.replayTraces.size() - 3]
                        .components[c.control.component[second]].evaluations != 0,
                    "unchanged shared endpoint aggregate was reused across an edited component");
        }
        c.ledger.append(first, {Command::Barrier, P, P, 0}, EndpointPurpose::Fixed);
        c.ledger.append(second, {Command::Barrier, P, P, 0}, EndpointPurpose::Fixed);
        compare(true, false);
        require(c.result.replayTraces[c.result.replayTraces.size() - 3]
                    .components[c.control.component[second]].evaluations != 0,
                "an edit to both branches reused the second loop");
        const auto invalid = c.ledger.append(first, {Command::Acquire, P, Q, 0}, EndpointPurpose::Fixed);
        compare(false, false);
        c.ledger.erase(invalid);
        compare(true, false); // Failed cache must fall back to a fresh solution.
    }
    // Sweep every legal original cut as the edit position. On a refined loop
    // this covers words inside the body, words reached only across a backedge,
    // words shared by several original occurrences, and the surrounding words.
    static std::size_t compareEveryCut(
        const Program& program, Pipe pipe, bool contextual, const char* label) {
        std::size_t edits = 0, kept = 0, total = 0, refused = 0;
        for (Cut edit = 0; edit < commandCutCount(program); ++edit) {
            if (!legalCommandCut(program, edit)) continue;
            const auto outcome = compareEdit(program, edit, pipe, contextual);
            kept += outcome.first != 0;
            total += outcome.first;
            refused += !outcome.second;
            ++edits;
        }
        require(edits != 0, "no legal edit position in this program");
        std::cout << label << " edits=" << edits << " edits_with_reuse=" << kept
                  << " components_kept=" << total << " refused=" << refused << '\n';
        return refused;
    }
};
} // namespace mlir::pto::oahs::selected
namespace {
const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
o::Command post(o::Pipe a, o::Pipe b, unsigned key = 0) { return {o::Command::Publish, a, b, key}; }
o::Command wait(o::Pipe a, o::Pipe b, unsigned key = 0) { return {o::Command::Acquire, a, b, key}; }
void sourceTimeAndNeighbors()
{
    auto p = base(3);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(R, {{2, true, false}}), op(Q, {{1, true, false}})};
    o::Commands fixed(5);
    fixed[1] = {post(P, Q)};
    fixed[3] = {wait(P, Q)};
    auto result = accepted(p, fixed);
    require(result.commands[2].size() == 1 && result.commands[2][0].key == 1,
            "source-time occupied key must not borrow discovery-time emptiness");
    // The new early publication would precede a previously finalized fixed SET.
    // Its matching fixed WAIT is later than this consumer, so key 0 is blocked.
    p.operations = {op(P, {{0, false, true, true}}), op(R, {{2, true, false}}),
                    op(R, {{2, true, false}}), op(Q, {{0, true, false}})};
    fixed.assign(5, {});
    fixed[2] = {post(P, Q)};
    fixed[4] = {wait(P, Q)};
    result = accepted(p, fixed);
    require(result.commands[1].size() == 1 && result.commands[1][0].key == 1,
            "allocation must check the following already-selected key use");
    require(result.commands[2][0].key == 0 && result.commands[4][0].key == 0,
            "fixed endpoints must not be recolored");
    // A future fixed publication loses virgin-key legality after the selected
    // pair. Rechecking must refuse, not manufacture an acknowledgment/replan.
    p = base(2, 1);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true, true}})};
    fixed.assign(4, {});
    fixed[2] = {post(P, Q)};
    fixed[3] = {wait(P, Q)};
    result = o::constructSelectedPlan(p, fixed);
    require(!result.success && result.commands.empty() && result.certificate.cuts.empty(),
            "invalidated selected key certificate must reject atomically");
    require(result.failure == o::SelectedFailure::SelectedUpdate, "classify stale selected ledger");
}
void retirementAlternatives()
{
    auto p = base(1);
    p.operations = {op(P, {{0, true, false}})};
    p.target.barrierAll = true;
    p.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    o::CausalFrontier f(p);
    const auto start = f.initial();
    const auto retired = f.command(start, {o::Command::BarrierAll}, {1, 0});
    require(retired.applied, "qualified terminal retirement");
    const auto merged = f.join(retired.state, start);
    require(merged.applied && !f.issue(merged.state, 0).applied,
            "a maybe-retired interface cannot launch a new payload");
    require(!f.exit(merged.state).applied, "retirement must hold on every exiting path");
    auto full = f.command(start, post(P, Q), {0, 0});
    require(full.applied, "fresh publication");
    auto drain = f.command(full.state, {o::Command::BarrierAll}, {1, 0});
    require(drain.applied && !f.exit(drain.state).applied, "drain does not consume a live event");
}
void joinedPrecisionBoundary()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    o::CausalFrontier f(p);
    auto source = f.issue(f.initial(), 0);
    require(source.applied, "writer");
    auto transfer = [&](o::FrontierState state, o::Pipe a, o::Pipe b) {
        auto publication = f.command(state, post(a, b), {0, 0});
        require(publication.applied, "publication");
        auto acquisition = f.command(publication.state, wait(a, b), {1, 0});
        require(acquisition.applied, "acquisition");
        return acquisition.state;
    };
    auto left = transfer(source.state, P, Q);
    auto right = transfer(source.state, P, R);
    require(f.issue(transfer(left, R, Q), 1).applied && f.issue(transfer(right, R, Q), 1).applied,
            "both concrete continuations are safe");
    auto joined = f.join(left, right);
    require(joined.applied && !f.issue(transfer(joined.state, R, Q), 1).applied,
            "must-join may lose a disjunction resolved by a later transfer");
}
void replayReusesUnchangedPrefix()
{
    // Five unrelated writes precede the first consumer. Every selected update
    // must recompute only the sites from its earliest changed word to the
    // consumer; the unchanged prefix components keep their previous solution.
    auto p = base(6);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(P, {{2, false, true, true}}), op(P, {{3, false, true, true}}),
                    op(P, {{4, false, true, true}}), op(Q, {{4, true, false}}),
                    op(Q, {{3, true, false}}), op(R, {{2, true, false}})};
    const auto result = accepted(p);
    require(result.work.selectedUpdates >= 2 && result.updates.size() == result.work.selectedUpdates,
            "two independent transfers are selected");
    for (const auto& update : result.updates) {
        require(!update.changedCuts.empty(), "an update records its changed words");
        const auto first = *std::min_element(update.changedCuts.begin(), update.changedCuts.end());
        const auto last = *std::max_element(update.changedCuts.begin(), update.changedCuts.end());
        require(update.siteEvaluations <= last - first + 1,
                "replay recomputes only from the earliest changed word to the consumer");
    }
    require(result.work.replaySiteEvaluations < result.commands.size() * result.work.selectedUpdates,
            "reused prefix components are not re-evaluated");
}
void sharedObservationReplay()
{
    auto p = base(1, 1);
    p.operations = {op(P, {{0, false, true, true}})};
    o::ObservedControl graph;
    graph.qualification = "original shared boundary observations";
    graph.sites.resize(4);
    graph.observations = {{10, {}, true}, {11, {}, true}, {12, {}, true}};
    // Execution order differs from numeric order. One emitted word is visible
    // at both sites 1 and 0, including the earlier noncanonical occurrence.
    graph.entry = 2;
    graph.exit = 3;
    graph.sites[2] = {0, 0, {1}, {}, 0};
    graph.sites[1] = {o::NoControlId, 1, {0}, {}, 0};
    graph.sites[0] = {o::NoControlId, 1, {3}, {}, 0};
    graph.sites[3] = {o::NoControlId, 2, {}, {}, 0};
    p.observed = graph;
    o::selected::ReplayTestAccess::compare(p, 0, P);
    // A canonical member can be unreachable while another member is live.
    p.observed->sites[1].successors = {3};
    o::selected::ReplayTestAccess::compare(p, 0, P);
    p = base(1, 1);
    p.operations = {op(P, {{0, true, false}}), op(P, {{0, true, false}})};
    p.body = seq({{o::Region::For, {leaf(0)}, 0, true}, leaf(1)});
    o::selected::ReplayTestAccess::compare(p, 0, P);
    // The same sweep on the component path, whose boundary now reads the
    // precomputed occurrence and span index rather than rescanning the sites.
    auto shared = base(3, 2);
    shared.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                         op(P, {{1, false, true, true}}), op(R, {{1, true, false}}),
                         op(P, {{2, false, true, true}}), op(Q, {{2, true, false}})};
    shared.body = seq({leaf(0), {o::Region::For, {seq({leaf(1), leaf(2)})}, 0, true}, leaf(3),
                       {o::Region::Choice, {leaf(4), leaf(5)}}});
    auto bounded = o::addStructuredBoundaryCuts(shared);
    require(bounded.success, bounded.reason);
    o::selected::ReplayTestAccess::compareEveryCut(bounded.program, P, false, "component-path");
}
void contextualPrefixReuse()
{
    // A qualified periodic loop takes the contextual replay path, which solved
    // the whole original graph per edit and kept nothing. Every legal original
    // cut is used as an edit position, so the sweep covers words inside the
    // refined body, words reached only across the backedge, words shared by
    // several original occurrences, and the surrounding words.
    // A qualified loop with independent work before and after it, so the
    // condensation really has components on both sides of the body.
    auto body = base(2, 8);
    body.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}, {1, false, true}}),
                       op(Q, {{0, true, false}})};
    auto input = o::makePeriodicLoop(body, 1, {});
    require(input.success, input.reason);
    auto& p = input.program;
    auto& q = *p.observed;
    const auto preOp = p.operations.size();
    p.operations.push_back(op(R, {{0, false, true}}));
    const auto postOp = p.operations.size();
    p.operations.push_back(op(o::Pipe::MTE1, {{0, true, false}, {1, false, true}}));
    auto node = [&](std::size_t operation) {
        const auto site = q.sites.size();
        const auto observation = q.observations.size();
        q.observations.push_back({1000 + site, {}, true});
        q.sites.push_back({operation, observation, {}, {}, 0});
        return site;
    };
    const auto oldEntry = q.entry, oldExit = q.exit;
    q.entry = node(preOp);
    const auto post = node(postOp);
    q.exit = node(o::NoControlId);
    q.sites[q.entry].successors = {oldEntry};
    q.sites[oldExit].successors = {post};
    q.sites[post].successors = {q.exit};
    p.target.barrierAll = true;
    p.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    const auto plan = accepted(p);
    require(plan.work.recurringChannels != 0, "this program must qualify recurrence");
    require(plan.work.contextualReplays != 0, "this program must use contextual replay");
    o::SelectedOptions tracedOptions;
    tracedOptions.traceReplay = true;
    const auto traced = o::constructSelectedPlan(p, {}, tracedOptions);
    require(traced.success && traced.commands.size() == plan.commands.size(),
            "replay attribution changed construction acceptance");
    for (o::Cut cut = 0; cut < plan.commands.size(); ++cut) {
        const auto& a = plan.commands[cut];
        const auto& b = traced.commands[cut];
        require(a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                [](const o::Command& x, const o::Command& y) {
                    return x.kind == y.kind && x.source == y.source && x.observer == y.observer && x.key == y.key;
                }), "replay attribution changed a selected command word");
    }
    require(plan.replayTraces.empty() && traced.replayTraces.size() == traced.work.contextualReplays,
            "replay traces must be opt-in and include every contextual solve");
    uint64_t evaluations = 0;
    for (const auto& trace : traced.replayTraces) {
        uint64_t unique = 0, visits = 0;
        for (const auto& component : trace.components) {
            unique += component.uniqueSites;
            visits += component.evaluations;
            require(component.uniqueSites <= component.sites && component.uniqueSites <= component.evaluations,
                    "component replay attribution is inconsistent");
        }
        require(unique == trace.uniqueSites && visits == trace.evaluations &&
                trace.resume <= trace.fixedBoundary && trace.fixedBoundary <= trace.changedBoundary &&
                trace.changedBoundary <= trace.activeComponent && trace.changedJoins <= trace.successorJoins,
                "contextual replay attribution is inconsistent");
        evaluations += visits;
    }
    require(evaluations == traced.work.replaySiteEvaluations && evaluations == plan.work.replaySiteEvaluations,
            "trace totals do not reconcile with replay work");
    o::selected::ReplayTestAccess::compareEveryCut(p, P, true, "contextual-named-fence");
    // An endpoint on an engine without a named fence is an invalid candidate. It
    // must be refused identically whether or not a prefix was kept.
    auto unavailable = p;
    unavailable.target.barriers[unsigned(R)] = false;
    const auto refusals = o::selected::ReplayTestAccess::compareEveryCut(
        unavailable, R, true, "contextual-invalid");
    require(refusals != 0, "the invalid-candidate sweep never actually refused an edit");
}
void recurringRoleIsolation()
{
    auto p = base(2, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(Q, {{1, false, true, true}}), op(P, {{1, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0), leaf(1), leaf(2), leaf(3)})}, 0, true};
    const auto result = accepted(p);
    require(result.work.commonCutTransfers != 0, "generic recurrence uses real local channels");
    for (const auto& source : result.sources) {
        require(source.snapshot.reachable(), "exported source is reconstructed under final ledger");
    }
}
} // namespace
int main()
{
    o::selected::ReplayTestAccess::dischargeRestore();
    for (unsigned words : {32u, 128u, 512u})
        o::selected::ReplayTestAccess::prefixComparisonCost(words);
    sourceTimeAndNeighbors();
    retirementAlternatives();
    joinedPrecisionBoundary();
    replayReusesUnchangedPrefix();
    sharedObservationReplay();
    contextualPrefixReuse();
    for (unsigned length : {2u, 8u, 32u}) {
        o::selected::ReplayTestAccess::siblingReuse(length, false, false);
        o::selected::ReplayTestAccess::siblingReuse(length, true, false);
        o::selected::ReplayTestAccess::siblingReuse(length, false, true);
    }
    for (unsigned siblings : {4u, 8u})
        for (unsigned prefix : {0u, 16u})
            o::selected::ReplayTestAccess::siblingReuse(8, false, false, siblings, prefix);
    for (unsigned length : {4u, 16u, 64u}) {
        auto p = base(2);
        std::vector<o::Region> body;
        for (unsigned i = 0; i < length; ++i) {
            p.operations.push_back(op(P, {{i % 2, true, true}}));
            body.push_back(leaf(i));
        }
        // Branches inside nested loops exercise joins and inner-header seeds.
        p.body = {o::Region::For, {{o::Region::Choice,
            {{o::Region::For, {{o::Region::Sequence, body}}, 0, true},
             {o::Region::Sequence, {}}}}}, 0, true};
        auto bounded = o::addStructuredBoundaryCuts(p);
        require(bounded.success, bounded.reason);
        o::selected::ReplayTestAccess::compareAdvances(bounded.program, P);
        for (auto& operation : bounded.program.operations)
            for (auto& access : operation.accesses) access.write = false;
        o::selected::ReplayTestAccess::compareAdvances(bounded.program, P, false);
        if (length == 16) {
            o::selected::ReplayTestAccess::compareAdvances(bounded.program, P, false, true);
            o::selected::ReplayTestAccess::compareAdvances(bounded.program, P, false, false, true);
        }
    }
    recurringRoleIsolation();
    std::cout << "selected-ledger update, boundary and refusal tests passed\n";
}
