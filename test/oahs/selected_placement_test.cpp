// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include <array>
#include <functional>
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
using namespace selected_test;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void joinedConsumptionReturn() {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        auto input = base(3, 1);
        input.operations = {op(P, {{0,false,true}}), op(Q, {{0,true,false}}),
            op(Q, {{0,true,false}}), op(P, {{1,false,true}}),
            op(P, {{2,false,true}}), op(Q, {{1,true,false}})};
        input.body = seq({leaf(0), {Region::Choice, {leaf(1),leaf(2)}}, leaf(3), leaf(4), leaf(5)});
        auto imported = addStructuredBoundaryCuts(input);
        require(imported.success, imported.reason);
        const auto& p = imported.program;
        auto cut = [&](unsigned op) {
            for (Cut i = 0; i < p.observed->sites.size(); ++i)
                if (p.observed->sites[i].operation == op) return i;
            return NoAnalysisId;
        };
        Commands fixed(commandCutCount(p));
        const auto firstPublication = p.observed->sites[cut(0)].successors.front();
        fixed[firstPublication] = {{Command::Publish,P,Q,0}};
        fixed[cut(1)] = fixed[cut(2)] = {{Command::Acquire,P,Q,0}};
        SelectedOptions options; options.finalHelperTrials = false;
        const auto plan = constructSelectedPlan(p,fixed,options);
        require(plan.success, "joined consumption construction: " + plan.reason);
        require(plan.work.joinedAcknowledgments == 1 && plan.work.acknowledgmentChecks == 1,
                "alternative consumptions did not select one checked return");
        require(checkCausalFrontier(p,plan.commands).accepted, "joined return failed cold validation");
        require(plan.decisions.back().publication == cut(4), "joined repair widened early source past unrelated load");
        std::function<void(Cut,Program,Commands,std::vector<Command>)> walk;
        walk = [&](Cut at, Program flat, Commands words, std::vector<Command> pending) {
            pending.insert(pending.end(),plan.commands[at].begin(),plan.commands[at].end());
            const auto& site = p.observed->sites[at];
            if (site.operation != NoAnalysisId) {
                flat.operations.push_back(p.operations[site.operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            if (at == p.observed->exit) {
                words.push_back(std::move(pending));
                std::vector<unsigned> trace(flat.operations.size());
                std::iota(trace.begin(),trace.end(),0);
                require(bool(oahs_oracle::graph(flat,words,trace)), "joined return failed independent oracle");
            } else for (auto next : site.successors) walk(next,flat,words,pending);
        };
        auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
        walk(p.observed->entry,flat,{},{});
        auto missing = plan.commands;
        for (auto& word : missing)
            word.erase(std::remove_if(word.begin(),word.end(),[&](const Command& c) {
                return c.source == Q && c.observer == P;
            }),word.end());
        require(!checkCausalFrontier(p,missing).accepted, "missing joined consumption support was accepted");
        auto unavailable = p;
        unavailable.target.keys[unsigned(Q)][unsigned(P)].clear();
        const auto refused = constructSelectedPlan(unavailable,fixed,options);
        require(!refused.success && refused.commands.empty(), "joined return invented an unavailable reverse key");
        // A branch which does not consume leaves a maybe-full key: a return
        // must not turn this into empty occupancy.
        auto unconsumed = fixed;
        unconsumed[cut(2)].clear();
        require(!constructSelectedPlan(p,unconsumed,options).success,
                "joined return accepted an unconsumed branch");
    }
    static void reusedGapCertificates() {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        // All states are obtained by production replay of actual endpoints.
        // Modes: earlier return, no return, return in the source word,
        // next generation in the source word, next generation after the gap.
        for (unsigned mode = 0; mode < 5; ++mode) {
            auto p = base(1, 1);
            for (unsigned i = 0; i < 8; ++i) p.operations.push_back(op(Q, {{0,true,false}}));
            Commands words(commandCutCount(p));
            words[0] = {{Command::Publish,Q,P,0}};
            words[1] = {{Command::Acquire,Q,P,0}};
            if (mode != 1) {
                words[2] = {{Command::Publish,P,Q,0}};
                words[mode == 2 ? 4 : 3] = {{Command::Acquire,P,Q,0}};
            }
            if (mode >= 3) {
                words[mode == 3 ? 4 : 5] = {{Command::Publish,Q,P,0}};
                words[6] = {{Command::Acquire,Q,P,0}};
            }
            Constructor c(p);
            std::string reason;
            require(c.ledger.initialize(words, reason), reason);
            c.current = 7; c.activeComponent = c.control.component[c.current];
            require(c.replay(), "gap certificate fixture replay failed: " + c.result.reason);
            Id key = NoAnalysisId;
            for (Id i = 0; i < c.frontier.keys().size(); ++i) {
                const auto& candidate = c.frontier.keys()[i];
                if (candidate.source == Q && candidate.observer == P && candidate.key == 0) key = i;
            }
            require(key != NoAnalysisId, "missing gap fixture key");
            require(c.canPublish(c.cache.cuts[4].incoming,key) == (mode == 0 || mode >= 3),
                    "gap fixture did not distinguish occupancy from rearming");
            if (mode == 2) require(c.canPublish(c.cache.cuts[4].before,key),
                    "source word did not supply the deliberately too-late credit");
            require((c.reusableAtStart(4,Q,P) != NoAnalysisId) == (mode == 0),
                    "gap accepted missing/late rearming or a neighboring selected use");
        }
    }
    static void proposalWordOrder() {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        auto p = base(1, 1);
        for (unsigned i = 0; i < 6; ++i) p.operations.push_back(op(P, {{0, true, false}}));
        RecurringRequirement forward, reverse;
        forward.cell = reverse.cell = 0;
        forward.cells = reverse.cells = {0};
        forward.source = P; forward.observer = Q;
        reverse.source = Q; reverse.observer = P;
        forward.publications = {0, 3}; forward.acquisitions = {1, 4};
        reverse.publications = {1, 4}; reverse.acquisitions = {2, 5};
        Commands ordered(commandCutCount(p));
        ordered[0] = {{Command::Publish, P, Q, 0}};
        ordered[1] = {{Command::Acquire, P, Q, 0}, {Command::Publish, Q, P, 0}};
        ordered[2] = {{Command::Acquire, Q, P, 0}};
        ordered[3] = ordered[0]; ordered[4] = ordered[1]; ordered[5] = ordered[2];
        require(checkCausalFrontier(p, ordered).accepted &&
                bool(oahs_oracle::graph(p, ordered, {0,1,2,3,4,5})),
                "reciprocal request-order witness must be valid");
        auto reordered = ordered;
        std::reverse(reordered[1].begin(), reordered[1].end());
        std::reverse(reordered[4].begin(), reordered[4].end());
        require(!checkCausalFrontier(p, reordered).accepted &&
                !bool(oahs_oracle::graph(p, reordered, {0,1,2,3,4,5})),
                "publication-first witness must lose rearming");
        for (bool trials : {false, true}) {
            SelectedOptions options; options.recurringOmissionTrials = trials;
            Constructor c(p, options);
            std::string reason;
            require(c.ledger.initialize({}, reason), reason);
            const auto version = c.ledger.version();
            require(c.recurring({forward, reverse}), "ordered rejection must remain optional");
            require(c.result.work.rejectedProtocolProposals == 1 &&
                    c.result.work.recurringTrials == 0,
                    "mandatory admission checked request order instead of committed order");
            require(c.ledger.version() == version && c.ledger.records().empty() &&
                    c.recurringKeys.empty() && c.result.channels.empty() && !c.needsContextualReplay,
                    "reordered proposal rejection leaked state");
            require(c.run({}).success, "ordinary fallback after word-order rejection failed");
        }
        // Moving the returns to distinct later cuts yields valid canonical
        // words. Commitment must retain that exact validated endpoint order.
        reverse.publications = {2, 5};
        ordered[1].pop_back(); ordered[4].pop_back();
        ordered[2].insert(ordered[2].begin(), {Command::Publish, Q, P, 0});
        ordered[5].insert(ordered[5].begin(), {Command::Publish, Q, P, 0});
        Constructor c(p);
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        require(c.recurring({forward, reverse}) && c.result.channels.size() == 2,
                "valid canonical proposal was declined");
        const auto committed = c.ledger.commands();
        for (Cut cut = 0; cut < ordered.size(); ++cut)
            require(committed[cut].size() == ordered[cut].size() &&
                    std::equal(committed[cut].begin(), committed[cut].end(), ordered[cut].begin(),
                        [](const Command& a, const Command& b) {
                            return a.kind == b.kind && a.source == b.source &&
                                   a.observer == b.observer && a.key == b.key;
                        }), "committed proposal differs from validated words");
        require(checkCausalFrontier(p, committed).accepted &&
                bool(oahs_oracle::graph(p, committed, {0,1,2,3,4,5})),
                "accepted canonical proposal lost protocol validity");
    }
    static void proposalOmissionWords() {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE3;
        auto p = base(1, 1);
        for (unsigned i = 0; i < 4; ++i) p.operations.push_back(op(P, {{0, true, false}}));
        RecurringRequirement direct, first, second;
        direct.source = first.source = P; direct.observer = second.observer = Q;
        first.observer = second.source = R;
        direct.publications = first.publications = {0};
        direct.acquisitions = second.acquisitions = {3};
        first.acquisitions = {1}; second.publications = {2};
        Constructor c(p);
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        require(c.recurring({direct, first, second}) && c.result.channels.size() == 2 &&
                c.result.work.recurringTrials == 1 && c.result.work.redundantRecurringChannels == 1,
                "fixture did not accept the indirect-route omission");
        const auto words = c.ledger.commands();
        require(words[0].size() == 1 && words[0][0].observer == R &&
                words[1].size() == 1 && words[1][0].kind == Command::Acquire &&
                words[2].size() == 1 && words[2][0].source == R &&
                words[3].size() == 1 && words[3][0].source == R,
                "omission committed stale endpoints or channel identities");
        for (const auto& endpoint : c.ledger.records()) {
            const auto& channel = c.result.channels.at(endpoint.request);
            require(channel.source == endpoint.command.source && channel.observer == endpoint.command.observer,
                    "omission failed to remap retained request to its channel");
        }
        require(checkCausalFrontier(p, words).accepted && bool(oahs_oracle::graph(p, words, {0,1,2,3})),
                "accepted omission did not commit a valid checked plan");
    }
    static void uncoveredProducerProposal() {
        auto p = base(2);
        p.operations = {op(Pipe::MTE2, {{0, false, true}}), op(Pipe::MTE2, {{0, false, true}}),
                        op(Pipe::V, {{1, false, true}}), op(Pipe::MTE2, {{1, true, false}})};
        Constructor c(p);
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        const auto version = c.ledger.version();
        RecurringRequirement supported;
        supported.cell = 1; supported.cells = {1}; supported.source = Pipe::V; supported.observer = Pipe::MTE2;
        supported.publications = {3}; supported.acquisitions = {3}; supported.qualifiedCycle = true;
        supported.repairFreeProducers.insert(Pipe::MTE2);
        require(c.recurring({supported}), "uncovered producer proposal escaped optional fallback");
        require(c.result.work.rejectedSupportProposals == 1 && c.result.work.rejectedProtocolProposals == 0,
                "valid event protocol hid an uncovered producer repair");
        require(c.ledger.version() == version && c.ledger.records().empty() &&
                c.recurringKeys.empty() && c.result.channels.empty() && !c.needsContextualReplay,
                "rejected producer support leaked committed state");
        require(c.run({}).success, "ordinary construction failed after support rejection");
    }
    static void invalidProposal() {
        auto p = base(1);
        p.operations = {op(Pipe::MTE2, {{0, false, true}}), op(Pipe::V, {{0, true, false}})};
        Constructor c(p);
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        const auto version = c.ledger.version();
        RecurringRequirement bad;
        bad.cell = 0; bad.cells = {0}; bad.source = Pipe::MTE2; bad.observer = Pipe::V;
        bad.publications = {1}; bad.qualifiedCycle = true;
        require(c.recurring({bad}), "optional rejection escaped as constructor failure");
        require(c.ledger.version() == version && c.ledger.records().empty() &&
                c.recurringKeys.empty() && c.result.channels.empty() && !c.needsContextualReplay,
                "invalid optional proposal leaked committed state");
        require(c.result.work.rejectedProtocolProposals == 1, "invalid protocol not rejected");
        require(c.run({}).success, "ordinary construction failed after optional rejection");
    }
};
}
namespace {
const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
o::Program cohort() {
    auto p = base(3, 2);
    // Only the ordinary two-engine vocabulary: no relay hides starvation.
    for (unsigned a = 0; a < o::PipeCount; ++a) for (unsigned b = 0; b < o::PipeCount; ++b)
        if (!((a == unsigned(P) && b == unsigned(Q)) || (a == unsigned(Q) && b == unsigned(P))))
            p.target.keys[a][b].clear();
    p.operations = {op(P, {{2,false,true}}), op(Q, {{2,true,false}}),
        op(P, {{0,false,true}}), op(Q, {{0,true,false}}),
        op(P, {{1,false,true}}), op(Q, {{1,true,false}})};
    p.body = seq({leaf(0), leaf(1), {o::Region::For,{seq({leaf(2),leaf(3),leaf(4),leaf(5)})},0,true}});
    auto imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    return imported.program;
}
void starvation() {
    auto p = cohort();
    o::SelectedOptions ordinary; ordinary.recurring = false;
    auto reference = o::constructSelectedPlan(p, {}, ordinary);
    require(reference.success, "ordinary reference must work: " + reference.reason);
    auto plan = o::constructSelectedPlan(p);
    std::cout << "cohort channels=" << plan.channels.size() << " rejected=" << plan.work.rejectedResourceProposals << " success=" << plan.success << '\n';
    require(plan.success, "optional cohort starved ordinary X: " + plan.reason);
    require(plan.work.rejectedResourceProposals == 1 && plan.channels.empty(), "exact-fit cohort was not declined");
}
void deferredRequiredReturn()
{
    auto p = base(3, 1);
    p.operations = {op(Q, {{2, true, false}}), op(P, {{0, false, true}}),
                    op(Q, {{0, true, false}}), op(P, {{0, false, true}}),
                    op(P, {{1, false, true}}), op(Q, {{1, true, false}})};
    p.body = seq({leaf(0), {o::Region::Choice, {leaf(1), seq({})}},
                  leaf(2), leaf(3), leaf(4), leaf(5)});
    o::SelectedOptions options;
    options.finalHelperTrials = false;
    const auto baseline = o::constructSelectedPlan(p, {}, options);
    options.deferredAcyclicAcknowledgments = true;
    const auto plan = o::constructSelectedPlan(p, {}, options);
    require(baseline.success && plan.success, "required-return deferral: " + plan.reason);
    require(!plan.rearming.empty(), "deferred consumption has no explicit obligation");
    const auto& obligation = plan.rearming.front();
    require(obligation.key.source == P && obligation.key.observer == Q &&
            obligation.acquisition < plan.ledger.size(), "obligation lost its physical key or receipt");
    require(std::find(obligation.returnDeadlines.begin(), obligation.returnDeadlines.end(), 3) !=
            obligation.returnDeadlines.end(), "obligation lost the physical storage-return candidate");
    require(!obligation.reusePublications.empty(), "actual rearmed reuse was not linked to consumption");
    for (const auto& endpoint : plan.ledger) {
        require(endpoint.purpose != o::EndpointPurpose::ConsumptionAcknowledgment ||
                endpoint.acknowledges != obligation.acquisition,
                "required storage return still received a private acknowledgment");
    }
    for (const auto& visits : {std::vector<unsigned>{0, 1, 2, 3, 4, 5},
                             std::vector<unsigned>{0, 2, 3, 4, 5}}) {
        oahs_oracle::PayloadOrder before, after;
        require(bool(oahs_oracle::graph(p, baseline.commands, visits, {}, nullptr, nullptr, &before)) &&
                bool(oahs_oracle::graph(p, plan.commands, visits, {}, nullptr, nullptr, &after)),
                "deferred return failed independent order checks");
        require(std::includes(before.begin(), before.end(), after.begin(), after.end()),
                "required return introduced payload ordering");
    }
}
void deferredSharedReceipt(bool requiredReturn, bool skippedReceipt = false, bool downstreamChoice = false)
{
    auto p = base(2, 2);
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true}}), op(P, {}), op(Q, {{1, true, false}}), op(Q, {})};
    if (requiredReturn) {
        p.operations[2] = op(P, {{0, false, true}});
        p.operations[3] = op(P, {{1, false, true}});
    }
    o::ObservedControl graph;
    graph.qualification = "mutually exclusive receipts with a common continuation";
    graph.entry = 12;
    graph.exit = 11;
    graph.sites.resize(13);
    for (unsigned at = 0; at < graph.sites.size(); ++at) {
        graph.observations.push_back({at, {}, true});
        graph.sites[at].observation = at;
        if (at != graph.exit) {
            graph.sites[at].successors = {at + 1};
        }
    }
    graph.sites[0].operation = 0;
    graph.sites[2].successors = {3, 4};
    graph.sites[3].operation = graph.sites[4].operation = 1;
    graph.sites[3].successors = {5};
    graph.sites[4].observation = 3;
    graph.sites[6].operation = 2;
    graph.sites[8].operation = 3;
    graph.sites[10].operation = 4;
    graph.sites[12].operation = 5;
    graph.sites[12].successors = {0};
    if (skippedReceipt) {
        graph.sites[4].observation = 4;
        graph.sites[4].operation = o::NoAnalysisId;
    }
    if (downstreamChoice) {
        graph.sites[5].successors = {6, 7};
    }
    p.observed = graph;
    o::SelectedOptions options;
    options.finalHelperTrials = false;
    const auto baseline = o::constructSelectedPlan(p, {}, options);
    options.deferredAcyclicAcknowledgments = true;
    const auto plan = o::constructSelectedPlan(p, {}, options);
    require(baseline.success && plan.success, "shared receipt construction: " + baseline.reason +
            " at " + std::to_string(baseline.cut) + " / " + plan.reason + " at " + std::to_string(plan.cut));
    const auto debt = std::find_if(plan.rearming.begin(), plan.rearming.end(), [&](const auto& obligation) {
        return plan.ledger.at(obligation.acquisition).cut == 3;
    });
    const bool qualified = !skippedReceipt;
    require((debt != plan.rearming.end()) == qualified, "shared receipt deferral ignored participation");
    if (qualified) {
        require(!debt->reusePublications.empty(), "shared receipt lost its actual reuse deadline");
        const bool hasPrivateReturn = std::any_of(plan.ledger.begin(), plan.ledger.end(), [&](const auto& endpoint) {
            return endpoint.purpose == o::EndpointPurpose::ConsumptionAcknowledgment &&
                endpoint.acknowledges == debt->acquisition;
        });
        require(hasPrivateReturn != requiredReturn,
                "shared receipt did not select the required return before a helper");
    }
    require(o::checkCausalFrontier(p, plan.commands).accepted, "shared receipt cold validation failed");
    std::function<void(o::Cut, std::vector<unsigned>)> walk;
    walk = [&](o::Cut at, std::vector<unsigned> path) {
        path.push_back(at);
        if (at != graph.exit) {
            for (auto next : graph.sites[at].successors) {
                walk(next, path);
            }
            return;
        }
        auto flat = p;
        flat.observed.reset();
        flat.body = {};
        flat.operations.clear();
        for (auto at : path) {
            if (graph.sites[at].operation != o::NoAnalysisId) {
                flat.operations.push_back(p.operations[graph.sites[at].operation]);
            }
        }
        auto ordering = [&](const o::Commands& commands) {
            o::Commands words;
            std::vector<o::Command> pending;
            for (auto at : path) {
                pending.insert(pending.end(), commands[at].begin(), commands[at].end());
                if (graph.sites[at].operation != o::NoAnalysisId) {
                    words.push_back(std::move(pending));
                    pending.clear();
                }
            }
            words.push_back(std::move(pending));
            std::vector<unsigned> visits(flat.operations.size());
            std::iota(visits.begin(), visits.end(), 0);
            oahs_oracle::PayloadOrder order;
            require(bool(oahs_oracle::graph(flat, words, visits, {}, nullptr, nullptr, &order)),
                    "shared receipt independent oracle rejected plan");
            return order;
        };
        const auto before = ordering(baseline.commands), after = ordering(plan.commands);
        require(std::includes(before.begin(), before.end(), after.begin(), after.end()),
                "shared receipt introduced payload ordering");
        if (qualified && !requiredReturn && !downstreamChoice) {
            require(before != after, "shared receipt helper did not release unrelated producer work");
        } else if (!qualified) {
            require(before == after, "unsupported shared receipt changed conservative ordering");
        }
    };
    walk(graph.entry, {});
}
void deferredAcknowledgment() {
    auto p = base(3, 2);
    p.operations = {op(Q,{{2,true,false}}), op(P,{{0,false,true}}),
                    op(Q,{{0,true,false}}), op(P,{{1,false,true}})};
    p.body = seq({leaf(0), {o::Region::Choice,{leaf(1),seq({})}}, leaf(2),leaf(3)});
    o::SelectedOptions before; before.finalHelperTrials = false;
    auto baseline = o::constructSelectedPlan(p,{},before);
    auto after = before; after.deferredAcyclicAcknowledgments = true;
    auto plan = o::constructSelectedPlan(p,{},after);
    require(baseline.success && plan.success, "deferred acyclic fixture construction failed: " + plan.reason);
    require(plan.work.deferredAcknowledgments != 0 && plan.work.acknowledgments < baseline.work.acknowledgments,
            "later payload without key reuse still forces acknowledgment");
    for (auto visits : {std::vector<unsigned>{0,1,2,3},std::vector<unsigned>{0,2,3}})
        require(bool(oahs_oracle::graph(p,plan.commands,visits,{{0,unsigned(visits.size()-1)}})),
                "deferred acknowledgment lost safety or unrelated-work independence");
    // A later actual reuse still needs its consumption path. No return means
    // either another virgin key, a real helper, or a refusal, never invented credit.
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    p.operations.push_back(op(P,{{1,false,true}}));
    p.operations.push_back(op(Q,{{1,true,false}}));
    p.body.children.push_back(leaf(4)); p.body.children.push_back(leaf(5));
    plan = o::constructSelectedPlan(p,{},after);
    require(plan.success, "real later reuse lost its repair: " + plan.reason);
    require(bool(oahs_oracle::graph(p,plan.commands,{0,1,2,3,4,5})), "real reuse lacks consumption evidence");
    require(!plan.rearming.empty() && !plan.rearming.front().reusePublications.empty(),
            "helper-backed reuse lost its deferred obligation");

    // The common consumption is exactly once. A helper demanded by a later
    // conditional publication must also be consumed on the skipped path.
    p.body = seq({leaf(0), {o::Region::Choice,{leaf(1),seq({})}}, leaf(2),leaf(3),
                  {o::Region::Choice,{seq({leaf(4),leaf(5)}),seq({})}}});
    baseline = o::constructSelectedPlan(p,{},before);
    plan = o::constructSelectedPlan(p,{},after);
    require(baseline.success, "closed conditional-reuse baseline failed: " + baseline.reason);
    require(plan.success, "conditional future key reuse lost its repair: " + plan.reason);
    require(plan.work.deferredAcknowledgments != 0 && !plan.rearming.empty(),
            "conditional repair did not retain its unresolved obligation");
    require(o::checkCausalFrontier(p,plan.commands).accepted, "conditional-reuse cold check failed");
    for (auto visits : {std::vector<unsigned>{0,1,2,3,4,5}, std::vector<unsigned>{0,2,3,4,5},
                        std::vector<unsigned>{0,1,2,3}, std::vector<unsigned>{0,2,3}}) {
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        require(bool(oahs_oracle::graph(p, baseline.commands, visits, {}, nullptr, nullptr, &oldOrder)),
                "closed conditional-reuse oracle failed");
        require(bool(oahs_oracle::graph(p, plan.commands, visits, {}, nullptr, nullptr, &newOrder)),
                "conditional reuse lacks real rearming");
        require(std::includes(oldOrder.begin(), oldOrder.end(), newOrder.begin(), newOrder.end()),
                "conditional rearming added payload ordering");
    }
    auto broken = plan.commands;
    for (auto& word : broken) word.erase(std::remove_if(word.begin(),word.end(),[&](const auto& command) {
        return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
               command.source == Q && command.observer == P;
    }),word.end());
    require(!o::checkCausalFrontier(p,broken).accepted &&
            !bool(oahs_oracle::graph(p,broken,{0,1,2,3,4,5})),
            "conditional reuse must reject missing consumption support");
    after.finalHelperTrials = true;
    plan = o::constructSelectedPlan(p,{},after);
    require(plan.success, "conditional-reuse final helper trials failed: " + plan.reason);
    for (auto visits : {std::vector<unsigned>{0,1,2,3,4,5}, std::vector<unsigned>{0,2,3,4,5},
                        std::vector<unsigned>{0,1,2,3}, std::vector<unsigned>{0,2,3}})
        require(bool(oahs_oracle::graph(p,plan.commands,visits)), "helper pruning lost conditional rearming");
}
void wordGapBaseline() {
    auto p = base(3, 4);
    p.operations = {op(R,{{1,true,false}}),op(P,{{0,false,true}}),
        op(Q,{{0,true,false},{2,false,true}}),op(Q,{{1,false,true}}),op(P,{{0,false,true}})};
    auto baseline = o::constructSelectedPlan(p);
    require(baseline.success,baseline.reason);
    require(bool(oahs_oracle::graph(p,baseline.commands,{0,1,2,3,4})), "baseline must be safe");
    o::SelectedOptions options; options.sourceGaps = true;
    auto candidate = o::constructSelectedPlan(p, {}, options);
    require(candidate.success, candidate.reason);
    require(bool(oahs_oracle::graph(p,candidate.commands,{0,1,2,3,4},{{0,4}})),
            "exact source gap still imports unrelated R completion");
    require(candidate.work.gapPublications != 0, "source-gap fixture did not select a gap");
    std::cout << "word_gap_baseline_forbidden=" << !bool(oahs_oracle::graph(p,baseline.commands,{0,1,2,3,4},{{0,4}})) << '\n';
    // An outward publication at the shared cut must retain its own source
    // meaning; gap insertion may not export the unrelated incoming wait.
    p.operations.push_back(op(o::Pipe::MTE1,{{2,true,false}}));
    o::Commands fixed(o::commandCutCount(p));
    fixed[3] = {{o::Command::Publish,Q,o::Pipe::MTE1,0}};
    fixed[5] = {{o::Command::Acquire,Q,o::Pipe::MTE1,0}};
    auto outward = o::constructSelectedPlan(p,fixed,options);
    require(outward.success,outward.reason);
    require(bool(oahs_oracle::graph(p,outward.commands,{0,1,2,3,4,5},{{0,4},{0,5}})),
            "source gap broadened the outward publication");
    // Removing real readiness must still fail, even in a gap-aware plan.
    auto broken = outward.commands;
    for (auto& word : broken) word.erase(std::remove_if(word.begin(),word.end(),[&](const auto& command) {
        return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
               command.source == P && command.observer == Q;
    }),word.end());
    require(!o::checkCausalFrontier(p,broken).accepted, "gap admission invented missing readiness");

}
void reusedSourceGap() {
    // Q->P has one key. Its first use transfers W; the later X readiness
    // (P->Q) can carry the consumption back before Q's X-release source.
    auto p = base(4, 1);
    p.operations = {op(Q,{{3,false,true}}), op(P,{{3,true,false}}),
        op(R,{{1,true,false}}), op(P,{{0,false,true}}),
        op(Q,{{0,true,false},{2,false,true}}), op(Q,{{1,false,true}}), op(P,{{0,false,true}})};
    o::SelectedOptions options; options.sourceGaps = true; options.finalHelperTrials = false;
    // Isolate the older source-gap policy. The publication-prefix certificate
    // independently recovers this early release in ordinary construction.
    options.publicationPrefixes = false;
    const auto plan = o::constructSelectedPlan(p,{},options);
    require(plan.success, "reused-key source gap construction failed: " + plan.reason);
    require(bool(oahs_oracle::graph(p,plan.commands,{0,1,2,3,4,5,6},{{2,6}})),
            "reused-key X release imported unrelated Y completion");
    auto baselineOptions = options; baselineOptions.sourceGaps = false;
    const auto baseline = o::constructSelectedPlan(p,{},baselineOptions);
    require(baseline.success,baseline.reason);
    oahs_oracle::PayloadOrder before, after;
    require(bool(oahs_oracle::graph(p,baseline.commands,{0,1,2,3,4,5,6},{},nullptr,nullptr,&before)) &&
            bool(oahs_oracle::graph(p,plan.commands,{0,1,2,3,4,5,6},{},nullptr,nullptr,&after)),
            "reused gap ordering comparison failed");
    require(before != after && std::includes(before.begin(),before.end(),after.begin(),after.end()),
            "reused gap added payload order");
    unsigned publications = 0;
    for (const auto& word : plan.commands) for (const auto& command : word)
        publications += command.kind == o::Command::Publish && command.source == Q &&
                        command.observer == P && command.key == 0;
    require(publications >= 2, "source-gap witness did not reuse the one eligible key");
    auto broken = plan.commands;
    for (auto& word : broken) word.erase(std::remove_if(word.begin(),word.end(),[&](const auto& command) {
        return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
               command.source == P && command.observer == Q;
    }),word.end());
    require(!o::checkCausalFrontier(p,broken).accepted &&
            !bool(oahs_oracle::graph(p,broken,{0,1,2,3,4,5,6})),
            "reused gap invented its readiness/rearming support");
    std::cout << "reused_gap_relations=" << before.size() << "->" << after.size() << '\n';
    require(std::any_of(plan.decisions.begin(),plan.decisions.end(),[&](const auto& decision) {
        return decision.source == Q && decision.observer == P && decision.consumer == 6 &&
               decision.publicationAtWordStart;
    }), "reused release did not select the source gap");
}
// Supplied-protocol witness for the *contextual* merge certificate. It does
// not claim the constructor emits either plan: all fixed endpoints are explicit.
void commonFrontierContext() {
    using C = o::Command;
    unsigned cases = 0;
    for (unsigned episodes : {1u, 2u, 4u}) for (unsigned outwardPosition : {0u, 1u, 2u}) {
        auto p = base(3, 4);
        for (unsigned i = 0; i < episodes; ++i) {
            p.operations.push_back(op(Q, {{2,false,true}})); // z producer
            p.operations.push_back(op(P, {{0,false,true}})); // A
            p.operations.push_back(op(P, {{1,false,true}})); // B
            p.operations.push_back(op(Q, {{0,true,false},{1,true,false}}));
            p.operations.push_back(op(R, {{2,true,false}}));
        }
        o::Commands separate(p.operations.size()+1), merged(p.operations.size()+1);
        auto add = [](o::Commands& commands, unsigned cut, C::Kind kind,
                      o::Pipe source, o::Pipe observer, unsigned key = 0) {
            commands[cut].push_back({kind,source,observer,key});
        };
        for (unsigned i = 0; i < episodes; ++i) {
            const unsigned start = 5*i;
            for (auto* commands : {&separate, &merged}) {
                if (i) {
                    add(*commands,start,C::Acquire,R,Q);
                    add(*commands,start+1,C::Acquire,Q,P);
                }
                add(*commands,start+4,C::Publish,Q,P); // actual AB reader return
                add(*commands,start+4,C::Acquire,Q,R);
                add(*commands,start+5,C::Publish,R,Q); // actual z reader return
            }
            add(separate,start+2,C::Publish,P,Q,0); // early A source
            add(separate,start+3,C::Publish,P,Q,1); // later B source
            if (outwardPosition == 0) add(separate,start+3,C::Publish,Q,R);
            add(separate,start+3,C::Acquire,P,Q,0);
            if (outwardPosition == 1) add(separate,start+3,C::Publish,Q,R);
            add(separate,start+3,C::Acquire,P,Q,1);
            if (outwardPosition == 2) add(separate,start+3,C::Publish,Q,R);
            // One later source, same consumer cut. Both original waits are
            // sufficient for the Q payload, but their intermediate export differs.
            add(merged,start+3,C::Publish,P,Q);
            if (outwardPosition == 0) add(merged,start+3,C::Publish,Q,R);
            add(merged,start+3,C::Acquire,P,Q);
            if (outwardPosition != 0) add(merged,start+3,C::Publish,Q,R);
        }
        for (auto* commands : {&separate, &merged}) {
            add(*commands,p.operations.size(),C::Acquire,Q,P);
            add(*commands,p.operations.size(),C::Acquire,R,Q);
        }
        std::vector<unsigned> visits(p.operations.size());
        std::iota(visits.begin(),visits.end(),0);
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        require(bool(oahs_oracle::graph(p,separate,visits,{},nullptr,nullptr,&oldOrder)),
                "separate-frontier fixture lacks memory/balance/rearming");
        require(bool(oahs_oracle::graph(p,merged,visits,{},nullptr,nullptr,&newOrder)),
                "merged-frontier fixture lacks memory/balance/rearming");
        require(o::checkCausalFrontier(p,separate).accepted &&
                o::checkCausalFrontier(p,merged).accepted,
                "production checker disagrees with supplied frontier protocols");
        if (outwardPosition != 1) {
            require(oldOrder == newOrder, "private common consumer changed payload order");
        } else {
            require(std::includes(newOrder.begin(),newOrder.end(),oldOrder.begin(),oldOrder.end()) &&
                    oldOrder != newOrder, "outward publication did not expose broader order");
            for (unsigned i = 0; i < episodes; ++i) {
                const auto unwanted = std::make_pair(2*(5*i+2)+1,2*(5*i+4));
                require(!oldOrder.count(unwanted) && newOrder.count(unwanted),
                        "later B load must newly gate the unrelated z reader");
            }
        }
        // Delete a complete return channel so event balance alone cannot mask
        // the loss of real reuse and forward-key consumption evidence.
        if (episodes > 1) {
            auto broken = merged;
            for (auto& word : broken)
                word.erase(std::remove_if(word.begin(),word.end(),[&](const C& c) {
                    return c.source == Q && c.observer == P;
                }),word.end());
            const auto verdict = oahs_oracle::graph(p,broken,visits);
            require(verdict.balanced && !verdict.hazards && !verdict.rearm,
                    "missing return must lose memory and rearming despite balanced tokens");
            require(!o::checkCausalFrontier(p,broken).accepted,
                    "production checker accepted missing return");
        }
        std::cout << "frontier_context episodes=" << episodes << " outward_position=" << outwardPosition
                  << " relations=" << oldOrder.size() << "->" << newOrder.size() << '\n';
        ++cases;
    }
    require(cases == 9, "frontier context campaign incomplete");
}

void noMotionAndRandom() {
    auto p = base(2,4);
    p.operations = {op(P,{{0,false,true}}),op(P,{{1,false,true}}),op(Q,{{0,true,false},{1,true,false}})};
    auto loop = o::makePeriodicLoop(p,1,{}); require(loop.success,loop.reason);
    o::SelectedOptions options; options.movingFrontiers = false;
    auto plan = o::constructSelectedPlan(loop.program,{},options);
    require(plan.success, "no-motion protocol failed: " + plan.reason);
    std::cout << "no_motion_channels=" << plan.channels.size() << '\n';
    // Broad finite safety campaign for the two narrow acyclic options. The
    // independent oracle contains actual primitive edges, never desired edges.
    std::mt19937 random(2031);
    unsigned accepted = 0;
    for (unsigned sample = 0; sample < 200; ++sample) {
        auto input = base(3,2);
        for (unsigned i = 0; i < 7; ++i) {
            const auto pipe = std::array<o::Pipe,3>{P,Q,R}[random()%3];
            const unsigned cell = random()%3;
            const bool write = random()%2;
            input.operations.push_back(op(pipe,{{cell,!write,write}}));
        }
        options.sourceGaps = true; options.deferredAcyclicAcknowledgments = true;
        options.equalCoverageBinding = true;
        auto candidate = o::constructSelectedPlan(input,{},options);
        require(candidate.success,"finite acyclic construction failure: " + candidate.reason);
        require(bool(oahs_oracle::graph(input,candidate.commands,{0,1,2,3,4,5,6})),"independent finite oracle rejected plan");
        ++accepted;
    }
    std::cout << "acyclic_option_cases=" << accepted << '\n';
}
}
int main() {
    o::selected::ReplayTestAccess::joinedConsumptionReturn();
    o::selected::ReplayTestAccess::reusedGapCertificates();
    o::selected::ReplayTestAccess::proposalWordOrder();
    o::selected::ReplayTestAccess::proposalOmissionWords();
    o::selected::ReplayTestAccess::invalidProposal();
    o::selected::ReplayTestAccess::uncoveredProducerProposal();
    deferredRequiredReturn();
    deferredSharedReceipt(false);
    deferredSharedReceipt(true);
    deferredSharedReceipt(false, true);
    deferredSharedReceipt(false, false, true);
    starvation(); wordGapBaseline(); reusedSourceGap(); deferredAcknowledgment(); commonFrontierContext(); noMotionAndRandom();
}
