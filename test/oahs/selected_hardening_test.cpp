// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
#include <numeric>
#include <functional>
using namespace selected_test;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void pendingAccumulator(const Program& p) {
        Constructor c(p);
        auto state = c.initial();
        Replay replay;
        require(c.payload(state, 0, replay, true), replay.reason);
        require(c.payload(state, 1, replay, true), replay.reason);
        require(!c.frontier.inspect(state.causal, 2).applied,
                "pending compatible issue erased incompatible ACC history");
    }

    static void exactSourceGap()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE1;
        auto p = base(2, 2);
        p.operations = {op(Q, {{1, false, true}}), op(P, {{0, false, true}}),
                        op(P, {{1, true, false}}), op(R, {{0, true, false}, {1, true, false}})};
        Constructor c(p);
        c.current = 3;
        c.ledger.append(1, {Command::Publish, Q, P, 0}, EndpointPurpose::Fixed);
        const auto wait = c.ledger.append(2, {Command::Acquire, Q, P, 0}, EndpointPurpose::Fixed);
        require(c.contextualReplay(), c.result.reason);
        Id key = NoAnalysisId;
        for (Id i = 0; i < c.frontier.keys().size(); ++i) {
            const auto& identity = c.frontier.keys()[i];
            if (identity.source == P && identity.observer == R && identity.key == 0) {
                key = i;
            }
        }
        require(key != NoAnalysisId, "missing source-gap fixture key");
        const WordGap before{2, NoAnalysisId, wait};
        const std::vector<FrontierRequirement> own{{0, P, true, 3, true, false}};
        const std::vector<FrontierRequirement> incoming{{1, Q, true, 3, true, false}};
        require(c.sourceGap(before, key, own).proved(), "exact prefix lost its own completed class");
        require(!c.sourceGap(before, key, incoming).proved(),
                "word-tail credit was borrowed before its incoming receipt");
        require(c.sourceGap(c.ledger.tail(2), key, incoming).proved(),
                "actual incoming receipt did not establish gap coverage");
        SelectedDecision provider;
        provider.required = own;
        require(c.earlyPublicationGap(2, key, provider).has_value(),
                "fixture did not expose earlier motivating source");
        provider.supporting = incoming;
        require(!c.earlyPublicationGap(2, key, provider).has_value(),
                "earlier publication erased the coverage used to select its provider");
        const auto publication = c.ledger.append(3, {Command::Publish, P, R, 0}, EndpointPurpose::Fixed);
        const auto receipt = c.ledger.append(3, {Command::Acquire, P, R, 0}, EndpointPurpose::Fixed);
        require(!c.sourceGap(before, key, own).proved(), "stale source-gap replay was reused");
        require(c.contextualReplay(), c.result.reason);
        require(!c.sourceGap(before, key, own).proved(), "future key use lacked neighboring-generation check");
        c.ledger.erase(publication);
        c.ledger.erase(receipt);
        require(c.contextualReplay(), c.result.reason);
        require(!c.sourceGap(before, key, own).proved(), "dormant key use was mistaken for a virgin key");
    }

    static void sourceGapOccurrences()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE1;
        for (bool missing : {false, true}) {
            auto p = base(3, 2);
            p.operations = {op(Q, {{1, false, true}}), op(P, {{0, false, true}}),
                            op(P, {{missing ? 2u : 0u, false, true}}), op(R, {{0, true, false}})};
            ObservedControl graph;
            graph.qualification = "shared source and receipt words across two original alternatives";
            graph.scopes = {{0, NoControlId, NoControlId}};
            graph.entry = 0;
            graph.exit = 8;
            graph.sites.resize(9);
            const std::vector<std::vector<std::size_t>> edges{{1}, {2, 5}, {3}, {4}, {8}, {6}, {7}, {8}, {}};
            for (Id site = 0; site < graph.sites.size(); ++site) {
                graph.sites[site].successors = edges[site];
                graph.sites[site].observation = site;
                graph.observations.push_back({site, {}, true});
            }
            graph.sites[0].operation = 0;
            graph.sites[2].operation = 1;
            graph.sites[5].operation = 2;
            graph.sites[4].operation = graph.sites[7].operation = 3;
            graph.sites[6].observation = 3;
            graph.sites[7].observation = 4;
            p.observed = std::move(graph);
            Constructor c(p);
            require(c.control.complete, c.control.reason);
            c.current = 4;
            c.ledger.append(1, {Command::Publish, Q, P, 0}, EndpointPurpose::Fixed);
            const auto wait = c.ledger.append(3, {Command::Acquire, Q, P, 0}, EndpointPurpose::Fixed);
            require(c.contextualReplay(), c.result.reason);
            Id key = 0;
            while (c.frontier.keys()[key].source != P || c.frontier.keys()[key].observer != R) {
                ++key;
            }
            const auto proof = c.sourceGap({3, NoAnalysisId, wait}, key, {{0, P, true, 3, true, false}});
            require(proof.proved() == !missing,
                    "source gap did not check every shared-word occurrence's actual prefix");
        }
        auto p = base(1, 2);
        p.operations = {op(P, {{0, false, true}}), op(R, {{0, true, false}})};
        p.body = {Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
        const auto imported = addStructuredBoundaryCuts(p);
        require(imported.success, imported.reason);
        Constructor c(imported.program);
        for (Cut site = 0; site < c.control.graph.operations.size(); ++site) {
            if (c.control.graph.operations[site] == 1) {
                c.current = site;
            }
        }
        require(c.contextualReplay(), c.result.reason);
        Id key = 0;
        while (c.frontier.keys()[key].source != P || c.frontier.keys()[key].observer != R) {
            ++key;
        }
        require(!c.sourceGap(c.ledger.tail(c.current), key, {{0, P, true, 1, true, false}}).proved(),
                "acyclic source-gap certificate admitted a recurring occurrence");
    }

    static void sourceGapBarriers()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE1;
        for (auto barrier : {Command::Barrier, Command::BarrierAll, Command::Publish}) {
            auto p = base(2, 2);
            p.operations = {op(Q, {{1, false, true}}), op(P, {{0, false, true}}),
                            op(P, {{1, true, false}}), op(R, {{0, true, false}})};
            Constructor c(p);
            c.current = 3;
            c.ledger.append(1, {Command::Publish, Q, P, 0}, EndpointPurpose::Fixed);
            const auto wait = c.ledger.append(2, {Command::Acquire, Q, P, 0}, EndpointPurpose::Fixed);
            c.ledger.append(2, {barrier, P, Q, 1}, EndpointPurpose::Fixed);
            if (barrier == Command::Publish) {
                c.ledger.append(3, {Command::Acquire, P, Q, 1}, EndpointPurpose::Fixed);
            }
            if (barrier != Command::BarrierAll) {
                require(c.contextualReplay(), c.result.reason);
            }
            Id key = 0;
            while (c.frontier.keys()[key].source != P || c.frontier.keys()[key].observer != R) {
                ++key;
            }
            SelectedDecision decision;
            decision.required = {{0, P, true, 3, true, false}};
            require(!c.earlyPublicationGap(2, key, decision),
                    "publication crossed an outward source publication or fence");
            require(c.result.work.sourceGapQueries == 0,
                    "outward boundary was crossed before consulting the source-gap proof");
            require(c.ledger.active(wait), "gap query mutated its incoming receipt");
        }
    }

    static SelectedPlan contextual(const Program& p) {
        Constructor constructor(p);
        constructor.needsContextualReplay = true;
        return constructor.run({}, false);
    }
    static void ownedPackets(bool acknowledgment = false)
    {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        auto p = base(1, 2);
        p.operations = {op(P, {{0, true, false}}), op(Q, {{0, true, false}}),
                        op(P, {{0, true, false}}), op(Q, {{0, true, false}})};
        Constructor c(p);
        std::vector<std::pair<Id, Id>> helpers;
        for (Cut cut : {1u, 2u}) {
            c.ledger.append(cut, {Command::Publish, P, Q, 0}, EndpointPurpose::Completion);
            const auto forward = c.ledger.append(cut, {Command::Acquire, P, Q, 0}, EndpointPurpose::Completion);
            const auto pub = c.ledger.append(cut, {Command::Publish, Q, P, 0},
                EndpointPurpose::ConsumptionAcknowledgment, 0, forward);
            const auto wait = c.ledger.append(cut, {Command::Acquire, Q, P, 0},
                EndpointPurpose::ConsumptionAcknowledgment, 0, forward);
            c.rememberReturn(pub, wait);
            helpers.emplace_back(pub, wait);
            c.ledger.append(cut, {Command::Publish, Q, P, 1}, EndpointPurpose::Completion);
            c.ledger.append(cut, {Command::Acquire, Q, P, 1}, EndpointPurpose::Completion);
        }
        c.ledger.append(3, {Command::Publish, P, Q, 0}, EndpointPurpose::Completion);
        const auto latestWait = c.ledger.append(3, {Command::Acquire, P, Q, 0}, EndpointPurpose::Completion);
        const auto marker = c.ledger.append(3, {Command::Barrier, Pipe::MTE1}, EndpointPurpose::Fixed);
        for (const auto& helper : helpers) {
            c.ledger.erase(helper.first);
            c.ledger.erase(helper.second);
        }
        c.result.work.rearmingDischarged = helpers.size();
        c.current = 3;
        require(checkCausalFrontier(p, c.ledger.commands()).accepted, "dormant fixture's actual returns are invalid");
        if (acknowledgment) {
            c.needsContextualReplay = true;
            c.activeComponent = c.control.component[c.current];
            require(c.update(), c.result.reason);
            Cut publication = 3;
            Id key = NoAnalysisId;
            SelectedDecision decision;
            bool completed = false;
            require(c.acknowledgment(P, Q, publication, key, decision, completed) && completed,
                    "single-consumption repair did not realize its complete owned packet");
            require(c.result.work.ownershipBindings == 1 && c.result.work.rearmingRestored == 2,
                    "single-consumption repair bypassed shared owner closure");
            const auto& word = c.ledger.word(3);
            const auto anchor = std::find(word.begin(), word.end(), latestWait);
            require(anchor != word.end() && *std::next(anchor) == decision.endpoints.front() &&
                    std::find(word.begin(), word.end(), marker) > std::next(anchor),
                    "dormant reverse ownership moved the early acknowledgment behind unrelated work");
            require(checkCausalFrontier(p, c.ledger.commands()).accepted, "owned acknowledgment broke protocol");
            return;
        }

        // An earlier forward-key deadline seeds one obligation. Its reverse
        // identity is also owned by the other generation: both must participate.
        const auto recovery = c.qualifyOwnedPacket({}, true, {helpers.front().second});
        require(recovery && recovery->restoredWaits.size() == 2 && recovery->prepared.size() == 4,
                "forward deadline did not close the other reverse-key owner");
        require(!c.ledger.active(helpers.front().second) && !c.ledger.active(helpers.back().second),
                "deadline qualification committed fallback credit");
        const auto queries = c.result.work.ownershipQueries;
        require(!c.restoreReturns(NoAnalysisId) && c.result.work.ownershipQueries == queries,
                "unrelated rearming key performed an ownership solve");

        const OrderedPacket packet{
            {3, {Command::Publish, Q, P, 0}, EndpointPurpose::Completion},
            {3, {Command::Acquire, Q, P, 0}, EndpointPurpose::Completion}};
        const auto incomplete = c.prepareOwnedPacket(packet);
        require(bool(incomplete), "structural ownership packet could not be prepared");
        const auto unqualifiedVersion = c.ledger.version();
        SelectedDecision unchecked;
        require(!c.commitOwnedPacket(*incomplete, unchecked) && c.ledger.version() == unqualifiedVersion,
                "structural preparation bypassed shared qualification");
        const auto partialGap = *c.ledger.gapAfter(c.ledger.endpoint(helpers.front().second).acknowledges);
        const auto partial = *c.ledger.restoration(helpers.front().first, partialGap);
        require(c.ledger.appendPacket(c.ledger.preparePacket({partial})).size() == 1,
                "partial-owner negative fixture failed to install");
        const auto partialVersion = c.ledger.version();
        require(!c.qualifyOwnedPacket(packet) && c.ledger.version() == partialVersion,
                "partial active/inactive ownership was treated as a complete helper");
        c.ledger.erase(helpers.front().first);
        const auto before = c.ledger.commands();
        const auto revision = c.ledger.version();
        const auto prepared = c.qualifyOwnedPacket(packet);
        require(prepared && prepared->restoredWaits.size() == 2 && prepared->restoredEndpoints == 4,
                "shared physical key did not close ALL dormant helper owners");
        require(c.ledger.version() == revision && !c.rearming.at(helpers.front().second).required &&
                !c.rearming.at(helpers.back().second).required, "qualification mutated ownership");
        const auto staged = c.ledger.withPacket(prepared->prepared);
        require(staged && checkCausalFrontier(p, *staged).accepted, "complete owner closure failed causal checking");
        c.helperOwners.erase(helpers.front().first);
        require(!c.qualifyOwnedPacket(packet), "unaccounted dormant record was ignored");
        c.helperOwners[helpers.front().first] = helpers.front().second;
        auto conflict = packet;
        conflict.back().command.kind = Command::Publish;
        require(!c.qualifyOwnedPacket(conflict), "invalid neighboring publication borrowed a dormant key");
        require(c.ledger.version() == revision && !c.rearming.at(helpers.front().second).required &&
                !c.rearming.at(helpers.back().second).required, "refused packet partially committed");
        SelectedDecision decision;
        require(c.commitOwnedPacket(*prepared, decision), c.result.reason);
        require(decision.endpoints.size() == 2 && c.rearming.at(helpers.front().second).required &&
                c.rearming.at(helpers.back().second).required &&
                c.result.work.rearmingRestored == 2 && c.result.work.rearmingDischarged == 0,
                "owner closure bookkeeping lost an owner or mixed it with new requirements");
        require(checkCausalFrontier(p, c.ledger.commands()).accepted, "committed owner closure changed checked words");
        require(!c.ledger.hasDormantUses({Q, P, 0}), "closed owners remained dormant");
        const auto committed = c.ledger.commands();
        require(staged->size() == committed.size(), "ownership packet changed command population");
        for (Cut cut = 0; cut < committed.size(); ++cut) {
            require((*staged)[cut].size() == committed[cut].size() &&
                std::equal((*staged)[cut].begin(), (*staged)[cut].end(), committed[cut].begin(), identical),
                "owner closure committed a different ordered word");
        }
    }

    static void ownedCandidateOrder()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE3;
        auto p = base(1, 2);
        p.operations = {op(P, {{0, true, false}}), op(Q, {{0, true, false}}),
                        op(P, {{0, true, false}}), op(Q, {{0, true, false}})};
        Constructor c(p);
        auto helper = [&](Cut cut, unsigned key) {
            c.ledger.append(cut, {Command::Publish, P, Q, 0}, EndpointPurpose::Completion);
            const auto anchor = c.ledger.append(cut, {Command::Acquire, P, Q, 0}, EndpointPurpose::Completion);
            const auto pub = c.ledger.append(cut, {Command::Publish, Q, P, key},
                EndpointPurpose::ConsumptionAcknowledgment, 0, anchor);
            const auto wait = c.ledger.append(cut, {Command::Acquire, Q, P, key},
                EndpointPurpose::ConsumptionAcknowledgment, 0, anchor);
            c.rememberReturn(pub, wait);
            c.ledger.erase(pub);
            c.ledger.erase(wait);
            return std::make_pair(pub, wait);
        };
        const auto early = helper(1, 1);
        c.ledger.append(2, {Command::Publish, Q, R, 0}, EndpointPurpose::Completion);
        c.ledger.append(2, {Command::Acquire, Q, R, 0}, EndpointPurpose::Completion);
        c.ledger.append(2, {Command::Publish, R, P, 0}, EndpointPurpose::Completion);
        c.ledger.append(2, {Command::Acquire, R, P, 0}, EndpointPurpose::Completion);
        const auto late = helper(3, 0);
        c.result.work.rearmingDischarged = 2;
        c.needsContextualReplay = true;
        c.current = 3;
        c.activeComponent = c.control.component[3];
        require(c.update(), c.result.reason);
        SelectedDecision decision;
        decision.source = Q;
        decision.observer = P;
        Cut publication = 3;
        require(c.edge(Q, P, publication, false, decision), c.result.reason);
        require(c.result.work.ownershipChecks >= 2 && c.result.work.ownershipBindings == 1,
                "lower unbindable dormant candidate hid a later complete certificate");
        require(c.ledger.endpoint(decision.endpoints.front()).command.key == 1 &&
                c.ledger.active(early.first) && c.ledger.active(early.second) &&
                !c.ledger.active(late.first) && !c.ledger.active(late.second),
                "candidate refusal restored a partial ledger or selected the wrong owner");
        require(c.rearming.at(early.second).required && !c.rearming.at(late.second).required,
                "failed ownership candidate changed the retained support set");
        require(checkCausalFrontier(p, c.ledger.commands()).accepted, "candidate retry broke causal/event legality");
    }

};
}
namespace {
const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
void pendingAccumulatorHistories()
{
    auto p = base(1, 2);
    p.target = mlir::pto::a3SyncProfile(mlir::pto::SyncCore::Cube);
    p.nativeAccumulatorClasses = 2;
    auto& cell = p.cells[0];
    cell.domain = o::Cell::Domain::Accumulator;
    cell.storage = o::Cell::Storage::CanonicalInterval;
    cell.coordinateSpace = "physical-local";
    cell.ranges = {{0, 131072}};
    p.operations = {op(o::Pipe::M, {{0, false, true, false, 1}}),
                    op(o::Pipe::M, {{0, true, true, false, 0}}),
                    op(o::Pipe::M, {{0, true, true, false, 0}})};
    p.operations[1].nativeMmadAccumulate = p.operations[2].nativeMmadAccumulate = true;
    for (auto contract : {std::size_t(1), o::NoControlId}) {
        p.operations[0].accesses[0].nativeAccumulatorClass = contract;
        o::selected::ReplayTestAccess::pendingAccumulator(p);
    }
}

void compareWords(const o::Commands& left, const o::Commands& right)
{
    require(left.size() == right.size(), "ordinary retry changed command-word population");
    for (unsigned word = 0; word < left.size(); ++word) {
        require(left[word].size() == right[word].size(), "ordinary retry retained optional endpoints");
        for (unsigned index = 0; index < left[word].size(); ++index) {
            require(o::selected::identical(left[word][index], right[word][index]),
                    "ordinary retry retained optional reservations or credit");
        }
    }
}
void joinedConsumptionReturn()
{
    using namespace mlir::pto::oahs;
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
        for (Cut i = 0; i < p.observed->sites.size(); ++i) {
            if (p.observed->sites[i].operation == op) {
                return i;
            }
        }
        return NoAnalysisId;
    };
    Commands fixed(commandCutCount(p));
    const auto firstPublication = p.observed->sites[cut(0)].successors.front();
    fixed[firstPublication] = {{Command::Publish,P,Q,0}};
    fixed[cut(1)] = fixed[cut(2)] = {{Command::Acquire,P,Q,0}};
    const auto plan = constructSelectedPlan(p,fixed);
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
        } else {
            for (auto next : site.successors) {
                walk(next, flat, words, pending);
            }
        }
    };
    auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
    walk(p.observed->entry,flat,{},{});
    auto missing = plan.commands;
    for (auto& word : missing) {
        word.erase(std::remove_if(word.begin(),word.end(),[&](const Command& c) {
            return c.source == Q && c.observer == P;
        }),word.end());
    }
    require(!checkCausalFrontier(p,missing).accepted, "missing joined consumption support was accepted");
    auto unavailable = p;
    unavailable.target.keys[unsigned(Q)][unsigned(P)].clear();
    const auto refused = constructSelectedPlan(unavailable,fixed);
    require(!refused.success && refused.commands.empty(), "joined return invented an unavailable reverse key");
    // A branch which does not consume leaves a maybe-full key: a return
    // must not turn this into empty occupancy.
    auto unconsumed = fixed;
    unconsumed[cut(2)].clear();
    require(!constructSelectedPlan(p,unconsumed).success,
            "joined return accepted an unconsumed branch");
}

void repeatedJoinedPacket()
{
    auto input = base(2, 1);
    input.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                        op(Q, {{0, true, false}}), op(P, {{1, false, true}}),
                        op(Q, {{1, true, false}})};
    input.body = seq({leaf(0), {o::Region::Choice, {leaf(1), leaf(2)}},
                      leaf(3), {o::Region::For, {leaf(4)}}});
    auto imported = o::addStructuredBoundaryCuts(input);
    require(imported.success, imported.reason);
    const auto& p = imported.program;
    const auto& graph = *p.observed;
    auto cut = [&](unsigned operation) {
        for (o::Cut at = 0; at < graph.sites.size(); ++at) {
            if (graph.sites[at].operation == operation) {
                return at;
            }
        }
        return o::NoAnalysisId;
    };
    o::Commands fixed(o::commandCutCount(p));
    fixed[graph.sites[cut(0)].successors.front()] = {{o::Command::Publish, P, Q, 0}};
    fixed[cut(1)] = fixed[cut(2)] = {{o::Command::Acquire, P, Q, 0}};
    const auto plan = o::constructSelectedPlan(p, fixed);
    require(plan.success, "normal repeated joined construction: " + plan.reason);
    require(plan.work.joinedAcknowledgments != 0, "normal constructor did not discover the joined packet");
    require(!plan.declinedRecurring, "joined packet was discovered only through a retry");
    const auto& packet = plan.commands[cut(4)];
    require(packet.size() == 4 && packet[0].source == Q && packet[2].source == P,
            "same-word packet did not return consumption before forward reuse");
    auto half = plan.commands;
    half[cut(4)].resize(2);
    require(!o::checkCausalFrontier(p, half).accepted,
            "reverse-only half incorrectly rearms its next recurring publication");
    auto missingReceipt = plan.commands;
    missingReceipt[cut(4)].pop_back();
    require(!o::checkCausalFrontier(p, missingReceipt).accepted,
            "missing forward receipt incorrectly rearms the reverse key");
    unsigned paths = 0;
    std::function<void(o::Cut, o::Program, o::Commands, std::vector<o::Command>, std::vector<unsigned>)> walk;
    walk = [&](o::Cut at, o::Program flat, o::Commands words, std::vector<o::Command> pending,
               std::vector<unsigned> visits) {
        if (++visits[at] > 4) {
            return;
        }
        pending.insert(pending.end(), plan.commands[at].begin(), plan.commands[at].end());
        const auto& site = graph.sites[at];
        if (site.operation != o::NoAnalysisId) {
            flat.operations.push_back(p.operations[site.operation]);
            words.push_back(std::move(pending));
            pending.clear();
        }
        if (at == graph.exit) {
            words.push_back(std::move(pending));
            std::vector<unsigned> trace(flat.operations.size());
            std::iota(trace.begin(), trace.end(), 0);
            require(bool(oahs_oracle::graph(flat, words, trace)), "repeated joined packet failed independent trace");
            ++paths;
        } else {
            for (auto next : site.successors) {
                walk(next, flat, words, pending, visits);
            }
        }
    };
    auto flat = p;
    flat.observed.reset();
    flat.body = {};
    flat.operations.clear();
    walk(graph.entry, flat, {}, {}, std::vector<unsigned>(graph.sites.size()));
    require(paths >= 8, "joined packet did not exercise branch alternatives and repeated visits");
    auto missing = fixed;
    missing[cut(2)].clear();
    const auto unconsumed = o::constructSelectedPlan(p, missing);
    require(!unconsumed.success && unconsumed.commands.empty(), "repeated joined packet ignored an unconsumed branch");
    auto noReverse = p;
    noReverse.target.keys[unsigned(Q)][unsigned(P)].clear();
    const auto unavailable = o::constructSelectedPlan(noReverse, fixed);
    require(!unavailable.success && unavailable.commands.empty(),
            "repeated joined packet invented a reverse direction");
}

void orderedPacketMaterialization()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}})};
    o::selected::Control control(p);
    o::selected::Ledger ledger(p, control.canonicalCut);
    ledger.append(1, {o::Command::Barrier, Q}, o::EndpointPurpose::Fixed);
    const auto original = ledger.commands();
    const auto revision = ledger.version();
    const o::selected::OrderedPacket packet{
        {1, {o::Command::Acquire, P, Q, 0}, o::EndpointPurpose::RecurringCompletion, 0},
        {1, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::RecurringCompletion, 1},
        {2, {o::Command::Acquire, Q, P, 0}, o::EndpointPurpose::RecurringCompletion, 1}};
    const auto prepared = ledger.preparePacket(packet);
    const auto staged = ledger.withPacket(prepared);
    require(bool(staged), "packet preparation failed");
    compareWords(original, ledger.commands());
    require(ledger.version() == revision, "private packet changed live revision");
    const auto ids = ledger.appendPacket(prepared);
    compareWords(*staged, ledger.commands());
    require(ids.size() == 3 && ledger.endpoint(ids[1]).request == 1,
            "packet lost logical matching provenance");
    require((*staged)[1].size() == 3 && (*staged)[1][0].kind == o::Command::Barrier &&
                (*staged)[1][1].kind == o::Command::Acquire && (*staged)[1][2].kind == o::Command::Publish,
            "packet sorted publications ahead of an earlier receipt or fixed command");
}
void stablePacketGaps()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}})};
    o::selected::Control control(p);
    o::selected::Ledger ledger(p, control.canonicalCut);
    const auto wait = ledger.append(1, {o::Command::Acquire, P, Q, 0}, o::EndpointPurpose::Fixed);
    const auto outward = ledger.append(1, {o::Command::Publish, Q, P, 1}, o::EndpointPurpose::Fixed);
    const auto gap = ledger.gapAfter(wait);
    require(gap && gap->right == outward, "gap lost outward publication boundary");
    const o::selected::OrderedPacket packet{
        {1, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::ConsumptionAcknowledgment, 0, wait, gap},
        {1, {o::Command::Acquire, Q, P, 0}, o::EndpointPurpose::ConsumptionAcknowledgment, 0, wait, gap},
        {1, {o::Command::Publish, P, Q, 0}, o::EndpointPurpose::Completion, 0, o::NoAnalysisId, gap, 1}};
    const auto prepared = ledger.preparePacket(packet);
    auto foreign = ledger;
    require(!foreign.withPacket(prepared) && foreign.appendPacket(prepared).empty(),
            "same-revision foreign ledger accepted another ledger's packet");
    const auto staged = ledger.withPacket(prepared);
    require(prepared.valid() && staged && ledger.word(1).size() == 2, "preparation mutated its ledger");
    const auto ids = ledger.appendPacket(prepared);
    require(ids.size() == 3 && ledger.word(1) == std::vector<std::size_t>{wait, ids[0], ids[1], ids[2], outward},
            "same-gap endpoints lost their selected order");
    require(ledger.endpoint(ids[2]).acknowledges == ids[1], "packet-local identity was not resolved");
    compareWords(*staged, ledger.commands());
    const auto committed = ledger.commands();
    const auto revision = ledger.version();
    require(!ledger.withPacket(prepared) && ledger.appendPacket(prepared).empty(), "stale packet was accepted");
    require(ledger.version() == revision, "stale packet partially mutated the ledger");
    compareWords(committed, ledger.commands());
    require(!ledger.preparePacket(packet).valid(), "separated neighbors remained an adjacent gap");
    const auto restorationPacket = ledger.preparePacket({
        {1, {o::Command::Barrier, Q}, o::EndpointPurpose::LocalFence}});
    ledger.erase(outward);
    const auto restored = ledger.restoration(outward, *ledger.gapAfter(ids.back()));
    require(restored && ledger.appendPacket(ledger.preparePacket({*restored})).size() == 1,
            "exact inactive endpoint restoration failed");
    compareWords(committed, ledger.commands());
    require(!ledger.withPacket(restorationPacket) && ledger.appendPacket(restorationPacket).empty(),
            "restoring identical words revived a stale proof revision");
    ledger.erase(wait);
    require(!ledger.gapAfter(wait), "erased endpoint retained a usable gap");
    require(!ledger.preparePacket(packet).valid(), "erased gap neighbor was accepted");
    auto malformed = packet;
    malformed[0].gap.reset();
    malformed[0].acknowledges = o::NoAnalysisId;
    malformed[0].acknowledgesPacket = 1;
    require(!ledger.preparePacket(malformed).valid(), "forward packet-local reference was accepted");
}
void mixedRestorationPacket()
{
    auto p = base(1, 1);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(Q, {{0, true, false}})};
    o::selected::Control control(p);
    o::selected::Ledger ledger(p, control.canonicalCut);
    ledger.append(1, {o::Command::Publish, P, Q, 0}, o::EndpointPurpose::Completion);
    const auto wait = ledger.append(1, {o::Command::Acquire, P, Q, 0}, o::EndpointPurpose::Completion);
    const auto publication = ledger.append(1, {o::Command::Publish, Q, P, 0},
        o::EndpointPurpose::ConsumptionAcknowledgment, 0, wait);
    const auto acquisition = ledger.append(1, {o::Command::Acquire, Q, P, 0},
        o::EndpointPurpose::ConsumptionAcknowledgment, 0, wait);
    const auto marker = ledger.append(1, {o::Command::Barrier, o::Pipe::MTE1}, o::EndpointPurpose::Fixed);
    ledger.erase(publication);
    ledger.erase(acquisition);
    require(ledger.hasDormantUses({Q, P, 0}), "erased protocol lost physical ownership history");
    const auto gap = *ledger.gapAfter(wait);
    const auto pub = *ledger.restoration(publication, gap);
    const auto acq = *ledger.restoration(acquisition, gap);
    const o::selected::OrderedPacket proposal{pub, acq,
        {1, {o::Command::Publish, P, Q, 0}, o::EndpointPurpose::Completion,
            1, o::NoAnalysisId, gap, 1},
        {2, {o::Command::Acquire, P, Q, 0}, o::EndpointPurpose::Completion}};
    const auto prepared = ledger.preparePacket(proposal);
    const auto staged = ledger.withPacket(prepared);
    require(prepared.valid() && staged && o::checkCausalFrontier(p, *staged).accepted,
            "complete restored/new packet failed independent protocol validation");
    auto foreign = ledger;
    require(foreign.appendPacket(prepared).empty(), "foreign ledger accepted a restoration certificate");
    auto duplicate = proposal;
    duplicate.push_back(pub);
    const auto revision = ledger.version();
    const auto before = ledger.commands();
    require(!ledger.preparePacket(duplicate).valid(), "duplicate restoration was accepted");
    auto changed = proposal;
    changed[0].command.key = 1;
    require(!ledger.preparePacket(changed).valid(), "restoration changed physical identity");
    changed = proposal;
    changed[0].acknowledges = marker;
    require(!ledger.preparePacket(changed).valid(), "restoration changed consumption provenance");
    changed = proposal;
    changed[0].cut = 2;
    require(!ledger.preparePacket(changed).valid(), "restoration crossed its original word");
    require(ledger.version() == revision && ledger.hasDormantUses({Q, P, 0}),
            "rejected restoration mutated version or ownership");
    compareWords(before, ledger.commands());
    const auto oldCount = ledger.records().size();
    const auto ids = ledger.appendPacket(prepared);
    require(ids == std::vector<std::size_t>{publication, acquisition, oldCount, oldCount + 1},
            "mixed packet changed restored identities or fresh endpoint numbering");
    require(ledger.endpoint(ids[2]).acknowledges == acquisition,
            "packet-local acknowledgment confused restored and newly allocated identities");
    require(ledger.word(1) == std::vector<std::size_t>{0, wait, publication, acquisition, ids[2], marker},
            "restored and new endpoints lost exact shared-gap order");
    require(!ledger.hasDormantUses({Q, P, 0}) && ledger.eventUses({Q, P, 0}).size() == 2,
            "restoration duplicated event-use membership or retained stale dormant ownership");
    compareWords(*staged, ledger.commands());
    require(ledger.appendPacket(prepared).empty(), "stale restoration certificate was committed twice");
    require(!ledger.restoration(publication, gap), "active endpoint offered for restoration");
    ledger.erase(publication);
    ledger.erase(acquisition);
    const auto pending = ledger.preparePacket({*ledger.restoration(publication, *ledger.gapAfter(wait)),
                                              *ledger.restoration(acquisition, *ledger.gapAfter(wait))});
    ledger.erase(wait);
    require(!ledger.withPacket(pending) && ledger.appendPacket(pending).empty(),
            "inactive consumption anchor retained its old restoration certificate");
    const auto newGap = ledger.tail(1);
    require(!ledger.preparePacket({*ledger.restoration(publication, newGap),
                                   *ledger.restoration(acquisition, newGap)}).valid(),
            "fresh restoration accepted an inactive consumption anchor");
}
void aliasPacketGaps()
{
    auto p = base(1, 1);
    p.operations = {op(P, {{0, false, true, true}})};
    o::ObservedControl graph;
    graph.qualification = "original shared packet boundary";
    graph.sites.resize(4);
    graph.observations = {{10, {}, true}, {11, {}, true}, {12, {}, true}};
    graph.entry = 2;
    graph.exit = 3;
    graph.sites[2] = {0, 0, {1}, {}, 0};
    graph.sites[1] = {o::NoControlId, 1, {0}, {}, 0};
    graph.sites[0] = {o::NoControlId, 1, {3}, {}, 0};
    graph.sites[3] = {o::NoControlId, 2, {}, {}, 0};
    p.observed = graph;
    o::selected::Control control(p);
    require(control.complete && control.canonicalCut[0] == control.canonicalCut[1], "invalid alias fixture");
    o::selected::Ledger ledger(p, control.canonicalCut);
    const auto wait = ledger.append(1, {o::Command::Acquire, P, Q, 0}, o::EndpointPurpose::Fixed);
    const auto prepared = ledger.preparePacket({
        {0, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::ConsumptionAcknowledgment,
         0, wait, ledger.gapAfter(wait)}});
    const auto staged = ledger.withPacket(prepared);
    require(staged && ledger.appendPacket(prepared).size() == 1, "aliased gap did not materialize");
    compareWords(*staged, ledger.commands());
    require(ledger.word(0) == ledger.word(1) && (*staged)[0].size() == 2 && (*staged)[1].size() == 2,
            "canonical packet word was not replicated at its original aliases");
    const auto original = ledger.word(0).back();
    ledger.erase(original);
    const auto restoration = ledger.restoration(original, *ledger.gapAfter(wait));
    require(bool(restoration), "canonical alias lost restoration anchor");
    const auto restored = ledger.preparePacket({*restoration});
    const auto restoredCommands = ledger.withPacket(restored);
    require(restoredCommands && ledger.appendPacket(restored).size() == 1,
            "canonical alias could not restore its original endpoint");
    compareWords(*staged, *restoredCommands);
    compareWords(*restoredCommands, ledger.commands());
}
void resourceAdmission(unsigned keys)
{
    auto p = base(3, keys);
    for (unsigned a = 0; a < o::PipeCount; ++a) {
        for (unsigned b = 0; b < o::PipeCount; ++b) {
            if (!((a == unsigned(P) && b == unsigned(Q)) ||
                  (a == unsigned(Q) && b == unsigned(P)))) {
                p.target.keys[a][b].clear();
            }
        }
    }
    p.operations = {op(P, {{2, false, true}}), op(Q, {{2, true, false}}),
                    op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true}}), op(Q, {{1, true, false}})};
    p.body = seq({leaf(0), leaf(1),
                  {o::Region::For, {seq({leaf(2), leaf(3), leaf(4), leaf(5)})}, 0, true}});
    auto input = o::addStructuredBoundaryCuts(p);
    require(input.success, input.reason);
    o::CountedLoopRegion loop;
    const auto& control = *input.program.observed;
    for (const auto& scope : control.scopes) {
        if (scope.kind == o::AnalysisContext::ForBody) {
            loop.owner = scope.ownerSite;
        }
    }
    require(loop.owner != o::NoControlId, "cohort has no loop owner");
    loop.header = control.sites[loop.owner].successors.front();
    loop.bodyEntry = control.sites[loop.header].successors.front();
    loop.continuation = control.sites[loop.header].successors.back();
    std::vector<std::size_t> todo{loop.bodyEntry};
    std::set<std::size_t> seen;
    while (!todo.empty()) {
        const auto site = todo.back();
        todo.pop_back();
        if (site == loop.header || !seen.insert(site).second) {
            continue;
        }
        loop.bodySites.push_back(site);
        const auto& next = control.sites[site].successors;
        todo.insert(todo.end(), next.begin(), next.end());
    }
    input = o::refineCountedLoop(input.program, loop);
    require(input.success, input.reason);
    o::Commands fixed(o::commandCutCount(input.program));
    fixed[input.program.observed->entry] = {{o::Command::Barrier, P}};
    o::selected::Constructor ordinary(input.program);
    const auto reference = ordinary.run(fixed, false);
    const auto plan = o::constructSelectedPlan(input.program, fixed);
    require(plan.success == reference.success && plan.failure == reference.failure,
            "declined proposal changed ordinary construction outcome");
    if (plan.success) {
        require(o::checkCausalFrontier(input.program, plan.commands).accepted,
                "ordinary retry failed independent validation");
    } else {
        require(plan.commands.empty(), "failed ordinary retry exported executable commands");
    }
    if (keys == 2) {
        require(plan.success, "exact-fit cohort stranded a feasible ordinary handoff: " + plan.reason);
    }
    require(plan.declinedRecurring || plan.work.recurringDeclines != 0,
            "unbound optional support must be visibly declined");
    if (plan.declinedRecurring) {
        require(plan.channels.empty(), "whole-attempt retry retained recurring reservations");
        require(plan.declinedRecurring->work.recurringProposals != 0, "retry lost proposal accounting");
        compareWords(plan.commands, reference.commands);
    } else {
        require(plan.work.recurringActivations == 1 && plan.channels.size() == 2,
                "local decline did not preserve exactly the admitted complete family");
    }
    std::cout << "keys=" << keys << " local_declines=" << plan.work.recurringDeclines
              << " retry=" << bool(plan.declinedRecurring) << '\n';
}
void separateReleaseFrontiers()
{
    auto body = base(2, 8);
    body.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                       op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    const auto input = o::makePeriodicLoop(body, 1, {});
    require(input.success, input.reason);
    const auto& p = input.program;
    const auto plan = accepted(p);
    const auto& graph = *p.observed;
    const unsigned iterations = 6, total = 4 * iterations;
    std::vector<o::Cut> path;
    std::vector<unsigned> visits(graph.sites.size());
    std::function<bool(o::Cut, unsigned, unsigned)> trace = [&](o::Cut site, unsigned payloads, unsigned iteration) {
        if (payloads > total || visits[site] > iterations + 1) {
            return false;
        }
        const auto& node = graph.sites[site];
        if (node.observation != o::NoAnalysisId) {
            for (const auto& atom : graph.observations[node.observation].atoms) {
                if (atom.kind == o::ObservationAtom::LoopHasPrevious &&
                    (iteration >= atom.parameter) != bool(atom.value)) {
                    return false;
                }
                if (atom.kind == o::ObservationAtom::LoopHasNext &&
                    (iterations > iteration && iterations - iteration > atom.parameter) != bool(atom.value)) {
                    return false;
                }
            }
        }
        if (node.operation != o::NoAnalysisId) {
            if (payloads == total || p.operations[node.operation].original != payloads % 4) {
                return false;
            }
            ++payloads;
        }
        path.push_back(site);
        ++visits[site];
        if (site == graph.exit && payloads == total) {
            return true;
        }
        for (std::size_t edge = 0; edge < node.successors.size(); ++edge) {
            const bool backedge = !node.backedgeOwners.empty() && node.backedgeOwners[edge] != o::NoControlId;
            if (trace(node.successors[edge], payloads, iteration + unsigned(backedge))) {
                return true;
            }
        }
        --visits[site];
        path.pop_back();
        return false;
    };
    require(trace(graph.entry, 0, 0), "missing six-visit recurring release witness");
    auto flat = body;
    flat.operations.clear();
    o::Commands words(1);
    for (auto site : path) {
        words.back().insert(words.back().end(), plan.commands[site].begin(), plan.commands[site].end());
        if (graph.sites[site].operation != o::NoAnalysisId) {
            flat.operations.push_back(p.operations[graph.sites[site].operation]);
            words.emplace_back();
        }
    }
    std::vector<unsigned> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::vector<std::pair<unsigned, unsigned>> forbidden;
    for (unsigned visit = 1; visit < iterations; ++visit) {
        forbidden.emplace_back(4 * visit - 1, 4 * visit);
    }
    require(bool(oahs_oracle::graph(flat, words, order, forbidden)),
            "B reader completion gates the next A refill");
}

void earlyPublication()
{
    const auto R = o::Pipe::MTE1;
    auto p = base(2, 2);
    p.operations = {op(Q, {{1, false, true}}), op(P, {{0, false, true}}),
                    op(P, {{1, true, false}}), op(R, {{0, true, false}})};
    const auto plan = o::constructSelectedPlan(p);
    require(plan.success, plan.reason);
    require(plan.work.earlyPublications == 1, "ordinary construction missed its exact early source gap");
    require(o::checkCausalFrontier(p, plan.commands).accepted, "early publication lost independent safety");
    const auto& word = plan.commands[2];
    require(word.size() == 2 && word[0].kind == o::Command::Publish && word[0].source == P &&
                word[0].observer == R && word[1].kind == o::Command::Acquire && word[1].observer == P,
            "new release publication did not precede the unrelated incoming wait");
    std::vector<unsigned> visits{0, 1, 2, 3};
    require(bool(oahs_oracle::graph(p, plan.commands, visits, {{0, 3}})),
            "unrelated incoming completion still gates the independent reader");
    auto tail = plan.commands;
    std::swap(tail[2][0], tail[2][1]);
    require(o::checkCausalFrontier(p, tail).accepted, "tail-placement reference is not valid");
    require(!bool(oahs_oracle::graph(p, tail, visits, {{0, 3}})),
            "ordering witness did not distinguish an enlarged publication prefix");
    unsigned removed = 0;
    for (unsigned from = 0; from < visits.size(); ++from) {
        for (unsigned to = 0; to < visits.size(); ++to) {
            const bool before = !bool(oahs_oracle::graph(p, tail, visits, {{from, to}}));
            const bool after = !bool(oahs_oracle::graph(p, plan.commands, visits, {{from, to}}));
            require(!after || before, "early source gap added a complete payload-order relation");
            removed += before && !after;
        }
    }
    require(removed == 1, "unexpected complete-order difference in early-publication witness");

}

void deadlineFence()
{
    auto p = base(3, 4);
    p.operations = {op(P, {{0, false, true}}), op(P, {{0, false, true}}),
                    op(P, {{1, false, true}}), op(Q, {{1, true, false}, {2, false, true}}),
                    op(P, {{2, true, false}}), op(P, {{0, false, true}})};
    for (unsigned i = 0; i < p.operations.size(); ++i) {
        p.operations[i].original = i;
    }
    p.operations.back().original = 1;
    const auto imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    const auto& input = imported.program;
    const auto plan = o::selected::ReplayTestAccess::contextual(input);
    require(plan.success, "contextual fixture: " + plan.reason);
    require(o::checkCausalFrontier(input, plan.commands).accepted, "deadline repair lost safety");
    auto flat = p;
    flat.body = {};
    o::Commands words(1);
    const auto& control = *input.observed;
    for (auto site = control.entry;; site = control.sites[site].successors.front()) {
        words.back().insert(words.back().end(), plan.commands[site].begin(), plan.commands[site].end());
        if (control.sites[site].operation != o::NoControlId) {
            words.emplace_back();
        }
        if (site == control.exit) {
            break;
        }
        require(control.sites[site].successors.size() == 1, "fixture must be linear");
    }
    std::vector<unsigned> visits(p.operations.size());
    std::iota(visits.begin(), visits.end(), 0);
    require(bool(oahs_oracle::graph(flat, words, visits, {{4, 5}})),
            "future fence imports unrelated read-z completion into later F");
    auto eager = words;
    eager[5].push_back({o::Command::Barrier, P});
    require(bool(oahs_oracle::graph(flat, eager, visits)), "eager-fence control should remain safe");
    require(!bool(oahs_oracle::graph(flat, eager, visits, {{4, 5}})),
            "oracle failed to distinguish speculative future fencing");
}
}
int main()
{
    o::selected::ReplayTestAccess::ownedPackets();
    o::selected::ReplayTestAccess::ownedPackets(true);
    o::selected::ReplayTestAccess::ownedCandidateOrder();
    o::selected::ReplayTestAccess::exactSourceGap();
    o::selected::ReplayTestAccess::sourceGapOccurrences();
    o::selected::ReplayTestAccess::sourceGapBarriers();
    earlyPublication();
    pendingAccumulatorHistories();
    repeatedJoinedPacket();
    joinedConsumptionReturn();
    orderedPacketMaterialization();
    stablePacketGaps();
    aliasPacketGaps();
    mixedRestorationPacket();
    resourceAdmission(1);
    resourceAdmission(2);
    deadlineFence();
    separateReleaseFrontiers();
    return 0;
}
