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

namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void pairEnumeration(unsigned count)
    {
        auto p = selected_test::base(1, 2);
        const auto P = Pipe::MTE2, Q = Pipe::V;
        p.operations = {selected_test::op(P, {{0, true, false}})};
        Constructor c(p);
        c.needsContextualReplay = true;
        std::string reason;
        selected_test::require(c.ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        // Structural-query stress fixture, not an accepted event protocol:
        // the intervening P payload must reject every possible replacement.
        for (unsigned i = 0; i < count; ++i) {
            auto publish = c.ledger.append(0, {Command::Publish, Q, P, 0}, EndpointPurpose::ConsumptionAcknowledgment);
            auto wait = c.ledger.append(0, {Command::Acquire, Q, P, 0}, EndpointPurpose::ConsumptionAcknowledgment);
            c.rememberReturn(publish, wait);
            c.ledger.append(1, {Command::Publish, Q, P, 1}, EndpointPurpose::Completion);
            auto actual = c.ledger.append(1, {Command::Acquire, Q, P, 1}, EndpointPurpose::Completion);
            SelectedDecision decision;
            decision.endpoints = {wait, actual};
            selected_test::require(c.settleRearming(decision), "pair enumeration changed ledger");
            selected_test::require(c.result.work.rearmingPairVisits == uint64_t(i+1)*(i+1),
                                   "old helper/return product was enumerated again");
            // Relevant decision with no newly appended helper or necessary return.
            decision.endpoints = {wait};
            selected_test::require(c.settleRearming(decision), "unchanged pair population");
            selected_test::require(c.result.work.rearmingPairVisits == uint64_t(i+1)*(i+1),
                                   "unchanged population revisited old pairs");
        }
        selected_test::require(c.result.work.rearmingDischarged == 0, "protected P payload ignored");
    }
    static void latentSourceSupport(unsigned position)
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE3;
        auto p = selected_test::base(1, 2);
        p.operations = {selected_test::op(P, {{0, true, false}}),
                        selected_test::op(P, {{0, true, false}})};
        Constructor c(p);
        c.ledger.append(0, {Command::Publish, Q, R, 0}, EndpointPurpose::Completion);
        const auto supportAnchor = c.ledger.append(0, {Command::Acquire, Q, R, 0}, EndpointPurpose::Completion);
        const auto supportPub = c.ledger.append(0, {Command::Publish, R, Q, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, supportAnchor);
        const auto supportWait = c.ledger.append(0, {Command::Acquire, R, Q, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, supportAnchor);
        c.rememberReturn(supportPub, supportWait);
        const auto actualReturn = [&](Cut cut) {
            c.ledger.append(cut, {Command::Publish, R, Q, 1}, EndpointPurpose::Completion);
            return c.ledger.append(cut, {Command::Acquire, R, Q, 1}, EndpointPurpose::Completion);
        };
        Id actual = NoAnalysisId;
        if (position == 0) { actual = actualReturn(1); }
        c.ledger.append(1, {Command::Publish, P, Q, 0}, EndpointPurpose::Completion);
        const auto latentAnchor = c.ledger.append(1, {Command::Acquire, P, Q, 0}, EndpointPurpose::Completion);
        const auto latentPub = c.ledger.append(1, {Command::Publish, Q, P, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, latentAnchor);
        const auto latentWait = c.ledger.append(1, {Command::Acquire, Q, P, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, latentAnchor);
        c.rememberReturn(latentPub, latentWait);
        c.deferReturn(latentWait);
        if (position != 0) { actual = actualReturn(position == 1 ? 1 : 2); }
        selected_test::require(c.returnBeforeUse(supportWait, actual) == (position == 0),
            "latent publication support ignored the actual within-word gap");
        c.rearming.at(latentWait).deferred = false;
        selected_test::require(c.returnBeforeUse(supportWait, actual),
            "inactive latent-use index kept imposing a support obligation");
    }
    static void skippedLatentSupport()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE3;
        auto p = selected_test::base(1, 2);
        p.operations = {selected_test::op(P, {{0, true, false}})};
        ObservedControl graph;
        graph.entry = 0;
        graph.exit = 5;
        graph.qualification = "original choice with a skipped replacement receipt";
        const std::vector<std::vector<Id>> edges{{1}, {2, 3}, {4}, {4}, {5}, {}};
        for (Id site = 0; site < edges.size(); ++site) {
            graph.observations.push_back({site, {}, true});
            graph.sites.push_back({NoAnalysisId, site, edges[site], {}, 0});
        }
        graph.sites[0].operation = 0;
        p.observed = graph;
        Constructor c(p);
        c.needsContextualReplay = true;
        c.ledger.append(0, {Command::Publish, Q, R, 0}, EndpointPurpose::Completion);
        const auto anchor = c.ledger.append(0, {Command::Acquire, Q, R, 0}, EndpointPurpose::Completion);
        const auto pub = c.ledger.append(0, {Command::Publish, R, Q, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, anchor);
        const auto wait = c.ledger.append(0, {Command::Acquire, R, Q, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, anchor);
        c.rememberReturn(pub, wait);
        c.ledger.append(2, {Command::Publish, R, Q, 1}, EndpointPurpose::Completion);
        const auto actual = c.ledger.append(2, {Command::Acquire, R, Q, 1}, EndpointPurpose::Completion);
        c.ledger.append(4, {Command::Publish, P, Q, 0}, EndpointPurpose::Completion);
        const auto forward = c.ledger.append(4, {Command::Acquire, P, Q, 0}, EndpointPurpose::Completion);
        const auto latentPub = c.ledger.append(4, {Command::Publish, Q, P, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, forward);
        const auto latentWait = c.ledger.append(4, {Command::Acquire, Q, P, 0},
            EndpointPurpose::ConsumptionAcknowledgment, 0, forward);
        c.rememberReturn(latentPub, latentWait);
        c.deferReturn(latentWait);
        c.result.work.acknowledgments = 1;
        c.current = graph.exit;
        c.activeComponent = c.control.component[c.current];
        selected_test::require(c.update(), c.result.reason);
        SelectedDecision decision;
        decision.endpoints = {actual};
        selected_test::require(c.settleRearming(decision), c.result.reason);
        selected_test::require(c.ledger.active(wait) && c.result.work.latentSupportRetained != 0,
            "helper deletion ignored a path reaching latent support without the actual replacement");
        selected_test::require(checkCausalFrontier(p, c.ledger.commands()).accepted,
            "latent support fixture has an invalid selected protocol");
    }
    static SelectedPlan contextual(const Program& program)
    {
        Constructor constructor(program);
        constructor.needsContextualReplay = true;
        return constructor.run({});
    }
};
}
using namespace selected_test;
namespace {
constexpr auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
void reuseThroughRequiredOverlapReadiness()
{
    for (bool connected : {true, false}) {
        auto p = base(3, 2);
        p.cells[2].storage = o::Cell::Storage::OverlapWitness;
        p.operations = {op(P, {{0, false, true, true}}),
                        op(Q, {{0, true, false}, {1, false, true, true}}),
                        op(R, {{2, false, true}}),
                        op(P, {{0, false, true, true}, {2, true, false}})};
        if (connected) { p.operations[2].accesses.push_back({1, true, false}); }
        p.body = seq({leaf(0), leaf(1), leaf(2), leaf(3)});
        const auto plan = accepted(p);
        const auto direct = std::count_if(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 3 && d.source == Q;
        });
        require(direct == (connected ? 0 : 1),
                "overlap readiness must carry actual reader completion to replace a direct return");
        require(bool(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3})),
                "three-engine completion support failed the independent oracle");
        if (!connected) { continue; }
        require(plan.decisions.size() == 3 && plan.decisions.back().source == R &&
                plan.decisions.back().stage == o::RequirementStage::Known,
                "necessary overlap provider was considered after the known reuse repair");
        // Removing the supporting middle handoff must expose the old reader;
        // a direction-level route or a future promised receipt is insufficient.
        auto broken = plan.commands;
        for (auto& word : broken) { word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
            return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) &&
                   c.source == Q && c.observer == R;
        }), word.end()); }
        require(!o::checkCausalFrontier(p, broken).accepted &&
                !bool(oahs_oracle::graph(p, broken, {0, 1, 2, 3})),
                "removing supporting readiness retained manufactured completion");

        p.body = {o::Region::For, {p.body}, 0, true};
        const auto repeated = accepted(p);
        require(std::none_of(repeated.decisions.begin(), repeated.decisions.end(), [](const auto& d) {
            return d.consumer == 3 && d.source == Q;
        }), "repeated indirect readiness acquired a duplicate direct reader release");
        for (unsigned count : {0u, 1u, 2u, 4u}) {
            std::vector<unsigned> visits;
            for (unsigned i = 0; i < count; ++i) visits.insert(visits.end(), {0, 1, 2, 3});
            require(bool(oahs_oracle::graph(p, repeated.commands, visits)),
                    "repeated third-engine receipt lost memory or consumption knowledge");
        }
    }
}
void alternativeReturnCoverage()
{
    for (unsigned variant = 0; variant < 3; ++variant) {
        auto p = base(3, 4);
        p.cells[2].storage = o::Cell::Storage::OverlapWitness;
        p.operations = {op(P, {{0, false, true, true}}),
                        op(Q, {{0, true, false}, {1, false, true, true}}),
                        op(R, {{1, true, false}, {2, false, true}}),
                        op(R, {{2, false, true}}), op(Q, {{0, true, false}}),
                        op(P, {{0, false, true, true}, {2, true, false}})};
        if (variant != 1) { p.operations[3].accesses.push_back({1, true, false}); }
        o::ObservedControl graph;
        graph.qualification = "required return from alternative original readers";
        graph.entry = 0; graph.exit = 10;
        const std::vector<std::size_t> operations{0, 1, o::NoAnalysisId, 2, o::NoAnalysisId,
            o::NoAnalysisId, 3, o::NoAnalysisId, variant == 2 ? 4u : o::NoAnalysisId, 5, o::NoAnalysisId};
        const std::vector<std::vector<std::size_t>> edges{
            {1}, {2}, {3, 6}, {4}, {5}, {8}, {7}, {8}, {9}, {10}, {}};
        for (std::size_t site = 0; site < operations.size(); ++site) {
            graph.observations.push_back({site, {}, true});
            graph.sites.push_back({operations[site], site, edges[site], {}, 0});
        }
        // Every represented physical phase must occur in the original graph.
        if (variant != 2) {
            p.operations.erase(p.operations.begin() + 4);
            graph.sites[9].operation = 4;
        }
        p.observed = std::move(graph);
        const auto plan = accepted(p);
        const auto found = std::find_if(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 9 && d.source == R && !d.supporting.empty();
        });
        require((found != plan.decisions.end()) == (variant == 0),
                "alternative provider used missing or refreshed extra coverage");
        if (variant == 0) {
            require(found->publicationFrontier == std::vector<o::Cut>{4, 7},
                    "additional coverage moved the selected alternative source boundaries");
        }
        for (const auto& path : {std::vector<o::Cut>{0, 1, 2, 3, 4, 5, 8, 9, 10},
                                 std::vector<o::Cut>{0, 1, 2, 6, 7, 8, 9, 10}}) {
            auto flat = p;
            flat.observed.reset(); flat.operations.clear(); flat.body = {};
            o::Commands words;
            std::vector<o::Command> pending;
            std::vector<unsigned> visits;
            for (auto site : path) {
                pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
                const auto operation = p.observed->sites[site].operation;
                if (operation == o::NoAnalysisId) { continue; }
                visits.push_back(unsigned(flat.operations.size()));
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            words.push_back(std::move(pending));
            require(bool(oahs_oracle::graph(flat, words, visits)),
                    "alternative actual return failed independent memory/event validation");
        }
    }
}
void loopReturnCoverage()
{
    for (bool refresh : {false, true}) {
        auto p = base(3, 4);
        p.cells[2].storage = o::Cell::Storage::OverlapWitness;
        p.operations = {op(P, {{0, false, true, true}}),
                        op(Q, {{0, true, false}, {1, false, true, true}}),
                        op(R, {{1, true, false}, {2, false, true}}),
                        op(P, {{0, false, true, true}, {2, true, false}})};
        if (refresh) { p.operations.push_back(op(Q, {{0, true, false}})); }
        o::ObservedControl graph;
        graph.qualification = "actual extra return coverage through original loop entry";
        graph.entry = 0; graph.exit = 8;
        const std::vector<std::size_t> operations{0, 1, 2, o::NoAnalysisId, o::NoAnalysisId,
            refresh ? 4u : o::NoAnalysisId, 3, o::NoAnalysisId, o::NoAnalysisId};
        const std::vector<std::vector<std::size_t>> edges{{1}, {2}, {3}, {4}, {5, 8}, {6}, {7}, {4}, {}};
        for (std::size_t site = 0; site < operations.size(); ++site) {
            graph.observations.push_back({site, {}, true});
            graph.sites.push_back({operations[site], site, edges[site], {}, 0});
        }
        graph.sites[7].backedgeOwners = {3};
        graph.loops.push_back({3, 3, 8, {4, 5, 6, 7}, 5, true});
        p.observed = std::move(graph);
        const auto plan = accepted(p);
        const bool promoted = std::any_of(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 6 && d.source == R && !d.supporting.empty();
        });
        require(promoted == !refresh, "loop entry promoted extra history invalidated by a child access");
        require(refresh || plan.work.loopEntryTransfers != 0,
                "positive loop-entry coverage fixture used no entry transfer");
        for (unsigned iterations : {1u, 2u, 4u}) {
            std::vector<o::Cut> path{0, 1, 2, 3};
            for (unsigned i = 0; i < iterations; ++i) path.insert(path.end(), {4, 5, 6, 7});
            path.insert(path.end(), {4, 8});
            auto flat = p;
            flat.observed.reset(); flat.operations.clear(); flat.body = {};
            o::Commands words;
            std::vector<o::Command> pending;
            std::vector<unsigned> visits;
            for (auto site : path) {
                pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
                const auto operation = p.observed->sites[site].operation;
                if (operation == o::NoAnalysisId) { continue; }
                visits.push_back(unsigned(flat.operations.size()));
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            words.push_back(std::move(pending));
            require(bool(oahs_oracle::graph(flat, words, visits)),
                    "loop-entry shared return failed independent repeated-trace validation");
        }
    }
}
void preserveWinnerCoverageAtExactGap()
{
    for (bool overlap : {false, true}) {
        auto p = base(2, 4);
        if (overlap) { p.cells[0].storage = o::Cell::Storage::OverlapWitness; }
        p.operations = {op(Q, {{1, false, true, true}}), op(P, {{0, false, true, !overlap}}),
                        op(P, {{1, true, false}}), op(R, {{0, true, false}, {1, true, false}})};
        p.body = seq({leaf(0), leaf(1), leaf(2), leaf(3)});
        const auto plan = accepted(p);
        const auto found = std::find_if(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 3 && d.source == P;
        });
        require(found != plan.decisions.end(), "required physical provider disappeared");
        // Class-first policy can prefer the exact earlier milestone, then acquire
        // the independent prerequisite directly. Compare the COMPLETE payload
        // order with the former forwarding packet, not its transfer count.
        o::Commands forwarding(p.operations.size()+1);
        forwarding[1].push_back({o::Command::Publish,Q,P,0});
        forwarding[2].push_back({o::Command::Acquire,Q,P,0});
        forwarding[2].push_back({o::Command::Publish,P,R,0});
        forwarding[3].push_back({o::Command::Acquire,P,R,0});
        std::set<std::pair<unsigned,unsigned>> selectedOrder, oldOrder;
        require(bool(oahs_oracle::graph(p,plan.commands,{0,1,2,3},{},nullptr,nullptr,&selectedOrder)) &&
                bool(oahs_oracle::graph(p,forwarding,{0,1,2,3},{},nullptr,nullptr,&oldOrder)),
                "forwarding comparison is not a valid complete protocol");
        require(std::includes(oldOrder.begin(),oldOrder.end(),selectedOrder.begin(),selectedOrder.end()),
                "milestone-preserving normal choices added payload ordering");
    }
}
void keepKnownPrefixSeparateFromOverlap()
{
    auto p = base(2, 2);
    p.cells[1].storage = o::Cell::Storage::OverlapWitness;
    p.operations = {op(P, {{0, false, true, true}}),
                    op(P, {{1, false, true}}),
                    op(Q, {{0, true, false}, {1, true, false}})};
    p.body = seq({leaf(0), leaf(1), leaf(2)});
    o::StorageFrontierAnalysis storage(p);
    bool known = false, overlap = false;
    for (const auto& relation : storage.relationshipsAt(2)) {
        const auto reasons = storage.describeRequirement(relation).reasons;
        known |= relation.cell == 0 && bool(reasons & o::KnownReadiness);
        overlap |= relation.cell == 1 && !(reasons & (o::KnownReadiness | o::KnownReuse));
    }
    require(known && overlap, "fixture did not exercise both priority stages");
    auto plan = accepted(p);
    require(!plan.decisions.empty() && plan.decisions[0].stage == o::RequirementStage::Known &&
            plan.decisions[0].required.size() == 1,
            "known readiness was widened by an additional-overlap demand");
    o::Commands separate(4);
    separate[1].push_back({o::Command::Publish,P,Q,0});
    separate[2].push_back({o::Command::Publish,P,Q,1});
    separate[2].push_back({o::Command::Acquire,P,Q,0});
    separate[2].push_back({o::Command::Acquire,P,Q,1});
    std::set<std::pair<unsigned,unsigned>> actual, reference;
    require(bool(oahs_oracle::graph(p,plan.commands,{0,1,2},{},nullptr,nullptr,&actual)) &&
            bool(oahs_oracle::graph(p,separate,{0,1,2},{},nullptr,nullptr,&reference)) && actual==reference,
            "independently required later provider changed complete payload order at the shared deadline");
}
void keepDifferentDeadlines()
{
    auto p = base(2, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    p.body = seq({leaf(0), leaf(1), leaf(2), leaf(3)});
    o::selected::Control control(p);
    o::StorageFrontierAnalysis storage(p);
    o::selected::RequirementFrontiers frontiers(p, control, storage);
    require(frontiers.complete() && frontiers.size() == 2,
            "first pass did not retain the two readiness requirements");
    require(frontiers.sourceBoundaries() == 2,
            "acyclic frontier qualification was not recorded");
    require(frontiers.occurrenceCounts()[unsigned(o::selected::RequirementOccurrence::Acyclic)] == 2,
            "acyclic occurrence class was not recorded");
    const auto& first = frontiers.at(2);
    const auto& second = frontiers.at(3);
    require(first.size() == 1 && first.front().publication == 1 && first.front().deadline == 2,
            "first source boundary/deadline pair changed");
    require(second.size() == 1 && second.front().publication == 2 && second.front().deadline == 3,
            "second source boundary/deadline pair changed");
    require(first.front().source == second.front().source &&
            first.front().observer == second.front().observer,
            "fixture does not exercise one pipeline pair with distinct frontiers");
    const auto plan = accepted(p);
    require(plan.decisions.size() == 2, "different consumers were merged");
    require(plan.decisions[0].publication == 1 && plan.decisions[0].consumer == 2,
            "first consumer now waits for the second load");
    require(plan.decisions[1].consumer == 3, "second deadline lost");
    require(bool(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3}, {{1, 2}})),
            "first consumer unnecessarily waits for completion of the second load");
}
o::Program joinedTerminal()
{
    auto p = base(1, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{0, false, true, true}}),
                    op(Q, {{0, true, false}})};
    p.body = seq({{o::Region::Choice, {leaf(0), leaf(1)}}, leaf(2)});
    return p;
}
void alternativeEarlySources()
{
    auto p = base(3, 2);
    p.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}),
                    op(P, {{0, false, true}}), op(P, {{2, false, true}}),
                    op(Q, {{0, true, false}})};
    o::ObservedControl graph;
    graph.qualification = "test-original-branch-cuts";
    graph.entry = 0;
    graph.exit = 8;
    const std::vector<std::size_t> operations{
        o::NoAnalysisId, 0, 1, o::NoAnalysisId, 2, 3, o::NoAnalysisId, 4, o::NoAnalysisId};
    const std::vector<std::vector<std::size_t>> successors{{1, 4}, {2}, {3}, {7}, {5}, {6}, {7}, {8}, {}};
    for (std::size_t site = 0; site < operations.size(); ++site) {
        graph.observations.push_back({site, {}, true});
        graph.sites.push_back({operations[site], site, successors[site], {}, 0});
    }
    p.observed = graph;
    const auto plan = accepted(p);
    require(plan.decisions.size() == 1 &&
            plan.decisions[0].publicationFrontier == std::vector<o::Cut>{2, 5},
            "missing alternative publication frontier immediately after the required writers");
    require(count(plan, o::Command::Publish) == 2 && count(plan, o::Command::Acquire) == 1,
            "alternative sources need one SET per arm, one shared WAIT");
    require(plan.commands[7].size() == 1 && plan.commands[7][0].kind == o::Command::Acquire,
            "late whole-prefix publication remains at the join");
    const auto& before = *plan.certificate.cuts[7].beforeIssue.facts();
    for (unsigned cell : {1u, 2u}) {
        const auto access = (std::size_t(cell) * o::PipeCount + unsigned(P)) * 2 + 1;
        const auto* history = before.history.find(access);
        require(history && !o::frontierContains(*history, unsigned(Q)),
                "consumer unnecessarily acquires unrelated branch-load completion");
    }
    // Absence from a must-state is not alone a proof that an unwanted edge is
    // absent concretely. Check both original branch paths using the unchanged
    // independent full-history graph oracle as well. Fold control-only words
    // into the next real payload; do not introduce dummy payload operations.
    for (const auto& path : {std::vector<o::Cut>{0, 1, 2, 3, 7, 8},
                             std::vector<o::Cut>{0, 4, 5, 6, 7, 8}}) {
        auto flat = p;
        flat.observed.reset();
        flat.body = {};
        flat.operations.clear();
        o::Commands words;
        std::vector<o::Command> pending;
        for (auto site : path) {
            pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
            const auto operation = p.observed->sites[site].operation;
            if (operation == o::NoAnalysisId) continue;
            flat.operations.push_back(p.operations[operation]);
            words.push_back(std::move(pending));
            pending.clear();
        }
        words.push_back(std::move(pending));
        require(flat.operations.size() == 3, "unexpected branch-path payload population");
        require(bool(oahs_oracle::graph(flat, words, {0, 1, 2}, {{1, 2}})),
                "branch protocol imposes unrelated-load completion before consumer issue");
    }
    // A backward source search cannot erase an earlier source occurrence on a
    // bypass path. Add a path which bypasses both writers: its missing source
    // participation must make the early-frontier candidate unavailable.
    auto bypass = p;
    bypass.observed->sites[0].successors.push_back(7);
    const auto conservative = accepted(bypass);
    require(std::all_of(conservative.decisions.begin(), conservative.decisions.end(),
        [](const auto& decision) { return decision.publicationFrontier.empty(); }),
        "branch bypass was silently matched to a nonparticipating publisher");
}
void noUnusedTerminalReturn()
{
    auto p = joinedTerminal();
    p.target.keys[unsigned(Q)][unsigned(P)].clear();
    const auto plan = accepted(p);
    require(plan.decisions.size() == 1 && plan.decisions[0].commonCut,
            "fixture did not select a common-cut transfer");
    require(count(plan, o::Command::Publish) == 1 && count(plan, o::Command::Acquire) == 1,
            "terminal common cut still requires a reverse key");
    require(plan.work.acknowledgments == 0, "unused terminal consumption was acknowledged");
}
void deferFuturePayloadReturn()
{
    auto p = joinedTerminal();
    p.operations.push_back(op(R, {{0, true, false}}));
    p.body.children.push_back(leaf(3));
    const auto plan = accepted(p);
    require(plan.work.rearmingDeferred != 0 && plan.work.acknowledgments == 0,
            "unrelated future payload forced a return before an actual key deadline");
}
void keepFutureWordReturn()
{
    const auto p = joinedTerminal();
    o::Commands fixed(o::commandCutCount(p));
    fixed[o::invocationExitCut(p)] = {
        {o::Command::Publish, o::Pipe::MTE1, R, 0},
        {o::Command::Acquire, o::Pipe::MTE1, R, 0}};
    const auto plan = accepted(p, fixed);
    require(plan.work.rearmingDeferred != 0 && plan.work.acknowledgments == 0,
            "unrelated fixed event identities forced an immediate return");
}
void deferredResourcePressure()
{
    for (unsigned variant = 0; variant < 4; ++variant) {
        const bool reverse = variant == 1;
        auto p = joinedTerminal();
        p.cells.push_back(p.cells.front());
        p.target.keys[unsigned(P)][unsigned(Q)] = {0};
        p.target.keys[unsigned(Q)][unsigned(P)] = {0};
        const auto producer = reverse ? Q : P;
        const auto consumer = reverse ? P : Q;
        p.operations.push_back(op(producer, {{1, false, true, true}}));
        p.operations.push_back(op(consumer, {{1, true, false}}));
        p.body.children.push_back(leaf(3));
        p.body.children.push_back(leaf(4));
        if (variant == 2) {
            const auto first = seq({p.body.children[0], p.body.children[1]});
            p.body = seq({{o::Region::Choice, {first, seq({})}}, leaf(3), leaf(4)});
        }
        o::Commands fixed;
        if (variant == 3) {
            const auto bounded = o::addStructuredBoundaryCuts(p);
            require(bounded.success, bounded.reason);
            p = bounded.program;
            o::selected::Control control(p);
            fixed.resize(o::commandCutCount(p));
            fixed[control.graph.entry].push_back({o::Command::Publish, Q, P, 1});
            const auto at = std::find(control.graph.operations.begin(), control.graph.operations.end(), 3);
            require(at != control.graph.operations.end(), "stale-return fixture lost its later payload");
            fixed[at - control.graph.operations.begin()].push_back({o::Command::Acquire, Q, P, 1});
            p.target.keys[unsigned(Q)][unsigned(P)] = {0, 1};
        }
        const auto plan = accepted(p, fixed);
        require(plan.work.rearmingDeferred != 0 && plan.work.deferredMaterialized != 0,
                "actual key pressure did not close the reserved deferred fallback");
        require(plan.work.ownershipBindings != 0,
                "deferred fallback bypassed complete ownership qualification");
    }
}
void restoreAtReuseDeadline()
{
    auto p = base(4, 2);
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    p.target.keys[unsigned(Q)][unsigned(P)] = {0};
    p.operations = {op(Q, {{3, false, true}}),
        op(P, {{0, false, true, true}, {1, false, true, true}}),
        op(P, {{0, false, true, true}, {1, false, true, true}}),
        op(Q, {{0, true, false}}), op(R, {{1, true, false}}),
        op(P, {{2, false, true, true}}), op(Q, {{2, true, false}})};
    p.body = seq({leaf(0), {o::Region::Choice, {leaf(1), leaf(2)}},
                  leaf(3), leaf(4), leaf(5), leaf(6)});
    const auto plan = accepted(p);
    require(plan.work.deadlineRestorations == 1 && plan.restorations.size() == 1,
            "normal construction did not restore at the selected republication deadline");
    const auto& restored = plan.restorations.front();
    const auto& wait = plan.ledger[restored.acquisition];
    require(restored.fallbackCut == 3 && restored.placedCut == 6 && restored.fallbackReason.empty(),
            "restoration lost original gap or selected deadline provenance");
    require(plan.ledger[restored.deadlinePublication].cut == wait.cut,
            "restoration deadline does not name the actual selected publication");
    auto old = plan.commands;
    auto& late = old[wait.cut];
    const auto found = std::find_if(late.begin(), late.end(), [&](const auto& command) {
        return o::selected::identical(command, wait.command);
    });
    require(found != late.end(), "restored wait missing from emitted word");
    late.erase(found);
    old[restored.fallbackCut].push_back(wait.command);
    require(o::checkCausalFrontier(p, old).accepted, "original-gap reference is not legal");
    for (const auto& trace : oahs_oracle::expand(p.body, 1, 100)) {
        std::set<std::pair<unsigned, unsigned>> before, after;
        require(bool(oahs_oracle::graph(p, old, trace, {}, nullptr, nullptr, &before)),
                "original-gap reference failed independent validation");
        require(bool(oahs_oracle::graph(p, plan.commands, trace, {{0, 3}}, nullptr, nullptr, &after)),
                "late restoration imported unrelated completion into another publication");
        require(after.size() < before.size() && std::includes(before.begin(), before.end(), after.begin(), after.end()),
                "deadline placement did not strictly remove complete payload ordering");
    }
    auto missing = plan.commands;
    missing[wait.cut].erase(missing[wait.cut].begin());
    require(!o::checkCausalFrontier(p, missing).accepted,
            "republication accepted without actual consumption knowledge");
    auto tooLate = plan.commands;
    std::swap(tooLate[wait.cut][0], tooLate[wait.cut][1]);
    require(!o::checkCausalFrontier(p, tooLate).accepted, "wait after republication granted anticipated credit");
}
void keepLoopReturn()
{
    auto p = base(1, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
    const auto plan = accepted(p);
    require(plan.work.acknowledgments + plan.work.rearmingDischarged != 0 && plan.work.rearmingDeferred == 0,
            "recurring consumption was incorrectly admitted to acyclic deferral");
    auto missingReturn = plan.commands;
    for (auto& word : missingReturn) word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
        return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) && c.source == Q && c.observer == P;
    }), word.end());
    require(!o::checkCausalFrontier(p, missingReturn).accepted,
            "loop republication lost its actual consumption path");
    o::selected::Control control(p);
    for (std::size_t site = 0; site < control.graph.sites.size(); ++site) {
        if (control.graph.operations[site] == 1)
            require(control.lookahead.mayIssueAfter(site), "loop lookahead erased backedge");
    }
}
o::Program invariantLoopProgram(unsigned keys = 4)
{
    auto p = base(3, keys);
    p.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}),
                    op(Q, {{0, true, false}})};
    o::ObservedControl q;
    q.qualification = "test-original-nonempty-loop-entry";
    q.entry = 0; q.exit = 7;
    const std::vector<std::size_t> operations{0, o::NoAnalysisId, 1, o::NoAnalysisId,
        o::NoAnalysisId, 2, o::NoAnalysisId, o::NoAnalysisId};
    const std::vector<std::vector<std::size_t>> edges{{1}, {2}, {3}, {4}, {5, 7}, {6}, {4}, {}};
    for (std::size_t i = 0; i < operations.size(); ++i) {
        q.observations.push_back({i, {}, true});
        q.sites.push_back({operations[i], i, edges[i], {}, 0});
    }
    q.sites[6].backedgeOwners = {3};
    q.loops.push_back({3, 3, 7, {4, 5, 6}, 5, true});
    p.observed = q;
    return p;
}
void invariantLoopEntry()
{
    auto p = invariantLoopProgram();
    const o::selected::Control original(p);
    require(original.loopEntries.size() == 1 &&
            original.loopEntries.front().firstConsumer[unsigned(Q)] == 5,
            "first pass lost the unique observer deadline");
    const auto plan = accepted(p);
    require(plan.work.loopEntryTransfers == 1, "invariant readiness not acquired at loop entry");
    require(plan.commands[1].size() == 1 && plan.commands[1][0].kind == o::Command::Publish,
            "loop-entry readiness includes unrelated source work");
    require(plan.commands[3].size() == 1 && plan.commands[3][0].kind == o::Command::Acquire,
            "invariant readiness must be acquired once per entry");
    require(plan.commands[5].empty(), "loop body repeats invariant acquisition");
    // Fold control-only entry words into the first body visit for a concrete
    // three-visit trace, retaining the actual command order and payload sites.
    auto flat = p;
    flat.observed.reset(); flat.operations.clear(); flat.body = {};
    o::Commands words;
    std::vector<o::Command> pending;
    for (auto site : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 4u, 5u, 6u, 4u, 5u, 6u, 4u, 7u}) {
        pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
        const auto operation = p.observed->sites[site].operation;
        if (operation == o::NoAnalysisId) continue;
        flat.operations.push_back(p.operations[operation]);
        words.push_back(std::move(pending)); pending.clear();
    }
    words.push_back(std::move(pending));
    require(bool(oahs_oracle::graph(flat, words, {0, 1, 2, 3, 4}, {{1, 2}, {1, 3}, {1, 4}})),
            "unrelated load orders a loop consumer");
    auto earlier = p;
    earlier.operations.push_back(op(Q, {}));
    auto& entry = *earlier.observed;
    entry.observations.push_back({8, {}, true});
    entry.sites.push_back({3, 8, {5}, {}, 0});
    entry.sites[4].successors[0] = 8;
    entry.loops.front().bodyEntry = 8;
    entry.loops.front().sites.push_back(8);
    require(accepted(earlier).work.loopEntryTransfers == 0,
            "entry acquisition unnecessarily gates an earlier observer payload");
    p.observed->loops.front().atLeastOnce = false;
    require(accepted(p).work.loopEntryTransfers == 0, "unknown/zero-trip entry was acquired unconditionally");
}
void reuseOneShotEntryKey()
{
    auto p = invariantLoopProgram(1);
    p.operations.push_back(op(P, {{2, false, true}}));
    p.operations.push_back(op(Q, {{2, true, false}}));
    auto& q = *p.observed;
    q.exit = 10;
    q.sites[7].successors = {8};
    for (o::Cut i = 8; i <= 10; ++i) {
        q.observations.push_back({i, {}, true});
        q.sites.push_back({i < 10 ? i - 5 : o::NoAnalysisId, i,
            i < 10 ? std::vector<o::Cut>{i + 1} : std::vector<o::Cut>{}, {}, 0});
    }
    o::Commands fixed(o::commandCutCount(p));
    fixed[7] = {{o::Command::Publish, Q, P, 0}, {o::Command::Acquire, Q, P, 0}};
    const auto plan = accepted(p, fixed);
    require(plan.work.loopEntryTransfers == 1 && plan.commands[1].size() == 1 &&
            plan.commands[1][0].kind == o::Command::Publish && plan.commands[3].size() == 1 &&
            plan.commands[3][0].kind == o::Command::Acquire,
            "one-shot key reuse lost the useful early entry placement");
    require(plan.work.acknowledgments == 0,
            "valid fixed return was ignored when proving one-shot key reuse");
    unsigned publications = 0;
    for (const auto& endpoint : plan.ledger)
        if (endpoint.command.kind == o::Command::Publish && endpoint.command.source == P &&
            endpoint.command.observer == Q) {
            require(endpoint.command.key == 0, "single-key pool was enlarged");
            ++publications;
        }
    require(publications == 2, "consumed one-shot key was permanently reserved");
    require(std::any_of(plan.updates.begin(), plan.updates.end(),
                       [](const auto& update) { return update.contextual; }),
            "ending a compiler reservation disabled contextual replay");
    auto missingReturn = plan.commands;
    missingReturn[7].clear();
    require(!o::checkCausalFrontier(p, missingReturn).accepted,
            "ending a reservation manufactured consumption knowledge");
}

void helperFreeLoopEntry()
{
    auto p = base(2, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true, true}}), op(Q, {{1, true, false}})};
    o::ObservedControl g;
    g.entry = 0; g.exit = 8; g.qualification = "bypass producer then a distinct reader child";
    const std::vector<o::Cut> operations{o::NoAnalysisId, 0, 1, 2, o::NoAnalysisId,
                                       o::NoAnalysisId, 3, o::NoAnalysisId, o::NoAnalysisId};
    const std::vector<std::vector<o::Cut>> edges{{1, 2}, {2}, {3}, {4}, {5}, {6, 8}, {7}, {5}, {}};
    for (o::Cut site = 0; site < operations.size(); ++site) {
        g.observations.push_back({site, {}, true});
        g.sites.push_back({operations[site], site, edges[site], {}, 0});
    }
    g.sites[7].backedgeOwners = {4};
    g.loops.push_back({4, 4, 8, {5, 6, 7}, 6, true});
    p.observed = g;
    const auto plan = accepted(p);
    require(plan.work.rearmingDeferred != 0 && plan.work.loopEntryTransfers == 1,
            "structured helper-free fixture did not exercise a dormant owner and entry request");
    require(plan.work.rearmingRestored == 0 && plan.work.acknowledgments == 0,
            "low dormant key was restored before examining the higher helper-free key");
    require(plan.commands[4].size() == 2 && plan.commands[4][0].kind == o::Command::Publish &&
            plan.commands[4][0].key == 1 && plan.commands[4][1].kind == o::Command::Acquire &&
            plan.commands[4][1].key == 1,
            "loop-entry helper-free key did not preserve the frozen endpoints");
    for (bool producer : {false, true}) {
        for (unsigned trips : {1u, 2u, 4u}) {
            std::vector<o::Cut> path{0};
            if (producer) { path.push_back(1); }
            path.insert(path.end(), {2, 3, 4});
            for (unsigned i = 0; i < trips; ++i) { path.insert(path.end(), {5, 6, 7}); }
            path.insert(path.end(), {5, 8});
            auto flat = p; flat.observed.reset(); flat.operations.clear(); flat.body = {};
            o::Commands words;
            std::vector<o::Command> pending;
            std::vector<unsigned> visits;
            for (auto site : path) {
                pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
                const auto operation = g.sites[site].operation;
                if (operation == o::NoAnalysisId) { continue; }
                visits.push_back(unsigned(flat.operations.size()));
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            words.push_back(std::move(pending));
            require(bool(oahs_oracle::graph(flat, words, visits)),
                    "helper-free entry failed independent occurrence/rearming validation");
        }
    }
}

void entryWaitMustNotCrossPublication()
{
    auto p = invariantLoopProgram();
    p.operations.push_back(op(R, {}));
    auto& q = *p.observed;
    q.observations.push_back({8, {}, true});
    q.sites.push_back({3, 8, {5}, {}, 0});
    q.sites[4].successors[0] = 8;
    q.loops.front().bodyEntry = 8;
    q.loops.front().sites.push_back(8);
    const std::vector<o::Command> relay{
        {o::Command::Publish, Q, R, 0}, {o::Command::Acquire, Q, R, 0},
        {o::Command::Publish, R, Q, 0}, {o::Command::Acquire, R, Q, 0}};
    o::Commands fixed(o::commandCutCount(p));
    fixed[8] = relay;
    const auto plan = accepted(p, fixed);
    require(plan.work.loopEntryTransfers == 0,
            "entry acquisition crossed an observer publication to another engine");
    auto flat = p;
    flat.observed.reset(); flat.operations.clear(); flat.body = {};
    o::Commands words;
    std::vector<o::Command> pending;
    for (auto site : {0u, 1u, 2u, 3u, 4u, 8u, 5u, 6u,
                     4u, 8u, 5u, 6u, 4u, 8u, 5u, 6u, 4u, 7u}) {
        pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
        const auto operation = q.sites[site].operation;
        if (operation == o::NoAnalysisId) continue;
        flat.operations.push_back(p.operations[operation]);
        words.push_back(std::move(pending)); pending.clear();
    }
    words.push_back(std::move(pending));
    require(bool(oahs_oracle::graph(flat, words, {0, 1, 2, 3, 4, 5, 6, 7}, {{0, 2}})),
            "entry placement ordered A completion before independent R work");

    // A fixed publication in the deadline's word also precedes the original
    // acquisition; checking only strictly earlier sites is insufficient.
    fixed[8].clear(); fixed[5] = relay;
    require(accepted(p, fixed).work.loopEntryTransfers == 0,
            "entry placement crossed the deadline's existing publication");

    // Keep useful placements when these words are not crossed: the new WAIT
    // is appended after entry commands, and body-exit commands follow Q's use.
    fixed[5].clear(); fixed[3] = relay;
    require(accepted(p, fixed).work.loopEntryTransfers == 1,
            "publication before the entry WAIT disabled useful early placement");
    fixed[3].clear(); fixed[6] = relay;
    require(accepted(p, fixed).work.loopEntryTransfers == 1,
            "publication after the consumer disabled useful early placement");
}
// A region with alternative first writers, re-entered after a foreign reader.
// The source does no work inside the child; no kernel/opcode recognition is used.
o::Program enclosingChoice()
{
    auto p = base(1, 3);
    p.operations = {op(Q, {{0, false, true}}), op(Q, {{0, false, true}}),
                    op(P, {{0, true, false}})};
    o::ObservedControl q;
    q.entry = 0; q.exit = 8; q.qualification = "original nested choice/re-entry";
    for (o::Cut i = 0; i <= 8; ++i) {
        q.observations.push_back({i, {}, true});
        q.sites.push_back({o::NoAnalysisId, i, {}, {}, 0});
    }
    q.sites[0].successors = {1, 8};
    q.sites[1].successors = {2};
    q.sites[2].successors = {3, 4};
    q.sites[3].operation = 0; q.sites[3].successors = {5};
    q.sites[4].operation = 1; q.sites[4].successors = {5};
    q.sites[5].successors = {2, 6}; q.sites[5].backedgeOwners = {1, o::NoAnalysisId};
    q.sites[6].successors = {7};
    q.sites[7].operation = 2; q.sites[7].successors = {1, 8};
    q.sites[7].backedgeOwners = {0, o::NoAnalysisId};
    q.loops.push_back({1, 1, 6, {2, 3, 4, 5}, 2, true});
    p.observed = std::move(q);
    return p;
}
void normalDormantOwnership()
{
    // One selected generation spans a choice child and parent continuation.
    // These are ordinary physical effects; discovery must use the public path.
    auto p = base(4, 3);
    p.operations = {op(Q, {{0, false, true}}), op(Q, {{0, false, true}}),
        op(P, {{0, true, false}}), op(P, {{1, true, false}}),
        op(P, {{1, true, true}}), op(P, {{2, true, true}}),
        op(P, {{1, false, true}}), op(Q, {{0, false, true}}),
        op(Q, {{1, true, true}}), op(Q, {{2, true, true}}),
        op(R, {{1, true, false}}), op(R, {{1, false, true}}),
        op(Q, {{0, true, true}}), op(Q, {{1, true, true}})};
    o::ObservedControl g;
    g.qualification = "original choice child, enclosing recurrence and parent continuation";
    g.entry = 0;
    g.exit = 20;
    for (unsigned i = 0; i < 21; ++i) {
        g.observations.push_back({i, {}, true});
        g.sites.push_back({o::NoAnalysisId, i, {}, {}, 0});
    }
    g.sites[0].successors = {1, 8};
    g.sites[1].successors = {2};
    g.sites[2].successors = {3, 4};
    g.sites[3].operation = 0; g.sites[3].successors = {5};
    g.sites[4].operation = 1; g.sites[4].successors = {5};
    g.sites[5].successors = {2, 6}; g.sites[5].backedgeOwners = {1, o::NoAnalysisId};
    g.sites[6].successors = {7};
    g.sites[7].operation = 2; g.sites[7].successors = {9};
    for (unsigned i = 9; i < 20; ++i) {
        g.sites[i].operation = i - 6;
        g.sites[i].successors = {i + 1};
    }
    g.sites[13].successors = {1, 8}; g.sites[13].backedgeOwners = {0, o::NoAnalysisId};
    g.sites[8].successors = {14};
    g.loops.push_back({1, 1, 6, {2, 3, 4, 5}, 2, true});
    p.observed = g;
    const auto plan = accepted(p);
    require(plan.work.ownershipBindings != 0, "normal constructor did not discover dormant ownership closure");
    for (bool unrelated : {false, true}) {
        auto variant = p;
        // Physical relabeling and independent continuation do not change the
        // ownership fact or introduce a spelling-dependent allocation gate.
        for (auto& operation : variant.operations) {
            for (auto& access : operation.accesses) {
                access.cell = (access.cell + 1) % 3;
            }
        }
        if (unrelated) {
            variant.operations.push_back(op(o::Pipe::MTE1, {{3, true, false}}));
            variant.observed->sites[20].operation = variant.operations.size() - 1;
            variant.observed->sites[20].successors = {21};
            variant.observed->observations.push_back({21, {}, true});
            variant.observed->sites.push_back({o::NoAnalysisId, 21, {}, {}, 0});
            variant.observed->exit = 21;
        }
        const auto selected = accepted(variant);
        require(selected.work.ownershipBindings != 0, "unrelated context erased ownership realization");
    }
}
void changedRepublicationDeadline()
{
    auto p = base(1, 3);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(R, {{0, true, true}}), op(Q, {{0, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0),
        {o::Region::Choice, {seq({leaf(1)}), seq({leaf(2)})}}, leaf(3)})}, 0, true};
    const auto plan = o::selected::ReplayTestAccess::contextual(p);
    require(plan.success, "new publication deadline: " + plan.reason);
    require(plan.work.rearmingRestored != 0, "changed-deadline fixture never restored a helper");
    require(o::checkCausalFrontier(p, plan.commands).accepted, "earlier deadline lost its rearming path");
    for (const auto& visits : oahs_oracle::traces(p, 3))
        require(bool(oahs_oracle::graph(p, plan.commands, visits)), "changed deadline failed independent protocol check");
}
void enclosingAcquisitionAndRearming()
{
    const auto p = enclosingChoice();
    const auto plan = accepted(p);
    require(plan.work.loopEntryTransfers == 1, "alternative first consumers lost enclosing acquisition");
    require(plan.commands[1].size() == 2 && plan.commands[1][0].kind == o::Command::Publish &&
            plan.commands[1][1].kind == o::Command::Acquire,
            "enclosing completion must be acquired once, without an immediate return");
    require(plan.work.rearmingDischarged == 2 && plan.work.acknowledgments == 0,
            "necessary entry/result transfers did not discharge each other's rearming");
    for (auto cut : {3u, 4u}) for (const auto& c : plan.commands[cut])
        require(c.kind != o::Command::Publish && c.kind != o::Command::Acquire,
                "invariant source completion was reacquired inside the region");
    for (auto cut : {1u, 7u}) {
        auto missing = plan.commands;
        missing[cut].clear();
        require(!o::checkCausalFrontier(p, missing).accepted,
                "necessary return deletion manufactured consumption knowledge");
    }
    o::Commands fixed(o::commandCutCount(p));
    fixed[6] = {{o::Command::Publish, P, R, 0}, {o::Command::Acquire, P, R, 0},
                {o::Command::Publish, R, P, 0}, {o::Command::Acquire, R, P, 0}};
    const auto exporting = accepted(p, fixed);
    require(exporting.work.loopEntryTransfers == 1 && exporting.work.acknowledgments != 0,
            "pending rearming ignored a publication before the necessary return");
    auto unrelated = p;
    unrelated.operations[1].accesses.clear();
    require(accepted(unrelated).work.loopEntryTransfers == 0,
            "one branch's deadline was broadened to an unrelated first consumer");
    auto optional = p;
    optional.observed->loops.front().atLeastOnce = false;
    require(accepted(optional).work.loopEntryTransfers == 0,
            "unqualified nonempty region received an enclosing acquisition");
}
} // namespace
int main()
{
    for (unsigned gap = 0; gap < 3; ++gap) { o::selected::ReplayTestAccess::latentSourceSupport(gap); }
    o::selected::ReplayTestAccess::skippedLatentSupport();
    for (auto n : {32u, 64u, 128u}) o::selected::ReplayTestAccess::pairEnumeration(n);
    keepKnownPrefixSeparateFromOverlap();
    reuseThroughRequiredOverlapReadiness();
    alternativeReturnCoverage();
    loopReturnCoverage();
    preserveWinnerCoverageAtExactGap();
    keepDifferentDeadlines();
    alternativeEarlySources();
    noUnusedTerminalReturn();
    deferFuturePayloadReturn();
    keepFutureWordReturn();
    deferredResourcePressure();
    restoreAtReuseDeadline();
    keepLoopReturn();
    normalDormantOwnership();
    changedRepublicationDeadline();
    enclosingAcquisitionAndRearming();
    invariantLoopEntry();
    reuseOneShotEntryKey();
    helperFreeLoopEntry();
    entryWaitMustNotCrossPublication();
    std::cout << "selected lookahead, deadline and terminal-return tests passed\n";
}
