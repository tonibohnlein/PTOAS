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

using namespace selected_test;
namespace {
constexpr auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
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
    require(plan.decisions[0].publication == 1,
            "lookahead moved the known source prefix past the later load");
}
void keepDifferentDeadlines()
{
    auto p = base(2, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true, true}}), op(Q, {{1, true, false}})};
    p.body = seq({leaf(0), leaf(1), leaf(2), leaf(3)});
    const auto plan = accepted(p);
    require(plan.decisions.size() == 2, "different consumers were merged");
    require(plan.decisions[0].publication == 1 && plan.decisions[0].consumer == 1,
            "first consumer now waits for the second load");
    require(plan.decisions[1].consumer == 3, "second deadline lost");
    require(bool(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3}, {{2, 1}})),
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
void keepFuturePayloadReturn()
{
    auto p = joinedTerminal();
    p.operations.push_back(op(R, {{0, true, false}}));
    p.body.children.push_back(leaf(3));
    const auto plan = accepted(p);
    require(plan.work.acknowledgments != 0,
            "terminal optimization crossed a possible future payload");
}
void keepFutureWordReturn()
{
    const auto p = joinedTerminal();
    o::Commands fixed(o::commandCutCount(p));
    fixed[o::invocationExitCut(p)] = {
        {o::Command::Publish, o::Pipe::MTE1, R, 0},
        {o::Command::Acquire, o::Pipe::MTE1, R, 0}};
    const auto plan = accepted(p, fixed);
    require(plan.work.acknowledgments != 0,
            "later fixed event words were ignored by terminal optimization");
}
void keepLoopReturn()
{
    auto p = base(1, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
    const auto plan = accepted(p);
    require(plan.work.acknowledgments != 0, "last body issue was mistaken for invocation exit");
    o::selected::Control control(p);
    for (std::size_t site = 0; site < control.graph.sites.size(); ++site) {
        if (control.graph.operations[site] == 1)
            require(control.lookahead.mayIssueAfter(site), "loop lookahead erased backedge");
    }
}
} // namespace
int main()
{
    keepKnownPrefixSeparateFromOverlap();
    keepDifferentDeadlines();
    alternativeEarlySources();
    noUnusedTerminalReturn();
    keepFuturePayloadReturn();
    keepFutureWordReturn();
    keepLoopReturn();
    std::cout << "selected lookahead, deadline and terminal-return tests passed\n";
}
