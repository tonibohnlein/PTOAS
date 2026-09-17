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
    sourceTimeAndNeighbors();
    retirementAlternatives();
    joinedPrecisionBoundary();
    replayReusesUnchangedPrefix();
    sharedObservationReplay();
    contextualPrefixReuse();
    recurringRoleIsolation();
    std::cout << "selected-ledger update, boundary and refusal tests passed\n";
}
