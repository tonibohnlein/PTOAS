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
    static SelectedPlan contextual(const Program& p) {
        Constructor constructor(p);
        constructor.needsContextualReplay = true;
        return constructor.run({}, false);
    }
};
}
namespace {
const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
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
    const auto staged = ledger.withPacket(packet);
    compareWords(original, ledger.commands());
    require(ledger.version() == revision, "private packet changed live revision");
    const auto ids = ledger.appendPacket(packet);
    compareWords(staged, ledger.commands());
    require(ids.size() == 3 && ledger.endpoint(ids[1]).request == 1,
            "packet lost logical matching provenance");
    require(staged[1].size() == 3 && staged[1][0].kind == o::Command::Barrier &&
                staged[1][1].kind == o::Command::Acquire && staged[1][2].kind == o::Command::Publish,
            "packet sorted publications ahead of an earlier receipt or fixed command");
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
    require(plan.declinedRecurring.has_value(), "optional cohort should be declined");
    require(plan.declinedRecurring->work.recurringProposals != 0 && plan.channels.empty(),
            "optional cohort was not discarded completely");
    compareWords(plan.commands, reference.commands);
    std::cout << "keys=" << keys << " declined=" << plan.declinedRecurring->reason << '\n';
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
    orderedPacketMaterialization();
    resourceAdmission(1);
    resourceAdmission(2);
    deadlineFence();
    separateReleaseFrontiers();
    return 0;
}
