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
constexpr auto P = o::Pipe::MTE1, Q = o::Pipe::M, R = o::Pipe::MTE2;
o::Program fixture(bool repeated = false)
{
    auto p = base(2, 4);
    p.operations = {
        op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}), op(Q, {{0, true, false}}),
        op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    o::ObservedControl g;
    g.qualification = "test-original-choice-frontiers";
    g.sites.resize(8);
    g.entry = 0;
    g.exit = 7;
    g.scopes = {{0, o::NoControlId, o::NoControlId}, {1, 0, 3}, {2, 0, 3}};
    for (unsigned i = 0; i < 8; ++i) {
        g.observations.push_back({i, {}, true});
        g.sites[i].observation = i;
    }
    g.sites[0].successors = repeated ? std::vector<std::size_t>{1, 7} : std::vector<std::size_t>{1};
    g.sites[1].operation = 0;
    g.sites[1].successors = {2};
    g.sites[2].operation = 1;
    g.sites[2].successors = {3};
    g.sites[3].successors = {4, 5};
    g.sites[4].operation = 2;
    g.sites[4].context = 1;
    g.sites[4].successors = {6};
    g.sites[5].operation = 3;
    g.sites[5].context = 2;
    g.sites[5].successors = {6};
    g.sites[6].operation = 4;
    g.sites[6].successors = {repeated ? 0u : 7u};
    if (repeated)
        g.sites[6].backedgeOwners = {0};
    p.observed = g;
    return p;
}
o::SelectedPlan construct(const o::Program& p, bool enabled, const o::Commands& fixed = {})
{
    o::SelectedOptions options;
    options.choiceConsumerFrontiers = enabled;
    options.recurringOmissionTrials = false;
    options.finalHelperTrials = false;
    auto r = o::constructSelectedPlan(p, fixed, options);
    require(r.success, r.reason);
    require(o::verify(p, r.commands).success, "cold choice check");
    return r;
}
oahs_oracle::PayloadOrder order(
    const o::Program& p, const o::Commands& commands, const std::vector<o::Cut>& path, bool forbid)
{
    auto flat = p;
    flat.observed.reset();
    flat.body = {};
    flat.operations.clear();
    o::Commands words;
    std::vector<o::Command> pending;
    std::vector<std::pair<unsigned, unsigned>> forbidden;
    for (auto site : path) {
        pending.insert(pending.end(), commands[site].begin(), commands[site].end());
        auto op = p.observed->sites[site].operation;
        if (op == o::NoControlId)
            continue;
        if (forbid && (op == 2 || op == 3))
            forbidden.emplace_back(flat.operations.size() - 1, flat.operations.size());
        flat.operations.push_back(p.operations[op]);
        words.push_back(std::move(pending));
        pending.clear();
    }
    words.push_back(std::move(pending));
    std::vector<unsigned> visits(flat.operations.size());
    std::iota(visits.begin(), visits.end(), 0);
    oahs_oracle::PayloadOrder out;
    require(
        bool(oahs_oracle::graph(flat, words, visits, forbidden, nullptr, nullptr, &out)), "choice independent oracle");
    return out;
}
void positive()
{
    for (bool repeat : {false, true}) {
        auto p = fixture(repeat);
        auto baseline = construct(p, false);
        auto candidate = construct(p, true);
        require(candidate.work.choiceTransfers == 1, "alternative consumer not selected");
        require(candidate.work.choiceTrials == 1, "choice performs more than one staged solve");
        for (unsigned trips : {0u, 1u, 2u, 4u}) {
            if (!repeat && trips != 1)
                continue;
            for (unsigned mask = 0; mask < (1u << trips); ++mask) {
                std::vector<o::Cut> path;
                for (unsigned i = 0; i < trips; ++i)
                    path.insert(path.end(), {0, 1, 2, 3, (mask & (1u << i)) ? 4u : 5u, 6});
                if (repeat)
                    path.push_back(0);
                path.push_back(7);
                auto a = order(p, baseline.commands, path, false), b = order(p, candidate.commands, path, true);
                require(std::includes(a.begin(), a.end(), b.begin(), b.end()), "choice added payload order");
                require(trips == 0 || b.size() < a.size(), "choice did not remove broad readiness");
            }
        }
        auto broken = candidate.commands;
        broken[2].clear();
        require(!o::verify(p, broken).success, "missing early readiness accepted");
    }
}
// The analysis splits a loop's final visit, while its ordinary command words
// remain shared. Existing placement opportunities must survive that refinement.
o::Program refinedFixture()
{
    auto p = fixture(true);
    auto& g = *p.observed;
    g.sites.resize(9);
    g.observations.push_back({8, {}, true});
    g.sites[8].observation = 8;
    g.sites[8].successors = {0};
    g.entry = 8;
    g.sites[6].backedgeOwners = {8};
    g.loops.push_back({8, 8, 7, {0, 1, 2, 3, 4, 5, 6}, 1, true});
    return p;
}
o::Program refine(const o::Program& p)
{
    o::ReaderVisitRegion request;
    request.owner = 8;
    request.lastPublications = {2};
    request.finalSourceGaps = true;
    auto r = o::refineReaderVisits(p, request);
    require(r.success, r.reason);
    return r.program;
}
std::vector<o::Cut> visits(const o::Program& p, unsigned trips, unsigned mask, bool refined)
{
    const auto& g = *p.observed;
    std::vector<o::Cut> path;
    unsigned iteration = 0;
    auto at = g.entry;
    for (unsigned steps = 0; steps < 200; ++steps) {
        path.push_back(at);
        const auto& site = g.sites[at];
        if (at == g.exit) {
            return path;
        }
        if (site.operation == 4) {
            ++iteration;
        }
        if (at == 0) {
            at = site.successors[refined ? iteration + 1 == trips : iteration == trips];
        } else if (site.observation == 3) {
            at = site.successors[(mask >> iteration) & 1];
        } else {
            require(site.successors.size() == 1, "unexpected path fork");
            at = site.successors.front();
        }
    }
    require(false, "refined path did not terminate");
    return {};
}
void sharedChoice()
{
    const auto original = refinedFixture();
    const auto refined = refine(original);
    const auto before = construct(original, true);
    const auto after = construct(refined, true);
    require(before.work.choiceTransfers == 1, "unrefined opportunity missing");
    require(after.work.choiceTransfers == 1, "refinement lost choice placement");
    for (unsigned trips : {1u, 2u, 4u}) {
        for (unsigned mask = 0; mask < (1u << trips); ++mask) {
            const auto a = order(original, before.commands, visits(original, trips, mask, false), true);
            const auto b = order(refined, after.commands, visits(refined, trips, mask, true), true);
            require(a == b, "analysis refinement changed complete payload order");
        }
    }
    o::selected::Control control(refined);
    require(control.complete, control.reason);
    const auto& paired = control.correspondence(2, 3);
    require(paired.qualified && paired.pairs.size() == 2, "shared visits not paired");
    for (const auto& pair : paired.pairs) {
        require(control.straight(pair.first, pair.second), "incorrect visit pairing");
        require(&paired == &control.correspondence(pair.first, pair.second), "noncanonical query cache");
    }
    require(control.correspondence(2, 2).pairs.size() == 2, "identity omitted shared visit");
    require(!control.correspondence(3, 2).qualified, "receipt before publication admitted");
    // Only the final analytical choice has an empty arm. Qualification must
    // inspect all copies instead of admitting the ordinary copy's evidence.
    auto empty = refined;
    const auto finalChoice = control.wordOccurrences[3].back();
    empty.observed->sites[finalChoice].successors.back() = control.wordOccurrences[6].back();
    o::selected::Control rejected(empty);
    require(rejected.complete, rejected.reason);
    require(rejected.choiceFrontiers.empty(), "empty final arm admitted");
    // An outward publication in a shared consumer word also blocks moving its
    // receipt in front of the choice. Initialize through the canonical ledger.
    o::selected::Ledger fixed(refined, control.canonicalCut, control.wordSpan);
    std::string reason;
    require(fixed.initialize(o::Commands(o::commandCutCount(refined)), reason), reason);
    fixed.append(4, {o::Command::Publish, Q, R, 0}, o::EndpointPurpose::Fixed);
    fixed.append(4, {o::Command::Acquire, Q, R, 0}, o::EndpointPurpose::Fixed);
    const auto blocked = o::constructSelectedPlan(refined, fixed.commands());
    require(blocked.work.choiceTransfers == 0 && blocked.work.choiceTrials == 0,
            "shared outward publication gained prerequisites");
    // Ordinary fallback can decline this repeated shared exchange for lack of
    // rearming. The placement rule must not bypass that independent check.
    if (blocked.success) {
        require(o::verify(refined, blocked.commands).success, "invalid fallback accepted");
    }
}
void unmatchedOccurrences()
{
    auto p = fixture();
    p.observed->sites[0].successors.push_back(3);
    require(!o::selected::Control(p).correspondence(2, 3).qualified, "unmatched receipt admitted");
    p = fixture();
    p.observed->sites[2].successors.push_back(7);
    require(!o::selected::Control(p).correspondence(2, 3).qualified, "unconsumed exit admitted");
    p = fixture();
    p.observed->sites[2].successors.push_back(1);
    p.observed->sites[2].backedgeOwners = {o::NoControlId, 1};
    require(!o::selected::Control(p).correspondence(2, 3).qualified, "double publication admitted");
    p = fixture();
    p.observed->sites[0].successors.push_back(7);
    require(o::selected::Control(p).correspondence(2, 3).qualified, "skipped matched pair declined");
}
void tightCapacity()
{
    auto p = fixture();
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    auto a = construct(p, false), b = construct(p, true);
    const std::vector<o::Cut> path{0, 1, 2, 3, 4, 6, 7};
    require(b.work.choiceTransfers == 0, "tight-capacity specialization was not declined");
    auto old = order(p, a.commands, path, false), now = order(p, b.commands, path, false);
    require(std::includes(old.begin(), old.end(), now.begin(), now.end()), "tight capacity added ordering");
}
void negatives()
{
    auto p = fixture();
    p.operations[1].accesses.push_back({0, false, true, true});
    require(construct(p, true).work.choiceTransfers == 0, "regenerated source history admitted");
    p = fixture();
    p.operations[2].pipe = R;
    require(construct(p, true).work.choiceTransfers == 0, "earlier other-pipeline deadline crossed");
    p = fixture();
    p.operations[3].accesses.clear();
    require(construct(p, true).work.choiceTransfers == 0, "independent first consumer gated");
    p = fixture();
    p.observed->sites[3].successors = {4, 6};
    require(construct(p, true).work.choiceTransfers == 0, "empty choice arm admitted");
    p = fixture();
    o::Commands fixed(o::commandCutCount(p));
    fixed[4] = {{o::Command::Publish, Q, R, 0}, {o::Command::Acquire, Q, R, 0}};
    auto noMotion = construct(p, true, fixed);
    require(noMotion.work.choiceTransfers == 0, "outward publication gained prerequisite");
    p = fixture();
    p.target.keys[unsigned(P)][unsigned(Q)].clear();
    // A relay remains ordinary fallback; the direct choice vocabulary declines.
    require(construct(p, true).work.choiceTransfers == 0, "unsupported direction admitted");
}
} // namespace
int main()
{
    positive();
    sharedChoice();
    unmatchedOccurrences();
    tightCapacity();
    negatives();
    std::cout << "choice consumer frontier tests passed\n";
}
