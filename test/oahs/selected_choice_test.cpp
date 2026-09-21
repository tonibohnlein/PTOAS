// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
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
    tightCapacity();
    negatives();
    std::cout << "choice consumer frontier tests passed\n";
}
