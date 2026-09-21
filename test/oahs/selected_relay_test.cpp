// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include <numeric>
using namespace selected_test;
namespace {
// Bounded relay selection regressions, not a general ordering guarantee.
// The explicit slot view and target vocabulary are portable fixture contracts;
// this does not qualify a native FIFO lowering or its cross-core protocol.
const auto P = o::Pipe::FIX, Q = o::Pipe::MTE2, R = o::Pipe::M, T = o::Pipe::MTE1;
o::Command set(o::Pipe a, o::Pipe b, unsigned key) { return {o::Command::Publish, a, b, key}; }
o::Command wait(o::Pipe a, o::Pipe b, unsigned key) { return {o::Command::Acquire, a, b, key}; }
o::Program observe(o::Program p, std::vector<std::size_t> reads, std::vector<std::size_t> writes)
{
    p.observed.reset();
    p.staticFifoSlots.reset();
    p.body = {o::Region::Sequence, {}};
    for (unsigned i = 0; i < p.operations.size(); ++i)
        p.body.children.push_back(leaf(i));
    auto observed = o::addStructuredBoundaryCuts(p);
    require(observed.success, observed.reason);
    p = std::move(observed.program);
    p.staticFifoSlots = o::Program::StaticFifoSlots{{0, 1}, std::move(reads), std::move(writes)};
    require(o::validateProgram(p).success, "relay fixture contract invalid");
    return p;
}
o::Cut before(const o::Program& p, unsigned operation)
{
    for (o::Cut at = 0; at < p.observed->sites.size(); ++at)
        if (p.observed->sites[at].operation == operation)
            return at;
    require(false, "missing payload deadline");
    return 0;
}
o::Program fixture(bool requiredReader)
{
    auto p = base(5, 4);
    for (auto& row : p.target.keys)
        for (auto& keys : row)
            keys.clear();
    for (auto [a, b] : std::vector<std::pair<o::Pipe, o::Pipe>>{{P, R}, {P, T}, {R, Q}, {T, Q}})
        p.target.keys[unsigned(a)][unsigned(b)] = {0, 1, 2, 3};
    if (requiredReader)
        p.operations = {
            op(R, {{2, true, false}}),
            op(P, {{0, false, true}}),
            op(P, {{1, false, true}}),
            op(T, {{3, true, false}}),
            op(Q, {{0, true, false}, {2, false, true}}),
            op(Q, {{1, true, false}})};
    else
        p.operations = {
            op(R, {{2, true, false}}), op(P, {{0, false, true}}), op(Q, {{3, true, false}}), op(T, {{4, true, false}}),
            op(Q, {{0, true, false}})};
    return observe(
        std::move(p), requiredReader ? std::vector<std::size_t>{4, 5} : std::vector<std::size_t>{4},
        requiredReader ? std::vector<std::size_t>{1, 2} : std::vector<std::size_t>{1});
}

o::Commands flatten(const o::Program& p, const o::Commands& words)
{
    o::Commands out;
    std::vector<o::Command> pending;
    const auto& g = *p.observed;
    for (auto at = g.entry;;) {
        pending.insert(pending.end(), words[at].begin(), words[at].end());
        if (g.sites[at].operation != o::NoControlId) {
            require(g.sites[at].operation == out.size(), "relay fixture phase order changed");
            out.push_back(std::move(pending));
            pending.clear();
        }
        if (at == g.exit) {
            out.push_back(std::move(pending));
            break;
        }
        require(g.sites[at].successors.size() == 1, "relay fixture is not straight");
        at = g.sites[at].successors.front();
    }
    return out;
}
oahs_oracle::Verdict evaluate(const o::Program& p, const o::Commands& words, oahs_oracle::PayloadOrder* out = nullptr)
{
    auto flat = p;
    flat.observed.reset();
    flat.staticFifoSlots.reset();
    flat.body = {};
    std::vector<unsigned> visits(flat.operations.size());
    std::iota(visits.begin(), visits.end(), 0);
    return oahs_oracle::graph(flat, words, visits, {}, nullptr, nullptr, out);
}
oahs_oracle::PayloadOrder order(const o::Program& p, const o::Commands& words)
{
    oahs_oracle::PayloadOrder out;
    require(bool(evaluate(p, words, &out)), "relay oracle rejected memory/matching/rearming");
    return out;
}
void missingSupport(const o::Program& p, o::Commands words, o::Pipe source, o::Pipe observer, unsigned key)
{
    unsigned erased = 0;
    for (auto& word : words)
        word.erase(
            std::remove_if(
                word.begin(), word.end(),
                [&](const auto& command) {
                    const bool remove = (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
                                        command.source == source && command.observer == observer && command.key == key;
                    erased += remove;
                    return remove;
                }),
            word.end());
    const auto verdict = evaluate(p, words);
    require(
        erased == 2 && verdict.balanced && verdict.acyclic && verdict.rearm && !verdict.hazards,
        "removing a complete relay leg did not isolate missing memory support");
}
unsigned pairs(const o::Commands& words)
{
    unsigned out = 0;
    for (const auto& word : words)
        for (const auto& c : word)
            out += c.kind == o::Command::Publish;
    return out;
}
void compare(const char* name, const oahs_oracle::PayloadOrder& a, const oahs_oracle::PayloadOrder& b)
{
    unsigned removed = 0, added = 0;
    for (auto r : a)
        removed += !b.count(r);
    for (auto r : b)
        added += !a.count(r);
    std::cout << name << " before=" << a.size() << " after=" << b.size() << " removed=" << removed << " added=" << added
              << '\n';
}
void witness(bool requiredReader)
{
    const auto p = fixture(requiredReader);
    o::SelectedOptions options;
    options.recurringOmissionTrials = options.finalHelperTrials = false;
    const auto plan = o::constructSelectedPlan(p, {}, options);
    require(plan.success, "linked relay construction failed: " + plan.reason);
    require(o::checkCausalFrontier(p, plan.commands).accepted, "cold relay check failed");
    const auto actual = flatten(p, plan.commands);
    const auto actualOrder = order(p, actual);
    std::cout << "linked required_reader=" << requiredReader << " split_relays=" << plan.work.splitRelays
              << " pairs=" << pairs(actual) << " order=" << actualOrder.size() << '\n';
    for (unsigned cut = 0; cut < actual.size(); ++cut)
        for (const auto& c : actual[cut])
            std::cout << "  cut=" << cut << " kind=" << unsigned(c.kind) << " source=" << unsigned(c.source)
                      << " observer=" << unsigned(c.observer) << " key=" << c.key << '\n';
    if (!requiredReader) {
        o::Commands late(6), early(6);
        late[2] = {set(P, R, 0)};
        late[4] = {wait(P, R, 0), set(R, Q, 0), wait(R, Q, 0)};
        early[2] = {set(P, T, 0)};
        early[3] = {wait(P, T, 0), set(T, Q, 0)};
        early[4] = {wait(T, Q, 0)};
        const auto a = order(p, late), b = order(p, early);
        require(
            plan.work.splitRelays == 1 && pairs(actual) == 2 && actualOrder == a,
            "incomparable relay tradeoff did not retain the deterministic late route");
        require(a.size() == 15 && b.size() == 15 && a != b, "relay incomparable model witness changed");
        require(
            a.count({1, 8}) && !b.count({1, 8}) && b.count({3, 6}) && !a.count({3, 6}),
            "relay did not exchange final-consumer and middle-engine prerequisites");
        compare("model_incomparable", a, b);
        compare("linked_vs_late", actualOrder, a);
        missingSupport(p, actual, P, R, 0);
        missingSupport(p, actual, R, Q, 0);
        missingSupport(p, early, P, T, 0);
        missingSupport(p, early, T, Q, 0);
    } else {
        o::Commands early(7), late(7);
        for (auto* words : {&early, &late}) {
            (*words)[1] = {set(R, Q, 0)};
            (*words)[4] = {wait(R, Q, 0)};
            (*words)[3].push_back(set(P, R, 1));
            (*words)[5] = {wait(P, R, 1), set(R, Q, 2), wait(R, Q, 2)};
        }
        early[2] = {set(P, T, 0)};
        early[3].push_back(wait(P, T, 0));
        early[3].push_back(set(T, Q, 0));
        early[4].push_back(wait(T, Q, 0));
        late[2] = {set(P, R, 0)};
        late[4].push_back(wait(P, R, 0));
        late[4].push_back(set(R, Q, 1));
        late[4].push_back(wait(R, Q, 1));
        const auto a = order(p, early), b = order(p, late);
        require(
            plan.work.splitRelays == 2 && pairs(actual) == 5 && actualOrder == b,
            "relay discarded useful receiver completion or gated unrelated middle work");
        auto shared = late;
        shared[1].clear();
        shared[4].erase(shared[4].begin());
        require(
            order(p, shared) == b && pairs(early) == 5 && pairs(late) == 5 && pairs(shared) == 4,
            "required receiver completion did not compose with relay");
        require(
            a.size() == 34 && b.size() == 30 && std::includes(a.begin(), a.end(), b.begin(), b.end()),
            "required-completion model witness changed");
        require(
            a.count({3, 6}) && !b.count({3, 6}) && b.count({1, 8}),
            "better relay lost required reader completion or retained the unrelated middle gate");
        compare("model_required", a, b);
        compare("linked_vs_required_route", actualOrder, b);
        // Advancing the later slot's receipt into the first receive word is
        // safe but broadens the already selected earlier forwarding publication.
        auto widened = late;
        widened[4].insert(widened[4].begin(), widened[5].begin(), widened[5].begin() + 2);
        widened[5].erase(widened[5].begin(), widened[5].begin() + 2);
        const auto widenedOrder = order(p, widened);
        require(
            widenedOrder.size() == 34 && std::includes(widenedOrder.begin(), widenedOrder.end(), b.begin(), b.end()),
            "late receipt mutation failed to expose earlier-receipt widening");
        compare("receipt_widening", b, widenedOrder);
        missingSupport(p, early, P, T, 0);
        missingSupport(p, early, T, Q, 0);
        missingSupport(p, early, R, Q, 0);
        missingSupport(p, early, P, R, 1);
        missingSupport(p, early, R, Q, 2);
        missingSupport(p, actual, P, R, 0);
        missingSupport(p, actual, R, Q, 1);
        missingSupport(p, actual, P, R, 1);
        missingSupport(p, actual, R, Q, 2);
        missingSupport(p, shared, P, R, 0);
        missingSupport(p, shared, R, Q, 1);
        missingSupport(p, shared, P, R, 1);
        missingSupport(p, shared, R, Q, 2);
    }
}
// Equal late gaps: the second enumerated middle carries useful history,
// while the first carries unrelated work. Exercise pending and actual credit.
void receiverCredit(bool acquired)
{
    auto input = fixture(false);
    input.operations = {
        op(R, {{4, true, false}}), op(T, {{2, false, true}}), op(P, {{0, false, true}}), op(Q, {{3, true, false}}),
        op(Q, {{0, true, false}, {2, true, false}})};
    auto p = observe(std::move(input), {4}, {2});
    o::Commands fixed(p.observed->sites.size());
    if (acquired) {
        fixed[before(p, 2)] = {set(T, Q, 3)};
        fixed[before(p, 4)] = {wait(T, Q, 3)};
    }
    o::SelectedOptions options;
    options.recurringOmissionTrials = options.finalHelperTrials = false;
    const auto selected = o::constructSelectedPlan(p, fixed, options);
    require(
        selected.success && o::checkCausalFrontier(p, selected.commands).accepted,
        "receiver credit construction failed");
    auto actual = flatten(p, selected.commands);
    o::Commands good = flatten(p, fixed), bad = good;
    good[3].insert(good[3].begin(), set(P, T, 0));
    good[4].insert(good[4].end(), {wait(P, T, 0), set(T, Q, 0), wait(T, Q, 0)});
    bad[3].insert(bad[3].begin(), set(P, R, 0));
    bad[4].insert(bad[4].end(), {wait(P, R, 0), set(R, Q, 0), wait(R, Q, 0)});
    if (!acquired) {
        bad[2] = {set(T, Q, 3)};
        bad[4].push_back(wait(T, Q, 3));
    }
    const auto a = order(p, actual), b = order(p, bad), g = order(p, good);
    std::cout << "receiver_credit acquired=" << acquired << " pairs=" << pairs(actual) << '\n';
    require(
        selected.decisions.size() == 1 && selected.decisions.front().source == P &&
            pairs(actual) == (acquired ? 3u : 2u),
        "pending receiver credit was supplied by an earlier construction decision");
    missingSupport(p, actual, P, T, 0);
    missingSupport(p, actual, T, Q, 0);
    compare("receiver_credit", b, a);
    require(
        selected.work.splitRelays == 1 && a == g && a != b && std::includes(b.begin(), b.end(), a.begin(), a.end()),
        "destination-required/acquired completion was penalized as incidental history");
}

// Existing credit can make early forwarding harmless to a middle payload.
// An earlier outward SET still needs its own exact-word check.
void outwardBoundary(unsigned placement)
{
    auto input = fixture(false);
    const auto other = o::Pipe::MTE3;
    input.target.keys[unsigned(T)][unsigned(other)] = {0};
    input.operations.push_back(op(other, {{3, true, false}}));
    auto p = observe(std::move(input), {4}, {1});
    o::Commands fixed(p.observed->sites.size());
    fixed[before(p, 2)] = {set(P, T, 3)};
    if (placement == 1)
        fixed[before(p, 3)].push_back(set(T, other, 0));
    fixed[before(p, 3)].push_back(wait(P, T, 3));
    if (placement == 2)
        fixed[before(p, 3)].push_back(set(T, other, 0));
    if (placement)
        fixed[before(p, 5)] = {wait(T, other, 0)};
    o::SelectedOptions options;
    options.recurringOmissionTrials = options.finalHelperTrials = false;
    const auto selected = o::constructSelectedPlan(p, fixed, options);
    require(
        selected.success && o::checkCausalFrontier(p, selected.commands).accepted,
        "outward boundary construction failed");
    const auto actual = flatten(p, selected.commands);
    auto late = flatten(p, fixed), early = late;
    late[2].push_back(set(P, R, 0));
    late[4] = {wait(P, R, 0), set(R, Q, 0), wait(R, Q, 0)};
    early[2].push_back(set(P, T, 0));
    early[3].insert(early[3].begin(), {wait(P, T, 0), set(T, Q, 0)});
    early[4] = {wait(T, Q, 0)};
    const auto a = order(p, actual), l = order(p, late), e = order(p, early);
    require(selected.work.splitRelays == 1 && a == (placement == 1 ? l : e), "middle gate or outward boundary ignored");
    require(placement != 1 || (!l.count({3, 10}) && e.count({3, 10})), "outward model lost its independent receiver");
    require(
        placement == 1 || (l != e && std::includes(l.begin(), l.end(), e.begin(), e.end())),
        "preexisting middle credit did not preserve useful early forwarding");
    std::cout << "outward_boundary placement=" << placement << '\n';
    compare("late_vs_selected", l, a);
}
} // namespace
int main()
{
    witness(false);
    witness(true);
    receiverCredit(false);
    receiverCredit(true);
    outwardBoundary(0);
    outwardBoundary(1);
    outwardBoundary(2);
}
