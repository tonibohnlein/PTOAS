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
using namespace selected_test;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void dormantRelay(Program p, Pipe a, Pipe b) {
        p.target.keys[unsigned(b)][unsigned(a)] = {0};
        SelectedOptions options; options.finalHelperTrials = options.recurringOmissionTrials = false;
        Constructor c(p,options); c.needsContextualReplay = true;
        const auto entry = c.control.graph.entry;
        auto append = [&](Pipe from, Pipe to, unsigned key, Command::Kind kind, EndpointPurpose purpose, Id ack = NoAnalysisId) {
            return c.ledger.append(entry,{kind,from,to,key},purpose,NoAnalysisId,ack);
        };
        append(b,a,0,Command::Publish,EndpointPurpose::Fixed);
        const auto forward = append(b,a,0,Command::Acquire,EndpointPurpose::Fixed);
        const auto helperSet = append(a,b,0,Command::Publish,EndpointPurpose::ConsumptionAcknowledgment,forward);
        const auto helperWait = append(a,b,0,Command::Acquire,EndpointPurpose::ConsumptionAcknowledgment,forward);
        c.rememberReturn(helperSet,helperWait); c.result.work.acknowledgments = 1;
        append(a,b,3,Command::Publish,EndpointPurpose::Completion);
        const auto actual = append(a,b,3,Command::Acquire,EndpointPurpose::Completion);
        c.current = c.control.graph.exit; c.activeComponent = c.control.component[c.current];
        require(c.update(),"dormant relay initial protocol: " + c.cache.reason);
        SelectedDecision receipt; receipt.endpoints = {actual};
        require(c.settleRearming(receipt) && !c.ledger.active(helperWait),"relay helper never discharged");
        const auto plan = c.run({});
        require(plan.success && plan.work.splitRelays != 0,"dormant relay lost its selectable route: " + plan.reason);
        bool alternative = false;
        for (const auto& e : c.ledger.records())
            if (c.ledger.active(e.id) && e.command.kind == Command::Publish &&
                e.command.source == a && e.command.observer == b && e.cut != entry) {
                require(e.command.key != 0,"relay stole dormant helper ownership");
                alternative = true;
            }
        require(alternative,"dormant relay did not exercise the affected leg");
        Id originalForward = NoAnalysisId;
        for (Id k = 0; k < c.frontier.keys().size(); ++k)
            if (c.frontier.keys()[k].source == b && c.frontier.keys()[k].observer == a &&
                c.frontier.keys()[k].key == 0) originalForward = k;
        require(c.restoreReturns(originalForward),"relay helper restoration was not exercised");
        require(checkCausalFrontier(p,c.ledger.commands()).accepted,"relay binding broke restored ownership");
    }
};
}
namespace {
// Bounded relay selection regressions, not a general ordering guarantee.
// Target vocabulary is a portable fixture contract. The same imported storage
// obligations must work without FIFO provenance.
const auto P = o::Pipe::FIX, Q = o::Pipe::MTE2, R = o::Pipe::M, T = o::Pipe::MTE1;
o::Command set(o::Pipe a, o::Pipe b, unsigned key) { return {o::Command::Publish, a, b, key}; }
o::Command wait(o::Pipe a, o::Pipe b, unsigned key) { return {o::Command::Acquire, a, b, key}; }
o::Program observe(o::Program p)
{
    p.observed.reset();
    p.staticFifoSlots.reset();
    p.body = {o::Region::Sequence, {}};
    for (unsigned i = 0; i < p.operations.size(); ++i)
        p.body.children.push_back(leaf(i));
    auto observed = o::addStructuredBoundaryCuts(p);
    require(observed.success, observed.reason);
    p = std::move(observed.program);
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
    return observe(std::move(p));
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
    auto tagged = p;
    tagged.staticFifoSlots = o::Program::StaticFifoSlots{
        {0, 1}, requiredReader ? std::vector<std::size_t>{4, 5} : std::vector<std::size_t>{4},
        requiredReader ? std::vector<std::size_t>{1, 2} : std::vector<std::size_t>{1}};
    const auto taggedPlan = o::constructSelectedPlan(tagged, {}, options);
    require(taggedPlan.success && taggedPlan.work.splitRelays == plan.work.splitRelays &&
                order(tagged, flatten(tagged, taggedPlan.commands)) == actualOrder,
            "FIFO provenance changed the ordinary storage relay");
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
    auto p = observe(std::move(input));
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
    auto p = observe(std::move(input));
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
// The default ranking prefers the late R route. A physically occupied key
// must not hide the independently bindable T route, on either relay leg.
void bindingAlternative(bool secondLeg, unsigned state = 0, bool reusedAlternative = false)
{
    auto p = fixture(false);
    for (auto& row : p.target.keys)
        for (auto& keys : row)
            if (!keys.empty()) keys = {0};
    const auto a = secondLeg ? R : P, b = secondLeg ? Q : R;
    o::Commands fixed(p.observed->sites.size());
    // Full at the source, empty but without return credit, or a future use
    // within the first leg's interval. All are eligible directional keys.
    fixed[before(p, state == 2 ? 3 : 1)] = {set(a, b, 0)};
    fixed[state == 1 ? before(p, 2) : p.observed->exit] = {wait(a, b, 0)};
    if (reusedAlternative) {
        p.target.keys[unsigned(Q)][unsigned(T)] = {0};
        fixed[before(p, 0)] = {set(T, Q, 0)};
        auto& word = fixed[before(p, 1)];
        word.insert(word.end(), {wait(T, Q, 0), set(Q, T, 0), wait(Q, T, 0)});
    }
    o::SelectedOptions options;
    options.recurringOmissionTrials = options.finalHelperTrials = false;
    auto control = o::constructSelectedPlan(p, {}, options);
    require(control.success && control.work.splitRelays == 1, "unblocked relay control failed");
    const auto unblocked = flatten(p, control.commands);
    require(std::any_of(unblocked[2].begin(), unblocked[2].end(), [](const auto& c) {
                return c.kind == o::Command::Publish && c.source == P && c.observer == R;
            }), "fixture no longer prefers the route whose key is blocked");
    auto selected = o::constructSelectedPlan(p, fixed, options);
    std::cout << "binding_alternative second_leg=" << secondLeg << " state=" << state
              << " reused=" << reusedAlternative << " success=" << selected.success
              << " split=" << selected.work.splitRelays << " reason=" << selected.reason << std::endl;
    require(selected.success, "occupied preferred route hid a bindable alternative");
    require(o::checkCausalFrontier(p, selected.commands).accepted, "alternative binding cold check failed");
    auto expected = flatten(p, fixed);
    expected[2].push_back(set(P, T, 0));
    expected[3].insert(expected[3].begin(), {wait(P, T, 0), set(T, Q, 0)});
    expected[4].push_back(wait(T, Q, 0));
    const auto actual = flatten(p, selected.commands);
    require(selected.work.splitRelays == 1 && selected.work.relayTrials == 1 &&
                pairs(actual) == (reusedAlternative ? 5u : 3u) && order(p, actual) == order(p, expected),
            "binding alternative did not select the certified two-leg route");
    missingSupport(p, actual, P, T, 0);
    if (!reusedAlternative) missingSupport(p, actual, T, Q, 0);
    else {
        auto noReturn = actual;
        for (auto& word : noReturn)
            word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
                return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) &&
                       c.source == Q && c.observer == T;
            }), word.end());
        const auto verdict = evaluate(p, noReturn);
        require(verdict.balanced && verdict.acyclic && !verdict.rearm && verdict.hazards,
                "reused alternative did not require actual consumption return credit");
    }
    // The alternative must be genuinely selectable; failure of all candidates
    // does not authorize a speculative receipt or extra staged searches.
    auto unavailable = p;
    unavailable.target.keys[unsigned(P)][unsigned(T)].clear();
    const auto rejected = o::constructSelectedPlan(unavailable, fixed, options);
    require(!rejected.success && rejected.work.relayTrials == 0 && rejected.work.splitRelays == 0,
            "unqualified binding was staged despite no certified route");
}
} // namespace
int main()
{
    o::selected::ReplayTestAccess::dormantRelay(fixture(true),P,R);
    o::selected::ReplayTestAccess::dormantRelay(fixture(true),R,Q);
    witness(false);
    witness(true);
    receiverCredit(false);
    receiverCredit(true);
    outwardBoundary(0);
    outwardBoundary(1);
    outwardBoundary(2);
    bindingAlternative(false);
    bindingAlternative(true);
    bindingAlternative(false, 1);
    bindingAlternative(true, 1);
    bindingAlternative(false, 2);
    bindingAlternative(false, 0, true);
}
