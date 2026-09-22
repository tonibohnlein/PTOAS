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
constexpr auto P = o::Pipe::MTE2;
constexpr auto Q = o::Pipe::MTE1;
constexpr auto M = o::Pipe::M;
constexpr auto R = o::Pipe::V;

void releaseBeforeUnrelatedWait(bool requiredReceipt, bool outward)
{
    auto p = base(2, 3);
    p.operations = {op(M, {{1, false, true}}), op(P, {{0, false, true}}),
                    op(Q, {{0, true, false}}), op(P, {{0, false, true}}), op(R, {})};
    if (requiredReceipt) {
        p.operations[3].accesses.push_back({1, true, false});
    }
    o::Commands fixed(o::commandCutCount(p));
    fixed[1] = {{o::Command::Publish, M, Q, 0}};
    fixed[2] = {{o::Command::Publish, P, Q, 0}, {o::Command::Acquire, P, Q, 0}};
    fixed[3] = {{o::Command::Acquire, M, Q, 0}};
    if (outward) {
        fixed[3].push_back({o::Command::Publish, Q, R, 0});
        fixed[4].push_back({o::Command::Acquire, Q, R, 0});
    }
    o::SelectedOptions options;
    options.publicationPrefixes = false;
    const auto old = o::constructSelectedPlan(p, fixed, options);
    options.publicationPrefixes = true;
    const auto plan = o::constructSelectedPlan(p, fixed, options);
    require(old.success && plan.success, "release construction: " + old.reason + " / " + plan.reason);
    require(o::checkCausalFrontier(p, plan.commands).accepted, "early publication cold certificate");
    oahs_oracle::PayloadOrder before, after;
    const std::vector<unsigned> visits{0, 1, 2, 3, 4};
    require(bool(oahs_oracle::graph(p, old.commands, visits, {}, nullptr, nullptr, &before)), "baseline graph");
    require(bool(oahs_oracle::graph(p, plan.commands, visits, {}, nullptr, nullptr, &after)), "candidate graph");
    require(std::includes(before.begin(), before.end(), after.begin(), after.end()), "publication added ordering");
    if (!requiredReceipt && !outward) {
        require(plan.work.prefixPublications == 1 && before.count({1, 6}) && !after.count({1, 6}),
                "constructor retained unrelated M completion before refill");
    } else if (outward) {
        require(plan.work.prefixPublications == 0 && before == after, "crossed an outward source publication");
    } else {
        require(after.count({1, 6}), "required M completion was lost");
    }
}

void reusedRelease()
{
    auto p = base(4, 1);
    p.operations = {op(Q, {{3, false, true}}), op(P, {{3, true, false}}),
                    op(M, {{1, true, false}}), op(P, {{0, false, true}}),
                    op(Q, {{0, true, false}, {2, false, true}}),
                    op(Q, {{1, false, true}}), op(P, {{0, false, true}})};
    o::SelectedOptions options;
    options.publicationPrefixes = false;
    options.finalHelperTrials = false;
    const auto old = o::constructSelectedPlan(p, {}, options);
    options.publicationPrefixes = true;
    const auto plan = o::constructSelectedPlan(p, {}, options);
    require(old.success && plan.success, "reused release construction: " + plan.reason);
    oahs_oracle::PayloadOrder before, after;
    const std::vector<unsigned> visits{0, 1, 2, 3, 4, 5, 6};
    require(bool(oahs_oracle::graph(p, old.commands, visits, {}, nullptr, nullptr, &before)), "reused baseline");
    require(bool(oahs_oracle::graph(p, plan.commands, visits, {}, nullptr, nullptr, &after)), "reused candidate");
    require(plan.work.prefixPublications && before != after &&
            std::includes(before.begin(), before.end(), after.begin(), after.end()), "reused prefix added ordering");
    require(before.count({5, 12}) && !after.count({5, 12}), "reused release retained unrelated compute");
}

oahs_oracle::PayloadOrder branchOrder(const o::Program& program, const o::Commands& commands, unsigned arm)
{
    auto flat = program;
    flat.observed.reset();
    flat.operations.clear();
    o::Commands words;
    std::vector<o::Command> pending;
    const auto& graph = *program.observed;
    auto at = graph.entry;
    unsigned branch = 0;
    for (unsigned steps = 0; steps < graph.sites.size(); ++steps) {
        pending.insert(pending.end(), commands[at].begin(), commands[at].end());
        const auto& site = graph.sites[at];
        if (site.operation != o::NoAnalysisId) {
            flat.operations.push_back(program.operations[site.operation]);
            words.push_back(std::move(pending));
            pending.clear();
        }
        if (at == graph.exit) {
            break;
        }
        at = site.successors[site.successors.size() == 2 ? ((arm >> branch++) & 1) : 0];
    }
    require(at == graph.exit, "branch trace did not terminate");
    words.push_back(std::move(pending));
    std::vector<unsigned> visits(flat.operations.size());
    std::iota(visits.begin(), visits.end(), 0);
    oahs_oracle::PayloadOrder order;
    require(bool(oahs_oracle::graph(flat, words, visits, {}, nullptr, nullptr, &order)), "branch order invalid");
    return order;
}

void crossWordRelease(bool readAgain, bool finalHelpers, bool shared = false)
{
    auto p = base(4, 1);
    p.operations = {op(Q, {{3, false, true}}), op(P, {{3, true, false}}),
                    op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(M, {{1, false, true}}), op(R, {}), op(R, {}), op(P, {{0, false, true}})};
    if (readAgain) {
        p.operations[5] = op(Q, {{0, true, false}});
        p.operations[6] = op(Q, {{0, true, false}});
    }
    o::ObservedControl graph;
    graph.qualification = "publication corridor with retained outward event";
    graph.entry = 0;
    graph.exit = 9;
    graph.sites.resize(10);
    for (unsigned i = 0; i < 10; ++i) {
        graph.observations.push_back({i, {}, true});
        graph.sites[i].observation = i;
        if (i < 9) {
            graph.sites[i].successors = {i + 1};
        }
    }
    for (unsigned i = 0; i < 5; ++i) {
        graph.sites[i].operation = i;
    }
    graph.sites[5].successors = {6, 7};
    graph.sites[6].operation = 5;
    graph.sites[6].successors = {8};
    graph.sites[7].operation = 6;
    graph.sites[8].operation = 7;
    p.observed = graph;
    o::Commands fixed(o::commandCutCount(p));
    fixed[5] = {{o::Command::Publish, M, Q, 0}, {o::Command::Acquire, M, Q, 0},
                {o::Command::Publish, Q, R, 0}, {o::Command::Acquire, Q, R, 0}};
    if (shared) {
        auto& g = *p.observed;
        for (o::Cut at = 0; at < 9; ++at) {
            auto copy = g.sites[at];
            for (auto& next : copy.successors) {
                if (next != 9) {
                    next += 10;
                }
            }
            g.sites.push_back(std::move(copy));
            fixed.push_back(fixed[at]);
        }
        g.sites.emplace_back();
        g.sites.back().successors = {0, 10};
        g.entry = 19;
        g.observations.push_back({19, {}, true});
        g.sites.back().observation = g.observations.size() - 1;
        fixed.emplace_back();
    }
    o::SelectedOptions options;
    options.finalHelperTrials = finalHelpers;
    const auto old = o::constructSelectedPlan(p, fixed, options);
    options.crossWordPrefixes = true;
    const auto plan = o::constructSelectedPlan(p, fixed, options);
    require(old.success && plan.success, "cross-word constructor: " + old.reason + " / " + plan.reason);
    require(o::checkCausalFrontier(p, plan.commands).accepted, "cross-word cold certificate");
    require(readAgain ? plan.work.crossWordPublications == 0 : plan.work.crossWordPublications != 0,
            "cross-word release ignored the last physical reader");
    for (unsigned arm = 0; arm < (shared ? 4u : 2u); ++arm) {
        const auto before = branchOrder(p, old.commands, arm);
        const auto after = branchOrder(p, plan.commands, arm);
        require(std::includes(before.begin(), before.end(), after.begin(), after.end()),
                "cross-word certificate added ordering");
        if (readAgain) {
            require(before == after && after.count({9, 12}), "later physical read lost its release boundary");
        } else {
            require(before != after && before.count({9, 12}) && !after.count({9, 12}),
                    "compute still gates refill");
            require(after.count({9, 10}), "crossed outward publication lost its original completion");
        }
    }
}
} // namespace

namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void spanBoundaries()
    {
        auto p = base(1, 2);
        p.operations = {op(Q, {}), op(M, {}), op(P, {})};
        Control control(p);
        CausalFrontier frontier(p);
        Ledger ledger(p, control.canonicalCut, control.wordSpan);
        std::string reason;
        require(ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        const auto outward = ledger.append(1, {Command::Publish, Q, R, 0}, EndpointPurpose::Fixed);
        const auto publication = ledger.append(2, {Command::Publish, Q, P, 0}, EndpointPurpose::Completion);
        const auto continuation = ledger.append(2, {Command::Acquire, Q, P, 0}, EndpointPurpose::Completion);
        std::vector<Cut> crossed;
        require(certifyPublicationOrder(p, control, ledger, frontier.keys(), publication, 1, 0, crossed),
                "open-fragment certificate refused retained outward publication");
        auto moved = ledger;
        require(moved.movePublicationTo(publication, 1, 0, crossed) && moved.publicationPrefixesValid(),
                "fresh cross-word certificate invalid");
        require(ledger.endpoint(publication).cut == 2, "motion trial changed the live ledger");
        for (const auto command : std::vector<Command>{{Command::Acquire, M, Q, 0},
                 {Command::Publish, Q, R, 1}, {Command::Barrier, Q}, {Command::BarrierAll}}) {
            auto changed = moved;
            changed.append(1, command, EndpointPurpose::Completion);
            require(!changed.publicationPrefixesValid(), "changed span retained a stale certificate");
        }
        auto erased = moved;
        erased.erase(outward);
        require(!erased.publicationPrefixesValid(), "endpoint deletion retained a stale span certificate");
        auto afterGap = moved;
        afterGap.append(2, {Command::Acquire, M, Q, 0}, EndpointPurpose::Completion);
        require(afterGap.publicationPrefixesValid(), "same-word continuation invalidated the earlier fragment");
        auto beforeGap = moved;
        beforeGap.prepend(2, {Command::Acquire, M, Q, 0}, EndpointPurpose::Completion);
        require(!beforeGap.publicationPrefixesValid(), "changed original gap retained a stale certificate");
        auto lostAnchor = moved;
        lostAnchor.erase(continuation);
        require(!lostAnchor.publicationPrefixesValid(), "deleted continuation anchor retained its gap");
        auto outside = moved;
        outside.append(3, {Command::BarrierAll}, EndpointPurpose::Fixed);
        require(outside.publicationPrefixesValid(), "unchanged fragment was invalidated by its continuation");
        const auto version = moved.version();
        require(!moved.movePublicationTo(publication, 1, 0, crossed) && moved.version() == version,
                "invalid destination changed the ledger");
        require(!certifyPublicationOrder(p, control, ledger, frontier.keys(), publication, 1, 9, crossed) &&
                crossed.empty(), "invalid gap retained a certificate");
        for (const auto kind : {Command::Publish, Command::Acquire}) {
            auto reuse = ledger;
            reuse.append(1, {kind, Q, P, 0}, EndpointPurpose::Fixed);
            require(!certifyPublicationOrder(p, control, reuse, frontier.keys(), publication, 1, 0, crossed),
                    "certificate crossed a same-key generation boundary");
        }
    }
    static void sharedSpan()
    {
        auto p = base(1, 2);
        p.operations = {op(Q, {}), op(M, {}), op(P, {})};
        ObservedControl g;
        g.qualification = "alternative analytical words with distinct crossed interfaces";
        g.entry = 0;
        g.exit = 7;
        g.sites.resize(8);
        for (Cut at = 0; at < 8; ++at) {
            g.observations.push_back({at, {}, true});
            g.sites[at].observation = at;
            if (at < 7) {
                g.sites[at].successors = {at + 1};
            }
        }
        g.sites[0].successors = {1, 4};
        g.sites[3].successors = {7};
        for (Cut at = 1; at <= 3; ++at) {
            g.sites[at].operation = at - 1;
            g.sites[at + 3].operation = at - 1;
        }
        g.sites[4].observation = 1;
        g.sites[6].observation = 3;
        p.observed = g;
        Control control(p);
        CausalFrontier frontier(p);
        Ledger ledger(p, control.canonicalCut, control.wordSpan);
        std::string reason;
        require(control.complete && ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        ledger.append(2, {Command::Acquire, M, Q, 0}, EndpointPurpose::Fixed);
        ledger.append(5, {Command::Acquire, R, Q, 0}, EndpointPurpose::Fixed);
        const auto set = ledger.append(3, {Command::Publish, Q, P, 0}, EndpointPurpose::Completion);
        ledger.append(3, {Command::Acquire, Q, P, 0}, EndpointPurpose::Completion);
        std::vector<Cut> crossed;
        require(certifyPublicationOrder(p, control, ledger, frontier.keys(), set, 4, 0, crossed),
                "paired shared endpoints were not certified");
        require(std::find(crossed.begin(), crossed.end(), 2) != crossed.end() &&
                std::find(crossed.begin(), crossed.end(), 5) != crossed.end(),
                "certificate omitted one occurrence's outward interface");
        auto moved = ledger;
        require(moved.movePublicationTo(set, 4, 0, crossed) && moved.publicationPrefixesValid(),
                "shared endpoint move lost its certificate");
        moved.append(5, {Command::Publish, Q, R, 1}, EndpointPurpose::Fixed);
        require(!moved.publicationPrefixesValid(), "edit in second occurrence retained stale proof");
        auto reused = ledger;
        reused.append(5, {Command::Acquire, Q, P, 0}, EndpointPurpose::Fixed);
        require(!certifyPublicationOrder(p, control, reused, frontier.keys(), set, 1, 0, crossed) && crossed.empty(),
                "same-key endpoint in one occurrence crossed");
        auto unmatched = p;
        unmatched.observed->sites[0].successors.back() = 5;
        Control missing(unmatched);
        Ledger words(unmatched, missing.canonicalCut, missing.wordSpan);
        require(missing.complete && words.initialize(ledger.commands(), reason), reason);
        const auto publication = words.word(3).front();
        require(!certifyPublicationOrder(unmatched, missing, words, frontier.keys(), publication, 1, 0, crossed),
                "unmatched shared publication occurrence admitted");
    }
    static void cyclicSpan()
    {
        auto p = base(1, 1);
        p.operations = {op(Q, {}), op(M, {}), op(P, {})};
        p.body = seq({leaf(0), {Region::For, {leaf(1)}, 0, true}, leaf(2)});
        Control control(p);
        CausalFrontier frontier(p);
        Ledger ledger(p, control.canonicalCut, control.wordSpan);
        std::string reason;
        require(control.complete && ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        const auto& operations = control.graph.operations;
        const auto source = Id(std::find(operations.begin(), operations.end(), 0) - operations.begin());
        const auto consumer = Id(std::find(operations.begin(), operations.end(), 2) - operations.begin());
        const auto publication = ledger.append(consumer, {Command::Publish, Q, P, 0}, EndpointPurpose::Completion);
        std::vector<Cut> crossed;
        require(!certifyPublicationOrder(p, control, ledger, frontier.keys(), publication,
                    control.after(source), 0, crossed) && crossed.empty(),
                "bounded certificate admitted an unqualified loop corridor");
    }
    static void certificateBoundaries()
    {
        auto p = base(1, 2);
        p.operations = {op(Q, {})};
        Control control(p);
        Ledger ledger(p, control.canonicalCut, control.wordSpan);
        std::string reason;
        require(ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        const auto wait = ledger.append(0, {Command::Acquire, M, Q, 0}, EndpointPurpose::Fixed);
        const auto publication = ledger.append(0, {Command::Publish, Q, P, 0}, EndpointPurpose::Completion);
        Ledger privateCopy = ledger;
        require(privateCopy.movePublicationBefore(publication, wait), "structural prefix certificate refused");
        require(ledger.word(0).front() == wait, "trial changed live ledger");
        require(privateCopy.publicationPrefixesValid(), "fresh prefix certificate invalid");
        const auto inserted = privateCopy.prepend(0, {Command::Acquire, R, Q, 1}, EndpointPurpose::Completion);
        require(privateCopy.word(0).front() == publication && privateCopy.word(0)[1] == inserted,
                "later receipt widened protected prefix");
        privateCopy.erase(wait);
        privateCopy.restoreAfter(wait, publication);
        require(privateCopy.publicationPrefixesValid(), "restoration lost protected ordering");

        for (const auto crossed : std::vector<Command>{{Command::Publish, Q, R, 0},
                 {Command::Barrier, Q}, {Command::Acquire, Q, P, 0}, {Command::BarrierAll}}) {
            Ledger negative(p, control.canonicalCut, control.wordSpan);
            require(negative.initialize(Commands(commandCutCount(p)), reason), reason);
            const auto anchor = negative.append(0, crossed, EndpointPurpose::Fixed);
            const auto set = negative.append(0, {Command::Publish, Q, P, 0}, EndpointPurpose::Completion);
            const auto version = negative.version();
            require(!negative.movePublicationBefore(set, anchor) && negative.version() == version,
                    "invalid motion changed the ledger");
        }
    }
};
} // namespace mlir::pto::oahs::selected

int main()
{
    releaseBeforeUnrelatedWait(false, false);
    releaseBeforeUnrelatedWait(true, false);
    releaseBeforeUnrelatedWait(false, true);
    reusedRelease();
    for (const bool finalHelpers : {false, true}) {
        crossWordRelease(false, finalHelpers);
        crossWordRelease(true, finalHelpers);
        crossWordRelease(false, finalHelpers, true);
        crossWordRelease(true, finalHelpers, true);
    }
    o::selected::ReplayTestAccess::spanBoundaries();
    o::selected::ReplayTestAccess::sharedSpan();
    o::selected::ReplayTestAccess::cyclicSpan();
    o::selected::ReplayTestAccess::certificateBoundaries();
    std::cout << "publication prefix tests passed\n";
}
