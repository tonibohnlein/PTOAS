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
} // namespace

namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
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
    o::selected::ReplayTestAccess::certificateBoundaries();
    std::cout << "publication prefix tests passed\n";
}
