// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// Rearming deadlines and generation evidence, independent of storage credit.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"

using namespace selected_test;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static Id key(const Constructor& c, Pipe source, Pipe observer, unsigned number)
    {
        const auto& keys = c.frontier.keys();
        for (Id i = 0; i < keys.size(); ++i) {
            if (keys[i].source == source && keys[i].observer == observer && keys[i].key == number) {
                return i;
            }
        }
        require(false, "missing rearming fixture key");
        return NoAnalysisId;
    }

    static void earlierDeadline(bool staleReturn, bool scarce = false)
    {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        auto p = base(1, 2);
        p.target.keys[unsigned(P)][unsigned(Q)] = {0};
        if (scarce) {
            p.target.keys[unsigned(Q)][unsigned(P)] = {0};
        }
        p.operations = {op(Q, {}), op(Q, {}), op(P, {}), op(Q, {}), op(P, {}), op(Q, {})};
        SelectedOptions options;
        options.deferredAcyclicAcknowledgments = true;
        Constructor c(p, options);
        c.needsContextualReplay = true;
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        const auto forward = key(c, P, Q, 0);
        const auto purpose = EndpointPurpose::Completion;
        c.ledger.append(1, {Command::Publish, P, Q, 0}, purpose);
        const auto receipt = c.ledger.append(1, {Command::Acquire, P, Q, 0}, purpose);
        c.current = 1;
        c.activeComponent = c.control.component[c.current];
        require(c.deferCommonRearming(forward, receipt, {}), "receipt did not retain its debt");
        // The stale variant receives a prefix published BEFORE the receipt.
        // The other variant carries the right generation, but arrives too late
        // for the earlier source gap selected below.
        c.ledger.append(staleReturn ? 0 : 3, {Command::Publish, Q, P, 0}, purpose);
        c.ledger.append(4, {Command::Acquire, Q, P, 0}, purpose);
        c.current = 5;
        c.activeComponent = c.control.component[c.current];
        require(c.update(), c.result.reason);
        require(!c.canPublishAt(2, forward), "late or stale return supplied earlier source credit");
        require(c.canPublishAt(5, forward) != staleReturn, "return generation was not distinguished");
        Cut publication = 2;
        SelectedDecision decision;
        if (scarce) {
            const auto version = c.ledger.version();
            require(!c.edge(P, Q, publication, false, decision) && c.ledger.version() == version,
                    "scarce-key refusal changed the ledger or borrowed an unavailable return");
            return;
        }
        require(c.edge(P, Q, publication, false, decision), "earlier deadline lost its helper: " + c.result.reason);
        require(publication == 2 && decision.repairedAcquisition == receipt && decision.repairReverseKey == 1,
                "earlier reuse moved its source or stole the neighboring return key");
        c.observeDeferredRearming(decision);
        require(c.result.rearming.front().reusePublications.size() == 1, "repaired reuse was not recorded");
        const auto words = c.ledger.commands();
        require(checkCausalFrontier(p, words).accepted && bool(oahs_oracle::graph(p, words, {0,1,2,3,4,5})),
                "earlier deadline failed independent checking");
        auto closed = words;
        auto& sourceWord = closed[2];
        const auto wait = std::find_if(sourceWord.begin(), sourceWord.end(), [&](const Command& command) {
            return command.kind == Command::Acquire && command.source == Q && command.observer == P;
        });
        require(wait != sourceWord.end(), "missing helper at the source gap");
        const auto command = *wait;
        sourceWord.erase(wait);
        closed[1].push_back(command);
        oahs_oracle::PayloadOrder before, after;
        require(bool(oahs_oracle::graph(p, closed, {0,1,2,3,4,5}, {}, nullptr, nullptr, &before)) &&
                bool(oahs_oracle::graph(p, words, {0,1,2,3,4,5}, {}, nullptr, nullptr, &after)) &&
                std::includes(before.begin(), before.end(), after.begin(), after.end()),
                "earlier deadline repair added payload ordering");
    }

    static void sharedReturnAndLaterEdit()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V;
        auto p = base(1, 2);
        p.operations = {op(P, {}), op(Q, {}), op(Q, {}), op(Q, {}), op(P, {}), op(Q, {})};
        SelectedOptions options;
        options.deferredAcyclicAcknowledgments = true;
        Constructor c(p, options);
        c.needsContextualReplay = true;
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        const auto purpose = EndpointPurpose::Completion;
        for (unsigned number : {0, 1}) {
            c.current = number + 1;
            c.ledger.append(c.current, {Command::Publish, P, Q, number}, purpose);
            const auto receipt = c.ledger.append(c.current, {Command::Acquire, P, Q, number}, purpose);
            require(c.deferCommonRearming(key(c, P, Q, number), receipt, {}), "shared debt was not recorded");
        }
        const auto sent = c.ledger.append(3, {Command::Publish, Q, P, 0}, purpose);
        const auto received = c.ledger.append(4, {Command::Acquire, Q, P, 0}, purpose);
        c.current = 5;
        c.activeComponent = c.control.component[c.current];
        require(c.update(), c.result.reason);
        for (unsigned number : {0, 1}) {
            require(c.canPublishAt(5, key(c, P, Q, number)), "one actual return did not cover both consumptions");
            SelectedDecision decision;
            decision.endpoints.push_back(c.ledger.append(5, {Command::Publish, P, Q, number}, purpose));
            decision.endpoints.push_back(c.ledger.append(5, {Command::Acquire, P, Q, number}, purpose));
            require(c.update(), c.result.reason);
            c.observeDeferredRearming(decision);
            require(c.result.rearming[number].reusePublications.size() == 1, "shared return lost a generation");
        }
        require(c.result.work.acknowledgments == 0 &&
                bool(oahs_oracle::graph(p, c.ledger.commands(), {0,1,2,3,4,5})),
                "shared actual return needed private helpers or lost rearming");
        // Provenance is not cached authority: invalidating the actual return
        // must invalidate every reuse it supported, in incremental and cold replay.
        c.ledger.erase(sent);
        c.ledger.erase(received);
        require(!c.update() && c.cache.failure == FrontierFailure::ConsumptionNotEstablished,
                "later edit retained obsolete deferred-return credit");
        require(!checkCausalFrontier(p, c.ledger.commands()).accepted &&
                !bool(oahs_oracle::graph(p, c.ledger.commands(), {0,1,2,3,4,5})),
                "independent checking accepted the removed supporting return");
    }
};
} // namespace mlir::pto::oahs::selected

namespace {
void conditionalNoReuse()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    auto p = base(2, 1);
    p.operations = {op(Q, {}), op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true}})};
    p.body = seq({leaf(0), {o::Region::Choice, {leaf(1), seq({})}}, leaf(2),
                  {o::Region::Choice, {leaf(3), seq({})}}});
    o::SelectedOptions options;
    options.finalHelperTrials = false;
    const auto before = o::constructSelectedPlan(p, {}, options);
    options.deferredAcyclicAcknowledgments = true;
    const auto after = o::constructSelectedPlan(p, {}, options);
    require(before.success && after.success && before.work.acknowledgments && !after.work.acknowledgments,
            "a possible downstream branch allocated a helper without key reuse");
    require(after.rearming.size() == 1 && after.rearming.front().reusePublications.empty(),
            "unused debt invented a publication deadline");
    for (const auto& visits : {std::vector<unsigned>{0,1,2,3}, std::vector<unsigned>{0,2,3},
                              std::vector<unsigned>{0,1,2}, std::vector<unsigned>{0,2}}) {
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        require(bool(oahs_oracle::graph(p, before.commands, visits, {}, nullptr, nullptr, &oldOrder)) &&
                bool(oahs_oracle::graph(p, after.commands, visits, {}, nullptr, nullptr, &newOrder)),
                "unused conditional debt failed independent checking");
        require(std::includes(oldOrder.begin(), oldOrder.end(), newOrder.begin(), newOrder.end()),
                "unused conditional debt added ordering");
        if (visits.back() == 3) {
            require(oldOrder != newOrder, "unused conditional debt retained unnecessary serialization");
        }
    }
}

void independentReaders()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
    auto p = base(2, 2);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}), op(R, {{0, true, false}}),
                    op(P, {{0, false, true}}), op(P, {{1, false, true}}), op(Q, {{1, true, false}})};
    p.body = seq({{o::Region::Choice, {leaf(0), seq({})}}, leaf(1), leaf(2), leaf(3), leaf(4), leaf(5)});
    o::SelectedOptions options;
    options.finalHelperTrials = false;
    const auto before = o::constructSelectedPlan(p, {}, options);
    options.deferredAcyclicAcknowledgments = true;
    const auto after = o::constructSelectedPlan(p, {}, options);
    require(before.success && after.success && !after.rearming.empty(),
            "independent-reader return construction failed: " + before.reason + " / " + after.reason);
    for (const auto& visits : {std::vector<unsigned>{0,1,2,3,4,5}, std::vector<unsigned>{1,2,3,4,5}}) {
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        require(bool(oahs_oracle::graph(p, before.commands, visits, {}, nullptr, nullptr, &oldOrder)) &&
                bool(oahs_oracle::graph(p, after.commands, visits, {}, nullptr, nullptr, &newOrder)),
                "event-consumption return lost an independent physical reader");
        require(std::includes(oldOrder.begin(), oldOrder.end(), newOrder.begin(), newOrder.end()),
                "independent-reader rearming added ordering");
    }
}

void conditionalRequiredReturn()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    auto p = base(3, 2);
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    p.operations = {op(Q, {}), op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(P, {{1, false, true}}), op(P, {{0, false, true}}),
                    op(P, {{2, false, true}}), op(Q, {{2, true, false}})};
    p.body = seq({leaf(0), {o::Region::Choice, {leaf(1), seq({})}}, leaf(2), leaf(3),
                  {o::Region::Choice, {seq({leaf(4), leaf(5), leaf(6)}), seq({})}}});
    o::SelectedOptions options;
    options.finalHelperTrials = false;
    const auto before = o::constructSelectedPlan(p, {}, options);
    options.deferredAcyclicAcknowledgments = true;
    const auto after = o::constructSelectedPlan(p, {}, options);
    require(before.success && after.success, "conditional required return: " + before.reason + " / " + after.reason);
    require(!after.rearming.empty() && !after.rearming.front().reusePublications.empty(),
            "conditional actual return was not used before key reuse");
    for (const auto& endpoint : after.ledger) {
        require(endpoint.purpose != o::EndpointPurpose::ConsumptionAcknowledgment ||
                endpoint.acknowledges != after.rearming.front().acquisition,
                "conditional required return still allocated the original private helper");
    }
    for (const auto& visits : {std::vector<unsigned>{0,1,2,3,4,5,6}, std::vector<unsigned>{0,2,3,4,5,6},
                              std::vector<unsigned>{0,1,2,3}, std::vector<unsigned>{0,2,3}}) {
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        require(bool(oahs_oracle::graph(p, before.commands, visits, {}, nullptr, nullptr, &oldOrder)) &&
                bool(oahs_oracle::graph(p, after.commands, visits, {}, nullptr, nullptr, &newOrder)),
                "conditional required return lost matching or consumption");
        require(std::includes(oldOrder.begin(), oldOrder.end(), newOrder.begin(), newOrder.end()),
                "conditional required return added ordering");
    }
}
} // namespace

int main()
{
    o::selected::ReplayTestAccess::earlierDeadline(false);
    o::selected::ReplayTestAccess::earlierDeadline(true);
    o::selected::ReplayTestAccess::earlierDeadline(false, true);
    o::selected::ReplayTestAccess::earlierDeadline(true, true);
    o::selected::ReplayTestAccess::sharedReturnAndLaterEdit();
    conditionalNoReuse();
    independentReaders();
    conditionalRequiredReturn();
}
