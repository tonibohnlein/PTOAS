// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include <array>
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
using namespace selected_test;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void uncoveredProducerProposal() {
        auto p = base(2);
        p.operations = {op(Pipe::MTE2, {{0, false, true}}), op(Pipe::MTE2, {{0, false, true}}),
                        op(Pipe::V, {{1, false, true}}), op(Pipe::MTE2, {{1, true, false}})};
        Constructor c(p);
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        const auto version = c.ledger.version();
        RecurringRequirement supported;
        supported.cell = 1; supported.cells = {1}; supported.source = Pipe::V; supported.observer = Pipe::MTE2;
        supported.publications = {3}; supported.acquisitions = {3}; supported.qualifiedCycle = true;
        supported.repairFreeProducers.insert(Pipe::MTE2);
        require(c.recurring({supported}), "uncovered producer proposal escaped optional fallback");
        require(c.result.work.rejectedSupportProposals == 1 && c.result.work.rejectedProtocolProposals == 0,
                "valid event protocol hid an uncovered producer repair");
        require(c.ledger.version() == version && c.ledger.records().empty() &&
                c.recurringKeys.empty() && c.result.channels.empty() && !c.needsContextualReplay,
                "rejected producer support leaked committed state");
        require(c.run({}).success, "ordinary construction failed after support rejection");
    }
    static void invalidProposal() {
        auto p = base(1);
        p.operations = {op(Pipe::MTE2, {{0, false, true}}), op(Pipe::V, {{0, true, false}})};
        Constructor c(p);
        std::string reason;
        require(c.ledger.initialize({}, reason), reason);
        const auto version = c.ledger.version();
        RecurringRequirement bad;
        bad.cell = 0; bad.cells = {0}; bad.source = Pipe::MTE2; bad.observer = Pipe::V;
        bad.publications = {1}; bad.qualifiedCycle = true;
        require(c.recurring({bad}), "optional rejection escaped as constructor failure");
        require(c.ledger.version() == version && c.ledger.records().empty() &&
                c.recurringKeys.empty() && c.result.channels.empty() && !c.needsContextualReplay,
                "invalid optional proposal leaked committed state");
        require(c.result.work.rejectedProtocolProposals == 1, "invalid protocol not rejected");
        require(c.run({}).success, "ordinary construction failed after optional rejection");
    }
};
}
namespace {
const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
o::Program cohort() {
    auto p = base(3, 2);
    // Only the ordinary two-engine vocabulary: no relay hides starvation.
    for (unsigned a = 0; a < o::PipeCount; ++a) for (unsigned b = 0; b < o::PipeCount; ++b)
        if (!((a == unsigned(P) && b == unsigned(Q)) || (a == unsigned(Q) && b == unsigned(P))))
            p.target.keys[a][b].clear();
    p.operations = {op(P, {{2,false,true}}), op(Q, {{2,true,false}}),
        op(P, {{0,false,true}}), op(Q, {{0,true,false}}),
        op(P, {{1,false,true}}), op(Q, {{1,true,false}})};
    p.body = seq({leaf(0), leaf(1), {o::Region::For,{seq({leaf(2),leaf(3),leaf(4),leaf(5)})},0,true}});
    auto imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    return imported.program;
}
void starvation() {
    auto p = cohort();
    o::SelectedOptions ordinary; ordinary.recurring = false;
    auto reference = o::constructSelectedPlan(p, {}, ordinary);
    require(reference.success, "ordinary reference must work: " + reference.reason);
    auto plan = o::constructSelectedPlan(p);
    std::cout << "cohort channels=" << plan.channels.size() << " rejected=" << plan.work.rejectedResourceProposals << " success=" << plan.success << '\n';
    require(plan.success, "optional cohort starved ordinary X: " + plan.reason);
    require(plan.work.rejectedResourceProposals == 1 && plan.channels.empty(), "exact-fit cohort was not declined");
}
void deferredAcknowledgment() {
    auto p = base(3, 2);
    p.operations = {op(Q,{{2,true,false}}), op(P,{{0,false,true}}),
                    op(Q,{{0,true,false}}), op(P,{{1,false,true}})};
    p.body = seq({leaf(0), {o::Region::Choice,{leaf(1),seq({})}}, leaf(2),leaf(3)});
    o::SelectedOptions before; before.finalHelperTrials = false;
    auto baseline = o::constructSelectedPlan(p,{},before);
    auto after = before; after.deferredAcyclicAcknowledgments = true;
    auto plan = o::constructSelectedPlan(p,{},after);
    require(baseline.success && plan.success, "deferred acyclic fixture construction failed: " + plan.reason);
    require(plan.work.deferredAcknowledgments != 0 && plan.work.acknowledgments < baseline.work.acknowledgments,
            "later payload without key reuse still forces acknowledgment");
    for (auto visits : {std::vector<unsigned>{0,1,2,3},std::vector<unsigned>{0,2,3}})
        require(bool(oahs_oracle::graph(p,plan.commands,visits,{{0,unsigned(visits.size()-1)}})),
                "deferred acknowledgment lost safety or unrelated-work independence");
    // A later actual reuse still needs its consumption path. No return means
    // either another virgin key, a real helper, or a refusal, never invented credit.
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    p.operations.push_back(op(P,{{1,false,true}}));
    p.operations.push_back(op(Q,{{1,true,false}}));
    p.body.children.push_back(leaf(4)); p.body.children.push_back(leaf(5));
    plan = o::constructSelectedPlan(p,{},after);
    require(plan.success, "real later reuse lost its repair: " + plan.reason);
    require(bool(oahs_oracle::graph(p,plan.commands,{0,1,2,3,4,5})), "real reuse lacks consumption evidence");
}
void wordGapBaseline() {
    auto p = base(3, 4);
    p.operations = {op(R,{{1,true,false}}),op(P,{{0,false,true}}),
        op(Q,{{0,true,false},{2,false,true}}),op(Q,{{1,false,true}}),op(P,{{0,false,true}})};
    auto baseline = o::constructSelectedPlan(p);
    require(baseline.success,baseline.reason);
    require(bool(oahs_oracle::graph(p,baseline.commands,{0,1,2,3,4})), "baseline must be safe");
    o::SelectedOptions options; options.sourceGaps = true;
    auto candidate = o::constructSelectedPlan(p, {}, options);
    require(candidate.success, candidate.reason);
    require(bool(oahs_oracle::graph(p,candidate.commands,{0,1,2,3,4},{{0,4}})),
            "exact source gap still imports unrelated R completion");
    require(candidate.work.gapPublications != 0, "source-gap fixture did not select a gap");
    std::cout << "word_gap_baseline_forbidden=" << !bool(oahs_oracle::graph(p,baseline.commands,{0,1,2,3,4},{{0,4}})) << '\n';
    // An outward publication at the shared cut must retain its own source
    // meaning; gap insertion may not export the unrelated incoming wait.
    p.operations.push_back(op(o::Pipe::MTE1,{{2,true,false}}));
    o::Commands fixed(o::commandCutCount(p));
    fixed[3] = {{o::Command::Publish,Q,o::Pipe::MTE1,0}};
    fixed[5] = {{o::Command::Acquire,Q,o::Pipe::MTE1,0}};
    auto outward = o::constructSelectedPlan(p,fixed,options);
    require(outward.success,outward.reason);
    require(bool(oahs_oracle::graph(p,outward.commands,{0,1,2,3,4,5},{{0,4},{0,5}})),
            "source gap broadened the outward publication");
    // Removing real readiness must still fail, even in a gap-aware plan.
    auto broken = outward.commands;
    for (auto& word : broken) word.erase(std::remove_if(word.begin(),word.end(),[&](const auto& command) {
        return (command.kind == o::Command::Publish || command.kind == o::Command::Acquire) &&
               command.source == P && command.observer == Q;
    }),word.end());
    require(!o::checkCausalFrontier(p,broken).accepted, "gap admission invented missing readiness");

}
// Supplied-protocol witness for the *contextual* merge certificate. It does
// not claim the constructor emits either plan: all fixed endpoints are explicit.
void commonFrontierContext() {
    using C = o::Command;
    unsigned cases = 0;
    for (unsigned episodes : {1u, 2u, 4u}) for (bool outwardBetween : {false, true}) {
        auto p = base(3, 4);
        for (unsigned i = 0; i < episodes; ++i) {
            p.operations.push_back(op(Q, {{2,false,true}})); // z producer
            p.operations.push_back(op(P, {{0,false,true}})); // A
            p.operations.push_back(op(P, {{1,false,true}})); // B
            p.operations.push_back(op(Q, {{0,true,false},{1,true,false}}));
            p.operations.push_back(op(R, {{2,true,false}}));
        }
        o::Commands separate(p.operations.size()+1), merged(p.operations.size()+1);
        auto add = [](o::Commands& commands, unsigned cut, C::Kind kind,
                      o::Pipe source, o::Pipe observer, unsigned key = 0) {
            commands[cut].push_back({kind,source,observer,key});
        };
        for (unsigned i = 0; i < episodes; ++i) {
            const unsigned start = 5*i;
            for (auto* commands : {&separate, &merged}) {
                if (i) {
                    add(*commands,start,C::Acquire,R,Q);
                    add(*commands,start+1,C::Acquire,Q,P);
                }
                add(*commands,start+4,C::Publish,Q,P); // actual AB reader return
                add(*commands,start+4,C::Acquire,Q,R);
                add(*commands,start+5,C::Publish,R,Q); // actual z reader return
            }
            add(separate,start+2,C::Publish,P,Q,0); // early A source
            add(separate,start+3,C::Publish,P,Q,1); // later B source
            add(separate,start+3,C::Acquire,P,Q,0);
            if (outwardBetween) add(separate,start+3,C::Publish,Q,R);
            add(separate,start+3,C::Acquire,P,Q,1);
            if (!outwardBetween) add(separate,start+3,C::Publish,Q,R);
            // One later source, same consumer cut. Both original waits are
            // sufficient for the Q payload, but their intermediate export differs.
            add(merged,start+3,C::Publish,P,Q);
            add(merged,start+3,C::Acquire,P,Q);
            add(merged,start+3,C::Publish,Q,R);
        }
        for (auto* commands : {&separate, &merged}) {
            add(*commands,p.operations.size(),C::Acquire,Q,P);
            add(*commands,p.operations.size(),C::Acquire,R,Q);
        }
        std::vector<unsigned> visits(p.operations.size());
        std::iota(visits.begin(),visits.end(),0);
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        require(bool(oahs_oracle::graph(p,separate,visits,{},nullptr,nullptr,&oldOrder)),
                "separate-frontier fixture lacks memory/balance/rearming");
        require(bool(oahs_oracle::graph(p,merged,visits,{},nullptr,nullptr,&newOrder)),
                "merged-frontier fixture lacks memory/balance/rearming");
        require(o::checkCausalFrontier(p,separate).accepted &&
                o::checkCausalFrontier(p,merged).accepted,
                "production checker disagrees with supplied frontier protocols");
        if (!outwardBetween) {
            require(oldOrder == newOrder, "private common consumer changed payload order");
        } else {
            require(std::includes(newOrder.begin(),newOrder.end(),oldOrder.begin(),oldOrder.end()) &&
                    oldOrder != newOrder, "outward publication did not expose broader order");
            for (unsigned i = 0; i < episodes; ++i) {
                const auto unwanted = std::make_pair(2*(5*i+2)+1,2*(5*i+4));
                require(!oldOrder.count(unwanted) && newOrder.count(unwanted),
                        "later B load must newly gate the unrelated z reader");
            }
        }
        // Delete a complete return channel so event balance alone cannot mask
        // the loss of real reuse and forward-key consumption evidence.
        if (episodes > 1) {
            auto broken = merged;
            for (auto& word : broken)
                word.erase(std::remove_if(word.begin(),word.end(),[&](const C& c) {
                    return c.source == Q && c.observer == P;
                }),word.end());
            const auto verdict = oahs_oracle::graph(p,broken,visits);
            require(verdict.balanced && !verdict.hazards && !verdict.rearm,
                    "missing return must lose memory and rearming despite balanced tokens");
            require(!o::checkCausalFrontier(p,broken).accepted,
                    "production checker accepted missing return");
        }
        std::cout << "frontier_context episodes=" << episodes << " outward_between=" << outwardBetween
                  << " relations=" << oldOrder.size() << "->" << newOrder.size() << '\n';
        ++cases;
    }
    require(cases == 6, "frontier context campaign incomplete");
}

void noMotionAndRandom() {
    auto p = base(2,4);
    p.operations = {op(P,{{0,false,true}}),op(P,{{1,false,true}}),op(Q,{{0,true,false},{1,true,false}})};
    auto loop = o::makePeriodicLoop(p,1,{}); require(loop.success,loop.reason);
    o::SelectedOptions options; options.movingFrontiers = false;
    auto plan = o::constructSelectedPlan(loop.program,{},options);
    require(plan.success, "no-motion protocol failed: " + plan.reason);
    std::cout << "no_motion_channels=" << plan.channels.size() << '\n';
    // Broad finite safety campaign for the two narrow acyclic options. The
    // independent oracle contains actual primitive edges, never desired edges.
    std::mt19937 random(2031);
    unsigned accepted = 0;
    for (unsigned sample = 0; sample < 200; ++sample) {
        auto input = base(3,2);
        for (unsigned i = 0; i < 7; ++i) {
            const auto pipe = std::array<o::Pipe,3>{P,Q,R}[random()%3];
            const unsigned cell = random()%3;
            const bool write = random()%2;
            input.operations.push_back(op(pipe,{{cell,!write,write}}));
        }
        options.sourceGaps = true; options.deferredAcyclicAcknowledgments = true;
        options.equalCoverageBinding = true;
        auto candidate = o::constructSelectedPlan(input,{},options);
        require(candidate.success,"finite acyclic construction failure: " + candidate.reason);
        require(bool(oahs_oracle::graph(input,candidate.commands,{0,1,2,3,4,5,6})),"independent finite oracle rejected plan");
        ++accepted;
    }
    std::cout << "acyclic_option_cases=" << accepted << '\n';
}
}
int main() {
    o::selected::ReplayTestAccess::invalidProposal();
    o::selected::ReplayTestAccess::uncoveredProducerProposal();
    starvation(); wordGapBaseline(); deferredAcknowledgment(); commonFrontierContext(); noMotionAndRandom();
}
