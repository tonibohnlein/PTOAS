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

namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void finalReadSource()
    {
        using namespace selected_test;
        const auto P = Pipe::MTE2, Q = Pipe::MTE1, M = Pipe::M;
        for (bool single : {false, true}) for (bool repeated : {false, true}) for (bool reused : {false, true}) {
            auto p = base(1, 2);
            p.operations = {op(P, {{0,false,true}}), op(Q, {{0,true,false}}),
                            op(M, {}), op(P, {{0,false,true}})};
            ObservedControl g; g.qualification = "final reader with a shared acknowledgment word";
            g.sites.resize(11); g.entry=0; g.exit=10;
            for (unsigned i=0;i<11;++i) { g.observations.push_back({i,{},true});g.sites[i].observation=i; }
            for (unsigned i=0;i<10;++i) g.sites[i].successors={i+1};
            g.sites[3].successors={4,8}; g.sites[7].successors={3};g.sites[7].backedgeOwners={2};
            if(repeated){g.sites[9].successors={0,10};g.sites[9].backedgeOwners={0,NoControlId};}
            g.sites[0].operation=0;g.sites[4].operation=1;g.sites[6].operation=2;g.sites[8].operation=3;
            g.loops.push_back({2,2,8,{3,4,5,6,7},4,true});p.observed=g;
            ReaderVisitRegion request;request.owner=2;request.step=64;
            request.lastPublications={5};request.singleVisit=single;request.finalSourceGaps=true;
            auto refined=refineReaderVisits(p,request);
            require(refined.success,"final-only refinement: "+refined.reason);p=std::move(refined.program);
            SelectedOptions settings;settings.finalReadSources=true;
            Constructor c(p,settings);std::string reason;
            require(c.control.complete,c.control.reason);
            require(c.ledger.initialize(Commands(commandCutCount(p)),reason),reason);
            c.ledger.append(0,{Command::Barrier,P,P,0},EndpointPurpose::Fixed);
            if(reused) for(auto cmd:std::vector<Command>{{Command::Publish,Q,P,0},{Command::Acquire,Q,P,0}})
                c.ledger.append(1,cmd,EndpointPurpose::Fixed);
            for(auto cmd:std::vector<Command>{{Command::Publish,P,Q,0},{Command::Acquire,P,Q,0},
                    {Command::Publish,Q,P,1},{Command::Acquire,Q,P,1}})
                c.ledger.append(1,cmd,EndpointPurpose::Fixed);
            for(auto cmd:std::vector<Command>{{Command::Publish,Q,M,0},{Command::Acquire,Q,M,0},
                    {Command::Publish,M,Q,0},{Command::Acquire,M,Q,0}})
                c.ledger.append(5,cmd,EndpointPurpose::Fixed);
            for(auto cmd:std::vector<Command>{{Command::Publish,P,Q,1},{Command::Acquire,P,Q,1}})
                c.ledger.append(9,cmd,EndpointPurpose::Fixed);
            c.current=8;c.activeComponent=c.control.component[8];c.needsContextualReplay=true;
            require(c.replay(),c.result.reason);
            std::vector<FrontierRequirement> required;
            for(auto r:c.residual()) if(r.source==Q)required.push_back(r);
            require(!required.empty(),"missing reader obligation");
            Group group;group.source=Q;group.requirements=required;
            require(c.finalReadFrontier(Q,required,group),"qualified final-read source declined");
            const auto gap=group.publication;
            require(c.ledger.word(5).size()==4,"source selection changed acknowledgment");
            if(repeated){
                auto commands=c.ledger.commands();
                const auto readySet=c.ledger.word(1)[reused ? 2 : 0];
                const auto readyWait=c.ledger.word(1)[reused ? 3 : 1];
                const auto wait=c.ledger.word(9).back();c.ledger.erase(wait);c.ledger.erase(readyWait);
                require(!c.consumptionBeforeNextPublication(gap,8,group.forwardKey),
                        "missing return manufactured rearming");
                c.ledger.restoreAfter(wait,c.ledger.word(9).back());
                c.ledger.restoreAfter(readyWait,readySet);
                require(c.replay(),c.result.reason);group.version=c.ledger.version();
            }
            require(c.bind(group,RequirementStage::Known),c.result.reason);
            require(c.result.work.finalReadPublications==1,"final source not selected");
            require(c.ledger.word(5).size()==4,"final binding removed acknowledgment");
            require(checkCausalFrontier(p,c.ledger.commands()).accepted,"final source protocol invalid");
            for (unsigned trips : (single ? std::vector<unsigned>{1} : std::vector<unsigned>{2,3,4})) {
                auto flat=p;flat.observed.reset();flat.operations.clear();flat.body={};
                Commands words;std::vector<Command> pending;
                unsigned iteration=0,entry=0;Cut at=p.observed->entry;
                const auto commands=c.ledger.commands();
                for(unsigned steps=0;steps<200;++steps) {
                    const auto& site=p.observed->sites[at];
                    pending.insert(pending.end(),commands[at].begin(),commands[at].end());
                    if(site.operation!=NoControlId) {
                        flat.operations.push_back(p.operations[site.operation]);words.push_back(pending);pending.clear();
                    }
                    if(at==p.observed->exit)break;
                    if(at==2)iteration=0;
                    if(at==3) at=site.successors[(++iteration==trips)?1:0];
                    else if(at==9 && repeated)at=site.successors[(++entry==2)?1:0];
                    else at=site.successors.front();
                }
                require(at==p.observed->exit,"finite final-reader trace did not terminate");
                words.push_back(pending);auto broad=words;
                for(auto& word:broad) {
                    auto release=std::find_if(word.begin(),word.end(),[&](const auto& cmd){
                        return cmd.kind==Command::Publish&&cmd.source==Q&&cmd.observer==P&&cmd.key==0;
                    });
                    if(release!=word.end() && std::any_of(word.begin(),word.end(),[&](const auto& cmd){
                        return cmd.kind==Command::Acquire&&cmd.source==M&&cmd.observer==Q;
                    })){auto command=*release;word.erase(release);word.push_back(command);}
                }
                std::vector<unsigned> visits(flat.operations.size());std::iota(visits.begin(),visits.end(),0);
                oahs_oracle::PayloadOrder earlyOrder,broadOrder;
                require(bool(oahs_oracle::graph(flat,words,visits,{},nullptr,nullptr,&earlyOrder)),
                        "constructed final source lost finite matching/rearming");
                require(bool(oahs_oracle::graph(flat,broad,visits,{},nullptr,nullptr,&broadOrder)),
                        "late release control invalid");
                require(std::includes(broadOrder.begin(),broadOrder.end(),earlyOrder.begin(),earlyOrder.end()),
                        "constructed final source added order");
                if(trips>1)require(earlyOrder.size()<broadOrder.size(),"constructed release failed to preserve overlap");
            }
            // A subsequent word-start insertion cannot move before the saved
            // publication boundary. Its endpoint identity, not offset 0, pins it.
            const auto release=c.ledger.word(gap).front();
            const auto later=c.ledger.prepend(gap,{Command::Barrier,Q,Q,0},EndpointPurpose::Fixed);
            require(c.ledger.word(gap).front()==release,"later insertion broadened protected source gap");
            c.ledger.erase(later);
            auto reload=p;reload.operations[2].pipe=Q;reload.operations[2].accesses={{0,true,false}};
            Constructor negative(reload,settings);
            require(negative.ledger.initialize(c.ledger.commands(),reason),reason);
            negative.current=8;negative.activeComponent=negative.control.component[8];
            negative.needsContextualReplay=true;
            // Freshness is structural and must reject a later physical reader,
            // independent of whether another transfer happens to cover it.
            negative.replay();
            require(!negative.finalReadGap(gap,Q,required),"later reader reused stale final prefix");
        }
    }
    static void crossControlAcknowledgment()
    {
        using namespace selected_test;
        const auto P = Pipe::MTE2, Q = Pipe::MTE1;
        auto p = base(1, 2);
        p.target.keys[unsigned(P)][unsigned(Q)] = {0};
        p.operations = {op(P, {}), op(P, {}), op(Q, {})};
        ObservedControl g; g.qualification = "return through an original branch";
        g.sites.resize(9); g.entry = 0; g.exit = 8;
        for (unsigned i = 0; i < 9; ++i) {
            g.observations.push_back({i, {}, true}); g.sites[i].observation = i;
        }
        g.sites[0].successors = {1}; g.sites[1].successors = {2};
        g.sites[2].successors = {3,4}; g.sites[3].successors = {5}; g.sites[4].successors = {5};
        g.sites[5].successors = {6}; g.sites[6].successors = {7}; g.sites[7].successors = {1,8};
        g.sites[7].backedgeOwners = {1, NoAnalysisId};
        g.sites[3].operation = 0; g.sites[5].operation = 1; g.sites[6].operation = 2;
        p.observed = g;
        SelectedOptions settings; settings.firstWriteConsumers = true;
        Constructor c(p, settings); std::string reason;
        require(c.control.complete, c.control.reason);
        require(c.ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        Id oldWait = NoAnalysisId;
        for (auto command : std::vector<Command>{{Command::Publish,Q,P,0},{Command::Acquire,Q,P,0},
                {Command::Publish,P,Q,0},{Command::Acquire,P,Q,0}})
            oldWait = c.ledger.append(1,command,EndpointPurpose::Completion);
        c.ledger.append(7,{Command::Publish,Q,P,1},EndpointPurpose::Completion);
        c.ledger.append(7,{Command::Acquire,Q,P,1},EndpointPurpose::Completion);
        c.current=6; c.activeComponent=c.control.component[6]; c.needsContextualReplay=true;
        require(c.replay(),c.result.reason);
        require(!c.control.straight(1,6), "fixture failed to cross control");
        Id forward=NoAnalysisId, reverse=NoAnalysisId;
        for (Id key=0;key<c.frontier.keys().size();++key) {
            const auto& k=c.frontier.keys()[key];
            if(k.source==P && k.observer==Q && k.key==0) forward=key;
            if(k.source==Q && k.observer==P && k.key==0) reverse=key;
        }
        require(c.crossControlReturn(oldWait,5,forward,reverse), "balanced branch return declined");
        require(!c.crossControlReturn(oldWait,3,forward,reverse), "bypassed return acquisition accepted");
        const auto interference=c.ledger.append(3,{Command::Publish,Q,P,0},EndpointPurpose::Completion);
        require(!c.crossControlReturn(oldWait,5,forward,reverse), "intervening reverse-key use accepted");
        c.ledger.erase(interference);
        Cut publication=5; SelectedDecision decision;
        require(c.edge(P,Q,publication,false,decision),c.result.reason);
        require(publication==5 && decision.repairedAcquisition==oldWait,
                "cross-control return lost source gap or actual consumption");
        require(checkCausalFrontier(p,c.ledger.commands()).accepted,"cross-control generations invalid");
        auto missing=c.ledger.commands();
        missing[5].erase(std::remove_if(missing[5].begin(),missing[5].end(),[&](const auto& cmd){
            return cmd.kind==Command::Acquire && cmd.source==Q && cmd.observer==P && cmd.key==0;
        }),missing[5].end());
        require(!checkCausalFrontier(p,missing).accepted,"missing return accepted");
        for(unsigned trips:{1u,2u,4u}) for(unsigned branch:{3u,4u}) {
            auto flat=p;flat.observed.reset();flat.operations.clear();flat.body={};
            Commands words;std::vector<Command> pending;
            std::vector<unsigned> middle;
            auto commands=c.ledger.commands();
            auto append=[&](unsigned site){
                pending.insert(pending.end(),commands[site].begin(),commands[site].end());
                if(g.sites[site].operation==NoAnalysisId)return;
                if(site==5)middle.push_back(flat.operations.size());
                flat.operations.push_back(p.operations[g.sites[site].operation]);
                words.push_back(std::move(pending));pending.clear();
            };
            append(0);
            for(unsigned t=0;t<trips;++t){append(1);append(2);append(t%2?7-branch:branch);append(5);append(6);append(7);}
            append(8);words.push_back(std::move(pending));
            auto late=words;
            for(auto at:middle){
                auto& word=late[at];
                auto lastSet=std::find_if(word.rbegin(),word.rend(),[&](const auto& cmd){
                    return cmd.kind==Command::Publish&&cmd.source==P&&cmd.observer==Q;
                });
                require(lastSet!=word.rend(),"missing early forward set");
                auto set=std::prev(lastSet.base());
                late[at+1].insert(late[at+1].begin(),*set);word.erase(set);
            }
            std::vector<unsigned> visits(flat.operations.size());std::iota(visits.begin(),visits.end(),0);
            oahs_oracle::PayloadOrder earlyOrder,lateOrder;
            require(bool(oahs_oracle::graph(flat,words,visits,{},nullptr,nullptr,&earlyOrder)),"finite return invalid");
            require(bool(oahs_oracle::graph(flat,late,visits,{},nullptr,nullptr,&lateOrder)),"late return control invalid");
            require(std::includes(lateOrder.begin(),lateOrder.end(),earlyOrder.begin(),earlyOrder.end())&&
                earlyOrder.size()<lateOrder.size(),"cross-control repair lost early overlap");
        }
    }
    static void sharedWordNewConsumption(bool existingReturn = false, bool noSpareReturn = false, bool reserved = false, bool closedRole = false)
    {
        using namespace selected_test;
        const auto P = Pipe::M, Q = Pipe::MTE1;
        auto p = base(1, 2);
        p.operations = {op(Q, {}), op(P, {}), op(P, {}), op(Q, {})};
        ObservedControl g;
        g.qualification = "shared first/later publication words";
        g.sites.resize(13);
        g.entry = 0; g.exit = 7;
        for (unsigned i = 0; i < 8; ++i) g.observations.push_back({i, {}, true});
        for (unsigned i = 0; i < 8; ++i) g.sites[i].observation = i;
        g.sites[0].successors = {1};
        for (unsigned i = 1; i < 6; ++i) g.sites[i].successors = {i + 1};
        g.sites[6].successors = {8, 7};
        for (unsigned i = 8; i < 13; ++i) {
            g.sites[i].observation = i - 7;
            g.sites[i].successors = {i == 12 ? 6u : i + 1};
        }
        for (auto start : {1u, 8u}) {
            g.sites[start].operation = 0;
            g.sites[start + 1].operation = 1;
            g.sites[start + 3].operation = 2;
            g.sites[start + 4].operation = 3;
        }
        g.sites[12].backedgeOwners = {6};
        p.observed = g;
        if (noSpareReturn) p.target.keys[unsigned(Q)][unsigned(P)] = {0};
        if (reserved || closedRole) p.target.keys[unsigned(P)][unsigned(Q)] = {0};
        SelectedOptions settings;
        settings.firstWriteConsumers = true;
        Constructor c(p, settings);
        require(c.control.complete, c.control.reason);
        std::string reason;
        require(c.ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        for (auto command : std::vector<Command>{{Command::Publish, P, Q, 0},
                {Command::Acquire, P, Q, 0}, {Command::Publish, Q, P, 0},
                {Command::Acquire, Q, P, 0}})
            c.ledger.append(1, command, reserved && command.source == P ?
                EndpointPurpose::RecurringCompletion : EndpointPurpose::Completion, 0);
        if (reserved) {
            c.result.channels.push_back({0, 0, {0}, P, Q, {1}, {1}, 6, 1});
            for (Id key = 0; key < c.frontier.keys().size(); ++key)
                if (c.frontier.keys()[key].source == P && c.frontier.keys()[key].observer == Q) {
                    c.recurringKeys.insert(key); c.closedKeys.insert(key);
                }
        }
        Id borrowedClosed = NoAnalysisId;
        if (closedRole) {
            Id reverse = NoAnalysisId;
            for (Id key = 0; key < c.frontier.keys().size(); ++key) {
                const auto& k = c.frontier.keys()[key];
                if (k.source == P && k.observer == Q && k.key == 0) borrowedClosed = key;
                if (k.source == Q && k.observer == P && k.key == 0) reverse = key;
            }
            c.closedBindings[{P,Q}] = {borrowedClosed,reverse};
            c.closedKeys.insert(borrowedClosed); c.closedKeys.insert(reverse);
        }
        if (existingReturn) {
            c.ledger.append(6, {Command::Publish, Q, P, 1}, EndpointPurpose::Completion);
            c.ledger.append(6, {Command::Acquire, Q, P, 1}, EndpointPurpose::Completion);
        }
        c.current = 5;
        c.activeComponent = c.control.component[c.current];
        c.needsContextualReplay = true;
        require(c.replay(), c.result.reason);
        require(checkCausalFrontier(p, c.ledger.commands()).accepted, "initial closed exchange invalid");
        Cut publication = 3;
        SelectedDecision decision;
        if (reserved) {
            const auto key = *c.recurringKeys.begin();
            require(c.inactiveReservation(3, 5, key), "inactive reservation lacks certificate");
            c.result.channels[0].acquisitions = {2};
            require(!c.inactiveReservation(3, 5, key), "unmaterialized owner endpoint accepted");
            c.result.channels[0].acquisitions = {1};
            c.entryProtocolKeys.insert(key);
            require(!c.inactiveReservation(3, 5, key), "lazy entry reservation borrowed");
            c.entryProtocolKeys.erase(key);
            require(!c.inactiveReservation(3, 1, key), "overlapping owner interval borrowed");
            const auto returnSet = c.ledger.word(6).front();
            const auto returnWait = c.ledger.word(6).back();
            c.ledger.erase(returnWait);
            require(!c.inactiveReservation(3, 5, key), "missing next-publication support accepted");
            c.ledger.restoreAfter(returnWait, returnSet);
        }
        if (closedRole) {
            require(c.inactiveClosedReservation(3,5,borrowedClosed), "closed role has no borrow certificate");
            require(!c.inactiveClosedReservation(3,1,borrowedClosed), "overlapping closed role accepted");
            const auto wait = c.ledger.word(6).back(), sent = c.ledger.word(6).front();
            c.ledger.erase(wait);
            require(!c.inactiveClosedReservation(3,5,borrowedClosed), "missing closed-role rearming accepted");
            c.ledger.restoreAfter(wait,sent);
            c.entryProtocolKeys.insert(borrowedClosed);
            require(!c.inactiveClosedReservation(3,5,borrowedClosed), "entry-owned key borrowed");
            c.entryProtocolKeys.erase(borrowedClosed);
        }
        const auto version = c.ledger.version();
        const bool bound = c.edge(P, Q, publication, false, decision);
        if (noSpareReturn) {
            require(!bound && c.result.failure == SelectedFailure::EventResource,
                    "unproved reverse-key consumption was borrowed");
            require(c.ledger.version() == version && publication == 3,
                    "failed return qualification changed endpoints or source gap");
            return;
        }
        require(bound, c.result.reason);
        if (closedRole) {
            require(c.result.work.closedReservationBorrows == 1 && decision.endpoints.size() == 2,
                    "closed borrow did not preserve helper-free transfer");
            require(c.closedKeys.count(borrowedClosed), "closed owner reservation was released");
        }
        require(publication == 3, "rearming moved the early source gap");
        if (reserved) {
            require(decision.endpoints.size() == 2, "reservation borrowing added a helper");
            require(c.closedKeys.count(*c.recurringKeys.begin()), "borrowing released owner reservation");
        }
        require(checkCausalFrontier(p, c.ledger.commands()).accepted, "shared-word edge lacks rearming");
        require(c.result.work.acknowledgments == (existingReturn ? 0u : 1u),
                "selected return ignored or missing new consumption helper");
        require((reserved || closedRole || c.result.work.splitRearmingQueries == (existingReturn ? 1u : 2u)) && c.result.work.splitRearmingSites != 0,
                "missing bounded neighboring-use query");
        // The earlier source cannot import the unrelated P operation between
        // publication and acquisition. Check multiple complete visits using
        // the independent issue/completion graph, including shared words.
        for (unsigned trips : {1u, 2u, 4u}) {
            auto flat = p; flat.observed.reset(); flat.operations.clear(); flat.body = {};
            Commands words; std::vector<Command> pending;
            std::vector<std::pair<unsigned, unsigned>> forbidden;
            const auto commands = c.ledger.commands();
            auto append = [&](unsigned site) {
                pending.insert(pending.end(), commands[site].begin(), commands[site].end());
                const auto operation = g.sites[site].operation;
                if (operation == NoAnalysisId) return;
                if (operation == 3) forbidden.emplace_back(flat.operations.size() - 1, flat.operations.size());
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            };
            append(0);
            for (unsigned visit = 0; visit < trips; ++visit) {
                for (unsigned site = 1; site <= 5; ++site) append(site + (visit ? 7 : 0));
                append(6);
            }
            append(7); words.push_back(std::move(pending));
            std::vector<unsigned> visits(flat.operations.size());
            std::iota(visits.begin(), visits.end(), 0);
            oahs_oracle::PayloadOrder narrowOrder, lateOrder;
            require(bool(oahs_oracle::graph(flat, words, visits, forbidden, nullptr, nullptr, &narrowOrder)),
                    "rearming broadened early source gap");
            auto late = words;
            for (unsigned visit = 0; visit < trips; ++visit) {
                auto& sourceWord = late[4 * visit + 2];
                const auto at = std::find_if(sourceWord.begin(), sourceWord.end(), [&](const auto& cmd) {
                    return cmd.kind == Command::Publish && cmd.source == P && cmd.observer == Q;
                });
                require(at != sourceWord.end(), "early publication not present at original gap");
                late[4 * visit + 3].insert(late[4 * visit + 3].begin(), *at);
                sourceWord.erase(at);
            }
            require(bool(oahs_oracle::graph(flat, late, visits, {}, nullptr, nullptr, &lateOrder)),
                    "late-source control invalid");
            require(std::includes(lateOrder.begin(), lateOrder.end(), narrowOrder.begin(), narrowOrder.end()) &&
                    narrowOrder.size() < lateOrder.size(), "early-source repair did not preserve order inclusion");
        }
        auto missing = c.ledger.commands();
        for (auto at : {5u, 12u, 6u}) {
            auto& word = missing[at];
            word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& cmd) {
                return cmd.source == Q && cmd.observer == P;
            }), word.end());
        }
        require(!checkCausalFrontier(p, missing).accepted, "old acknowledgment reused for new consumption");
    }
    static void pairEnumeration(unsigned count)
    {
        auto p = selected_test::base(1, 2);
        const auto P = Pipe::MTE2, Q = Pipe::V;
        p.operations = {selected_test::op(P, {{0, true, false}})};
        Constructor c(p);
        c.needsContextualReplay = true;
        std::string reason;
        selected_test::require(c.ledger.initialize(Commands(commandCutCount(p)), reason), reason);
        // Structural-query stress fixture, not an accepted event protocol:
        // the intervening P payload must reject every possible replacement.
        for (unsigned i = 0; i < count; ++i) {
            auto publish = c.ledger.append(0, {Command::Publish, Q, P, 0}, EndpointPurpose::ConsumptionAcknowledgment);
            auto wait = c.ledger.append(0, {Command::Acquire, Q, P, 0}, EndpointPurpose::ConsumptionAcknowledgment);
            c.rememberReturn(publish, wait);
            c.ledger.append(1, {Command::Publish, Q, P, 1}, EndpointPurpose::Completion);
            auto actual = c.ledger.append(1, {Command::Acquire, Q, P, 1}, EndpointPurpose::Completion);
            SelectedDecision decision;
            decision.endpoints = {wait, actual};
            selected_test::require(c.settleRearming(decision), "pair enumeration changed ledger");
            selected_test::require(c.result.work.rearmingPairVisits == uint64_t(i+1)*(i+1),
                                   "old helper/return product was enumerated again");
            // Relevant decision with no newly appended helper or necessary return.
            decision.endpoints = {wait};
            selected_test::require(c.settleRearming(decision), "unchanged pair population");
            selected_test::require(c.result.work.rearmingPairVisits == uint64_t(i+1)*(i+1),
                                   "unchanged population revisited old pairs");
        }
        selected_test::require(c.result.work.rearmingDischarged == 0, "protected P payload ignored");
    }
    static SelectedPlan contextual(const Program& program)
    {
        Constructor constructor(program);
        constructor.needsContextualReplay = true;
        return constructor.run({});
    }
};
}
using namespace selected_test;
namespace {
constexpr auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
void reuseThroughRequiredOverlapReadiness()
{
    for (bool connected : {true, false}) {
        auto p = base(3, 2);
        p.cells[2].storage = o::Cell::Storage::OverlapWitness;
        p.operations = {op(P, {{0, false, true, true}}),
                        op(Q, {{0, true, false}, {1, false, true, true}}),
                        op(R, {{2, false, true}}),
                        op(P, {{0, false, true, true}, {2, true, false}})};
        if (connected) p.operations[2].accesses.push_back({1, true, false});
        p.body = seq({leaf(0), leaf(1), leaf(2), leaf(3)});
        const auto plan = accepted(p);
        const auto direct = std::count_if(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 3 && d.source == Q;
        });
        require(direct == (connected ? 0 : 1),
                "overlap readiness must carry actual reader completion to replace a direct return");
        require(bool(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3})),
                "three-engine completion support failed the independent oracle");
        if (!connected) continue;
        require(plan.decisions.size() == 3 && plan.decisions.back().source == R &&
                plan.decisions.back().stage == o::RequirementStage::Known,
                "necessary overlap provider was considered after the known reuse repair");
        // Removing the supporting middle handoff must expose the old reader;
        // a direction-level route or a future promised receipt is insufficient.
        auto broken = plan.commands;
        for (auto& word : broken) word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
            return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) &&
                   c.source == Q && c.observer == R;
        }), word.end());
        require(!o::checkCausalFrontier(p, broken).accepted &&
                !bool(oahs_oracle::graph(p, broken, {0, 1, 2, 3})),
                "removing supporting readiness retained manufactured completion");

        p.body = {o::Region::For, {p.body}, 0, true};
        const auto repeated = accepted(p);
        require(std::none_of(repeated.decisions.begin(), repeated.decisions.end(), [](const auto& d) {
            return d.consumer == 3 && d.source == Q;
        }), "repeated indirect readiness acquired a duplicate direct reader release");
        for (unsigned count : {0u, 1u, 2u, 4u}) {
            std::vector<unsigned> visits;
            for (unsigned i = 0; i < count; ++i) visits.insert(visits.end(), {0, 1, 2, 3});
            require(bool(oahs_oracle::graph(p, repeated.commands, visits)),
                    "repeated third-engine receipt lost memory or consumption knowledge");
        }
    }
}
void alternativeProviderCredit()
{
    for (unsigned variant : {0u, 1u, 2u}) {
        auto p = base(4, 4);
        p.cells[2].storage = o::Cell::Storage::OverlapWitness;
        p.operations = {op(P, {{0, false, true, true}}),
                        op(Q, {{0, true, false}, {1, false, true, true}}),
                        op(R, {{1, true, false}, {2, false, true}}),
                        op(R, {{2, false, true}}),
                        op(R, {{3, false, true}}), op(R, {{3, false, true}}),
                        op(P, {{0, false, true, true}, {2, true, false}})};
        if (variant != 1) p.operations[3].accesses.push_back({1, true, false});
        if (variant == 2) p.operations[4] = op(Q, {{0, true, false}});
        o::ObservedControl g;
        g.qualification = "alternative required provider with source-time joint credit";
        g.entry = 0; g.exit = 10;
        const std::vector<std::size_t> operations{0, 1, o::NoAnalysisId, 2, 4,
            o::NoAnalysisId, 3, 5, o::NoAnalysisId, 6, o::NoAnalysisId};
        const std::vector<std::vector<std::size_t>> edges{
            {1}, {2}, {3, 6}, {4}, {5}, {9}, {7}, {8}, {9}, {10}, {}};
        for (std::size_t i = 0; i < operations.size(); ++i) {
            g.observations.push_back({i, {}, true});
            g.sites.push_back({operations[i], i, edges[i], {}, 0});
        }
        p.observed = g;
        const auto plan = accepted(p);
        const bool direct = std::any_of(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 9 && d.source == Q;
        });
        require(direct == (variant != 0),
                "alternative provider must cover every arm and the current reader occurrence");
        if (variant == 0) {
            require(std::any_of(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
                return d.consumer == 9 && d.source == R && d.stage == o::RequirementStage::Known &&
                    d.publicationFrontier == std::vector<o::Cut>{4, 7};
            }), "Known promotion lost alternative early publication boundaries");
            auto broken = plan.commands;
            for (auto& word : broken) word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
                return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) &&
                    c.source == Q && c.observer == R;
            }), word.end());
            require(!o::checkCausalFrontier(p, broken).accepted,
                    "alternative source manufactured credit without supporting readiness");
        }
        for (const auto& path : {std::vector<o::Cut>{0,1,2,3,4,5,9,10},
                                 std::vector<o::Cut>{0,1,2,6,7,8,9,10}}) {
            auto flat = p; flat.observed.reset(); flat.body = {}; flat.operations.clear();
            o::Commands words; std::vector<o::Command> pending;
            for (auto cut : path) {
                pending.insert(pending.end(), plan.commands[cut].begin(), plan.commands[cut].end());
                const auto operation = g.sites[cut].operation;
                if (operation == o::NoAnalysisId) continue;
                flat.operations.push_back(p.operations[operation]);
                words.push_back(std::move(pending)); pending.clear();
            }
            words.push_back(std::move(pending));
            const std::vector<std::pair<unsigned, unsigned>> forbidden = variant == 0 ?
                std::vector<std::pair<unsigned, unsigned>>{{3, 4}} :
                std::vector<std::pair<unsigned, unsigned>>{};
            require(bool(oahs_oracle::graph(flat, words, {0,1,2,3,4}, forbidden)),
                    "alternative provider lost memory/rearming or imported the later unrelated R work");
        }
    }
}

void entryProviderCredit()
{
    for (bool connected : {false, true}) {
        auto p = base(4, 4);
        p.cells[2].storage = o::Cell::Storage::OverlapWitness;
        p.operations = {op(P, {{0, false, true, true}}),
                        op(Q, {{0, true, false}, {1, false, true, true}}),
                        op(R, {{2, false, true}}),
                        op(P, {{0, false, true, true}, {2, true, false}}),
                        op(R, {{3, false, true}})};
        if (connected) p.operations[2].accesses.push_back({1, true, false});
        o::ObservedControl g;
        g.qualification = "nonempty consumer region with invariant extra provider credit";
        g.entry = 0; g.exit = 9;
        const std::vector<std::size_t> operations{0, 1, 2, o::NoAnalysisId,
            4, o::NoAnalysisId, o::NoAnalysisId, 3, o::NoAnalysisId, o::NoAnalysisId};
        const std::vector<std::vector<std::size_t>> edges{{1}, {2}, {3}, {4}, {5}, {6}, {7,9}, {8}, {6}, {}};
        for (std::size_t i = 0; i < operations.size(); ++i) {
            g.observations.push_back({i, {}, true});
            g.sites.push_back({operations[i], i, edges[i], {}, 0});
        }
        g.sites[8].backedgeOwners = {5};
        g.loops.push_back({5, 5, 9, {6,7,8}, 7, true});
        p.observed = g;
        const auto plan = accepted(p);
        const bool direct = std::any_of(plan.decisions.begin(), plan.decisions.end(), [](const auto& d) {
            return d.consumer == 7 && d.source == Q;
        });
        require(direct == !connected, "entry provider failed to distinguish actual extra credit");
        require(plan.work.loopEntryTransfers != 0, "fixture did not select entry acquisition");
        if (connected) {
            auto broken = plan.commands;
            for (auto& word : broken) word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
                return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) &&
                    c.source == Q && c.observer == R;
            }), word.end());
            require(!o::checkCausalFrontier(p, broken).accepted,
                    "entry provider kept credit after deleting its supporting transfer");
        }
    }
}

void keepKnownPrefixSeparateFromOverlap()
{
    auto p = base(2, 2);
    p.cells[1].storage = o::Cell::Storage::OverlapWitness;
    p.operations = {op(P, {{0, false, true, true}}),
                    op(P, {{1, false, true}}),
                    op(Q, {{0, true, false}, {1, true, false}})};
    p.body = seq({leaf(0), leaf(1), leaf(2)});
    o::StorageFrontierAnalysis storage(p);
    bool known = false, overlap = false;
    for (const auto& relation : storage.relationshipsAt(2)) {
        const auto reasons = storage.describeRequirement(relation).reasons;
        known |= relation.cell == 0 && bool(reasons & o::KnownReadiness);
        overlap |= relation.cell == 1 && !(reasons & (o::KnownReadiness | o::KnownReuse));
    }
    require(known && overlap, "fixture did not exercise both priority stages");
    auto plan = accepted(p);
    require(!plan.decisions.empty() && plan.decisions[0].stage == o::RequirementStage::Known &&
            plan.decisions[0].required.size() == 1,
            "known readiness was widened by an additional-overlap demand");
    require(plan.decisions[0].publication == 1,
            "lookahead moved the known source prefix past the later load");
}
void sharedLifecycleView()
{
    auto p = base(1, 3);
    const auto R = o::Pipe::V;
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(R, {{0, true, false}}), op(P, {{0, false, true, true}})};
    o::selected::Control control(p);
    o::StorageFrontierAnalysis storage(p);
    o::selected::RequirementFrontiers facts(p, control, storage);
    require(facts.complete(), facts.reason());
    const auto& reader = facts.use(1, 0);
    require(reader.roles == 1 && reader.pipe == Q && reader.release == 2 &&
            reader.returns == std::vector<o::Cut>{3}, "reader lost physical return deadline");
    const auto& physical = facts.lifetime(3, 0);
    std::set<o::Cut> readers;
    for (const auto& origin : physical.previousReaders) {
        readers.insert(origin.site);
    }
    require(readers == std::set<o::Cut>{1, 2}, "lifecycle merged independent reader engines");
    require(&physical == &facts.lifetime(3, 0), "lifecycle summary not shared");
    require(!facts.lifetime(100, 0).reachable && !facts.use(1, 4).roles, "invalid lifecycle query accepted");
    require(facts.recurringRelease(1, 0) == o::NoAnalysisId, "unqualified recurrence invented");
    const auto plan = accepted(p);
    bool readiness = false, release = false;
    for (const auto& decision : plan.decisions) {
        for (const auto& demand : decision.lifecycles) {
            if (demand.requirement.source.site == 0 && demand.requirement.target.site == 1) {
                readiness = demand.release == 1 && demand.deadline == 1 &&
                    demand.returnDeadlines == std::vector<o::Cut>{3};
            }
            if (demand.requirement.kind == o::StorageRelationship::WAR) {
                release |= demand.deadline == 3;
            }
        }
    }
    require(readiness && release, "binding lost its first-pass lifecycle evidence");
    require(bool(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3}, {})), "lifecycle binding order invalid");
}
void keepDifferentDeadlines()
{
    auto p = base(2, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    p.body = seq({leaf(0), leaf(1), leaf(2), leaf(3)});
    o::selected::Control control(p);
    o::StorageFrontierAnalysis storage(p);
    o::selected::RequirementFrontiers frontiers(p, control, storage);
    require(frontiers.complete() && frontiers.size() == 2,
            "first pass did not retain the two readiness requirements");
    require(frontiers.sourceBoundaries() == 2,
            "acyclic frontier qualification was not recorded");
    require(frontiers.occurrenceCounts()[unsigned(o::selected::RequirementOccurrence::Acyclic)] == 2,
            "acyclic occurrence class was not recorded");
    const auto& first = frontiers.at(2);
    const auto& second = frontiers.at(3);
    require(first.size() == 1 && first.front().publication == 1 && first.front().deadline == 2 &&
            first.front().lifecycleRelease == 1,
            "first source boundary/deadline pair changed");
    require(second.size() == 1 && second.front().publication == 2 && second.front().deadline == 3 &&
            second.front().lifecycleRelease == 2,
            "second source boundary/deadline pair changed");
    require(first.front().source == second.front().source &&
            first.front().observer == second.front().observer,
            "fixture does not exercise one pipeline pair with distinct frontiers");
    const auto plan = accepted(p);
    require(plan.decisions.size() == 2, "different consumers were merged");
    require(plan.decisions[0].publication == 1 && plan.decisions[0].consumer == 2,
            "first consumer now waits for the second load");
    require(plan.decisions[1].consumer == 3, "second deadline lost");
    require(bool(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3}, {{1, 2}})),
            "first consumer unnecessarily waits for completion of the second load");
}
o::Program joinedTerminal()
{
    auto p = base(1, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{0, false, true, true}}),
                    op(Q, {{0, true, false}})};
    p.body = seq({{o::Region::Choice, {leaf(0), leaf(1)}}, leaf(2)});
    return p;
}
void alternativeEarlySources()
{
    auto p = base(3, 2);
    p.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}),
                    op(P, {{0, false, true}}), op(P, {{2, false, true}}),
                    op(Q, {{0, true, false}})};
    o::ObservedControl graph;
    graph.qualification = "test-original-branch-cuts";
    graph.entry = 0;
    graph.exit = 8;
    const std::vector<std::size_t> operations{
        o::NoAnalysisId, 0, 1, o::NoAnalysisId, 2, 3, o::NoAnalysisId, 4, o::NoAnalysisId};
    const std::vector<std::vector<std::size_t>> successors{{1, 4}, {2}, {3}, {7}, {5}, {6}, {7}, {8}, {}};
    for (std::size_t site = 0; site < operations.size(); ++site) {
        graph.observations.push_back({site, {}, true});
        graph.sites.push_back({operations[site], site, successors[site], {}, 0});
    }
    p.observed = graph;
    const auto plan = accepted(p);
    require(plan.decisions.size() == 1 &&
            plan.decisions[0].publicationFrontier == std::vector<o::Cut>{2, 5},
            "missing alternative publication frontier immediately after the required writers");
    require(count(plan, o::Command::Publish) == 2 && count(plan, o::Command::Acquire) == 1,
            "alternative sources need one SET per arm, one shared WAIT");
    require(plan.commands[7].size() == 1 && plan.commands[7][0].kind == o::Command::Acquire,
            "late whole-prefix publication remains at the join");
    const auto& before = *plan.certificate.cuts[7].beforeIssue.facts();
    for (unsigned cell : {1u, 2u}) {
        const auto access = (std::size_t(cell) * o::PipeCount + unsigned(P)) * 2 + 1;
        const auto* history = before.history.find(access);
        require(history && !o::frontierContains(*history, unsigned(Q)),
                "consumer unnecessarily acquires unrelated branch-load completion");
    }
    // Absence from a must-state is not alone a proof that an unwanted edge is
    // absent concretely. Check both original branch paths using the unchanged
    // independent full-history graph oracle as well. Fold control-only words
    // into the next real payload; do not introduce dummy payload operations.
    for (const auto& path : {std::vector<o::Cut>{0, 1, 2, 3, 7, 8},
                             std::vector<o::Cut>{0, 4, 5, 6, 7, 8}}) {
        auto flat = p;
        flat.observed.reset();
        flat.body = {};
        flat.operations.clear();
        o::Commands words;
        std::vector<o::Command> pending;
        for (auto site : path) {
            pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
            const auto operation = p.observed->sites[site].operation;
            if (operation == o::NoAnalysisId) continue;
            flat.operations.push_back(p.operations[operation]);
            words.push_back(std::move(pending));
            pending.clear();
        }
        words.push_back(std::move(pending));
        require(flat.operations.size() == 3, "unexpected branch-path payload population");
        require(bool(oahs_oracle::graph(flat, words, {0, 1, 2}, {{1, 2}})),
                "branch protocol imposes unrelated-load completion before consumer issue");
    }
    // A backward source search cannot erase an earlier source occurrence on a
    // bypass path. Add a path which bypasses both writers: its missing source
    // participation must make the early-frontier candidate unavailable.
    auto bypass = p;
    bypass.observed->sites[0].successors.push_back(7);
    const auto conservative = accepted(bypass);
    require(std::all_of(conservative.decisions.begin(), conservative.decisions.end(),
        [](const auto& decision) { return decision.publicationFrontier.empty(); }),
        "branch bypass was silently matched to a nonparticipating publisher");
}
void noUnusedTerminalReturn()
{
    auto p = joinedTerminal();
    p.target.keys[unsigned(Q)][unsigned(P)].clear();
    const auto plan = accepted(p);
    require(plan.decisions.size() == 1 && plan.decisions[0].commonCut,
            "fixture did not select a common-cut transfer");
    require(count(plan, o::Command::Publish) == 1 && count(plan, o::Command::Acquire) == 1,
            "terminal common cut still requires a reverse key");
    require(plan.work.acknowledgments == 0, "unused terminal consumption was acknowledged");
}
void keepFuturePayloadReturn()
{
    auto p = joinedTerminal();
    p.operations.push_back(op(R, {{0, true, false}}));
    p.body.children.push_back(leaf(3));
    const auto plan = accepted(p);
    require(plan.work.acknowledgments != 0,
            "terminal optimization crossed a possible future payload");
}
void keepFutureWordReturn()
{
    const auto p = joinedTerminal();
    o::Commands fixed(o::commandCutCount(p));
    fixed[o::invocationExitCut(p)] = {
        {o::Command::Publish, o::Pipe::MTE1, R, 0},
        {o::Command::Acquire, o::Pipe::MTE1, R, 0}};
    const auto plan = accepted(p, fixed);
    require(plan.work.acknowledgments != 0,
            "later fixed event words were ignored by terminal optimization");
}
void keepLoopReturn()
{
    auto p = base(1, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
    const auto plan = accepted(p);
    require(plan.work.acknowledgments + plan.work.rearmingDischarged != 0,
            "last body issue was mistaken for invocation exit");
    auto missingReturn = plan.commands;
    for (auto& word : missingReturn) word.erase(std::remove_if(word.begin(), word.end(), [](const auto& c) {
        return (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) && c.source == Q && c.observer == P;
    }), word.end());
    require(!o::checkCausalFrontier(p, missingReturn).accepted,
            "loop republication lost its actual consumption path");
    o::selected::Control control(p);
    for (std::size_t site = 0; site < control.graph.sites.size(); ++site) {
        if (control.graph.operations[site] == 1)
            require(control.lookahead.mayIssueAfter(site), "loop lookahead erased backedge");
    }
}
o::Program invariantLoopProgram(unsigned keys = 4)
{
    auto p = base(3, keys);
    p.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}),
                    op(Q, {{0, true, false}})};
    o::ObservedControl q;
    q.qualification = "test-original-nonempty-loop-entry";
    q.entry = 0; q.exit = 7;
    const std::vector<std::size_t> operations{0, o::NoAnalysisId, 1, o::NoAnalysisId,
        o::NoAnalysisId, 2, o::NoAnalysisId, o::NoAnalysisId};
    const std::vector<std::vector<std::size_t>> edges{{1}, {2}, {3}, {4}, {5, 7}, {6}, {4}, {}};
    for (std::size_t i = 0; i < operations.size(); ++i) {
        q.observations.push_back({i, {}, true});
        q.sites.push_back({operations[i], i, edges[i], {}, 0});
    }
    q.sites[6].backedgeOwners = {3};
    q.loops.push_back({3, 3, 7, {4, 5, 6}, 5, true});
    p.observed = q;
    return p;
}
void invariantLoopEntry()
{
    auto p = invariantLoopProgram();
    const o::selected::Control original(p);
    require(original.loopEntries.size() == 1 &&
            original.loopEntries.front().firstConsumer[unsigned(Q)] == 5,
            "first pass lost the unique observer deadline");
    const auto plan = accepted(p);
    require(plan.work.loopEntryTransfers == 1, "invariant readiness not acquired at loop entry");
    require(plan.commands[1].size() == 1 && plan.commands[1][0].kind == o::Command::Publish,
            "loop-entry readiness includes unrelated source work");
    require(plan.commands[3].size() == 1 && plan.commands[3][0].kind == o::Command::Acquire,
            "invariant readiness must be acquired once per entry");
    require(plan.commands[5].empty(), "loop body repeats invariant acquisition");
    // Fold control-only entry words into the first body visit for a concrete
    // three-visit trace, retaining the actual command order and payload sites.
    auto flat = p;
    flat.observed.reset(); flat.operations.clear(); flat.body = {};
    o::Commands words;
    std::vector<o::Command> pending;
    for (auto site : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 4u, 5u, 6u, 4u, 5u, 6u, 4u, 7u}) {
        pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
        const auto operation = p.observed->sites[site].operation;
        if (operation == o::NoAnalysisId) continue;
        flat.operations.push_back(p.operations[operation]);
        words.push_back(std::move(pending)); pending.clear();
    }
    words.push_back(std::move(pending));
    require(bool(oahs_oracle::graph(flat, words, {0, 1, 2, 3, 4}, {{1, 2}, {1, 3}, {1, 4}})),
            "unrelated load orders a loop consumer");
    auto earlier = p;
    earlier.operations.push_back(op(Q, {}));
    auto& entry = *earlier.observed;
    entry.observations.push_back({8, {}, true});
    entry.sites.push_back({3, 8, {5}, {}, 0});
    entry.sites[4].successors[0] = 8;
    entry.loops.front().bodyEntry = 8;
    entry.loops.front().sites.push_back(8);
    require(accepted(earlier).work.loopEntryTransfers == 0,
            "entry acquisition unnecessarily gates an earlier observer payload");
    p.observed->loops.front().atLeastOnce = false;
    require(accepted(p).work.loopEntryTransfers == 0, "unknown/zero-trip entry was acquired unconditionally");
}
void reuseOneShotEntryKey()
{
    auto p = invariantLoopProgram(1);
    p.operations.push_back(op(P, {{2, false, true}}));
    p.operations.push_back(op(Q, {{2, true, false}}));
    auto& q = *p.observed;
    q.exit = 10;
    q.sites[7].successors = {8};
    for (o::Cut i = 8; i <= 10; ++i) {
        q.observations.push_back({i, {}, true});
        q.sites.push_back({i < 10 ? i - 5 : o::NoAnalysisId, i,
            i < 10 ? std::vector<o::Cut>{i + 1} : std::vector<o::Cut>{}, {}, 0});
    }
    o::Commands fixed(o::commandCutCount(p));
    fixed[7] = {{o::Command::Publish, Q, P, 0}, {o::Command::Acquire, Q, P, 0}};
    const auto plan = accepted(p, fixed);
    require(plan.work.loopEntryTransfers == 1 && plan.commands[1].size() == 1 &&
            plan.commands[1][0].kind == o::Command::Publish && plan.commands[3].size() == 1 &&
            plan.commands[3][0].kind == o::Command::Acquire,
            "one-shot key reuse lost the useful early entry placement");
    require(plan.work.acknowledgments == 0,
            "valid fixed return was ignored when proving one-shot key reuse");
    unsigned publications = 0;
    for (const auto& endpoint : plan.ledger)
        if (endpoint.command.kind == o::Command::Publish && endpoint.command.source == P &&
            endpoint.command.observer == Q) {
            require(endpoint.command.key == 0, "single-key pool was enlarged");
            ++publications;
        }
    require(publications == 2, "consumed one-shot key was permanently reserved");
    require(std::any_of(plan.updates.begin(), plan.updates.end(),
                       [](const auto& update) { return update.contextual; }),
            "ending a compiler reservation disabled contextual replay");
    auto missingReturn = plan.commands;
    missingReturn[7].clear();
    require(!o::checkCausalFrontier(p, missingReturn).accepted,
            "ending a reservation manufactured consumption knowledge");
}

void entryWaitMustNotCrossPublication()
{
    auto p = invariantLoopProgram();
    p.operations.push_back(op(R, {}));
    auto& q = *p.observed;
    q.observations.push_back({8, {}, true});
    q.sites.push_back({3, 8, {5}, {}, 0});
    q.sites[4].successors[0] = 8;
    q.loops.front().bodyEntry = 8;
    q.loops.front().sites.push_back(8);
    const std::vector<o::Command> relay{
        {o::Command::Publish, Q, R, 0}, {o::Command::Acquire, Q, R, 0},
        {o::Command::Publish, R, Q, 0}, {o::Command::Acquire, R, Q, 0}};
    o::Commands fixed(o::commandCutCount(p));
    fixed[8] = relay;
    const auto plan = accepted(p, fixed);
    require(plan.work.loopEntryTransfers == 0,
            "entry acquisition crossed an observer publication to another engine");
    auto flat = p;
    flat.observed.reset(); flat.operations.clear(); flat.body = {};
    o::Commands words;
    std::vector<o::Command> pending;
    for (auto site : {0u, 1u, 2u, 3u, 4u, 8u, 5u, 6u,
                     4u, 8u, 5u, 6u, 4u, 8u, 5u, 6u, 4u, 7u}) {
        pending.insert(pending.end(), plan.commands[site].begin(), plan.commands[site].end());
        const auto operation = q.sites[site].operation;
        if (operation == o::NoAnalysisId) continue;
        flat.operations.push_back(p.operations[operation]);
        words.push_back(std::move(pending)); pending.clear();
    }
    words.push_back(std::move(pending));
    require(bool(oahs_oracle::graph(flat, words, {0, 1, 2, 3, 4, 5, 6, 7}, {{0, 2}})),
            "entry placement ordered A completion before independent R work");

    // A fixed publication in the deadline's word also precedes the original
    // acquisition; checking only strictly earlier sites is insufficient.
    fixed[8].clear(); fixed[5] = relay;
    require(accepted(p, fixed).work.loopEntryTransfers == 0,
            "entry placement crossed the deadline's existing publication");

    // Keep useful placements when these words are not crossed: the new WAIT
    // is appended after entry commands, and body-exit commands follow Q's use.
    fixed[5].clear(); fixed[3] = relay;
    require(accepted(p, fixed).work.loopEntryTransfers == 1,
            "publication before the entry WAIT disabled useful early placement");
    fixed[3].clear(); fixed[6] = relay;
    require(accepted(p, fixed).work.loopEntryTransfers == 1,
            "publication after the consumer disabled useful early placement");
}
// Incoming reader completion is needed at the first conflicting write, not
// before the observer exports an unrelated prefix. Supply an original
// first-visit witness to distinguish occurrence precision from new causal state.
void firstConflictingWriterAfterPublication()
{
    auto p = invariantLoopProgram();
    p.operations[0].accesses = {{0, true, false}};
    p.operations[2].accesses = {{0, false, true}};
    p.operations.push_back(op(R, {}));
    auto& q = *p.observed;
    q.observations.push_back({8, {}, true});
    q.sites.push_back({3, 8, {5}, {}, 0});
    q.sites[4].successors[0] = 8;
    q.loops.front().bodyEntry = 8;
    q.loops.front().sites.push_back(8);
    const std::vector<o::Command> relay{
        {o::Command::Publish, Q, R, 0}, {o::Command::Acquire, Q, R, 0},
        {o::Command::Publish, R, Q, 0}, {o::Command::Acquire, R, Q, 0}};
    o::Commands fixed(o::commandCutCount(p));
    fixed[8] = relay;
    const auto baseline = accepted(p, fixed);
    require(baseline.work.loopEntryTransfers == 0,
            "external-reader receipt crossed an outward publication");

    auto refined = p;
    auto& r = *refined.observed;
    // Original first/later participation: one prefix, then zero or more
    // backedge visits. The outward-publication word stays shared.
    r.sites.push_back({3, 8, {10}, {}, 0});
    r.observations.push_back({10, {{o::ObservationAtom::LoopHasPrevious, 3, 1, 0}}, true});
    r.sites.push_back({2, r.observations.size() - 1, {6}, {}, 0});
    r.sites[3].successors = {9};
    r.loops.front().bodyEntry = 9;
    r.loops.front().sites.insert(r.loops.front().sites.end(), {9, 10});
    fixed.resize(o::commandCutCount(refined));
    fixed[9] = relay;
    const auto selected = accepted(refined, fixed);
    require(std::any_of(selected.commands[10].begin(), selected.commands[10].end(), [&](const auto& c) {
        return c.kind == o::Command::Acquire && c.source == P && c.observer == Q;
    }), "first conflicting writer did not receive incoming reader completion");
    require(std::none_of(selected.commands[5].begin(), selected.commands[5].end(), [&](const auto& c) {
        return c.kind == o::Command::Acquire && c.source == P && c.observer == Q;
    }), "later writes reacquire the same invariant external reader");

    auto evaluate = [&](const o::Program& program, const o::Commands& commands,
                        const std::vector<o::Cut>& path, oahs_oracle::PayloadOrder& order) {
        auto flat = program; flat.observed.reset(); flat.body = {}; flat.operations.clear();
        o::Commands words; std::vector<o::Command> pending;
        for (auto site : path) {
            pending.insert(pending.end(), commands[site].begin(), commands[site].end());
            const auto operation = program.observed->sites[site].operation;
            if (operation == o::NoAnalysisId) continue;
            flat.operations.push_back(program.operations[operation]);
            words.push_back(std::move(pending)); pending.clear();
        }
        words.push_back(std::move(pending));
        std::vector<unsigned> visits(flat.operations.size());
        std::iota(visits.begin(), visits.end(), 0);
        return bool(oahs_oracle::graph(flat, words, visits, {{0, 2}}, nullptr, nullptr, &order));
    };
    for (unsigned n : {1u, 2u, 5u}) {
        std::vector<o::Cut> before{0, 1, 2, 3}, after{0, 1, 2, 3, 9, 10, 6};
        for (unsigned i = 0; i < n; ++i) before.insert(before.end(), {4, 8, 5, 6});
        for (unsigned i = 1; i < n; ++i) after.insert(after.end(), {4, 8, 5, 6});
        before.insert(before.end(), {4, 7}); after.insert(after.end(), {4, 7});
        oahs_oracle::PayloadOrder a, b;
        require(evaluate(p, baseline.commands, before, a) && evaluate(refined, selected.commands, after, b),
                "first-write protocol lost safety/rearming or gated the outward consumer");
        require(std::includes(a.begin(), a.end(), b.begin(), b.end()),
                "first-write receipt added payload ordering");
    }
    auto missing = selected.commands;
    auto& word = missing[10];
    word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& c) {
        return c.kind == o::Command::Acquire && c.source == P && c.observer == Q;
    }), word.end());
    require(!o::checkCausalFrontier(refined, missing).accepted,
            "first-write completion was assumed without its actual receipt");
    auto broad = selected.commands;
    const auto receipt = std::find_if(broad[10].begin(), broad[10].end(), [&](const auto& c) {
        return c.kind == o::Command::Acquire && c.source == P && c.observer == Q;
    });
    broad[3].push_back(*receipt);
    broad[10].erase(receipt);
    require(o::checkCausalFrontier(refined, broad).accepted,
            "broad first-write witness should remain safe");
    oahs_oracle::PayloadOrder broadOrder;
    require(!evaluate(refined, broad, {0, 1, 2, 3, 9, 10, 6, 4, 7}, broadOrder),
            "broad entry receipt did not expose the forbidden outward ordering");
    std::cout << "first-conflicting-write paths=3 repeated-receipt=0 broad-entry-safe-but-orders-export=1\n";
}

// A region with alternative first writers, re-entered after a foreign reader.
// The source does no work inside the child; no kernel/opcode recognition is used.
o::Program enclosingChoice()
{
    auto p = base(1, 3);
    p.operations = {op(Q, {{0, false, true}}), op(Q, {{0, false, true}}),
                    op(P, {{0, true, false}})};
    o::ObservedControl q;
    q.entry = 0; q.exit = 8; q.qualification = "original nested choice/re-entry";
    for (o::Cut i = 0; i <= 8; ++i) {
        q.observations.push_back({i, {}, true});
        q.sites.push_back({o::NoAnalysisId, i, {}, {}, 0});
    }
    q.sites[0].successors = {1, 8};
    q.sites[1].successors = {2};
    q.sites[2].successors = {3, 4};
    q.sites[3].operation = 0; q.sites[3].successors = {5};
    q.sites[4].operation = 1; q.sites[4].successors = {5};
    q.sites[5].successors = {2, 6}; q.sites[5].backedgeOwners = {1, o::NoAnalysisId};
    q.sites[6].successors = {7};
    q.sites[7].operation = 2; q.sites[7].successors = {1, 8};
    q.sites[7].backedgeOwners = {0, o::NoAnalysisId};
    q.loops.push_back({1, 1, 6, {2, 3, 4, 5}, 2, true});
    p.observed = std::move(q);
    return p;
}
void changedRepublicationDeadline()
{
    auto p = base(1, 3);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(R, {{0, true, true}}), op(Q, {{0, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0),
        {o::Region::Choice, {seq({leaf(1)}), seq({leaf(2)})}}, leaf(3)})}, 0, true};
    const auto plan = o::selected::ReplayTestAccess::contextual(p);
    require(plan.success, "new publication deadline: " + plan.reason);
    require(plan.work.rearmingRestored != 0, "changed-deadline fixture never restored a helper");
    require(o::checkCausalFrontier(p, plan.commands).accepted, "earlier deadline lost its rearming path");
    for (const auto& visits : oahs_oracle::traces(p, 3))
        require(bool(oahs_oracle::graph(p, plan.commands, visits)), "changed deadline failed independent protocol check");
}
// Supplied-plan diagnostic for the KDA acknowledgment -> MAT-release path.
// This is not a claim that native construction already selects the early word.
void releaseBeforeAcknowledgment()
{
    const auto dma = o::Pipe::MTE2, reader = o::Pipe::MTE1, compute = o::Pipe::M;
    for (unsigned trips : {2u, 3u, 4u}) {
        auto p = base(trips + 1, 1);
        p.operations.push_back(op(dma, {{0, false, true}}));
        for (unsigned i = 0; i < trips; ++i) {
            p.operations.push_back(op(reader, {{0, true, false}, {i + 1, false, true}}));
            p.operations.push_back(op(compute, {{i + 1, true, false}}));
        }
        p.operations.push_back(op(dma, {{0, false, true}}));
        o::Commands early(p.operations.size() + 1);
        early[1] = {{o::Command::Publish, dma, reader, 0},
                    {o::Command::Acquire, dma, reader, 0}};
        for (unsigned i = 0; i < trips; ++i) {
            early[2 + 2*i] = {{o::Command::Publish, reader, compute, 0},
                             {o::Command::Acquire, reader, compute, 0},
                             {o::Command::Publish, compute, reader, 0},
                             {o::Command::Acquire, compute, reader, 0}};
        }
        const unsigned finalReadGap = 2 * trips;
        early[finalReadGap].insert(early[finalReadGap].begin(),
                                  {o::Command::Publish, reader, dma, 0});
        early[finalReadGap + 1] = {{o::Command::Acquire, reader, dma, 0}};
        auto broad = early;
        broad[finalReadGap].erase(broad[finalReadGap].begin());
        broad[finalReadGap].push_back({o::Command::Publish, reader, dma, 0});
        std::vector<unsigned> visits(p.operations.size());
        std::iota(visits.begin(), visits.end(), 0);
        oahs_oracle::PayloadOrder earlyOrder, broadOrder;
        require(o::checkCausalFrontier(p, early).accepted, "early release loses causal support");
        require(o::checkCausalFrontier(p, broad).accepted, "broad release control invalid");
        require(bool(oahs_oracle::graph(p, early, visits,
                    {{2 * trips - 2, 2 * trips + 1}}, nullptr, nullptr, &earlyOrder)),
                "early release imports penultimate compute or loses rearming");
        require(bool(oahs_oracle::graph(p, broad, visits, {}, nullptr, nullptr, &broadOrder)),
                "broad release oracle failure");
        require(std::includes(broadOrder.begin(), broadOrder.end(), earlyOrder.begin(), earlyOrder.end()) &&
                    broadOrder.size() > earlyOrder.size(), "release gap does not strictly reduce order");
        auto missing = early;
        missing[2].erase(missing[2].begin() + 2, missing[2].end());
        require(!oahs_oracle::graph(p, missing, visits).rearm,
                "missing acknowledgment did not expose actual key-reuse obligation");
        require(!o::checkCausalFrontier(p, missing).accepted,
                "causal checker accepted missing rearming support");
    }
}
void enclosingAcquisitionAndRearming()
{
    const auto p = enclosingChoice();
    const auto plan = accepted(p);
    require(plan.work.loopEntryTransfers == 1, "alternative first consumers lost enclosing acquisition");
    require(plan.commands[1].size() == 2 && plan.commands[1][0].kind == o::Command::Publish &&
            plan.commands[1][1].kind == o::Command::Acquire,
            "enclosing completion must be acquired once, without an immediate return");
    require(plan.work.rearmingDischarged == 2 && plan.work.acknowledgments == 0,
            "necessary entry/result transfers did not discharge each other's rearming");
    for (auto cut : {3u, 4u}) for (const auto& c : plan.commands[cut])
        require(c.kind != o::Command::Publish && c.kind != o::Command::Acquire,
                "invariant source completion was reacquired inside the region");
    for (auto cut : {1u, 7u}) {
        auto missing = plan.commands;
        missing[cut].clear();
        require(!o::checkCausalFrontier(p, missing).accepted,
                "necessary return deletion manufactured consumption knowledge");
    }
    o::Commands fixed(o::commandCutCount(p));
    fixed[6] = {{o::Command::Publish, P, R, 0}, {o::Command::Acquire, P, R, 0},
                {o::Command::Publish, R, P, 0}, {o::Command::Acquire, R, P, 0}};
    const auto exporting = accepted(p, fixed);
    require(exporting.work.loopEntryTransfers == 1 && exporting.work.rearmingComposed != 0 &&
            exporting.work.acknowledgments == 0,
            "sealed reciprocal transfers did not discharge the conservative helper");
    auto unrelated = p;
    unrelated.operations[1].accesses.clear();
    require(accepted(unrelated).work.loopEntryTransfers == 0,
            "one branch's deadline was broadened to an unrelated first consumer");
    auto optional = p;
    optional.observed->loops.front().atLeastOnce = false;
    require(accepted(optional).work.loopEntryTransfers == 0,
            "unqualified nonempty region received an enclosing acquisition");
}
} // namespace
int main()
{
    for (auto n : {32u, 64u, 128u}) o::selected::ReplayTestAccess::pairEnumeration(n);
    keepKnownPrefixSeparateFromOverlap();
    reuseThroughRequiredOverlapReadiness();
    sharedLifecycleView();
    keepDifferentDeadlines();
    alternativeEarlySources();
    alternativeProviderCredit();
    entryProviderCredit();
    noUnusedTerminalReturn();
    keepFuturePayloadReturn();
    keepFutureWordReturn();
    keepLoopReturn();
    changedRepublicationDeadline();
    enclosingAcquisitionAndRearming();
    releaseBeforeAcknowledgment();
    invariantLoopEntry();
    reuseOneShotEntryKey();
    entryWaitMustNotCrossPublication();
    firstConflictingWriterAfterPublication();
    o::selected::ReplayTestAccess::finalReadSource();
    o::selected::ReplayTestAccess::crossControlAcknowledgment();
    o::selected::ReplayTestAccess::sharedWordNewConsumption();
    o::selected::ReplayTestAccess::sharedWordNewConsumption(true);
    o::selected::ReplayTestAccess::sharedWordNewConsumption(true, false, true);
    o::selected::ReplayTestAccess::sharedWordNewConsumption(true, false, false, true);
    o::selected::ReplayTestAccess::sharedWordNewConsumption(false, true);
    std::cout << "selected lookahead, deadline and terminal-return tests passed\n";
}
