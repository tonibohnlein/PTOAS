// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "ObservedFixtures.h"
#include "GraphOracle.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
using namespace selected_test;
namespace s = o::selected;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void normalization(unsigned mutation)
    {
        auto p = base(2);
        p.operations = {op(Pipe::MTE2, {{0,false,true,true}, {1,false,true,true}}),
                        op(Pipe::V, {{0,true,false}, {1,true,false}})};
        if (mutation == 1) { p.operations[0].accesses[1].definiteWrite = false; }
        if (mutation == 2) { p.operations.push_back(op(Pipe::MTE1, {{1,true,false}})); }
        if (mutation == 3) { p.operations[0].accesses[1].nativeAccumulatorClass = 0; p.nativeAccumulatorClasses = 1; }
        Constructor c(p); c.current = 1;
        const std::vector<FrontierRequirement> due{{0,Pipe::MTE2,true,1,true,false},
                                                  {1,Pipe::MTE2,true,1,true,false}};
        const auto normalized = c.normalizeDue(due);
        require(normalized.size() == (mutation ? 2u : 1u), "normalization lost physical interpretation");
        const auto visits = c.result.work.normalizationIncidences;
        c.normalizeDue(due);
        require(c.result.work.normalizationIncidences == visits, "normalization repeated original incidence scan");
        if (!mutation) {
            require(c.normalizedCoverage(normalized,due,{accessClass(due[0])}).empty(),
                    "partial witness coverage counted as a complete normalized obligation");
        }
    }
    static void alternativeKeys(unsigned mutation)
    {
        const auto P=Pipe::MTE2, Q=Pipe::V;
        auto p=base(2,1);
        p.operations={op(P,{{0,false,true}}),op(Q,{{0,true,false}}),
                      op(P,{{1,false,true}}),op(Q,{{1,true,false}})};
        p.body={Region::Choice,{seq({leaf(0),leaf(1)}),seq({leaf(2),leaf(3)})}};
        if (mutation==1) { p.body=seq({leaf(0),leaf(1),leaf(2),leaf(3)}); }
        if (mutation==2) { p.body={Region::For,{p.body}}; }
        Constructor c(p); c.current=3;
        c.ledger.append(1,{Command::Publish,P,Q,0},EndpointPurpose::Completion);
        c.ledger.append(1,{Command::Acquire,P,Q,0},EndpointPurpose::Completion);
        // Re-entry's incomplete old protocol can itself fail replay; that is a
        // valid rejection and never licenses a local occupancy shortcut.
        const bool replay=c.contextualReplay();
        if (!replay) { require(mutation!=0,"alternative setup failed"); return; }
        const std::vector<FrontierRequirement> due{{1,P,true,3,true,false}};
        const auto facts=c.sourceGapFacts({3,NoAnalysisId,NoAnalysisId},P,Q,due);
        Id key=NoAnalysisId;
        for (Id i=0;i<c.frontier.keys().size();++i) {
            const auto& k=c.frontier.keys()[i];
            if(k.source==P&&k.observer==Q) { key=i;break; }
        }
        require(key!=NoAnalysisId,"alternative key absent");
        require(c.sourceGapKey(facts,key)==(mutation==0),"control alternatives were confused with key rearming");
        if (!mutation) {
            c.deferredByKey[key].insert(0);
            require(!c.sourceGapKey(facts,key),"dormant helper ownership was omitted from normal binding");
        }
    }
    static void tailOnlyCoverage()
    {
        const auto P=Pipe::MTE2,Q=Pipe::V,R=Pipe::MTE1;
        auto p=base(2);
        p.operations={op(Q,{{1,false,true}}),op(P,{{0,false,true}}),
                      op(P,{{1,true,false}}),op(R,{{0,true,false},{1,true,false}})};
        Constructor c(p); c.current=3;
        c.ledger.append(1,{Command::Publish,Q,P,0},EndpointPurpose::Completion);
        c.ledger.append(2,{Command::Acquire,Q,P,0},EndpointPurpose::Completion);
        require(c.contextualReplay(),c.result.reason);
        Group group; group.source=P;group.publication=2;
        group.requirements={{0,P,true,3,true,false},{1,Q,true,3,true,false}};
        const auto before=c.ledger.version();
        const auto normal=c.normalOrdinary(group,group.requirements,c.normalizeDue(group.requirements));
        require(!normal && c.ledger.version()==before,"normal early placement lost its complete required coverage");
    }
    static void adoptionNegatives(const Program& p, unsigned mutation)
    {
        if (mutation<2) {
            Constructor complete(p);
            const auto plan=complete.run({},true);
            require(plan.success && !complete.activeRoles.empty(),"adoption negative setup failed");
            complete.result=plan;
            const auto& channel=plan.channels.front();
            auto qualifies = [&]() {
                RecurringPacket packet;
                const auto owned=complete.prepareOwnedPacket({});
                require(bool(owned),"empty complete embedding setup failed");
                packet.packet=*owned;
                return complete.qualifyRecurringInterface({0},packet,{});
            };
            require(qualifies(),"unchanged selected role did not embed");
            const auto cut=mutation==0 ? complete.control.graph.exit : channel.publications.front();
            complete.ledger.append(cut,{Command::Publish,channel.source,channel.observer,channel.key},
                                   EndpointPurpose::Completion);
            require(!qualifies(),"duplicate/outside use escaped the complete binding certificate");
            return;
        }
        auto input=p;
        for(auto& row:input.target.keys) {
            for(auto& keys:row) {
                if(!keys.empty() && std::find(keys.begin(),keys.end(),1)==keys.end()) {
                    keys.push_back(1);
                }
            }
        }
        Constructor c(input);
        c.recurringFrontiers=qualifyCyclicFrontiers(input,c.control,c.requirements);
        c.activeFamilies.assign(c.recurringFrontiers.families.size(),false);
        const auto& family=c.recurringFrontiers.families.front();
        const auto& role=c.recurringFrontiers.roles[family.roles.front()];
        c.current=role.acquisitions.front();
        const auto cut=role.publications.front();
        if (mutation==2) {
            const auto other=role.observer;
            c.ledger.append(cut,{Command::Publish,other,role.source,0},EndpointPurpose::Completion);
            c.ledger.append(cut,{Command::Acquire,other,role.source,0},EndpointPurpose::Completion);
        }
        c.ledger.append(cut,{Command::Publish,role.source,role.observer,0},EndpointPurpose::Completion);
        c.ledger.append(c.current,{Command::Acquire,role.source,role.observer,0},EndpointPurpose::Completion);
        require(c.contextualReplay(),c.result.reason);
            std::vector<FrontierRequirement> due{{family.cells.front(),role.source,true,
                c.control.graph.operations[c.current],true,false}};
            const auto version=c.ledger.version();
            require(!c.normalRecurring({0},due,c.normalizeDue(due)) && c.ledger.version()==version,
                    "adopted SET after WAIT was classified as normal at an unchanged anchor");
    }
    static void milestoneMutation(const Program& p)
    {
        Constructor c(p);
        c.recurringFrontiers = qualifyCyclicFrontiers(p,c.control,c.requirements);
        require(!c.recurringFrontiers.roles.empty(), "missing original role");
        auto role = c.recurringFrontiers.roles.front();
        require(c.physicalMilestones(role), "complete physical role lost its milestone");
        role.physicalQualified.reset();
        role.publicationOrigins.clear();
        require(!c.physicalMilestones(role), "missing own-source premise passed through an existing-publication proof");
        const auto version = c.ledger.version();
        require(c.ledger.records().empty() && c.ledger.version() == version, "milestone query selected commands");
    }
};
}
namespace {
void policy()
{
    s::CertifiedRealization normal, repair;
    normal.coverage = {0}; normal.ownCoverage = {0}; normal.order = {1,1,1,o::Pipe::MTE2,o::Pipe::V,{1},{2}};
    repair = normal; repair.placementClass = 2; repair.coverage = {0,1};
    require(s::selectRealization({repair,normal}) == 1, "broader repair beat normal milestone");
    require(s::selectRealization({normal,repair}) == 0, "candidate enumeration changed class priority");
    repair.placementClass = 0;
    require(s::selectRealization({normal,repair}) == 1, "strict coverage dominance was lost within class");
    repair.coverage = {1}; repair.ownCoverage = {1}; repair.order = {2,1,1,o::Pipe::MTE2,o::Pipe::V,{1},{2}};
    require(s::selectRealization({repair,normal}) == 1, "incomparable candidates used discovery order");
    repair.ownCoverage.clear(); repair.coverage={0,1,2};
    require(s::selectRealization({repair,normal}) == 1,"support-only progress selected a nonrequired realization");
    repair.ownCoverage={3};
    require(s::selectRealization({repair,normal}) == 1,"unproved own obligation passed the selector");
}
void rings()
{
    for (unsigned banks=1; banks<=4; ++banks) {
        const auto input = observed_fixtures::ring(banks);
        require(input.success,input.reason);
        const auto plan = accepted(input.program);
        require(!plan.declinedRecurring && plan.channels.size() == 2*banks,
                "ordinary-first selection stranded later complete recurring roles");
        bool competition = false;
        for (const auto& choice : plan.realizationChoices) {
            competition |= !choice.recurring && choice.ordinaryCandidates && choice.recurringCandidates;
        }
        require(competition, "ordinary and recurring providers did not compete through normal construction");
        if (banks == 1) {
            require(plan.work.normalRecurringSelected, "selected return was not realized through normal selector");
            s::ReplayTestAccess::milestoneMutation(input.program);
            for(unsigned mutation=0;mutation!=3;++mutation) {
                s::ReplayTestAccess::adoptionNegatives(input.program,mutation);
            }
        }
    }
}
void subdivision()
{
    std::size_t choices = 0, population = 0;
    for (unsigned cells : {1u,8u,32u}) {
        auto p = base(cells);
        p.operations = {op(o::Pipe::MTE2,{}),op(o::Pipe::V,{})};
        for (unsigned cell=0;cell<cells;++cell) {
            p.operations[0].accesses.push_back({cell,false,true});
            p.operations[1].accesses.push_back({cell,true,false});
        }
        const auto input=o::makePeriodicLoop(p,1,{});
        require(input.success,input.reason);
        const auto plan=accepted(input.program);
        require(plan.channels.size()==2 && plan.work.normalRecurringSelected==1 && plan.activations.size()==1,
                "physical subdivision: cells="+std::to_string(cells)+
                " channels="+std::to_string(plan.channels.size())+
                " normal="+std::to_string(plan.work.normalRecurringSelected)+
                " activations="+std::to_string(plan.activations.size()));
        if (cells == 1) { choices=plan.realizationChoices.size(); population=plan.work.normalCandidates; }
        require(plan.realizationChoices.size()==choices && plan.work.normalCandidates==population,
                "uniform witnesses multiplied normalized decision weight or candidate population");
    }
}
}
int main()
{
    policy(); rings(); subdivision();
    s::ReplayTestAccess::tailOnlyCoverage();
    for (unsigned mutation=0;mutation!=3;++mutation) { s::ReplayTestAccess::alternativeKeys(mutation); }
    for (unsigned mutation=0;mutation!=4;++mutation) { s::ReplayTestAccess::normalization(mutation); }
}
