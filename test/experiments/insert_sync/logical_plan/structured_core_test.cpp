// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
using namespace mlir::pto::structured_sync;
static uint64_t checks=0, finiteRuns=0, acceptedModels=0;
static void require(bool p,const char *message) {
    ++checks;
    if (!p) { std::cerr<<"FAILED: "<<message<<'\n'; std::exit(1); }
}
using Key=std::tuple<Lane,Lane,unsigned>;
struct Expanded {
    std::vector<std::vector<unsigned>> graph;
    std::map<std::pair<std::size_t,uint64_t>,std::pair<unsigned,unsigned>> payload;
    std::map<Key,std::vector<std::pair<unsigned,unsigned>>> episodes;
    bool valid=true;
    unsigned node() { graph.emplace_back(); return unsigned(graph.size()-1); }
    void edge(unsigned a,unsigned b) { graph[a].push_back(b); }
    bool reaches(unsigned a,unsigned b) const {
        std::vector<bool> seen(graph.size(),false); std::vector<unsigned> todo{a}; seen[a]=true;
        while (!todo.empty()) { unsigned p=todo.back();todo.pop_back();
            if(p==b)return true;
            for(unsigned q:graph[p])if(!seen[q]){seen[q]=true;todo.push_back(q);}
        }
        return false;
    }
    bool acyclic() const {
        std::vector<unsigned> degree(graph.size(),0); for(const auto& row:graph)for(unsigned q:row)++degree[q];
        std::vector<unsigned> ready;for(unsigned i=0;i<degree.size();++i)if(!degree[i])ready.push_back(i);
        unsigned count=0;while(!ready.empty()){unsigned p=ready.back();ready.pop_back();++count;
            for(unsigned q:graph[p])if(--degree[q]==0)ready.push_back(q);}
        return count==graph.size();
    }
};

// Independent finite asynchronous semantics: payload issue and completion are
// DIFFERENT nodes. A set has separate enqueue and fire nodes. No implicit edge
// from set fire to the next source command is invented. This oracle does not
// use the production two-layer cut matrices or production event pairing.
static Expanded expand(const Model &m,const std::vector<Action> &actions,uint64_t trips) {
    struct Item { uint64_t t,rank; unsigned side; uint64_t actionOrder;
        std::size_t atom; const Action *action; };
    std::vector<Item> items;
    for(uint64_t t=0;t<trips;++t)for(std::size_t p=0;p<m.atoms.size();++p) {
        if(m.recurring?t%m.period!=m.atoms[p].residue:t!=m.atoms[p].residue)continue;
        items.push_back({t,m.atoms[p].order,1,0,p,nullptr});
        for(const auto &a:actions)if(a.anchor==p) {
            bool active=a.kind==Action::Set?a.distanceInIterations<trips-t:
                a.kind==Action::Wait?t>=a.distanceInIterations:true;
            if(active)items.push_back({t,m.atoms[p].order,a.after?2u:0u,a.order,p,&a});
        }
    }
    std::sort(items.begin(),items.end(),[](const auto&a,const auto&b){
        return std::tie(a.t,a.rank,a.side,a.actionOrder)<std::tie(b.t,b.rank,b.side,b.actionOrder);});
    Expanded x;
    std::map<Lane,unsigned> previous;
    std::map<Lane,std::vector<unsigned>> completed;
    std::map<Key,std::deque<unsigned>> pending;
    auto issue=[&](Lane lane) { unsigned n=x.node();auto p=previous.find(lane);
        if (p!=previous.end()) x.edge(p->second,n);
        previous[lane]=n;
        return n;
    };
    for(const auto &item:items) {
        if(!item.action) {
            Lane lane=m.atoms[item.atom].lane;
            unsigned start=issue(lane),done=x.node();x.edge(start,done);
            if(m.target.synchronous(lane))previous[lane]=done;
            completed[lane].push_back(done);
            x.payload[{item.atom,m.recurring?item.t/m.period:0}]={start,done};
            continue;
        }
        const auto &a=*item.action;Key key{a.source,a.target,a.key};
        if(a.kind==Action::Set) {
            unsigned enqueued=issue(a.source),fire=x.node();x.edge(enqueued,fire);
            for(unsigned done:completed[a.source])x.edge(done,fire);
            pending[key].push_back(fire); // queuing is NOT accepted as a flag-capacity proof
        } else if(a.kind==Action::Wait) {
            unsigned take=issue(a.target);
            if(pending[key].empty()){x.valid=false;continue;}
            unsigned fire=pending[key].front();pending[key].pop_front();
            x.edge(fire,take);x.episodes[key].push_back({fire,take});
        } else {
            unsigned pass=issue(a.source);for(unsigned done:completed[a.source])x.edge(done,pass);
        }
    }
    for(const auto &p:pending)if(!p.second.empty())x.valid=false;
    return x;
}
static bool finiteCorrect(const Model &m,const std::vector<Action> &actions,uint64_t trips) {
    ++finiteRuns;auto x=expand(m,actions,trips);if(!x.valid||!x.acyclic())return false;
    // All older conflicting occurrences are checked, not just the latest one
    // compressed into each immutable native requirement.
    for(const auto &r:m.requirements)for(const auto &q:x.payload)if(q.first.first==r.target)
        for(const auto &p:x.payload)if(p.first.first==r.source) {
            uint64_t pt=p.first.second*m.period+m.atoms[r.source].residue;
            uint64_t qt=q.first.second*m.period+m.atoms[r.target].residue;
            if(std::make_pair(pt,m.atoms[r.source].order)>=std::make_pair(qt,m.atoms[r.target].order))continue;
            if(!x.reaches(p.second.second,q.second.first))return false;
        }
    for(const auto &f:x.episodes)for(std::size_t i=1;i<f.second.size();++i)
        if(!x.reaches(f.second[i-1].second,f.second[i].first))return false;
    return true;
}
static void hazard(Model&m,std::size_t p,std::size_t q,Property property=Property::Completion){
    auto d=priorDistance(m,p,q);if(d)m.requirements.push_back({p,q,*d,property});
}
static Model buffering(unsigned d,bool stores=true){
    Model m;m.period=d;
    for(unsigned r=0;r<d;++r){
        m.atoms.push_back({r,0,{Core::AIV,Pipe::MTE2}});
        m.atoms.push_back({r,1,{Core::AIV,Pipe::V}});
        m.atoms.push_back({r,2,{Core::AIV,Pipe::MTE3}});
        for(auto a:{std::pair<unsigned,unsigned>{3*r,3*r+1},{3*r+1,3*r},{3*r+1,3*r+2},
                    {3*r+2,3*r+1},{3*r,3*r},{3*r+1,3*r+1}})hazard(m,a.first,a.second);
    }
    if(stores)for(unsigned r=0;r<d;++r)for(unsigned s=0;s<d;++s)hazard(m,3*r+2,3*s+2);
    return m;
}
static Result challenge(const Model&m,const char *name){
    auto r=construct(m);require(r.status==Status::Applied,name);++acceptedModels;
    auto actions=actionsForPlan(m,r.plan);
    require(verify(m,actions).status==Status::Applied,"fresh plan check");
    for(uint64_t n=0;n<=3*m.period+1;++n)require(finiteCorrect(m,actions,n),"finite async semantics");
    return r;
}
int main(){
    auto started=std::chrono::steady_clock::now();
    for(unsigned d=1;d<=6;++d)challenge(buffering(d),"ordinary buffering");
    Model early;early.atoms={{0,0,{Core::AIV,Pipe::MTE2}},{0,1,{Core::AIV,Pipe::MTE2}},
                            {0,2,{Core::AIV,Pipe::V}},{0,3,{Core::AIV,Pipe::V}}};
    for(auto pair:{std::pair<unsigned,unsigned>{0,2},{2,0},{1,3},{3,1},{0,0},{1,1}})hazard(early,pair.first,pair.second);
    auto first=challenge(early,"independent preloads");
    auto earlyActions=actionsForPlan(early,first.plan);
    auto graph=expand(early,earlyActions,3);
    require(!graph.reaches(graph.payload.at({1,0}).second,graph.payload.at({2,0}).first),"second preload blocks first consumer");
    // ID permutation is not schedule order.
    Model shuffled=early;
    std::reverse(shuffled.atoms.begin(),shuffled.atoms.end());
    for(auto &r:shuffled.requirements){r.source=3-r.source;r.target=3-r.target;}
    challenge(shuffled,"arbitrary phase IDs");
    // Dedicated keys have recurrence requirements too. Removing the reverse
    // handoff from a repeated producer/consumer is not safe event reuse.
    Model recurrence;recurrence.atoms={{0,0,{Core::AIV,Pipe::MTE2}},{0,1,{Core::AIV,Pipe::V}}};
    hazard(recurrence,0,1);
    require(construct(recurrence).status==Status::AllocationFailure,"unacknowledged recurring event accepted");
    recurrence.recurring=false;recurrence.requirements.clear();hazard(recurrence,0,1);
    auto once=challenge(recurrence,"one shot");
    auto mutations=actionsForPlan(recurrence,once.plan);
    require(!finiteCorrect(recurrence,{},1),"negative async oracle");
    for(std::size_t i=0;i<mutations.size();++i){auto bad=mutations;bad.erase(bad.begin()+i);
        require(verify(recurrence,bad).status!=Status::Applied,"missing action accepted");}
    auto wrong=mutations;wrong.back().key=1;
    require(verify(recurrence,wrong).status!=Status::Applied,"wrong key accepted");
    wrong=mutations;wrong.back().distanceInIterations=1;
    require(verify(recurrence,wrong).status!=Status::Applied,"wrong generation accepted");
    wrong=mutations;wrong.push_back(wrong.front());wrong.back().order+=99;
    require(verify(recurrence,wrong).status!=Status::Applied,"duplicate notification accepted");
    auto reserved=recurrence;reserved.target.reservations.push_back({recurrence.atoms[0].lane,recurrence.atoms[1].lane,0});
    auto reservePlan=challenge(reserved,"reservation holes");
    require(reservePlan.plan.handoffs.front().key==1,"reserved key allocated");
    auto scarce=early;scarce.target.compilerKeys={0};
    require(construct(scarce).status==Status::AllocationFailure,"scarcity silently serialized independent work");
    // Same-pipe issue is not completion; an MTE2 WAW needs a barrier unless
    // acquired completion really supplies it through another lane.
    Model waw;waw.recurring=false;waw.atoms={{0,0,{Core::AIV,Pipe::MTE2}},{0,1,{Core::AIV,Pipe::MTE2}}};
    hazard(waw,0,1);auto w=challenge(waw,"overlapping loads");
    require(w.plan.barriers.size()==1,"overlapping MTE2 writes exempted");
    require(!supplies(waw,{},waw.requirements[0]),"issue-only completion");
    auto scalar=waw;for(auto &a:scalar.atoms)a.lane.pipe=Pipe::S;
    auto sp=challenge(scalar,"scalar intrinsic ordering");require(sp.plan.barriers.empty(),"PIPE_S emitted");
    auto vis=recurrence;vis.requirements.front().property=Property::Visibility;
    require(construct(vis).status==Status::Unsupported,"event treated as GM visibility");
    auto cross=recurrence;cross.atoms[1].lane={Core::AIC,Pipe::M};
    require(construct(cross).status==Status::Unsupported,"cross-core flag accepted");
    Target t;require(!t.barrier({Core::AIC,Pipe::S})&&!t.barrier({Core::AIV,Pipe::S}),"scalar barrier legality");
    require(!t.event({Core::AIC,Pipe::S},{Core::AIC,Pipe::M}),"invented scalar AIC event");
    require(!t.event({Core::AIC,Pipe::M},{Core::AIC,Pipe::MTE3}),"invented M to MTE3 event");
    require(t.event({Core::AIC,Pipe::M},{Core::AIC,Pipe::FIX}),"qualified M/FIX event lost");
    auto acc=recurrence;acc.recurring=true;acc.atoms[0].lane={Core::AIC,Pipe::M};acc.atoms[1].lane={Core::AIC,Pipe::FIX};
    acc.requirements.clear();hazard(acc,0,1,Property::AccResource);hazard(acc,1,0,Property::AccResource);
    challenge(acc,"ACC reader resource ordering");
    // Large periods are represented numerically, never enumerated by the core.
    Model large;large.period=uint64_t(INT64_MAX);large.atoms={{0,0,{Core::AIV,Pipe::MTE2}},
        {0,1,{Core::AIV,Pipe::V}}};hazard(large,0,1);hazard(large,1,0);
    auto lp=construct(large);require(lp.status==Status::Applied,"large period scalar map");
    require(lp.completionRelaxations<1000,"numeric period enumerated");

    std::mt19937 rng(0x0a450051);
    const Pipe lanes[]={Pipe::MTE2,Pipe::V,Pipe::MTE3};
    uint64_t refused=0;
    for(unsigned test=0;test<400;++test){
        Model m;m.period=1+rng()%4;m.recurring=(test%5)!=0;
        unsigned n=2+rng()%7;
        std::vector<unsigned> storage(n),writes(n);
        std::set<std::pair<uint64_t,uint64_t>> used;
        for(unsigned i=0;i<n;++i){
            uint64_t r=rng()%m.period,rank=i;
            m.atoms.push_back({r,rank,{Core::AIV,lanes[rng()%3]}});
            storage[i]=rng()%3;writes[i]=rng()%2;
        }
        for(unsigned p=0;p<n;++p)for(unsigned q=0;q<n;++q)
            if(storage[p]==storage[q]&&(writes[p]||writes[q]))hazard(m,p,q);
        auto result=construct(m);
        if(result.status!=Status::Applied){require(result.status==Status::AllocationFailure,"unexpected randomized planner status");++refused;continue;}
        ++acceptedModels;auto a=actionsForPlan(m,result.plan);
        for(uint64_t trips=0;trips<=2*m.period+1;++trips)
            require(finiteCorrect(m,a,trips),"random finite asynchronous reference");
    }
    std::cout<<"{\"status\":\"passed\",\"checks\":"<<checks<<",\"accepted_models\":"<<acceptedModels
        <<",\"finite_executions\":"<<finiteRuns<<",\"random_allocation_refusals\":"<<refused
        <<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";
}
