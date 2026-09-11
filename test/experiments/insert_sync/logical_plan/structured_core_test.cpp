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
    struct Item { Segment segment; uint64_t t,rank; unsigned side; uint64_t actionOrder;
        std::size_t atom; const Action *action; };
    std::vector<Item> items;
    for(std::size_t p=0;p<m.atoms.size();++p) {
        const auto &atom=m.atoms[p];
        auto append=[&](uint64_t t) {
            items.push_back({atom.segment,t,atom.order,1,0,p,nullptr});
            for(const auto &a:actions)if(a.anchor==p) {
                bool active=true;
                if (a.participation==Action::IfBody) active=trips>a.guardResidue;
                else if (a.participation==Action::First) active=t==atom.residue;
                else if (a.participation==Action::Last) active=trips-t<=m.period;
                else if (atom.segment==Segment::Body) active=a.kind==Action::Set?a.distanceInIterations<trips-t:
                    a.kind==Action::Wait?t>=a.distanceInIterations:true;
                if(active)items.push_back({atom.segment,t,atom.order,a.after?2u:0u,a.order,p,&a});
            }
        };
        if (atom.segment!=Segment::Body) append(0);
        else for(uint64_t t=0;t<trips;++t) {
            if(m.recurring?t%m.period!=atom.residue:t!=atom.residue)continue;
            append(t);
        }
    }
    std::sort(items.begin(),items.end(),[](const auto&a,const auto&b){
        return std::tie(a.segment,a.t,a.rank,a.side,a.actionOrder)<
               std::tie(b.segment,b.t,b.rank,b.side,b.actionOrder);});
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
            if(std::make_tuple(m.atoms[r.source].segment,pt,m.atoms[r.source].order)>=
               std::make_tuple(m.atoms[r.target].segment,qt,m.atoms[r.target].order))continue;
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
    uint64_t boundaryAccepted=0,boundaryRefused=0;
    Model bridges;bridges.period=3;
    bridges.atoms={{0,0,{Core::AIV,Pipe::MTE2},Segment::Prelude},
                   {0,1,{Core::AIV,Pipe::MTE2},Segment::Prelude},
                   {0,2,{Core::AIV,Pipe::V}}, {1,3,{Core::AIV,Pipe::V}},
                   {2,4,{Core::AIV,Pipe::V}},
                   {0,5,{Core::AIV,Pipe::MTE3},Segment::Epilogue}};
    hazard(bridges,0,2);hazard(bridges,0,3);hazard(bridges,1,4);
    hazard(bridges,2,5);hazard(bridges,3,5);hazard(bridges,4,5);
    auto bridge=challenge(bridges,"first/last cross-region handoffs");++boundaryAccepted;
    auto bridgeActions=actionsForPlan(bridges,bridge.plan);
    for (uint64_t trips=1;trips<11;++trips) {
        auto x=expand(bridges,bridgeActions,trips);
        require(!x.reaches(x.payload.at({1,0}).second,x.payload.at({2,0}).first),
                "independent later preload was acquired by first reader");
    }
    Plan partial;
    for (const auto &h:bridge.plan.handoffs) if(h.source==0&&h.target==2)partial.handoffs.push_back(h);
    require(supplies(bridges,partial,bridges.requirements[0]),"supply query depends on unrelated unresolved requirements");
    require(!supplies(bridges,partial,bridges.requirements[2]),"unacquired second preload was inferred");
    unsigned entryCount=0;
    for (const auto &a:bridgeActions) if(a.kind==Action::Wait&&a.participation==Action::First)++entryCount;
    require(entryCount==2,"repeated readers acquired the same preload again");
    unsigned prematureWitnesses=0;
    for(std::size_t i=0;i<bridgeActions.size();++i) {
        auto bad=bridgeActions;bad.erase(bad.begin()+i);
        require(verify(bridges,bad).status!=Status::Applied,"missing boundary endpoint accepted");
        bad=bridgeActions;
        if(bad[i].participation==Action::IfBody) {
            ++bad[i].guardResidue;
            require(verify(bridges,bad).status!=Status::Applied,"wrong boundary population accepted");
        }
        bad=bridgeActions;
        if(bad[i].participation==Action::Last) {
            bad[i].participation=Action::First;
            require(verify(bridges,bad).status!=Status::Applied,"first reader substituted for last reader");
            if (!finiteCorrect(bridges,bad,10)) ++prematureWitnesses;
        }
        bad=bridgeActions;
        if(bad[i].participation==Action::First) {
            bad[i].participation=Action::Every;
            require(verify(bridges,bad).status!=Status::Applied,"repeated consuming wait accepted");
        }
    }
    require(prematureWitnesses>0,"finite oracle missed every premature release witness");
    auto zero=bridges;hazard(zero,0,5);
    auto z=challenge(zero,"direct zero-trip bypass");++boundaryAccepted;
    bool bypass=false;for(const auto &h:z.plan.handoffs)if(h.source==0&&h.target==5)bypass=true;
    require(bypass,"loop completion used on zero-trip bypass");
    // A first same-pipe barrier supplies the preload once, not every iteration.
    Model same;same.period=2;
    same.atoms={{0,0,{Core::AIV,Pipe::MTE2},Segment::Prelude},
                {0,1,{Core::AIV,Pipe::MTE2}}, {1,2,{Core::AIV,Pipe::MTE2}},
                {0,3,{Core::AIV,Pipe::V},Segment::Epilogue}};
    hazard(same,0,1);hazard(same,0,2);hazard(same,1,3);hazard(same,2,3);
    auto sm=challenge(same,"first-only barrier and final release");++boundaryAccepted;
    require(sm.plan.firstBarriers.size()==1,"entry barrier repeats unnecessarily");
    Model fanout;fanout.atoms={{0,0,{Core::AIV,Pipe::MTE2},Segment::Prelude},
        {0,1,{Core::AIV,Pipe::V}}, {0,2,{Core::AIV,Pipe::MTE3}},
        {0,3,{Core::AIV,Pipe::MTE2},Segment::Epilogue}};
    for(auto e:{std::pair<unsigned,unsigned>{0,1},{0,2},{1,3},{2,3},{0,3}})hazard(fanout,e.first,e.second);
    auto fo=challenge(fanout,"all consumer lanes release before overwrite");++boundaryAccepted;
    auto fa=actionsForPlan(fanout,fo.plan);
    fa.erase(std::remove_if(fa.begin(),fa.end(),[](const Action &a){
        return a.kind!=Action::Barrier && a.source.pipe==Pipe::MTE3 && a.target.pipe==Pipe::MTE2;
    }),fa.end());
    require(verify(fanout,fa).status!=Status::Applied,"one reader lane was silently forgotten");
    require(!finiteCorrect(fanout,fa,3),"finite oracle missed outstanding second reader");
    Model mixed;mixed.atoms={{0,0,{Core::AIV,Pipe::MTE2},Segment::Prelude},
        {0,1,{Core::AIV,Pipe::V}}, {0,2,{Core::AIV,Pipe::MTE2}}, {0,3,{Core::AIV,Pipe::V}}};
    hazard(mixed,0,1);hazard(mixed,2,3);hazard(mixed,3,2);hazard(mixed,2,2);
    auto mix=challenge(mixed,"one allocation sees boundary and recurring streams");++boundaryAccepted;
    auto ma=actionsForPlan(mixed,mix.plan);
    std::optional<unsigned> bodyKey;
    for(const auto &a:ma)if(a.kind==Action::Set&&a.source.pipe==Pipe::MTE2&&a.participation==Action::Every)bodyKey=a.key;
    require(bool(bodyKey),"mixed fixture lost recurring handoff");
    for(auto &a:ma)if(a.kind!=Action::Barrier&&a.source.pipe==Pipe::MTE2&&a.participation!=Action::Every)a.key=*bodyKey;
    require(verify(mixed,ma).status!=Status::Applied,"boundary/recurrence key collision accepted");
    auto tight=mixed;tight.target.compilerKeys={0};
    require(construct(tight).status==Status::AllocationFailure,"boundary scarcity silently serialized");
    auto noBoundary=same;noBoundary.atoms[0].segment=static_cast<Segment>(99);
    require(construct(noBoundary).status==Status::Unsupported,"invalid segment accepted");
    auto largeBoundary=bridges;largeBoundary.period=uint64_t(INT64_MAX);
    auto lb=construct(largeBoundary);
    require(lb.status==Status::Applied && lb.completionRelaxations<1000000,
            "large numeric boundary period was expanded");
    // Hand-transcribed static storage summary of the pinned QK input, NOT a
    // native import result. The native gate below remains required.
    {
    struct QKAccess {unsigned space;uint64_t begin,end;bool write;};
Model m;std::vector<std::vector<QKAccess>> access;
auto phase=[&](Pipe p,Segment segment,std::vector<QKAccess> a){m.atoms.push_back({0,m.atoms.size(),{Core::AIC,p},segment});access.push_back(a);};
phase(Pipe::MTE2,Segment::Prelude,{{1,0,4096,true}});
phase(Pipe::MTE2,Segment::Prelude,{{1,4096,8192,true}});
for(unsigned half=0;half<2;++half){
phase(Pipe::MTE2,Segment::Body,{{1,8192,73728,true}});
phase(Pipe::MTE1,Segment::Body,{{1,half*4096,(half+1)*4096,false},{2,0,2048,true}});
phase(Pipe::MTE1,Segment::Body,{{1,8192,73728,false},{3,32768,65536,true}});
phase(Pipe::MTE1,Segment::Body,{{1,half*4096,(half+1)*4096,false},{2,2048,4096,true}});
phase(Pipe::MTE1,Segment::Body,{{1,8192,73728,false},{3,0,32768,true}});
phase(Pipe::M,Segment::Body,{{2,0,2048,false},{3,32768,65536,false},{4,0,16384,true}});
phase(Pipe::M,Segment::Body,{{2,2048,4096,false},{3,0,32768,false},{4,0,16384,false},{4,0,16384,true}});
phase(Pipe::FIX,Segment::Body,{{4,0,16384,false},{5,0,1,true}});
}
for(std::size_t p=0;p<access.size();++p)for(std::size_t q=0;q<access.size();++q){
auto d=priorDistance(m,p,q);if(!d)continue;bool conflict=false,acc=false;
for(auto a:access[p])for(auto b:access[q])if(a.space==b.space&&a.begin<b.end&&b.begin<a.end){
bool resource=a.space==4&&m.atoms[p].lane!=m.atoms[q].lane;conflict|=a.write||b.write||resource;acc|=resource;}
if(conflict)m.requirements.push_back({p,q,*d,acc?Property::AccResource:Property::Completion});}
    auto qk=challenge(m,"QK-shaped cross-region summary");++boundaryAccepted;
    auto qkGraph=expand(m,actionsForPlan(m,qk.plan),3);
    require(!qkGraph.reaches(qkGraph.payload.at({1,0}).second,qkGraph.payload.at({3,0}).first),
            "QK second Q preload blocks first panel extraction");
    require(!qkGraph.reaches(qkGraph.payload.at({2,0}).second,qkGraph.payload.at({3,0}).first),
            "QK independent K load blocks first panel extraction");
    }
    // Randomized complete conflict populations, not hand-picked channel names.
    // Finite checking independently tests every older conflicting occurrence.
    for(unsigned test=0;test<800;++test) {
        Model m;m.period=1+rng()%4;
        unsigned n=3+rng()%7;
        std::vector<unsigned> storage(n),writes(n);
        for(unsigned i=0;i<n;++i) {
            Segment segment=i==0?Segment::Prelude:i+1==n?Segment::Epilogue:Segment(rng()%3);
            m.atoms.push_back({segment==Segment::Body?rng()%m.period:0,i,
                              {Core::AIV,lanes[rng()%3]},segment});
            storage[i]=rng()%3;writes[i]=rng()%2;
        }
        for(unsigned p=0;p<n;++p)for(unsigned q=0;q<n;++q)
            if(storage[p]==storage[q]&&(writes[p]||writes[q]))hazard(m,p,q);
        auto result=construct(m);
        if(result.status!=Status::Applied) {
            require(result.status==Status::AllocationFailure,"unexpected boundary construction failure");
            ++boundaryRefused;continue;
        }
        ++boundaryAccepted;++acceptedModels;
        auto actions=actionsForPlan(m,result.plan);
        for(uint64_t trips=0;trips<=4*m.period+1;++trips) {
            if (!finiteCorrect(m,actions,trips)) {
                std::cerr<<"boundary test "<<test<<" trip "<<trips<<" period "<<m.period<<"\n";
                for (unsigned j=0;j<n;++j) std::cerr<<j<<":"<<unsigned(m.atoms[j].segment)<<","<<m.atoms[j].residue<<","<<unsigned(m.atoms[j].lane.pipe)<<" storage "<<storage[j]<<" write "<<writes[j]<<"\n";
                require(false,"random boundary finite asynchronous reference");
            }
            require(true,"random boundary finite asynchronous reference");
        }
    }
    std::cout<<"{\"status\":\"passed\",\"checks\":"<<checks<<",\"accepted_models\":"<<acceptedModels
        <<",\"finite_executions\":"<<finiteRuns<<",\"random_allocation_refusals\":"<<refused
        <<",\"boundary_accepted_models\":"<<boundaryAccepted<<",\"boundary_allocation_refusals\":"<<boundaryRefused
        <<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"}\n";
}
