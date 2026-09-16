// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Replay.h"
#include "PTO/Transforms/OAHS/Bundles.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "AnalysisEquality.h"
#include "ObservedFixtures.h"
#include "PhaseFixtures.h"
#include "GraphOracle.h"
#include <cstdlib>
#include <iostream>
#include <random>
namespace o = mlir::pto::oahs;
namespace {
std::size_t checks=0, comparisons=0, traces=0, reused=0;
void check(bool x, unsigned line) { ++checks; if (!x) { std::cerr << "M6 replay assertion " << line << " after " << comparisons << " comparisons\n"; std::abort(); } }
#define CHECK(x) check(bool(x), __LINE__)
o::Program base(unsigned n) {
  auto p=observed_fixtures::target(3,4,3); p.target.barrierAll=true;
  for (unsigned i=0;i<n;++i) p.operations.push_back(observed_fixtures::op(i%3,i%4,i%2?1:2));
  return p;
}
o::Command post(unsigned a,unsigned b,unsigned k=0) { return {o::Command::Publish,o::Pipe(a),o::Pipe(b),k}; }
o::Command wait(unsigned a,unsigned b,unsigned k=0) { return {o::Command::Acquire,o::Pipe(a),o::Pipe(b),k}; }
void same(o::ReplaySession &session,const o::Program &p,const o::Commands &c,bool states=true) {
  auto hot=session.analyze(c,{states}); auto cold=o::analyze(p,c,{states}); ++comparisons;
  if (oahs_test::report(hot)!=oahs_test::report(cold)) {
    std::cerr << "different reports on case " << comparisons << " (hot " << hot.reason << "; cold " << cold.reason << ")\n";
    CHECK(false);
  }
  CHECK(hot.complete==cold.complete);
  reused+=session.lastReplay().reusedSites;
  if (hot.verified()&&!p.observed&&!p.finalBlocks&&p.operations.size()<10)
    for (const auto &t:oahs_oracle::traces(p,2)) { ++traces; CHECK(oahs_oracle::graph(p,c,t)); }
}
void word(o::Commands &c,const o::Program&p,o::Cut at,std::vector<o::Command> v) {
  const auto leader=o::canonicalCommandCut(p,at);
  for (o::Cut i=0;i<c.size();++i) if(o::canonicalCommandCut(p,i)==leader)c[i]=v;
}
}
int main() {
  { // Removing a fence must retract old completion, not monotonically add facts.
    auto p=base(80); o::Commands c(81); c[70]={{o::Command::BarrierAll}};
    o::ReplaySession session(p); same(session,p,c); c[70].clear(); same(session,p,c);
    CHECK(session.lastReplay().kind==o::ReplayStats::Incremental);
    CHECK(session.lastReplay().reusedSites>=70); CHECK(session.lastReplay().invalidatedSites<15);
    same(session,p,c); CHECK(session.lastReplay().kind==o::ReplayStats::Unchanged);
    CHECK(session.analyze(c).stats.siteEvaluations==0);
    c[75]={post(0,1),wait(0,1)}; same(session,p,c);
    CHECK(session.lastReplay().kind==o::ReplayStats::KeyLayoutChanged);
    std::swap(c[75][0],c[75][1]); same(session,p,c); // same key, invalid event order
    CHECK(session.lastReplay().kind==o::ReplayStats::Incremental);
    auto saved=session.analyze(c); auto broken=c; broken[74]={post(0,1,999)};
    same(session,p,broken); CHECK(session.lastReplay().kind==o::ReplayStats::Invalid);
    CHECK(oahs_test::report(saved)==oahs_test::report(session.analyze(c)));
    CHECK(session.lastReplay().kind==o::ReplayStats::Unchanged);
    same(session,p,c,false); same(session,p,c,true); // reporting choice is not precision
    session.clear(); same(session,p,c); CHECK(session.lastReplay().kind==o::ReplayStats::Cold);
    // Caller changes cannot change the session's original program/target.
    auto original=p; p.operations[0].accesses.clear(); same(session,original,c);
  }
  { // Invalid early receipt must invalidate a DIFFERENT later relay key.
    auto p=base(4); o::Commands c(5);
    c[1]={post(0,1),wait(0,1)}; c[2]={post(1,2),wait(1,2)};
    o::ReplaySession session(p); same(session,p,c);
    c[1].erase(c[1].begin()); same(session,p,c); // same used-key order
    CHECK(session.lastReplay().kind==o::ReplayStats::Incremental);
    CHECK(session.analyze(c).stats.suppressedCommands>0);
    c[1].insert(c[1].begin(),post(0,1)); same(session,p,c); // no sticky suppressions
  }
  { // Backedge must dirty an EARLIER lexical cut and the complete loop SCC.
    auto p=base(5);
    p.body=observed_fixtures::seq({observed_fixtures::leaf(0),
      {o::Region::While,{observed_fixtures::seq({observed_fixtures::leaf(1)}),
       observed_fixtures::seq({{o::Region::Choice,{observed_fixtures::seq({observed_fixtures::leaf(2)}),observed_fixtures::seq({observed_fixtures::leaf(3)})}}})}},
      observed_fixtures::leaf(4)});
    o::Commands c(6); o::ReplaySession session(p); same(session,p,c);
    c[3]={{o::Command::BarrierAll}}; same(session,p,c);
    CHECK(session.lastReplay().invalidatedSites>=5);
    c[3].clear(); same(session,p,c);
  }
  { // Native-style retirement and synchronous lanes retain their exact contract.
    auto p=base(6);
    p.target.synchronous[0]=true;
    p.cells[1].exclusive=true;
    p.invocation.retirement=o::Program::InvocationContract::DrainAllAtReturn;
    o::Commands c(7);c[6]={{o::Command::BarrierAll}};
    o::ReplaySession session(p);same(session,p,c);
    c[6].clear();same(session,p,c);CHECK(!session.analyze(c).retirement.empty());
    c[5]={{o::Command::BarrierAll}};same(session,p,c);
    c[6]={{o::Command::BarrierAll}};same(session,p,c);
    // The same numeric keys in a different first-use order change their layout.
    c[1]={post(0,1,0),wait(0,1,0),post(1,2,1),wait(1,2,1)};same(session,p,c);
    std::rotate(c[1].begin(),c[1].begin()+2,c[1].end());same(session,p,c);
    CHECK(session.lastReplay().kind==o::ReplayStats::KeyLayoutChanged);
  }
  { // Invalid original contracts stay invalid even for an empty command edit.
    auto p=base(2);p.operations[0].complete=false;
    o::ReplaySession session(p);o::Commands c(3);same(session,p,c);
    CHECK(!session.analyze(c).complete);CHECK(session.lastReplay().kind==o::ReplayStats::Invalid);
    c[1]={post(0,1),wait(0,1)};same(session,p,c);
  }
  std::mt19937 rng(0x6a1173u);
  for(unsigned program=0;program<120;++program) {
    auto p=base(7);
    for(auto &op:p.operations) {
      const unsigned lane=rng()%3,cell=rng()%4,mode=rng()%3;
      op=observed_fixtures::op(lane,cell,mode+1);
    }
    if(program%3==1) p.body=observed_fixtures::seq({observed_fixtures::leaf(0),
      {o::Region::For,{observed_fixtures::seq({observed_fixtures::leaf(1),
       {o::Region::Choice,{observed_fixtures::seq({observed_fixtures::leaf(2),observed_fixtures::leaf(3)}),observed_fixtures::seq({observed_fixtures::leaf(4)})}},
        observed_fixtures::leaf(5)})},0,true},observed_fixtures::leaf(6)});
    if(program%3==2) p.body=observed_fixtures::seq({{o::Region::While,{
      observed_fixtures::seq({observed_fixtures::leaf(0),observed_fixtures::leaf(1)}),
      observed_fixtures::seq({observed_fixtures::leaf(2),{o::Region::For,{observed_fixtures::seq({observed_fixtures::leaf(3),observed_fixtures::leaf(4)})},0,true},observed_fixtures::leaf(5)})}},observed_fixtures::leaf(6)});
    auto plan=o::construct(p); CHECK(plan.success);
    auto c=plan.commands; o::ReplaySession hot(p); same(hot,p,c);
    for(unsigned edit=0;edit<30;++edit) {
      const auto at=rng()%c.size(); const auto mutation=rng()%6;
      if(mutation==0) c[at].clear();
      if(mutation==1 && !c[at].empty())c[at].erase(c[at].begin());
      if(mutation==2) c[at].push_back({o::Command::Barrier,o::Pipe(rng()%3)});
      if(mutation==3){unsigned a=rng()%3,b=(a+1+rng()%2)%3,k=rng()%3; c[at].push_back(rng()%2?post(a,b,k):wait(a,b,k));}
      if(mutation==4) { auto other=rng()%c.size(); std::swap(c[at],c[other]); }
      if(mutation==5)c=plan.commands;
      same(hot,p,c,edit%7!=0);
    }
  }
  { // Qualified observations share one word: invalidate ALL matching members.
    auto input=observed_fixtures::ring(2); CHECK(input.success);
    auto p=input.program; auto plan=o::construct(p); CHECK(plan.success);
    o::ReplaySession session(p); auto c=plan.commands; same(session,p,c);
    for (unsigned i=0;i<50;++i) {
      o::Cut at=rng()%c.size(); if(!o::legalCommandCut(p,at))continue;
      word(c,p,at,i%3?std::vector<o::Command>{{o::Command::Barrier,o::Pipe(0)}}:plan.commands[at]);
      same(session,p,c);
    }
  }
  { // Phase/resource history never reuses an ordinary compact checkpoint.
    auto p=phase_fixture::program();
    p.operations={phase_fixture::ordinary(phase_fixture::Q),phase_fixture::producer(p),phase_fixture::consumer(p)};
    phase_fixture::loop(p); o::Commands c(4); c[2]={{o::Command::Barrier,o::Pipe(phase_fixture::F)}};
    o::ReplaySession session(p); same(session,p,c);
    CHECK(session.lastReplay().kind==o::ReplayStats::PhaseFull);
    same(session,p,c); CHECK(session.lastReplay().kind==o::ReplayStats::Unchanged);
    c[1]=phase_fixture::handoff(phase_fixture::Q,phase_fixture::M); same(session,p,c);
    CHECK(session.lastReplay().kind==o::ReplayStats::PhaseFull);
    c[1].clear();same(session,p,c);c[2].clear();same(session,p,c);
    CHECK(!o::phaseNativeQualification().enabled);
  }
  { // Prefix indexes are exact per immutable query, not shared across plan edits.
    auto p=base(8);o::PrefixQuery query(p);auto a=query.coverByPrefixes(6);auto first=query.statistics();
    auto b=query.coverByPrefixes(6);auto second=query.statistics();
    CHECK(a.candidates.size()==b.candidates.size());
    CHECK(first.backwardTraversals==second.backwardTraversals && second.backwardHits>first.backwardHits);
    CHECK(first.correspondenceTraversals==second.correspondenceTraversals && second.correspondenceHits>first.correspondenceHits);
    CHECK(first.sourceSnapshots==second.sourceSnapshots && second.sourceHits>first.sourceHits);
  }
  std::cout<<checks<<" M6 replay assertions; "<<comparisons<<" complete-report comparisons; "<<traces<<" concrete traces; "<<reused<<" reused sites\n";
}
