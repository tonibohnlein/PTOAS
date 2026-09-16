// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Replay.h"
#include "PTO/Transforms/OAHS/Bundles.h"
#include "ObservedFixtures.h"
#include "GraphOracle.h"
#include <cstdlib>
#include <iostream>
#include <random>
namespace o=mlir::pto::oahs;
namespace {
unsigned checks=0, traces=0, improvements=0, accepted=0;
void check(bool b,unsigned line){++checks;if(!b){std::cerr<<"resource-choice assertion "<<line<<'\n';std::abort();}}
#define CHECK(x) check(bool(x),__LINE__)
o::Program twoTransfers(){
 auto p=observed_fixtures::target(3,2,1);
 for(auto &row:p.target.keys)for(auto &pool:row)pool.clear();
 p.operations={observed_fixtures::op(0,0,2),observed_fixtures::op(1,0,1),
               observed_fixtures::op(0,1,2),observed_fixtures::op(1,1,1)};
 p.target.keys[0][1]={0};return p;
}
void oracle(const o::Program &p,const o::Result&r){
 CHECK(r.success);CHECK(o::verify(p,r.commands).success);
 for(auto t:oahs_oracle::traces(p,3)){++traces;CHECK(oahs_oracle::graph(p,r.commands,t));}
}
unsigned count(const o::Commands&c,o::Command::Kind k){unsigned n=0;for(auto &word:c)for(auto x:word)n+=x.kind==k;return n;}
bool same(const o::Commands&a,const o::Commands&b){
 if(a.size()!=b.size())return false;
 for(unsigned i=0;i<a.size();++i){if(a[i].size()!=b[i].size())return false;
 for(unsigned j=0;j<a[i].size();++j){auto x=a[i][j],y=b[i][j];if(x.kind!=y.kind||x.source!=y.source||x.observer!=y.observer||x.key!=y.key)return false;}}
 return true;
}
}
int main(){
 { // A direct direction EXISTS, but its reused key is not causally available.
   // There is no return direction. A longer forward route uses fresh keys.
  auto p=twoTransfers();p.target.keys[0][2]={0};p.target.keys[2][1]={0};
  auto old=o::construct(p,{false,true,false});CHECK(!old.success);
  auto plan=o::construct(p,{false,true,true});oracle(p,plan);
  CHECK(plan.work.longerRouteTrials>0);CHECK(plan.work.routedReplies==0);
  CHECK(count(plan.commands,o::Command::Publish)==3);
  CHECK(count(plan.commands,o::Command::Acquire)==3);
  CHECK(count(plan.commands,o::Command::BarrierAll)==0);
  CHECK(plan.stages.size()>=2);CHECK(!plan.stages.front().success);
  CHECK(plan.stages.back().resourcePass==o::ResourcePass::Expanded);
  CHECK(same(plan.commands,o::construct(p,{false,false,true}).commands));
  p.reservations.push_back({o::Pipe(2),o::Pipe(1),0,false});
  CHECK(!o::construct(p,{false,true,true}).success);
 }
 { // No longer forward route. A real two-hop acknowledgment is required.
  auto p=twoTransfers();p.target.keys[1][2]={0};p.target.keys[2][0]={0};
  CHECK(!o::construct(p,{false,true,false}).success);
  auto plan=o::construct(p,{false,true,true});oracle(p,plan);
  CHECK(plan.work.routedReplies==1);
  CHECK(count(plan.commands,o::Command::Publish)==4);
  CHECK(count(plan.commands,o::Command::Acquire)==4);
  CHECK(count(plan.commands,o::Command::BarrierAll)==0);
  // Receipt-consumption reply BEFORE the first read is not a reader release.
  CHECK(oahs_oracle::graph(p,plan.commands,{0,1,2,3},{{1,2}}));
  auto broken=plan.commands;
  for(auto &word:broken)for(auto i=word.begin();i!=word.end();++i)
    if(i->kind==o::Command::Acquire&&i->source==o::Pipe(2)){word.erase(i);goto removed;}
 removed: CHECK(!o::verify(p,broken).success);
  p.reservations.push_back({o::Pipe(1),o::Pipe(2),0,false});CHECK(!o::construct(p).success);
 }
 { // Genuine reader release carries the return: no automatic extra reply.
  auto p=twoTransfers();p.operations[2].accesses[0].cell=0;p.operations[3].accesses[0].cell=0;
  p.target.keys[1][0]={0};auto plan=o::construct(p);oracle(p,plan);
  CHECK(count(plan.commands,o::Command::Publish)==3);CHECK(plan.work.routedReplies==0);
 }
 { // Finite resource exhaustion remains distinct from work budgets.
  auto p=twoTransfers();CHECK(!o::construct(p).success);
  p.target.barrierAll=true;auto plan=o::construct(p);oracle(p,plan);
  CHECK(plan.stages.back().resourcePass==o::ResourcePass::BaselineRecovery);
  CHECK(plan.conservativeBarriers>0&&plan.bundleTrials>0);
  o::ConstructionWork sum;std::size_t trials=0;
  for(const auto&s:plan.stages){sum+=s.work;trials+=s.bundleTrials;}
  CHECK(sum.routeKeyTrials==plan.work.routeKeyTrials);
  CHECK(sum.replaySiteEvaluations==plan.work.replaySiteEvaluations);
  CHECK(trials==plan.bundleTrials);
 }
 { // A different used key can satisfy recurrence once the complete plan exists.
  auto p=observed_fixtures::target(3,3,2);
  p.target.keys[0][1].clear();p.target.keys[0][2]={0};p.target.keys[1][2]={0};
  p.operations={observed_fixtures::op(1,2,1),observed_fixtures::op(2,2,1),
    observed_fixtures::op(1,2,2),observed_fixtures::op(2,0,1),
    observed_fixtures::op(1,2,2),observed_fixtures::op(1,0,3),
    observed_fixtures::op(0,2,1),observed_fixtures::op(0,2,3)};
  p.body={o::Region::For,{observed_fixtures::seq({observed_fixtures::leaf(0),observed_fixtures::leaf(1),
    {o::Region::Choice,{observed_fixtures::seq({observed_fixtures::leaf(2),observed_fixtures::leaf(3)}),
      observed_fixtures::seq({observed_fixtures::leaf(4),observed_fixtures::leaf(5)})}},
    observed_fixtures::leaf(6),observed_fixtures::leaf(7)})},0,true};
  auto plan=o::construct(p);oracle(p,plan);CHECK(plan.work.rekeyTrials>0&&plan.work.rekeys==1);
  auto cold=o::construct(p,{true,false,true});CHECK(cold.success&&same(plan.commands,cold.commands));
 }
 std::mt19937 rng(0x6b8731u);
 for(unsigned n=0;n<160;++n){
  auto p=observed_fixtures::target(3,3,1);
  for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)if(a!=b){
    auto r=rng()%4;if(r==0)p.target.keys[a][b].clear();else if(r==3)p.target.keys[a][b].push_back(1);
  }
  for(unsigned i=0;i<6;++i){auto a=rng()%3,c=rng()%3,m=rng()%3;p.operations.push_back(observed_fixtures::op(a,c,m+1));}
  if(n%2)p.body={o::Region::For,{observed_fixtures::seq({observed_fixtures::leaf(0),
    {o::Region::Choice,{observed_fixtures::seq({observed_fixtures::leaf(1),observed_fixtures::leaf(2)}),observed_fixtures::seq({observed_fixtures::leaf(3)})}},
    observed_fixtures::leaf(4),observed_fixtures::leaf(5)})},0,true};
  auto old=o::construct(p,{true,true,false});auto fresh=o::construct(p,{true,true,true});
  CHECK(!old.success||fresh.success); // exact old policy remains a final recovery
  if(fresh.success){++accepted;oracle(p,fresh);improvements+=!old.success;}
  auto cold=o::construct(p,{true,false,true});CHECK(cold.success==fresh.success);
  if(fresh.success)CHECK(same(cold.commands,fresh.commands));
 }
 std::cout<<checks<<" M6 resource assertions; "<<traces<<" graph traces; "<<accepted
          <<" seeded accepts; "<<improvements<<" seeded admission gains over retained policy\n";
}
