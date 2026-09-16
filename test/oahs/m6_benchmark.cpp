// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Replay.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "ObservedFixtures.h"
#include "PhaseFixtures.h"
#include "AnalysisEquality.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
namespace o=mlir::pto::oahs;
using Clock=std::chrono::steady_clock;
struct Run { double ms=0;uint64_t evaluations=0,reused=0,work=0;std::size_t full=0,hot=0;std::vector<std::string> results;};
bool sameWords(const o::Commands&a,const o::Commands&b) {
 if(a.size()!=b.size())return false;
 for(std::size_t i=0;i<a.size();++i){if(a[i].size()!=b[i].size())return false;
  for(std::size_t j=0;j<a[i].size();++j){auto x=a[i][j],y=b[i][j];
   if(x.kind!=y.kind||x.source!=y.source||x.observer!=y.observer||x.key!=y.key)return false;}}
 return true;
}
void require(bool b){if(!b){std::cerr<<"M6 benchmark semantic comparison failed\n";std::exit(2);}}
Run replay(const o::Program&p,const std::vector<o::Commands>&plans,bool incremental){
 Run out;
 std::unique_ptr<o::ReplaySession> session;
 if(incremental){auto setup=Clock::now();session=std::make_unique<o::ReplaySession>(p);out.ms+=std::chrono::duration<double,std::milli>(Clock::now()-setup).count();}
 for(const auto&plan:plans){auto start=Clock::now();auto r=incremental?session->analyze(plan,{false}):o::analyze(p,plan,{false});out.ms+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  require(r.complete);out.evaluations+=r.stats.siteEvaluations;out.work+=r.stats.work;auto s=incremental?session->lastReplay():o::ReplayStats{};out.reused+=s.reusedSites;
  out.hot+=s.kind==o::ReplayStats::Incremental||s.kind==o::ReplayStats::Unchanged;
  out.full+=s.kind!=o::ReplayStats::Incremental&&s.kind!=o::ReplayStats::Unchanged;
  // Equality serialization is intentionally outside the timed interval.
  out.results.push_back(oahs_test::report(r));}
 return out;
}
void replayCase(const std::string&name,const o::Program&p,const std::vector<o::Commands>&plans){
 std::vector<double> coldTimes,hotTimes;Run cold,hot;
 for(unsigned repeat=0;repeat<5;++repeat){if(repeat%2){hot=replay(p,plans,true);cold=replay(p,plans,false);}else{cold=replay(p,plans,false);hot=replay(p,plans,true);}
  require(cold.results==hot.results);coldTimes.push_back(cold.ms);hotTimes.push_back(hot.ms);}
 std::sort(coldTimes.begin(),coldTimes.end());std::sort(hotTimes.begin(),hotTimes.end());
 std::cout<<"{\"case\":\""<<name<<"\",\"phases\":"<<p.operations.size()<<",\"plans\":"<<plans.size()
 <<",\"samples\":5,\"cold_ms_median\":"<<coldTimes[2]<<",\"incremental_ms_median\":"<<hotTimes[2]
 <<",\"cold_site_evaluations\":"<<cold.evaluations<<",\"incremental_site_evaluations\":"<<hot.evaluations
 <<",\"reused_sites\":"<<hot.reused<<",\"cold_work\":"<<cold.work<<",\"incremental_work\":"<<hot.work
 <<",\"incremental_or_cached_calls\":"<<hot.hot<<",\"full_calls\":"<<hot.full<<",\"reports_equal\":true,\"session_setup_included\":true,\"cold_ms_samples\":[";
 for(unsigned i=0;i<coldTimes.size();++i){if(i)std::cout<<',';std::cout<<coldTimes[i];}
 std::cout<<"],\"incremental_ms_samples\":[";for(unsigned i=0;i<hotTimes.size();++i){if(i)std::cout<<',';std::cout<<hotTimes[i];}std::cout<<"]}\n";
}
int main(){
 {auto p=observed_fixtures::target(3,240,2);for(unsigned i=0;i<240;++i)p.operations.push_back(observed_fixtures::op(i%3,i,2));
  o::Commands c(241);c[0]=phase_fixture::handoff(0,1);std::vector<o::Commands> plans{c};
  for(unsigned n=0;n<40;++n){auto at=220+n%18;c[at].clear();if(n%3)c[at].push_back({o::Command::Barrier,o::Pipe(n%3)});plans.push_back(c);}
  replayCase("late_edits_acyclic",p,plans);}
 {auto p=observed_fixtures::target(3,45,2);for(unsigned i=0;i<45;++i)p.operations.push_back(observed_fixtures::op(i%3,i,2));
  o::Region body;for(unsigned i=10;i<40;++i)body.children.push_back(observed_fixtures::leaf(i));
  o::Region all;for(unsigned i=0;i<10;++i)all.children.push_back(observed_fixtures::leaf(i));all.children.push_back({o::Region::For,{body},0,true});
  for(unsigned i=40;i<45;++i) { all.children.push_back(observed_fixtures::leaf(i)); }
  p.body=all;
  o::Commands c(46);c[0]=phase_fixture::handoff(0,1);std::vector<o::Commands> plans{c};
  for(unsigned n=0;n<24;++n){c[35].clear();if(n%2)c[35]={{o::Command::Barrier,o::Pipe(2)}};plans.push_back(c);}
  replayCase("loop_backedge_invalidation",p,plans);}
 {auto p=phase_fixture::program(2);p.operations={phase_fixture::producer(p),phase_fixture::consumer(p)};phase_fixture::loop(p);
  o::Commands c(3);std::vector<o::Commands> plans;
  for(unsigned n=0;n<12;++n){c[1].clear();if(n%2)c[1]={{o::Command::Barrier,o::Pipe(phase_fixture::F)}};plans.push_back(c);}
  replayCase("phase_full_replay_control",p,plans);}
 {auto p=observed_fixtures::target(3,20,3);for(unsigned i=0;i<20;++i)p.operations.push_back(observed_fixtures::op(i%3,i,2));
  o::Commands c(21);std::vector<o::Commands> plans;
  for(unsigned n=0;n<24;++n){c[15]=phase_fixture::handoff(0,1,n%3);plans.push_back(c);}
  replayCase("changed_key_layout_control",p,plans);}
 for(unsigned n:{12u,24u}){
  auto p=observed_fixtures::target(3,n,2);p.target.barrierAll=true;
  for(unsigned i=0;i<n/2;++i)p.operations.push_back(observed_fixtures::op(0,i,2));
  for(unsigned i=0;i<n/2;++i)p.operations.push_back(observed_fixtures::op(1,i,1));
  std::vector<double> coldTimes,hotTimes;o::Result cold,hot;
  for(unsigned rep=0;rep<3;++rep){auto run=[&](bool inc,o::Result &out,std::vector<double>&times){auto start=Clock::now();out=o::construct(p,{true,inc,true});times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());};
    if(rep%2){run(true,hot,hotTimes);run(false,cold,coldTimes);}else{run(false,cold,coldTimes);run(true,hot,hotTimes);}
    require(cold.success&&hot.success);require(sameWords(cold.commands,hot.commands));require(oahs_test::report(o::analyze(p,cold.commands))==oahs_test::report(o::analyze(p,hot.commands)));}
  std::sort(coldTimes.begin(),coldTimes.end());std::sort(hotTimes.begin(),hotTimes.end());
  std::cout<<"{\"case\":\"constructor_fanin_"<<n<<"\",\"samples\":3,\"cold_ms_median\":"<<coldTimes[1]<<",\"incremental_ms_median\":"<<hotTimes[1]
   <<",\"cold_proposal_evaluations\":"<<cold.work.provisionalSiteEvaluations<<",\"incremental_proposal_evaluations\":"<<hot.work.provisionalSiteEvaluations
   <<",\"cold_candidate_evaluations\":"<<cold.work.replaySiteEvaluations<<",\"incremental_candidate_evaluations\":"<<hot.work.replaySiteEvaluations
   <<",\"bundle_trials\":"<<hot.bundleTrials<<",\"reports_equal\":true,\"commands_equal\":true,\"cold_ms_samples\":[";
  for(unsigned i=0;i<coldTimes.size();++i){if(i)std::cout<<',';std::cout<<coldTimes[i];}
  std::cout<<"],\"incremental_ms_samples\":[";for(unsigned i=0;i<hotTimes.size();++i){if(i)std::cout<<',';std::cout<<hotTimes[i];}std::cout<<"]}\n";
 }
}
