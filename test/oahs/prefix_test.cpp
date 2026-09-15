// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "GraphOracle.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <tuple>
namespace o = mlir::pto::oahs;
namespace {
std::size_t checks=0, graphChecks=0, coverageChecks=0, candidateChecks=0, nonRearming=0, snapshotChecks=0;
void check(bool value, unsigned line) {
  ++checks;
  if (!value) { std::cerr << "prefix check failed at " << line << '\n'; std::abort(); }
}
#define CHECK(x) check(bool(x), __LINE__)
o::Program base() {
  o::Program p; p.cells.resize(6);
  p.target.contract="test-only independent issue-ordered engines";
  p.target.barrierAll=true;
  for (unsigned a=0;a<3;++a) {
    p.target.supported[a]=p.target.barriers[a]=true;
    for (unsigned b=0;b<3;++b) if(a!=b) p.target.keys[a][b]={0,1,2,3,4,5};
  }
  return p;
}
o::Operation op(unsigned pipe,unsigned cell,bool read=false,bool write=true) {
  o::Operation a; a.pipe=o::Pipe(pipe); a.complete=true;
  a.accesses={{cell,read,write}}; return a;
}
o::Operation empty(unsigned pipe) { auto a=op(pipe,0); a.accesses.clear(); return a; }
o::Region leaf(unsigned id) { return {o::Region::Operation,{},id}; }
o::Region seq(std::initializer_list<o::Region> cs) { return {o::Region::Sequence,cs}; }
o::Region loop(o::Region body) { return {o::Region::For,{std::move(body)}}; }
o::Command pub(unsigned a,unsigned b,unsigned key=0) { return {o::Command::Publish,o::Pipe(a),o::Pipe(b),key}; }
o::Command wait(unsigned a,unsigned b,unsigned key=0) { return {o::Command::Acquire,o::Pipe(a),o::Pipe(b),key}; }
const o::ProspectivePrefix &selected(const o::PrefixCover &cover) {
  CHECK(!cover.selected.empty()); return cover.candidates[cover.selected.front()];
}
void oraclePlan(const o::Program &p,const o::Commands &commands,unsigned trips=3,
                const std::vector<std::pair<unsigned,unsigned>> &forbidden={}) {
  for (const auto &trace:oahs_oracle::traces(p,trips)) {
    CHECK(oahs_oracle::graph(p,commands,trace,forbidden)); ++graphChecks;
  }
}
// Independent reference: materialize one VIRTUAL handoff with split endpoints.
// Its fresh test key has no allocation restriction. Reference alternation and
// payload coverage are checked; causal rearm is deliberately NOT inferred from
// a prospective prefix and can fail until actual event realization adds a return.
void oraclePrefix(const o::Program &p,const o::Commands &commands,
                  const o::ProspectivePrefix &prefix,
                  const std::vector<o::CompletionRequirement> &requirements,unsigned trips=2) {
  if (!prefix.matchingEstablished) return;
  ++candidateChecks;
  o::Commands actual=commands;
  const auto a=unsigned(prefix.source), b=unsigned(prefix.observer);
  if (a==b) {
    if (prefix.publication!=prefix.acquisition) return;
    actual[prefix.acquisition].push_back({o::Command::Barrier,prefix.source});
  } else {
    actual[prefix.publication].insert(actual[prefix.publication].begin(),pub(a,b,1000));
    actual[prefix.acquisition].push_back(wait(a,b,1000));
  }
  for (const auto &trace:oahs_oracle::traces(p,trips)) {
    oahs_oracle::PrefixProbe observation;
    observation.source=prefix.source;
    observation.publication=prefix.publication;
    observation.acquisition=prefix.acquisition;
    for (auto i:prefix.coveredRequirements) observation.claims.push_back(requirements[i].demand);
    const auto old=oahs_oracle::graph(p,commands,trace,{},nullptr,&observation);
    if (!old.balanced || !old.acyclic || !old.rearm) continue;
    CHECK(observation.matching); CHECK(observation.coverage); ++snapshotChecks;
    std::vector<oahs_oracle::UncoveredConflict> missing;
    const auto now=oahs_oracle::graph(p,actual,trace,{},&missing);
    CHECK(now.balanced); CHECK(now.acyclic); nonRearming+=!now.rearm;
    ++graphChecks;
    for (auto component:prefix.coveredRequirements) {
      CHECK(component<requirements.size());
      const auto &d=requirements[component].demand;
      for (const auto &conflict:missing) {
        CHECK(!(trace[conflict.producerVisit]==d.producer &&
                trace[conflict.consumerVisit]==d.consumer && conflict.cell==d.cell));
        ++coverageChecks;
      }
    }
  }
}
void validateCover(const o::Program &p,const o::Commands &commands,const o::PrefixCover &cover,unsigned trips=2) {
  CHECK(cover.complete); CHECK(cover.backward.complete);
  std::vector<bool> covered(cover.requirements.size());
  for (const auto &candidate:cover.candidates) oraclePrefix(p,commands,candidate,cover.requirements,trips);
  for (auto index:cover.selected) {
    CHECK(index<cover.candidates.size()); CHECK(cover.candidates[index].selectable());
    for (auto c:cover.candidates[index].coveredRequirements) covered[c]=true;
  }
  for (std::size_t i=0;i<covered.size();++i) {
    CHECK(covered[i]==!bool(cover.jointUncoveredOperations[cover.requirements[i].demand.producer]));
    CHECK(!covered[i]==(std::find(cover.remaining.begin(),cover.remaining.end(),i)!=cover.remaining.end()));
  }
}
}
int main() {
  auto p=base();
  p.operations={op(0,0),op(0,1),op(1,0,true,false)};
  const o::Commands none(p.operations.size()+1);
  o::PrefixQuery early(p,none);
  const auto cover=early.coverByPrefixes(2);
  CHECK(cover.coversAll()); CHECK(cover.selected.size()==1);
  CHECK(selected(cover).source==o::Pipe(0)); CHECK(selected(cover).publication==1);
  CHECK(!selected(cover).uncoveredOperations[0]);
  CHECK(selected(cover).uncoveredOperations[1]);
  validateCover(p,none,cover);
  const auto beforeProducer=early.inspectPrefix(o::Pipe(0),0,2);
  CHECK(beforeProducer.matchingEstablished); CHECK(beforeProducer.coveredRequirements.empty());
  const auto late=early.inspectPrefix(o::Pipe(0),2,2);
  CHECK(late.matchingEstablished); CHECK(!late.uncoveredOperations[0] && !late.uncoveredOperations[1]);
  CHECK(!early.inspectPrefix(o::Pipe::Count,1,2).complete);
  CHECK(!early.backwardCuts(99).complete);
  CHECK(!early.coverByPrefixes(99).complete);
  CHECK(!early.inspectPrefix(o::Pipe(0),99,2).complete);
  auto plan=o::construct(p);
  CHECK(plan.success); oraclePlan(p,plan.commands,2,{{1,2}});

  // Irrelevant structured nodes, and real unrelated work in either choice arm,
  // do not remove the early ordinary cut or force a recognized episode.
  p.operations.push_back(op(2,3)); p.operations.push_back(op(2,4));
  p.body=seq({leaf(0),leaf(1),{o::Region::Choice,{leaf(3),leaf(4)}},leaf(2)});
  o::PrefixQuery structured(p);
  auto throughChoice=structured.coverByPrefixes(2);
  CHECK(throughChoice.coversAll()); CHECK(selected(throughChoice).publication==1);
  validateCover(p,o::Commands(6),throughChoice);
  plan=o::construct(p); CHECK(plan.success);
  for (const auto &t:oahs_oracle::traces(p,2)) {
    CHECK(oahs_oracle::graph(p,plan.commands,t,{{1,unsigned(t.size()-1)}})); ++graphChecks;
  }

  // Ordinary early placement survives an enclosing loop when the existing
  // memory-release cycle supplies event recurrence. No episode provider exists.
  p=base(); p.operations={op(0,0),op(0,1),op(1,0,true,false)};
  p.body=loop(seq({leaf(0),leaf(1),leaf(2)}));
  plan=o::construct(p); CHECK(plan.success);
  bool earlyLoop=false;
  for(const auto &h:plan.handoffs)
    earlyLoop |= h.source==o::Pipe(0) && h.observer==o::Pipe(1) &&
                 h.publication==1 && h.acquisition==2;
  CHECK(earlyLoop);
  for(const auto &t:oahs_oracle::traces(p,4)) {
    const auto forbidden=t.empty() ? std::vector<std::pair<unsigned,unsigned>>{} :
                                   std::vector<std::pair<unsigned,unsigned>>{{1,2}};
    CHECK(oahs_oracle::graph(p,plan.commands,t,forbidden)); ++graphChecks;
  }

  // Same-arm source and target: finite original participation certifies early
  // placement. A source in a different arm never becomes an unconditional wait.
  p=base(); p.operations={op(0,0),op(0,1),op(1,0,true,false),op(2,4)};
  p.body={o::Region::Choice,{seq({leaf(0),leaf(1),leaf(2)}),leaf(3)}};
  o::PrefixQuery branch(p);
  auto local=branch.coverByPrefixes(2);
  CHECK(local.coversAll()); CHECK(selected(local).publication==1);
  CHECK(selected(local).sourceContext==selected(local).targetContext);
  CHECK(!branch.inspectPrefix(o::Pipe(0),3,2).matchingEstablished);
  validateCover(p,o::Commands(5),local);
  plan=o::construct(p); CHECK(plan.success); oraclePlan(p,plan.commands);
  p.body=seq({{o::Region::Choice,{seq({leaf(0),leaf(1)}),leaf(3)}},leaf(2)});
  o::PrefixQuery optional(p);
  CHECK(!optional.inspectPrefix(o::Pipe(0),1,2).availableAtEveryAcquisition);
  CHECK(optional.inspectPrefix(o::Pipe(0),2,2).matchingEstablished);

  // Independent fan-in is a conjunction. No numeric maximum, no last-writer
  // compression across engines, and no scalar union interpreted as one receipt.
  p=base();p.operations={op(0,0),op(2,1),op(1,0,true,false)};
  p.operations[2].accesses.push_back({1,true,false});
  o::PrefixQuery fanin(p); auto pair=fanin.coverByPrefixes(2);
  CHECK(pair.coversAll()); CHECK(pair.selected.size()==2);
  for(auto i:pair.selected) CHECK(pair.candidates[i].coveredRequirements.size()==1);
  validateCover(p,o::Commands(4),pair);
  plan=o::construct(p); CHECK(plan.success); oraclePlan(p,plan.commands);

  // A REAL acquired receipt on a relay can cover both independent components.
  p=base();p.operations={op(0,0),op(2,1),op(0,4),op(1,0,true,false)};
  p.operations[3].accesses.push_back({1,true,false});
  o::Commands relay(5); relay[1]={pub(0,2),wait(0,2)};
  o::PrefixQuery acquired(p,relay);auto together=acquired.coverByPrefixes(3);
  CHECK(together.coversAll()); CHECK(together.selected.size()==1);
  CHECK(selected(together).source==o::Pipe(2)); CHECK(selected(together).publication==2);
  CHECK(selected(together).coveredRequirements.size()==2);
  validateCover(p,relay,together);
  // The real constructor obtains the same relay from a required first
  // consumer, instead of merely exposing an unused query API.
  auto relayProgram=p;
  relayProgram.operations[1].accesses.push_back({0,true,false});
  const auto relayPlan=o::construct(relayProgram);
  CHECK(relayPlan.success); CHECK(relayPlan.handoffs.size()==2);
  bool selectedRelay=false;
  for(const auto &h:relayPlan.handoffs)
    selectedRelay |= h.source==o::Pipe(2) && h.observer==o::Pipe(1) &&
                     h.publication==2 && h.acquisition==3;
  CHECK(selectedRelay); oraclePlan(relayProgram,relayPlan.commands,2,{{2,3}});
  // Snapshot before the cut's existing acquisition has NOT learned that fact.
  CHECK(acquired.inspectPrefix(o::Pipe(2),1,3).coveredRequirements.empty());
  // An input edit cannot silently change this immutable query's meaning.
  relay[1].clear(); o::PrefixQuery notAcquired(p,relay);
  CHECK(notAcquired.coverByPrefixes(3).selected.size()==2);
  CHECK(acquired.coverByPrefixes(3).selected.size()==1);
  // Invalid receipts cannot be laundered into a relay source snapshot.
  relay[1]={wait(0,2)}; o::PrefixQuery invalidRelay(p,relay);
  CHECK(!invalidRelay.analysis().protocol.empty());
  auto incomplete=invalidRelay.inspectPrefix(o::Pipe(2),2,3);
  CHECK(incomplete.matchingEstablished);
  CHECK(incomplete.uncoveredOperations[0]); // P is not completed by the unseeded wait.
  CHECK(!incomplete.uncoveredOperations[1]); // relay's own earlier payload is covered.

  // Fresh visits of the SAME static write extend a retained prefix remainder.
  p=base();p.operations={op(0,0),op(1,0,true,false)};
  p.body=loop(seq({leaf(0),leaf(1)}));
  o::PrefixQuery freshness(p);
  auto stale=freshness.inspectPrefix(o::Pipe(0),0,1);
  CHECK(stale.matchingEstablished); CHECK(stale.uncoveredOperations[0]);
  CHECK(stale.coveredRequirements.empty());
  auto fresh=freshness.inspectPrefix(o::Pipe(0),1,1);
  CHECK(fresh.matchingEstablished); CHECK(!fresh.coveredRequirements.empty());
  CHECK(fresh.crossesBackedge);
  oraclePrefix(p,o::Commands(3),fresh,freshness.consumerRequirements(1),4);
  const auto backwards=freshness.backwardCuts(0);
  CHECK(!backwards.crossedLoopOwners.empty());
  const auto previous=std::find_if(backwards.cuts.begin(),backwards.cuts.end(),[](const o::BackwardCut &c){return c.cut==1;});
  CHECK(previous!=backwards.cuts.end()); CHECK(previous->throughBackedge && !previous->withoutBackedge);
  CHECK(!freshness.inspectPrefix(o::Pipe(1),1,0).matchingEstablished); // unprimed prior generation
  auto multiple=base();multiple.operations={op(0,0),op(0,0),op(1,0,true,false)};
  o::PrefixQuery rewriting(multiple); auto old=rewriting.inspectPrefix(o::Pipe(0),1,2);
  CHECK(old.matchingEstablished); CHECK(old.uncoveredOperations[1]);
  CHECK(old.coveredRequirements.size()==1);

  // Source outside a zero-trip inner loop: bypass leaks the hypothetical event;
  // additional trips consume it twice. Neither is a valid one-to-one frontier.
  p=base();p.operations={op(0,0),empty(0),op(1,0,true,false)};
  p.body=seq({leaf(0),leaf(1),loop(leaf(2))});
  o::PrefixQuery manyReaders(p); auto bad=manyReaders.inspectPrefix(o::Pipe(0),1,2);
  CHECK(!bad.matchingEstablished); CHECK(bad.unconsumedAtExit);
  p.body=seq({leaf(0),loop(leaf(1)),leaf(2)});
  o::PrefixQuery manyPublishers(p); bad=manyPublishers.inspectPrefix(o::Pipe(0),1,2);
  CHECK(!bad.matchingEstablished); CHECK(bad.repeatedPublication);

  // Two readers, including two on the same issue-only engine, remain distinct.
  p=base();p.operations={op(0,0,true,false),op(0,0,true,false),op(2,0,true,false),op(1,0)};
  o::PrefixQuery readers(p); auto releases=readers.coverByPrefixes(3);
  CHECK(releases.requirements.size()==3); CHECK(releases.selected.size()==2);
  CHECK(releases.coversAll()); validateCover(p,o::Commands(5),releases);
  auto earlyReader=readers.inspectPrefix(o::Pipe(0),1,3);
  CHECK(earlyReader.coveredRequirements.size()==1 && earlyReader.uncoveredOperations[1]);

  // Keys/capabilities are eligibility, not analysis budgets or allocations.
  p=base();p.operations={op(0,0),op(1,0,true,false)};
  p.target.keys[0][1].clear(); o::PrefixQuery noKeys(p);
  CHECK(noKeys.inspectPrefix(o::Pipe(0),1,1).matchingEstablished);
  CHECK(!noKeys.coverByPrefixes(1).coversAll());
  p.target.keys[0][1]={0};p.reservations={{o::Pipe(0),o::Pipe(1),0,true}};
  o::PrefixQuery reserved(p);CHECK(!reserved.coverByPrefixes(1).coversAll());
  p.reservations.clear(); o::PrefixQuery unreserved(p); CHECK(unreserved.coverByPrefixes(1).coversAll());
  p.operations[1].pipe=o::Pipe(0);o::PrefixQuery sameLane(p);
  CHECK(sameLane.coverByPrefixes(1).coversAll());
  p.target.barriers[0]=false;o::PrefixQuery noFence(p);CHECK(!noFence.coverByPrefixes(1).coversAll());
  p.operations[0].complete=false;o::PrefixQuery unsupported(p);
  CHECK(!unsupported.analysis().complete && !unsupported.coverByPrefixes(1).complete);
  p.operations[0].complete=true;o::PrefixQuery malformed(p,o::Commands(2));
  CHECK(!malformed.analysis().complete && !malformed.backwardCuts(1).complete);
  p=base();o::PrefixQuery emptyProgram(p);CHECK(emptyProgram.analysis().verified());
  CHECK(!emptyProgram.coverByPrefixes(0).complete);

  // Original order is not assumed to be the numerical order of phase IDs.
  p=base();p.operations={op(1,0,true,false),empty(1),op(0,0),op(0,1)};
  p.body=seq({leaf(2),leaf(3),leaf(0),leaf(1)});
  o::PrefixQuery reordered(p);auto reorderedCover=reordered.coverByPrefixes(0);
  CHECK(reorderedCover.coversAll()); CHECK(selected(reorderedCover).publication==3);
  plan=o::construct(p); CHECK(plan.success);oraclePlan(p,plan.commands,2,{{1,2}});

  // While-before always executes; an after-only publication cannot supply the
  // first before-region acquisition. Nested loop contexts keep separate owners.
  p=base();p.operations={op(1,0,true,false),op(0,0),empty(2)};
  p.body={o::Region::While,{leaf(0),seq({leaf(1),loop(leaf(2))})}};
  o::PrefixQuery whileQuery(p);CHECK(!whileQuery.inspectPrefix(o::Pipe(0),1,0).matchingEstablished);
  const auto whileBack=whileQuery.backwardCuts(0);CHECK(whileBack.crossedLoopOwners.size()==2);
  validateCover(p,o::Commands(4),whileQuery.coverByPrefixes(0));

  // M1 multiple-writer example and its nested variants, without ANY recognizer.
  p=base();p.operations={op(0,0),op(1,0,true,false),op(2,0,true,true),op(1,0,true,false)};
  auto body=seq({leaf(0),{o::Region::Choice,{leaf(1),leaf(2)}},leaf(3)});
  p.body=loop(body);
  o::PrefixQuery writers(p);
  for(unsigned i=0;i<4;++i)validateCover(p,o::Commands(5),writers.coverByPrefixes(i));
  plan=o::construct(p);CHECK(plan.success);oraclePlan(p,plan.commands,3);
  p.body={o::Region::While,{seq({}),loop(body)}};
  o::PrefixQuery nestedWriters(p);
  for(unsigned i=0;i<4;++i)validateCover(p,o::Commands(5),nestedWriters.coverByPrefixes(i),1);
  plan=o::construct(p);CHECK(plan.success);oraclePlan(p,plan.commands,1);

  // Nesting changes static state count, not dynamic occurrence enumeration.
  p=base();p.operations={op(0,0),op(1,0,true,false)};
  p.body=seq({leaf(0),leaf(1)});
  for(unsigned depth=0;depth<128;++depth)p.body=loop(std::move(p.body));
  o::PrefixQuery deep(p);auto deepCandidate=deep.inspectPrefix(o::Pipe(0),1,1);
  CHECK(deepCandidate.matchingEstablished); CHECK(deepCandidate.siteEvaluations<8*deep.analysis().stats.staticSites);
  CHECK(deep.backwardCuts(1).visitedStates<=2*deep.analysis().stats.staticSites);

  // Seeded finite differential checks against the concrete command graph.
  std::mt19937 rng(20260916);
  for(unsigned sample=0;sample<120;++sample) {
    p=base();
    for(unsigned i=0;i<5;++i) {
      const unsigned mode=rng()%3;
      // Sequence draws explicitly: GCC and Clang must test the same population.
      const unsigned cell=rng()%3;
      const unsigned pipe=rng()%3;
      p.operations.push_back(op(pipe,cell,mode!=0,mode!=1));
    }
    if(sample%4==0)p.body=loop(seq({leaf(0),{o::Region::Choice,{leaf(1),leaf(2)}},leaf(3),leaf(4)}));
    else if(sample%4==1)p.body={o::Region::While,{seq({leaf(0),leaf(1)}),seq({leaf(2),leaf(3),leaf(4)})}};
    else if(sample%4==2)p.body=seq({leaf(0),{o::Region::Choice,{seq({leaf(1),leaf(2)}),leaf(3)}},leaf(4)});
    o::Commands original(6);
    // A partial plan containing a genuine direct handoff on acyclic inputs.
    if(sample%4==3 && p.operations[0].pipe!=p.operations[1].pipe) {
      original[1]={{o::Command::Publish,p.operations[0].pipe,p.operations[1].pipe,0},
                   {o::Command::Acquire,p.operations[0].pipe,p.operations[1].pipe,0}};
    }
    o::PrefixQuery query(p,original);
    CHECK(query.analysis().complete);
    for(unsigned consumer=0;consumer<5;++consumer)
      validateCover(p,original,query.coverByPrefixes(consumer),2);
    const auto built=o::construct(p); CHECK(built.success); oraclePlan(p,built.commands,2);
  }
  CHECK(nonRearming>0);
  std::cout<<"OAHS prefixes: "<<checks<<" assertions, "<<candidateChecks<<" prospective candidates, "
           <<graphChecks<<" graph traces, "<<snapshotChecks<<" immutable-source probes, "<<coverageChecks<<" residual comparisons, "
           <<nonRearming<<" virtual traces deliberately not certified for rearm\n";
}
