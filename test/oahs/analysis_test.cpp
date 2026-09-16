// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "PTO/Transforms/OAHS/Analysis.h"
#include "GraphOracle.h"
#include "../../lib/PTO/Transforms/OAHS/Transfer.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
namespace o = mlir::pto::oahs;
namespace {
unsigned checks = 0, traces = 0, mutations = 0, residualTraces = 0, residualWitnesses = 0;
void check(bool value, unsigned line) {
  ++checks;
  if (!value) { std::cerr << "analysis check failed at " << line << '\n'; std::abort(); }
}
#define CHECK(x) check(bool(x), __LINE__)
o::Program base() {
  o::Program p; p.cells.resize(3);
  p.target.contract = "test-only independent issue-ordered engines";
  p.target.barrierAll = true;
  for (unsigned a=0; a<3; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b=0; b<3; ++b) if (a!=b) p.target.keys[a][b] = {0,1,2};
  }
  return p;
}
o::Operation op(unsigned pipe, unsigned cell, bool read, bool write) {
  o::Operation a; a.pipe=o::Pipe(pipe); a.accesses={{cell,read,write}}; a.complete=true; return a;
}
o::Region leaf(unsigned id) { return {o::Region::Operation,{},id}; }
o::Region seq(std::initializer_list<o::Region> children) { return {o::Region::Sequence,children}; }
bool has(const o::AnalysisResult &r, unsigned from, unsigned to, o::CompletionRequirement::Kind kind) {
  for (const auto &d:r.residuals)
    if (d.demand.producer==from && d.demand.consumer==to && d.kind==kind) return true;
  return false;
}
bool protocol(const o::AnalysisResult &r, o::ProtocolObligation::Kind kind) {
  for (const auto &d:r.protocol) if(d.kind==kind) return true;
  return false;
}
void oracle(const o::Program &p, const o::Commands &commands, unsigned trips=3) {
  for (const auto &t:oahs_oracle::traces(p,trips)) {
    CHECK(oahs_oracle::graph(p,commands,t)); ++traces;
  }
}
void accepted(const o::Program &p, const o::Commands &commands, unsigned trips=3) {
  const auto r=o::analyze(p,commands);
  CHECK(r.complete); CHECK(r.verified()); CHECK(o::verify(p,commands).success);
  CHECK(r.stats.certificationPasses==1); CHECK(r.stats.suppressedCommands==0);
  for (const auto &c:r.commands) CHECK(c.preconditionsEstablished);
  oracle(p,commands,trips);
}
// Check residual completeness against the independent finite command graph,
// not the abstract interpreter or a desired-edge-augmented graph. Only protocol-
// legal reference graphs are used to interpret uncovered payload order.
void residualOracle(const o::Program &p, const o::Commands &commands,
                    const o::AnalysisResult &report) {
  for (const auto &t:oahs_oracle::traces(p,2)) {
    std::vector<oahs_oracle::UncoveredConflict> missing;
    const auto verdict=oahs_oracle::graph(p,commands,t,{},&missing);
    if (!verdict.acyclic || !verdict.balanced || !verdict.rearm) continue;
    ++residualTraces;
    for (const auto &d:missing) {
      const auto source=t[d.producerVisit], target=t[d.consumerVisit];
      for (const auto &a:p.operations[source].accesses)
        for (const auto &b:p.operations[target].accesses) {
          if (a.cell!=d.cell || b.cell!=d.cell) continue;
          auto contains=[&](o::CompletionRequirement::Kind kind) {
            for (const auto &r:report.residuals)
              if (r.kind==kind && r.demand.producer==source &&
                  r.demand.consumer==target && r.demand.cell==d.cell) return true;
            return false;
          };
          if (a.write && b.read) CHECK(contains(o::CompletionRequirement::RAW));
          if (a.read && b.write) CHECK(contains(o::CompletionRequirement::WAR));
          if (a.write && b.write) CHECK(contains(o::CompletionRequirement::WAW));
          if (p.cells[d.cell].exclusive) CHECK(contains(o::CompletionRequirement::ExclusiveResource));
          ++residualWitnesses;
        }
    }
  }
}
void compare(const o::Program &p, const o::Commands &commands) {
  const auto report=o::analyze(p,commands,{false});
  const auto first=o::detail::Transfer(p,commands).run();
  CHECK(report.complete);
  CHECK(report.verified()==(first.kind==o::detail::Failure::None));
  for (const auto &cut:report.cuts)
    CHECK(!cut.incoming && !cut.beforeIssue && !cut.outgoing);
  std::size_t endpoints=0;
  for (const auto &at:commands) for (const auto &c:at)
    endpoints += c.kind==o::Command::Publish || c.kind==o::Command::Acquire;
  CHECK(report.stats.certificationPasses<=endpoints+1);
  if (report.verified()) oracle(p,commands,2);
  residualOracle(p,commands,report);
  ++mutations;
}
}
int main() {
  // Full residual collection, not just the first consumer. Fresh writes neither
  // kill readers nor establish completion. Read/read is not a memory conflict.
  auto p=base();
  p.operations={op(0,0,false,true),op(1,0,true,false),op(2,0,true,true),op(1,0,true,false)};
  auto empty=o::analyze(p);
  CHECK(empty.complete && !empty.verified()); CHECK(empty.protocol.empty());
  CHECK(has(empty,0,1,o::CompletionRequirement::RAW));
  CHECK(has(empty,0,2,o::CompletionRequirement::RAW));
  CHECK(has(empty,0,2,o::CompletionRequirement::WAW));
  CHECK(has(empty,1,2,o::CompletionRequirement::WAR));
  CHECK(has(empty,2,3,o::CompletionRequirement::RAW));
  CHECK(!has(empty,1,3,o::CompletionRequirement::RAW));
  CHECK(empty.cuts[3].beforeIssue->pending[1][0]);
  CHECK(empty.cuts[3].beforeIssue->pending[1][1]);
  auto plan=o::constructSelectedPlan(p); CHECK(plan.success); accepted(p,plan.commands);

  // v0.5 section 7.5, with NO episode, rank or guard refinement prerequisites.
  p.body={o::Region::For,{seq({leaf(0),{o::Region::Choice,{seq({leaf(1)}),seq({leaf(2)})}},leaf(3)})},0,true};
  auto multiple=o::analyze(p);
  CHECK(multiple.complete && !multiple.verified());
  CHECK(has(multiple,0,1,o::CompletionRequirement::RAW));
  CHECK(has(multiple,0,2,o::CompletionRequirement::RAW));
  CHECK(has(multiple,2,3,o::CompletionRequirement::RAW));
  CHECK(has(multiple,1,0,o::CompletionRequirement::WAR));
  CHECK(has(multiple,2,0,o::CompletionRequirement::WAW));
  CHECK(has(multiple,0,0,o::CompletionRequirement::WAW));
  auto then=multiple.cuts[1].context, otherwise=multiple.cuts[2].context;
  CHECK(then!=otherwise);
  CHECK(multiple.contexts[then].kind==o::AnalysisContext::ThenArm);
  CHECK(multiple.contexts[otherwise].kind==o::AnalysisContext::ElseArm);
  CHECK(multiple.contexts[then].ownerSite==multiple.contexts[otherwise].ownerSite);
  CHECK(multiple.contexts[multiple.contexts[then].parent].kind==o::AnalysisContext::ForBody);
  plan=o::constructSelectedPlan(p); CHECK(plan.success); accepted(p,plan.commands,5);
  const auto reduced=o::analyze(p,plan.commands,{false}); CHECK(reduced.verified());

  // While before executes even when after does not; contexts must not be folded.
  auto w=base(); w.operations={op(0,0,false,true),op(1,0,true,false),op(2,0,true,false)};
  w.body=seq({{o::Region::While,{seq({leaf(0)}),seq({leaf(1)})}},leaf(2)});
  auto wr=o::analyze(w);
  CHECK(has(wr,0,2,o::CompletionRequirement::RAW));
  CHECK(wr.contexts[wr.cuts[0].context].kind==o::AnalysisContext::WhileBefore);
  CHECK(wr.contexts[wr.cuts[1].context].kind==o::AnalysisContext::WhileAfter);
  plan=o::constructSelectedPlan(w); CHECK(plan.success); accepted(w,plan.commands,4);

  // Independent fan-in and multiple readers remain separate components.
  auto fan=base(); fan.operations={op(0,0,false,true),op(1,1,false,true),op(2,0,true,false)};
  fan.operations[2].accesses.push_back({1,true,false});
  auto fr=o::analyze(fan);
  CHECK(has(fr,0,2,o::CompletionRequirement::RAW));
  CHECK(has(fr,1,2,o::CompletionRequirement::RAW));
  auto read=base(); read.operations={op(0,0,true,false),op(1,0,true,false),op(2,0,false,true)};
  auto rr=o::analyze(read);
  CHECK(has(rr,0,2,o::CompletionRequirement::WAR)); CHECK(has(rr,1,2,o::CompletionRequirement::WAR));
  read.cells[0].exclusive=true;
  CHECK(has(o::analyze(read),0,1,o::CompletionRequirement::ExclusiveResource));
  CHECK(!has(o::analyze(read),0,1,o::CompletionRequirement::RAW));

  // Unseeded waits supply neither completion nor acknowledgment evidence.
  auto bad=base(); bad.operations={op(0,0,false,true),op(1,0,true,false),op(2,0,true,false)};
  o::Commands missing(4);
  missing[1]={{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  missing[2]={{o::Command::Publish,o::Pipe(1),o::Pipe(2),0},
              {o::Command::Acquire,o::Pipe(1),o::Pipe(2),0}};
  auto br=o::analyze(bad,missing);
  CHECK(br.complete && !br.verified());
  CHECK(protocol(br,o::ProtocolObligation::AcquisitionNotFull));
  CHECK(has(br,0,1,o::CompletionRequirement::RAW));
  CHECK(has(br,0,2,o::CompletionRequirement::RAW)); // cannot launder a bad receipt
  CHECK(br.cuts[2].beforeIssue->pending[2][0]);
  CHECK(br.commands[1].preconditionsEstablished); // independent key remains usable
  CHECK(!br.commands[0].preconditionsEstablished);

  // A publication absent on one branch cannot cover a post-join acquisition.
  auto choice=base(); choice.operations={op(0,0,false,true),op(0,1,true,false),op(0,1,true,false),op(1,0,true,false)};
  choice.body=seq({leaf(0),{o::Region::Choice,{seq({leaf(1)}),seq({leaf(2)})}},leaf(3)});
  o::Commands optional(5);
  optional[1]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0}};
  optional[3]={{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  auto cr=o::analyze(choice,optional);
  CHECK(cr.complete && !cr.verified());
  CHECK(protocol(cr,o::ProtocolObligation::AcquisitionNotFull));
  CHECK(has(cr,0,3,o::CompletionRequirement::RAW));

  // A lexically consumed key is not causally reusable at the publisher. The
  // provisional second receipt cannot enter the public completion snapshots.
  auto reuse=base(); reuse.operations={op(0,0,false,true),op(0,0,false,true),op(1,0,true,false)};
  o::Commands unsafe(4);
  unsafe[1]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0},
             {o::Command::Acquire,o::Pipe(0),o::Pipe(1),0},
             {o::Command::Barrier,o::Pipe(0)}};
  unsafe[2]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0},
             {o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  auto ur=o::analyze(reuse,unsafe);
  CHECK(protocol(ur,o::ProtocolObligation::ConsumptionNotEstablished));
  CHECK(has(ur,1,2,o::CompletionRequirement::RAW));
  CHECK(ur.cuts[2].beforeIssue->pending[1][1]);
  CHECK(ur.stats.certificationPasses>=2);
  auto safe=unsafe;
  safe[1].insert(safe[1].end(),{{o::Command::Publish,o::Pipe(1),o::Pipe(0),0},
                              {o::Command::Acquire,o::Pipe(1),o::Pipe(0),0}});
  accepted(reuse,safe);

  // A reply before the read acknowledges consumption, NOT storage release.
  auto release=base();
  release.operations={op(0,0,false,true),op(1,0,true,false),op(0,0,false,true)};
  o::Commands reply(4);
  reply[1]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0},
            {o::Command::Acquire,o::Pipe(0),o::Pipe(1),0},
            {o::Command::Publish,o::Pipe(1),o::Pipe(0),0},
            {o::Command::Acquire,o::Pipe(1),o::Pipe(0),0}};
  auto beforeRead=o::analyze(release,reply);
  CHECK(beforeRead.protocol.empty());
  CHECK(has(beforeRead,1,2,o::CompletionRequirement::WAR));
  reply[2]={{o::Command::Publish,o::Pipe(1),o::Pipe(0),1},
            {o::Command::Acquire,o::Pipe(1),o::Pipe(0),1}};
  accepted(release,reply);

  // Reusing an interpreter object never retains a prior call's credit masks.
  o::detail::Transfer reusedInterpreter(bad,missing);
  CHECK(!reusedInterpreter.inspect({}).verified());
  CHECK(reusedInterpreter.run().kind==o::detail::Failure::Occupancy);
  CHECK(!reusedInterpreter.inspect({}).verified());

  // Original source snapshots are not enlarged by later source work; valid
  // transitive receipts do cover all components they actually acquired.
  auto fresh=base(); fresh.operations={op(0,0,false,true),op(0,0,false,true),op(1,0,true,false)};
  o::Commands old(4);
  old[1]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0},{o::Command::Barrier,o::Pipe(0)}};
  old[2]={{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  auto stale=o::analyze(fresh,old);
  CHECK(stale.protocol.empty()); CHECK(has(stale,1,2,o::CompletionRequirement::RAW));
  CHECK(stale.cuts[1].outgoing->events[0].uncoveredOperations[1]);
  auto transitive=base(); transitive.operations={op(0,0,false,true),op(1,0,true,false),op(2,0,true,false)};
  o::Commands chain(4);
  chain[1]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0},{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  chain[2]={{o::Command::Publish,o::Pipe(1),o::Pipe(2),0},{o::Command::Acquire,o::Pipe(1),o::Pipe(2),0}};
  accepted(transitive,chain);
  CHECK(!o::analyze(transitive,chain).cuts[2].beforeIssue->pending[2][0]);

  // Exit completion never consumes a full event; collect both kinds of unmet
  // invocation obligations, rather than returning only the first one.
  auto exit=base(); exit.operations={op(0,0,false,true),op(1,1,false,true)};
  exit.invocation.retirement=o::Program::InvocationContract::DrainAllAtReturn;
  o::Commands live(3); live[2]={{o::Command::Publish,o::Pipe(0),o::Pipe(1),0}};
  auto er=o::analyze(exit,live);
  CHECK(protocol(er,o::ProtocolObligation::UnconsumedAtExit)); CHECK(er.retirement.size()==2);
  live[2].push_back({o::Command::BarrierAll}); er=o::analyze(exit,live);
  CHECK(er.retirement.empty()); CHECK(protocol(er,o::ProtocolObligation::UnconsumedAtExit));

  // Unsupported semantics are not analyzed as empty effects; malformed streams
  // are rejected before indexing their pipelines, cells or command cuts.
  auto unsupported=base(); unsupported.operations={op(0,0,false,true)};
  unsupported.operations[0].resources.push_back({"private queue",true,false});
  auto noModel=o::analyze(unsupported);
  CHECK(!noModel.complete && !noModel.verified()); CHECK(noModel.diagnostics[0].operation==0);
  CHECK(noModel.diagnostics[0].kind==o::AnalysisDiagnostic::UnsupportedSemantics);
  auto malformed=base(); malformed.operations={op(0,0,true,false)};
  CHECK(!o::analyze(malformed,o::Commands{}).complete);
  o::Commands invalid(2); invalid[0]={{o::Command::Acquire,o::Pipe::Count,o::Pipe(0),0}};
  CHECK(!o::analyze(malformed,invalid).complete);
  malformed.operations[0].complete=false; CHECK(!o::analyze(malformed).complete);
  CHECK(o::analyze(base()).verified());

  // Nested structural tests exercise the service itself without any episode
  // premise, including loops around an effectful while-before region.
  for(unsigned depth:{1u,8u,32u,128u}) {
    auto deep=w;
    for(unsigned i=0;i<depth;++i) deep.body={o::Region::For,{deep.body},0,true};
    auto d=o::analyze(deep,o::Commands(deep.operations.size()+1),{false});
    CHECK(d.complete); CHECK(d.stats.staticSites<3*depth+20);
    CHECK(has(d,0,2,o::CompletionRequirement::RAW));
  }

  std::mt19937 random(20260916);
  for(unsigned sample=0;sample<240;++sample) {
    auto q=base();
    for(unsigned i=0;i<5;++i) {
      bool write=random()%2, readBit=!write || random()%2;
      // Fix draw order independently of function-argument evaluation order.
      // Preserve the population previously generated by GCC (right to left).
      const unsigned cell = random() % 3;
      const unsigned pipe = random() % 3;
      q.operations.push_back(op(pipe,cell,readBit,write));
    }
    if(sample%2)
      q.body=seq({leaf(0),{o::Region::For,{seq({leaf(1),
          {o::Region::Choice,{seq({leaf(2)}),seq({leaf(3)})}}})},0,true},leaf(4)});
    else
      q.body=seq({leaf(0),{o::Region::While,{seq({leaf(1)}),
          seq({{o::Region::Choice,{seq({leaf(2)}),seq({leaf(3)})}}})}},leaf(4)});
    auto result=o::constructSelectedPlan(q); CHECK(result.success);
    accepted(q,result.commands,2);
    compare(q,o::Commands(q.operations.size()+1));
    for(unsigned trial=0;trial<8;++trial) {
      auto changed=result.commands;
      auto cut=random()%changed.size();
      if(!changed[cut].empty()) {
        auto at=random()%changed[cut].size(); auto c=changed[cut][at];
        changed[cut].erase(changed[cut].begin()+at);
        if(trial%2) changed[random()%changed.size()].push_back(c);
      }
      compare(q,changed);
    }
  }
  std::cout<<checks<<" M1 assertions; "<<traces<<" concrete graph checks; "
           <<mutations<<" fixed-plan differential trials; "
           <<residualTraces<<" residual graph traces; "<<residualWitnesses<<" uncovered witnesses\n";
}
