// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only interchange driver. Links the actual production implementation.
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "PTO/Transforms/OAHS/Phases.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <map>
namespace o=mlir::pto::oahs;
using IDs=std::vector<unsigned>;
unsigned number() {
  unsigned x;
  if(!(std::cin>>x))throw std::runtime_error("invalid test input");
  return x;
}
IDs ids() {
  IDs a;
  auto n=number();
  while(n--)a.push_back(number());
  return a;
}
void str(const std::string&s) {
  std::cout<<std::quoted(s);
}
template<class T>void array(const std::vector<T>&a) {
  std::cout<<'[';
  bool first=true;
  for(auto x:a) {
    if(!first)std::cout<<',';
    first=false;
    std::cout<<uint64_t(x);
  }
  std::cout<<']';
}
void snapshot(const std::optional<o::BoundaryFacts>&s) {
  if(!s) {
    std::cout<<"null";
    return;
  }
  std::cout<<"{\"pending\":[";
  for(unsigned q=0;q<o::PipeCount;++q) {
    if(q)std::cout<<',';
    array(s->pending[q]);
  }
  std::cout<<"],\"events\":[";
  for(std::size_t e=0;e<s->events.size();++e) {
    if(e)std::cout<<',';
    auto &v=s->events[e];
    std::cout<<"{\"occupancy\":"<<unsigned(v.possibleOccupancy)<<",\"valid\":"<<v.receiptValidOnEveryPath<<",\"known\":"<<unsigned(v.consumptionKnownAt)<<",\"uncovered\":";
    array(v.uncoveredOperations);
    std::cout<<",\"carried\":";
    array(v.carriedConsumptions);
    std::cout<<'}';
  }
  std::cout<<"],\"resources\":[";
  for(std::size_t u=0;u<s->phaseResources.size();++u) {
    if(u)std::cout<<',';
    auto &v=s->phaseResources[u];
    std::cout<<"{\"cell\":"<<v.cell<<",\"permission\":"<<v.possiblePermission<<",\"completed\":";
    array(v.completedOperations);
    std::cout<<",\"carried\":";
    array(v.carriedConsumptions);
    std::cout<<'}';
  }
  std::cout<<"]}";
}
int main() {
  try {
    o::Program p;
    p.target.contract="reference-interchange-only";
    unsigned lanes=number();
    if(lanes>o::PipeCount)throw std::runtime_error("too many lanes");
    for(unsigned q=0;q<lanes;++q)p.target.supported[q]=p.target.barriers[q]=true;
    auto cells=number();
    while(cells--) {
      o::Cell c;
      uint64_t b,n;
      std::cin>>c.addressSpace>>b>>n;
      c.ranges= {
        {
          b,n
        }
      };
      p.cells.push_back(c);
    }
    o::FinalBlockProfile profile;
    profile.entryWritable=number();
    profile.orderedPerBlockService=number();
    profile.enabledPhaseProgress=number();
    profile.commonForwardControl=number();
    auto groups=number();
    while(groups--) {
      o::FinalBlockGroup g;
      std::cin>>g.name;
      g.producer=o::Pipe(number());
      g.consumer=o::Pipe(number());
      g.blocks=ids();
      profile.groups.push_back(g);
    }
    p.finalBlocks=profile;
    auto keys=number();
    while(keys--) {
      auto a=number(),b=number(),k=number();
      if(a>=lanes||b>=lanes)throw std::runtime_error("invalid lane");
      p.target.keys[a][b].push_back(k);
    }
    auto operations=number();
    while(operations--) {
      o::Operation op;
      op.pipe=o::Pipe(number());
      op.complete=true;
      auto reads=ids(),writes=ids();
      for(auto c:reads)op.accesses.push_back( {
        c,true,false,false
      });
      for(auto c:writes)op.accesses.push_back( {
        c,false,true,true
      });
      auto role=number();
      if (role > 2) throw std::runtime_error("invalid phase role tag");
      if(role) {
        o::FinalBlockOperation f;
        f.role=role==1?o::FinalBlockOperation::Role::Producer:o::FinalBlockOperation::Role::Consumer;
        f.group=number();
        f.blocks=ids();
        f.operands=ids();
        f.outputs=ids();
        op.finalBlock=f;
      }
      p.operations.push_back(op);
    }
    o::ObservedControl q;
    auto sites=number();
    q.entry=number();
    q.exit=number();
    q.qualification="explicit original control from pinned phase fixture";
    while(sites--) {
      long long op;
      std::cin>>op;
      o::ObservedSite site;
      site.operation=op<0?o::NoControlId:std::size_t(op);
      site.observation=q.observations.size();
      site.successors= {
      };
      for(auto n:ids())site.successors.push_back(n);
      o::OriginalObservation observation;
      observation.anchor=site.observation;
      observation.available=true;
      q.observations.push_back(observation);
      q.sites.push_back(site);
    }
    p.observed=q;
    o::Commands c(o::commandCutCount(p));
    for(auto &word:c) {
      auto commands=number();
      while(commands--) {
        o::Command x;
        const auto kind=number();
        if (kind > unsigned(o::Command::BarrierAll)) throw std::runtime_error("invalid command tag");
        x.kind=o::Command::Kind(kind);
        x.source=o::Pipe(number());
        x.observer=o::Pipe(number());
        x.key=number();
        word.push_back(x);
      }
    }
    const bool construction=number();
    bool constructed=false;
    std::string reason;
    if(construction) {
      auto plan=o::constructSelectedPlan(p);
      constructed=plan.success;
      reason=plan.reason;
      if(plan.success)c=plan.commands;
    }
    auto a=o::analyze(p,c);
    auto order=o::checkPhaseOrder(p,c);
    std::cout<<"{\"complete\":"<<a.complete<<",\"accepted\":"<<a.verified()<<",\"exact\":"<<order.exact<<",\"constructed\":"<<constructed<<",\"reason\":";
    str(construction?reason:a.reason);
    std::cout<<",\"states\":"<<a.stats.phaseStateCount<<",\"commands\":[";
    for(std::size_t s=0;s<c.size();++s) {
      if(s)std::cout<<',';
      std::cout<<'[';
      for(std::size_t j=0;j<c[s].size();++j) {
        if(j)std::cout<<',';
        auto &x=c[s][j];
        std::cout<<'['<<x.kind<<','<<unsigned(x.source)<<','<<unsigned(x.observer)<<','<<x.key<<']';
      }
      std::cout<<']';
    }
    std::cout<<"],\"keys\":[";
    for(std::size_t i=0;i<a.keys.size();++i) {
      if(i)std::cout<<',';
      auto &x=a.keys[i];
      std::cout<<'['<<unsigned(x.source)<<','<<unsigned(x.observer)<<','<<x.key<<']';
    }
    std::cout<<"],\"cuts\":[";
    for(std::size_t i=0;i<a.cuts.size();++i) {
      if(i)std::cout<<',';
      auto &x=a.cuts[i];
      std::cout<<"{\"incoming\":";
      snapshot(x.incoming);
      std::cout<<",\"before\":";
      snapshot(x.beforeIssue);
      std::cout<<",\"outgoing\":";
      snapshot(x.outgoing);
      std::cout<<'}';
    }
    std::cout<<"],\"fragments\":[";
    for(std::size_t i=0;i<p.operations.size()&&a.complete;++i) {
      if(i)std::cout<<',';
      o::PhaseFragment f;
      std::string why;
      if(!o::phaseFragment(p,i,f,why))throw std::runtime_error(why);
      std::cout<<"{\"nodes\":[";
      for(std::size_t j=0;j<f.endpoints.size();++j) {
        if(j)std::cout<<',';
        str(f.endpoints[j]);
      }
      std::cout<<"],\"edges\":[";
      for(std::size_t j=0;j<f.edges.size();++j) {
        if(j)std::cout<<',';
        std::cout<<'['<<f.edges[j].first<<','<<f.edges[j].second<<']';
      }
      std::cout<<"],\"accesses\":[";
      for(std::size_t j=0;j<f.accesses.size();++j) {
        if(j)std::cout<<',';
        auto &x=f.accesses[j];
        std::cout<<'['<<x.effect.cell<<','<<x.effect.read<<','<<x.effect.write<<','<<x.begin<<','<<x.end<<','<<x.permission<<']';
      }
      std::cout<<"]}";
    }
    std::cout<<"]}\n";
    return 0;
  }
  catch(const std::exception&e) {
    std::cerr<<e.what()<<'\n';
    return 2;
  }
}
