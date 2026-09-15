// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TEST_OAHS_GRAPH_ORACLE_H
#define PTO_TEST_OAHS_GRAPH_ORACLE_H
#include "PTO/Transforms/OAHS/Plan.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
namespace oahs_oracle {
namespace o = mlir::pto::oahs;
struct Verdict { bool hazards=true, rearm=true, balanced=true, acyclic=true;
  explicit operator bool() const { return hazards && rearm && balanced && acyclic; } };
// Independent finite graph: all command launches are ordered per engine;
// publication depends on source prefix finishes, but does not gate launches.
// Edges never include the memory demands or physical-key reuse obligations.
inline Verdict graph(const o::Program &truth, const o::Commands &commands,
              const std::vector<unsigned> &visits,
              const std::vector<std::pair<unsigned,unsigned>> &forbidden = {}) {
  using K=std::tuple<o::Pipe,o::Pipe,unsigned>;
  std::vector<std::vector<unsigned>> edges;
  auto vertex=[&]() { edges.emplace_back(); return unsigned(edges.size()-1); };
  auto edge=[&](unsigned a,unsigned b) { edges[a].push_back(b); };
  std::array<unsigned,o::PipeCount> launch,gate;
  std::array<std::vector<unsigned>,o::PipeCount> finishes;
  for(unsigned a=0;a<o::PipeCount;++a) launch[a]=gate[a]=vertex();
  std::map<K,unsigned> live,lastWait;
  std::vector<std::pair<unsigned,unsigned>> rearms;
  std::vector<unsigned> starts,dones;
  Verdict v;
  auto sync=[&](const o::Command &c) {
    auto i=vertex(),f=vertex(); edge(i,f);
    auto a=unsigned(c.source),b=unsigned(c.observer);
    if(c.kind==o::Command::BarrierAll) {
      for(unsigned p=0;p<o::PipeCount;++p) {
        edge(launch[p],i); edge(gate[p],i);
        for(auto x:finishes[p]) edge(x,f);
        launch[p]=i; gate[p]=f; finishes[p].push_back(f);
      }
      return;
    }
    unsigned engine=c.kind==o::Command::Acquire ? b:a;
    edge(launch[engine],i); edge(gate[engine],i); launch[engine]=i;
    K key{c.source,c.observer,c.key};
    if(c.kind==o::Command::Publish) {
      for(auto x:finishes[a]) edge(x,f);
      if(live.count(key)) v.balanced=false;
      if(lastWait.count(key)) rearms.push_back({lastWait[key],f});
      live[key]=f;
    } else if(c.kind==o::Command::Acquire) {
      if(!live.count(key)) v.balanced=false;
      else { edge(live[key],f); live.erase(key); }
      lastWait[key]=f; gate[b]=f;
    } else {
      for(auto x:finishes[a]) edge(x,f);
      gate[a]=f;
    }
    finishes[engine].push_back(f);
  };
  for(unsigned id:visits) {
    for(const auto &c:commands.at(id)) sync(c);
    unsigned p=unsigned(truth.operations[id].pipe);
    auto i=vertex(),f=vertex(); edge(launch[p],i); edge(gate[p],i); edge(i,f);
    launch[p]=i; if(truth.target.synchronous[p]) gate[p]=f;
    finishes[p].push_back(f); starts.push_back(i); dones.push_back(f);
  }
  for(const auto &c:commands.back()) sync(c);
  v.balanced &= live.empty();
  auto reaches=[&](unsigned a,unsigned b) {
    std::vector<unsigned> work{a}; std::vector<bool> seen(edges.size());
    while(!work.empty()) {
      unsigned x=work.back(); work.pop_back(); if(x==b) return true;
      if(seen[x]) continue;
      seen[x]=true;
      for(auto y:edges[x]) work.push_back(y);
    }
    return false;
  };
  std::vector<unsigned> deg(edges.size()),todo;
  for(const auto &out:edges) for(auto y:out) ++deg[y];
  for(unsigned i=0;i<deg.size();++i) if(!deg[i]) todo.push_back(i);
  unsigned count=0;
  while(!todo.empty()) { auto x=todo.back(); todo.pop_back(); ++count;
    for(auto y:edges[x]) if(--deg[y]==0) todo.push_back(y); }
  v.acyclic=count==edges.size();
  for(unsigned i=0;i<visits.size();++i) for(unsigned j=i+1;j<visits.size();++j)
    for(const auto &a:truth.operations[visits[i]].accesses)
      for(const auto &b:truth.operations[visits[j]].accesses)
        if(a.cell==b.cell && (a.write||b.write||truth.cells[a.cell].exclusive))
          v.hazards &= reaches(dones[i],starts[j]);
  for(auto [a,b]:rearms) v.rearm &= reaches(a,b);
  for(auto [a,b]:forbidden) v.hazards &= !reaches(dones.at(a), starts.at(b));
  return v;
}

using Trace = std::vector<unsigned>;
using Traces = std::vector<Trace>;
inline Traces product(const Traces &a, const Traces &b, std::size_t limit) {
  Traces out;
  for (const auto &x : a) for (const auto &y : b) {
    if (out.size() >= limit) throw std::runtime_error("oracle trace budget exhausted");
    Trace z = x; z.insert(z.end(), y.begin(), y.end()); out.push_back(std::move(z));
  }
  return out;
}
// Enumerates common-control traces for tests only. Each loop visit can choose
// either branch independently. The production checker must not enumerate trips.
inline Traces expand(const o::Region &r, unsigned trips, std::size_t limit) {
  if (r.kind == o::Region::Operation) return {{unsigned(r.operation)}};
  if (r.kind == o::Region::Sequence) {
    Traces result{{}};
    for (const auto &child : r.children) result = product(result, expand(child,trips,limit),limit);
    return result;
  }
  if (r.kind == o::Region::Choice) {
    auto a=expand(r.children[0],trips,limit), b=expand(r.children[1],trips,limit);
    if (a.size()+b.size()>limit) throw std::runtime_error("oracle choice budget exhausted");
    a.insert(a.end(),b.begin(),b.end()); return a;
  }
  Traces prefix{{}}, body, result;
  if (r.kind == o::Region::While) {
    prefix=expand(r.children[0],trips,limit);
    body=product(expand(r.children[1],trips,limit),prefix,limit);
  } else body=expand(r.children[0],trips,limit);
  for (unsigned i=0;i<=trips;++i) {
    if (result.size()+prefix.size()>limit) throw std::runtime_error("oracle loop budget exhausted");
    result.insert(result.end(),prefix.begin(),prefix.end());
    if (i<trips) prefix=product(prefix,body,limit);
  }
  return result;
}
inline Traces traces(const o::Program &p, unsigned trips=3, std::size_t limit=20000) {
  if (p.body.kind==o::Region::Sequence && p.body.children.empty()) {
    Trace t(p.operations.size()); std::iota(t.begin(),t.end(),0); return {t};
  }
  return expand(p.body,trips,limit);
}
} // namespace oahs_oracle
#endif
