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
#include <set>
#include <limits>
#include <stdexcept>
#include <tuple>
namespace oahs_oracle {
namespace o = mlir::pto::oahs;
struct UncoveredConflict {
  unsigned producerVisit, consumerVisit, cell;
};
// Test-only immutable-prefix probe. Observation vertices have incoming edges
// from actual source finishes and NO outgoing edges: they cannot repair any
// missing memory order or improve a later source receipt.
struct PrefixProbe {
  o::Pipe source = o::Pipe::S;
  o::Cut publication = 0, acquisition = 0;
  std::vector<o::Demand> claims;
  bool matching = true, coverage = true;
};
struct Verdict { bool hazards=true, rearm=true, balanced=true, acyclic=true;
  explicit operator bool() const { return hazards && rearm && balanced && acyclic; } };
// Payload vertex 2*i is visit i's launch; 2*i+1 is its finish. Export the
// complete strict relation for paired-plan tests, not only its cardinality.
using PayloadOrder = std::set<std::pair<unsigned, unsigned>>;
// Independent finite graph: all command launches are ordered per engine;
// publication depends on source prefix finishes, but does not gate launches.
// Edges never include the memory demands or physical-key reuse obligations.
inline Verdict graph(const o::Program &truth, const o::Commands &commands,
              const std::vector<unsigned> &visits,
              const std::vector<std::pair<unsigned,unsigned>> &forbidden = {},
              std::vector<UncoveredConflict> *uncovered = nullptr,
              PrefixProbe *prefix = nullptr, PayloadOrder *payloadOrder = nullptr) {
  using K=std::tuple<o::Pipe,o::Pipe,unsigned>;
  std::vector<std::vector<unsigned>> edges;
  auto vertex=[&]() { edges.emplace_back(); return unsigned(edges.size()-1); };
  auto edge=[&](unsigned a,unsigned b) { edges[a].push_back(b); };
  std::array<unsigned,o::PipeCount> launch,gate;
  std::array<std::vector<unsigned>,o::PipeCount> finishes;
  for(unsigned a=0;a<o::PipeCount;++a) launch[a]=gate[a]=vertex();
  std::map<K,unsigned> live,lastWait;
  std::vector<std::pair<unsigned,unsigned>> rearms;
  std::vector<unsigned> starts,dones, observedPrefixes;
  const unsigned noPrefix = std::numeric_limits<unsigned>::max();
  unsigned livePrefix = noPrefix;
  if (prefix) { prefix->matching = true; prefix->coverage = true; }
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
    if (prefix && id == prefix->publication) {
      prefix->matching &= livePrefix == noPrefix;
      livePrefix = vertex();
      for (auto f : finishes[unsigned(prefix->source)]) edge(f, livePrefix);
    }
    for(const auto &c:commands.at(id)) sync(c);
    unsigned observed = noPrefix;
    if (prefix && id == prefix->acquisition) {
      prefix->matching &= livePrefix != noPrefix;
      observed = livePrefix; livePrefix = noPrefix;
    }
    observedPrefixes.push_back(observed);
    unsigned p=unsigned(truth.operations[id].pipe);
    auto i=vertex(),f=vertex(); edge(launch[p],i); edge(gate[p],i); edge(i,f);
    launch[p]=i; if(truth.target.synchronous[p]) gate[p]=f;
    finishes[p].push_back(f); starts.push_back(i); dones.push_back(f);
  }
  for(const auto &c:commands.back()) sync(c);
  v.balanced &= live.empty();
  if (prefix) prefix->matching &= livePrefix == noPrefix;
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
        if(a.cell==b.cell && (a.write||b.write||truth.cells[a.cell].exclusive)) {
          const bool covered = reaches(dones[i],starts[j]);
          v.hazards &= covered;
          if (!covered && uncovered) uncovered->push_back({i,j,a.cell});
        }
  if (prefix) {
    for (unsigned j = 0; j < visits.size(); ++j) {
      if (visits[j] != prefix->acquisition) continue;
      if (observedPrefixes[j] == noPrefix) { prefix->coverage = false; continue; }
      for (unsigned i = 0; i < j; ++i)
        for (const auto &claim : prefix->claims)
          if (claim.producer == visits[i] && claim.consumer == visits[j])
            prefix->coverage &= reaches(dones[i], observedPrefixes[j]);
    }
  }
  for(auto [a,b]:rearms) v.rearm &= reaches(a,b);
  for(auto [a,b]:forbidden) v.hazards &= !reaches(dones.at(a), starts.at(b));
  if (payloadOrder) {
    payloadOrder->clear();
    for (unsigned a = 0; a < 2 * visits.size(); ++a)
      for (unsigned b = 0; b < 2 * visits.size(); ++b)
        if (a != b && reaches(a % 2 ? dones[a / 2] : starts[a / 2],
                              b % 2 ? dones[b / 2] : starts[b / 2]))
          payloadOrder->emplace(a, b);
  }
  return v;
}

} // namespace oahs_oracle
#endif
