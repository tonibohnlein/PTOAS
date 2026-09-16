// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "PTO/Transforms/OAHS/Plan.h"
#include "PTO/Transforms/OAHS/StorageWitnesses.h"
#include "PTO/Transforms/InsertSync/SyncOriginPropagation.h"
#include "../../lib/PTO/Transforms/OAHS/Transfer.h"
#include "GraphOracle.h"
#include <cstdlib>
#include <iostream>
#include <set>
#include <random>

namespace o = mlir::pto::oahs;
static unsigned checks = 0;
static void check(bool condition, unsigned line) {
  ++checks;
  if (!condition) { std::cerr << "scaling check failed at " << line << '\n'; std::abort(); }
}
#define CHECK(x) check(bool(x),__LINE__)
static o::Program program() {
  o::Program p; p.target.contract = "test-only issue-ordered asynchronous engines";
  for (unsigned a=0;a<3;++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for(unsigned b=0;b<3;++b) if(a!=b) p.target.keys[a][b]={0,1};
  }
  p.target.barrierAll=true; return p;
}
static o::Region leaf(unsigned id) { return {o::Region::Operation,{},id}; }
static o::Operation op(unsigned lane) {
  o::Operation a; a.pipe=o::Pipe(lane); a.complete=true; return a;
}
static bool conflict(const o::Program &p,unsigned a,unsigned b) {
  for(const auto &x:p.operations[a].accesses)
    for(const auto &y:p.operations[b].accesses)
      if(x.cell==y.cell && (x.write||y.write||p.cells[x.cell].exclusive)) return true;
  return false;
}
int main() {
  // Former recursive solving took 2^(depth+1)-1 charged visits, even empty.
  for(unsigned depth : {4,8,12,32,128,512}) {
    auto p=program(); o::Region r;
    for(unsigned i=0;i<depth;++i) r={o::Region::For,{std::move(r)},0,true};
    p.body=std::move(r); o::Commands commands(1);
    o::detail::Transfer transfer(p,commands);
    CHECK(transfer.run().kind==o::detail::Failure::None);
    CHECK(transfer.evaluationCount()<=2*depth+3);
    std::cout << "empty depth=" << depth << " sites=" << transfer.siteCount()
              << " evaluations=" << transfer.evaluationCount() << '\n';
  }
  // Not an empty-subtree special case: pending payload facts also stabilize
  // through nested loops in the same static worklist.
  for(unsigned depth : {4,16,64,128}) {
    auto p=program();p.cells.resize(1);p.operations={op(0)};
    p.operations[0].accesses={{0,false,true}};
    o::Region r=leaf(0);
    for(unsigned i=0;i<depth;++i) r={o::Region::For,{std::move(r)},0,true};
    p.body=std::move(r);o::Commands commands(2);
    commands[0]={{o::Command::Barrier,o::Pipe(0)}};
    o::detail::Transfer transfer(p,commands);
    CHECK(transfer.run().kind==o::detail::Failure::None);
    CHECK(transfer.evaluationCount()<=10*(depth+2));
    commands[0].clear();CHECK(!o::verify(p,commands).success);
    std::cout<<"payload depth="<<depth<<" evaluations="<<transfer.evaluationCount()<<'\n';
  }
  // Root propagation exceeds the former 2^20 work allowance. Every distinct
  // root still reaches every node in a cyclic forwarding graph.
  struct Node { std::vector<unsigned> users; std::set<unsigned> roots; bool unknown=false; };
  const unsigned n=1025;
  std::vector<Node> nodes(n);
  for(unsigned i=0;i<n;++i) { nodes[i].users={(i+1)%n}; nodes[i].roots.insert(i); }
  const auto visits=mlir::pto::propagateSyncOrigins(nodes);
  CHECK(visits==std::size_t(n)*n);
  for(const auto &node:nodes) { CHECK(node.roots.size()==n);CHECK(!node.unknown); }
  std::vector<Node> unknownCycle(4);
  for(unsigned i=0;i<4;++i) unknownCycle[i].users={(i+1)%4};
  unknownCycle[0].unknown=true;
  mlir::pto::propagateSyncOrigins(unknownCycle);
  for(const auto &node:unknownCycle) CHECK(node.unknown && node.roots.empty());
  std::cout<<"origin nodes="<<n<<" propagated facts="<<visits<<'\n';
  // A~C, B~C does not imply A~B. Also compare arbitrary pair relations against
  // the conflicts represented by the actual shared native witness builder.
  std::mt19937 rng(20260915);
  for(unsigned sample=0;sample<100;++sample) {
    constexpr unsigned groups=4,operations=12;
    bool alias[groups][groups]{};
    for(unsigned a=0;a<groups;++a) for(unsigned b=a;b<groups;++b)
      alias[a][b]=alias[b][a]=(a==b || rng()%2);
    if(!sample) { alias[0][1]=alias[1][0]=false;alias[0][2]=alias[2][0]=true;alias[1][2]=alias[2][1]=true; }
    auto p=program();std::vector<o::FootprintGroup> population(groups);
    unsigned groupOf[operations]{};bool write[operations]{};
    for(unsigned i=0;i<operations;++i) {
      p.operations.push_back(op(i%3)); groupOf[i]=i%groups;write[i]=rng()%2;
      population[groupOf[i]].uses.push_back({i,!write[i],write[i]});
    }
    o::appendStorageWitnesses(p,population,[&](auto a,auto b){return alias[a][b];});
    for(unsigned a=0;a<operations;++a) for(unsigned b=0;b<operations;++b)
      CHECK(conflict(p,a,b)==(alias[groupOf[a]][groupOf[b]] && (write[a]||write[b])));
  }
  // Same physical pair used 730 times used to trigger whole-function ALL.
  // It now has just two exact footprint groups and no cross-space alias query.
  auto p=program();std::vector<o::FootprintGroup> groups(2);
  groups[0].description.addressSpace="gm";groups[1].description.addressSpace="vec";
  o::Region sequence;
  for(unsigned i=0;i<730;++i) {
    p.operations.push_back(op(0));sequence.children.push_back(leaf(i));
    groups[0].uses.push_back({i,true,false});groups[1].uses.push_back({i,false,true});
  }
  auto counts=o::appendStorageWitnesses(p,groups,[](auto,auto){return false;});
  CHECK(counts.groups==2 && counts.aliasQueries==0 && p.cells.size()==2);
  p.body={o::Region::For,{std::move(sequence)},0,true};
  o::Commands serial(p.operations.size()+1);
  for(unsigned i=0;i<p.operations.size();++i) serial[i]={{o::Command::Barrier,o::Pipe(0)}};
  CHECK(o::verify(p,serial).success);
  serial[0].clear();CHECK(!o::verify(p,serial).success);
  const auto constructed = o::constructSelectedPlan(p);
  CHECK(constructed.success);
  unsigned named=0, all=0;
  for(const auto &at:constructed.commands) for(const auto &c:at) {
    named += c.kind==o::Command::Barrier;
    all += c.kind==o::Command::BarrierAll;
  }
  CHECK(named==730 && all==0);
  std::cout<<"730-load witnesses="<<p.cells.size()<<" alias queries="<<counts.aliasQueries
           <<" named barriers="<<named<<" ALL="<<all<<'\n';
  std::cout<<checks<<" scaling assertions passed\n";
}
