// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Plan.h"
#include "GraphOracle.h"
#include <cstdlib>
#include <iostream>
#include <random>
namespace o = mlir::pto::oahs;
namespace {
unsigned checks = 0, executions = 0, acceptedMutations = 0, rejectedMutations = 0;
void require(bool value, unsigned line) {
  ++checks;
  if (!value) { std::cerr << "structured check failed at " << line << '\n'; std::abort(); }
}
#define CHECK(x) require(bool(x), __LINE__)
o::Program base(unsigned keys = 3) {
  o::Program p;
  p.cells.resize(3); p.target.contract = "test-only independent issue-ordered engines";
  p.target.barrierAll = true;
  for (unsigned a = 0; a < 3; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b = 0; b < 3; ++b) if (a != b)
      for (unsigned k = 0; k < keys; ++k) p.target.keys[a][b].push_back(k);
  }
  return p;
}
o::Operation op(unsigned pipe, unsigned cell, bool write) {
  o::Operation result; result.pipe = o::Pipe(pipe); result.complete = true;
  result.accesses.push_back({cell, !write, write}); return result;
}
o::Region leaf(unsigned id) { return {o::Region::Operation, {}, id}; }
o::Region seq(std::initializer_list<o::Region> children) {
  return {o::Region::Sequence, children};
}
void oracle(const o::Program &p, const o::Commands &commands, unsigned trips = 3) {
  for (const auto &trace : oahs_oracle::traces(p, trips)) {
    auto v = oahs_oracle::graph(p, commands, trace);
    if (!v) std::cerr << "hazards=" << v.hazards << " rearm=" << v.rearm
                     << " balanced=" << v.balanced << " DAG=" << v.acyclic << '\n';
    CHECK(v); ++executions;
  }
}
o::Result build(const o::Program &p, unsigned trips = 3) {
  auto result = o::construct(p);
  if (!result.success) std::cerr << result.reason << '\n';
  CHECK(result.success); CHECK(o::verify(p, result.commands).success);
  oracle(p, result.commands, trips); return result;
}
unsigned allCount(const o::Commands &commands) {
  unsigned count = 0;
  for (const auto &at : commands) for (const auto &c : at) count += c.kind == o::Command::BarrierAll;
  return count;
}
void mutation(const o::Program &p, const o::Commands &commands, unsigned trips = 3) {
  if (o::verify(p, commands).success) { ++acceptedMutations; oracle(p, commands, trips); }
  else ++rejectedMutations;
}
}
int main() {
  // Storage release already supplies the causality for both recurring keys.
  auto ring = base(1);
  ring.operations = {op(0,0,true), op(1,0,false)};
  ring.body = {o::Region::For, {seq({leaf(0),leaf(1)})}, 0, true};
  auto ringPlan = build(ring, 8);
  CHECK(allCount(ringPlan.commands) == 0);
  CHECK(ringPlan.handoffs.size() == 2);
  CHECK(ringPlan.protocolRepairs == 0); // no additional acknowledgment packet
  auto noReturn = ringPlan.commands;
  noReturn[0].clear();
  CHECK(!o::verify(ring, noReturn).success);

  auto multiReader = base();
  multiReader.operations = {op(0,0,false),op(1,0,false),op(2,0,true)};
  auto releases = build(multiReader);
  CHECK(releases.handoffs.size() == 2);
  auto missingReader = releases.commands; missingReader[2].pop_back();
  CHECK(!o::verify(multiReader,missingReader).success);

  // The empty branch must not switch the straight-line case to a drain planner.
  auto empty = base(); empty.operations = {op(0,0,true),op(1,0,false)};
  empty.body = seq({leaf(0),leaf(1),{o::Region::Choice,{seq({}),seq({})}}});
  CHECK(allCount(build(empty).commands) == 0);

  // A live receipt crosses a whole child loop, including zero trips. The child
  // neither consumes the token nor changes X's generation.
  auto open = base();
  open.operations = {op(0,0,true),op(0,1,false),op(0,1,false),op(1,0,false)};
  open.body = seq({leaf(0),leaf(1),{o::Region::For,{seq({leaf(2)})},0,true},leaf(3)});
  o::Commands live(5);
  live[1] = {{o::Command::Publish,o::Pipe(0),o::Pipe(1),0}};
  live[3] = {{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  CHECK(o::verify(open,live).success); oracle(open,live,8);
  // Reusing the same static producer class inside the child invalidates the old
  // publication. A pre-loop receipt must not certify the new X generation.
  open.operations[2] = op(0,0,true);
  CHECK(!o::verify(open,live).success);

  auto conditional = base();
  conditional.operations = {op(0,0,true),op(0,1,false),op(0,1,false),op(1,0,false)};
  conditional.body = seq({leaf(0),{o::Region::Choice,{seq({leaf(1)}),seq({leaf(2)})}},leaf(3)});
  CHECK(!o::verify(conditional,live).success); // missing publication on else

  // A primed full token is an actual loop interface: each visit consumes and
  // republishes it, and cleanup consumes the final generation even at zero trips.
  auto circulating = base(1);
  circulating.operations = {op(0,1,false),op(1,1,false),op(1,1,false)};
  circulating.body = seq({leaf(0),{o::Region::For,{seq({leaf(1)})},0,true},leaf(2)});
  o::Commands circulation(4);
  circulation[0] = {{o::Command::Publish,o::Pipe(0),o::Pipe(1),0}};
  circulation[1] = {{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0},
                    {o::Command::Publish,o::Pipe(1),o::Pipe(0),0},
                    {o::Command::Acquire,o::Pipe(1),o::Pipe(0),0},
                    {o::Command::Publish,o::Pipe(0),o::Pipe(1),0}};
  circulation[2] = {{o::Command::Acquire,o::Pipe(0),o::Pipe(1),0}};
  CHECK(o::verify(circulating,circulation).success);
  oracle(circulating,circulation,8);
  auto noCleanup = circulation; noCleanup[2].clear();
  CHECK(!o::verify(circulating,noCleanup).success);
  auto noReceipt = circulation;
  noReceipt[1].erase(noReceipt[1].begin()+1,noReceipt[1].begin()+3);
  CHECK(!o::verify(circulating,noReceipt).success);

  // Global reservations are fixed constraints, not imaginary acquired events.
  auto reserved = base(); reserved.operations = {op(0,0,true),op(1,0,false)};
  reserved.reservations.push_back({o::Pipe(0),o::Pipe(1),0,true});
  auto reservedPlan = build(reserved);
  for (const auto &at : reservedPlan.commands) for (const auto &c : at)
    if (c.kind == o::Command::Publish) CHECK(c.key != 0);

  auto vector = mlir::pto::a3SyncProfile(mlir::pto::SyncCore::Vector);
  auto cube = mlir::pto::a3SyncProfile(mlir::pto::SyncCore::Cube);
  CHECK(vector.supported[unsigned(o::Pipe::V)] && !vector.supported[unsigned(o::Pipe::M)]);
  CHECK(vector.keys[unsigned(o::Pipe::V)][unsigned(o::Pipe::MTE2)].size() == 6);
  CHECK(cube.keys[unsigned(o::Pipe::MTE2)][unsigned(o::Pipe::FIX)].empty());
  CHECK(cube.keys[unsigned(o::Pipe::FIX)][unsigned(o::Pipe::MTE3)].empty());
  CHECK(cube.keys[unsigned(o::Pipe::M)][unsigned(o::Pipe::FIX)].size() == 6);
  CHECK(cube.keys[unsigned(o::Pipe::S)][unsigned(o::Pipe::M)].empty());

  // Essential verification has no work cutoff. Real key limits still apply.
  auto noAll = ring; noAll.target.barrierAll = false;
  CHECK(build(noAll,8).success);

  std::mt19937 random(20260915);
  for (unsigned sample = 0; sample < 180; ++sample) {
    auto p = base(1 + random()%3);
    for (unsigned i = 0; i < 5; ++i) p.operations.push_back(op(random()%3,random()%3,random()%2));
    if (sample%3 == 0) p.cells[0].exclusive = true;
    if (sample%2 == 0)
      p.body = seq({leaf(0),{o::Region::For,{seq({
        {o::Region::Choice,{seq({leaf(1)}),seq({leaf(2)})}},leaf(3)})},0,true},leaf(4)});
    else
      p.body = seq({leaf(0),{o::Region::For,{seq({leaf(1),
        {o::Region::While,{seq({leaf(2)}),seq({leaf(3)})}}})},0,true},leaf(4)});
    auto result = build(p,2);
    for (unsigned cut = 0; cut < result.commands.size(); ++cut) {
      for (unsigned j = 0; j < result.commands[cut].size(); ++j) {
        auto deleted = result.commands; deleted[cut].erase(deleted[cut].begin()+j);
        mutation(p,deleted,2);
        auto moved = deleted;
        moved[(cut+1)%moved.size()].push_back(result.commands[cut][j]);
        mutation(p,moved,2);
        auto changed = result.commands; changed[cut][j].key = 0;
        mutation(p,changed,2);
      }
    }
  }
  std::cout << checks << " structured assertions; " << executions << " concrete traces; "
            << acceptedMutations << " verified mutations independently checked; "
            << rejectedMutations << " mutations rejected\n";
}
