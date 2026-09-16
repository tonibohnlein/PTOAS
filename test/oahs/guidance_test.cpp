// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "GraphOracle.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <cstdlib>
#include <iostream>
#include <random>
namespace o = mlir::pto::oahs;
namespace {
unsigned checks = 0, traces = 0;
void check(bool b, unsigned line) {
  ++checks;
  if (!b) {
    std::cerr << "guided check " << line << '\n';
    std::abort();
  }
}
#define CHECK(x) check(bool(x), __LINE__)
o::Program base() {
  o::Program p;
  p.cells.resize(3);
  p.target.contract = "test-only issue-ordered";
  for (unsigned a = 0; a < 3; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b = 0; b < 3; ++b)
      if (a != b)
        p.target.keys[a][b] = {0, 1};
  }
  return p;
}
o::Operation op(unsigned pipe, unsigned cell, bool write) {
  o::Operation a;
  a.pipe = o::Pipe(pipe);
  a.complete = true;
  a.accesses = {{cell, !write, write, write}};
  return a;
}
o::Region leaf(unsigned i) { return {o::Region::Operation, {}, i}; }
o::Region seq(std::initializer_list<o::Region> r) {
  return {o::Region::Sequence, r};
}
void oracle(const o::Program &p, const o::Result &r) {
  CHECK(r.success && o::verify(p, r.commands).success);
  for (auto t : oahs_oracle::traces(p, 2)) {
    ++traces;
    CHECK(oahs_oracle::graph(p, r.commands, t));
  }
}
} // namespace
int main() {
  {
    auto p = base();
    p.operations = {op(0, 0, true), op(2, 0, false)};
    for (auto &row : p.target.keys) {
      for (auto &keys : row)
        keys.clear();
    }
    p.target.keys[0][1] = {0};
    p.target.keys[1][2] = {0};
    auto r = o::construct(p);
    oracle(p, r);
    CHECK(r.stages.size() == 3);
    CHECK(!r.stages[0].success && !r.stages[1].success && r.stages[2].success);
    CHECK(r.stages[2].stage == o::CandidateStage::Relays);
    auto b = o::construct(p, {false});
    oracle(p, b);
    CHECK(b.stages.size() == 1 &&
          b.stages[0].stage == o::CandidateStage::Original);
  }
  std::mt19937 random(0x4B01u);
  unsigned finalStages[4] = {};
  for (unsigned sample = 0; sample < 180; ++sample) {
    auto p = base();
    p.target.barrierAll = true;
    for (unsigned i = 0; i < 6; ++i) {
      const auto pipe = random() % 3, cell = random() % 3;
      const bool write = random() % 2;
      p.operations.push_back(op(pipe, cell, write));
    }
    p.body = seq(
        {leaf(0),
         {o::Region::For,
          {seq({leaf(1), {o::Region::Choice, {leaf(2), leaf(3)}}, leaf(4)})}},
         leaf(5)});
    auto guided = o::construct(p), plain = o::construct(p, {false});
    oracle(p, guided);
    oracle(p, plain);
    std::size_t trials = 0, selections = 0, repairs = 0;
    for (const auto &s : guided.stages) {
      trials += s.bundleTrials;
      selections += s.bundleSelections;
      repairs += s.protocolRepairs;
    }
    CHECK(trials == guided.bundleTrials &&
          selections == guided.bundleSelections &&
          repairs == guided.protocolRepairs);
    ++finalStages[unsigned(guided.stages.back().stage)];
    if (guided.stages.back().stage == o::CandidateStage::Original) {
      CHECK(guided.commands.size() == plain.commands.size());
      for (unsigned c = 0; c < guided.commands.size(); ++c) {
        CHECK(guided.commands[c].size() == plain.commands[c].size());
        for (unsigned i = 0; i < guided.commands[c].size(); ++i) {
          auto a = guided.commands[c][i], b = plain.commands[c][i];
          CHECK(a.kind == b.kind && a.source == b.source &&
                a.observer == b.observer && a.key == b.key);
        }
      }
    }
  }
  std::cout << checks << " guided assertions; " << traces
            << " traces; final stages " << finalStages[0] << ','
            << finalStages[1] << ',' << finalStages[2] << ',' << finalStages[3]
            << '\n';
}
