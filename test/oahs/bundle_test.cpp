// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "GraphOracle.h"
#include "PTO/Transforms/OAHS/Bundles.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include <cstdlib>
#include <iostream>
#include <random>
namespace o = mlir::pto::oahs;
namespace {
unsigned checks = 0, traces = 0;
void check(bool b, unsigned line) {
  ++checks;
  if (!b) {
    std::cerr << "bundle check " << line << '\n';
    std::abort();
  }
}
#define CHECK(x) check(bool(x), __LINE__)
o::Program base(unsigned lanes = 3) {
  o::Program p;
  p.cells.resize(3);
  p.target.contract = "test-only issue-ordered core";
  for (unsigned a = 0; a < lanes; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b = 0; b < lanes; ++b)
      if (a != b)
        p.target.keys[a][b] = {0, 1, 2};
  }
  return p;
}
o::Operation op(unsigned engine, unsigned cell, bool read, bool write) {
  o::Operation a;
  a.pipe = o::Pipe(engine);
  a.complete = true;
  if (read || write)
    a.accesses.push_back({cell, read, write});
  return a;
}
o::Region leaf(unsigned i) { return {o::Region::Operation, {}, i}; }
o::Region seq(std::initializer_list<o::Region> a) {
  return {o::Region::Sequence, a};
}
o::Command pub(unsigned a, unsigned b, unsigned k = 0) {
  return {o::Command::Publish, o::Pipe(a), o::Pipe(b), k};
}
o::Command wait(unsigned a, unsigned b, unsigned k = 0) {
  return {o::Command::Acquire, o::Pipe(a), o::Pipe(b), k};
}
unsigned count(const o::Commands &c, o::Command::Kind k) {
  unsigned n = 0;
  for (auto &at : c)
    for (auto x : at)
      n += x.kind == k;
  return n;
}
void oracle(const o::Program &p, const o::Commands &c, unsigned visits = 3) {
  CHECK(o::verify(p, c).success);
  for (auto t : oahs_oracle::traces(p, visits)) {
    ++traces;
    CHECK(oahs_oracle::graph(p, c, t));
  }
}
} // namespace
int main() {
  { // Necessary release covers WAW and WAR, not unrelated later source work.
    auto p = base();
    p.operations = {op(0, 0, 0, 1), op(0, 1, 0, 1), op(1, 0, 1, 0),
                    op(0, 0, 0, 1), op(0, 1, 1, 0)};
    o::Commands initial(6);
    initial[1] = {pub(0, 1)};
    initial[2] = {wait(0, 1)};
    auto actual = initial;
    actual[3] = {pub(1, 0), wait(1, 0)};
    o::BundleQuery query(p, initial);
    auto result = query.evaluate(actual);
    CHECK(result.complete && result.memoryProgressAt(3));
    bool waw = false, war = false;
    for (auto &r : result.discharged)
      if (r.demand.consumer == 3) {
        waw |= r.kind == o::CompletionRequirement::WAW;
        war |= r.kind == o::CompletionRequirement::WAR;
      }
    CHECK(waw && war);
    CHECK(result.analysis.cuts[3].beforeIssue->pending[0][1]);
    CHECK(!result.analysis
               .verified()); // the later Y read is not silently repaired
    CHECK(query.analysis().residuals.size() > result.analysis.residuals.size());
    auto plan = o::construct(p);
    CHECK(plan.success);
    oracle(p, plan.commands);
    for (auto c : plan.commands[3])
      CHECK(c.kind != o::Command::Barrier && c.kind != o::Command::BarrierAll);
    CHECK(oahs_oracle::graph(p, plan.commands, {0, 1, 2, 3, 4},
                             {{1, 2}, {1, 3}}));
  }
  { // A lane without payload memory must survive route construction.
    auto p = base();
    p.operations = {op(0, 0, 0, 1), op(2, 0, 1, 0)};
    for (auto &row : p.target.keys)
      for (auto &pool : row)
        pool.clear();
    p.target.keys[0][1] = {0};
    p.target.keys[1][2] = {0};
    auto plan = o::construct(p);
    CHECK(plan.success);
    CHECK(plan.bundleSelections == 1);
    CHECK(count(plan.commands, o::Command::Publish) == 2);
    CHECK(count(plan.commands, o::Command::Acquire) == 2);
    CHECK(plan.commands[1].size() == 4);
    CHECK(plan.commands[1][1].kind == o::Command::Acquire);
    CHECK(plan.commands[1][2].source == o::Pipe(1));
    oracle(p, plan.commands);
    p.reservations.push_back({o::Pipe(1), o::Pipe(2), 0, false});
    auto blocked = o::construct(p);
    CHECK(!blocked.success);
    CHECK(blocked.reason.find("not a target infeasibility proof") !=
          std::string::npos);
  }
  { // Two original arms, one dynamic publication, one common acquisition.
    auto p = base();
    p.operations = {op(0, 0, 0, 1), op(0, 1, 0, 1), op(0, 0, 0, 1),
                    op(0, 2, 0, 1), op(1, 0, 1, 0)};
    p.body = seq({{o::Region::Choice,
                   {seq({leaf(0), leaf(1)}), seq({leaf(2), leaf(3)})}},
                  leaf(4)});
    o::PrefixQuery query(p);
    CHECK(!query.inspectPrefix(o::Pipe(0), 1, 4).matchingEstablished);
    CHECK(!query.inspectPrefix(o::Pipe(0), 3, 4).matchingEstablished);
    auto plan = o::construct(p);
    CHECK(plan.success);
    CHECK(count(plan.commands, o::Command::Publish) == 2);
    CHECK(count(plan.commands, o::Command::Acquire) == 1);
    CHECK(plan.commands[1].size() == 1 && plan.commands[3].size() == 1);
    oracle(p, plan.commands);
    for (auto t : oahs_oracle::traces(p, 1))
      CHECK(oahs_oracle::graph(p, plan.commands, t, {{1, 2}}));
    auto bad = plan.commands;
    bad[3].clear();
    auto rejected = o::analyze(p, bad);
    CHECK(!rejected.verified() && !rejected.protocol.empty());
  }
  { // The earlier SET timing must still not acquire a future write.
    auto p = base();
    p.operations = {op(0, 0, 0, 1), op(0, 0, 0, 1), op(1, 0, 1, 0)};
    o::Commands old(4);
    old[1] = {pub(0, 1)};
    old[2] = {wait(0, 1)};
    o::BundleQuery query(p, old);
    auto r = query.evaluate(old);
    CHECK(!r.analysis.verified());
    CHECK(r.discharged.empty());
    CHECK(!r.analysis.residuals.empty());
  }
  { // A helper is useful for protocol progress, not positive per-command memory
    // credit.
    auto p = base();
    p.operations = {op(0, 0, 0, 0), op(1, 0, 0, 0)};
    o::Commands c(3);
    c[0] = {pub(0, 1), wait(0, 1)};
    c[1] = {pub(0, 1), wait(0, 1)};
    o::BundleQuery query(p, c);
    auto trial = c;
    trial[0].push_back(pub(1, 0));
    trial[0].push_back(wait(1, 0));
    auto r = query.evaluate(trial);
    CHECK(r.complete && r.analysis.verified());
    CHECK(r.discharged.empty());
    CHECK(r.protocolProgress());
    oracle(p, trial);
  }
  { // Replay does not mutate current plan or its immutable report; invalid key
    // refuses.
    auto p = base();
    p.operations = {op(0, 0, 0, 1), op(1, 0, 1, 0)};
    o::Commands c(3);
    o::BundleQuery query(p, c);
    c[1] = {pub(0, 1), wait(0, 1)};
    auto r = query.evaluate(c);
    CHECK(r.analysis.verified());
    CHECK(!query.analysis().verified());
    auto broken = c;
    broken[1][1].key = 100;
    auto bad = query.evaluate(broken);
    CHECK(!bad.complete);
    CHECK(!query.analysis().verified());
    CHECK(query.evaluate(c).analysis.verified());
  }
  { // Whole-plan fallback must retain the work spent on discarded bundles.
    auto p = base(2);
    p.target.barrierAll = true;
    for (auto &row : p.target.keys)
      for (auto &pool : row)
        pool.clear();
    p.target.keys[0][1] = {0};
    p.operations = {op(0, 0, 0, 1), op(0, 1, 0, 1), op(1, 0, 1, 0),
                    op(1, 1, 1, 0)};
    auto plan = o::construct(p);
    CHECK(plan.success);
    CHECK(plan.conservativeBarriers == 4 && plan.scarcityBarriers == 4);
    CHECK(plan.bundleTrials > 0 && plan.bundleSelections > 0);
    CHECK(
        plan.handoffs.empty()); // The final fallback has no surviving handoffs.
    oracle(p, plan.commands);
  }
  // Build and check repeated/branching protocols, not just individual
  // candidates.
  std::mt19937 rng(0x4D3301u);
  unsigned accepted = 0;
  for (unsigned n = 0; n < 180; ++n) {
    auto p = base();
    p.target.barrierAll = true;
    for (unsigned i = 0; i < 5; ++i) {
      unsigned lane = rng() % 3, cell = rng() % 3;
      bool write = rng() % 2;
      p.operations.push_back(op(lane, cell, !write, write));
    }
    p.body = seq(
        {{o::Region::For,
          {seq({leaf(0),
                {o::Region::Choice, {seq({leaf(1), leaf(2)}), seq({leaf(3)})}},
                leaf(4)})},
          0,
          true}});
    auto plan = o::construct(p);
    CHECK(plan.success);
    ++accepted;
    oracle(p, plan.commands, 2);
    o::BundleQuery q(p, plan.commands, {false});
    auto replay = q.evaluate(plan.commands);
    CHECK(replay.analysis.verified());
    CHECK(replay.changedCuts.empty());
    CHECK(replay.discharged.empty());
  }
  std::cout << checks << " M3 bundle assertions; " << traces
            << " concrete graph traces; " << accepted
            << " generated programs\n";
}
