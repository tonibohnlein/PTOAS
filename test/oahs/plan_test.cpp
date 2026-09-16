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
#include <map>
#include <random>
#include <tuple>

namespace o = mlir::pto::oahs;
static unsigned checks = 0;
static void checkAt(bool value, unsigned line) {
  ++checks;
  if (!value) {
    std::cerr << "failed assertion " << checks << " at line " << line << '\n';
    std::abort();
  }
}
#define check(value) checkAt((value), __LINE__)
static o::Program program(unsigned keys = 6) {
  o::Program p;
  p.cells.resize(3);
  p.target.contract = "test-only asynchronous three-pipeline target";
  p.target.barrierAll = true;
  for (unsigned a = 0; a < 3; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b = 0; b < 3; ++b)
      if (a != b)
        for (unsigned k = 0; k < keys; ++k) p.target.keys[a][b].push_back(k);
  }
  return p;
}
static o::Operation op(unsigned pipe, unsigned cell, bool write) {
  o::Operation result;
  result.pipe = o::Pipe(pipe);
  result.accesses.push_back({cell, !write, write});
  result.complete = true;
  return result;
}

// Independent command graph, including causal rearming and original issue order.
static void oracle(const o::Program &p, const o::Commands &commands,
                   std::vector<std::pair<unsigned, unsigned>> forbidden = {}) {
  for (const auto &trace : oahs_oracle::traces(p)) {
    auto result = oahs_oracle::graph(p, commands, trace, forbidden);
    check(result.hazards); check(result.rearm);
    check(result.balanced); check(result.acyclic);
  }
}
static o::Result run(const o::Program &p) {
  auto result = o::construct(p);
  if (!result.success) std::cerr << result.reason << '\n';
  check(result.success);
  check(o::verify(p, result.commands).success);
  oracle(p, result.commands);
  return result;
}
int main() {
  auto p = program();
  p.operations = {op(0, 0, true), op(0, 1, true), op(1, 1, false), op(1, 0, false)};
  auto shared = run(p);
  check(shared.handoffs.size() == 1);
  check(shared.handoffs[0].publication == 2 && shared.handoffs[0].acquisition == 2);
  auto deleted = shared.commands;
  deleted[2].pop_back();
  check(!o::verify(p, deleted).success);
  auto early = shared.commands;
  auto publication = early[2].front(); early[2].erase(early[2].begin());
  early[0].push_back(publication);
  check(!o::verify(p, early).success);

  // A's first consumer must not acquire B's later production on the same lane.
  p.operations = {op(0, 0, true), op(0, 1, true), op(1, 0, false)};
  auto independent = run(p);
  check(independent.handoffs.size() == 1 && independent.handoffs[0].publication == 1);
  check(independent.commands.back().empty()); // No unrequested retirement/reply.
  oracle(p, independent.commands, {{1, 2}});

  // A release follows the actual last reader, before unrelated downstream work.
  p.operations = {op(1, 0, false), op(1, 1, true), op(0, 0, true)};
  auto release = run(p);
  check(release.handoffs[0].publication == 1);
  oracle(p, release.commands, {{1, 2}});

  // The second transfer carries the first lane's previously acquired prefix.
  p.operations = {op(0, 0, true), op(1, 0, false), op(1, 1, true),
                  op(2, 1, false), op(2, 0, false)};
  auto transitive = run(p);
  check(transitive.handoffs.size() == 2);

  p.operations = {op(0, 0, true), op(1, 0, false), op(0, 0, true), op(1, 0, false)};
  auto fresh = run(p);
  check(fresh.handoffs.size() == 3); // readiness, release, new readiness
  auto reused = fresh.commands;
  for (auto &cut : reused)
    for (auto &c : cut)
      if (c.kind == o::Command::Publish || c.kind == o::Command::Acquire) c.key = 0;
  // The real return handoff acknowledges consumption: no extra reply required.
  check(o::verify(p, reused).success);
  auto unsafe = program();
  unsafe.operations = {op(0, 0, true), op(1, 0, false), op(0, 1, true), op(1, 1, false)};
  auto unsafePlan = run(unsafe);
  for (auto &cut : unsafePlan.commands)
    for (auto &c : cut) c.key = 0;
  check(!o::verify(unsafe, unsafePlan.commands).success);
  check(!oahs_oracle::graph(unsafe, unsafePlan.commands, {0,1,2,3}).rearm);

  auto scarce = program(1); scarce.operations = unsafe.operations;
  auto scarcePlan = run(scarce);
  check(scarcePlan.protocolRepairs > 0 && scarcePlan.scarcityBarriers == 0);
  scarce.target.barrierAll = false;
  run(scarce); // Causally checked reuse must not require ALL to be available.
  scarce.target.keys[1][0].clear();
  // The old direct-reply policy fails; a payload-free relay is a real route.
  check(!o::construct(scarce, {true, true, false}).success);
  auto relayed = run(scarce);
  check(relayed.work.routedReplies || relayed.work.longerRouteTrials);
  p.operations = {op(0, 0, false), op(1, 0, false)};
  check(run(p).handoffs.empty());
  p.cells[0].exclusive = true;
  check(run(p).handoffs.size() == 1);
  check(!p.operations[0].accesses[0].write); // Exclusion did not change roles.
  p.invocation.retirement =
      o::Program::InvocationContract::DrainAllAtReturn;
  run(p);
  auto retired = o::construct(p).commands; retired.back().clear();
  check(!o::verify(p, retired).success);
  p.operations[0].complete = false;
  check(!o::construct(p).success);
  p.operations[0].complete = true;
  p.operations[0].resources.push_back({"private queue", true, false});
  check(o::validateProgram(p).success);
  check(!o::construct(p).success);
  p.operations[0].resources.clear();

  // Analysis retains actual structured control and reports synthesis coverage
  // separately. Every physical phase must occur in the region tree.
  auto structured = program();
  structured.operations = {op(0, 0, true), op(1, 0, true),
                           op(2, 0, false)};
  o::Region thenRegion{o::Region::Sequence,
                       {{o::Region::Operation, {}, 0}}};
  o::Region elseRegion{o::Region::Sequence,
                       {{o::Region::Operation, {}, 1}}};
  o::Region choice{o::Region::Choice, {thenRegion, elseRegion}};
  structured.body = {o::Region::Sequence,
                     {choice, {o::Region::Operation, {}, 2}}};
  check(o::validateProgram(structured).success);
  auto structuredPlan = o::construct(structured);
  check(structuredPlan.success);
  oracle(structured, structuredPlan.commands);
  auto missingStructured = structuredPlan.commands;
  missingStructured[2].clear();
  check(!o::verify(structured, missingStructured).success);
  structured.body.children[0].children[1].children.clear();
  check(!o::validateProgram(structured).success);

  auto loop = program();
  loop.operations = {op(0, 0, true)};
  o::Region loopBody{o::Region::Sequence,
                     {{o::Region::Operation, {}, 0}}};
  loop.body = {o::Region::For, {loopBody}, 0, true};
  check(o::validateProgram(loop).success);
  auto loopPlan = o::construct(loop);
  check(loopPlan.success);
  oracle(loop, loopPlan.commands);
  check(!loopPlan.commands[0].empty()); // repeated write generation

  auto whileProgram = program();
  whileProgram.operations = {op(0, 0, true), op(1, 0, false)};
  o::Region before{o::Region::Sequence,
                   {{o::Region::Operation, {}, 0}}};
  o::Region after{o::Region::Sequence,
                  {{o::Region::Operation, {}, 1}}};
  whileProgram.body = {o::Region::While, {before, after}};
  auto whilePlan = o::construct(whileProgram);
  check(whilePlan.success);
  oracle(whileProgram, whilePlan.commands);
  check(!whilePlan.commands[0].empty() && !whilePlan.commands[1].empty());

  // A fully serialized reference must cover the FIRST repeated phase too.
  // There is no work-budget flag that can hide a missing original obligation.
  for (o::Program cyclic : {loop, whileProgram}) {
    o::Commands serialized(cyclic.operations.size()+1);
    for (unsigned i=0;i<cyclic.operations.size();++i)
      serialized[i].push_back({o::Command::BarrierAll});
    check(o::verify(cyclic,serialized).success);
    oracle(cyclic,serialized);
    serialized[0].clear();
    check(!o::verify(cyclic,serialized).success);
  }

  std::mt19937 rng(7321);
  for (unsigned sample = 0; sample < 200; ++sample) {
    auto random = program(1 + rng() % 3);
    random.invocation.retirement = sample % 2
        ? o::Program::InvocationContract::DrainAllAtReturn
        : o::Program::InvocationContract::NoRetirement;
    random.cells[0].exclusive = sample % 3 == 0;
    for (unsigned i = 0; i < 16; ++i) {
      // Fix draw order independently of function-argument evaluation order.
      // Preserve the population previously generated by GCC (right to left).
      const bool write = rng() % 2;
      const unsigned cell = rng() % 3;
      const unsigned pipe = rng() % 3;
      random.operations.push_back(op(pipe, cell, write));
    }
    run(random);
  }
  std::cout << checks << " OAHS assertions passed (including 200 independent graph checks)\n";
}
