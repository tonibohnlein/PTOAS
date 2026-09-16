// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef OAHS_OBSERVED_FIXTURES_H
#define OAHS_OBSERVED_FIXTURES_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
namespace observed_fixtures {
namespace o = mlir::pto::oahs;
inline o::Program target(unsigned lanes, unsigned cells, unsigned keys = 4) {
  o::Program p;
  p.cells.resize(cells);
  p.target.contract = "test-only exact issue-ordered core";
  for (unsigned a = 0; a < lanes; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b = 0; b < lanes; ++b)
      if (a != b)
        for (unsigned k = 0; k < keys; ++k)
          p.target.keys[a][b].push_back(k);
  }
  return p;
}
inline o::Operation op(unsigned pipe, unsigned cell, unsigned mode) {
  o::Operation a;
  a.pipe = o::Pipe(pipe);
  a.complete = true;
  if (mode)
    a.accesses.push_back(
        {cell, bool(mode & 1), bool(mode & 2), bool(mode & 2)});
  return a;
}
inline o::Region leaf(unsigned i) { return {o::Region::Operation, {}, i}; }
inline o::Region seq(std::initializer_list<o::Region> a) {
  return {o::Region::Sequence, a};
}
inline o::ObservedImport ring(unsigned b, bool stride = false) {
  auto p = target(2, stride ? 4 : b, b);
  p.operations = {op(0, 0, 2), op(1, 0, 1)};
  std::vector<unsigned> cells;
  for (unsigned i = 0; i < p.cells.size(); ++i)
    cells.push_back(i);
  return o::makePeriodicLoop(
      p, b,
      {{0, 0, cells, stride ? 2u : 1u, stride ? 1u : 0u},
       {1, 0, cells, stride ? 2u : 1u, stride ? 1u : 0u}});
}
inline o::ObservedImport refinedRing(unsigned b) {
  auto p = target(2, b, b);
  for (unsigned i = 0; i < b; ++i) {
    p.operations.push_back(op(0, i, 2));
    p.operations.push_back(op(1, i, 1));
  }
  if (b == 1)
    p.body = {o::Region::For, {seq({leaf(0), leaf(1)})}};
  else if (b == 2) {
    o::Region choice{o::Region::Choice,
                     {seq({leaf(0), leaf(1)}), seq({leaf(2), leaf(3)})}};
    p.body = {o::Region::For, {choice}};
  } else
    return {};
  auto original = o::addStructuredBoundaryCuts(p);
  if (!original.success)
    return original;
  auto &q = *original.program.observed;
  o::CountedLoopRegion spec;
  spec.owner = q.entry;
  spec.header = q.sites[spec.owner].successors.front();
  spec.bodyEntry = q.sites[spec.header].successors[0];
  spec.continuation = q.sites[spec.header].successors[1];
  spec.period = b;
  std::vector<bool> seen(q.sites.size());
  std::vector<std::size_t> todo{spec.bodyEntry};
  while (!todo.empty()) {
    auto at = todo.back();
    todo.pop_back();
    if (at == spec.header || seen[at])
      continue;
    seen[at] = true;
    spec.bodySites.push_back(at);
    for (auto next : q.sites[at].successors)
      todo.push_back(next);
  }
  if (b == 2)
    spec.decisions.push_back({spec.bodyEntry, 2, 0, true});
  return o::refineCountedLoop(original.program, spec);
}
inline o::ObservedImport nested() {
  auto p = target(2, 1);
  p.operations = {op(0, 0, 2), op(1, 0, 1)};
  p.body = {o::Region::For, {seq({leaf(0), {o::Region::For, {leaf(1)}}})}};
  return o::addStructuredBoundaryCuts(p);
}
inline o::ObservedImport mixed() {
  auto p = target(3, 1);
  p.operations = {op(0, 0, 2), op(1, 0, 1), op(2, 0, 3), op(1, 0, 1)};
  p.body = {o::Region::For,
            {seq({leaf(0), {o::Region::Choice, {leaf(1), leaf(2)}}, leaf(3)})}};
  return o::addStructuredBoundaryCuts(p);
}
inline o::ObservedImport independent() {
  auto p = target(3, 1);
  p.operations = {op(0, 0, 2), op(1, 0, 1), op(2, 0, 1)};
  p.body = {o::Region::For, {seq({leaf(0), leaf(1), leaf(2)})}};
  return o::addStructuredBoundaryCuts(p);
}
} // namespace observed_fixtures
#endif
