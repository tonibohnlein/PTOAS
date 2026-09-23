// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "../../lib/PTO/Transforms/OAHS/Control.h"
#include "GraphOracle.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <cstdlib>
#include <iostream>
#include <random>
namespace o = mlir::pto::oahs;
namespace {
std::size_t checks = 0, relations = 0;
void check(bool b, unsigned line) {
  ++checks;
  if (!b) {
    std::cerr << "storage check " << line << '\n';
    std::abort();
  }
}
#define CHECK(x) check(bool(x), __LINE__)
o::Program base() {
  o::Program p;
  p.cells.resize(3);
  p.target.contract = "test-only exact cells";
  for (unsigned a = 0; a < 3; ++a) {
    p.target.supported[a] = p.target.barriers[a] = true;
    for (unsigned b = 0; b < 3; ++b)
      if (a != b)
        p.target.keys[a][b] = {0, 1, 2};
  }
  return p;
}
o::Operation op(unsigned pipe, unsigned cell, unsigned mode,
                bool definite = true) {
  o::Operation a;
  a.pipe = o::Pipe(pipe);
  a.complete = true;
  if (mode)
    a.accesses.push_back(
        {cell, bool(mode & 1), bool(mode & 2), definite && bool(mode & 2)});
  return a;
}
o::Region leaf(unsigned i) { return {o::Region::Operation, {}, i}; }
o::Region seq(std::initializer_list<o::Region> a) {
  return {o::Region::Sequence, a};
}
bool has(const std::vector<o::StorageOrigin> &a, std::size_t s) {
  return std::any_of(a.begin(), a.end(),
                     [&](const auto &x) { return x.site == s; });
}
o::Access access(const o::Program &p, std::size_t s, unsigned c) {
  o::Access a;
  a.cell = c;
  if (s >= p.operations.size())
    return a;
  for (const auto &x : p.operations[s].accesses)
    if (x.cell == c) {
      a.read |= x.read;
      a.write |= x.write;
      a.definiteWrite |= x.definiteWrite;
    }
  return a;
}
bool path(const o::Program &p, const o::detail::ControlGraph &g, std::size_t s,
          std::size_t t, unsigned c) {
  std::vector<bool> seen(g.sites.size());
  std::vector<std::size_t> todo = g.sites[s].successors;
  while (!todo.empty()) {
    auto v = todo.back();
    todo.pop_back();
    if (v == t)
      return true;
    if (seen[v])
      continue;
    seen[v] = true;
    const auto a = access(p, v, c);
    if (a.write && a.definiteWrite)
      continue;
    for (auto n : g.sites[v].successors)
      todo.push_back(n);
  }
  return false;
}
// Independent graph definition of the former flags, deliberately keeping
// path/dominance queries in this test oracle rather than in construction.
bool reaches(const o::detail::ControlGraph& graph, std::vector<std::size_t> pending,
             std::size_t target, std::size_t excluded = o::NoAnalysisId)
{
    std::vector<bool> seen(graph.sites.size());
    while (!pending.empty()) {
        const auto at = pending.back();
        pending.pop_back();
        if (at == excluded || seen[at]) {
            continue;
        }
        if (at == target) {
            return true;
        }
        seen[at] = true;
        for (auto next : graph.sites[at].successors) {
            pending.push_back(next);
        }
    }
    return false;
}
unsigned originalFlags(const o::Program& p, const o::detail::ControlGraph& graph,
                       const o::StorageFrontierAnalysis& analysis, const o::StorageRelationship& r)
{
    unsigned flags = o::AdditionalOverlap;
    const auto s = r.source.site, t = r.target.site;
    if (!reaches(graph, {graph.entry}, s) || !reaches(graph, {graph.entry}, t) || !path(p, graph, s, t, r.cell)) {
        return flags;
    }
    const auto source = access(p, s, r.cell), target = access(p, t, r.cell);
    const bool reuse = target.write && ((r.kind == o::StorageRelationship::WAR && source.read) ||
                                       (r.kind == o::StorageRelationship::WAW && source.write));
    if (reuse && p.cells[r.cell].storage == o::Cell::Storage::CanonicalInterval) {
        flags |= o::KnownReuse;
    }
    const auto writers = analysis.previousWriters(t, r.cell);
    if (r.kind == o::StorageRelationship::RAW && source.write && target.read && source.definiteWrite &&
        writers.size() == 1 && writers.front().site == s && s != t &&
        !reaches(graph, {graph.entry}, t, s) && !reaches(graph, graph.sites[s].successors, s) &&
        !reaches(graph, graph.sites[t].successors, t)) {
        flags |= o::KnownReadiness;
    }
    return flags;
}
void classificationScaling()
{
    for (unsigned count : {4u, 8u, 16u, 32u}) {
        for (bool singleton : {false, true}) {
            auto p = base();
            const auto writers = singleton ? 1u : count;
            for (unsigned i = 0; i < writers; ++i) {
                p.operations.push_back(op(0, 0, 2));
            }
            auto alternatives = leaf(0);
            for (unsigned i = 1; i < writers; ++i) {
                alternatives = {o::Region::Choice, {alternatives, leaf(i)}};
            }
            p.body = seq({alternatives});
            for (unsigned i = 0; i < count; ++i) {
                p.operations.push_back(op(1, 0, 1));
                p.body.children.push_back(leaf(writers + i));
            }
            o::StorageFrontierAnalysis analysis(p);
            CHECK(analysis.complete());
            for (unsigned i = 0; i < count; ++i) {
                for (const auto& r : analysis.relationshipsAt(writers + i)) {
                    const auto flags = analysis.classifyRequirement(r);
                    CHECK(bool(flags & o::KnownReadiness) == singleton);
                }
            }
            const auto& work = analysis.stats();
            CHECK(work.classificationQueries == writers * count);
            CHECK(work.witnessQueries == 0 && work.witnessSites == 0);
            CHECK(work.classificationSites <= 2 * work.staticSites);
            CHECK(work.classificationOrigins <= writers * count);
            std::cout << "classification writers=" << writers << " readers=" << count
                      << " pairs=" << work.classificationQueries << " sites=" << work.classificationSites
                      << " origins=" << work.classificationOrigins << '\n';
        }
    }
}
void sparseMatchesAll(const o::Program &p,
                      const std::vector<unsigned> &visits) {
  using Matrix = std::vector<std::vector<bool>>;
  const auto count = 2 * visits.size();
  Matrix all(count, std::vector<bool>(count)), sparse = all;
  auto edges = [&](const Matrix &input) {
    auto result = input;
    for (std::size_t k = 0; k < count; ++k) {
      for (std::size_t i = 0; i < count; ++i) {
        if (result[i][k]) {
          for (std::size_t j = 0; j < count; ++j)
            result[i][j] = result[i][j] || result[k][j];
        }
      }
    }
    return result;
  };
  for (std::size_t i = 0; i < visits.size(); ++i) {
    all[2 * i][2 * i + 1] = sparse[2 * i][2 * i + 1] = true;
    for (std::size_t j = 0; j < i; ++j)
      for (unsigned c = 0; c < p.cells.size(); ++c) {
        auto a = access(p, visits[j], c), b = access(p, visits[i], c);
        if ((a.read || a.write) && (b.read || b.write) && (a.write || b.write))
          all[2 * j + 1][2 * i] = true;
      }
  }
  for (unsigned c = 0; c < p.cells.size(); ++c) {
    std::size_t writer = o::NoAnalysisId;
    std::vector<std::size_t> readers;
    for (std::size_t i = 0; i < visits.size(); ++i) {
      auto a = access(p, visits[i], c);
      if (!a.read && !a.write)
        continue;
      if (writer != o::NoAnalysisId)
        sparse[2 * writer + 1][2 * i] = true;
      if (a.write) {
        for (auto reader : readers)
          sparse[2 * reader + 1][2 * i] = true;
        readers.clear();
        writer = i;
      } else
        readers.push_back(i);
    }
  }
  CHECK(edges(all) == edges(sparse));
}

} // namespace
int main() {
  classificationScaling();
  {
    auto p = base();
    p.operations = {op(0, 0, 2), op(1, 0, 1), op(1, 0, 1), op(0, 0, 2)};
    auto before = o::analyze(p);
    o::StorageFrontierAnalysis f(p);
    CHECK(f.complete());
    CHECK(f.previousReaders(3, 0).size() == 2);
    CHECK(f.previousWriters(3, 0).size() == 1);
    CHECK(!f.readerBoundary(2, 0, o::Pipe(1), true).boundary);
    auto after = o::analyze(p);
    CHECK(before.residuals.size() == after.residuals.size());
    for (std::size_t i = 0; i < before.cuts.size(); ++i)
      CHECK(before.cuts[i].beforeIssue->pending ==
            after.cuts[i].beforeIssue->pending);
  }
  {
    auto p = base();
    p.operations = {op(0, 0, 2), op(1, 0, 1), op(0, 0, 2)};
    p.body = seq({leaf(0), {o::Region::For, {leaf(1)}}, leaf(2)});
    o::StorageFrontierAnalysis f(p);
    CHECK(f.readerBoundary(1, 0, o::Pipe(1), true).ambiguous());
    auto w = f.witness(1, 1, 0);
    CHECK(w.exists && w.sites.size() > 1 && !w.crossedLoopOwners.empty());
  }
  {
    auto p = base();
    p.operations = {op(0, 0, 2), op(1, 0, 3, false), op(2, 0, 1)};
    o::StorageFrontierAnalysis weak(p);
    CHECK(has(weak.previousWriters(2, 0), 0));
    CHECK(!weak.witness(0, 2, 0).definiteWriteFree);
    p.operations[1].accesses[0].definiteWrite = true;
    o::StorageFrontierAnalysis strong(p);
    CHECK(!has(strong.previousWriters(2, 0), 0));
    CHECK(has(strong.previousWriters(1, 0), 0));
  }
  std::mt19937 rng(0x4A01u);
  for (unsigned trial = 0; trial < 180; ++trial) {
    auto p = base();
    for (unsigned i = 0; i < 6; ++i) {
      const auto a = rng() % 3, c = rng() % 3, m = rng() % 4;
      p.operations.push_back(op(a, c, m, trial % 3 != 0));
    }
    p.body = seq(
        {leaf(0),
         {o::Region::For,
          {seq({leaf(1), {o::Region::Choice, {leaf(2), leaf(3)}}, leaf(4)})}},
         leaf(5)});
    if (trial % 2)
      p.body = seq(
          {leaf(0),
           {o::Region::While,
            {seq({leaf(1), {o::Region::Choice, {leaf(2), leaf(3)}}}), leaf(4)}},
           leaf(5)});
    for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
        p.cells[cell].storage = o::Cell::Storage::CanonicalInterval;
        p.cells[cell].coordinateSpace = "physical";
        p.cells[cell].ranges = {{cell * 16u, 16}};
    }
    auto g = o::detail::buildControlGraph(p);
    o::StorageFrontierAnalysis f(p);
    CHECK(f.complete());
    if (trial % 3)
      for (const auto &walk : oahs_oracle::traces(p, 2))
        sparseMatchesAll(p, walk);
    for (unsigned c = 0; c < p.cells.size(); ++c)
      for (std::size_t s = 0; s < 6; ++s)
        for (std::size_t t = 0; t < 6; ++t) {
          const auto a = access(p, s, c), b = access(p, t, c);
          const bool exists = path(p, g, s, t, c);
          if (a.read || a.write) {
              for (auto kind : {o::StorageRelationship::RAW, o::StorageRelationship::WAR,
                                o::StorageRelationship::WAW}) {
                  o::StorageRelationship r{kind, c, {s}, {t}};
                  CHECK(f.classifyRequirement(r) == originalFlags(p, g, f, r));
              }
          }
          if (a.write) {
            CHECK(has(f.previousWriters(t, c), s) == exists);
            ++relations;
          }
          if (a.read && !a.write) {
            CHECK(has(f.previousReaders(t, c), s) == exists);
            ++relations;
          }
          if (b.write)
            CHECK(has(f.nextWriters(s, c), t) == exists);
          if (b.read && !b.write)
            CHECK(has(f.nextReaders(s, c), t) == exists);
          if ((a.read || a.write) && exists) {
            auto w = f.witness(s, t, c);
            CHECK(w.exists && w.sites.front() == s && w.sites.back() == t &&
                  w.sites.size() > 1);
            for (std::size_t k = 0; k + 1 < w.sites.size(); ++k) {
              const auto &next = g.sites[w.sites[k]].successors;
              CHECK(std::find(next.begin(), next.end(), w.sites[k + 1]) !=
                    next.end());
            }
          }
        }
  }
  for (unsigned i = 0; i < 1000; ++i) {
    auto make = [&]() {
      o::SuccessionSummary s;
      s.preservesIncoming = rng() % 2;
      for (unsigned j = 0; j < 4; ++j) {
        if (rng() % 2)
          s.writers.push_back(j);
        if (rng() % 2)
          s.readers.push_back(j + 4);
      }
      return s;
    };
    auto a = make(), b = make();
    std::vector<std::size_t> w{12, 13}, r{14, 15};
    auto first = a.apply(w, r), twice = b.apply(first.first, first.second);
    CHECK(o::SuccessionSummary::sequence(a, b).apply(w, r) == twice);
    CHECK(a.apply(first.first, first.second) == first);
    CHECK(o::SuccessionSummary::repeat(a).apply(w, r) ==
          o::SuccessionSummary::choice(a, o::SuccessionSummary{}).apply(w, r));
  }
  std::cout << checks << " storage assertions; " << relations
            << " origin queries\n";
}
