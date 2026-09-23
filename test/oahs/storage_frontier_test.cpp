// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "../../lib/PTO/Transforms/OAHS/Control.h"
#include "GraphOracle.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <functional>
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

// The origin enumerator is deliberately independent of the compact fixed point.
void summaryMatchesOrigins(const o::Program& p, const o::detail::ControlGraph& graph,
                           o::StorageFrontierAnalysis& facts) {
  for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
    for (std::size_t site = 0; site < graph.sites.size(); ++site) {
      for (bool backward : {false, true}) {
        for (const auto& stops : {std::vector<std::size_t>{}, std::vector<std::size_t>{site}}) {
          const auto& exact = facts.nearestUses({site}, cell, stops, backward);
          unsigned roles = 0;
          for (const auto& origin : exact.accesses) {
            const auto a = access(p, origin.operation, cell);
            roles |= a.read && a.write ? o::PhysicalUseSummary::ReadWrite :
                a.read ? o::PhysicalUseSummary::Read :
                a.definiteWrite ? o::PhysicalUseSummary::FullWrite : o::PhysicalUseSummary::PartialWrite;
          }
          for (auto boundary : exact.boundaries) {
            roles |= std::find(stops.begin(), stops.end(), boundary) != stops.end() ?
                o::PhysicalUseSummary::IntervalStop :
                backward ? o::PhysicalUseSummary::Entry : o::PhysicalUseSummary::Exit;
          }
          o::OriginalUseQuery query;
          query.cell = cell;
          query.starts.assign(1, site);
          query.stops = stops;
          query.backward = backward;
          const auto summary = facts.nearestUseSummary(query);
          CHECK(summary.complete() && summary.roles == roles);
        }
        o::OriginalUseQuery exclusive;
        exclusive.cell = cell;
        exclusive.starts.assign(1, site);
        exclusive.backward = backward;
        exclusive.includeStarts = false;
        o::OriginalUseQuery inclusive = exclusive;
        inclusive.includeStarts = true;
        inclusive.starts.clear();
        if (backward) {
          for (std::size_t from = 0; from < graph.sites.size(); ++from) {
            for (auto to : graph.sites[from].successors) {
              if (to == site) { inclusive.starts.push_back(from); }
            }
          }
        } else { inclusive.starts = graph.sites[site].successors; }
        auto expected = facts.nearestUseSummary(inclusive).roles;
        const bool boundary = inclusive.starts.empty() || (backward && site == graph.entry);
        if (boundary) {
          expected = backward ? o::PhysicalUseSummary::Entry : o::PhysicalUseSummary::Exit;
        }
        CHECK(facts.nearestUseSummary(exclusive).roles == expected);

      }
    }
  }
}
void summarySparseClosure() {
  auto p = base();
  o::ObservedControl graph;
  graph.qualification = "test original alternative with a closed access-free cycle";
  graph.scopes = {{0, o::NoControlId, o::NoControlId}};
  graph.entry = 0;
  graph.exit = 2;
  graph.sites.resize(3);
  graph.sites[0].successors = {1, 2};
  graph.sites[1].successors = {1};
  for (std::size_t site = 0; site < 3; ++site) {
    graph.sites[site].observation = site;
    graph.observations.push_back({site, {}, true});
  }
  p.observed = graph;
  o::StorageFrontierAnalysis facts(p);
  CHECK(facts.complete());
  o::OriginalUseQuery query;
  query.starts.assign(1, 1);
  auto cycle = facts.nearestUseSummary(query);
  CHECK(cycle.status == o::PhysicalUseSummary::Status::NoHit && cycle.roles == 0);
  const auto work = facts.stats().useSummarySites;
  CHECK(facts.nearestUseSummary(query).roles == 0 && facts.stats().useSummarySites == work);
  query.starts.assign(1, 0);
  CHECK(facts.nearestUseSummary(query).roles == o::PhysicalUseSummary::Exit);
  CHECK(facts.stats().useSummarySolves == 2);
  const auto joinedWork = facts.stats().useSummarySites;
  query.starts.assign(1, 1);
  CHECK(facts.nearestUseSummary(query).roles == 0 && facts.stats().useSummarySites == joinedWork);

  p = base();
  p.operations.push_back(op(1, 0, 1));
  for (unsigned i = 0; i < 256; ++i) { p.operations.push_back(op(0, 1, 0)); }
  o::StorageFrontierAnalysis sparse(p);
  query.starts.assign(1, 0);
  CHECK(sparse.nearestUseSummary(query).roles == o::PhysicalUseSummary::Read);
  CHECK(sparse.stats().useSummarySites == 2 && sparse.stats().useSummaryEdges == 0);
}

void summaryIntervalsAndScaling() {
  for (unsigned length : {16u, 64u, 256u}) {
    auto p = base();
    for (unsigned i = 0; i < length; ++i) { p.operations.push_back(op(0, 1, 0)); }
    p.operations.push_back(op(1, 0, 1));
    o::StorageFrontierAnalysis facts(p);
    o::OriginalUseQuery query;
    query.starts.assign(1, 0);
    CHECK(facts.nearestUseSummary(query).roles == o::PhysicalUseSummary::Read);
    const auto work = facts.stats().useSummarySites;
    for (unsigned i = 0; i < length; ++i) {
      query.starts.assign(1, i);
      CHECK(facts.nearestUseSummary(query).roles == o::PhysicalUseSummary::Read);
    }
    CHECK(facts.stats().useSummarySolves == 1 && facts.stats().useSummarySites == work);
    CHECK(work <= 3 * (length + 2));
    query.starts.assign(1, 0);
    query.stops.assign(1, length);
    CHECK(facts.nearestUseSummary(query).roles == o::PhysicalUseSummary::IntervalStop);
    query.includeStops = true;
    CHECK(facts.nearestUseSummary(query).roles == o::PhysicalUseSummary::Read);
    query.starts.assign(1, length);
    query.includeStarts = false;
    const auto empty = facts.nearestUseSummary(query);
    CHECK(empty.status == o::PhysicalUseSummary::Status::NoHit &&
          empty.roles == o::PhysicalUseSummary::IntervalStop);
    query.starts.assign(1, length + 100);
    CHECK(!facts.nearestUseSummary(query).complete());
    query.starts.assign(1, 0);
    query.owner = length + 100;
    CHECK(!facts.nearestUseSummary(query).complete());
    query.owner = o::NoAnalysisId;
    query.occurrence = length + 100;
    CHECK(!facts.nearestUseSummary(query).complete());
  }
}

// Interpret the returned DAG with occurrence-local atoms. The test deliberately
// changes the choice value between loop visits rather than treating it as a
// globally invariant Boolean.
bool predicate(const o::StorageFrontierAnalysis& facts, std::size_t id,
               const std::function<bool(const o::ObservationAtom&)>& value) {
  const auto p = facts.participationExpression(id);
  switch (p.kind) {
  case o::ParticipationExpression::Invalid: CHECK(false); return false;
  case o::ParticipationExpression::False: return false;
  case o::ParticipationExpression::True: return true;
  case o::ParticipationExpression::Atom: return value(p.atom);
  case o::ParticipationExpression::Not: return !predicate(facts, p.left, value);
  case o::ParticipationExpression::And: return predicate(facts, p.left, value) && predicate(facts, p.right, value);
  case o::ParticipationExpression::Or: return predicate(facts, p.left, value) || predicate(facts, p.right, value);
  }
  return false;
}
std::set<std::size_t> frontier(const o::StorageFrontierAnalysis& facts, std::size_t id,
               const std::function<bool(const o::ObservationAtom&)>& value) {
  const auto f = facts.guardedReadFrontier(id);
  if (f.kind == o::GuardedReadFrontier::Empty) { return {}; }
  if (f.kind == o::GuardedReadFrontier::Access) { return {f.operation}; }
  if (f.kind == o::GuardedReadFrontier::Guard) {
    return predicate(facts, f.predicate, value) ? frontier(facts, f.left, value) : std::set<std::size_t>{};
  }
  auto result = frontier(facts, f.left, value);
  const auto right = frontier(facts, f.right, value);
  result.insert(right.begin(), right.end());
  return result;
}
void originalReadIntervals() {
  auto p = base();
  p.operations = {op(1, 0, 1), op(1, 0, 1), op(0, 0, 2)};
  o::Region a{o::Region::For, {leaf(0)}}, b{o::Region::For, {leaf(1)}};
  a.originalOwner = 10; b.originalOwner = 20;
  a.qualifiedCounted = b.qualifiedCounted = true;
  a.zeroTripPossible = b.zeroTripPossible = true;
  p.body = seq({a, b, leaf(2)});
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis facts(p);
  o::ReaderIntervalQuery q{o::NoAnalysisId, 0, o::Pipe(1), o::ReaderIntervalQuery::BodyInterval, 0, 2};
  const auto& result = facts.originalReaderFrontiers(q);
  CHECK(result.status == o::OriginalReaderFrontiers::Status::Exact);
  for (unsigned mask = 0; mask != 4; ++mask) {
    auto value = [&](const o::ObservationAtom& atom) {
      if (atom.kind == o::ObservationAtom::LoopNonEmpty) { return bool(mask & (atom.owner == 10 ? 1 : 2)); }
      return true; // each participating child has a single visit
    };
    CHECK(predicate(facts, result.nonempty, value) == bool(mask));
    // A frontier inside a zero-trip child is unreachable; filter by original
    // participation exactly as an actual observation would be.
    auto first = frontier(facts, result.first, value), last = frontier(facts, result.last, value);
    if (!(mask & 1)) { first.erase(0); last.erase(0); }
    if (!(mask & 2)) { first.erase(1); last.erase(1); }
    CHECK(first == (mask ? std::set<std::size_t>{mask & 1 ? 0u : 1u} : std::set<std::size_t>{}));
    CHECK(last == (mask ? std::set<std::size_t>{mask & 2 ? 1u : 0u} : std::set<std::size_t>{}));
  }
  const auto work = facts.stats().originalReadRegions;
  CHECK(&result == &facts.originalReaderFrontiers(q));
  CHECK(work == facts.stats().originalReadRegions);
  q.end = 3;
  CHECK(!facts.originalReaderFrontiers(q).complete()); // reload ends the episode
  q.owner = 10; q.scope = o::ReaderIntervalQuery::WholeRegion; q.end = o::NoAnalysisId;
  CHECK(facts.originalReaderFrontiers(q).separatesVisits);
  // Refinement may change a current operation; original facts remain immutable.
  p.operations[0].accesses.clear();
  o::StorageFrontierAnalysis refined(p);
  CHECK(!refined.originalReaderFrontiers(q).complete());
  CHECK(p.originalStructure->operations[0].accesses.size() == 1);
  p.originalOperations.clear();
  o::StorageFrontierAnalysis malformed(p);
  CHECK(!malformed.originalReaderFrontiers(q).complete());

  p = base(); p.operations = {op(1, 0, 1), op(1, 0, 1)};
  o::Region choice{o::Region::Choice, {leaf(0), leaf(1)}};
  choice.originalOwner = 30;
  a.children = {choice}; p.body = seq({a});
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis alternatives(p);
  const auto& both = alternatives.originalReaderFrontiers(q);
  CHECK(both.status == o::OriginalReaderFrontiers::Status::Exact);
  for (unsigned visit = 0; visit != 4; ++visit) {
    auto value = [&](const o::ObservationAtom& atom) {
      if (atom.kind == o::ObservationAtom::OriginalBoolean) { return bool(visit & 1); }
      if (atom.kind == o::ObservationAtom::LoopHasPrevious) { return visit == 0; }
      if (atom.kind == o::ObservationAtom::LoopHasNext) { return visit == 3; }
      return true;
    };
    CHECK(frontier(alternatives, both.first, value) ==
          (visit == 0 ? std::set<std::size_t>{1} : std::set<std::size_t>{}));
    CHECK(frontier(alternatives, both.last, value) ==
          (visit == 3 ? std::set<std::size_t>{0} : std::set<std::size_t>{}));
  }
  p.originalStructure.reset(); p.operations[1].accesses.clear();
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis conditional(p);
  CHECK(!conditional.originalReaderFrontiers(q).complete());

  p = base(); p.operations = {op(1, 0, 1)};
  CHECK(o::captureOriginalStructure(p).success); // implicit flat normalization
  o::StorageFrontierAnalysis flat(p);
  CHECK(flat.originalReaderFrontiers({o::NoAnalysisId, 0, o::Pipe(1)}).nonempty == 1);
  auto imported = o::addStructuredBoundaryCuts(p);
  CHECK(imported.success && o::hasOriginalIdentityMap(imported.program));
  auto periodic = o::makePeriodicLoop(p, 2, {});
  CHECK(periodic.success && !periodic.program.originalStructure);
  p.originalStructure.reset(); p.cells[0].unknownRange = true;
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis uncertain(p);
  CHECK(!uncertain.originalReaderFrontiers({o::NoAnalysisId, 0, o::Pipe(1)}).complete());
}

void originalReadRefusalsAndScaling() {
  auto p = base(); p.operations = {op(1, 0, 1)};
  p.operations[0].accesses.push_back({1, true, false});
  p.body = {o::Region::For, {leaf(0)}};
  auto observed = o::addStructuredBoundaryCuts(p);
  CHECK(observed.success);
  p = observed.program;
  p.originalStructure.reset();
  p.physicalUses = {{p.body.originalOwner, 2, {{0, {{{1, true, false}}, {{1, true, false}}}}}}};
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis variants(p);
  CHECK(!variants.originalReaderFrontiers({o::NoAnalysisId, 0, o::Pipe(1)}).complete());
  CHECK(variants.originalReaderFrontiers({o::NoAnalysisId, 1, o::Pipe(1)}).nonempty == 1);
  CHECK(variants.participationExpression(o::NoAnalysisId).kind == o::ParticipationExpression::Invalid);
  CHECK(variants.guardedReadFrontier(o::NoAnalysisId).kind == o::GuardedReadFrontier::Invalid);
  p.operations[0].accesses.push_back({2, true, false});
  o::StorageFrontierAnalysis changed(p);
  CHECK(!changed.originalReaderFrontiers({o::NoAnalysisId, 2, o::Pipe(1)}).complete());
  p = base(); p.operations = {op(1, 0, 1), op(1, 0, 1)};
  o::Region a{o::Region::Choice, {leaf(0), seq({})}}, b{o::Region::Choice, {seq({}), leaf(1)}};
  a.originalOwner = b.originalOwner = 8;
  p.body = seq({a, b});
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis ambiguous(p);
  CHECK(!ambiguous.originalReaderFrontiers({o::NoAnalysisId, 0, o::Pipe(1)}).complete());
  CHECK(ambiguous.originalReaderFrontiers({o::NoAnalysisId, 2, o::Pipe(1)}).status ==
        o::OriginalReaderFrontiers::Status::NoHit);
  p.originalStructure.reset(); p.operations[1].accesses = {{1, true, false}};
  p.body.children[1].originalOwner = o::NoControlId;
  CHECK(o::captureOriginalStructure(p).success);
  o::StorageFrontierAnalysis unrelated(p);
  CHECK(unrelated.originalReaderFrontiers({o::NoAnalysisId, 0, o::Pipe(1)}).complete());
  for (unsigned size : {16u, 32u, 64u}) {
    p = base(); p.body = seq({});
    for (unsigned i = 0; i < size; ++i) {
      p.operations.push_back(op(1, 0, 1));
      o::Region choice{o::Region::Choice, {leaf(i), seq({})}};
      choice.originalOwner = i;
      p.body.children.push_back(choice);
    }
    CHECK(o::captureOriginalStructure(p).success);
    o::StorageFrontierAnalysis many(p);
    o::ReaderIntervalQuery q{o::NoAnalysisId, 0, o::Pipe(1)};
    CHECK(many.originalReaderFrontiers(q).status == o::OriginalReaderFrontiers::Status::Exact);
    const auto work = many.stats();
    CHECK(work.originalReadRegions == 3 * size + 1);
    CHECK(work.participationNodes < 12 * size && work.readFrontierNodes < 12 * size);
    CHECK(work.readCompositionParts == size);
    for (unsigned repeat = 0; repeat != size; ++repeat) { CHECK(many.originalReaderFrontiers(q).complete()); }
    CHECK(many.stats().originalReadRegions == work.originalReadRegions);
    CHECK(many.stats().participationNodes == work.participationNodes);
    CHECK(many.stats().readCompositionParts == work.readCompositionParts);
  }
}

void readSegmentIndexScaling() {
  for (unsigned size : {16u, 32u, 64u}) {
    auto p = base(); p.operations = {op(0, 0, 2)};
    auto body = seq({leaf(0)});
    for (unsigned i = 1; i <= size; ++i) {
      p.operations.push_back(op(1, 0, 1)); body.children.push_back(leaf(i));
    }
    p.operations.push_back(op(0, 0, 2)); body.children.push_back(leaf(size + 1));
    p.body = {o::Region::For, {body}};
    p.body.originalOwner = 100; p.body.qualifiedCounted = true;
    CHECK(o::captureOriginalStructure(p).success);
    o::StorageFrontierAnalysis facts(p);
    for (unsigned i = 1; i <= size; ++i) {
      const auto& segment = facts.originalReadSegment(100, i, 0, o::Pipe(1));
      CHECK(segment.complete && !segment.startsAtOwnerEntry && !segment.endsAtOwnerExit);
      CHECK(segment.interval.begin == 1 && segment.interval.end == size + 1);
      const auto& frontiers = facts.originalReaderFrontiers(segment.interval);
      CHECK(frontiers.complete());
      CHECK(facts.readerFrontierCondition(frontiers.first, i) == unsigned(i == 1));
      CHECK(facts.readerFrontierCondition(frontiers.last, i) == unsigned(i == size));
    }
    const auto work = facts.stats().readCompositionParts;
    CHECK(work < 8 * size);
    for (unsigned i = 1; i <= size; ++i) {
      const auto& segment = facts.originalReadSegment(100, i, 0, o::Pipe(1));
      const auto& frontiers = facts.originalReaderFrontiers(segment.interval);
      CHECK(facts.readerFrontierCondition(frontiers.first, i) == unsigned(i == 1));
    }
    CHECK(facts.stats().readCompositionParts == work);
  }
}

} // namespace
int main() {
  originalReadIntervals();
  readSegmentIndexScaling();
  originalReadRefusalsAndScaling();
  classificationScaling();
  summaryIntervalsAndScaling();
  summarySparseClosure();
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
    summaryMatchesOrigins(p, g, f);
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
