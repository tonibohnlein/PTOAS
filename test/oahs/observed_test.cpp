// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "GraphOracle.h"
#include "ObservedFixtures.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <cstdlib>
#include <functional>
#include <iostream>
#include <set>
namespace o = mlir::pto::oahs;
namespace {
std::size_t checks = 0, concrete = 0, frontEndPaths = 0;
void check(bool b, unsigned line) {
  ++checks;
  if (!b) {
    std::cerr << "observed check " << line << '\n';
    std::abort();
  }
}
#define CHECK(x) check(bool(x), __LINE__)
// Test-only concrete paths. Production never expands dynamic visits.
void paths(const o::Program &p, unsigned maxPayload,
           const std::function<void(const std::vector<std::size_t> &)> &f) {
  const auto &q = *p.observed;
  std::vector<std::size_t> path;
  std::function<void(std::size_t, unsigned)> dfs = [&](std::size_t at,
                                                       unsigned count) {
    count += q.sites[at].operation != o::NoControlId;
    if (count > maxPayload)
      return;
    // Fixtures contain a payload on each cycle. This is a test-data invariant.
    CHECK(path.size() < 16 * (maxPayload + 1) + q.sites.size());
    path.push_back(at);
    if (at == q.exit)
      f(path);
    else
      for (auto next : q.sites[at].successors)
        dfs(next, count);
    path.pop_back();
  };
  dfs(q.entry, 0);
}
void oracle(const o::Program &p, const o::Commands &commands,
            const std::vector<std::size_t> &path) {
  o::Program concreteProgram = p;
  concreteProgram.observed.reset();
  concreteProgram.body = {};
  concreteProgram.operations.clear();
  o::Commands word;
  std::vector<o::Command> pending;
  std::vector<unsigned> visits;
  for (auto at : path) {
    pending.insert(pending.end(), commands[at].begin(), commands[at].end());
    const auto op = p.observed->sites[at].operation;
    if (op != o::NoControlId) {
      word.push_back(std::move(pending));
      pending.clear();
      visits.push_back(visits.size());
      concreteProgram.operations.push_back(p.operations[op]);
    }
  }
  word.push_back(std::move(pending));
  ++concrete;
  CHECK(oahs_oracle::graph(concreteProgram, word, visits));
}
void ringFrontend(const o::Program &p, unsigned b, bool stride, unsigned maxN) {
  std::vector<unsigned> byN(maxN + 1);
  paths(p, 2 * maxN, [&](const auto &path) {
    unsigned total = 0;
    for (auto at : path)
      total += p.observed->sites[at].operation != o::NoControlId;
    CHECK(total % 2 == 0);
    const unsigned n = total / 2;
    ++byN[n];
    ++frontEndPaths;
    unsigned iteration = 0, position = 0;
    for (auto at : path) {
      const auto &node = p.observed->sites[at];
      if (node.observation != o::NoControlId) {
        const auto &obs = p.observed->observations[node.observation];
        if (!obs.atoms.empty())
          for (const auto &a : obs.atoms) {
            if (a.kind == o::ObservationAtom::LoopResidue)
              CHECK(a.value == iteration % b);
            else if (a.kind == o::ObservationAtom::LoopHasPrevious)
              CHECK(a.value == (iteration >= b));
            else if (a.kind == o::ObservationAtom::LoopHasNext)
              CHECK(a.value == (iteration + b < n));
            else
              CHECK(false);
          }
        if (obs.anchor == 2 && !obs.atoms.empty()) {
          CHECK(position == 2);
          position = 0;
          ++iteration;
        }
      }
      if (node.operation != o::NoControlId) {
        CHECK(node.operation == 2 * (iteration % b) + position);
        const auto &access = p.operations[node.operation].accesses[0];
        CHECK(access.cell ==
              (stride ? (2 * iteration + 1) % 4 : iteration % b));
        ++position;
      }
    }
    CHECK(iteration == n);
  });
  for (auto count : byN)
    CHECK(count == 1); // all and only finite normalized lengths
}
} // namespace
int main() {
  for (unsigned b = 1; b <= 4; ++b) {
    auto imported = observed_fixtures::ring(b);
    CHECK(imported.success);
    const auto &p = imported.program;
    ringFrontend(p, b, false, b + 5);
    auto r = o::constructSelectedPlan(p);
    CHECK(r.success && o::analyze(p, r.commands).verified());
    auto schema = o::exportObservedSchema(p, r.commands);
    CHECK(schema.complete);
    auto readback = o::reconstructObservedSchema(p, schema);
    CHECK(readback.success);
    CHECK(!schema.correspondence.empty());
    auto wrongMatching = schema;
    wrongMatching.correspondence.front().publications.clear();
    CHECK(!o::reconstructObservedSchema(p, wrongMatching).success);
    paths(p, 2 * (b + 4),
          [&](const auto &path) { oracle(p, readback.commands, path); });
    auto broken = schema;
    CHECK(!broken.words.empty());
    broken.words[0].observation.available = false;
    CHECK(!o::reconstructObservedSchema(p, broken).success);
    auto nonuniform = r.commands;
    bool mutated = false;
    for (o::Cut c = 0; c < nonuniform.size(); ++c)
      if (o::canonicalCommandCut(p, c) != c && !nonuniform[c].empty()) {
        nonuniform[c].clear();
        mutated = true;
        break;
      }
    if (mutated)
      CHECK(!o::analyze(p, nonuniform).complete);
  }
  {
    auto imported = observed_fixtures::ring(2, true);
    CHECK(imported.success);
    ringFrontend(imported.program, 2, true, 8);
    // Strided frontend semantics do not imply admission by the isolated
    // two-role cyclic constructor. Check the original finite quotient itself.
    CHECK(o::analyze(imported.program).complete);
  }
  for (unsigned b = 1; b <= 2; ++b) {
    auto imported = observed_fixtures::refinedRing(b);
    CHECK(imported.success);
    const auto &p = imported.program;
    // Isolate observation-schema validation from synchronization synthesis.
    auto schemaProgram = p;
    for (auto& operation : schemaProgram.operations)
      operation.accesses.clear();
    auto schema = o::exportObservedSchema(schemaProgram, o::Commands(o::commandCutCount(p)));
    CHECK(schema.complete && o::analyze(p).complete);
    std::set<std::size_t> used;
    for (const auto &site : p.observed->sites)
      if (site.observation != o::NoControlId)
        used.insert(site.observation);
    bool orphan = false;
    for (std::size_t id = 0; id < p.observed->observations.size(); ++id)
      if (!used.count(id)) {
        auto bad = schema;
        bad.words.push_back({p.observed->observations[id],
                             {{o::Command::Barrier, o::Pipe(0)}}});
        CHECK(!o::reconstructObservedSchema(schemaProgram, bad).success);
        orphan = true;
        break;
      }
    CHECK(orphan);
    std::vector<unsigned> counts(8);
    paths(p, 14, [&](const auto &path) {
      unsigned n = 0;
      for (auto at : path)
        n += p.observed->sites[at].operation != o::NoControlId;
      CHECK(n % 2 == 0);
      ++counts[n / 2];
      unsigned step = 0;
      for (auto at : path) {
        const auto op = p.observed->sites[at].operation;
        if (op == o::NoControlId)
          continue;
        CHECK(op == (step / 2 % b) * 2 + step % 2);
        ++step;
      }
    });
    for (auto c : counts)
      CHECK(c == 1);
  }
  { // A shared word can have an unreachable first representative.
    auto p = observed_fixtures::target(2, 1);
    p.operations = {observed_fixtures::op(0, 0, 2),
                    observed_fixtures::op(1, 0, 1)};
    o::ObservedControl q;
    q.qualification = "test-only original CFG";
    q.entry = 1;
    q.exit = 4;
    q.sites.resize(5);
    q.observations = {
        {100, {}, true}, {101, {}, true}, {102, {}, true}, {103, {}, true}};
    q.sites[0].observation = q.sites[2].observation = 0;
    q.sites[0].successors = {4};
    q.sites[1].operation = 0;
    q.sites[1].observation = 1;
    q.sites[1].successors = {2};
    q.sites[2].successors = {3};
    q.sites[3].operation = 1;
    q.sites[3].observation = 2;
    q.sites[3].successors = {4};
    q.sites[4].observation = 3;
    p.observed = q;
    o::PrefixQuery prefixes(p);
    CHECK(prefixes.inspectPrefix(o::Pipe(0), 0, 3).selectable());
    o::Commands c(5);
    c[0] = c[2] = {{o::Command::Publish, o::Pipe(0), o::Pipe(1), 0}};
    c[3] = {{o::Command::Acquire, o::Pipe(0), o::Pipe(1), 0}};
    CHECK(o::verify(p, c).success);
    // Removing the only reachable representative must not make a dead word an
    // admissible emission at its physical original anchor.
    p.observed->sites[1].successors = {3};
    CHECK(!o::analyze(p, c).complete);
  }
  { // These are semantic vocabulary errors even if no payload hazard exists.
    auto x = observed_fixtures::ring(2);
    CHECK(x.success);
    auto &p = x.program;
    o::Commands empty(o::commandCutCount(p));
    bool illegal = false, nonuniform = false;
    for (o::Cut c = 0; c < empty.size(); ++c) {
      if (!o::legalCommandCut(p, c) && !illegal) {
        auto bad = empty;
        bad[c].push_back({o::Command::Barrier, o::Pipe(0)});
        CHECK(!o::analyze(p, bad).complete);
        illegal = true;
      }
      if (o::canonicalCommandCut(p, c) != c && !nonuniform) {
        auto bad = empty;
        bad[c].push_back({o::Command::Barrier, o::Pipe(0)});
        CHECK(!o::analyze(p, bad).complete);
        nonuniform = true;
      }
    }
    CHECK(illegal && nonuniform);
    auto q = observed_fixtures::target(1, 1);
    q.operations = {observed_fixtures::op(0, 0, 1)};
    CHECK(!o::makePeriodicLoop(q, std::numeric_limits<unsigned>::max(), {})
               .success);
  }
  for (auto imported : {observed_fixtures::nested(), observed_fixtures::mixed(),
                        observed_fixtures::independent()}) {
    CHECK(imported.success);
    const auto report = o::analyze(imported.program);
    CHECK(report.complete && !report.residuals.empty());
    bool boundary = false;
    for (o::Cut c = 0; c < o::commandCutCount(imported.program); ++c)
      boundary |= o::operationAtCut(imported.program, c) == o::NoControlId &&
                  o::legalCommandCut(imported.program, c);
    CHECK(boundary);
  }

  {
    auto x = observed_fixtures::ring(2);
    CHECK(x.success);
    auto &q = *x.program.observed;
    std::size_t first = o::NoControlId, second = o::NoControlId;
    for (std::size_t i = 0; i < q.observations.size(); ++i)
      if (!q.observations[i].atoms.empty()) {
        if (first == o::NoControlId)
          first = i;
        else if (q.observations[i].anchor == q.observations[first].anchor) {
          second = i;
          break;
        }
      }
    CHECK(second != o::NoControlId);
    q.observations[second].atoms.pop_back();
    CHECK(!o::validateProgram(x.program).success);
  }
  {
    auto x = observed_fixtures::ring(1);
    CHECK(x.success);
    auto p = x.program;
    p.observed->observations[0].available = false;
    CHECK(!o::validateProgram(p).success);
    p = x.program;
    p.observed->observations[0].atoms = {
        {o::ObservationAtom::Kind(99), 0, 0, 0}};
    CHECK(!o::validateProgram(p).success);
    p = x.program;
    p.observed->sites[p.observed->entry].successors = {
        p.observed->sites.size()};
    CHECK(!o::validateProgram(p).success);
    p = x.program;
    p.observed->qualification.clear();
    CHECK(!o::validateProgram(p).success);
  }
  { // A checked initializer must dominate every entry to the replaced loop.
    auto p = observed_fixtures::target(2, 1);
    p.operations = {observed_fixtures::op(0, 0, 2),
                    observed_fixtures::op(1, 0, 1)};
    p.body = {o::Region::For,
              {observed_fixtures::seq({observed_fixtures::leaf(0),
                                       observed_fixtures::leaf(1)})}};
    auto original = o::addStructuredBoundaryCuts(p);
    CHECK(original.success);
    const auto &q = *original.program.observed;
    o::CountedLoopRegion spec;
    spec.owner = q.entry;
    spec.header = q.sites[spec.owner].successors.front();
    spec.bodyEntry = q.sites[spec.header].successors.front();
    spec.continuation = q.sites[spec.header].successors.back();
    std::vector<bool> seen(q.sites.size());
    std::vector<std::size_t> pending{spec.bodyEntry};
    while (!pending.empty()) {
      const auto at = pending.back();
      pending.pop_back();
      if (at == spec.header || seen[at])
        continue;
      seen[at] = true;
      spec.bodySites.push_back(at);
      for (auto next : q.sites[at].successors)
        pending.push_back(next);
    }
    CHECK(o::refineCountedLoop(original.program, spec).success);
    auto malformedIdentity = original.program;
    malformedIdentity.originalOperations.clear();
    auto withoutIdentity = o::refineCountedLoop(malformedIdentity, spec);
    CHECK(withoutIdentity.success && !withoutIdentity.program.originalStructure);
    auto bypassesBody = [](const o::Program& p) {
      const auto& graph = *p.observed;
      std::vector<bool> seen(graph.sites.size());
      std::vector<std::size_t> todo{graph.entry};
      while (!todo.empty()) {
        const auto at = todo.back(); todo.pop_back();
        if (at == graph.exit) return true;
        if (seen[at] || graph.sites[at].operation != o::NoAnalysisId) continue;
        seen[at] = true;
        for (auto next : graph.sites[at].successors) todo.push_back(next);
      }
      return false;
    };
    CHECK(bypassesBody(o::refineCountedLoop(original.program, spec).program));
    spec.atLeastOnce = true;
    const auto nonempty = o::refineCountedLoop(original.program, spec);
    CHECK(nonempty.success);
    CHECK(!bypassesBody(nonempty.program));
    spec.atLeastOnce = false;

    for (unsigned variant = 0; variant < 3; ++variant) {
      auto bad = original.program;
      if (variant == 0)
        bad.observed->entry = spec.header;
      else if (variant == 1)
        bad.observed->entry = spec.bodyEntry;
      else {
        const auto incoming = bad.observed->sites.size();
        bad.observed->sites.emplace_back();
        bad.observed->sites.back().successors = {spec.owner, spec.header};
        bad.observed->entry = incoming;
      }
      CHECK(o::validateProgram(bad).success); // valid CFG, not a normalized loop
      const auto before = o::analyze(bad);
      CHECK(before.complete && !before.residuals.empty());
      const auto refinement = o::refineCountedLoop(bad, spec);
      CHECK(!refinement.success && !refinement.reason.empty());
      CHECK(o::analyze(bad).residuals.size() == before.residuals.size());
    }
    // A legitimate external entry through the initializer remains supported.
    auto wrapper = original.program;
    const auto incoming = wrapper.observed->sites.size();
    wrapper.observed->sites.emplace_back();
    wrapper.observed->sites.back().successors = {spec.owner};
    wrapper.observed->entry = incoming;
    CHECK(o::refineCountedLoop(wrapper, spec).success);
  }
  { // Refining an outer loop must not erase an inner loop's edge owners.
    auto p = observed_fixtures::target(2, 1);
    p.operations = {observed_fixtures::op(0, 0, 2),
                    observed_fixtures::op(1, 0, 1),
                    observed_fixtures::op(0, 0, 2)};
    p.operations[0].accesses[0].definiteWrite = true;
    p.operations[2].accesses[0].definiteWrite = true;
    p.body = {o::Region::For,
              {observed_fixtures::seq({
                  observed_fixtures::leaf(0),
                  {o::Region::For,
                   {observed_fixtures::seq({observed_fixtures::leaf(1)})}},
                  observed_fixtures::leaf(2)})}};
    auto original = o::addStructuredBoundaryCuts(p);
    CHECK(original.success);
    const auto &q = *original.program.observed;
    o::CountedLoopRegion spec;
    spec.owner = q.entry;
    spec.header = q.sites[spec.owner].successors.front();
    spec.bodyEntry = q.sites[spec.header].successors.front();
    spec.continuation = q.sites[spec.header].successors.back();
    std::set<std::size_t> seen;
    std::vector<std::size_t> pending{spec.bodyEntry};
    while (!pending.empty()) {
      const auto at = pending.back();
      pending.pop_back();
      if (at == spec.header || !seen.insert(at).second)
        continue;
      spec.bodySites.push_back(at);
      for (auto next : q.sites[at].successors)
        pending.push_back(next);
    }
    o::StorageFrontierAnalysis before(original.program);
    const auto source = before.sitesForOperation(1).front();
    const auto oldWitness = before.witness(source, source, 0);
    CHECK(oldWitness.exists && oldWitness.crossedLoopOwners.size() == 1);
    const auto innerOwner = oldWitness.crossedLoopOwners.front();
    CHECK(innerOwner != spec.owner);
    const auto refined = o::refineCountedLoop(original.program, spec);
    CHECK(refined.success);
    o::StorageFrontierAnalysis after(refined.program);
    o::PrefixQuery prefixes(refined.program);
    const auto readers = after.sitesForOperation(1);
    CHECK(!readers.empty());
    for (auto reader : readers) {
      const auto witness = after.witness(reader, reader, 0);
      CHECK(witness.exists && witness.definiteWriteFree);
      CHECK(witness.crossedLoopOwners == oldWitness.crossedLoopOwners);
      const auto cuts = prefixes.backwardCuts(reader);
      CHECK(cuts.complete);
      CHECK(std::find(cuts.crossedLoopOwners.begin(),
                      cuts.crossedLoopOwners.end(), innerOwner) !=
            cuts.crossedLoopOwners.end());
    }
  }
  std::cout << checks << " observation assertions; " << frontEndPaths
            << " independent normalized frontend paths; " << concrete
            << " full-history graph traces\n";
}
