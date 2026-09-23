// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include <functional>
using namespace selected_test;

namespace {
o::Program fixture() {
  auto p = base(1);
  p.operations = {op(o::Pipe::M, {{0, false, true, false}}),
                  op(o::Pipe::M, {{0, true, false, false}}),
                  op(o::Pipe::MTE2, {}), op(o::Pipe::FIX, {{0, true, false, false}})};
  o::ObservedControl q;
  q.qualification = "test original nested first-use predicate";
  q.scopes = {{0, o::NoControlId, o::NoControlId}};
  q.entry = 0;
  q.exit = 14;
  q.sites.resize(15);
  const std::vector<std::vector<std::size_t>> edges = {
      {1}, {2, 14}, {3}, {4, 12}, {5}, {6}, {7, 11}, {8, 9},
      {10}, {10}, {6}, {3}, {13}, {1}, {}};
  for (std::size_t site = 0; site < q.sites.size(); ++site) {
    q.sites[site].successors = edges[site];
    q.sites[site].observation = site;
    q.observations.push_back({site, {}, true});
  }
  q.sites[4].operation = 2;
  q.sites[8].operation = 0;
  q.sites[9].operation = 1;
  q.sites[12].operation = 3;
  q.sites[10].backedgeOwners = {5};
  q.sites[11].backedgeOwners = {2};
  q.sites[13].backedgeOwners = {0};
  q.loops = {{0, 0, 14, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}},
             {2, 2, 12, {3, 4, 5, 6, 7, 8, 9, 10, 11}},
             {5, 5, 11, {6, 7, 8, 9, 10}}};
  p.observed = std::move(q);
  return p;
}

void checkPaths(const o::Program &p, const o::Commands &commands) {
  const auto &q = *p.observed;
  std::vector<std::size_t> path;
  unsigned traces = 0, repeated = 0, bypassed = 0;
  std::function<void(std::size_t, unsigned, bool, unsigned)> visit;
  visit = [&](std::size_t at, unsigned payloads, bool first, unsigned entries) {
    payloads += q.sites[at].operation != o::NoControlId;
    if (payloads > 6) {
      return;
    }
    const auto anchor = q.observations[q.sites[at].observation].anchor;
    if (anchor == 2) {
      first = true;
      ++entries;
    }
    if (anchor == 7) {
      require(q.sites[at].successors == std::vector<std::size_t>{first ? 8u : 9u},
              "first-use predicate disagrees with original nested visits");
    }
    path.push_back(at);
    if (at == q.exit) {
      o::Program concrete = p;
      concrete.observed.reset();
      concrete.operations.clear();
      o::Commands words(1);
      std::vector<unsigned> order;
      for (auto site : path) {
        words.back().insert(words.back().end(), commands[site].begin(), commands[site].end());
        const auto operation = q.sites[site].operation;
        if (operation != o::NoControlId) {
          order.push_back(concrete.operations.size());
          concrete.operations.push_back(p.operations[operation]);
          words.emplace_back();
        }
      }
      require(bool(oahs_oracle::graph(concrete, words, order)), "first-use concrete causal graph");
      ++traces;
      repeated += entries > 1;
      bypassed += entries == 0;
    } else {
      for (std::size_t edge = 0; edge < q.sites[at].successors.size(); ++edge) {
        const auto owner = q.sites[at].backedgeOwners.empty()
                               ? o::NoControlId : q.sites[at].backedgeOwners[edge];
        visit(q.sites[at].successors[edge], payloads, first && owner != 2 && owner != 5, entries);
      }
    }
    path.pop_back();
  };
  visit(q.entry, 0, false, 0);
  require(traces && repeated && bypassed, "missing repeated-entry or zero-trip traces");
  std::cout << traces << " first-use concrete control/causal traces\n";
}

o::Commands fixtureCommands(const o::Program &p) {
  o::Commands commands(o::commandCutCount(p));
  const auto &q = *p.observed;
  for (std::size_t site = 0; site < q.sites.size(); ++site) {
    const auto anchor = q.observations[q.sites[site].observation].anchor;
    if (anchor == 2) {
      commands[site] = {{o::Command::Publish, o::Pipe::FIX, o::Pipe::M, 0},
                        {o::Command::Acquire, o::Pipe::FIX, o::Pipe::M, 0}};
    } else if (anchor == 12) {
      commands[site] = {{o::Command::Publish, o::Pipe::M, o::Pipe::FIX, 0},
                        {o::Command::Acquire, o::Pipe::M, o::Pipe::FIX, 0}};
    } else if (anchor == 9) {
      commands[site] = {{o::Command::Barrier, o::Pipe::M}};
    }
  }
  return commands;
}
} // namespace

int main() {
  const auto original = fixture();
  const o::FirstUseRegion region{2, 12, {7}, {2, 5}};
  const auto refined = o::refineFirstUse(original, region);
  require(refined.success, refined.reason);
  require(refined.program.observed->sites.size() == 21, "expanded beyond the first-use prefix");
  require(refined.program.operations.size() == original.operations.size(), "duplicated physical effects");
  for (std::size_t i = 0; i < original.observed->loops.size(); ++i) {
    const auto& retained = refined.program.observed->loops[i].sites;
    for (auto site : original.observed->loops[i].sites) {
      require(std::find(retained.begin(), retained.end(), site) != retained.end(),
              "refined entry reachability erased original owner membership");
    }
  }
  const auto& child = refined.program.observed->loops.back();
  require(child.occurrences.size() == 2, "first-use prefix lost the copied child entry");
  for (const auto& occurrence : o::loopEntryOccurrences(child)) {
    // The copied prefix can bypass the child through a copied exit, or join
    // its original suffix and exit there. Both preserve the same native gap.
    require(std::find(occurrence.exits.begin(), occurrence.exits.end(), 11) != occurrence.exits.end(),
            "prefix lost its original shared child exit");
    require(occurrence.exits.size() == (occurrence.entry == child.entry ? 1u : 2u),
            "prefix lost its zero-trip or shared-suffix exit");
    for (auto exit : occurrence.exits) {
      const auto& control = *refined.program.observed;
      require(control.observations[control.sites[exit].observation].anchor == 11,
              "child occurrence escaped its original exit boundary");
    }
    require(std::find(occurrence.sites.begin(), occurrence.sites.end(), 9) != occurrence.sites.end(),
            "prefix lost the original later-iteration suffix");
  }
  const auto commands = fixtureCommands(refined.program);
  const auto certificate = o::checkCausalFrontier(refined.program, commands);
  require(certificate.accepted, "first-use causal certificate: " + certificate.reason +
          " at " + std::to_string(certificate.cut));
  checkPaths(refined.program, commands);
  for (auto boundary : {2u, 12u}) {
    auto missing = commands;
    missing[boundary].clear();
    require(!o::checkCausalFrontier(refined.program, missing).accepted,
            "region boundary invented completion or event credit");
  }
  auto repeating = fixtureCommands(original);
  require(!o::checkCausalFrontier(original, repeating).accepted,
          "genuinely repeated initialization lost its completion requirement");
  repeating[8] = {{o::Command::Barrier, o::Pipe::M}};
  require(o::checkCausalFrontier(original, repeating).accepted, "repeating initializer completion repair");
  auto bypass = original;
  bypass.observed->sites[0].successors.push_back(7);
  require(!o::refineFirstUse(bypass, region).success, "accepted external region entry");
  auto successive = original;
  successive.observed->sites.resize(17);
  successive.observed->sites[8].operation = o::NoControlId;
  successive.observed->sites[8].successors = {15, 16};
  for (auto site : {15u, 16u}) {
    successive.observed->sites[site].operation = site == 15 ? 0 : 1;
    successive.observed->sites[site].successors = {10};
    successive.observed->sites[site].observation = site;
    successive.observed->observations.push_back({site, {}, true});
  }
  auto twoDecisions = region;
  twoDecisions.decisions = {7, 8};
  require(!o::refineFirstUse(successive, twoDecisions).success,
          "accepted successive first-use decisions before one tracked backedge");
  for (unsigned mutation = 0; mutation < 4; ++mutation) {
    auto invalid = region;
    if (mutation == 0) { invalid.entry = 99; }
    if (mutation == 1) { invalid.decisions = {99}; }
    if (mutation == 2) { invalid.backedgeOwners.clear(); }
    if (mutation == 3) { invalid.decisions = {0}; }
    require(!o::refineFirstUse(original, invalid).success, "accepted malformed first-use qualification");
  }
}
