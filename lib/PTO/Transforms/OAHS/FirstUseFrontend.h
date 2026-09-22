// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_FIRST_USE_FRONTEND_H
#define PTO_OAHS_FIRST_USE_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include "Control.h"
#include <algorithm>
#include <map>
#include <set>

namespace mlir::pto::oahs {
// Preserve one original first-use fact across nested region boundaries. The
// prefix copies share command observations and physical phases with their
// originals. No storage, causal, or event state is reset at the join.
namespace observed_detail {
class FirstUsePrefix {
public:
  FirstUsePrefix(const ObservedControl &original, ObservedControl &output,
                 const FirstUseRegion &descriptor)
      : old(original), q(output), region(descriptor),
        decisions(region.decisions.begin(), region.decisions.end()),
        owners(region.backedgeOwners.begin(), region.backedgeOwners.end()) {}

  std::string validate(const Program &input) {
    const auto shape = validateShape();
    if (!shape.empty()) {
      return shape;
    }
    collectMembers();
    for (auto site : decisions) {
      if (!members.count(site)) {
        return "first-use decision outside its region";
      }
    }
    const auto separated = validateDecisionSeparation();
    if (!separated.empty()) {
      return separated;
    }
    if (members.count(old.entry)) {
      return "original entry bypasses first-use boundary";
    }
    const auto reachable = detail::reachableSites(detail::buildControlGraph(input));
    return validateEntries(reachable);
  }

  void run() {
    // Do not retain a reference into sites while prefix() may grow it.
    std::vector<std::size_t> entries;
    for (auto next : old.sites[region.entry].successors) {
      entries.push_back(prefix(next));
    }
    q.sites[region.entry].successors = std::move(entries);
    for (std::size_t i = 0; i < pending.size(); ++i) {
      wirePrefix(pending[i]);
    }
    for (auto site : decisions) {
      selectArm(site, site, 1);
    }
    updateLoops();
    q.qualification += "; qualified-first-use-prefix-v1";
  }

private:
  std::string validateShape() const {
    const auto size = old.sites.size();
    if (region.entry >= size || region.exit >= size || region.entry == region.exit ||
        decisions.empty() || owners.empty()) {
      return "invalid first-use region boundary";
    }
    for (auto site : decisions) {
      if (site >= size || site == region.entry || site == region.exit ||
          old.sites[site].successors.size() != 2) {
        return "invalid first-use decision";
      }
    }
    for (auto owner : owners) {
      if (owner >= size) {
        return "invalid first-use backedge owner";
      }
    }
    return {};
  }

  void collectMembers() {
    auto todo = old.sites[region.entry].successors;
    while (!todo.empty()) {
      const auto at = todo.back();
      todo.pop_back();
      if (at == region.exit || at == region.entry || !members.insert(at).second) {
        continue;
      }
      const auto &next = old.sites[at].successors;
      todo.insert(todo.end(), next.begin(), next.end());
    }
  }

  std::string validateEntries(const std::vector<bool> &reachable) const {
    for (std::size_t site = 0; site < old.sites.size(); ++site) {
      if (!reachable[site]) {
        continue;
      }
      for (auto next : old.sites[site].successors) {
        const bool bypass = members.count(next) && !members.count(site) && site != region.entry;
        if (bypass) {
          return "external entry bypasses first-use boundary";
        }
      }
    }
    return {};
  }

  std::string validateDecisionSeparation() const {
    for (auto decision : decisions) {
      std::vector<std::size_t> todo;
      const auto &start = old.sites[decision];
      for (std::size_t edge = 0; edge < start.successors.size(); ++edge) {
        if (edge != 0) {
          continue;
        }
        const auto owner = start.backedgeOwners.empty()
                               ? NoControlId
                               : start.backedgeOwners[edge];
        if (!owners.count(owner) && start.successors[edge] != region.exit) {
          todo.push_back(start.successors[edge]);
        }
      }
      std::set<std::size_t> seen;
      while (!todo.empty()) {
        const auto site = todo.back();
        todo.pop_back();
        if (!seen.insert(site).second) {
          continue;
        }
        if (decisions.count(site)) {
          return "first-use decisions are not separated by a tracked backedge";
        }
        const auto &node = old.sites[site];
        for (std::size_t edge = 0; edge < node.successors.size(); ++edge) {
          const auto owner = node.backedgeOwners.empty()
                                 ? NoControlId
                                 : node.backedgeOwners[edge];
          const auto next = node.successors[edge];
          if (!owners.count(owner) && next != region.exit) {
            todo.push_back(next);
          }
        }
      }
    }
    return {};
  }

  std::size_t prefix(std::size_t site) {
    if (!members.count(site)) {
      return site;
    }
    const auto found = clones.find(site);
    if (found != clones.end()) {
      return found->second;
    }
    const auto id = q.sites.size();
    clones.emplace(site, id);
    pending.push_back(site);
    q.sites.push_back(old.sites[site]);
    return id;
  }

  void selectArm(std::size_t target, std::size_t source, unsigned arm) {
    const auto &node = old.sites[source];
    q.sites[target].successors = {node.successors[arm]};
    q.sites[target].backedgeOwners = {
        node.backedgeOwners.empty() ? NoControlId : node.backedgeOwners[arm]};
  }

  void wirePrefix(std::size_t site) {
    const auto clone = clones.at(site);
    const auto &node = old.sites[site];
    if (decisions.count(site)) {
      selectArm(clone, site, 0);
      return;
    }
    std::vector<std::size_t> next;
    for (std::size_t edge = 0; edge < node.successors.size(); ++edge) {
      const auto owner = node.backedgeOwners.empty() ? NoControlId : node.backedgeOwners[edge];
      next.push_back(owners.count(owner) ? node.successors[edge] : prefix(node.successors[edge]));
    }
    q.sites[clone].successors = std::move(next);
  }

  void updateLoops() {
    for (auto &loop : q.loops) {
      const auto originalOccurrences = loopEntryOccurrences(loop);
      const auto originalMembers = loop.sites;
      for (auto site : originalMembers) {
        const auto found = clones.find(site);
        if (found != clones.end()) {
          loop.sites.push_back(found->second);
        }
      }
      auto extendBoundaries = [&](std::vector<std::size_t> &boundaries,
                                  std::size_t primary) {
        const auto original = boundaries.empty()
                                  ? std::vector<std::size_t>{primary}
                                  : boundaries;
        for (auto site : original) {
          const auto found = clones.find(site);
          if (found == clones.end()) {
            continue;
          }
          if (boundaries.empty()) {
            boundaries.push_back(primary);
          }
          boundaries.push_back(found->second);
        }
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                         boundaries.end());
      };
      extendBoundaries(loop.entries, loop.entry);
      extendBoundaries(loop.exits, loop.exit);
      if (loop.entry == region.entry && clones.count(loop.bodyEntry)) {
        loop.bodyEntry = clones.at(loop.bodyEntry);
      }
      // Preserve paired original entry interfaces through prefix cloning.
      // A copied entry gets its mapped interface; an unchanged entry owns both
      // its first-use prefix and shared continuation. Neither resets state.
      loop.occurrences.clear();
      for (const auto& occurrence : originalOccurrences) {
        auto mapped = occurrence;
        auto remap = [&](std::size_t site) {
          const auto found = clones.find(site);
          return found == clones.end() ? site : found->second;
        };
        mapped.entry = remap(mapped.entry);
        mapped.exit = remap(mapped.exit);
        mapped.bodyEntry = remap(mapped.bodyEntry);
        for (auto* sites : {&mapped.sites, &mapped.firstVisitPrefix, &mapped.exits}) {
          for (auto& site : *sites) {
            site = remap(site);
          }
        }
        for (auto& frontier : mapped.firstWriteFrontiers) {
          frontier.first = remap(frontier.first);
          frontier.second = remap(frontier.second);
        }
        if (mapped.entry != occurrence.entry) {
          loop.occurrences.push_back(occurrence);
          loop.occurrences.push_back(std::move(mapped));
        } else {
          auto combined = occurrence;
          combined.bodyEntry = occurrence.entry == region.entry ? mapped.bodyEntry : occurrence.bodyEntry;
          auto mergeSites = [](auto& target, const auto& extra) {
            target.insert(target.end(), extra.begin(), extra.end());
            std::sort(target.begin(), target.end());
            target.erase(std::unique(target.begin(), target.end()), target.end());
          };
          mergeSites(combined.sites, mapped.sites);
          mergeSites(combined.firstVisitPrefix, mapped.firstVisitPrefix);
          for (const auto& frontier : mapped.firstWriteFrontiers) {
            if (std::find(combined.firstWriteFrontiers.begin(), combined.firstWriteFrontiers.end(), frontier) ==
                combined.firstWriteFrontiers.end()) {
              combined.firstWriteFrontiers.push_back(frontier);
            }
          }
          loop.occurrences.push_back(std::move(combined));
        }
      }
      if (!loop.occurrences.empty()) {
        loop.firstVisitPrefix.clear();
        loop.firstWriteFrontiers.clear();
        for (const auto& occurrence : loop.occurrences) {
          loop.firstVisitPrefix.insert(loop.firstVisitPrefix.end(),
              occurrence.firstVisitPrefix.begin(), occurrence.firstVisitPrefix.end());
          loop.firstWriteFrontiers.insert(loop.firstWriteFrontiers.end(),
              occurrence.firstWriteFrontiers.begin(), occurrence.firstWriteFrontiers.end());
        }
      }
    }
  }

  const ObservedControl &old;
  ObservedControl &q;
  const FirstUseRegion &region;
  const std::set<std::size_t> decisions, owners;
  std::set<std::size_t> members;
  std::map<std::size_t, std::size_t> clones;
  std::vector<std::size_t> pending;
};
} // namespace observed_detail

ObservedImport refineFirstUse(const Program &input, const FirstUseRegion &region) {
  ObservedImport out;
  const auto valid = validateProgram(input);
  if (!valid.success || !input.observed) {
    out.reason = valid.success ? "first-use refinement requires observed control" : valid.reason;
    return out;
  }
  out.program = input;
  observed_detail::FirstUsePrefix prefix(*input.observed, *out.program.observed, region);
  out.reason = prefix.validate(input);
  if (!out.reason.empty()) {
    return out;
  }
  prefix.run();
  if (!refreshLoopOccurrences(*out.program.observed)) {
    out.reason = "first-use refinement lacks a closed child occurrence interface";
    return out;
  }
  for (std::size_t i = 0; i < input.operations.size(); ++i) {
    out.originalPhases.push_back(i);
  }
  const auto checked = validateProgram(out.program);
  out.success = checked.success;
  out.reason = checked.reason;
  return out;
}
} // namespace mlir::pto::oahs
#endif
