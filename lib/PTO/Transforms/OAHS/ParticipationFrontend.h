// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_PARTICIPATION_FRONTEND_H
#define PTO_OAHS_PARTICIPATION_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <array>
#include <map>
#include <set>

namespace mlir::pto::oahs {
ParticipationImport refineParticipations(const Program& input, const std::vector<ParticipationRegion>& requests)
{
  ParticipationImport out;
  ++out.validations;
  const auto valid = validateProgram(input);
  if (!valid.success || !input.observed) {
    out.reason = "participation requires valid original control"; return out;
  }
  const auto& old = *input.observed;
  const auto size = old.sites.size();
  std::vector<std::vector<std::size_t>> predecessors(size), wordSites(old.observations.size());
  std::vector<bool> reachable(size);
  std::vector<std::size_t> pending{old.entry};
  while (!pending.empty()) {
    const auto site = pending.back(); pending.pop_back();
    if (reachable[site]) { continue; }
    reachable[site] = true; ++out.work;
    const auto& node = old.sites[site];
    if (node.observation != NoControlId) { wordSites[node.observation].push_back(site); }
    for (auto next : node.successors) {
      ++out.work; predecessors[next].push_back(site); pending.push_back(next);
    }
  }
  struct Prepared { std::size_t request; std::set<std::size_t> members; };
  std::vector<Prepared> prepared;
  std::set<std::size_t> occupied;
  out.refusals.resize(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& r = requests[index];
    auto qualify = [&]() -> std::string {
      const bool shape = r.available && r.entry < size && r.exit < size && r.decision < size &&
          r.entry != r.exit && !r.whenFalse.empty() && !r.whenTrue.empty() &&
          r.predicate.kind == ObservationAtom::LoopNonEmpty && r.predicate.value == 1 &&
          old.sites[r.entry].operation == NoControlId;
      if (!shape) { return "participation lacks an available Boolean and control-only entry gap"; }
      std::set<std::size_t> alternatives(r.whenFalse.begin(), r.whenFalse.end());
      for (auto site : r.whenTrue) {
        if (!alternatives.insert(site).second) { return "participation alternatives overlap"; }
      }
      const std::set<std::size_t> actual(old.sites[r.decision].successors.begin(),
                                          old.sites[r.decision].successors.end());
      if (alternatives != actual) { return "participation alternatives do not cover the actual decision"; }
      std::set<std::size_t> members;
      std::vector<std::size_t> todo{r.entry};
      while (!todo.empty()) {
        const auto site = todo.back(); todo.pop_back();
        if (site == r.exit || !members.insert(site).second) { continue; }
        ++out.work;
        if (occupied.count(site)) { return "overlapping participation needs composed occurrence support"; }
        const auto& next = old.sites[site].successors;
        if (next.empty()) { return "participation interval has an unrepresented exit"; }
        todo.insert(todo.end(), next.begin(), next.end());
      }
      const bool missingDecision = !members.count(r.decision) ||
          (members.count(old.entry) && old.entry != r.entry);
      if (missingDecision) { return "participation decision or original entry bypasses interval"; }
      for (auto predecessor : predecessors[r.entry]) {
        ++out.work;
        if (members.count(predecessor)) { return "participation entry repeats before its invocation exits"; }
      }
      std::set<std::size_t> words;
      for (auto site : members) {
        for (auto predecessor : predecessors[site]) {
          ++out.work;
          const bool bypass = site != r.entry && !members.count(predecessor);
          if (bypass) { return "external entry bypasses participation interval"; }
        }
        const auto word = old.sites[site].observation;
        if (word != NoControlId) { words.insert(word); }
      }
      for (auto word : words) {
        for (auto site : wordSites[word]) {
          ++out.work;
          if (!members.count(site)) { return "participation interval is not closed under original command words"; }
        }
        for (const auto& atom : old.observations[word].atoms) {
          if (atom.kind == ObservationAtom::LoopNonEmpty) {
            return "overlapping participation quotient needs composed occurrence support";
          }
        }
      }
      const auto entryWord = old.sites[r.entry].observation;
      const bool sharedEntry = entryWord != NoControlId && wordSites[entryWord].size() != 1;
      if (sharedEntry) { return "participation source gap has another original occurrence"; }
      std::set<std::size_t> seen;
      todo = old.sites[r.decision].successors;
      while (!todo.empty()) {
        const auto site = todo.back(); todo.pop_back();
        if (site == r.exit || !seen.insert(site).second) { continue; }
        ++out.work;
        if (site == r.decision || site == r.entry) { return "participation crosses its original invocation"; }
        const auto& next = old.sites[site].successors;
        todo.insert(todo.end(), next.begin(), next.end());
      }
      occupied.insert(members.begin(), members.end());
      prepared.push_back({index, std::move(members)});
      return {};
    };
    out.refusals[index] = qualify();
  }
  out.program = input; ++out.copies;
  if (prepared.empty()) { out.success = true; return out; }
  auto& q = *out.program.observed;
  std::array<std::map<std::size_t, std::size_t>, 2> copies;
  std::set<std::size_t> entries, members;
  for (const auto& item : prepared) {
    const auto& r = requests[item.request];
    const auto& members = item.members;
    entries.insert(r.entry);
    for (unsigned value = 0; value < 2; ++value) {
      std::map<std::size_t, std::size_t> observations;
      for (auto site : members) {
        const auto id = q.sites.size();
        copies[value].emplace(site, id);
        q.sites.push_back(old.sites[site]);
        const auto observation = old.sites[site].observation;
        if (site == r.entry || observation == NoControlId) {
          q.sites[id].observation = NoControlId; continue;
        }
        auto found = observations.find(observation);
        if (found == observations.end()) {
          auto word = old.observations[observation];
          auto atom = r.predicate; atom.value = value;
          word.atoms.push_back(atom);
          const auto index = q.observations.size();
          q.observations.push_back(std::move(word));
          found = observations.emplace(observation, index).first;
        }
        q.sites[id].observation = found->second;
      }
      for (auto site : members) {
        auto& node = q.sites[copies[value].at(site)];
        const auto& original = old.sites[site];
        node.successors.clear(); node.backedgeOwners.clear();
        const auto& selected = value ? r.whenTrue : r.whenFalse;
        for (std::size_t edge = 0; edge < original.successors.size(); ++edge) {
          const auto next = original.successors[edge];
          const bool omitted = site == r.decision &&
              std::find(selected.begin(), selected.end(), next) == selected.end();
          if (omitted) { continue; }
          node.successors.push_back(next == r.exit ? next : copies[value].at(next));
          node.backedgeOwners.push_back(original.backedgeOwners.empty() ? NoControlId : original.backedgeOwners[edge]);
        }
      }
    }
    // Retain an unguarded source gap before sampling. This is not a new payload;
    // cloned words retain the original payload and exact original guard.
    for (auto site : members) {
      q.sites[site].operation = NoControlId;
      q.sites[site].successors = {r.exit}; q.sites[site].backedgeOwners.clear();
      if (site != r.entry) { q.sites[site].observation = NoControlId; }
    }
    q.sites[r.entry].successors = {copies[0].at(r.entry), copies[1].at(r.entry)};
    out.accepted.push_back(item.request);
  }
  for (const auto& item : prepared) { members.insert(item.members.begin(), item.members.end()); }
  auto mapSite = [&](std::size_t site, unsigned value) {
    const auto found = copies[value].find(site);
    return found == copies[value].end() ? site : found->second;
  };
  auto expand = [&](const std::vector<std::size_t>& sites) {
    std::set<std::size_t> all;
    for (auto site : sites) {
      all.insert(mapSite(site, 0)); all.insert(mapSite(site, 1));
      if (entries.count(site)) { all.insert(site); }
    }
    return std::vector<std::size_t>(all.begin(), all.end());
  };
  std::set<std::size_t> affectedOwners;
  for (auto& loop : q.loops) {
    out.work += loop.sites.size() + loop.entries.size() + loop.exits.size();
    const bool entryAffected = members.count(loop.entry) != 0;
    const bool exitAffected = members.count(loop.exit) != 0;
    bool affected = entryAffected || exitAffected || members.count(loop.bodyEntry) != 0;
    for (const auto* sites : {&loop.sites, &loop.entries, &loop.exits}) {
      for (auto site : *sites) {
        affected |= members.count(site) != 0;
      }
    }
    if (!affected) { continue; }
    affectedOwners.insert(loop.owner);
    const auto occurrences = loopEntryOccurrences(loop);
    if (occurrences.empty()) { out.reason = "participation requires paired child interfaces"; return out; }
    loop.sites = expand(loop.sites);
    loop.entries = expand(loop.entries.empty() ? std::vector<std::size_t>{loop.entry} : loop.entries);
    loop.exits = expand(loop.exits.empty() ? std::vector<std::size_t>{loop.exit} : loop.exits);
    loop.entry = loop.entries.front(); loop.exit = loop.exits.front();
    loop.bodyEntry = mapSite(loop.bodyEntry, 0);
    loop.occurrences.clear();
    for (const auto& occurrence : occurrences) {
      if (members.count(occurrence.entry)) {
        for (unsigned value = 0; value < 2; ++value) {
          auto mapped = occurrence;
          mapped.entry = mapSite(mapped.entry, value); mapped.exit = mapSite(mapped.exit, value);
          mapped.bodyEntry = mapSite(mapped.bodyEntry, value);
          for (auto* sites : {&mapped.sites, &mapped.exits}) {
            for (auto& site : *sites) { site = mapSite(site, value); }
          }
          loop.occurrences.push_back(std::move(mapped));
        }
      } else {
        auto mapped = occurrence;
        mapped.sites = expand(mapped.sites);
        mapped.exits = expand(mapped.exits.empty() ? std::vector<std::size_t>{mapped.exit} : mapped.exits);
        if (members.count(mapped.bodyEntry)) { mapped.bodyEntry = occurrence.bodyEntry; }
        loop.occurrences.push_back(std::move(mapped));
      }
    }
  }

  if (!prepared.empty()) {
    ++out.refreshes;
    if (!refreshLoopOccurrences(q, &affectedOwners)) {
      out.reason = "participation lost a paired child continuation"; return out;
    }
    q.qualification += "; original-participation-interval-v1";
  }
  ++out.validations;
  const auto checked = validateProgram(out.program);
  out.success = checked.success; out.reason = checked.reason;
  for (std::size_t i = 0; i < input.operations.size(); ++i) { out.originalPhases.push_back(i); }
  return out;
}
ObservedImport refineParticipation(const Program& input, const ParticipationRegion& request)
{
  auto batch = refineParticipations(input, {request});
  if (batch.success && batch.accepted.empty()) {
    batch.success = false; batch.reason = batch.refusals.front();
  }
  return std::move(static_cast<ObservedImport&>(batch));
}
} // namespace mlir::pto::oahs
#endif
