// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_LAST_VISIT_FRONTEND_H
#define PTO_OAHS_LAST_VISIT_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <algorithm>
#include <map>
#include <set>
namespace mlir::pto::oahs {
ObservedImport refineLastVisit(const Program &input, std::size_t owner,
                               const std::vector<std::size_t> &anchors) {
  ObservedImport out;
  auto reject = [&](const char *reason) { out.reason = reason; return out; };
  const auto valid = validateProgram(input);
  if (!valid.success) { out.reason = valid.reason; return out; }
  if (!input.observed || anchors.empty()) return reject("last visit needs observed anchors");
  const auto &old = *input.observed;
  const auto found = std::find_if(old.loops.begin(), old.loops.end(),
      [&](const auto &loop) { return loop.owner == owner; });
  if (found == old.loops.end() || !found->atLeastOnce || found->entry != owner ||
      found->bodyEntry == NoControlId || !found->firstVisitPrefix.empty() ||
      !found->entries.empty() || !found->exits.empty() || owner >= old.sites.size() ||
      old.sites[owner].successors.size() != 1)
    return reject("last visit needs an unrefined nonempty loop");
  const auto header = old.sites[owner].successors.front();
  if (header >= old.sites.size() || old.sites[header].successors !=
      std::vector<std::size_t>{found->bodyEntry, found->exit})
    return reject("last visit needs original counted control");
  std::vector<std::size_t> body;
  std::set<std::size_t> members;
  auto at = found->bodyEntry;
  while (at != header) {
    if (at >= old.sites.size() || at == found->exit || !members.insert(at).second ||
        old.sites[at].successors.size() != 1)
      return reject("last visit needs a straight leaf body");
    const auto &site = old.sites[at];
    if (site.observation != NoControlId && !old.observations[site.observation].atoms.empty())
      return reject("last visit cannot combine existing observations");
    const auto next = site.successors.front();
    if (!site.backedgeOwners.empty() && site.backedgeOwners.front() !=
        (next == header ? owner : NoControlId))
      return reject("last visit has an unqualified backedge");
    body.push_back(at);
    at = next;
  }
  if (body.empty()) return reject("last visit needs a body");
  // No entry into the middle of either analytical visit, including shared
  // anchors that could otherwise execute an unqualified final-only command.
  for (std::size_t site = 0; site < old.sites.size(); ++site)
    for (auto next : old.sites[site].successors)
      if (members.count(next) && !members.count(site) &&
          !(site == header && next == found->bodyEntry))
        return reject("last visit has an external body entry");
  const std::set<std::size_t> selected(anchors.begin(), anchors.end());
  for (auto anchor : selected) {
    if (!members.count(anchor) || old.sites[anchor].observation == NoControlId)
      return reject("last visit anchor is unavailable");
    const auto observation = old.sites[anchor].observation;
    if (std::count_if(old.sites.begin(), old.sites.end(), [&](const auto &site) {
          return site.observation == observation;
        }) != 1) return reject("last visit anchor is shared");
  }
  out.program = input;
  auto &q = *out.program.observed;
  std::map<std::size_t, std::size_t> clones;
  for (auto site : body) {
    clones[site] = q.sites.size();
    q.sites.push_back(old.sites[site]);
    if (selected.count(site)) for (unsigned more = 0; more < 2; ++more) {
      auto observation = old.observations[old.sites[site].observation];
      observation.atoms.push_back({ObservationAtom::LoopHasNext, owner, 1, more});
      q.sites[more ? site : clones[site]].observation = q.observations.size();
      q.observations.push_back(std::move(observation));
    }
  }
  for (auto site : body) {
    auto &clone = q.sites[clones.at(site)];
    const auto next = clone.successors.front();
    clone.successors = {next == header ? found->exit : clones.at(next)};
    clone.backedgeOwners.clear();
  }
  // Nonfinal visits backedge; the final visit exits. Original payloads and
  // all unselected command words remain shared, with no runtime history.
  q.sites[header].successors = {found->bodyEntry, clones.at(found->bodyEntry)};
  q.sites[header].backedgeOwners.clear();
  for (auto &loop : q.loops) {
    if (loop.owner != owner && std::find(loop.sites.begin(), loop.sites.end(), owner) == loop.sites.end()) continue;
    for (auto site : body) loop.sites.push_back(clones.at(site));
    if (loop.owner == owner) {
      if (std::find(loop.sites.begin(), loop.sites.end(), header) == loop.sites.end()) loop.sites.push_back(header);
      loop.bodyEntry = header;
    }
  }
  if (!refreshLoopOccurrences(q)) {
    out.reason = "last-visit refinement lacks a closed child occurrence interface";
    return out;
  }
  q.qualification += "; last-visit-words-v1";
  for (std::size_t i = 0; i < input.operations.size(); ++i) out.originalPhases.push_back(i);
  const auto checked = validateProgram(out.program);
  out.success = checked.success;
  out.reason = checked.reason;
  return out;
}
} // namespace mlir::pto::oahs
#endif
