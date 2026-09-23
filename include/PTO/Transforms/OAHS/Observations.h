// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_OBSERVATIONS_H
#define PTO_TRANSFORMS_OAHS_OBSERVATIONS_H
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>
namespace mlir::pto::oahs {
constexpr std::size_t NoControlId = std::numeric_limits<std::size_t>::max();
// Only original-value observations. Occupancy, receipt state and writer history
// are deliberately absent. The frontend qualifies availability and arithmetic.
struct ObservationAtom {
  // Fixed underlying type permits malformed serialized tags to be rejected
  // explicitly, rather than invoking undefined behavior before validation.
  enum Kind : unsigned {
    OriginalBoolean,
    LoopNonEmpty,
    LoopResidue,
    LoopHasPrevious,
    LoopHasNext
  } kind = OriginalBoolean;
  std::size_t owner = 0;
  uint64_t parameter = 0, value = 0;
};
struct OriginalObservation {
  std::size_t anchor = 0;
  std::vector<ObservationAtom> atoms;
  bool available = false;
};
struct ObservedSite {
  // Optional original physical phase. A boundary site is NOT a dummy payload.
  std::size_t operation = NoControlId;
  // Sites with the same observation execute the same ordered word.
  std::size_t observation = NoControlId;
  std::vector<std::size_t> successors, backedgeOwners;
  std::size_t context = 0;
};
struct ObservedScope {
  unsigned kind = 0; // AnalysisContext kind, validated by the adapter
  std::size_t parent = NoControlId, ownerSite = NoControlId;
};
// Original normalized region boundaries, used only to qualify local roles.
// They confer no completion or event credit; construction replays the full graph.
struct ObservedLoopOccurrence {
  std::size_t entry = NoControlId, exit = NoControlId, bodyEntry = NoControlId;
  std::vector<std::size_t> sites;
  // A cloned prefix can rejoin original backedges and reach several exits.
  std::vector<std::size_t> exits = {};
};
struct ObservedLoop {
  std::size_t owner = NoControlId, entry = NoControlId, exit = NoControlId;
  std::vector<std::size_t> sites;
  // Original unrefined entry, qualified by the frontend's actual bounds.
  // These placement facts do not change the conservative execution graph or
  // grant completion. Refined quotients leave bodyEntry unavailable.
  std::size_t bodyEntry = NoControlId;
  bool atLeastOnce = false;
  // Copies of the same original child boundary under an enclosing bank
  // interface. Empty vectors denote the single entry/exit above. These remain
  // original-control positions and never imply a storage or event reset.
  std::vector<std::size_t> entries = {}, exits = {};
  // Explicitly paired original-control interfaces, not parallel lists whose
  // entries/exits may be arbitrarily zipped. No causal state lives here.
  std::vector<ObservedLoopOccurrence> occurrences = {};
};
inline std::vector<ObservedLoopOccurrence> loopEntryOccurrences(const ObservedLoop& loop)
{
  if (!loop.occurrences.empty()) {
    return loop.occurrences;
  }
  if (!loop.entries.empty() || !loop.exits.empty()) {
    return {}; // Legacy aggregate boundaries do not establish pairing.
  }
  return {{loop.entry, loop.exit, loop.bodyEntry, loop.sites}};
}
struct ObservedControl {
  std::vector<ObservedSite> sites;
  std::vector<OriginalObservation> observations;
  std::vector<ObservedScope> scopes;
  std::vector<ObservedLoop> loops;
  std::size_t entry = 0, exit = 0;
  // Named input/frontend proof boundary, not a causal-completion assertion.
  std::string qualification;
};
// Resolve each interface through the refined original graph. A pointwise
// clone map does not capture a prefix that rejoins a shared suffix/backedge.
inline bool refreshLoopOccurrences(ObservedControl& control, const std::set<std::size_t>* affectedOwners = nullptr)
{
  for (auto& loop : control.loops) {
    if (affectedOwners && !affectedOwners->count(loop.owner)) { continue; }
    if (loop.occurrences.empty()) {
      continue;
    }
    const std::set<std::size_t> allowed(loop.sites.begin(), loop.sites.end());
    const std::set<std::size_t> boundaries = loop.exits.empty() ? std::set<std::size_t>{loop.exit} :
        std::set<std::size_t>(loop.exits.begin(), loop.exits.end());
    for (auto& occurrence : loop.occurrences) {
      if (occurrence.entry >= control.sites.size() || occurrence.exit >= control.sites.size()) {
        return false;
      }
      std::set<std::size_t> members, exits;
      auto pending = control.sites[occurrence.entry].successors;
      while (!pending.empty()) {
        const auto site = pending.back();
        pending.pop_back();
        if (boundaries.count(site)) {
          exits.insert(site);
          continue;
        }
        if (!members.insert(site).second) {
          continue;
        }
        if (site >= control.sites.size() || !allowed.count(site)) {
          return false;
        }
        const auto& next = control.sites[site].successors;
        pending.insert(pending.end(), next.begin(), next.end());
      }
      if (exits.empty() || (loop.atLeastOnce && !members.count(occurrence.bodyEntry))) {
        return false;
      }
      occurrence.sites.assign(members.begin(), members.end());
      occurrence.exits.assign(exits.begin(), exits.end());
      occurrence.exit = *exits.begin();
    }
    // Keep the original owner membership. Reachability from one refined
    // entry is not permission to erase positions used by another lifetime or
    // by qualification of the surrounding recurrence interface.
  }
  return true;
}
} // namespace mlir::pto::oahs
#endif
