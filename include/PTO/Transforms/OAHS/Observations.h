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
#include <utility>
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
  // A separate endpoint gap immediately before this anchor's shared word.
  // It may use a narrower predicate vocabulary without splitting that word.
  bool beforeSharedWord = false;
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
  std::vector<std::size_t> sites, firstVisitPrefix;
  std::vector<std::pair<std::size_t, std::size_t>> firstWriteFrontiers;
  // A first-use prefix may rejoin original backedges and reach both copied
  // and original exits. Empty means the single representative exit above.
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
  // A frontend-qualified first-iteration prefix. Other words may remain shared
  // with later visits; they cannot consume a one-time publication repeatedly.
  std::vector<std::size_t> firstVisitPrefix = {};
  // Qualified original distance for a final-visit observation. Legacy unit
  // loops use one; a joint reader frontend supplies the original positive step.
  uint64_t lastVisitDistance = 1;
  // Exactly-once receipt boundary and its immediate first-write payload.
  // The boundary executes before the payload's ordinary shared command word.
  std::vector<std::pair<std::size_t, std::size_t>> firstWriteFrontiers = {};
  // Paired boundaries and endpoint obligations of each analytical copy. The
  // original owner and participation remain shared; entries do not reset state.
  std::vector<ObservedLoopOccurrence> occurrences = {};
};
inline std::vector<ObservedLoopOccurrence> loopEntryOccurrences(const ObservedLoop& loop)
{
  if (!loop.occurrences.empty()) {
    return loop.occurrences;
  }
  if (!loop.entries.empty() || !loop.exits.empty()) {
    return {}; // Older composed metadata does not establish boundary pairing.
  }
  return {{loop.entry, loop.exit, loop.bodyEntry, loop.sites,
           loop.firstVisitPrefix, loop.firstWriteFrontiers}};
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
// Recompute membership after a control refinement using its explicit paired
// interfaces. Cloned prefixes can rejoin original suffixes and backedges; a
// pointwise clone map alone does not describe those participating occurrences.
inline bool refreshLoopOccurrences(ObservedControl& control)
{
  for (auto& loop : control.loops) {
    if (loop.occurrences.empty()) {
      continue;
    }
    const std::set<std::size_t> allowed(loop.sites.begin(), loop.sites.end());
    const std::set<std::size_t> boundaries = loop.exits.empty() ? std::set<std::size_t>{loop.exit} :
        std::set<std::size_t>(loop.exits.begin(), loop.exits.end());
    std::set<std::size_t> covered;
    for (auto& occurrence : loop.occurrences) {
      if (occurrence.entry >= control.sites.size() || occurrence.exit >= control.sites.size()) {
        return false;
      }
      std::set<std::size_t> members;
      std::set<std::size_t> exits;
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
      occurrence.sites.assign(members.begin(), members.end());
      if (exits.empty()) {
        return false;
      }
      occurrence.exits.assign(exits.begin(), exits.end());
      occurrence.exit = *exits.begin();
      if (occurrence.entry == loop.entry) {
        occurrence.bodyEntry = loop.bodyEntry;
      }
      if (loop.atLeastOnce && !members.count(occurrence.bodyEntry)) {
        return false;
      }
      occurrence.firstVisitPrefix.clear();
      for (auto site : loop.firstVisitPrefix) {
        if (members.count(site)) {
          occurrence.firstVisitPrefix.push_back(site);
        }
      }
      occurrence.firstWriteFrontiers.clear();
      for (const auto& frontier : loop.firstWriteFrontiers) {
        if (members.count(frontier.first) != members.count(frontier.second)) {
          return false;
        }
        if (members.count(frontier.first)) {
          occurrence.firstWriteFrontiers.push_back(frontier);
        }
      }
      covered.insert(members.begin(), members.end());
    }
    loop.sites.assign(covered.begin(), covered.end());
    loop.firstVisitPrefix.clear();
    loop.firstWriteFrontiers.clear();
    for (const auto& occurrence : loop.occurrences) {
      loop.firstVisitPrefix.insert(loop.firstVisitPrefix.end(),
          occurrence.firstVisitPrefix.begin(), occurrence.firstVisitPrefix.end());
      loop.firstWriteFrontiers.insert(loop.firstWriteFrontiers.end(),
          occurrence.firstWriteFrontiers.begin(), occurrence.firstWriteFrontiers.end());
    }
    std::sort(loop.firstVisitPrefix.begin(), loop.firstVisitPrefix.end());
    loop.firstVisitPrefix.erase(std::unique(loop.firstVisitPrefix.begin(), loop.firstVisitPrefix.end()),
                               loop.firstVisitPrefix.end());
    std::sort(loop.firstWriteFrontiers.begin(), loop.firstWriteFrontiers.end());
    loop.firstWriteFrontiers.erase(std::unique(loop.firstWriteFrontiers.begin(), loop.firstWriteFrontiers.end()),
                                  loop.firstWriteFrontiers.end());
  }
  return true;
}
} // namespace mlir::pto::oahs
#endif
