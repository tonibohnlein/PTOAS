// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_STORAGEWITNESSES_H
#define PTO_OAHS_STORAGEWITNESSES_H
#include "PTO/Transforms/OAHS/Plan.h"
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <map>
#include <set>
#include <iterator>

namespace mlir::pto::oahs {
// A group means one IDENTICAL imported footprint, never a connected component
// of may-alias. Each original effect remains in its group's access population.
struct FootprintUse {
  std::size_t operation;
  bool read, write;
  bool definiteWrite = false;
};
struct FootprintGroup {
  Cell description;
  std::vector<FootprintUse> uses;
};
struct WitnessCounts {
  std::size_t groups = 0, aliasQueries = 0, pairCells = 0;
  std::size_t canonicalCells = 0;
};
// Work depends on distinct footprints, not on a cross product of repeated
// operations. Worst-case distinct-footprint overlap can still be quadratic;
// it is completed rather than converted to a different synchronization plan.
// Non-transitive alias relations remain pair witnesses: A~C and B~C do not
// turn A and B into aliases. Singleton cells retain self recurrence.
template <typename MayAlias>
WitnessCounts appendStorageWitnesses(Program &p,
    const std::vector<FootprintGroup> &groups, MayAlias mayAlias) {
  WitnessCounts counts; counts.groups = groups.size();
  auto add = [&](Cell description, const FootprintGroup &a,
                 const FootprintGroup *b) {
    if (p.cells.size() >= std::numeric_limits<unsigned>::max())
      std::abort(); // intrinsic unsigned cell-identity capacity, not a work budget
    const auto cell = static_cast<unsigned>(p.cells.size());
    p.cells.push_back(std::move(description));
    auto attach = [&](const FootprintGroup &g) {
      for (const auto &use : g.uses)
        p.operations.at(use.operation).accesses.push_back({cell, use.read, use.write});
    };
    attach(a); if (b) attach(*b);
  };
  for (const auto &group : groups) add(group.description, group, nullptr);
  for (std::size_t a = 0; a < groups.size(); ++a)
    for (std::size_t b = a + 1; b < groups.size(); ++b) {
      const auto &x = groups[a], &y = groups[b];
      if (x.description.addressSpace != y.description.addressSpace) continue;
      const auto writes = [](const FootprintGroup &g) {
        return std::any_of(g.uses.begin(), g.uses.end(),
                           [](const FootprintUse &u) { return u.write; });
      };
      if (!writes(x) && !writes(y) && !x.description.exclusive && !y.description.exclusive) continue;
      ++counts.aliasQueries;
      if (!mayAlias(a, b)) continue;
      Cell cell = x.description;
      cell.storage = Cell::Storage::OverlapWitness;
      cell.coordinateSpace.clear();
      cell.provenance = "pairwise imported footprint overlap";
      cell.exclusive |= y.description.exclusive;
      cell.unknownRange |= y.description.unknownRange;
      cell.ranges.insert(cell.ranges.end(), y.description.ranges.begin(), y.description.ranges.end());
      add(std::move(cell), x, &y); ++counts.pairCells;
    }
  return counts;
}

// Partition qualified bounding intervals at their endpoints. Unknown records
// keep independent recurrence cells and pairwise witnesses under the SAME
// production alias query. In particular, unknown U overlapping disjoint A and
// B cannot merge A and B. Coordinate identity must come from shared extraction,
// never from an SSA allocation name or a numeric address without its domain.
template <typename MayAlias>
WitnessCounts appendCanonicalStorage(Program& p, const std::vector<FootprintGroup>& groups, MayAlias mayAlias)
{
    WitnessCounts counts;
    counts.groups = groups.size();
    std::vector<bool> qualified(groups.size());
    using Space = std::pair<std::string, std::string>;
    std::map<Space, std::vector<std::size_t>> spaces;
    auto add = [&](Cell cell, const std::set<std::size_t>& members) {
        if (p.cells.size() >= std::numeric_limits<unsigned>::max())
            std::abort();
        const auto id = static_cast<unsigned>(p.cells.size());
        const bool exact = cell.storage == Cell::Storage::CanonicalInterval;
        for (auto g : members)
            cell.storageOrigins.insert(
                cell.storageOrigins.end(), groups[g].description.storageOrigins.begin(),
                groups[g].description.storageOrigins.end());
        std::sort(cell.storageOrigins.begin(), cell.storageOrigins.end());
        cell.storageOrigins.erase(
            std::unique(cell.storageOrigins.begin(), cell.storageOrigins.end()), cell.storageOrigins.end());
        p.cells.push_back(std::move(cell));
        for (auto g : members)
            for (const auto& u : groups[g].uses)
                p.operations.at(u.operation)
                    .accesses.push_back({id, u.read, u.write, exact && u.write && u.definiteWrite});
    };
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const auto& c = groups[g].description;
        // Exclusion is a separate resource relationship. Folding one exclusive
        // footprint into an ordinary byte atom could make two unrelated plain
        // readers exclude each other through that third footprint.
        qualified[g] = !c.exclusive && !c.unknownRange && !c.coordinateSpace.empty() && c.ranges.size() == 1 &&
                       c.ranges[0].second &&
                       c.ranges[0].second <= std::numeric_limits<uint64_t>::max() - c.ranges[0].first;
        if (qualified[g])
            spaces[{c.addressSpace, c.coordinateSpace}].push_back(g);
        else {
            auto conservative = c;
            conservative.storage = Cell::Storage::OverlapWitness;
            conservative.coordinateSpace.clear();
            add(std::move(conservative), {g});
        }
    }
    for (const auto& [space, members] : spaces) {
        struct Ends {
            std::vector<std::size_t> start, stop;
        };
        std::map<uint64_t, Ends> ends;
        for (auto g : members) {
            const auto [begin, size] = groups[g].description.ranges.front();
            ends[begin].start.push_back(g);
            ends[begin + size].stop.push_back(g);
        }
        std::set<std::size_t> active;
        for (auto at = ends.begin(); at != ends.end(); ++at) {
            for (auto g : at->second.stop)
                active.erase(g);
            for (auto g : at->second.start)
                active.insert(g);
            const auto next = std::next(at);
            if (active.empty() || next == ends.end())
                continue;
            Cell atom;
            atom.storage = Cell::Storage::CanonicalInterval;
            atom.addressSpace = space.first;
            atom.coordinateSpace = space.second;
            atom.provenance = "shared physical bounding-interval partition";
            atom.ranges = {{at->first, next->first - at->first}};
            for (auto g : active)
                atom.exclusive |= groups[g].description.exclusive;
            add(std::move(atom), active);
            ++counts.canonicalCells;
        }
    }
    for (std::size_t a = 0; a < groups.size(); ++a)
        for (std::size_t b = a + 1; b < groups.size(); ++b) {
            const auto &x = groups[a], &y = groups[b];
            if (x.description.addressSpace != y.description.addressSpace)
                continue;
            if (qualified[a] && qualified[b] && x.description.coordinateSpace == y.description.coordinateSpace)
                continue; // Their exact coordinate overlap is already represented.
            const auto writes = [](const FootprintGroup& g) {
                return std::any_of(g.uses.begin(), g.uses.end(), [](const FootprintUse& u) { return u.write; });
            };
            if (!writes(x) && !writes(y) && !x.description.exclusive && !y.description.exclusive)
                continue;
            ++counts.aliasQueries;
            if (!mayAlias(a, b))
                continue;
            Cell witness;
            witness.storage = Cell::Storage::OverlapWitness;
            witness.addressSpace = x.description.addressSpace;
            witness.provenance = "shared conservative alias relation";
            witness.unknownRange = true;
            witness.ranges = x.description.ranges;
            witness.ranges.insert(witness.ranges.end(), y.description.ranges.begin(), y.description.ranges.end());
            witness.exclusive = x.description.exclusive || y.description.exclusive;
            add(std::move(witness), {a, b});
            ++counts.pairCells;
        }
    return counts;
}
} // namespace mlir::pto::oahs
#endif
