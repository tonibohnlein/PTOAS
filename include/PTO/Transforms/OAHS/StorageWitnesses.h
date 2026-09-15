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

namespace mlir::pto::oahs {
// A group means one IDENTICAL imported footprint, never a connected component
// of may-alias. Each original effect remains in its group's access population.
struct FootprintUse {
  std::size_t operation;
  bool read, write;
};
struct FootprintGroup {
  Cell description;
  std::vector<FootprintUse> uses;
};
struct WitnessCounts {
  std::size_t groups = 0, aliasQueries = 0, pairCells = 0;
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
      cell.provenance = "pairwise imported footprint overlap";
      cell.exclusive |= y.description.exclusive;
      cell.unknownRange |= y.description.unknownRange;
      cell.ranges.insert(cell.ranges.end(), y.description.ranges.begin(), y.description.ranges.end());
      add(std::move(cell), x, &y); ++counts.pairCells;
    }
  return counts;
}
} // namespace mlir::pto::oahs
#endif
