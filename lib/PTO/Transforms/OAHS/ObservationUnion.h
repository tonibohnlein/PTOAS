// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_OBSERVATION_UNION_H
#define PTO_OAHS_OBSERVATION_UNION_H
#include "PTO/Transforms/OAHS/Observations.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::oahs::detail {
using ObservationTerm = std::vector<ObservationAtom>;
using ObservationUnion = std::vector<ObservationTerm>;
inline auto atomKey(const ObservationAtom &a) {
  return std::make_tuple(a.owner, unsigned(a.kind), a.parameter, a.value);
}
inline bool atomLess(const ObservationAtom &a, const ObservationAtom &b) {
  return atomKey(a) < atomKey(b);
}
inline bool termLess(const ObservationTerm &a, const ObservationTerm &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), atomLess);
}
// Exact disjunction algebra over qualified original-value observations. No
// reachability, event state, or selected completion is used to drop a guard.
// Terms can absorb supersets, or eliminate a dimension only when every value
// of that dimension occurs with exactly the same remaining conjunction.
inline ObservationUnion simplifyObservationUnion(ObservationUnion terms) {
  for (auto &term : terms) {
    std::sort(term.begin(), term.end(), atomLess);
    term.erase(std::unique(term.begin(), term.end(), [](const auto &a, const auto &b) {
      return atomKey(a) == atomKey(b);
    }), term.end());
  }
  while (true) {
    std::sort(terms.begin(), terms.end(), termLess);
    terms.erase(std::unique(terms.begin(), terms.end(), [](const auto &a, const auto &b) {
      return !termLess(a, b) && !termLess(b, a);
    }), terms.end());
    bool changed = false;
    for (std::size_t i = 0; i < terms.size() && !changed; ++i)
      for (std::size_t j = 0; j < terms.size(); ++j)
        if (i != j && std::includes(terms[j].begin(), terms[j].end(),
                                   terms[i].begin(), terms[i].end(), atomLess)) {
          terms.erase(terms.begin() + j);
          changed = true;
          break;
        }
    if (changed) continue;
    using Dimension = std::tuple<std::size_t, unsigned, uint64_t>;
    struct Group { ObservationTerm rest; std::set<uint64_t> values; std::vector<std::size_t> members; };
    std::map<Dimension, std::vector<Group>> groups;
    for (std::size_t i = 0; i < terms.size(); ++i) {
      for (std::size_t j = 0; j < terms[i].size(); ++j) {
        const auto &a = terms[i][j];
        auto rest = terms[i];
        rest.erase(rest.begin() + j);
        auto &byRest = groups[{a.owner, unsigned(a.kind), a.parameter}];
        auto at = std::find_if(byRest.begin(), byRest.end(), [&](const auto &g) {
          return !termLess(g.rest, rest) && !termLess(rest, g.rest);
        });
        if (at == byRest.end()) { byRest.push_back({std::move(rest), {}, {}}); at = byRest.end() - 1; }
        at->values.insert(a.value);
        at->members.push_back(i);
      }
    }
    for (const auto &[dimension, byRest] : groups) {
      const auto kind = std::get<1>(dimension);
      const auto count = kind == ObservationAtom::LoopResidue ? std::get<2>(dimension) : 2;
      if (!count) continue;
      for (const auto &g : byRest) {
        if (g.values.size() != count || *g.values.begin() != 0 || *g.values.rbegin() != count - 1) continue;
        ObservationUnion next;
        for (std::size_t i = 0; i < terms.size(); ++i)
          if (std::find(g.members.begin(), g.members.end(), i) == g.members.end()) next.push_back(terms[i]);
        next.push_back(g.rest);
        terms = std::move(next);
        changed = true;
        break;
      }
      if (changed) break;
    }
    if (!changed) return terms;
  }
}
} // namespace mlir::pto::oahs::detail
#endif
