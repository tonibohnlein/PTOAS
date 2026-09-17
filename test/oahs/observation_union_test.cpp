// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "../../lib/PTO/Transforms/OAHS/ObservationUnion.h"
#include <cstdlib>
#include <iostream>
#include <random>

namespace o = mlir::pto::oahs;
using namespace o::detail;
static void require(bool ok) { if (!ok) std::abort(); }
static bool evaluate(const ObservationUnion &u, unsigned residue, bool previous, bool next, bool other) {
  for (const auto &term : u) {
    bool match = true;
    for (const auto &a : term) {
      const auto value = a.owner == 1 ? unsigned(other) :
          a.kind == o::ObservationAtom::LoopResidue ? residue :
          a.kind == o::ObservationAtom::LoopHasPrevious ? unsigned(previous) : unsigned(next);
      match &= a.value == value;
    }
    if (match) return true;
  }
  return false;
}
int main() {
  using A = o::ObservationAtom;
  // Equal words for every first/tail alternative need only the residue test.
  ObservationUnion full;
  for (unsigned prev = 0; prev < 2; ++prev)
    for (unsigned next = 0; next < 2; ++next)
      full.push_back({{A::LoopResidue, 0, 2, 1}, {A::LoopHasPrevious, 0, 2, prev}, {A::LoopHasNext, 0, 2, next}});
  auto reduced = simplifyObservationUnion(full);
  require(reduced.size() == 1 && reduced[0].size() == 1 && reduced[0][0].kind == A::LoopResidue);
  full.pop_back();
  require(!evaluate(simplifyObservationUnion(full), 1, true, true, false));
  std::mt19937 random(41721);
  unsigned comparisons = 0;
  for (unsigned period = 1; period <= 4; ++period) {
    for (unsigned trial = 0; trial < 1000; ++trial) {
      ObservationUnion input;
      for (unsigned r = 0; r < period; ++r)
        for (unsigned bits = 0; bits < 8; ++bits) {
          if (random() % 3 == 0) continue;
          ObservationTerm term;
          if (random() % 3) term.push_back({A::LoopResidue, 0, period, r});
          if (random() % 3) term.push_back({A::LoopHasPrevious, 0, period, bits & 1});
          if (random() % 3) term.push_back({A::LoopHasNext, 0, period, (bits >> 1) & 1});
          if (random() % 3) term.push_back({A::LoopHasPrevious, 1, period, (bits >> 2) & 1});
          input.push_back(std::move(term));
        }
      auto output = simplifyObservationUnion(input);
      for (unsigned r = 0; r < period; ++r)
        for (unsigned bits = 0; bits < 8; ++bits) {
          require(evaluate(input, r, bits & 1, bits & 2, bits & 4) ==
                  evaluate(output, r, bits & 1, bits & 2, bits & 4));
          ++comparisons;
        }
    }
  }
  std::cout << comparisons << " original-observation union equivalence checks passed\n";
}
