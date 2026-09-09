// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>

using namespace mlir;
using namespace mlir::pto;
namespace {
using Pair = std::pair<unsigned, unsigned>;
using Lane = PipelineType;
using Space = AddressSpace;
struct Case {
  std::string name;
  std::vector<std::unique_ptr<BaseMemInfo>> storage;
  std::vector<SyncPhysicalAccess> accesses;
  explicit Case(std::string name) : name(std::move(name)) {}
  void add(Space scope, bool write, Lane lane, SmallVector<uint64_t> bases,
           uint64_t size, bool physical = true, bool unknown = false) {
    storage.push_back(std::make_unique<BaseMemInfo>(Value(), Value(), scope,
                                                   std::move(bases), size, physical, unknown));
    accesses.push_back({unsigned(accesses.size()), storage.back().get(), write, lane});
  }
};
unsigned assertions = 0;
bool check(bool condition, const std::string &message) {
  ++assertions;
  if (!condition) llvm::errs() << "candidate parity failure: " << message << "\n";
  return condition;
}
SyncAccessCandidates enumerate(const Case &test, uint64_t budget, uint64_t &charged) {
  charged = 0;
  return enumerateSyncAccessCandidates(test.accesses, [&](uint64_t amount) {
    if (amount > budget - charged) return false;
    charged += amount;
    return true;
  });
}
// This oracle deliberately enumerates every access pair. It uses the existing
// alias contract, not the sweep's intervals, active sets, or candidate output.
std::set<Pair> allPairs(const Case &test) {
  std::set<Pair> expected;
  for (unsigned a = 0; a < test.accesses.size(); ++a)
    for (unsigned b = a; b < test.accesses.size(); ++b) {
      const auto &x = test.accesses[a], &y = test.accesses[b];
      bool ordering = x.write || y.write ||
          (x.memory->scope == Space::ACC && y.memory->scope == Space::ACC && x.lane != y.lane);
      if (ordering && logicalSyncMayAlias(x.memory, y.memory, {}, InsertSyncGMAliasMode::MayAlias))
        expected.emplace(a, b);
    }
  return expected;
}
bool parity(const Case &test, llvm::json::Array &rows) {
  uint64_t charged;
  auto actual = enumerate(test, 1000000, charged);
  if (!check(actual.status == SyncAccessCandidates::Status::Complete, test.name + " complete") ||
      !check(actual.work == charged, test.name + " accounting")) return false;
  std::set<Pair> candidates(actual.pairs.begin(), actual.pairs.end()), filtered;
  if (!check(candidates.size() == actual.pairs.size() &&
             std::is_sorted(actual.pairs.begin(), actual.pairs.end()), test.name + " deterministic uniqueness"))
    return false;
  for (auto [a, b] : actual.pairs) {
    if (!check(a <= b && b < test.accesses.size(), test.name + " access identity")) return false;
    if (logicalSyncMayAlias(test.accesses[a].memory, test.accesses[b].memory, {}, InsertSyncGMAliasMode::MayAlias))
      filtered.emplace(a, b);
  }
  auto expected = allPairs(test);
  if (!check(filtered == expected, test.name + " exact all-pairs parity after original alias filter")) return false;
  rows.push_back(llvm::json::Object{{"case", test.name}, {"accesses", int64_t(test.accesses.size())},
      {"candidates", int64_t(actual.pairs.size())}, {"qualified_pairs", int64_t(expected.size())},
      {"work", int64_t(actual.work)}, {"interval_visits", int64_t(actual.intervalVisits)},
      {"candidate_visits", int64_t(actual.candidateVisits)}});
  return true;
}
} // namespace

int main() {
  llvm::json::Array rows, scaling;
  std::vector<Case> tests;
  auto sample = [&](const char *name) -> Case& { return tests.emplace_back(name); };
  auto &touch = sample("touching_half_open");
  touch.add(Space::VEC, true, Lane::PIPE_MTE2, {0}, 16);
  touch.add(Space::VEC, false, Lane::PIPE_V, {16}, 16);
  touch.add(Space::VEC, true, Lane::PIPE_MTE2, {32}, 16);
  if (!check(allPairs(touch) == std::set<Pair>{{0, 0}, {2, 2}}, "touching has no cross-access conflict")) return 1;
  auto &overlap = sample("one_byte_overlap");
  overlap.add(Space::VEC, true, Lane::PIPE_MTE2, {0}, 32);
  overlap.add(Space::VEC, false, Lane::PIPE_V, {31}, 16);
  if (!check(allPairs(overlap).count({0, 1}), "one-byte overlap retained")) return 1;
  auto &multi = sample("multiple_bases_and_duplicate_fragments");
  multi.add(Space::VEC, true, Lane::PIPE_MTE2, {0, 0, 128}, 32);
  multi.add(Space::VEC, false, Lane::PIPE_V, {64, 144}, 16);
  multi.add(Space::VEC, false, Lane::PIPE_V, {32, 96}, 16);
  auto &unknown = sample("unknown_geometry_and_overflow");
  unknown.add(Space::VEC, true, Lane::PIPE_MTE2, {0}, 16);
  unknown.add(Space::VEC, false, Lane::PIPE_V, {4096}, 16, false);
  unknown.add(Space::VEC, false, Lane::PIPE_V, {4096}, 16, true, true);
  unknown.add(Space::VEC, false, Lane::PIPE_V, {}, 16);
  unknown.add(Space::VEC, true, Lane::PIPE_MTE2, {8192}, 0);
  unknown.add(Space::VEC, true, Lane::PIPE_MTE2, {std::numeric_limits<uint64_t>::max() - 8}, 16);
  unknown.add(Space::VEC, false, Lane::PIPE_V, {std::numeric_limits<uint64_t>::max() - 16}, 16);
  auto &spaces = sample("distinct_address_spaces");
  for (Space scope : {Space::VEC, Space::MAT, Space::ACC, Space::GM}) {
    spaces.add(scope, true, Lane::PIPE_MTE2, {0}, 32);
    spaces.add(scope, false, Lane::PIPE_V, {0}, 32);
  }
  auto &acc = sample("acc_read_resources");
  acc.add(Space::ACC, false, Lane::PIPE_M, {0}, 64);
  acc.add(Space::ACC, false, Lane::PIPE_M, {0}, 64);
  acc.add(Space::ACC, false, Lane::PIPE_FIX, {32}, 64);
  acc.add(Space::ACC, false, Lane::PIPE_FIX, {256}, 64, false);
  if (!check(!allPairs(acc).count({0, 1}) && allPairs(acc).count({0, 2}) &&
             allPairs(acc).count({0, 3}), "ACC lane and unknown-range obligations")) return 1;
  auto &readonly = sample("non_acc_read_only");
  readonly.add(Space::VEC, false, Lane::PIPE_V, {0}, 64);
  readonly.add(Space::VEC, false, Lane::PIPE_MTE3, {0}, 64, false, true);
  if (!check(allPairs(readonly).empty(), "ordinary read/read is not a hazard")) return 1;
  auto &gm = sample("gm_retains_contract_filtering");
  gm.add(Space::GM, true, Lane::PIPE_MTE3, {0}, 16);
  gm.add(Space::GM, false, Lane::PIPE_MTE2, {1024}, 16);
  gm.add(Space::GM, false, Lane::PIPE_MTE2, {0}, 16, false, true);
  auto &samePhase = sample("separate_effects_in_one_phase");
  samePhase.add(Space::VEC, false, Lane::PIPE_V, {0}, 32);
  samePhase.add(Space::VEC, true, Lane::PIPE_V, {16}, 32);
  samePhase.accesses.back().phase = 0;

  uint32_t seed = 0x7384a9u;
  auto random = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
  for (unsigned trial = 0; trial < 64; ++trial) {
    Case test("seeded_" + std::to_string(trial));
    unsigned count = 8 + random() % 17;
    for (unsigned i = 0; i < count; ++i) {
      Space scope = (random() % 3 == 0) ? Space::ACC : Space::VEC;
      SmallVector<uint64_t> bases;
      unsigned fragments = random() % 4;
      for (unsigned f = 0; f < fragments; ++f) bases.push_back((random() % 12) * 16);
      test.add(scope, bool(random() & 1), (random() & 1) ? Lane::PIPE_M : Lane::PIPE_FIX,
               std::move(bases), random() % 65, random() % 7 != 0, random() % 11 == 0);
    }
    tests.push_back(std::move(test));
  }
  for (const auto &test : tests) if (!parity(test, rows)) return 1;

  for (bool known : {true, false}) for (unsigned n : {64u, 256u, 1024u}) {
    Case test("same_lane_ACC_readonly");
    for (unsigned i = 0; i < n; ++i) test.add(Space::ACC, false, Lane::PIPE_M, {0}, 64, known);
    uint64_t charged;
    auto result = enumerate(test, uint64_t(n) * 4, charged);
    if (!check(result.status == SyncAccessCandidates::Status::Complete && result.pairs.empty(), "ACC scaling empty") ||
        !check(result.candidateVisits == (known ? n : 0) && result.intervalVisits == (known ? n : 0),
               "ACC readonly visits stay linear") ||
        !check(result.work == charged && charged == uint64_t(n) * (known ? 3 : 2), "ACC scaling accounting")) return 1;
    scaling.push_back(llvm::json::Object{{"accesses", n}, {"known", known}, {"work", int64_t(charged)},
        {"candidate_visits", int64_t(result.candidateVisits)}, {"interval_visits", int64_t(result.intervalVisits)}});
  }
  Case limited("budget");
  for (unsigned i = 0; i < 8; ++i) limited.add(Space::VEC, true, Lane::PIPE_MTE2, {0}, 64);
  for (uint64_t budget : {0u, 1u, 16u, 20u}) {
    uint64_t charged;
    auto result = enumerate(limited, budget, charged);
    if (!check(result.status == SyncAccessCandidates::Status::AnalysisLimit && result.pairs.empty() &&
               result.work == charged && charged <= budget, "budget failure exposes no partial candidates")) return 1;
  }
  limited.accesses[0].memory = nullptr;
  uint64_t charged;
  auto invalid = enumerate(limited, 1000, charged);
  if (!check(invalid.status == SyncAccessCandidates::Status::InvalidInput && invalid.pairs.empty(), "null access rejected")) return 1;
  llvm::outs() << llvm::json::Value(llvm::json::Object{{"status", "passed"}, {"assertions", assertions},
      {"cases", std::move(rows)}, {"readonly_scaling", std::move(scaling)}, {"device", "not-run"}}) << "\n";
  return 0;
}
