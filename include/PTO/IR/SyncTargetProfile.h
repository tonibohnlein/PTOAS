// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_IR_SYNCTARGETPROFILE_H
#define PTO_IR_SYNCTARGETPROFILE_H
#include <array>
#include <string>
#include <vector>
namespace mlir::pto {
enum class SyncPipe : unsigned { S, V, M, MTE1, MTE2, MTE3, FIX, Count };
enum class SyncCore { Cube, Vector };
constexpr unsigned SyncPipeCount = unsigned(SyncPipe::Count);
struct SyncTargetProfile {
  std::string contract;
  std::array<bool, SyncPipeCount> supported{}, barriers{}, synchronous{};
  std::array<std::array<std::vector<unsigned>, SyncPipeCount>, SyncPipeCount> keys;
  bool barrierAll = false;
};
// A source-qualified conservative profile, not new device qualification.
// Topology: c73c04fb3 / StructuredSyncCore.cpp Target::{supports,event,barrier}.
// Pool 0..5 is the selected static-library-safe pool; hardware IDs 6/7 are not
// asserted nonexistent. Private/authored reservations further subtract keys.
// This shared semantic table contains no placement or allocation strategy.
inline SyncTargetProfile a3SyncProfile(SyncCore core) {
  using P = SyncPipe;
  SyncTargetProfile t;
  t.contract = core == SyncCore::Vector ? "a3-aiv-prefix-static-safe-v1" : "a3-aic-prefix-static-safe-v1";
  auto enable = [&](P p) {
    t.supported[unsigned(p)] = true;
    t.barriers[unsigned(p)] = p != P::S;
    t.synchronous[unsigned(p)] = p == P::S;
  };
  for (P p : {P::S, P::MTE2, P::MTE3}) enable(p);
  if (core == SyncCore::Vector) enable(P::V);
  else for (P p : {P::M, P::MTE1, P::FIX}) enable(p);
  auto direction = [&](P a, P b) {
    if (a == b || !t.supported[unsigned(a)] || !t.supported[unsigned(b)]) return false;
    if (core == SyncCore::Vector) return true;
    if (a == P::S || b == P::S) return false;
    if ((a == P::FIX && (b == P::MTE2 || b == P::MTE3)) ||
        (b == P::FIX && (a == P::MTE2 || a == P::MTE3))) return false;
    if (a == P::M) return b == P::MTE1 || b == P::MTE2 || b == P::FIX;
    if (a == P::MTE3) return b == P::MTE1 || b == P::MTE2;
    return true;
  };
  for (unsigned a = 0; a < SyncPipeCount; ++a)
    for (unsigned b = 0; b < SyncPipeCount; ++b)
      if (direction(P(a), P(b))) t.keys[a][b] = {0,1,2,3,4,5};
  t.barrierAll = true;
  return t;
}
} // namespace mlir::pto
#endif
