// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_OBSERVATIONS_H
#define PTO_TRANSFORMS_OAHS_OBSERVATIONS_H
#include <cstddef>
#include <cstdint>
#include <limits>
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
struct ObservedLoop {
  std::size_t owner = NoControlId, entry = NoControlId, exit = NoControlId;
  std::vector<std::size_t> sites;
};
struct ObservedControl {
  std::vector<ObservedSite> sites;
  std::vector<OriginalObservation> observations;
  std::vector<ObservedScope> scopes;
  std::vector<ObservedLoop> loops;
  std::size_t entry = 0, exit = 0;
  // Named input/frontend proof boundary, not a causal-completion assertion.
  std::string qualification;
};
} // namespace mlir::pto::oahs
#endif
