// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_REPLAY_H
#define PTO_TRANSFORMS_OAHS_REPLAY_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include <memory>

namespace mlir::pto::oahs {
struct ReplayStats {
  enum Kind { Cold, Incremental, Unchanged, KeyLayoutChanged, PhaseFull,
              Disabled, Invalid } kind = Cold;
  std::size_t changedCuts = 0, invalidatedSites = 0, reusedSites = 0;
  uint64_t boundaryEvaluations = 0;
};
// One immutable original program, one last-candidate checkpoint. Successive
// calls may insert, remove or move commands; they never mutate the program or
// a previously returned report. An unsuccessful trial is not committed to any
// caller plan. Malformed commands do not replace the last complete checkpoint.
//
// Compact replay reuses only the predecessor-closed, unchanged part of the
// PROVISIONAL fixed point, not old certificate decisions. Dirty successors,
// including entire reachable loop SCCs, restart from bottom. Certificate
// revocation is repeated from scratch; its subsequent passes use full solves.
// A changed key-index layout or the phase collecting backend uses full replay.
// Final verify() intentionally remains an independent cold analysis.
//
// The session is move-only and not thread-safe. The bool is a developer
// differential-test ablation, not an algorithm mode or analysis-work allowance.
class ReplaySession {
public:
  explicit ReplaySession(Program, bool incremental = true);
  ~ReplaySession();
  ReplaySession(ReplaySession &&) noexcept;
  ReplaySession &operator=(ReplaySession &&) noexcept;
  ReplaySession(const ReplaySession &) = delete;
  ReplaySession &operator=(const ReplaySession &) = delete;
  AnalysisResult analyze(const Commands &, AnalysisOptions = {});
  ReplayStats lastReplay() const;
  void clear();
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::oahs
#endif
