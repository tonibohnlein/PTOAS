// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_BUNDLES_H
#define PTO_TRANSFORMS_OAHS_BUNDLES_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include <memory>

namespace mlir::pto::oahs {
struct BundleResources {
  std::vector<EventIdentity> keys;
  std::size_t publications = 0, acquisitions = 0, fences = 0, allFences = 0;
};
struct BundleEvaluation {
  // A completed replay is not acceptance of the candidate. The full original
  // obligation population remains in analysis; only analysis.verified()
  // accepts.
  bool complete = false;
  std::string reason;
  Commands commands;
  AnalysisResult analysis;
  std::vector<CompletionRequirement> discharged, introduced;
  // Endpoint indices can shift after an edit. These differences use cut, key,
  // and failure kind (with multiplicity), not unstable command offsets.
  std::vector<ProtocolObligation> resolvedProtocol, introducedProtocol;
  std::vector<RetirementRequirement> retired, introducedRetirement;
  std::vector<PhaseResourceObligation> resolvedResources, introducedResources;
  std::vector<Cut> changedCuts;
  BundleResources resources;
  bool memoryProgressAt(Cut cut) const {
    for (const auto &r : discharged)
      if (r.consumerCut == cut)
        return true;
    return false;
  }
  bool protocolProgress() const { return !resolvedProtocol.empty() || !resolvedResources.empty(); }
};
// A read-only transaction evaluator pinned to original effects AND the current
// actual command population. The caller supplies the complete trial commands,
// including helpers and key assignments; there is no claimed-credit input.
// This API does not infer guard availability or permit new cut kinds.
class BundleQuery {
public:
  BundleQuery(Program, Commands, AnalysisOptions = {});
  ~BundleQuery();
  BundleQuery(BundleQuery &&) noexcept;
  BundleQuery &operator=(BundleQuery &&) noexcept;
  BundleQuery(const BundleQuery &) = delete;
  BundleQuery &operator=(const BundleQuery &) = delete;
  const AnalysisResult &analysis() const;
  BundleEvaluation evaluate(Commands) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::oahs
#endif
