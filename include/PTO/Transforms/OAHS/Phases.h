// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_PHASES_H
#define PTO_TRANSFORMS_OAHS_PHASES_H
#include "PTO/Transforms/OAHS/Analysis.h"
namespace mlir::pto::oahs {
  struct PhaseAccess {
    Access effect;
    std::size_t begin = 0, end = 1;
    bool permission = false;
  };
  struct PhaseFragment {
    std::vector<std::string> endpoints;
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::vector<PhaseAccess> accesses;
  };
  // Validates the ENTIRE supplied profile and all signatures before exposing a
  // fragment. Internal endpoints remain analytical and never become legal cuts.
  bool phaseFragment(const Program &, std::size_t operation, PhaseFragment &, std::string &reason);
  bool validatePhaseContract(const Program &, std::string &reason);
  struct PhaseOrderWitness {
    Cut cut = 0;
    std::size_t operation = 0;
    std::string target;
    bool withinOperation = false;
  };
  struct PhaseOrderResult {
    bool complete = false, safe = false, exact = false;
    std::string reason;
    std::vector<PhaseOrderWitness> excess;
    AnalysisStats stats;
  };
  // Uses the same phase semantics with a separate ideal graph. This is an
  // optional quality query, not a different planner and not a device cost model.
  PhaseOrderResult checkPhaseOrder(const Program &, const Commands &);
  struct PhaseNativeQualification {
    bool enabled = false;
    std::vector<std::string> missing;
  };
  // No native adapter is qualified by this source series. A supplied reference
  // contract (or a caller-written boolean) must not turn on UnitFlag credit.
  PhaseNativeQualification phaseNativeQualification();
}
// namespace mlir::pto::oahs
#endif
