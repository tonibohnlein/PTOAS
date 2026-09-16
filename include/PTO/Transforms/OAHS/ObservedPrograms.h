// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_OBSERVEDPROGRAMS_H
#define PTO_TRANSFORMS_OAHS_OBSERVEDPROGRAMS_H
#include "PTO/Transforms/OAHS/Plan.h"
#include <tuple>
namespace mlir::pto::oahs {
struct PeriodicAccess {
  std::size_t operation = 0, access = 0;
  std::vector<unsigned> cells;
  uint64_t stride = 1, offset = 0;
};
struct ObservedImport {
  bool success = false;
  std::string reason;
  Program program;
  std::vector<std::size_t> originalPhases;
};
// Qualifies ONE normalized i=0..N-1 loop with an arbitrary flat body and exact
// affine-modular cell selectors. It does not choose synchronization or expand
// dynamic iterations. The caller explicitly selects a common period; unrelated
// scopes are never silently multiplied. This is not a general MLIR frontend.
ObservedImport makePeriodicLoop(const Program &body, unsigned period,
                                const std::vector<PeriodicAccess> &bindings);
// Exposes original structured entry/exit positions as control-only cuts. The
// abstract Region input declares those boundaries legal; native import supplies
// its own exact anchors. No fake payload operation is introduced.
ObservedImport addStructuredBoundaryCuts(const Program &);
// A checked normalized loop region in an already qualified original CFG.
// The descriptor comes from the frontend, never from a selected protocol.
struct ResidueDecision {
  std::size_t site = 0;
  uint64_t modulus = 1, residue = 0;
  bool equal = true;
};
struct CountedLoopRegion {
  std::size_t owner = 0, header = 0, bodyEntry = 0, continuation = 0;
  unsigned period = 1;
  std::vector<std::size_t> bodySites;
  std::vector<ResidueDecision> decisions;
};
// Refines original control only; no physical phase or command is inserted.
// Assumes original i=0..N-1, step 1. Native import discharges this premise.
ObservedImport refineCountedLoop(const Program &, const CountedLoopRegion &);
struct ObservedWord {
  OriginalObservation observation;
  std::vector<Command> commands;
};
struct ObservedEndpoint {
  std::size_t observation = 0, command = 0;
  bool operator<(const ObservedEndpoint &b) const {
    return std::tie(observation, command) < std::tie(b.observation, b.command);
  }
  bool operator==(const ObservedEndpoint &b) const {
    return observation == b.observation && command == b.command;
  }
};
struct ObservedCorrespondence {
  ObservedEndpoint acquisition;
  std::vector<ObservedEndpoint> publications;
};
struct ObservedSchema {
  bool complete = false;
  std::string reason;
  std::vector<ObservedWord> words;
  // Exact marginal publisher alternatives under original reference control.
  // This is not an affine generation distance or a substitute for causal reuse.
  std::vector<ObservedCorrespondence> correspondence;
};
ObservedSchema exportObservedSchema(const Program &, const Commands &);
// Match actual original anchors AND predicate descriptors, not numeric solver
// state. Empty omitted words are permitted; verify() must validate the result.
Result reconstructObservedSchema(const Program &, const ObservedSchema &);
} // namespace mlir::pto::oahs
#endif
