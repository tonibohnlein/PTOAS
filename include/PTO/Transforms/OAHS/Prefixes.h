// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_PREFIXES_H
#define PTO_TRANSFORMS_OAHS_PREFIXES_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include <memory>

namespace mlir::pto::oahs {
struct BackwardCut {
  Cut cut = 0;
  std::size_t context = 0;
  bool withoutBackedge = false, throughBackedge = false;
};
struct BackwardCutResult {
  bool complete = false;
  std::string reason;
  Cut consumer = 0;
  std::vector<BackwardCut> cuts;
  std::vector<std::size_t> crossedLoopOwners;
  std::size_t visitedStates = 0;
};
struct ProspectivePrefix {
  // This names an ANALYSIS snapshot, not a live event. Publication is before
  // the existing commands at its original physical cut; acquisition is after
  // the existing commands at the consumer cut, immediately before its payload.
  // No command offset, executable guard, new key, or dynamic rank is invented.
  Pipe source = Pipe::S, observer = Pipe::S;
  Cut publication = 0, acquisition = 0;
  std::size_t sourceContext = 0, targetContext = 0;
  bool complete = false;
  std::string reason;
  bool availableAtEveryAcquisition = false;
  bool matchingEstablished = false; // reference alternation, NOT causal rearm
  bool routeAvailable = false;      // eligible mechanism exists, NOT allocated
  bool crossesBackedge =
      false; // some backward path; NOT a selected epoch/distance
  bool repeatedPublication = false, unconsumedAtExit = false;
  AnalysisBits uncoveredOperations;
  // Positions in this query's consumerRequirements(acquisition), not operation
  // numbers/ranks. Independent components remain a conjunction.
  std::vector<std::size_t> coveredRequirements;
  std::size_t siteEvaluations = 0;
  bool selectable() const {
    return complete && matchingEstablished && routeAvailable &&
           !coveredRequirements.empty();
  }
};
struct PrefixCover {
  bool complete = false; // query completed; may still have uncovered components
  std::string reason;
  Cut consumer = 0;
  std::vector<CompletionRequirement> requirements;
  BackwardCutResult backward;
  std::vector<ProspectivePrefix> candidates;
  std::vector<std::size_t> selected;  // indices into candidates
  std::vector<std::size_t> remaining; // indices into requirements
  AnalysisBits
      jointUncoveredOperations; // intersection of the selected remainders
  bool coversAll() const { return complete && remaining.empty(); }
};

// Read-only queries bound to an OWNED immutable copy of one program and actual
// plan. No externally supplied/stale AnalysisResult is accepted as a
// certificate. A caller editing its plan creates a new query. Returned records
// are proposals: event matching, generation reuse, progress and ALL original
// demands are still checked on the complete realized candidate. No keys are
// reserved by this API.
class PrefixQuery {
public:
  PrefixQuery(Program, Commands);
  explicit PrefixQuery(Program);
  ~PrefixQuery();
  PrefixQuery(PrefixQuery &&) noexcept;
  PrefixQuery &operator=(PrefixQuery &&) noexcept;
  PrefixQuery(const PrefixQuery &) = delete;
  PrefixQuery &operator=(const PrefixQuery &) = delete;
  const AnalysisResult &analysis() const;
  std::vector<CompletionRequirement> consumerRequirements(Cut) const;
  BackwardCutResult backwardCuts(Cut consumer) const;
  ProspectivePrefix inspectPrefix(Pipe source, Cut publication,
                                  Cut consumer) const;
  // Deterministic set-cover heuristic: greatest newly covered residual count,
  // then original lexical source cut, then lane. No global optimality claim.
  PrefixCover coverByPrefixes(Cut consumer) const;
  // Proposal restriction only; required effects and certified analysis
  // unchanged.
  PrefixCover coverByPrefixes(Cut consumer,
                              const std::vector<Cut> &allowed) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::oahs
#endif
