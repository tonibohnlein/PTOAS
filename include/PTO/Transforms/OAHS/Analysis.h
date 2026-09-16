// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_ANALYSIS_H
#define PTO_TRANSFORMS_OAHS_ANALYSIS_H
#include "PTO/Transforms/OAHS/Plan.h"
#include <limits>
#include <optional>

namespace mlir::pto::oahs {
constexpr std::size_t NoAnalysisId = std::numeric_limits<std::size_t>::max();
using AnalysisBits = std::vector<uint8_t>;

// Describes original STATIC control, not an executable predicate, a path
// partition, or a dynamic epoch. Different loop visits remain conservatively
// merged at base precision. The parent chain retains the owning scope.
struct AnalysisContext {
  enum Kind {
    Function,
    ThenArm,
    ElseArm,
    ForBody,
    WhileBefore,
    WhileAfter
  } kind = Function;
  std::size_t parent = NoAnalysisId;
  std::size_t ownerSite = NoAnalysisId;
};
struct EventFacts {
  uint8_t possibleOccupancy = 1; // bit 0: empty; bit 1: full
  bool receiptValidOnEveryPath = false;
  AnalysisBits uncoveredOperations;
  AnalysisBits carriedConsumptions; // indices in AnalysisResult::keys
  // Initial empty keys satisfy this vacuously: no earlier consumption needs
  // acknowledging. A set bit is not an assertion that a consumption occurred.
  uint8_t consumptionKnownAt = 0; // bit q: knowledge at engine q
};
struct BoundaryFacts {
  // Bits name original physical phases (all their represented dynamic visits).
  // Absence is NOT evidence that an operation executed; it says no pending
  // occurrence is represented at this observer under the current contracts.
  std::array<AnalysisBits, PipeCount> pending;
  std::vector<EventFacts> events;
  std::vector<PhaseResourceFacts> phaseResources;
};
struct CutFacts {
  Cut cut = 0;
  std::size_t context = 0;
  bool reachable = false;
  // Absent only when unreachable or captureStates=false. These are snapshots
  // for THIS candidate, not reusable certificates after an edit.
  std::optional<BoundaryFacts> incoming, beforeIssue, outgoing;
};
struct CompletionRequirement {
  enum Kind { RAW, WAR, WAW, ExclusiveResource } kind = RAW;
  Demand demand;
  Access producerAccess, consumerAccess;
  // NoAnalysisId when no single original source context is established.
  std::size_t producerContext = 0, consumerContext = 0;
  Cut consumerCut =
      NoAnalysisId; // distinct when a phase has several observed contexts
  // Analytical access endpoints, not legal command cuts. Whole-operation
  // backends leave these unspecified; phase-aware replay retains them.
  std::size_t producerEndpoint = NoAnalysisId, consumerEndpoint = NoAnalysisId;
  // The source is an earlier represented occurrence. producer==consumer can
  // denote self recurrence; no iteration distance is inferred from static IDs.
};
struct ProtocolObligation {
  enum Kind {
    PublicationNotEmpty,
    AcquisitionNotFull,
    ReceiptNotEstablished,
    ConsumptionNotEstablished,
    UnconsumedAtExit
  } kind = PublicationNotEmpty;
  Cut cut = 0;
  std::size_t command = NoAnalysisId;
  std::size_t context = 0;
  EventIdentity event;
  std::string reason;
};
struct CommandFacts {
  Cut cut = 0;
  std::size_t command = 0, context = 0;
  Command endpoint;
  // This is a local primitive-precondition certificate at a stabilized
  // invariant, not acceptance of the entire plan. Invalid commands never
  // contribute established completion/consumption to the reported snapshots.
  bool preconditionsEstablished = false;
};
struct RetirementRequirement {
  std::size_t operation = 0;
  Pipe observer = Pipe::S;
};
struct AnalysisDiagnostic {
  enum Kind {
    InvalidInput,
    UnsupportedSemantics,
    InvalidCommands
  } kind = InvalidInput;
  std::size_t operation = NoAnalysisId;
  std::string reason;
};
struct AnalysisOptions {
  // A reporting/memory choice, not a precision or correctness switch.
  bool captureStates = true;
};
struct AnalysisStats {
  uint64_t siteEvaluations = 0, merges = 0, work = 0;
  std::size_t phaseStateCount = 0, maxPhaseStatesPerSite = 0;
  std::size_t staticSites = 0, certificationPasses = 0, suppressedCommands = 0;
};
struct AnalysisResult {
  // complete means supported inputs were analyzed to convergence. It does NOT
  // mean synchronized: an empty plan normally has residual requirements.
  bool complete = false;
  std::string reason;
  std::vector<AnalysisDiagnostic> diagnostics;
  std::vector<AnalysisContext> contexts;
  std::vector<EventIdentity> keys; // index order for BoundaryFacts::events
  std::vector<CutFacts> cuts;      // [0,N) physical phases; N invocation exit
  std::vector<CommandFacts> commands;
  std::vector<CompletionRequirement> residuals;
  std::vector<ProtocolObligation> protocol;
  std::vector<RetirementRequirement> retirement;
  std::vector<PhaseResourceObligation> phaseResources;
  AnalysisStats stats;
  bool verified() const {
    return complete && diagnostics.empty() && residuals.empty() &&
           protocol.empty() && retirement.empty() && phaseResources.empty();
  }
};

// Read-only fixed-plan analysis. On incomplete protocols this is a conservative
// reference traversal, NOT a claim that the candidate can execute. Diagnostics
// may include abstraction-induced failures. No summaries/episodes are required.
AnalysisResult analyze(const Program &, const Commands &, AnalysisOptions = {});
// Analyze the unsynchronized program (empty command population).
AnalysisResult analyze(const Program &);
} // namespace mlir::pto::oahs
#endif
