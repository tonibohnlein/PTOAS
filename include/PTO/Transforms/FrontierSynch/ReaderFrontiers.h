// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License. D3 original reader vocabulary ported from OAHS StorageFrontiers.h. Expression IDs name
// structural facts; they grant no causal completion.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_READERFRONTIERS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_READERFRONTIERS_H
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
namespace mlir::pto::frontiersynch {
struct ObservationAtom {
  enum Kind : unsigned {
    OriginalBoolean,
    LoopNonEmpty,
    LoopResidue,
    LoopHasPrevious,
    LoopHasNext
  } kind = OriginalBoolean;
  std::size_t owner = NoControlId;
  uint64_t parameter = 0, value = 0;
};
struct ParticipationExpression {
  enum Kind { False, True, Atom, And, Or, Not, Invalid } kind = Invalid;
  std::size_t left = NoControlId, right = NoControlId;
  ObservationAtom atom;
};
struct GuardedReadFrontier {
  enum Kind { Empty, Access, Union, Guard, Invalid } kind = Invalid;
  std::size_t left = NoControlId, right = NoControlId;
  std::size_t predicate = 1, operation = NoControlId;
};
struct ReaderIntervalQuery {
  std::size_t owner = NoControlId;
  std::size_t cell = 0;
  PipelineType reader = PipelineType::PIPE_UNASSIGNED;
  // WholeRegion includes this owner's recurrence. BodyInterval describes one
  // original visit and indexes the children of its body sequence [begin,end).
  enum Scope { WholeRegion, BodyInterval } scope = WholeRegion;
  std::size_t begin = 0, end = NoControlId;
};
struct OriginalParticipationDemand {
  ReaderIntervalQuery interval;
  ObservationAtom predicate;
};
struct OriginalReaderFrontiers {
  enum class Status { Unknown, NoHit, Exact } status = Status::Unknown;
  ReaderIntervalQuery interval;
  // Exact whole counted-owner frontiers require initial/final visit vocabulary.
  bool separatesVisits = false;
  // Publication/receipt gaps require their own availability check later.
  bool guardsAvailableAtReadSites = false;
  std::size_t nonempty = 0, first = 0, last = 0;
  std::string reason;
  bool complete() const { return status != Status::Unknown; }
};
// A read-only structural segment, not a fresh-generation/strong-update proof.
struct OriginalReadSegment {
  bool complete = false;
  ReaderIntervalQuery interval;
  std::size_t before = NoControlId, after = NoControlId;
  bool startsAtOwnerEntry = true, endsAtOwnerExit = true;
  std::string reason;
};
} // namespace mlir::pto::frontiersynch
#endif
