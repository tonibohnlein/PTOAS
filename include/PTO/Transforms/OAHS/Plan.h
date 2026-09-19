// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_PLAN_H
#define PTO_TRANSFORMS_OAHS_PLAN_H

#include "PTO/IR/SyncTargetProfile.h"
#include "PTO/Transforms/OAHS/Observations.h"
#include "PTO/Transforms/OAHS/PhaseContracts.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::pto::oahs {

// Ported from StructuredSyncCore: pipelines are facts, never search variables.
using Pipe = ::mlir::pto::SyncPipe;
constexpr unsigned PipeCount = unsigned(Pipe::Count);
// Legacy programs: before operation i, with N the invocation exit. Observed
// programs: an original-control site; multiple sites may share one emitted
// word.
using Cut = std::size_t;
struct Access {
  unsigned cell = 0;
  bool read = false, write = false;
  // Reference succession ONLY: the frontend proves a definite full-cell write.
  // A strong update never clears acquired-completion or event state.
  bool definiteWrite = false;
};
enum class EffectTiming { Issue, Completion };
struct ResourceEffect {
  std::string resource;
  bool acquire = false, release = false;
  EffectTiming timing = EffectTiming::Completion;
};
struct VisibilityEffect {
  unsigned cell = 0;
  bool publish = false, acquire = false;
  EffectTiming timing = EffectTiming::Completion;
};
struct EventIdentity {
  Pipe source = Pipe::S, observer = Pipe::S;
  unsigned key = 0;
  bool privateIdentity = false;
};
struct Cell {
  // Stronger resource exclusion must not change the actual byte-access roles.
  bool exclusive = false;
  std::string addressSpace;
  std::string provenance;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  bool unknownRange = false;
  // A canonical interval is one atom in a qualified coordinate space. A
  // conservative overlap witness is NOT a physical atom and cannot justify
  // strong updates or produced-content identity.
  enum class Storage : unsigned { Abstract, CanonicalInterval, OverlapWitness };
  Storage storage = Storage::Abstract;
  std::string coordinateSpace;
  // Function-local translated root identities, retained even when geometry is
  // unknown or several SSA roots name one physical atom. Not disjointness facts.
  std::vector<std::size_t> storageOrigins;
  // Explicit target/lowering premise for all M accesses to this exact atom:
  // qualified ordinary large-shape MMADs permit a subsequent accumulating
  // consumer to rely on native ACC-local access ordering.
  // This does NOT complete an operation or clear any pending source history.
  bool nativeMmadAccOrder = false;
};
struct Operation {
  Pipe pipe = Pipe::S;
  std::vector<Access> accesses;
  std::vector<ResourceEffect> resources;
  std::vector<VisibilityEffect> visibility;
  std::vector<EventIdentity> authoredEvents;
  std::vector<EventIdentity> internalTransfers;
  std::size_t original = 0;
  unsigned phase = 0;
  // The lowering/importer must establish completeness explicitly. No opcode
  // name or absence of a typed effect can turn this default into true.
  bool complete = false;
  std::optional<FinalBlockOperation> finalBlock;
  // Only an accumulating consumer may use the qualified ACC-local rule.
  // A fresh matrix initialization is not a K-axis accumulation dependency.
  bool nativeMmadAccumulate = false;
};
struct Region {
  enum Kind { Sequence, Choice, For, While, Operation } kind = Sequence;
  std::vector<Region> children;
  std::size_t operation = 0;
  // A for may execute zero times. While has [before, after] children and always
  // executes before once before deciding whether to enter after.
  bool zeroTripPossible = false;
};
enum class Property { ByteCompletion, ResourceExclusion };
struct Demand {
  std::size_t producer = 0, consumer = 0;
  unsigned cell = 0;
  Property property = Property::ByteCompletion;
};
using Target = ::mlir::pto::SyncTargetProfile;

struct Program {
  std::vector<Cell> cells;
  std::vector<Operation> operations;
  Region body;
  // Optional qualified finite original-control quotient. It replaces, rather
  // than silently flattens, body. Effects still name original physical phases.
  std::optional<ObservedControl> observed;
  std::vector<EventIdentity> reservations;
  // Original two-slot FIFO use correspondence. It supplies byte identities,
  // not completion. Commands at both hidden slot phases share original words.
  struct AlternatingSlots {
    std::vector<unsigned> cells;
    std::vector<Cut> reads;
    Pipe reader = Pipe::S, writer = Pipe::S;
  };
  std::optional<AlternatingSlots> alternatingSlots;
  Target target;
  // No native frontend populates this until its lowering/service premises are
  // qualified. Model clients can explicitly supply the reference contract.
  std::optional<FinalBlockProfile> finalBlocks;
  // Essential analyses run to their finite fixed points. Resource limits are
  // target/ABI facts; no compiler-work allowance changes these obligations.
  struct InvocationContract {
    enum Retirement {
      NoRetirement,
      DrainAllAtReturn
    } retirement = NoRetirement;
    std::string alias;
    std::string visibility;
    std::string boundary;
  } invocation;
};
inline std::size_t commandCutCount(const Program &p) {
  return p.observed ? p.observed->sites.size() : p.operations.size() + 1;
}
inline Cut invocationExitCut(const Program &p) {
  return p.observed ? p.observed->exit : p.operations.size();
}
inline std::size_t operationAtCut(const Program &p, Cut c) {
  return p.observed ? p.observed->sites.at(c).operation
                    : (c < p.operations.size() ? c : NoControlId);
}
inline bool legalCommandCut(const Program &p, Cut c) {
  return c < commandCutCount(p) &&
         (!p.observed || p.observed->sites[c].observation != NoControlId);
}
inline Cut canonicalCommandCut(const Program &p, Cut c) {
  if (!p.observed || !legalCommandCut(p, c))
    return c;
  const auto observation = p.observed->sites[c].observation;
  for (Cut i = 0; i < c; ++i)
    if (p.observed->sites[i].observation == observation)
      return i;
  return c;
}
struct Command {
  enum Kind { Publish, Acquire, Barrier, BarrierAll } kind = BarrierAll;
  Pipe source = Pipe::S, observer = Pipe::S;
  unsigned key = 0;
};
using Commands = std::vector<std::vector<Command>>;
// Common status for structural validation, fixed-plan verification and native
// emission. Selected construction diagnostics live in SelectedPlan.
struct Result {
  bool success = false;
  std::string reason;
  std::vector<Demand> demands;
  Commands commands;
};

// Declaration/structural validation only; not residual analysis or a proof of
// synchronization. Use Analysis.h::analyze for the fixed-plan analysis service.
Result validateProgram(const Program &program);
// Read-only: reconstructs receipts and hazards from original operations and the
// actual command stream, never from Result::demands annotations.
Result verify(const Program &program, const Commands &commands);

} // namespace mlir::pto::oahs
#endif
