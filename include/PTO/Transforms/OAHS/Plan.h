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
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlir::pto::oahs {

// Ported from StructuredSyncCore: pipelines are facts, never search variables.
using Pipe = ::mlir::pto::SyncPipe;
constexpr unsigned PipeCount = unsigned(Pipe::Count);
using Cut = std::size_t; // Before operation i; size() is the lifetime exit.
struct Access {
  unsigned cell = 0;
  bool read = false, write = false;
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
  std::vector<EventIdentity> reservations;
  Target target;
  // A bounded analysis may require a target-supported conservative result.
  // The reason is retained as part of the imported contract and this still
  // flows through the same constructor and reconstructed verifier.
  uint64_t verificationWorkLimit = 1u << 26;
  uint64_t constructionStepLimit = 1u << 16;
  bool conservativeCompletion = false;
  std::string conservativeReason;
  struct InvocationContract {
    enum Retirement { NoRetirement, DrainAllAtReturn } retirement = NoRetirement;
    std::string alias;
    std::string visibility;
    std::string boundary;
  } invocation;
};
struct Handoff {
  Pipe source = Pipe::S, observer = Pipe::S;
  Cut publication = 0, acquisition = 0;
};
struct Command {
  enum Kind { Publish, Acquire, Barrier, BarrierAll } kind = BarrierAll;
  Pipe source = Pipe::S, observer = Pipe::S;
  unsigned key = 0;
};
using Commands = std::vector<std::vector<Command>>;
struct Result {
  bool success = false;
  std::string reason;
  std::vector<Demand> demands;
  std::vector<Handoff> handoffs;
  Commands commands;
  unsigned scarcityBarriers = 0;
  unsigned conservativeBarriers = 0;
  unsigned protocolRepairs = 0;
};

// Validates semantic completeness and structural integrity without constructing
// or emitting synchronization. This is the analysis-coverage boundary.
Result analyze(const Program &program);
// One constructor entry for all represented programs. Unsupported synthesis is
// diagnosed after complete analysis import; it is never flattened or delegated
// to the legacy constructor.
Result construct(const Program &program);
// Read-only: reconstructs receipts and hazards from original operations and the
// actual command stream, never from Result::demands/handoffs annotations.
Result verify(const Program &program, const Commands &commands);

} // namespace mlir::pto::oahs
#endif
