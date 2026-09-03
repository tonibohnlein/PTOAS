// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// This bounded oracle deliberately does not consume region summaries. It
// enumerates structured choices and bounded loop iterations, then executes the
// concrete mechanism program points. Ordinary worlds use zero through two
// iterations. ReadyRelease<2> worlds use zero through four so the first slot
// is observed after both its period and reuse distance. The bound is a
// differential
// testing aid, not a proof for arbitrary loop counts. Reaching the state cap
// marks the result inconclusive instead of rejecting an otherwise valid
// program; materialized verification remains the final correctness gate.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr std::size_t kMaxOracleStates = 512;
constexpr unsigned kDefaultOracleLoopUnroll = 2;
constexpr unsigned kReadyRelease2OracleLoopUnroll = 4;
// The structured oracle is a bounded differential test, not the coverage
// proof.  Avoid starting an enumeration whose straight-line demand checks
// already exceed this development-oracle budget; the mandatory regional and
// flattened analyses still have to agree, and the staged physical verifier
// remains authoritative for the selected plan.
constexpr std::size_t kMaxOracleStaticChecks = 32768;

std::size_t cappedAdd(std::size_t first, std::size_t second) {
  if (first > kMaxOracleStates || second > kMaxOracleStates - first) {
    return kMaxOracleStates + 1;
  }
  return first + second;
}

std::size_t cappedMultiply(std::size_t first, std::size_t second) {
  if (first == 0 || second == 0) {
    return 0;
  }
  if (first > kMaxOracleStates || second > kMaxOracleStates / first) {
    return kMaxOracleStates + 1;
  }
  return first * second;
}

std::size_t estimateBlockStates(Block &block, unsigned loopUnroll);

std::size_t estimateOperationStates(Operation *operation,
                                    unsigned loopUnroll) {
  if (auto choice = dyn_cast<scf::IfOp>(operation)) {
    const std::size_t thenStates =
        estimateBlockStates(choice.getThenRegion().front(), loopUnroll);
    const std::size_t elseStates =
        choice.getElseRegion().empty()
            ? 1
            : estimateBlockStates(choice.getElseRegion().front(), loopUnroll);
    return cappedAdd(thenStates, elseStates);
  }
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    const std::size_t bodyStates =
        estimateBlockStates(loop.getRegion().front(), loopUnroll);
    std::size_t total = 1;
    std::size_t atDepth = 1;
    for (unsigned count = 1; count <= loopUnroll; ++count) {
      atDepth = cappedMultiply(atDepth, bodyStates);
      total = cappedAdd(total, atDepth);
    }
    return total;
  }
  if (auto section = dyn_cast<SectionCubeOp>(operation)) {
    return estimateBlockStates(section.getBody().front(), loopUnroll);
  }
  if (auto section = dyn_cast<SectionVectorOp>(operation)) {
    return estimateBlockStates(section.getBody().front(), loopUnroll);
  }
  return 1;
}

std::size_t estimateBlockStates(Block &block, unsigned loopUnroll) {
  std::size_t states = 1;
  for (Operation &operation : block) {
    states = cappedMultiply(states,
                            estimateOperationStates(&operation, loopUnroll));
    if (states > kMaxOracleStates) {
      return states;
    }
  }
  return states;
}

struct LoopInstance {
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  unsigned iteration = 0;

  bool operator==(const LoopInstance &other) const {
    return loop == other.loop && iteration == other.iteration;
  }
};

struct PhaseInstance {
  CanonicalPhaseId phase = kInvalidCanonicalSyncId;
  SmallVector<LoopInstance, 2> loops;

  bool operator==(const PhaseInstance &other) const {
    return phase == other.phase && loops == other.loops;
  }
};

struct ResourceState {
  CanonicalPhysicalResource resource;
  SmallVector<PhaseInstance, 16> issued;
  SmallVector<PhaseInstance, 16> known;
};

struct EventPayload {
  enum class Kind : std::uint8_t {
    Ordinary,
    OwnershipReady,
    OwnershipRelease,
  };

  CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
  Kind kind = Kind::Ordinary;
  unsigned lane = 0;
  SmallVector<PhaseInstance, 16> phases;
};

struct OracleState {
  SmallVector<ResourceState, 8> resources;
  SmallVector<PhaseInstance, 16> globalKnown;
  SmallVector<EventPayload, 8> events;
  SmallVector<CanonicalControlAtom, 2> controlPath;
  SmallVector<LoopInstance, 2> loops;
  SmallVector<bool, 16> covered;
  bool valid = true;
};

struct OracleContext {
  bool exhaustive = true;
  const CanonicalSyncTarget *target = nullptr;
  CanonicalSyncStatistics *statistics = nullptr;
  SmallVector<SmallVector<CanonicalDemandId, 8>, 0> demandsByTarget;
  unsigned loopUnroll = kDefaultOracleLoopUnroll;
  SmallVector<CanonicalOwnershipChannelId, 2> ownershipChannels;
};

using PhaseIndex = DenseMap<Operation *, SmallVector<CanonicalPhaseId, 2>>;

ResourceState &getResource(OracleState &state,
                           CanonicalPhysicalResource resource) {
  auto found = llvm::find_if(state.resources, [&](const ResourceState &entry) {
    return entry.resource == resource;
  });
  if (found == state.resources.end()) {
    state.resources.push_back({resource, {}, {}});
    return state.resources.back();
  }
  return *found;
}

const ResourceState *getResource(const OracleState &state,
                                 CanonicalPhysicalResource resource) {
  auto found = llvm::find_if(state.resources, [&](const ResourceState &entry) {
    return entry.resource == resource;
  });
  return found == state.resources.end() ? nullptr : &*found;
}

void appendUnique(SmallVectorImpl<PhaseInstance> &destination,
                  ArrayRef<PhaseInstance> source) {
  for (const PhaseInstance &instance : source) {
    if (!llvm::is_contained(destination, instance)) {
      destination.push_back(instance);
    }
  }
}

bool guardEnabled(ArrayRef<CanonicalControlAtom> active,
                  ArrayRef<CanonicalControlAtom> required) {
  return llvm::all_of(required, [active](const CanonicalControlAtom &atom) {
    return llvm::is_contained(active, atom);
  });
}

std::optional<unsigned> loopIteration(const PhaseInstance &instance,
                                      CanonicalRegionId loop) {
  auto found = llvm::find_if(instance.loops, [loop](const LoopInstance &item) {
    return item.loop == loop;
  });
  return found == instance.loops.end()
             ? std::nullopt
             : std::optional<unsigned>(found->iteration);
}

bool matchesDistance(const CanonicalDemand &demand, const PhaseInstance &source,
                     const PhaseInstance &target) {
  for (const CanonicalLoopDistance &distance : demand.iterationDistance) {
    const std::optional<unsigned> sourceIteration =
        loopIteration(source, distance.loop);
    const std::optional<unsigned> targetIteration =
        loopIteration(target, distance.loop);
    if (!sourceIteration || !targetIteration) {
      return false;
    }
    const bool wrongSame =
        distance.relation == CanonicalIterationRelation::Same &&
        *sourceIteration != *targetIteration;
    const bool wrongPositive =
        distance.relation == CanonicalIterationRelation::AnyPositive &&
        *sourceIteration >= *targetIteration;
    if (wrongSame || wrongPositive) {
      return false;
    }
  }
  return true;
}

bool ownershipInstancesMayConflict(
    const CanonicalSyncProgram &program, const CanonicalDemand &demand,
    const PhaseInstance &source, const PhaseInstance &target,
    const OracleContext &context) {
  for (CanonicalOwnershipChannelId id : context.ownershipChannels) {
    const CanonicalOwnershipChannel &channel =
        program.getOwnershipChannels()[id];
    if (!canonicalOwnershipChannelCoversDemand(program, channel, demand)) {
      continue;
    }
    const CanonicalStorageGeneration &generation =
        program.getStorageGeneration(channel.generation);
    const std::optional<unsigned> sourceIteration =
        loopIteration(source, generation.loop);
    const std::optional<unsigned> targetIteration =
        loopIteration(target, generation.loop);
    if (!sourceIteration || !targetIteration) {
      return true;
    }
    const unsigned sourceLane =
        (*sourceIteration + generation.slotOffset) % generation.staticDepth;
    const unsigned targetLane =
        (*targetIteration + generation.slotOffset) % generation.staticDepth;
    return sourceLane == targetLane;
  }
  return true;
}

bool completionKnown(const CanonicalSyncProgram &program,
                     const PhaseInstance &source,
                     CanonicalPhysicalResource destination,
                     const CanonicalSyncTarget &target,
                     const OracleState &state) {
  const CanonicalPhase &sourcePhase = program.getPhase(source.phase);
  const bool intrinsicCompletion =
      getVPTOSchedulingSemantics(sourcePhase.operation)
          .completionIsSynchronous ||
      (sourcePhase.resource == destination &&
       target.hasIntrinsicCompletion(sourcePhase.resource));
  if (intrinsicCompletion) {
    return true;
  }
  const ResourceState *resource = getResource(state, destination);
  return llvm::is_contained(state.globalKnown, source) ||
         (resource && llvm::is_contained(resource->known, source));
}

bool demandUsesDirectMechanism(const CanonicalSyncProgram &program,
                               const CanonicalDemand &demand,
                               ArrayRef<CanonicalMechanismId> selected) {
  const CanonicalMechanismId direct = program.getDirectMechanisms()[demand.id];
  return llvm::is_contained(selected, direct);
}

bool recurringReleaseCovers(const CanonicalSyncProgram &program,
                            const CanonicalDemand &demand,
                            ArrayRef<CanonicalMechanismId> selected) {
  auto carrying = llvm::find_if(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::AnyPositive;
      });
  if (carrying == demand.iterationDistance.end()) {
    return false;
  }
  const CanonicalPhysicalResource source =
      program.getPhase(demand.source).resource;
  const CanonicalPhysicalResource target =
      program.getPhase(demand.target).resource;
  return llvm::any_of(selected, [&](CanonicalMechanismId id) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    return mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
           !mechanism.boundaryRecurring &&
           mechanism.recurrenceLoop == carrying->loop &&
           mechanism.target == source && mechanism.source == target;
  });
}

void checkTargetDemands(const CanonicalSyncProgram &program,
                        const PhaseInstance &target,
                        ArrayRef<CanonicalMechanismId> selected,
                        const OracleContext &context, OracleState &state) {
  for (CanonicalDemandId demandId : context.demandsByTarget[target.phase]) {
    if (context.statistics) {
      ++context.statistics->coverageOracleDemandTests;
    }
    const CanonicalDemand &demand = program.getDemand(demandId);
    if (!state.covered[demand.id] || demand.target != target.phase ||
        demand.kind == CanonicalDemandKind::ExitCompletion) {
      continue;
    }
    if (demand.requirement == CanonicalRequirement::Visibility) {
      state.covered[demand.id] =
          demandUsesDirectMechanism(program, demand, selected);
      continue;
    }
    if (recurringReleaseCovers(program, demand, selected)) {
      continue;
    }
    const CanonicalMechanismId direct =
        program.getDirectMechanisms()[demand.id];
    const bool recurringSelected = llvm::is_contained(selected, direct) &&
                                   program.getMechanism(direct).kind ==
                                       CanonicalMechanismKind::RecurringEvent;
    if (recurringSelected) {
      continue;
    }
    for (const ResourceState &resource : state.resources) {
      for (const PhaseInstance &source : resource.issued) {
        if (context.statistics) {
          ++context.statistics->coverageOracleSourceInstanceTests;
        }
        if (source.phase != demand.source ||
            !matchesDistance(demand, source, target) ||
            !ownershipInstancesMayConflict(program, demand, source, target,
                                           context)) {
          continue;
        }
        const CanonicalPhysicalResource destination =
            program.getPhase(target.phase).resource;
        if (!completionKnown(program, source, destination, *context.target,
                             state)) {
          state.covered[demand.id] = false;
          break;
        }
      }
      if (!state.covered[demand.id]) {
        break;
      }
    }
  }
}

void applyBarrier(const CanonicalMechanism &mechanism, OracleState &state) {
  const ResourceState *source =
      getResource(static_cast<const OracleState &>(state), mechanism.source);
  if (!source) {
    return;
  }
  const SmallVector<PhaseInstance, 16> completed = source->issued;
  appendUnique(getResource(state, mechanism.target).known, completed);
}

void applyFixedFence(const CanonicalSyncProgram &program,
                     const CanonicalMechanism &mechanism, OracleState &state) {
  const CanonicalFenceEffect &effect =
      program.getFenceEffect(*mechanism.fenceEffect);
  SmallVector<PhaseInstance, 16> completed;
  for (const ResourceState &resource : state.resources) {
    if (!llvm::is_contained(effect.drainedResources, resource.resource)) {
      continue;
    }
    appendUnique(completed, resource.issued);
    appendUnique(completed, resource.known);
  }
  appendUnique(state.globalKnown, completed);
}

void applyEventSet(const CanonicalMechanism &mechanism, OracleState &state) {
  EventPayload payload;
  payload.mechanism = mechanism.id;
  const ResourceState *source =
      getResource(static_cast<const OracleState &>(state), mechanism.source);
  if (source) {
    appendUnique(payload.phases, source->issued);
    appendUnique(payload.phases, source->known);
  }
  state.events.push_back(std::move(payload));
}

void applyEventWait(const CanonicalMechanism &mechanism, OracleState &state) {
  auto found = llvm::find_if(state.events, [&](const EventPayload &payload) {
    return payload.mechanism == mechanism.id &&
           payload.kind == EventPayload::Kind::Ordinary;
  });
  if (found == state.events.end()) {
    return;
  }
  appendUnique(getResource(state, mechanism.target).known, found->phases);
  state.events.erase(found);
}

void applyOwnershipSet(const CanonicalMechanism &mechanism,
                       EventPayload::Kind kind, unsigned lane,
                       CanonicalPhysicalResource source,
                       OracleState &state) {
  const bool duplicate = llvm::any_of(state.events, [&](const EventPayload &e) {
    return e.mechanism == mechanism.id && e.kind == kind && e.lane == lane;
  });
  if (duplicate) {
    state.valid = false;
    return;
  }
  EventPayload payload;
  payload.mechanism = mechanism.id;
  payload.kind = kind;
  payload.lane = lane;
  const ResourceState *resource =
      getResource(static_cast<const OracleState &>(state), source);
  if (resource) {
    appendUnique(payload.phases, resource->issued);
    appendUnique(payload.phases, resource->known);
  }
  state.events.push_back(std::move(payload));
}

void applyOwnershipWait(const CanonicalMechanism &mechanism,
                        EventPayload::Kind kind, unsigned lane,
                        CanonicalPhysicalResource target, OracleState &state) {
  auto found = llvm::find_if(state.events, [&](const EventPayload &payload) {
    return payload.mechanism == mechanism.id && payload.kind == kind &&
           payload.lane == lane;
  });
  if (found == state.events.end()) {
    state.valid = false;
    return;
  }
  appendUnique(getResource(state, target).known, found->phases);
  state.events.erase(found);
}

std::optional<unsigned>
getOwnershipLane(const CanonicalStorageGeneration &generation,
                 const OracleState &state) {
  auto found = llvm::find_if(state.loops, [&](const LoopInstance &instance) {
    return instance.loop == generation.loop;
  });
  if (found == state.loops.end()) {
    return std::nullopt;
  }
  return (found->iteration + generation.slotOffset) % generation.staticDepth;
}

void executeOwnershipPoint(const CanonicalSyncProgram &program,
                           const CanonicalMechanism &mechanism,
                           CanonicalProgramPoint point, OracleState &state) {
  const CanonicalOwnershipChannel &channel =
      program.getOwnershipChannels()[*mechanism.ownershipChannel];
  const CanonicalStorageGeneration &generation =
      program.getStorageGeneration(channel.generation);
  Operation *loop = program.getRegion(generation.loop).operation;
  if (point.operation == loop &&
      point.position == CanonicalProgramPointPosition::Before) {
    for (unsigned lane = 0; lane < generation.staticDepth; ++lane) {
      applyOwnershipSet(mechanism, EventPayload::Kind::OwnershipRelease, lane,
                        generation.consumer, state);
    }
    return;
  }
  if (point.operation == loop &&
      point.position == CanonicalProgramPointPosition::After) {
    for (unsigned lane = 0; lane < generation.staticDepth; ++lane) {
      applyOwnershipWait(mechanism, EventPayload::Kind::OwnershipRelease, lane,
                         generation.producer, state);
    }
    return;
  }
  const std::optional<unsigned> lane = getOwnershipLane(generation, state);
  if (!lane) {
    return;
  }
  if (point == generation.writeAcquire) {
    applyOwnershipWait(mechanism, EventPayload::Kind::OwnershipRelease, *lane,
                       generation.producer, state);
  }
  if (point == generation.ready) {
    applyOwnershipSet(mechanism, EventPayload::Kind::OwnershipReady, *lane,
                      generation.producer, state);
  }
  if (point == generation.readAcquire) {
    applyOwnershipWait(mechanism, EventPayload::Kind::OwnershipReady, *lane,
                       generation.consumer, state);
  }
  if (point == generation.lastUse) {
    applyOwnershipSet(mechanism, EventPayload::Kind::OwnershipRelease, *lane,
                      generation.consumer, state);
  }
}

void executePoint(const CanonicalSyncProgram &program,
                  CanonicalProgramPoint point,
                  ArrayRef<CanonicalMechanismId> selected,
                  const OracleContext &context, OracleState &state) {
  for (CanonicalMechanismId id : selected) {
    if (context.statistics) {
      ++context.statistics->coverageOracleMechanismTests;
    }
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (!guardEnabled(state.controlPath, mechanism.guard)) {
      continue;
    }
    if (mechanism.kind == CanonicalMechanismKind::ReadyRelease2) {
      executeOwnershipPoint(program, mechanism, point, state);
      continue;
    }
    // A boundary handshake transfers completion from the loop latch to the
    // next loop header.  Until summary transfers carry iteration transforms,
    // only its explicitly certified recurrence origins may use that effect.
    if (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
        mechanism.boundaryRecurring) {
      continue;
    }
    if (mechanism.targetPoint == point &&
        mechanism.kind == CanonicalMechanismKind::PipeBarrier) {
      applyBarrier(mechanism, state);
    }
    if (mechanism.targetPoint == point &&
        mechanism.kind == CanonicalMechanismKind::FixedFence) {
      applyFixedFence(program, mechanism, state);
    }
    if (mechanism.sourcePoint == point &&
        (mechanism.kind == CanonicalMechanismKind::Event ||
         mechanism.kind == CanonicalMechanismKind::CrossCoreEvent ||
         mechanism.kind == CanonicalMechanismKind::RecurringEvent)) {
      applyEventSet(mechanism, state);
    }
    if (mechanism.targetPoint == point &&
        (mechanism.kind == CanonicalMechanismKind::Event ||
         mechanism.kind == CanonicalMechanismKind::CrossCoreEvent ||
         mechanism.kind == CanonicalMechanismKind::RecurringEvent)) {
      applyEventWait(mechanism, state);
    }
  }
}

CanonicalRegionId findStructuredRegion(const CanonicalSyncProgram &program,
                                       Operation *operation,
                                       CanonicalRegionKind kind) {
  auto found =
      llvm::find_if(program.getRegions(), [&](const CanonicalRegion &region) {
        return region.operation == operation && region.kind == kind;
      });
  return found == program.getRegions().end() ? kInvalidCanonicalSyncId
                                             : found->id;
}

void executePhases(const CanonicalSyncProgram &program, Operation *operation,
                   const PhaseIndex &phases,
                   ArrayRef<CanonicalMechanismId> selected,
                   const OracleContext &context, OracleState &state) {
  auto found = phases.find(operation);
  if (found == phases.end()) {
    return;
  }
  for (CanonicalPhaseId phase : found->second) {
    PhaseInstance instance{phase, state.loops};
    checkTargetDemands(program, instance, selected, context, state);
    ResourceState &resource =
        getResource(state, program.getPhase(phase).resource);
    if (!llvm::is_contained(resource.issued, instance)) {
      resource.issued.push_back(std::move(instance));
    }
  }
}

FailureOr<SmallVector<OracleState, 8>>
executeBlock(const CanonicalSyncProgram &program, Block &block,
             const PhaseIndex &phases, ArrayRef<CanonicalMechanismId> selected,
             ArrayRef<OracleState> inputs, OracleContext &context);

FailureOr<SmallVector<OracleState, 8>>
executeChoice(const CanonicalSyncProgram &program, scf::IfOp choice,
              const PhaseIndex &phases, ArrayRef<CanonicalMechanismId> selected,
              ArrayRef<OracleState> inputs, OracleContext &context) {
  SmallVector<OracleState, 8> result;
  const CanonicalRegionId choiceId =
      findStructuredRegion(program, choice, CanonicalRegionKind::Choice);
  for (unsigned arm = 0; arm < 2U; ++arm) {
    SmallVector<OracleState, 8> armInputs(inputs.begin(), inputs.end());
    for (OracleState &state : armInputs) {
      state.controlPath.push_back({choiceId, arm});
    }
    SmallVector<OracleState, 8> armOutputs;
    if (arm == 0) {
      FailureOr<SmallVector<OracleState, 8>> executed =
          executeBlock(program, choice.getThenRegion().front(), phases,
                       selected, armInputs, context);
      if (failed(executed)) {
        return failure();
      }
      armOutputs = std::move(*executed);
    } else if (choice.getElseRegion().empty()) {
      armOutputs = std::move(armInputs);
    } else {
      FailureOr<SmallVector<OracleState, 8>> executed =
          executeBlock(program, choice.getElseRegion().front(), phases,
                       selected, armInputs, context);
      if (failed(executed)) {
        return failure();
      }
      armOutputs = std::move(*executed);
    }
    for (OracleState &state : armOutputs) {
      state.controlPath.pop_back();
      result.push_back(std::move(state));
    }
  }
  return result;
}

FailureOr<SmallVector<OracleState, 8>>
executeLoop(const CanonicalSyncProgram &program, scf::ForOp loop,
            const PhaseIndex &phases, ArrayRef<CanonicalMechanismId> selected,
            ArrayRef<OracleState> inputs, OracleContext &context) {
  SmallVector<OracleState, 8> result;
  const CanonicalRegionId loopId =
      findStructuredRegion(program, loop, CanonicalRegionKind::Loop);
  for (unsigned count = 0; count <= context.loopUnroll; ++count) {
    SmallVector<OracleState, 8> current(inputs.begin(), inputs.end());
    for (unsigned iteration = 0; iteration < count; ++iteration) {
      for (OracleState &state : current) {
        state.loops.push_back({loopId, iteration});
      }
      FailureOr<SmallVector<OracleState, 8>> body =
          executeBlock(program, loop.getRegion().front(), phases, selected,
                       current, context);
      if (failed(body)) {
        return failure();
      }
      current = std::move(*body);
      for (OracleState &state : current) {
        state.loops.pop_back();
      }
    }
    result.append(std::make_move_iterator(current.begin()),
                  std::make_move_iterator(current.end()));
    const bool stateCapExceeded = result.size() > kMaxOracleStates;
    if (stateCapExceeded) {
      context.exhaustive = false;
      result.resize(kMaxOracleStates);
    }
  }
  return result;
}

FailureOr<SmallVector<OracleState, 8>>
executeOperation(const CanonicalSyncProgram &program, Operation *operation,
                 const PhaseIndex &phases,
                 ArrayRef<CanonicalMechanismId> selected,
                 ArrayRef<OracleState> inputs, OracleContext &context) {
  SmallVector<OracleState, 8> states(inputs.begin(), inputs.end());
  if (context.statistics) {
    context.statistics->coverageOracleStateOperations += states.size();
  }
  for (OracleState &state : states) {
    executePoint(program, {operation, CanonicalProgramPointPosition::Before},
                 selected, context, state);
  }
  if (auto choice = dyn_cast<scf::IfOp>(operation)) {
    FailureOr<SmallVector<OracleState, 8>> result =
        executeChoice(program, choice, phases, selected, states, context);
    if (failed(result)) {
      return failure();
    }
    states = std::move(*result);
  } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    FailureOr<SmallVector<OracleState, 8>> result =
        executeLoop(program, loop, phases, selected, states, context);
    if (failed(result)) {
      return failure();
    }
    states = std::move(*result);
  } else if (auto section = dyn_cast<SectionCubeOp>(operation)) {
    FailureOr<SmallVector<OracleState, 8>> result = executeBlock(
        program, section.getBody().front(), phases, selected, states, context);
    if (failed(result)) {
      return failure();
    }
    states = std::move(*result);
  } else if (auto section = dyn_cast<SectionVectorOp>(operation)) {
    FailureOr<SmallVector<OracleState, 8>> result = executeBlock(
        program, section.getBody().front(), phases, selected, states, context);
    if (failed(result)) {
      return failure();
    }
    states = std::move(*result);
  } else {
    for (OracleState &state : states) {
      executePhases(program, operation, phases, selected, context, state);
    }
  }
  for (OracleState &state : states) {
    executePoint(program, {operation, CanonicalProgramPointPosition::After},
                 selected, context, state);
  }
  return states;
}

FailureOr<SmallVector<OracleState, 8>>
executeBlock(const CanonicalSyncProgram &program, Block &block,
             const PhaseIndex &phases, ArrayRef<CanonicalMechanismId> selected,
             ArrayRef<OracleState> inputs, OracleContext &context) {
  SmallVector<OracleState, 8> states(inputs.begin(), inputs.end());
  for (Operation &operation : block) {
    FailureOr<SmallVector<OracleState, 8>> result = executeOperation(
        program, &operation, phases, selected, states, context);
    if (failed(result)) {
      return failure();
    }
    states = std::move(*result);
    const bool stateCapExceeded = states.size() > kMaxOracleStates;
    if (stateCapExceeded) {
      context.exhaustive = false;
      states.resize(kMaxOracleStates);
    }
  }
  return states;
}

} // namespace

FailureOr<CanonicalUnrolledCoverageResult>
mlir::pto::canonical_sync_detail::evaluateCanonicalSyncUnrolledOracle(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalMechanismId> selected) {
  CanonicalSyncStatistics *statistics = program.getStatistics();
  if (statistics) {
    ++statistics->coverageOracleWorlds;
  }
  const std::size_t phaseCount =
      std::max<std::size_t>(program.getPhases().size(), 1);
  const bool hasReadyRelease2 = llvm::any_of(
      selected, [&](CanonicalMechanismId id) {
        return program.getMechanism(id).kind ==
               CanonicalMechanismKind::ReadyRelease2;
      });
  const unsigned loopUnroll = hasReadyRelease2
                                  ? kReadyRelease2OracleLoopUnroll
                                  : kDefaultOracleLoopUnroll;
  const bool staticBudgetExceeded =
      program.getDemands().size() > kMaxOracleStaticChecks / phaseCount;
  const bool stateBudgetExceeded =
      estimateBlockStates(program.getFunction().getBody().front(),
                          loopUnroll) >
      kMaxOracleStates;
  if (staticBudgetExceeded || stateBudgetExceeded) {
    if (statistics) {
      ++statistics->coverageOracleSkippedWorlds;
    }
    CanonicalUnrolledCoverageResult result;
    result.exhaustive = false;
    return result;
  }
  PhaseIndex phases;
  for (const CanonicalPhase &phase : program.getPhases()) {
    phases[phase.operation].push_back(phase.id);
  }
  OracleState initial;
  initial.covered.resize(program.getDemands().size(), true);
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demand.kind == CanonicalDemandKind::ExitCompletion) {
      initial.covered[demand.id] =
          demandUsesDirectMechanism(program, demand, selected);
    }
  }
  OracleContext context;
  context.statistics = statistics;
  context.loopUnroll = loopUnroll;
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }
  context.target = &*target;
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind == CanonicalMechanismKind::ReadyRelease2 &&
        mechanism.ownershipChannel) {
      context.ownershipChannels.push_back(*mechanism.ownershipChannel);
    }
  }
  context.demandsByTarget.resize(program.getPhases().size());
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demand.target < context.demandsByTarget.size()) {
      context.demandsByTarget[demand.target].push_back(demand.id);
    }
  }
  FailureOr<SmallVector<OracleState, 8>> states =
      executeBlock(program, program.getFunction().getBody().front(), phases,
                   selected, ArrayRef<OracleState>(&initial, 1), context);
  if (failed(states)) {
    return failure();
  }
  CanonicalUnrolledCoverageResult result;
  result.exhaustive = context.exhaustive;
  for (const CanonicalDemand &demand : program.getDemands()) {
    const bool coveredOnEveryPath =
        llvm::all_of(*states, [&](const OracleState &state) {
          return state.valid && state.covered[demand.id];
        });
    if (coveredOnEveryPath) {
      result.covered.push_back(demand.id);
    }
  }
  return result;
}
