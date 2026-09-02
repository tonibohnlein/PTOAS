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
// enumerates structured choices and enough loop iterations to witness each
// selected periodic ownership protocol, then executes the concrete mechanism
// program points and token transitions. The bound is a differential testing
// aid, not a proof for arbitrary loop counts. Reaching the state cap marks the
// result inconclusive instead of rejecting an otherwise valid program;
// materialized verification remains the final correctness gate.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr std::size_t kMaxOracleStates = 512;
constexpr unsigned kDefaultOracleLoopUnroll = 2;
constexpr unsigned kMaximumOracleLoopUnroll = 8;
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

CanonicalRegionId findStructuredRegion(const CanonicalSyncProgram &program,
                                       Operation *operation,
                                       CanonicalRegionKind kind);

std::size_t estimateBlockStates(
    const CanonicalSyncProgram &program, Block &block,
    ArrayRef<unsigned> loopUnrollByRegion);

std::size_t estimateOperationStates(
    const CanonicalSyncProgram &program, Operation *operation,
    ArrayRef<unsigned> loopUnrollByRegion) {
  if (auto choice = dyn_cast<scf::IfOp>(operation)) {
    const std::size_t thenStates = estimateBlockStates(
        program, choice.getThenRegion().front(), loopUnrollByRegion);
    const std::size_t elseStates =
        choice.getElseRegion().empty()
            ? 1
            : estimateBlockStates(program, choice.getElseRegion().front(),
                                  loopUnrollByRegion);
    return cappedAdd(thenStates, elseStates);
  }
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    const std::size_t bodyStates = estimateBlockStates(
        program, loop.getRegion().front(), loopUnrollByRegion);
    const CanonicalRegionId loopId = findStructuredRegion(
        program, loop, CanonicalRegionKind::Loop);
    const unsigned unroll =
        loopId < loopUnrollByRegion.size()
            ? loopUnrollByRegion[loopId]
            : kDefaultOracleLoopUnroll;
    std::size_t states = 1;
    std::size_t iterationStates = 1;
    for (unsigned count = 1; count <= unroll; ++count) {
      iterationStates = cappedMultiply(iterationStates, bodyStates);
      states = cappedAdd(states, iterationStates);
    }
    return states;
  }
  if (auto section = dyn_cast<SectionCubeOp>(operation)) {
    return estimateBlockStates(program, section.getBody().front(),
                               loopUnrollByRegion);
  }
  if (auto section = dyn_cast<SectionVectorOp>(operation)) {
    return estimateBlockStates(program, section.getBody().front(),
                               loopUnrollByRegion);
  }
  return 1;
}

std::size_t estimateBlockStates(
    const CanonicalSyncProgram &program, Block &block,
    ArrayRef<unsigned> loopUnrollByRegion) {
  std::size_t states = 1;
  for (Operation &operation : block) {
    states = cappedMultiply(
        states, estimateOperationStates(program, &operation,
                                        loopUnrollByRegion));
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
    Direct,
    OwnershipReady,
    OwnershipRelease,
  };

  CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
  Kind kind = Kind::Direct;
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
  SmallVector<SmallVector<CanonicalDemandId, 8>, 0> demandsByTarget;
  SmallVector<unsigned, 0> loopUnrollByRegion;
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

std::optional<std::uint64_t> evaluateIndexValue(
    Value value, const CanonicalSyncProgram &program, const OracleState &state,
    unsigned depth = 0U) {
  constexpr unsigned kMaximumExpressionDepth = 8U;
  if (!value || depth > kMaximumExpressionDepth) {
    return std::nullopt;
  }
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) {
    if (constant.isNegative() || constant.getActiveBits() > 64U) {
      return std::nullopt;
    }
    return constant.getZExtValue();
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast_or_null<scf::ForOp>(
        argument.getOwner()->getParentOp());
    if (!loop || argument != loop.getInductionVar()) {
      return std::nullopt;
    }
    const CanonicalRegionId loopId = findStructuredRegion(
        program, loop, CanonicalRegionKind::Loop);
    auto active = llvm::find_if(
        state.loops, [loopId](const LoopInstance &instance) {
          return instance.loop == loopId;
        });
    const std::optional<std::uint64_t> lower =
        evaluateIndexValue(loop.getLowerBound(), program, state, depth + 1U);
    const std::optional<std::uint64_t> step =
        evaluateIndexValue(loop.getStep(), program, state, depth + 1U);
    if (active == state.loops.end() || !lower || !step || *step == 0U ||
        active->iteration >
            (std::numeric_limits<std::uint64_t>::max() - *lower) / *step) {
      return std::nullopt;
    }
    return *lower + active->iteration * *step;
  }
  if (auto remainder = value.getDefiningOp<arith::RemSIOp>()) {
    const std::optional<std::uint64_t> lhs = evaluateIndexValue(
        remainder.getLhs(), program, state, depth + 1U);
    const std::optional<std::uint64_t> rhs = evaluateIndexValue(
        remainder.getRhs(), program, state, depth + 1U);
    return lhs && rhs && *rhs != 0U
               ? std::optional<std::uint64_t>(*lhs % *rhs)
               : std::nullopt;
  }
  if (auto remainder = value.getDefiningOp<arith::RemUIOp>()) {
    const std::optional<std::uint64_t> lhs = evaluateIndexValue(
        remainder.getLhs(), program, state, depth + 1U);
    const std::optional<std::uint64_t> rhs = evaluateIndexValue(
        remainder.getRhs(), program, state, depth + 1U);
    return lhs && rhs && *rhs != 0U
               ? std::optional<std::uint64_t>(*lhs % *rhs)
               : std::nullopt;
  }
  return std::nullopt;
}

std::optional<unsigned> evaluateChoiceArm(
    const CanonicalSyncProgram &program, scf::IfOp choice,
    const OracleState &state) {
  auto compare = choice.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!compare || (compare.getPredicate() != arith::CmpIPredicate::eq &&
                   compare.getPredicate() != arith::CmpIPredicate::ne)) {
    return std::nullopt;
  }
  const std::optional<std::uint64_t> lhs =
      evaluateIndexValue(compare.getLhs(), program, state);
  const std::optional<std::uint64_t> rhs =
      evaluateIndexValue(compare.getRhs(), program, state);
  if (!lhs || !rhs) {
    return std::nullopt;
  }
  const bool equal = *lhs == *rhs;
  const bool condition = compare.getPredicate() == arith::CmpIPredicate::eq
                             ? equal
                             : !equal;
  return condition ? 0U : 1U;
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
        if (source.phase != demand.source ||
            !matchesDistance(demand, source, target)) {
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

bool sameEvent(const EventPayload &payload, CanonicalMechanismId mechanism,
               EventPayload::Kind kind, unsigned lane) {
  return payload.mechanism == mechanism && payload.kind == kind &&
         payload.lane == lane;
}

void applyEventSet(const CanonicalMechanism &mechanism,
                   EventPayload::Kind kind, unsigned lane,
                   CanonicalPhysicalResource sourceResource,
                   OracleState &state) {
  if (llvm::any_of(state.events, [&](const EventPayload &payload) {
        return sameEvent(payload, mechanism.id, kind, lane);
      })) {
    state.valid = false;
    return;
  }
  EventPayload payload;
  payload.mechanism = mechanism.id;
  payload.kind = kind;
  payload.lane = lane;
  const ResourceState *source =
      getResource(static_cast<const OracleState &>(state), sourceResource);
  if (source) {
    appendUnique(payload.phases, source->issued);
    appendUnique(payload.phases, source->known);
  }
  state.events.push_back(std::move(payload));
}

void applyEventWait(const CanonicalMechanism &mechanism,
                    EventPayload::Kind kind, unsigned lane,
                    CanonicalPhysicalResource targetResource,
                    OracleState &state) {
  auto found = llvm::find_if(state.events, [&](const EventPayload &payload) {
    return sameEvent(payload, mechanism.id, kind, lane);
  });
  if (found == state.events.end()) {
    state.valid = false;
    return;
  }
  appendUnique(getResource(state, targetResource).known, found->phases);
  state.events.erase(found);
}

bool loopIsActive(const OracleState &state, CanonicalRegionId loop) {
  return llvm::any_of(state.loops, [loop](const LoopInstance &instance) {
    return instance.loop == loop;
  });
}

void applyOwnershipPoint(const CanonicalSyncProgram &program,
                         const CanonicalMechanism &mechanism,
                         const CanonicalOwnershipProtocol &protocol,
                         CanonicalProgramPoint point, OracleState &state) {
  Operation *loopOperation =
      program.getRegion(protocol.recurrenceLoop).operation;
  const CanonicalProgramPoint loopBefore{
      loopOperation, CanonicalProgramPointPosition::Before};
  const CanonicalProgramPoint loopAfter{
      loopOperation, CanonicalProgramPointPosition::After};
  if (point == loopBefore) {
    for (unsigned lane = 0; lane < protocol.lanes.size(); ++lane) {
      const bool seededByInitialProducer = llvm::any_of(
          protocol.stages, [lane](const CanonicalOwnershipStage &stage) {
            return stage.lane == lane && stage.initialProducer;
          });
      if (seededByInitialProducer) {
        continue;
      }
      applyEventSet(mechanism, EventPayload::Kind::OwnershipRelease, lane,
                    protocol.consumer, state);
    }
    return;
  }
  if (point == loopAfter) {
    for (unsigned lane = 0; lane < protocol.lanes.size(); ++lane) {
      const bool seededByInitialProducer = llvm::any_of(
          protocol.stages, [lane](const CanonicalOwnershipStage &stage) {
            return stage.lane == lane && stage.initialProducer;
          });
      const bool hasRelease = llvm::any_of(
          state.events, [&](const EventPayload &payload) {
            return sameEvent(payload, mechanism.id,
                             EventPayload::Kind::OwnershipRelease, lane);
          });
      if (seededByInitialProducer && !hasRelease) {
        applyEventWait(mechanism, EventPayload::Kind::OwnershipReady, lane,
                       protocol.consumer, state);
      } else {
        applyEventWait(mechanism, EventPayload::Kind::OwnershipRelease, lane,
                       protocol.producer, state);
      }
    }
    return;
  }
  const bool loopActive = loopIsActive(state, protocol.recurrenceLoop);
  BitVector releaseWaited(protocol.lanes.size());
  BitVector readySet(protocol.lanes.size());
  BitVector readyWaited(protocol.lanes.size());
  BitVector releaseSet(protocol.lanes.size());
  for (const CanonicalOwnershipStage &stage : protocol.stages) {
    const bool producerActive =
        guardEnabled(state.controlPath, stage.producerGuard);
    const bool consumerActive =
        guardEnabled(state.controlPath, stage.consumerGuard);
    const bool stageProducerActive =
        producerActive && (loopActive || stage.initialProducer);
    if (!stage.initialProducer && stageProducerActive &&
        stage.writeAcquire == point && !releaseWaited.test(stage.lane)) {
      applyEventWait(mechanism, EventPayload::Kind::OwnershipRelease,
                     stage.lane, protocol.producer, state);
      releaseWaited.set(stage.lane);
    }
    if (stageProducerActive && stage.ready == point &&
        !readySet.test(stage.lane)) {
      applyEventSet(mechanism, EventPayload::Kind::OwnershipReady, stage.lane,
                    protocol.producer, state);
      readySet.set(stage.lane);
    }
    if (loopActive && consumerActive && stage.readAcquire == point &&
        !readyWaited.test(stage.lane)) {
      applyEventWait(mechanism, EventPayload::Kind::OwnershipReady, stage.lane,
                     protocol.consumer, state);
      readyWaited.set(stage.lane);
    }
    if (loopActive && consumerActive && stage.release == point &&
        !releaseSet.test(stage.lane)) {
      applyEventSet(mechanism, EventPayload::Kind::OwnershipRelease,
                    stage.lane, protocol.consumer, state);
      releaseSet.set(stage.lane);
    }
  }
}

void executePoint(const CanonicalSyncProgram &program,
                  CanonicalProgramPoint point,
                  ArrayRef<CanonicalMechanismId> selected, OracleState &state) {
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (!guardEnabled(state.controlPath, mechanism.guard)) {
      continue;
    }
    // A boundary handshake transfers completion from the loop latch to the
    // next loop header.  Until summary transfers carry iteration transforms,
    // only its explicitly certified recurrence origins may use that effect.
    if (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
        mechanism.boundaryRecurring) {
      continue;
    }
    if (mechanism.kind == CanonicalMechanismKind::PeriodicOwnership) {
      if (!mechanism.ownershipProtocol) {
        state.valid = false;
        continue;
      }
      applyOwnershipPoint(
          program, mechanism,
          program.getOwnershipProtocol(*mechanism.ownershipProtocol), point,
          state);
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
      applyEventSet(mechanism, EventPayload::Kind::Direct, 0,
                    mechanism.source, state);
    }
    if (mechanism.targetPoint == point &&
        (mechanism.kind == CanonicalMechanismKind::Event ||
         mechanism.kind == CanonicalMechanismKind::CrossCoreEvent ||
         mechanism.kind == CanonicalMechanismKind::RecurringEvent)) {
      applyEventWait(mechanism, EventPayload::Kind::Direct, 0,
                     mechanism.target, state);
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
  for (const OracleState &input : inputs) {
    const std::optional<unsigned> knownArm =
        evaluateChoiceArm(program, choice, input);
    for (unsigned arm = 0; arm < 2U; ++arm) {
      if (knownArm && *knownArm != arm) {
        continue;
      }
      SmallVector<OracleState, 1> armInputs(1, input);
      armInputs.front().controlPath.push_back({choiceId, arm});
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
        armOutputs.assign(armInputs.begin(), armInputs.end());
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
  const unsigned unroll =
      loopId < context.loopUnrollByRegion.size()
          ? context.loopUnrollByRegion[loopId]
          : kDefaultOracleLoopUnroll;
  for (unsigned count = 0; count <= unroll; ++count) {
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
  for (OracleState &state : states) {
    executePoint(program, {operation, CanonicalProgramPointPosition::Before},
                 selected, state);
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
                 selected, state);
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

SmallVector<unsigned, 0> buildLoopUnrolls(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalMechanismId> selected, bool &withinBound) {
  SmallVector<unsigned, 0> result(program.getRegions().size(),
                                  kDefaultOracleLoopUnroll);
  for (CanonicalMechanismId id : selected) {
    const CanonicalMechanism &mechanism = program.getMechanism(id);
    if (mechanism.kind != CanonicalMechanismKind::PeriodicOwnership ||
        !mechanism.ownershipProtocol) {
      continue;
    }
    const CanonicalOwnershipProtocol &protocol =
        program.getOwnershipProtocol(*mechanism.ownershipProtocol);
    if (protocol.recurrenceLoop >= result.size()) {
      withinBound = false;
      continue;
    }
    if (protocol.witnessHorizon > kMaximumOracleLoopUnroll) {
      withinBound = false;
    }
    result[protocol.recurrenceLoop] =
        std::max(result[protocol.recurrenceLoop],
                 std::min(protocol.witnessHorizon,
                          kMaximumOracleLoopUnroll));
  }
  return result;
}

} // namespace

FailureOr<CanonicalUnrolledCoverageResult>
mlir::pto::canonical_sync_detail::evaluateCanonicalSyncUnrolledOracle(
    const CanonicalSyncProgram &program,
    ArrayRef<CanonicalMechanismId> selected) {
  OracleContext context;
  context.loopUnrollByRegion =
      buildLoopUnrolls(program, selected, context.exhaustive);
  const std::size_t phaseCount =
      std::max<std::size_t>(program.getPhases().size(), 1);
  const bool staticBudgetExceeded =
      program.getDemands().size() > kMaxOracleStaticChecks / phaseCount;
  const bool stateBudgetExceeded =
      estimateBlockStates(program, program.getFunction().getBody().front(),
                          context.loopUnrollByRegion) >
      kMaxOracleStates;
  if (staticBudgetExceeded || stateBudgetExceeded) {
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
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(program.getFunction());
  if (failed(target)) {
    return failure();
  }
  context.target = &*target;
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
          return state.valid && state.events.empty() &&
                 state.covered[demand.id];
        });
    if (coveredOnEveryPath) {
      result.covered.push_back(demand.id);
    }
  }
  return result;
}
