// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

std::optional<CanonicalDemandKind>
classifyHazard(const CanonicalSyncProgram &program,
               const CanonicalAccess &source, const CanonicalAccess &target) {
  const CanonicalPhysicalResource sourceResource =
      program.getPhase(source.phase).resource;
  const CanonicalPhysicalResource targetResource =
      program.getPhase(target.phase).resource;
  const bool sourceRead = accessReads(source.mode);
  const bool sourceWrite = accessWrites(source.mode);
  const bool targetRead = accessReads(target.mode);
  const bool targetWrite = accessWrites(target.mode);

  if (source.ordered || target.ordered) {
    return CanonicalDemandKind::OrderedMemory;
  }
  if (sourceWrite && targetRead) {
    return CanonicalDemandKind::Raw;
  }
  if (sourceRead && targetWrite) {
    return CanonicalDemandKind::War;
  }
  if (sourceWrite && targetWrite) {
    return CanonicalDemandKind::Waw;
  }
  const bool sourceMayBeAcc =
      source.unknownSpace || source.space == AddressSpace::ACC;
  const bool targetMayBeAcc =
      target.unknownSpace || target.space == AddressSpace::ACC;
  const bool accReadConflict = sourceRead && targetRead && sourceMayBeAcc &&
                               targetMayBeAcc &&
                               sourceResource.core == CanonicalCore::AIC &&
                               targetResource.core == CanonicalCore::AIC &&
                               sourceResource.pipe != targetResource.pipe;
  return accReadConflict ? std::optional<CanonicalDemandKind>(
                               CanonicalDemandKind::HardwareAccReadConflict)
                         : std::nullopt;
}

SmallVector<CanonicalRegionId, 2> commonLoops(const CanonicalPhase &source,
                                              const CanonicalPhase &target) {
  SmallVector<CanonicalRegionId, 2> result;
  for (CanonicalRegionId loop : source.loopPath) {
    if (llvm::is_contained(target.loopPath, loop)) {
      result.push_back(loop);
    }
  }
  return result;
}

bool regionIsNestedIn(const CanonicalSyncProgram &program,
                      CanonicalRegionId region, CanonicalRegionId ancestor) {
  for (CanonicalRegionId current = region; current != kInvalidCanonicalSyncId;
       current = program.getRegion(current).parent) {
    if (current == ancestor) {
      return true;
    }
  }
  return false;
}

bool controlsCanCoexecuteAtDistance(const CanonicalSyncProgram &program,
                                    ArrayRef<CanonicalControlAtom> source,
                                    ArrayRef<CanonicalControlAtom> target,
                                    CanonicalRegionId carryingLoop) {
  for (const CanonicalControlAtom &left : source) {
    for (const CanonicalControlAtom &right : target) {
      if (left.choice != right.choice || left.arm == right.arm) {
        continue;
      }
      // A choice nested in the carrying loop is re-evaluated in each dynamic
      // iteration. Opposite arms can therefore supply the two endpoints. A
      // choice outside the carrying loop remains mutually exclusive.
      if (!regionIsNestedIn(program, left.choice, carryingLoop)) {
        return false;
      }
    }
  }
  return true;
}

SmallVector<CanonicalLoopDistance, 2>
sameIterationRelation(ArrayRef<CanonicalRegionId> loops) {
  SmallVector<CanonicalLoopDistance, 2> result;
  for (CanonicalRegionId loop : loops) {
    result.push_back({loop, CanonicalIterationRelation::Same});
  }
  return result;
}

SmallVector<CanonicalLoopDistance, 2>
carriedIterationRelation(ArrayRef<CanonicalRegionId> loops,
                         size_t carryingIndex) {
  SmallVector<CanonicalLoopDistance, 2> result;
  for (auto [index, loop] : llvm::enumerate(loops)) {
    CanonicalIterationRelation relation = CanonicalIterationRelation::Same;
    if (index == carryingIndex) {
      relation = CanonicalIterationRelation::AnyPositive;
    } else if (index > carryingIndex) {
      relation = CanonicalIterationRelation::Any;
    }
    result.push_back({loop, relation});
  }
  return result;
}

bool sameDemand(const CanonicalDemand &left, const CanonicalDemand &right) {
  return left.source == right.source && left.target == right.target &&
         left.owner == right.owner && left.kind == right.kind &&
         left.requirement == right.requirement &&
         left.visibility == right.visibility &&
         left.iterationDistance == right.iterationDistance &&
         left.sourceGuard == right.sourceGuard &&
         left.targetGuard == right.targetGuard;
}

std::uint64_t hashDemand(const CanonicalDemand &demand) {
  llvm::hash_code hash =
      llvm::hash_combine(demand.source, demand.target, demand.owner,
                         static_cast<unsigned>(demand.kind),
                         static_cast<unsigned>(demand.requirement));
  if (demand.visibility) {
    hash = llvm::hash_combine(
        hash, static_cast<unsigned>(demand.visibility->direction),
        static_cast<unsigned>(demand.visibility->scope),
        static_cast<unsigned>(demand.visibility->cacheMaintenance));
  } else {
    hash = llvm::hash_combine(hash, 0U);
  }
  for (const CanonicalLoopDistance &distance : demand.iterationDistance) {
    hash = llvm::hash_combine(hash, distance.loop,
                              static_cast<unsigned>(distance.relation));
  }
  for (const CanonicalControlAtom &atom : demand.sourceGuard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  hash = llvm::hash_combine(hash, demand.sourceGuard.size());
  for (const CanonicalControlAtom &atom : demand.targetGuard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  hash = llvm::hash_combine(hash, demand.targetGuard.size());
  return static_cast<std::uint64_t>(static_cast<std::size_t>(hash));
}

class DemandIndex {
public:
  std::optional<CanonicalDemandId>
  find(const CanonicalSyncProgram &program,
       const CanonicalDemand &candidate) const {
    auto found = buckets.find(hashDemand(candidate));
    if (found == buckets.end()) {
      return std::nullopt;
    }
    for (CanonicalDemandId id : found->second) {
      if (sameDemand(program.getDemand(id), candidate)) {
        return id;
      }
    }
    return std::nullopt;
  }

  void insert(const CanonicalSyncProgram &program, CanonicalDemandId id) {
    buckets[hashDemand(program.getDemand(id))].push_back(id);
  }

private:
  llvm::DenseMap<std::uint64_t, SmallVector<CanonicalDemandId, 1>> buckets;
};

CanonicalDemandId appendOrMerge(CanonicalSyncProgram &program,
                                DemandIndex &index, CanonicalDemand demand,
                                CanonicalDemandCause cause) {
  if (std::optional<CanonicalDemandId> existing = index.find(program, demand)) {
    program.appendDemandCause(*existing, std::move(cause));
    return *existing;
  }
  demand.causes.push_back(std::move(cause));
  CanonicalDemandId id = program.appendDemand(std::move(demand));
  index.insert(program, id);
  return id;
}

using PhaseMap = llvm::DenseMap<Operation *, SmallVector<CanonicalPhaseId, 2>>;
using CompletionMap = llvm::DenseMap<Value, CanonicalPhaseId>;

struct AccessPair {
  CanonicalAccessId source = kInvalidCanonicalSyncId;
  CanonicalAccessId target = kInvalidCanonicalSyncId;
};

struct LocalIntervalRecord {
  AddressSpace space = AddressSpace::Zero;
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
  CanonicalAccessId access = kInvalidCanonicalSyncId;
};

std::uint64_t accessPairKey(CanonicalAccessId source,
                            CanonicalAccessId target) {
  return (static_cast<std::uint64_t>(source) << 32U) |
         static_cast<std::uint64_t>(target);
}

SmallVector<AccessPair, 32> buildMayAliasPairs(CanonicalSyncProgram &program) {
  SmallVector<LocalIntervalRecord, 32> intervals;
  SmallVector<CanonicalAccessId, 16> fallback;
  for (const CanonicalAccess &access : program.getAccesses()) {
    const bool knownLocalRange =
        !access.unknownSpace && access.space != AddressSpace::GM &&
        access.physical && !access.unknownRange && !access.intervals.empty();
    bool validIntervals = knownLocalRange;
    if (knownLocalRange) {
      for (const CanonicalByteInterval &interval : access.intervals) {
        const std::optional<std::uint64_t> end = interval.end();
        if (!end) {
          validIntervals = false;
          break;
        }
      }
    }
    if (!validIntervals) {
      fallback.push_back(access.id);
      continue;
    }
    for (const CanonicalByteInterval &interval : access.intervals) {
      const std::uint64_t end = *interval.end();
      if (interval.begin == end) {
        continue;
      }
      intervals.push_back({access.space, interval.begin, end, access.id});
    }
  }

  llvm::sort(intervals, [](const LocalIntervalRecord &left,
                           const LocalIntervalRecord &right) {
    return std::tie(left.space, left.begin, left.end, left.access) <
           std::tie(right.space, right.begin, right.end, right.access);
  });

  llvm::DenseSet<std::uint64_t> pairKeys;
  const auto addBoth = [&pairKeys](CanonicalAccessId first,
                                   CanonicalAccessId second) {
    pairKeys.insert(accessPairKey(first, second));
    pairKeys.insert(accessPairKey(second, first));
  };
  SmallVector<LocalIntervalRecord, 16> active;
  std::optional<AddressSpace> activeSpace;
  for (const LocalIntervalRecord &current : intervals) {
    if (!activeSpace || *activeSpace != current.space) {
      active.clear();
      activeSpace = current.space;
    }
    llvm::erase_if(active, [&](const LocalIntervalRecord &candidate) {
      return candidate.end <= current.begin;
    });
    for (const LocalIntervalRecord &candidate : active) {
      addBoth(candidate.access, current.access);
    }
    addBoth(current.access, current.access);
    active.push_back(current);
  }

  CanonicalSyncStatistics *statistics = program.getStatistics();
  if (statistics) {
    statistics->localIntervalRecords = intervals.size();
  }
  for (CanonicalAccessId first : fallback) {
    for (const CanonicalAccess &second : program.getAccesses()) {
      if (statistics) {
        ++statistics->aliasPairTests;
      }
      if (accessesMayAlias(program.getAccess(first), second,
                           program.getGmAliasPolicy())) {
        addBoth(first, second.id);
      }
    }
  }

  SmallVector<AccessPair, 32> result;
  result.reserve(pairKeys.size());
  for (std::uint64_t key : pairKeys) {
    result.push_back({static_cast<CanonicalAccessId>(key >> 32U),
                      static_cast<CanonicalAccessId>(key)});
  }
  llvm::sort(result, [](const AccessPair &left, const AccessPair &right) {
    return std::tie(left.source, left.target) <
           std::tie(right.source, right.target);
  });
  if (statistics) {
    statistics->aliasCandidatePairs = result.size();
  }
  return result;
}

LogicalResult indexSsaCompletionPhases(const CanonicalSyncProgram &program,
                                       PhaseMap &operationPhases,
                                       CompletionMap &completions) {
  for (const CanonicalPhase &phase : program.getPhases()) {
    operationPhases[phase.operation].push_back(phase.id);
  }
  for (const auto &entry : operationPhases) {
    Operation *operation = entry.first;
    if (operation->getResults().empty()) {
      continue;
    }
    const bool hasUniqueCompletion = entry.second.size() == 1;
    if (!hasUniqueCompletion) {
      return operation->emitError(
          "canonical sync cannot bind an SSA result to a multi-phase macro");
    }
    for (Value result : operation->getResults()) {
      completions[result] = entry.second.front();
    }
  }
  return success();
}

LogicalResult enqueueIfResult(OpResult result, scf::IfOp operation,
                              SmallVectorImpl<Value> &worklist) {
  const unsigned index = result.getResultNumber();
  for (Region *region :
       {&operation.getThenRegion(), &operation.getElseRegion()}) {
    if (region->empty()) {
      continue;
    }
    auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
    if (!yield || index >= yield.getNumOperands()) {
      return operation.emitError(
          "canonical sync cannot trace an scf.if result to its yields");
    }
    worklist.push_back(yield.getOperand(index));
  }
  return success();
}

LogicalResult enqueueForResult(OpResult result, scf::ForOp operation,
                               SmallVectorImpl<Value> &worklist) {
  const unsigned index = result.getResultNumber();
  auto yield = dyn_cast<scf::YieldOp>(operation.getBody()->getTerminator());
  const bool invalidInit = index >= operation.getInitArgs().size();
  const bool invalidYield = !yield || index >= yield.getNumOperands();
  if (invalidInit || invalidYield) {
    return operation.emitError(
        "canonical sync cannot trace an scf.for result to its values");
  }
  worklist.push_back(operation.getInitArgs()[index]);
  worklist.push_back(yield.getOperand(index));
  return success();
}

LogicalResult traceSsaProducers(func::FuncOp function,
                                CanonicalSyncStatistics *statistics, Value seed,
                                const PhaseMap &operationPhases,
                                const CompletionMap &completions,
                                llvm::SetVector<CanonicalPhaseId> &producers) {
  struct TraceValue {
    Value value;
    bool fromLoopBackedge = false;
  };
  SmallVector<TraceValue, 16> worklist{{seed, false}};
  llvm::DenseSet<Value> discoveredForward;
  llvm::DenseSet<Value> discoveredBackedge;
  while (!worklist.empty()) {
    const TraceValue current = worklist.pop_back_val();
    Value value = current.value;
    if (!value) {
      continue;
    }
    llvm::DenseSet<Value> &discovered = current.fromLoopBackedge
                                            ? discoveredBackedge
                                            : discoveredForward;
    if (!discovered.insert(value).second) {
      continue;
    }
    if (statistics) {
      ++statistics->ssaTraceVisits;
    }
    if (auto completion = completions.find(value);
        completion != completions.end()) {
      const bool synchronous = getVPTOSchedulingSemantics(value.getDefiningOp())
                                   .completionIsSynchronous;
      if (current.fromLoopBackedge && !synchronous) {
        return function.emitError(
            "canonical sync rejects a loop-carried physical SSA result "
            "until recurrence-aware SSA completion is implemented");
      }
      producers.insert(completion->second);
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Block *owner = argument.getOwner();
      if (owner == &function.getBody().front()) {
        continue;
      }
      auto loop = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
      if (loop && argument == loop.getInductionVar()) {
        continue;
      }
      if (loop) {
        // Scalar/index iter arguments are ordinary structured SSA and occur in
        // address calculations throughout generated PyPTO kernels. Follow
        // both the initialization and yielded value rather than treating the
        // block argument as an opaque producer. Keep the backedge label for
        // every type: if the chain reaches a physical producer, recurrence-
        // aware SSA completion is still required and we fail closed above.
        const unsigned argumentNumber = argument.getArgNumber();
        if (argumentNumber == 0) {
          return loop.emitError("canonical sync found an invalid induction "
                                "value while tracing SSA completion");
        }
        const unsigned iterationIndex = argumentNumber - 1;
        auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        const bool validBackedge = iterationIndex < loop.getInitArgs().size() &&
                                   yield &&
                                   iterationIndex < yield.getNumOperands();
        if (!validBackedge) {
          return loop.emitError(
              "canonical sync cannot trace an scf.for loop argument");
        }
        worklist.push_back({loop.getInitArgs()[iterationIndex], false});
        worklist.push_back({yield.getOperand(iterationIndex), true});
        continue;
      }
      return function.emitError(
          "canonical sync cannot trace SSA through this block argument");
    }
    Operation *definition = value.getDefiningOp();
    if (!definition) {
      continue;
    }
    if (auto result = dyn_cast<OpResult>(value)) {
      if (auto conditional = dyn_cast<scf::IfOp>(definition)) {
        SmallVector<Value, 4> predecessors;
        if (failed(enqueueIfResult(result, conditional, predecessors))) {
          return failure();
        }
        for (Value predecessor : predecessors) {
          worklist.push_back({predecessor, current.fromLoopBackedge});
        }
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(definition)) {
        SmallVector<Value, 2> predecessors;
        if (failed(enqueueForResult(result, loop, predecessors))) {
          return failure();
        }
        for (Value predecessor : predecessors) {
          worklist.push_back({predecessor, current.fromLoopBackedge});
        }
        continue;
      }
    }
    if (operationPhases.contains(definition)) {
      return definition->emitError(
          "canonical sync has no completion phase for this SSA result");
    }
    const bool structuralStorageRoot =
        areMemoryLikeTypes(value.getType()) &&
        isa<AllocTileOp, AllocMultiTileOp, DeclareTileOp, DeclareGlobalOp>(
            definition);
    const bool structuralPipeRoot =
        isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(definition);
    if (structuralStorageRoot || structuralPipeRoot) {
      continue;
    }
    const bool hasNestedRegions = definition->getNumRegions() != 0;
    if (hasNestedRegions || !isMemoryEffectFree(definition)) {
      return definition->emitError(
          "canonical sync cannot trace SSA through this effectful operation");
    }
    for (Value operand : definition->getOperands()) {
      worklist.push_back({operand, current.fromLoopBackedge});
    }
  }
  return success();
}

FailureOr<SmallVector<Value, 8>>
getPhaseSsaOperands(const CanonicalPhase &phase) {
  if (!phase.macroPhase) {
    return SmallVector<Value, 8>(phase.operation->operand_begin(),
                                 phase.operation->operand_end());
  }
  std::optional<SyncMacroModel> model = getSyncMacroModel(phase.operation);
  if (!model) {
    phase.operation->emitError(
        "canonical sync lost the operand model for a macro phase");
    return failure();
  }
  const auto found =
      llvm::find_if(model->phases, [&](const SyncMacroPhase &item) {
        return item.phaseId == *phase.macroPhase;
      });
  if (found == model->phases.end()) {
    phase.operation->emitError(
        "canonical sync cannot find the modeled macro phase");
    return failure();
  }
  return SmallVector<Value, 8>(found->useValues.begin(),
                               found->useValues.end());
}

LogicalResult deriveSsaDemands(CanonicalSyncProgram &program,
                               DemandIndex &index) {
  PhaseMap operationPhases;
  CompletionMap completions;
  if (failed(indexSsaCompletionPhases(program, operationPhases, completions))) {
    return failure();
  }
  for (const CanonicalPhase &target : program.getPhases()) {
    FailureOr<SmallVector<Value, 8>> operands = getPhaseSsaOperands(target);
    if (failed(operands)) {
      return failure();
    }
    llvm::SetVector<CanonicalPhaseId> producers;
    for (Value operand : *operands) {
      if (failed(traceSsaProducers(program.getFunction(),
                                   program.getStatistics(), operand,
                                   operationPhases, completions, producers))) {
        return failure();
      }
    }
    for (CanonicalPhaseId sourceId : producers) {
      const CanonicalPhase &source = program.getPhase(sourceId);
      if (source.operation == target.operation ||
          source.sourceOrder >= target.sourceOrder ||
          !controlsCanCoexecute(source.controlPath, target.controlPath)) {
        continue;
      }
      CanonicalDemand demand;
      demand.source = source.id;
      demand.target = target.id;
      demand.owner = findRegionLca(program, source.region, target.region);
      demand.kind = CanonicalDemandKind::SsaCompletion;
      demand.requirement = CanonicalRequirement::Completion;
      demand.sourceGuard = source.controlPath;
      demand.targetGuard = target.controlPath;
      demand.iterationDistance =
          sameIterationRelation(commonLoops(source, target));
      CanonicalDemandCause cause;
      cause.provenance = "SSA producer completion";
      appendOrMerge(program, index, std::move(demand), std::move(cause));
    }
  }
  return success();
}

bool requiresVisibility(const CanonicalSyncProgram &program,
                        const CanonicalSyncTarget &targetModel,
                        const CanonicalAccess &source,
                        const CanonicalAccess &target) {
  const bool sourceMayBeGm =
      source.unknownSpace || source.space == AddressSpace::GM;
  const bool targetMayBeGm =
      target.unknownSpace || target.space == AddressSpace::GM;
  if (!sourceMayBeGm || !targetMayBeGm) {
    return false;
  }
  const CanonicalPhysicalResource sourceResource =
      program.getPhase(source.phase).resource;
  const CanonicalPhysicalResource targetResource =
      program.getPhase(target.phase).resource;
  const PIPE sourcePipe = sourceResource.pipe;
  const PIPE targetPipe = targetResource.pipe;
  const bool scalarCrossing =
      (sourcePipe == PIPE::PIPE_S) != (targetPipe == PIPE::PIPE_S);
  const bool unprovenMteRoundTrip =
      sourcePipe == PIPE::PIPE_MTE3 && targetPipe == PIPE::PIPE_MTE2 &&
      accessWrites(source.mode) && accessReads(target.mode) &&
      !targetModel.supportsEventGmPublication(sourceResource, targetResource);
  // A pure WAR edge needs execution completion but transfers no value through
  // scalar DCache: the scalar read must finish before the non-scalar write (or
  // vice versa), which a directed event supplies.  RAW and WAW crossings can
  // expose or later write back cached data and retain the stronger fence/CMO
  // requirement.
  const bool pureWar = accessReads(source.mode) && !accessWrites(source.mode) &&
                       accessWrites(target.mode) && !accessReads(target.mode);
  return (scalarCrossing && !pureWar) || unprovenMteRoundTrip;
}

CanonicalVisibilityRequirement
buildVisibilityRequirement(const CanonicalSyncProgram &program,
                           const CanonicalAccess &source,
                           const CanonicalAccess &target) {
  const bool scalarSource =
      program.getPhase(source.phase).resource.pipe == PIPE::PIPE_S;
  const PIPE sourcePipe = program.getPhase(source.phase).resource.pipe;
  const PIPE targetPipe = program.getPhase(target.phase).resource.pipe;
  CanonicalVisibilityRequirement requirement;
  if (sourcePipe == PIPE::PIPE_MTE3 && targetPipe == PIPE::PIPE_MTE2) {
    requirement.direction = CanonicalVisibilityDirection::Mte3ToMte2Gm;
  } else {
    requirement.direction =
        scalarSource ? CanonicalVisibilityDirection::ScalarToNonScalar
                     : CanonicalVisibilityDirection::NonScalarToScalar;
  }
  requirement.scope = FenceScope::GM;
  if (requirement.direction == CanonicalVisibilityDirection::Mte3ToMte2Gm) {
    return requirement;
  }
  if (scalarSource && accessWrites(source.mode)) {
    requirement.cacheMaintenance = CanonicalCacheMaintenance::CleanSource;
  } else if (!scalarSource && accessReads(target.mode)) {
    requirement.cacheMaintenance = CanonicalCacheMaintenance::InvalidateTarget;
  }
  return requirement;
}

void addHazardDemand(CanonicalSyncProgram &program, DemandIndex &index,
                     const CanonicalSyncTarget &targetModel,
                     const CanonicalAccess &source,
                     const CanonicalAccess &target, CanonicalDemandKind hazard,
                     ArrayRef<CanonicalLoopDistance> distance,
                     CanonicalRegionId owner) {
  CanonicalDemand demand;
  demand.source = source.phase;
  demand.target = target.phase;
  demand.owner = owner;
  demand.kind = requiresVisibility(program, targetModel, source, target)
                    ? CanonicalDemandKind::Visibility
                    : hazard;
  demand.requirement = demand.kind == CanonicalDemandKind::Visibility
                           ? CanonicalRequirement::Visibility
                           : CanonicalRequirement::Completion;
  if (demand.requirement == CanonicalRequirement::Visibility) {
    demand.visibility = buildVisibilityRequirement(program, source, target);
  }
  demand.iterationDistance.assign(distance.begin(), distance.end());
  demand.sourceGuard = program.getPhase(source.phase).controlPath;
  demand.targetGuard = program.getPhase(target.phase).controlPath;
  CanonicalDemandCause cause;
  cause.sourceAccess = source.id;
  cause.targetAccess = target.id;
  cause.provenance =
      (stringifyCanonicalDemandKind(hazard) + Twine(" overlap between a") +
       Twine(source.id) + Twine(" and a") + Twine(target.id))
          .str();
  appendOrMerge(program, index, std::move(demand), std::move(cause));
}

void deriveMemoryDemands(CanonicalSyncProgram &program, DemandIndex &index,
                         const CanonicalSyncTarget &targetModel) {
  for (const AccessPair &pair : buildMayAliasPairs(program)) {
    const CanonicalAccess &source = program.getAccess(pair.source);
    const CanonicalPhase &sourcePhase = program.getPhase(source.phase);
    const CanonicalAccess &target = program.getAccess(pair.target);
    const CanonicalPhase &targetPhase = program.getPhase(target.phase);
    const bool samePhase = source.phase == target.phase;
    std::optional<CanonicalDemandKind> hazard =
        classifyHazard(program, source, target);
    if (!hazard) {
      continue;
    }
    const SmallVector<CanonicalRegionId, 2> loops =
        commonLoops(sourcePhase, targetPhase);
    const bool sameIterationCompatible =
        controlsCanCoexecute(sourcePhase.controlPath, targetPhase.controlPath);
    if (sameIterationCompatible && !samePhase &&
        sourcePhase.operation != targetPhase.operation &&
        sourcePhase.sourceOrder < targetPhase.sourceOrder) {
      const SmallVector<CanonicalLoopDistance, 2> zeroDistance =
          sameIterationRelation(loops);
      addHazardDemand(
          program, index, targetModel, source, target, *hazard, zeroDistance,
          findRegionLca(program, sourcePhase.region, targetPhase.region));
    }
    for (size_t carryingIndex = 0; carryingIndex < loops.size();
         ++carryingIndex) {
      const CanonicalRegionId carryingLoop = loops[carryingIndex];
      if (!controlsCanCoexecuteAtDistance(program, sourcePhase.controlPath,
                                          targetPhase.controlPath,
                                          carryingLoop)) {
        continue;
      }
      const SmallVector<CanonicalLoopDistance, 2> recurrence =
          carriedIterationRelation(loops, carryingIndex);
      addHazardDemand(program, index, targetModel, source, target, *hazard,
                      recurrence, carryingLoop);
    }
  }
}

void deriveExitDemands(CanonicalSyncProgram &program) {
  for (const CanonicalPhase &phase : program.getPhases()) {
    CanonicalDemand demand;
    demand.source = phase.id;
    demand.target = kInvalidCanonicalSyncId;
    demand.owner = 0;
    demand.kind = CanonicalDemandKind::ExitCompletion;
    demand.requirement = CanonicalRequirement::Completion;
    demand.sourceGuard = phase.controlPath;
    program.appendDemand(std::move(demand));
  }
}

} // namespace

LogicalResult mlir::pto::canonical_sync_detail::deriveCanonicalDemands(
    CanonicalSyncProgram &program, const CanonicalSyncTarget &target) {
  DemandIndex index;
  if (failed(deriveSsaDemands(program, index))) {
    return failure();
  }
  deriveMemoryDemands(program, index, target);
  deriveExitDemands(program);
  return success();
}
