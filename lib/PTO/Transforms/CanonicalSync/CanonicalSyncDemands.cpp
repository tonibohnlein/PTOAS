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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

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

bool sameDemand(const CanonicalDemand &left, const CanonicalDemand &right) {
  return left.source == right.source && left.target == right.target &&
         left.owner == right.owner && left.kind == right.kind &&
         left.requirement == right.requirement &&
         left.visibility == right.visibility &&
         left.iterationDistance == right.iterationDistance &&
         left.guard == right.guard;
}

CanonicalDemandId appendOrMerge(CanonicalSyncProgram &program,
                                CanonicalDemand demand,
                                CanonicalDemandCause cause) {
  for (const CanonicalDemand &existing : program.getDemands()) {
    if (sameDemand(existing, demand)) {
      program.appendDemandCause(existing.id, std::move(cause));
      return existing.id;
    }
  }
  demand.causes.push_back(std::move(cause));
  return program.appendDemand(std::move(demand));
}

using PhaseMap = llvm::DenseMap<Operation *, SmallVector<CanonicalPhaseId, 2>>;
using CompletionMap = llvm::DenseMap<Value, CanonicalPhaseId>;

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

LogicalResult traceSsaProducers(func::FuncOp function, Value seed,
                                const PhaseMap &operationPhases,
                                const CompletionMap &completions,
                                llvm::SetVector<CanonicalPhaseId> &producers) {
  SmallVector<Value, 16> worklist{seed};
  llvm::DenseSet<Value> discovered;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !discovered.insert(value).second) {
      continue;
    }
    if (auto completion = completions.find(value);
        completion != completions.end()) {
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
      return function.emitError(
          "canonical sync cannot trace SSA through this block argument");
    }
    Operation *definition = value.getDefiningOp();
    if (!definition) {
      continue;
    }
    if (auto result = dyn_cast<OpResult>(value)) {
      if (auto conditional = dyn_cast<scf::IfOp>(definition)) {
        if (failed(enqueueIfResult(result, conditional, worklist))) {
          return failure();
        }
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(definition)) {
        if (failed(enqueueForResult(result, loop, worklist))) {
          return failure();
        }
        continue;
      }
    }
    if (operationPhases.contains(definition)) {
      return definition->emitError(
          "canonical sync has no completion phase for this SSA result");
    }
    if (areMemoryLikeTypes(value.getType())) {
      continue;
    }
    const bool hasNestedRegions = definition->getNumRegions() != 0;
    if (hasNestedRegions || !isMemoryEffectFree(definition)) {
      return definition->emitError(
          "canonical sync cannot trace SSA through this effectful operation");
    }
    worklist.append(definition->operand_begin(), definition->operand_end());
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

LogicalResult deriveSsaDemands(CanonicalSyncProgram &program) {
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
      if (failed(traceSsaProducers(program.getFunction(), operand,
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
      demand.guard =
          intersectControlPaths(source.controlPath, target.controlPath);
      demand.iterationDistance.assign(commonLoops(source, target).size(), 0);
      CanonicalDemandCause cause;
      cause.provenance = "SSA producer completion";
      appendOrMerge(program, std::move(demand), std::move(cause));
    }
  }
  return success();
}

bool requiresVisibility(const CanonicalSyncProgram &program,
                        const CanonicalAccess &source,
                        const CanonicalAccess &target) {
  const bool sourceMayBeGm =
      source.unknownSpace || source.space == AddressSpace::GM;
  const bool targetMayBeGm =
      target.unknownSpace || target.space == AddressSpace::GM;
  if (!sourceMayBeGm || !targetMayBeGm) {
    return false;
  }
  const PIPE sourcePipe = program.getPhase(source.phase).resource.pipe;
  const PIPE targetPipe = program.getPhase(target.phase).resource.pipe;
  return (sourcePipe == PIPE::PIPE_S) != (targetPipe == PIPE::PIPE_S);
}

CanonicalVisibilityRequirement
buildVisibilityRequirement(const CanonicalSyncProgram &program,
                           const CanonicalAccess &source,
                           const CanonicalAccess &target) {
  const bool scalarSource =
      program.getPhase(source.phase).resource.pipe == PIPE::PIPE_S;
  CanonicalVisibilityRequirement requirement;
  requirement.direction = scalarSource
                              ? CanonicalVisibilityDirection::ScalarToNonScalar
                              : CanonicalVisibilityDirection::NonScalarToScalar;
  requirement.scope = FenceScope::GM;
  if (scalarSource && accessWrites(source.mode)) {
    requirement.cacheMaintenance = CanonicalCacheMaintenance::CleanSource;
  } else if (!scalarSource && accessReads(target.mode)) {
    requirement.cacheMaintenance = CanonicalCacheMaintenance::InvalidateTarget;
  }
  return requirement;
}

void addHazardDemand(CanonicalSyncProgram &program,
                     const CanonicalAccess &source,
                     const CanonicalAccess &target, CanonicalDemandKind hazard,
                     ArrayRef<int64_t> distance, CanonicalRegionId owner) {
  CanonicalDemand demand;
  demand.source = source.phase;
  demand.target = target.phase;
  demand.owner = owner;
  demand.kind = requiresVisibility(program, source, target)
                    ? CanonicalDemandKind::Visibility
                    : hazard;
  demand.requirement = demand.kind == CanonicalDemandKind::Visibility
                           ? CanonicalRequirement::Visibility
                           : CanonicalRequirement::Completion;
  if (demand.requirement == CanonicalRequirement::Visibility) {
    demand.visibility = buildVisibilityRequirement(program, source, target);
  }
  demand.iterationDistance.assign(distance.begin(), distance.end());
  demand.guard =
      intersectControlPaths(program.getPhase(source.phase).controlPath,
                            program.getPhase(target.phase).controlPath);
  CanonicalDemandCause cause;
  cause.sourceAccess = source.id;
  cause.targetAccess = target.id;
  cause.provenance =
      (stringifyCanonicalDemandKind(hazard) + Twine(" overlap between a") +
       Twine(source.id) + Twine(" and a") + Twine(target.id))
          .str();
  appendOrMerge(program, std::move(demand), std::move(cause));
}

void deriveMemoryDemands(CanonicalSyncProgram &program) {
  for (const CanonicalAccess &source : program.getAccesses()) {
    const CanonicalPhase &sourcePhase = program.getPhase(source.phase);
    for (const CanonicalAccess &target : program.getAccesses()) {
      const CanonicalPhase &targetPhase = program.getPhase(target.phase);
      const bool samePhase = source.phase == target.phase;
      if (!controlsCanCoexecute(sourcePhase.controlPath,
                                targetPhase.controlPath) ||
          !accessesMayAlias(source, target)) {
        continue;
      }
      std::optional<CanonicalDemandKind> hazard =
          classifyHazard(program, source, target);
      if (!hazard) {
        continue;
      }
      const SmallVector<CanonicalRegionId, 2> loops =
          commonLoops(sourcePhase, targetPhase);
      if (!samePhase && sourcePhase.operation != targetPhase.operation &&
          sourcePhase.sourceOrder < targetPhase.sourceOrder) {
        SmallVector<int64_t, 2> zeroDistance(loops.size(), 0);
        addHazardDemand(
            program, source, target, *hazard, zeroDistance,
            findRegionLca(program, sourcePhase.region, targetPhase.region));
      }
      if (!loops.empty()) {
        SmallVector<int64_t, 2> recurrence(loops.size(), 0);
        recurrence.back() = kCanonicalAnyPositiveDistance;
        addHazardDemand(program, source, target, *hazard, recurrence,
                        loops.back());
      }
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
    demand.guard = phase.controlPath;
    program.appendDemand(std::move(demand));
  }
}

} // namespace

LogicalResult mlir::pto::canonical_sync_detail::deriveCanonicalDemands(
    CanonicalSyncProgram &program) {
  if (failed(deriveSsaDemands(program))) {
    return failure();
  }
  deriveMemoryDemands(program);
  deriveExitDemands(program);
  return success();
}
