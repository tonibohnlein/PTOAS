// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

using SteadyClock = std::chrono::steady_clock;

std::uint64_t elapsedNanoseconds(SteadyClock::time_point begin) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           SteadyClock::now() - begin)
                           .count();
  return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

void addNanoseconds(std::uint64_t &total, std::uint64_t amount) {
  total = amount > std::numeric_limits<std::uint64_t>::max() - total
              ? std::numeric_limits<std::uint64_t>::max()
              : total + amount;
}

bool checkedWorkSum(std::size_t first, std::size_t second,
                    std::size_t &result) {
  const bool valid = second <= std::numeric_limits<std::size_t>::max() - first;
  result = valid ? first + second : 0;
  return valid;
}

bool checkedWorkProduct(std::size_t first, std::size_t second,
                        std::size_t &result) {
  const bool valid =
      first == 0 || second <= std::numeric_limits<std::size_t>::max() / first;
  result = valid ? first * second : 0;
  return valid;
}

using AllocationKey = std::pair<CanonicalSyncMechanismId, std::size_t>;

struct ConcreteAction {
  CanonicalSyncActionKind kind = CanonicalSyncActionKind::Barrier;
  Operation *anchor = nullptr;
  bool before = true;
  PipelineType source = PipelineType::PIPE_UNASSIGNED;
  PipelineType target = PipelineType::PIPE_UNASSIGNED;
  PipelineType barrier = PipelineType::PIPE_UNASSIGNED;
  unsigned eventId = 0;
  std::vector<unsigned> eventIds;
  scf::ForOp eventLaneLoop;
  CanonicalSyncActionGuardKind guard = CanonicalSyncActionGuardKind::None;
  scf::ForOp guardLoop;
  bool tailFence = false;
};

struct ActionGroup {
  Operation *anchor = nullptr;
  bool before = true;
  std::vector<ConcreteAction> actions;
};

bool isPhysicalEventPipe(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
  case PipelineType::PIPE_V:
  case PipelineType::PIPE_M:
  case PipelineType::PIPE_MTE1:
  case PipelineType::PIPE_MTE2:
  case PipelineType::PIPE_MTE3:
  case PipelineType::PIPE_FIX:
    return true;
  case PipelineType::PIPE_ALL:
  case PipelineType::PIPE_MTE4:
  case PipelineType::PIPE_MTE5:
  case PipelineType::PIPE_V2:
  case PipelineType::VIRTUAL_PIPE_MTE2_L1A:
  case PipelineType::VIRTUAL_PIPE_MTE2_L1B:
  case PipelineType::PIPE_NUM:
  case PipelineType::PIPE_UNASSIGNED:
    return false;
  }
  return false;
}

PipeAttr getPipeAttr(Builder &builder, PipelineType pipe) {
  return PipeAttr::get(builder.getContext(), static_cast<PIPE>(pipe));
}

EventAttr getEventAttr(Builder &builder, unsigned eventId) {
  return EventAttr::get(builder.getContext(), static_cast<EVENT>(eventId));
}

std::vector<std::uint32_t> getIssueResources(const SyncCoverGraph &graph) {
  std::vector<std::uint32_t> resources;
  for (const SyncCoverNode &node : graph.getNodes()) {
    resources.push_back(node.resource);
  }
  llvm::sort(resources);
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  return resources;
}

Operation *getFirstUnownedOperation(Block &block) {
  for (Operation &operation : block) {
    if (!isa_and_nonnull<UnitAttr>(operation.getAttr("pto.canonical_sync"))) {
      return &operation;
    }
  }
  return nullptr;
}

std::optional<std::pair<Operation *, bool>>
resolvePhysicalAnchor(const CanonicalSyncProgram &program,
                      const SyncCoverAnchor &anchor) {
  switch (anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode: {
    if (anchor.node >= program.getNodeBindings().size()) {
      return std::nullopt;
    }
    Operation *operation = program.getNodeBindings()[anchor.node].operation;
    if (!operation) {
      return std::nullopt;
    }
    return std::make_pair(operation,
                          anchor.kind == SyncCoverAnchorKind::BeforeNode);
  }
  case SyncCoverAnchorKind::ControlEntry:
  case SyncCoverAnchorKind::ControlExit: {
    if (anchor.node >= program.getControlBindings().size()) {
      return std::nullopt;
    }
    Operation *owner = program.getControlBindings()[anchor.node].owner;
    if (!owner || !isa<scf::IfOp>(owner)) {
      return std::nullopt;
    }
    return std::make_pair(owner,
                          anchor.kind == SyncCoverAnchorKind::ControlEntry);
  }
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit: {
    if (anchor.scope >= program.getScopeBindings().size()) {
      return std::nullopt;
    }
    const CanonicalSyncScopeBinding &binding =
        program.getScopeBindings()[anchor.scope];
    Operation *owner = binding.owner;
    if (!owner || owner == program.getFunction().getOperation()) {
      return std::nullopt;
    }
    if (isa<scf::ForOp>(owner)) {
      return std::make_pair(owner,
                            anchor.kind == SyncCoverAnchorKind::ScopeEntry);
    }
    const bool invalidRegion = !binding.region || binding.region->empty() ||
                               !binding.region->hasOneBlock() ||
                               binding.region->front().empty();
    if (invalidRegion) {
      return std::nullopt;
    }
    Block &block = binding.region->front();
    Operation *position = anchor.kind == SyncCoverAnchorKind::ScopeEntry
                              ? getFirstUnownedOperation(block)
                              : &block.back();
    if (!position) {
      return std::nullopt;
    }
    if (anchor.kind == SyncCoverAnchorKind::ScopeExit &&
        !position->hasTrait<OpTrait::IsTerminator>()) {
      return std::nullopt;
    }
    return std::make_pair(position, true);
  }
  case SyncCoverAnchorKind::LoopBodyEntry:
  case SyncCoverAnchorKind::LoopBodyExit: {
    if (anchor.scope >= program.getScopeBindings().size()) {
      return std::nullopt;
    }
    Operation *owner = program.getScopeBindings()[anchor.scope].owner;
    auto loop = dyn_cast_or_null<scf::ForOp>(owner);
    if (!loop || loop.getBody()->empty()) {
      return std::nullopt;
    }
    Operation *position = anchor.kind == SyncCoverAnchorKind::LoopBodyEntry
                              ? getFirstUnownedOperation(*loop.getBody())
                              : loop.getBody()->getTerminator();
    if (!position) {
      return std::nullopt;
    }
    return std::make_pair(position, true);
  }
  case SyncCoverAnchorKind::TimelinePoint:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<PipelineType>
resolveBarrierPipe(const CanonicalSyncAction &action,
                   const std::vector<std::uint32_t> &allResources) {
  switch (action.barrierKind) {
  case CanonicalSyncBarrierKind::Targeted: {
    const bool hasSingleResource = action.drainedResources.size() == 1;
    const bool targetsResource =
        hasSingleResource && action.drainedResources.front() == action.resource;
    if (!targetsResource) {
      return std::nullopt;
    }
    const PipelineType pipe = static_cast<PipelineType>(action.resource);
    return isPhysicalEventPipe(pipe) ? std::optional<PipelineType>(pipe)
                                     : std::nullopt;
  }
  case CanonicalSyncBarrierKind::All:
    return action.drainedResources == allResources
               ? std::optional<PipelineType>(PipelineType::PIPE_ALL)
               : std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::map<AllocationKey, std::vector<unsigned>>>
indexAllocations(const CanonicalSyncPatternProblem &problem,
                 const CanonicalSyncVerifiedPlan &plan) {
  std::size_t expectedUses = 0;
  for (std::size_t index = 0; index < plan.mechanisms.size(); ++index) {
    const CanonicalSyncMechanismId mechanism = plan.mechanisms[index];
    const bool invalidMechanism =
        mechanism >= problem.getMechanisms().size() ||
        (index != 0 && plan.mechanisms[index - 1] >= mechanism);
    if (invalidMechanism) {
      return std::nullopt;
    }
    const std::size_t uses =
        problem.getMechanisms()[mechanism].descriptor.eventUses.size();
    if (uses > std::numeric_limits<std::size_t>::max() - expectedUses) {
      return std::nullopt;
    }
    expectedUses += uses;
  }
  if (plan.allocation.domains.size() != problem.getDomains().size()) {
    return std::nullopt;
  }

  std::map<AllocationKey, std::vector<unsigned>> result;
  std::vector<bool> indexedDomains(problem.getDomains().size(), false);
  // Track ownership in the descriptor's physical domain. An allocation entry
  // must not evade the no-reuse rule by claiming a different outer domain.
  std::vector<std::vector<bool>> assignedIds;
  assignedIds.reserve(problem.getDomains().size());
  for (const CanonicalSyncEventDomain &domain : problem.getDomains()) {
    assignedIds.emplace_back(domain.budget, false);
  }
  for (const CanonicalSyncDomainAllocation &domain : plan.allocation.domains) {
    if (domain.domain >= problem.getDomains().size() ||
        indexedDomains[domain.domain]) {
      return std::nullopt;
    }
    indexedDomains[domain.domain] = true;
    const CanonicalSyncEventDomain &eventDomain =
        problem.getDomains()[domain.domain];
    for (const CanonicalSyncEventAllocation &use : domain.uses) {
      const bool selected = std::binary_search(
          plan.mechanisms.begin(), plan.mechanisms.end(), use.mechanism);
      if (!selected || use.mechanism >= problem.getMechanisms().size()) {
        return std::nullopt;
      }
      const CanonicalSyncMechanismDescriptor &descriptor =
          problem.getMechanisms()[use.mechanism].descriptor;
      if (use.eventUse >= descriptor.eventUses.size()) {
        return std::nullopt;
      }
      const CanonicalSyncEventUse &descriptorUse =
          descriptor.eventUses[use.eventUse];
      if (descriptorUse.domain != domain.domain ||
          use.ids.size() != descriptorUse.width) {
        return std::nullopt;
      }
      for (unsigned id : use.ids) {
        const bool outOfRange = id >= eventDomain.budget;
        const bool reserved =
            !outOfRange &&
            std::binary_search(eventDomain.reservedIds.begin(),
                               eventDomain.reservedIds.end(), id);
        const bool duplicate =
            !outOfRange && assignedIds[descriptorUse.domain][id];
        if (outOfRange || reserved || duplicate) {
          return std::nullopt;
        }
        assignedIds[descriptorUse.domain][id] = true;
      }
      const AllocationKey key{use.mechanism, use.eventUse};
      if (!result.emplace(key, use.ids).second) {
        return std::nullopt;
      }
    }
  }
  if (llvm::is_contained(indexedDomains, false) ||
      result.size() != expectedUses) {
    return std::nullopt;
  }
  for (CanonicalSyncMechanismId mechanism : plan.mechanisms) {
    const std::size_t uses =
        problem.getMechanisms()[mechanism].descriptor.eventUses.size();
    for (std::size_t use = 0; use < uses; ++use) {
      if (result.find({mechanism, use}) == result.end()) {
        return std::nullopt;
      }
    }
  }
  return result;
}

std::optional<ConcreteAction> makeConcreteAction(
    const CanonicalSyncProgram &program,
    const CanonicalSyncPatternProblem &problem,
    const CanonicalSyncMechanism &mechanism, const CanonicalSyncAction &action,
    const std::map<AllocationKey, std::vector<unsigned>> &allocations,
    const std::vector<std::uint32_t> &allResources) {
  const auto physicalAnchor = resolvePhysicalAnchor(program, action.anchor);
  if (!physicalAnchor ||
      !program.getFunction()->isAncestor(physicalAnchor->first)) {
    return std::nullopt;
  }
  ConcreteAction result;
  result.kind = action.kind;
  result.anchor = physicalAnchor->first;
  result.before = physicalAnchor->second ||
                  physicalAnchor->first->hasTrait<OpTrait::IsTerminator>();
  if (action.guard != CanonicalSyncActionGuardKind::None) {
    if (!action.guardScope ||
        *action.guardScope >= program.getScopeBindings().size()) {
      return std::nullopt;
    }
    result.guardLoop = dyn_cast_or_null<scf::ForOp>(
        program.getScopeBindings()[*action.guardScope].owner);
    if (!result.guardLoop) {
      return std::nullopt;
    }
    result.guard = action.guard;
  } else if (action.guardScope) {
    return std::nullopt;
  }
  if (action.kind == CanonicalSyncActionKind::Barrier) {
    const std::optional<PipelineType> pipe =
        resolveBarrierPipe(action, allResources);
    if (!pipe) {
      return std::nullopt;
    }
    result.barrier = *pipe;
    return result;
  }
  if (!action.eventUse) {
    return std::nullopt;
  }
  const AllocationKey key{mechanism.id, *action.eventUse};
  const auto allocation = allocations.find(key);
  const CanonicalSyncEventUse &use =
      mechanism.descriptor.eventUses[*action.eventUse];
  const bool hasAllocation = allocation != allocations.end();
  const bool hasEventLane =
      hasAllocation && action.eventLane < allocation->second.size();
  const bool hasDomain = use.domain < problem.getDomains().size();
  if (!hasEventLane || !hasDomain) {
    return std::nullopt;
  }
  const CanonicalSyncEventDomain &domain = problem.getDomains()[use.domain];
  result.source = static_cast<PipelineType>(domain.sourceResource);
  result.target = static_cast<PipelineType>(domain.targetResource);
  if (action.eventLaneKind == CanonicalSyncEventLaneKind::LoopIterationModulo) {
    if (!action.eventLaneScope ||
        *action.eventLaneScope >= program.getScopeBindings().size() ||
        allocation->second.size() != use.width) {
      return std::nullopt;
    }
    result.eventLaneLoop = dyn_cast_or_null<scf::ForOp>(
        program.getScopeBindings()[*action.eventLaneScope].owner);
    if (!result.eventLaneLoop) {
      return std::nullopt;
    }
    result.eventIds = allocation->second;
  } else {
    if (action.eventLaneScope) {
      return std::nullopt;
    }
    result.eventId = allocation->second[action.eventLane];
  }
  return isPhysicalEventPipe(result.source) &&
                 isPhysicalEventPipe(result.target)
             ? std::optional<ConcreteAction>(result)
             : std::nullopt;
}

std::optional<std::vector<ActionGroup>>
stageActions(const CanonicalSyncProgram &program,
             const CanonicalSyncPatternProblem &problem,
             const CanonicalSyncVerifiedPlan &plan) {
  const auto allocations = indexAllocations(problem, plan);
  if (!allocations) {
    return std::nullopt;
  }
  const std::vector<std::uint32_t> allResources =
      getIssueResources(program.getGraph());
  std::vector<ActionGroup> groups;
  std::map<std::pair<Operation *, bool>, std::size_t> groupIds;
  const auto appendAction = [&](const ConcreteAction &action) {
    const auto key = std::make_pair(action.anchor, action.before);
    auto [position, inserted] = groupIds.emplace(key, groups.size());
    if (inserted) {
      groups.push_back({action.anchor, action.before, {}});
    }
    groups[position->second].actions.push_back(action);
  };
  for (CanonicalSyncMechanismId mechanismId : plan.mechanisms) {
    if (mechanismId >= problem.getMechanisms().size()) {
      return std::nullopt;
    }
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    for (const CanonicalSyncSupplyBinding &binding :
         mechanism.descriptor.supplies) {
      const bool hasSource =
          binding.edge.source < program.getNodeBindings().size();
      const bool hasTarget =
          binding.edge.target < program.getNodeBindings().size();
      if (!hasSource || !hasTarget) {
        return std::nullopt;
      }
      const Operation *source =
          program.getNodeBindings()[binding.edge.source].operation;
      const Operation *target =
          program.getNodeBindings()[binding.edge.target].operation;
      if (binding.edge.distance == 0 && source == target) {
        return std::nullopt;
      }
    }
    for (const CanonicalSyncAction &action : mechanism.descriptor.actions) {
      const auto concrete = makeConcreteAction(
          program, problem, mechanism, action, *allocations, allResources);
      if (!concrete) {
        return std::nullopt;
      }
      appendAction(*concrete);
    }
  }
  SmallVector<func::ReturnOp, 4> returns;
  program.getFunction().walk(
      [&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  for (func::ReturnOp returnOp : returns) {
    ConcreteAction tail;
    tail.kind = CanonicalSyncActionKind::Barrier;
    tail.anchor = returnOp;
    tail.before = true;
    tail.barrier = PipelineType::PIPE_ALL;
    tail.tailFence = true;
    appendAction(tail);
  }
  return groups;
}

void markGenerated(Operation *operation, Builder &builder) {
  operation->setAttr("pto.canonical_sync", builder.getUnitAttr());
}

bool isGenerated(Operation *operation) {
  return operation &&
         isa_and_nonnull<UnitAttr>(operation->getAttr("pto.canonical_sync"));
}

SmallVector<Operation *> collectGeneratedRoots(func::FuncOp function) {
  SmallVector<Operation *> roots;
  function.walk([&](Operation *operation) {
    if (!isGenerated(operation)) {
      return;
    }
    bool ownedAncestor = false;
    for (Operation *parent = operation->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      ownedAncestor = ownedAncestor || isGenerated(parent);
    }
    if (!ownedAncestor) {
      roots.push_back(operation);
    }
  });
  return roots;
}

void emitPhysicalAction(IRRewriter &rewriter, func::FuncOp function,
                        const ConcreteAction &action) {
  if (action.kind == CanonicalSyncActionKind::Barrier) {
    Operation *created =
        rewriter
            .create<BarrierOp>(action.anchor->getLoc(),
                               getPipeAttr(rewriter, action.barrier))
            .getOperation();
    markGenerated(created, rewriter);
    if (action.tailFence) {
      created->setAttr("pto.auto_sync_tail_barrier", rewriter.getUnitAttr());
      if (StringAttr hint =
              function->getAttrOfType<StringAttr>("pto.auto_sync_tail_hint")) {
        created->setAttr("pto.auto_sync_tail_hint", hint);
      }
    }
    return;
  }
  const PipeAttr source = getPipeAttr(rewriter, action.source);
  const PipeAttr target = getPipeAttr(rewriter, action.target);
  if (action.eventLaneLoop) {
    const Location location = action.anchor->getLoc();
    scf::ForOp loop = action.eventLaneLoop;
    Value offset = rewriter.create<arith::SubIOp>(
        location, loop.getInductionVar(), loop.getLowerBound());
    markGenerated(offset.getDefiningOp(), rewriter);
    Value ordinal =
        rewriter.create<arith::DivUIOp>(location, offset, loop.getStep());
    markGenerated(ordinal.getDefiningOp(), rewriter);
    Value width = rewriter.create<arith::ConstantIndexOp>(
        location, action.eventIds.size());
    markGenerated(width.getDefiningOp(), rewriter);
    Value lane = rewriter.create<arith::RemUIOp>(location, ordinal, width);
    markGenerated(lane.getDefiningOp(), rewriter);
    Value selected = rewriter.create<arith::ConstantIndexOp>(
        location, action.eventIds.front());
    markGenerated(selected.getDefiningOp(), rewriter);
    for (std::size_t index = 1; index < action.eventIds.size(); ++index) {
      Value candidate =
          rewriter.create<arith::ConstantIndexOp>(location, index);
      markGenerated(candidate.getDefiningOp(), rewriter);
      Value matches = rewriter.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, lane, candidate);
      markGenerated(matches.getDefiningOp(), rewriter);
      Value event = rewriter.create<arith::ConstantIndexOp>(
          location, action.eventIds[index]);
      markGenerated(event.getDefiningOp(), rewriter);
      selected =
          rewriter.create<arith::SelectOp>(location, matches, event, selected);
      markGenerated(selected.getDefiningOp(), rewriter);
    }
    Operation *created =
        action.kind == CanonicalSyncActionKind::EventSet
            ? rewriter.create<SetFlagDynOp>(location, source, target, selected)
                  .getOperation()
            : rewriter.create<WaitFlagDynOp>(location, source, target, selected)
                  .getOperation();
    markGenerated(created, rewriter);
    return;
  }
  const EventAttr event = getEventAttr(rewriter, action.eventId);
  Operation *created = action.kind == CanonicalSyncActionKind::EventSet
                           ? rewriter
                                 .create<SetFlagOp>(action.anchor->getLoc(),
                                                    source, target, event)
                                 .getOperation()
                           : rewriter
                                 .create<WaitFlagOp>(action.anchor->getLoc(),
                                                     source, target, event)
                                 .getOperation();
  markGenerated(created, rewriter);
}

void emitGuardedActions(IRRewriter &rewriter, func::FuncOp function,
                        ArrayRef<ConcreteAction> actions) {
  assert(!actions.empty() &&
         actions.front().guard != CanonicalSyncActionGuardKind::None);
  const ConcreteAction &representative = actions.front();
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  const Location location = representative.anchor->getLoc();
  scf::ForOp guardLoop = representative.guardLoop;
  Value condition;
  switch (representative.guard) {
  case CanonicalSyncActionGuardKind::LoopNonEmpty:
    condition = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, guardLoop.getLowerBound(),
        guardLoop.getUpperBound());
    break;
  case CanonicalSyncActionGuardKind::LoopEmpty:
    condition = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::sge, guardLoop.getLowerBound(),
        guardLoop.getUpperBound());
    break;
  case CanonicalSyncActionGuardKind::NotFirstIteration:
    condition = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ne, guardLoop.getInductionVar(),
        guardLoop.getLowerBound());
    break;
  case CanonicalSyncActionGuardKind::HasSuccessor: {
    const Value next = rewriter.create<arith::AddIOp>(
        location, guardLoop.getInductionVar(), guardLoop.getStep());
    markGenerated(next.getDefiningOp(), rewriter);
    condition = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, next, guardLoop.getUpperBound());
    break;
  }
  case CanonicalSyncActionGuardKind::None:
    return;
  }
  markGenerated(condition.getDefiningOp(), rewriter);
  scf::IfOp guard = rewriter.create<scf::IfOp>(location, TypeRange{}, condition,
                                               /*withElseRegion=*/false);
  markGenerated(guard, rewriter);
  rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
  for (const ConcreteAction &action : actions) {
    assert(action.guard == representative.guard &&
           action.guardLoop == guardLoop);
    emitPhysicalAction(rewriter, function, action);
  }
}

std::vector<CanonicalSyncMechanismId>
firstConflictCore(const CanonicalSyncSelection &selection) {
  for (const CanonicalSyncDomainAllocation &domain :
       selection.allocation.domains) {
    if (domain.required > domain.available) {
      return domain.liveMechanisms;
    }
  }
  return {};
}

struct SelectionOutcome {
  CanonicalSyncSelection selection;
  CanonicalSyncSelectionError preciseError = CanonicalSyncSelectionError::None;
  CanonicalSyncGreedyStatistics preciseSearch;
  CanonicalSyncResourceAllocation preciseAllocation;
  std::optional<CanonicalSyncVerifiedPlan> verifiedPlan;
  std::unique_ptr<CanonicalSyncPatternProblem> ownedProblem;
  const CanonicalSyncPatternProblem *selectedProblem = nullptr;
  bool feasible = false;
  bool fatalConstructionError = false;
  bool repairAttempted = false;
  bool repairSearchExhausted = false;
  bool repairFrontierTruncated = false;
  bool repairBudgetExhausted = false;
  bool backstopDeletionTruncated = false;
  std::size_t repairRounds = 0;
  std::size_t repairCatalogRebuilds = 0;
  std::size_t firstRepairCatalogRebuildWorkUnits = 0;
  std::size_t repairTrials = 0;
  std::size_t repairWorkUnits = 0;
  std::size_t backstopDeletionTrials = 0;
  std::size_t backstopDeletionWorkUnits = 0;
  std::uint64_t selectionNanoseconds = 0;
  std::uint64_t repairNanoseconds = 0;

  const CanonicalSyncPatternProblem &getProblem() const {
    return ownedProblem ? *ownedProblem : *selectedProblem;
  }
};

class RepairBudget {
public:
  RepairBudget(std::size_t maximumTrials, std::size_t maximumWorkUnits)
      : maximumTrials_(maximumTrials), workBudget_(maximumWorkUnits) {}

  std::optional<CanonicalSyncSelection>
  run(const CanonicalSyncPatternProblem &problem,
      CanonicalSyncGreedyOptions options) {
    const std::size_t remainingWork =
        workBudget_.maximumWorkUnits - workBudget_.workUnits;
    if (trials_ == maximumTrials_ || remainingWork == 0) {
      workBudget_.exhausted = true;
      return std::nullopt;
    }
    options.maximumWorkUnits =
        std::min(options.maximumWorkUnits, remainingWork);
    ++trials_;
    CanonicalSyncSelection selection =
        selectCanonicalSyncPatterns(problem, std::move(options));
    if (!workBudget_.consume(
            std::min(selection.statistics.workUnits, remainingWork)) ||
        selection.error == CanonicalSyncSelectionError::WorkLimitExceeded) {
      workBudget_.exhausted = true;
    }
    return selection;
  }

  CanonicalSyncVerifiedPlan verify(const CanonicalSyncPatternProblem &problem,
                                   const CanonicalSyncSelection &selection) {
    const std::size_t remainingWork =
        workBudget_.maximumWorkUnits - workBudget_.workUnits;
    if (remainingWork == 0) {
      workBudget_.exhausted = true;
      CanonicalSyncVerifiedPlan result;
      result.error = CanonicalSyncSelectionError::WorkLimitExceeded;
      return result;
    }
    CanonicalSyncVerifiedPlan result =
        verifyCanonicalSyncSelection(problem, selection, &workBudget_);
    return result;
  }

  SyncCoverCoverageWorkBudget *sharedWorkBudget() { return &workBudget_; }
  bool exhausted() const { return workBudget_.exhausted; }
  std::size_t trials() const { return trials_; }
  std::size_t workUnits() const { return workBudget_.workUnits; }

private:
  std::size_t maximumTrials_ = 0;
  SyncCoverCoverageWorkBudget workBudget_;
  std::size_t trials_ = 0;
};

template <typename T>
bool stableSortAndUniqueRepairValues(std::vector<T> &values,
                                     SyncCoverCoverageWorkBudget *workBudget) {
  const bool needsSorting = values.size() > 1;
  if (needsSorting) {
    if (workBudget && !workBudget->consume(values.size())) {
      return false;
    }
    std::vector<T> scratch(values.size());
    for (std::size_t width = 1; width < values.size();) {
      if (workBudget && !workBudget->consume(values.size())) {
        return false;
      }
      for (std::size_t begin = 0; begin < values.size();) {
        const std::size_t middle =
            begin + std::min(width, values.size() - begin);
        const std::size_t end =
            middle + std::min(width, values.size() - middle);
        std::size_t left = begin;
        std::size_t right = middle;
        std::size_t output = begin;
        while (left < middle && right < end) {
          if (workBudget && !workBudget->consume()) {
            return false;
          }
          if (values[right] < values[left]) {
            scratch[output++] = values[right++];
          } else {
            scratch[output++] = values[left++];
          }
        }
        while (left < middle) {
          scratch[output++] = values[left++];
        }
        while (right < end) {
          scratch[output++] = values[right++];
        }
        begin = end;
      }
      values.swap(scratch);
      const std::size_t remainingWidth = values.size() - width;
      if (width >= remainingWidth) {
        break;
      }
      width *= 2;
    }
  }
  if (workBudget && !workBudget->consume(values.size())) {
    return false;
  }
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return true;
}

template <typename T>
std::optional<bool>
meteredRepairContains(ArrayRef<T> values, const T &value,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t begin = 0;
  std::size_t end = values.size();
  while (begin < end) {
    if (workBudget && !workBudget->consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (values[middle] < value) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (workBudget && !workBudget->consume()) {
    return std::nullopt;
  }
  return begin != values.size() && values[begin] == value;
}

template <typename T>
std::optional<bool>
meteredRepairEqual(ArrayRef<T> first, ArrayRef<T> second,
                   SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t comparisonWork = 0;
  const bool comparisonWorkAvailable =
      checkedWorkSum(first.size(), second.size(), comparisonWork) &&
      (!workBudget || workBudget->consume(comparisonWork));
  if (!comparisonWorkAvailable) {
    return std::nullopt;
  }
  return first == second;
}

std::optional<std::size_t>
meteredResourceOverflow(const CanonicalSyncSelection &selection,
                        SyncCoverCoverageWorkBudget *workBudget) {
  if (workBudget && !workBudget->consume(selection.allocation.domains.size())) {
    return std::nullopt;
  }
  std::size_t result = 0;
  for (const CanonicalSyncDomainAllocation &domain :
       selection.allocation.domains) {
    if (domain.required <= domain.available) {
      continue;
    }
    const std::size_t overflow = domain.required - domain.available;
    if (!checkedWorkSum(result, overflow, result)) {
      result = std::numeric_limits<std::size_t>::max();
    }
  }
  return result;
}

std::optional<std::vector<CanonicalSyncMechanismId>>
meteredConflictCore(const CanonicalSyncSelection &selection,
                    SyncCoverCoverageWorkBudget *workBudget) {
  if (workBudget && !workBudget->consume(selection.allocation.domains.size())) {
    return std::nullopt;
  }
  std::size_t liveMechanisms = 0;
  for (const CanonicalSyncDomainAllocation &domain :
       selection.allocation.domains) {
    if (domain.required > domain.available &&
        !checkedWorkSum(liveMechanisms, domain.liveMechanisms.size(),
                        liveMechanisms)) {
      return std::nullopt;
    }
  }
  std::size_t copyWork = 0;
  const bool copyWorkAvailable =
      checkedWorkSum(selection.allocation.domains.size(), liveMechanisms,
                     copyWork) &&
      (!workBudget || workBudget->consume(copyWork));
  if (!copyWorkAvailable) {
    return std::nullopt;
  }
  std::vector<CanonicalSyncMechanismId> result;
  result.reserve(liveMechanisms);
  for (const CanonicalSyncDomainAllocation &domain :
       selection.allocation.domains) {
    if (domain.required > domain.available) {
      result.insert(result.end(), domain.liveMechanisms.begin(),
                    domain.liveMechanisms.end());
    }
  }
  if (!stableSortAndUniqueRepairValues(result, workBudget)) {
    return std::nullopt;
  }
  return result;
}

struct VerifiedRepairCandidate {
  CanonicalSyncSelection selection;
  CanonicalSyncVerifiedPlan plan;
};

bool consumesEvent(const CanonicalSyncPatternProblem &problem,
                   CanonicalSyncMechanismId mechanism) {
  if (mechanism >= problem.getMechanisms().size()) {
    return false;
  }
  return !problem.getMechanisms()[mechanism].descriptor.eventUses.empty();
}

SelectionOutcome
selectWithBoundedRepair(const CanonicalSyncProgram &program,
                        const CanonicalSyncPatternProblem &problem,
                        const CanonicalSyncBuildOptions &options) {
  CanonicalSyncGreedyOptions current = options.selection;
  const SteadyClock::time_point selectionStart = SteadyClock::now();
  CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(problem, current);
  SelectionOutcome outcome;
  outcome.selectionNanoseconds = elapsedNanoseconds(selectionStart);
  outcome.selectedProblem = &problem;
  outcome.preciseError = selection.error;
  outcome.preciseSearch = selection.statistics;
  outcome.preciseAllocation = selection.allocation;
  if (selection ||
      selection.error != CanonicalSyncSelectionError::ResourceInfeasible ||
      !options.patterns.enableConflictCoreRepair ||
      options.maximumRepairRounds == 0) {
    outcome.feasible = static_cast<bool>(selection);
    outcome.selection = std::move(selection);
    return outcome;
  }

  const SteadyClock::time_point repairStart = SteadyClock::now();
  RepairBudget budget(options.maximumRepairTrials,
                      options.maximumRepairWorkUnits);
  outcome.repairAttempted = true;
  std::optional<std::vector<CanonicalSyncMechanismId>> meteredInitialCore =
      meteredConflictCore(selection, budget.sharedWorkBudget());
  if (!meteredInitialCore) {
    outcome.selection = std::move(selection);
    outcome.repairSearchExhausted = true;
    outcome.repairBudgetExhausted = budget.exhausted();
    outcome.repairTrials = budget.trials();
    outcome.repairWorkUnits = budget.workUnits();
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    return outcome;
  }
  std::vector<CanonicalSyncMechanismId> initialCore =
      std::move(*meteredInitialCore);
  if (initialCore.empty()) {
    outcome.selection = std::move(selection);
    outcome.fatalConstructionError = true;
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    program.getFunction().emitError(
        "canonical sync allocation pressure has no valid conflict core");
    return outcome;
  }
  CanonicalSyncProblemBuildResult repair = buildCanonicalSyncRepairProblem(
      program, problem, options, initialCore, selection.mechanisms,
      budget.sharedWorkBudget());
  if (!repair) {
    outcome.selection = std::move(selection);
    outcome.fatalConstructionError = !budget.exhausted();
    outcome.repairSearchExhausted = budget.exhausted();
    outcome.repairBudgetExhausted = budget.exhausted();
    outcome.repairTrials = budget.trials();
    outcome.repairWorkUnits = budget.workUnits();
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    return outcome;
  }
  outcome.ownedProblem = std::move(repair.problem);
  outcome.selectedProblem = outcome.ownedProblem.get();
  outcome.repairFrontierTruncated =
      outcome.ownedProblem->getPatternStatistics().repairFrontierTruncated;
  std::map<CanonicalSyncMechanismId, std::vector<CanonicalSyncMechanismId>>
      repairMechanismsByOwner = std::move(repair.repairMechanismsByOwner);
  std::map<CanonicalSyncMechanismId, std::vector<SyncCoverDemandId>>
      repairCriticalDemandsByOwner =
          std::move(repair.repairCriticalDemandsByOwner);
  std::vector<CanonicalSyncMechanismId> collectiveRepairMechanisms =
      std::move(repair.collectiveRepairMechanisms);
  const std::size_t preciseMechanismCount = problem.getMechanisms().size();
  std::vector<const std::vector<CanonicalSyncMechanismId> *>
      repairMechanismsByOwnerIndex;
  std::vector<const std::vector<SyncCoverDemandId> *>
      repairCriticalDemandsByOwnerIndex;
  std::vector<CanonicalSyncMechanismId> allRepairMechanisms;
  const auto prepareRepairMechanismIndex =
      [&](const std::map<CanonicalSyncMechanismId,
                         std::vector<CanonicalSyncMechanismId>>
              &mechanismsByOwner,
          const std::map<CanonicalSyncMechanismId,
                         std::vector<SyncCoverDemandId>> &demandsByOwner,
          ArrayRef<CanonicalSyncMechanismId> collectiveMechanisms,
          std::vector<const std::vector<CanonicalSyncMechanismId> *>
              &mechanismIndex,
          std::vector<const std::vector<SyncCoverDemandId> *> &demandIndex,
          std::vector<CanonicalSyncMechanismId> &allMechanisms) {
        std::size_t mapEntries = 0;
        std::size_t mapScanWork = 0;
        const bool mapScanWorkAvailable =
            checkedWorkSum(mechanismsByOwner.size(), demandsByOwner.size(),
                           mapEntries) &&
            checkedWorkProduct(mapEntries, 2, mapScanWork) &&
            budget.sharedWorkBudget()->consume(mapScanWork);
        if (!mapScanWorkAvailable) {
          return false;
        }
        std::size_t ownerMechanismEntries = 0;
        for (const auto &[owner, mechanisms] : mechanismsByOwner) {
          if (owner >= preciseMechanismCount ||
              !checkedWorkSum(ownerMechanismEntries, mechanisms.size(),
                              ownerMechanismEntries)) {
            return false;
          }
        }
        for (const auto &[owner, demands] : demandsByOwner) {
          (void)demands;
          if (owner >= preciseMechanismCount) {
            return false;
          }
        }
        std::size_t indexEntries = 0;
        std::size_t copiedMechanisms = 0;
        std::size_t allocationWork = 0;
        const bool allocationWorkAvailable =
            checkedWorkProduct(preciseMechanismCount, 2, indexEntries) &&
            checkedWorkSum(collectiveMechanisms.size(), ownerMechanismEntries,
                           copiedMechanisms) &&
            checkedWorkSum(indexEntries, copiedMechanisms, allocationWork) &&
            budget.sharedWorkBudget()->consume(allocationWork);
        if (!allocationWorkAvailable) {
          return false;
        }
        mechanismIndex.assign(preciseMechanismCount, nullptr);
        demandIndex.assign(preciseMechanismCount, nullptr);
        allMechanisms.clear();
        allMechanisms.reserve(copiedMechanisms);
        allMechanisms.insert(allMechanisms.end(), collectiveMechanisms.begin(),
                             collectiveMechanisms.end());
        for (const auto &[owner, mechanisms] : mechanismsByOwner) {
          mechanismIndex[owner] = &mechanisms;
          allMechanisms.insert(allMechanisms.end(), mechanisms.begin(),
                               mechanisms.end());
        }
        for (const auto &[owner, demands] : demandsByOwner) {
          demandIndex[owner] = &demands;
        }
        return stableSortAndUniqueRepairValues(allMechanisms,
                                               budget.sharedWorkBudget());
      };
  if (!prepareRepairMechanismIndex(
          repairMechanismsByOwner, repairCriticalDemandsByOwner,
          collectiveRepairMechanisms, repairMechanismsByOwnerIndex,
          repairCriticalDemandsByOwnerIndex, allRepairMechanisms)) {
    outcome.selection = std::move(selection);
    outcome.repairSearchExhausted = true;
    outcome.repairBudgetExhausted = true;
    outcome.repairTrials = budget.trials();
    outcome.repairWorkUnits = budget.workUnits();
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    return outcome;
  }
  // The repair catalog is never an unrestricted replacement catalog. Preserve
  // the precise pressure result and expose its extra mechanisms only in a
  // trial that forbids at least one live conflicting event.
  std::size_t initialCoreCopyWork = 0;
  const bool initialCoreCopyWorkAvailable =
      checkedWorkProduct(initialCore.size(), 2, initialCoreCopyWork) &&
      budget.sharedWorkBudget()->consume(initialCoreCopyWork);
  if (!initialCoreCopyWorkAvailable) {
    outcome.selection = std::move(selection);
    outcome.repairSearchExhausted = true;
    outcome.repairBudgetExhausted = true;
    outcome.repairTrials = budget.trials();
    outcome.repairWorkUnits = budget.workUnits();
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    return outcome;
  }
  CanonicalSyncSelection lastPressure = std::move(selection);
  std::vector<CanonicalSyncMechanismId> activeCatalogCore = initialCore;
  std::vector<CanonicalSyncMechanismId> activeLivePreciseCore = initialCore;
  std::vector<CanonicalSyncMechanismId> core = std::move(initialCore);
  std::vector<CanonicalSyncMechanismId> forbiddenRepairOwners;
  std::vector<CanonicalSyncMechanismId> forcedRepairExclusions;
  bool collectiveRepairEnabled = false;

  for (std::size_t round = 1; round <= options.maximumRepairRounds; ++round) {
    outcome.repairRounds = round;
    if (!budget.sharedWorkBudget()->consume(core.size())) {
      break;
    }
    std::vector<CanonicalSyncMechanismId> preciseCore;
    llvm::copy_if(core, std::back_inserter(preciseCore),
                  [&](CanonicalSyncMechanismId mechanism) {
                    return mechanism < preciseMechanismCount;
                  });
    if (!stableSortAndUniqueRepairValues(preciseCore,
                                         budget.sharedWorkBudget())) {
      break;
    }
    std::size_t catalogCoreEntries = 0;
    const bool catalogCoreWorkAvailable =
        checkedWorkSum(forbiddenRepairOwners.size(), preciseCore.size(),
                       catalogCoreEntries) &&
        budget.sharedWorkBudget()->consume(catalogCoreEntries);
    if (!catalogCoreWorkAvailable) {
      break;
    }
    std::vector<CanonicalSyncMechanismId> catalogCore = forbiddenRepairOwners;
    catalogCore.insert(catalogCore.end(), preciseCore.begin(),
                       preciseCore.end());
    if (!stableSortAndUniqueRepairValues(catalogCore,
                                         budget.sharedWorkBudget())) {
      break;
    }
    const std::optional<bool> sameCatalogCore = meteredRepairEqual(
        ArrayRef<CanonicalSyncMechanismId>(catalogCore),
        ArrayRef<CanonicalSyncMechanismId>(activeCatalogCore),
        budget.sharedWorkBudget());
    const std::optional<bool> sameLivePreciseCore = meteredRepairEqual(
        ArrayRef<CanonicalSyncMechanismId>(preciseCore),
        ArrayRef<CanonicalSyncMechanismId>(activeLivePreciseCore),
        budget.sharedWorkBudget());
    if (!sameCatalogCore || !sameLivePreciseCore) {
      break;
    }
    const bool catalogChanged =
        !preciseCore.empty() && (!*sameCatalogCore || !*sameLivePreciseCore);
    if (catalogChanged) {
      const bool rebuildWorkAvailable =
          budget.sharedWorkBudget()->consume(lastPressure.mechanisms.size()) &&
          budget.sharedWorkBudget()->consume(catalogCore.size());
      if (!rebuildWorkAvailable) {
        break;
      }
      std::vector<CanonicalSyncMechanismId> preciseSelectedMechanisms;
      llvm::copy_if(lastPressure.mechanisms,
                    std::back_inserter(preciseSelectedMechanisms),
                    [&](CanonicalSyncMechanismId mechanism) {
                      return mechanism < preciseMechanismCount;
                    });
      std::vector<CanonicalSyncRepairCriticalDemandSeed>
          retainedCriticalDemands;
      retainedCriticalDemands.reserve(catalogCore.size());
      for (CanonicalSyncMechanismId owner : catalogCore) {
        const bool hasRetainedCriticalDemands =
            owner < repairCriticalDemandsByOwnerIndex.size() &&
            repairCriticalDemandsByOwnerIndex[owner];
        if (hasRetainedCriticalDemands) {
          retainedCriticalDemands.push_back(
              {owner, *repairCriticalDemandsByOwnerIndex[owner]});
        }
      }
      CanonicalSyncProblemBuildResult nextRepair =
          buildCanonicalSyncRepairProblem(
              program, problem, options, catalogCore, preciseSelectedMechanisms,
              budget.sharedWorkBudget(), retainedCriticalDemands);
      if (!nextRepair) {
        outcome.fatalConstructionError = !budget.exhausted();
        break;
      }
      auto nextMechanismsByOwner =
          std::move(nextRepair.repairMechanismsByOwner);
      auto nextCriticalDemandsByOwner =
          std::move(nextRepair.repairCriticalDemandsByOwner);
      std::vector<CanonicalSyncMechanismId> nextCollectiveMechanisms =
          std::move(nextRepair.collectiveRepairMechanisms);
      std::vector<const std::vector<CanonicalSyncMechanismId> *>
          nextMechanismIndex;
      std::vector<const std::vector<SyncCoverDemandId> *> nextDemandIndex;
      std::vector<CanonicalSyncMechanismId> nextAllRepairMechanisms;
      if (!prepareRepairMechanismIndex(
              nextMechanismsByOwner, nextCriticalDemandsByOwner,
              nextCollectiveMechanisms, nextMechanismIndex, nextDemandIndex,
              nextAllRepairMechanisms)) {
        break;
      }
      std::size_t nextOptionWork = 0;
      const bool nextOptionWorkAvailable =
          checkedWorkSum(options.selection.forbiddenMechanisms.size(),
                         forbiddenRepairOwners.size(), nextOptionWork) &&
          budget.sharedWorkBudget()->consume(nextOptionWork);
      if (!nextOptionWorkAvailable) {
        break;
      }
      CanonicalSyncGreedyOptions nextCurrent = options.selection;
      nextCurrent.forbiddenMechanisms.insert(
          nextCurrent.forbiddenMechanisms.end(), forbiddenRepairOwners.begin(),
          forbiddenRepairOwners.end());
      const std::vector<CanonicalSyncMechanismId> nextForcedExclusions;
      if (!prepareCanonicalSyncRepairTrial(
              nextCurrent, nextAllRepairMechanisms, nextMechanismIndex,
              nextCollectiveMechanisms, forbiddenRepairOwners,
              collectiveRepairEnabled, nextForcedExclusions,
              budget.sharedWorkBudget())) {
        break;
      }
      if (!budget.sharedWorkBudget()->consume(
              nextCurrent.forbiddenMechanisms.size())) {
        break;
      }
      std::optional<CanonicalSyncSelection> rebuiltPressure =
          budget.run(*nextRepair.problem, nextCurrent);
      if (!rebuiltPressure) {
        break;
      }
      std::optional<CanonicalSyncVerifiedPlan> rebuiltVerified;
      std::optional<std::vector<CanonicalSyncMechanismId>> rebuiltCore;
      if (*rebuiltPressure) {
        CanonicalSyncVerifiedPlan verified =
            budget.verify(*nextRepair.problem, *rebuiltPressure);
        if (verified) {
          rebuiltVerified = std::move(verified);
        } else if (!budget.exhausted()) {
          outcome.fatalConstructionError = true;
        }
      } else if (rebuiltPressure->error ==
                 CanonicalSyncSelectionError::ResourceInfeasible) {
        rebuiltCore =
            meteredConflictCore(*rebuiltPressure, budget.sharedWorkBudget());
      } else {
        outcome.fatalConstructionError = !budget.exhausted();
      }
      const bool validRebuiltResult = rebuiltVerified || rebuiltCore;
      if (!validRebuiltResult) {
        break;
      }
      outcome.ownedProblem.swap(nextRepair.problem);
      repairMechanismsByOwner.swap(nextMechanismsByOwner);
      repairCriticalDemandsByOwner.swap(nextCriticalDemandsByOwner);
      collectiveRepairMechanisms.swap(nextCollectiveMechanisms);
      repairMechanismsByOwnerIndex = std::move(nextMechanismIndex);
      repairCriticalDemandsByOwnerIndex = std::move(nextDemandIndex);
      allRepairMechanisms = std::move(nextAllRepairMechanisms);
      activeCatalogCore = std::move(catalogCore);
      activeLivePreciseCore = std::move(preciseCore);
      current = std::move(nextCurrent);
      forcedRepairExclusions.clear();
      outcome.selectedProblem = outcome.ownedProblem.get();
      outcome.repairFrontierTruncated |=
          outcome.ownedProblem->getPatternStatistics().repairFrontierTruncated;
      if (outcome.repairCatalogRebuilds == 0) {
        outcome.firstRepairCatalogRebuildWorkUnits = budget.workUnits();
      }
      ++outcome.repairCatalogRebuilds;
      if (rebuiltVerified) {
        outcome.selection = std::move(*rebuiltPressure);
        outcome.verifiedPlan = std::move(*rebuiltVerified);
        outcome.feasible = true;
        break;
      }
      lastPressure = std::move(*rebuiltPressure);
      core = std::move(*rebuiltCore);
    }
    std::optional<VerifiedRepairCandidate> bestVerified;
    std::optional<CanonicalSyncSelection> bestPressureTrial;
    CanonicalSyncGreedyOptions bestPressureOptions;
    std::vector<CanonicalSyncMechanismId> bestPressureOwners;
    std::vector<CanonicalSyncMechanismId> bestPressureRepairExclusions;
    bool bestPressureUsesCollective = false;
    const std::optional<std::size_t> overflow =
        meteredResourceOverflow(lastPressure, budget.sharedWorkBudget());
    if (!overflow || !budget.sharedWorkBudget()->consume(core.size())) {
      break;
    }
    CanonicalSyncRepairRoundRanker ranker(options.selection.objective,
                                          *overflow);
    std::vector<CanonicalSyncMechanismId> replaceableCore;
    llvm::copy_if(core, std::back_inserter(replaceableCore),
                  [&](CanonicalSyncMechanismId mechanism) {
                    return consumesEvent(*outcome.ownedProblem, mechanism);
                  });
    // Try owners with a certified repair alternative first. Stable ordering
    // within each class keeps the search deterministic while avoiding work on
    // exclusions that can only reshuffle the same over-pressured event plan.
    if (!budget.sharedWorkBudget()->consume(replaceableCore.size())) {
      break;
    }
    std::vector<CanonicalSyncMechanismId> withoutOwnerRepair;
    withoutOwnerRepair.reserve(replaceableCore.size());
    std::size_t ownerRepairPosition = 0;
    for (CanonicalSyncMechanismId mechanism : replaceableCore) {
      const bool hasOwnerRepair =
          mechanism < repairMechanismsByOwnerIndex.size() &&
          repairMechanismsByOwnerIndex[mechanism];
      if (hasOwnerRepair) {
        replaceableCore[ownerRepairPosition++] = mechanism;
      } else {
        withoutOwnerRepair.push_back(mechanism);
      }
    }
    replaceableCore.resize(ownerRepairPosition);
    replaceableCore.insert(replaceableCore.end(), withoutOwnerRepair.begin(),
                           withoutOwnerRepair.end());
    const auto considerTrial =
        [&](CanonicalSyncGreedyOptions trialOptions,
            ArrayRef<CanonicalSyncMechanismId> trialOwners,
            ArrayRef<CanonicalSyncMechanismId> trialRepairExclusions,
            bool trialUsesCollective) {
          if (!budget.sharedWorkBudget()->consume(
                  trialOptions.forbiddenMechanisms.size())) {
            return;
          }
          std::optional<CanonicalSyncSelection> trial =
              budget.run(*outcome.ownedProblem, trialOptions);
          if (!trial) {
            return;
          }
          std::optional<CanonicalSyncVerifiedPlan> plan;
          if (static_cast<bool>(*trial)) {
            CanonicalSyncVerifiedPlan verified =
                budget.verify(*outcome.ownedProblem, *trial);
            if (verified) {
              plan = std::move(verified);
            }
          }
          std::size_t trialOverflow = 0;
          if (!plan) {
            const std::optional<std::size_t> diagnosedOverflow =
                meteredResourceOverflow(*trial, budget.sharedWorkBudget());
            if (!diagnosedOverflow) {
              return;
            }
            trialOverflow = *diagnosedOverflow;
          }
          const CanonicalSyncRepairRoundRanker::Decision decision =
              ranker.consider(*trial, plan.has_value(), trialOverflow,
                              budget.sharedWorkBudget());
          if (decision ==
              CanonicalSyncRepairRoundRanker::Decision::WorkLimitExceeded) {
            return;
          }
          if (decision ==
              CanonicalSyncRepairRoundRanker::Decision::ReplaceBestVerified) {
            bestVerified =
                VerifiedRepairCandidate{std::move(*trial), std::move(*plan)};
          } else if (decision == CanonicalSyncRepairRoundRanker::Decision::
                                     ReplaceBestPressure) {
            std::size_t retainedTrialWork = 0;
            const bool retainedTrialWorkAvailable =
                checkedWorkSum(trialOwners.size(), trialRepairExclusions.size(),
                               retainedTrialWork) &&
                budget.sharedWorkBudget()->consume(retainedTrialWork);
            if (!retainedTrialWorkAvailable) {
              return;
            }
            bestPressureTrial = std::move(*trial);
            bestPressureOptions = std::move(trialOptions);
            bestPressureOwners.assign(trialOwners.begin(), trialOwners.end());
            bestPressureRepairExclusions.assign(trialRepairExclusions.begin(),
                                                trialRepairExclusions.end());
            bestPressureUsesCollective = trialUsesCollective;
          }
        };
    // Required search: forbid one live conflicting event at a time from the
    // current pressure baseline.
    for (CanonicalSyncMechanismId mechanism : replaceableCore) {
      std::size_t trialCopyWork = 0;
      const bool trialCopyWorkAvailable =
          checkedWorkSum(current.forbiddenMechanisms.size(),
                         forbiddenRepairOwners.size(), trialCopyWork) &&
          checkedWorkSum(trialCopyWork, forcedRepairExclusions.size(),
                         trialCopyWork) &&
          checkedWorkSum(trialCopyWork, 1, trialCopyWork) &&
          budget.sharedWorkBudget()->consume(trialCopyWork);
      if (!trialCopyWorkAvailable) {
        break;
      }
      CanonicalSyncGreedyOptions trialOptions = current;
      trialOptions.forbiddenMechanisms.push_back(mechanism);
      std::vector<CanonicalSyncMechanismId> trialOwners = forbiddenRepairOwners;
      std::vector<CanonicalSyncMechanismId> trialRepairExclusions =
          forcedRepairExclusions;
      if (mechanism < preciseMechanismCount) {
        trialOwners.push_back(mechanism);
      } else {
        trialRepairExclusions.push_back(mechanism);
      }
      const bool trialPrepared =
          stableSortAndUniqueRepairValues(trialOwners,
                                          budget.sharedWorkBudget()) &&
          stableSortAndUniqueRepairValues(trialRepairExclusions,
                                          budget.sharedWorkBudget()) &&
          prepareCanonicalSyncRepairTrial(
              trialOptions, allRepairMechanisms, repairMechanismsByOwnerIndex,
              collectiveRepairMechanisms, trialOwners, collectiveRepairEnabled,
              trialRepairExclusions, budget.sharedWorkBudget());
      if (!trialPrepared) {
        break;
      }
      considerTrial(std::move(trialOptions), trialOwners, trialRepairExclusions,
                    collectiveRepairEnabled);
      if (budget.exhausted()) {
        break;
      }
    }
    // Optional acceleration: try replacing the entire live core, but never
    // allow an uncoverable/work-limited collective trial to replace the last
    // valid resource-pressure diagnosis.
    const bool runCollectiveRepair =
        !budget.exhausted() && !replaceableCore.empty() &&
        options.patterns.enableCollectiveRepairTrial;
    if (runCollectiveRepair) {
      std::size_t replaceableCopies = 0;
      std::size_t collectiveCopyWork = 0;
      const bool collectiveCopyWorkAvailable =
          checkedWorkProduct(replaceableCore.size(), 3, replaceableCopies) &&
          checkedWorkSum(current.forbiddenMechanisms.size(),
                         forbiddenRepairOwners.size(), collectiveCopyWork) &&
          checkedWorkSum(collectiveCopyWork, forcedRepairExclusions.size(),
                         collectiveCopyWork) &&
          checkedWorkSum(collectiveCopyWork, replaceableCopies,
                         collectiveCopyWork) &&
          budget.sharedWorkBudget()->consume(collectiveCopyWork);
      if (!collectiveCopyWorkAvailable) {
        break;
      }
      CanonicalSyncGreedyOptions collectiveOptions = current;
      std::vector<CanonicalSyncMechanismId> collectiveOwners =
          forbiddenRepairOwners;
      std::vector<CanonicalSyncMechanismId> collectiveRepairExclusions =
          forcedRepairExclusions;
      llvm::copy_if(replaceableCore, std::back_inserter(collectiveOwners),
                    [&](CanonicalSyncMechanismId mechanism) {
                      return mechanism < preciseMechanismCount;
                    });
      llvm::copy_if(replaceableCore,
                    std::back_inserter(collectiveRepairExclusions),
                    [&](CanonicalSyncMechanismId mechanism) {
                      return mechanism >= preciseMechanismCount;
                    });
      collectiveOptions.forbiddenMechanisms.insert(
          collectiveOptions.forbiddenMechanisms.end(), replaceableCore.begin(),
          replaceableCore.end());
      const bool collectivePrepared =
          stableSortAndUniqueRepairValues(collectiveOwners,
                                          budget.sharedWorkBudget()) &&
          stableSortAndUniqueRepairValues(collectiveRepairExclusions,
                                          budget.sharedWorkBudget()) &&
          prepareCanonicalSyncRepairTrial(
              collectiveOptions, allRepairMechanisms,
              repairMechanismsByOwnerIndex, collectiveRepairMechanisms,
              collectiveOwners, true, collectiveRepairExclusions,
              budget.sharedWorkBudget());
      if (collectivePrepared) {
        considerTrial(std::move(collectiveOptions), collectiveOwners,
                      collectiveRepairExclusions, true);
      }
    }
    if (bestVerified) {
      outcome.selection = std::move(bestVerified->selection);
      outcome.verifiedPlan = std::move(bestVerified->plan);
      outcome.feasible = true;
      outcome.repairRounds = round;
      break;
    }
    const bool cannotContinue = budget.exhausted() || !bestPressureTrial;
    if (cannotContinue) {
      break;
    }
    std::optional<std::vector<CanonicalSyncMechanismId>> nextCore =
        meteredConflictCore(*bestPressureTrial, budget.sharedWorkBudget());
    if (!nextCore) {
      break;
    }
    lastPressure = std::move(*bestPressureTrial);
    current = std::move(bestPressureOptions);
    forbiddenRepairOwners = std::move(bestPressureOwners);
    forcedRepairExclusions = std::move(bestPressureRepairExclusions);
    collectiveRepairEnabled = bestPressureUsesCollective;
    core = std::move(*nextCore);
  }
  if (!outcome.feasible) {
    outcome.selection = std::move(lastPressure);
  }
  outcome.repairBudgetExhausted = budget.exhausted();
  outcome.repairSearchExhausted = !outcome.feasible;
  outcome.repairTrials = budget.trials();
  outcome.repairWorkUnits = budget.workUnits();
  outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
  return outcome;
}

bool canUseLocalizedPipeAllBackstop(const SelectionOutcome &outcome) {
  return !outcome.feasible && !outcome.fatalConstructionError &&
         outcome.selection.error ==
             CanonicalSyncSelectionError::ResourceInfeasible &&
         outcome.selection.allocation.valid &&
         !outcome.selection.allocation.feasible &&
         !firstConflictCore(outcome.selection).empty() &&
         outcome.repairAttempted && outcome.repairSearchExhausted;
}

std::size_t
countPredictedSyncInstructions(const CanonicalSyncProgram &program,
                               const CanonicalSyncPatternProblem &problem,
                               const CanonicalSyncVerifiedPlan &plan) {
  std::size_t result = 0;
  for (CanonicalSyncMechanismId mechanism : plan.mechanisms) {
    const std::size_t actions =
        problem.getMechanisms()[mechanism].descriptor.actions.size();
    const bool actionCountOverflows =
        actions > std::numeric_limits<std::size_t>::max() - result;
    if (actionCountOverflows) {
      return std::numeric_limits<std::size_t>::max();
    }
    result += actions;
  }
  program.getFunction().walk([&](func::ReturnOp) {
    if (result != std::numeric_limits<std::size_t>::max()) {
      ++result;
    }
  });
  return result;
}

void hashPlanValue(std::uint64_t &hash, std::uint64_t value) {
  constexpr std::uint64_t hashPrime = 1099511628211ULL;
  for (unsigned byte = 0; byte < sizeof(value); ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffU;
    hash *= hashPrime;
  }
}

std::uint64_t computePlanSignature(const CanonicalSyncPatternProblem &problem,
                                   const CanonicalSyncVerifiedPlan &plan) {
  std::uint64_t hash = 1469598103934665603ULL;
  hashPlanValue(hash, plan.mechanisms.size());
  for (CanonicalSyncMechanismId mechanism : plan.mechanisms) {
    hashPlanValue(hash, mechanism);
    hashPlanValue(hash, problem.getMechanismSignature(mechanism));
  }
  hashPlanValue(hash, plan.allocation.domains.size());
  for (const CanonicalSyncDomainAllocation &domain : plan.allocation.domains) {
    hashPlanValue(hash, domain.domain);
    hashPlanValue(hash, domain.required);
    hashPlanValue(hash, domain.available);
    hashPlanValue(hash, domain.maximumPressurePoint.value_or(
                            std::numeric_limits<std::size_t>::max()));
    for (CanonicalSyncMechanismId mechanism : domain.liveMechanisms) {
      hashPlanValue(hash, mechanism);
    }
    for (const CanonicalSyncEventAllocation &use : domain.uses) {
      hashPlanValue(hash, use.mechanism);
      hashPlanValue(hash, use.eventUse);
      for (unsigned id : use.ids) {
        hashPlanValue(hash, id);
      }
    }
  }
  return hash;
}

bool consumeStagingVerificationWork(const CanonicalSyncProgram &program,
                                    const CanonicalSyncPatternProblem &problem,
                                    const CanonicalSyncVerifiedPlan &plan,
                                    SyncCoverCoverageWorkBudget &work) {
  const bool setupWorkUnavailable =
      !work.consume(plan.mechanisms.size()) ||
      !work.consume(plan.allocation.domains.size());
  if (setupWorkUnavailable) {
    return false;
  }
  for (CanonicalSyncMechanismId mechanism : plan.mechanisms) {
    const CanonicalSyncMechanismDescriptor &descriptor =
        problem.getMechanisms()[mechanism].descriptor;
    const bool descriptorWorkUnavailable =
        !work.consume(descriptor.supplies.size()) ||
        !work.consume(descriptor.actions.size()) ||
        !work.consume(descriptor.eventUses.size());
    if (descriptorWorkUnavailable) {
      return false;
    }
  }
  for (const CanonicalSyncDomainAllocation &domain : plan.allocation.domains) {
    const bool domainWorkUnavailable =
        !work.consume(domain.liveMechanisms.size()) ||
        !work.consume(domain.uses.size());
    if (domainWorkUnavailable) {
      return false;
    }
    for (const CanonicalSyncEventAllocation &use : domain.uses) {
      if (!work.consume(use.ids.size())) {
        return false;
      }
    }
  }
  std::size_t returns = 0;
  program.getFunction().walk([&](func::ReturnOp) {
    if (returns != std::numeric_limits<std::size_t>::max()) {
      ++returns;
    }
  });
  return work.consume(returns);
}

struct FreshVerificationResult {
  CanonicalSyncVerifiedPlan plan;
  std::size_t workUnits = 0;
  std::uint64_t nanoseconds = 0;
};

FreshVerificationResult
freshlyVerifySelection(const CanonicalSyncProgram &program,
                       const CanonicalSyncPatternProblem &problem,
                       const CanonicalSyncSelection &selection,
                       std::size_t maximumWorkUnits) {
  const SteadyClock::time_point verificationStart = SteadyClock::now();
  SyncCoverCoverageWorkBudget work(maximumWorkUnits);
  FreshVerificationResult result;
  result.plan = verifyCanonicalSyncSelection(problem, selection, &work);
  if (result.plan &&
      (!consumeStagingVerificationWork(program, problem, result.plan, work) ||
       !stageActions(program, problem, result.plan))) {
    result.plan.error =
        work.exhausted ? CanonicalSyncSelectionError::WorkLimitExceeded
                       : CanonicalSyncSelectionError::FinalValidationFailed;
  }
  result.workUnits = work.workUnits;
  result.nanoseconds = elapsedNanoseconds(verificationStart);
  return result;
}

CanonicalSyncStrategyReport
buildStrategyReport(CanonicalSyncSelectionStrategy strategy,
                    const CanonicalSyncProgram &program,
                    const SelectionOutcome &outcome,
                    const FreshVerificationResult &verification,
                    bool usedLocalizedPipeAll = false) {
  const CanonicalSyncPatternProblem &problem = outcome.getProblem();
  CanonicalSyncStrategyReport report;
  report.strategy = strategy;
  report.error = outcome.selection.error;
  report.preciseError = outcome.preciseError;
  report.verificationError = verification.plan.error;
  report.verified = static_cast<bool>(verification.plan);
  report.usedLocalizedPipeAll = usedLocalizedPipeAll;
  report.repairFrontierTruncated = outcome.repairFrontierTruncated;
  report.repairBudgetExhausted = outcome.repairBudgetExhausted;
  report.backstopDeletionTruncated = outcome.backstopDeletionTruncated;
  report.repairRounds = outcome.repairRounds;
  report.repairCatalogRebuilds = outcome.repairCatalogRebuilds;
  report.firstRepairCatalogRebuildWorkUnits =
      outcome.firstRepairCatalogRebuildWorkUnits;
  report.repairTrials = outcome.repairTrials;
  report.repairWorkUnits = outcome.repairWorkUnits;
  report.backstopDeletionTrials = outcome.backstopDeletionTrials;
  report.backstopDeletionWorkUnits = outcome.backstopDeletionWorkUnits;
  report.selectionNanoseconds = outcome.selectionNanoseconds;
  report.repairNanoseconds = outcome.repairNanoseconds;
  report.verificationNanoseconds = verification.nanoseconds;
  report.verificationWorkUnits = verification.workUnits;
  report.cost = outcome.selection.cost;
  report.search = outcome.selection.statistics;
  report.allocation = verification.plan ? verification.plan.allocation
                                        : outcome.selection.allocation;
  report.preciseSearch = outcome.preciseSearch;
  report.preciseAllocation = outcome.preciseAllocation;
  constexpr std::size_t maximumSelectedMechanismDetails = 4096;
  for (CanonicalSyncMechanismId mechanismId : outcome.selection.mechanisms) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[mechanismId];
    const CanonicalSyncMechanismDescriptor &descriptor = mechanism.descriptor;
    for (std::size_t origin = 0; origin < kCanonicalSyncMechanismOriginCount;
         ++origin) {
      const auto originKind = static_cast<CanonicalSyncMechanismOrigin>(origin);
      if ((mechanism.originMask &
           canonicalSyncMechanismOriginBit(originKind)) != 0) {
        ++report.selectedMechanismsByOrigin[origin];
      }
    }
    CanonicalSyncSelectedMechanismReport detail;
    detail.mechanism = mechanismId;
    detail.kind = descriptor.kind;
    detail.originMask = mechanism.originMask;
    detail.supplies = descriptor.supplies.size();
    detail.eventUses = descriptor.eventUses.size();
    detail.actions = descriptor.actions.size();
    const bool hasRecurrenceSupply =
        llvm::any_of(descriptor.supplies, [](const auto &binding) {
          return binding.edge.distance != 0;
        });
    const bool hasZeroDistanceSupply =
        llvm::any_of(descriptor.supplies, [](const auto &binding) {
          return binding.edge.distance == 0;
        });
    const bool isTargetLocalPipeDrain =
        llvm::any_of(descriptor.supplies, [](const auto &binding) {
          return binding.proof ==
                     CanonicalSyncSupplyProof::TargetLocalPipeDrainAction ||
                 binding.proof ==
                     CanonicalSyncSupplyProof::DominatingTargetedDrainCut;
        });
    const bool isLoopCarryPipeDrain =
        llvm::any_of(descriptor.supplies, [](const auto &binding) {
          return binding.proof == CanonicalSyncSupplyProof::LoopCarryPipeDrain;
        });
    const bool isSourceLocalPipeDrain =
        llvm::any_of(descriptor.supplies, [](const auto &binding) {
          return binding.proof ==
                 CanonicalSyncSupplyProof::SourceLocalPipeDrainAction;
        });
    const bool isSourcePrefixPipeDrain =
        llvm::any_of(descriptor.supplies, [](const auto &binding) {
          return binding.proof ==
                 CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
        });
    for (const CanonicalSyncSupplyBinding &binding : descriptor.supplies) {
      detail.maximumRecurrenceDistance =
          std::max(detail.maximumRecurrenceDistance, binding.edge.distance);
    }
    for (const CanonicalSyncAction &action : descriptor.actions) {
      if (action.kind == CanonicalSyncActionKind::EventSet) {
        ++report.emittedEventSets;
        ++detail.eventSets;
      } else if (action.kind == CanonicalSyncActionKind::EventWait) {
        ++report.emittedEventWaits;
        ++detail.eventWaits;
      } else if (action.barrierKind == CanonicalSyncBarrierKind::All) {
        ++report.emittedPipeAllBarriers;
        ++detail.pipeAllBarriers;
      } else {
        ++report.emittedTargetedBarriers;
        ++detail.targetedBarriers;
        if (hasRecurrenceSupply) {
          ++report.emittedRecurrenceTargetedBarriers;
        } else {
          ++report.emittedZeroDistanceTargetedBarriers;
        }
        if (hasZeroDistanceSupply && hasRecurrenceSupply) {
          ++report.emittedMixedDistanceTargetedBarriers;
        } else if (hasRecurrenceSupply) {
          ++report.emittedRecurrenceOnlyTargetedBarriers;
        } else {
          ++report.emittedZeroOnlyTargetedBarriers;
        }
        if (isTargetLocalPipeDrain) {
          ++report.emittedTargetLocalPipeDrainBarriers;
        }
        if (isLoopCarryPipeDrain) {
          ++report.emittedLoopCarryPipeDrainBarriers;
        }
        if (isSourceLocalPipeDrain) {
          ++report.emittedSourceLocalPipeDrainBarriers;
        }
        if (isSourcePrefixPipeDrain) {
          ++report.emittedSourcePrefixPipeDrainBarriers;
        }
      }
    }
    if (report.selectedMechanisms.size() < maximumSelectedMechanismDetails) {
      report.selectedMechanisms.push_back(detail);
    } else {
      report.selectedMechanismDetailsTruncated = true;
    }
    if (descriptor.kind != CanonicalSyncMechanismKind::Barrier) {
      ++report.selectedEvents;
      continue;
    }
    const bool usesPipeAll = std::any_of(
        descriptor.actions.begin(), descriptor.actions.end(),
        [](const CanonicalSyncAction &action) {
          return action.kind == CanonicalSyncActionKind::Barrier &&
                 action.barrierKind == CanonicalSyncBarrierKind::All;
        });
    if (usesPipeAll) {
      ++report.selectedPipeAllBarriers;
    } else {
      ++report.selectedTargetedBarriers;
    }
  }
  for (const CanonicalSyncPattern &pattern : problem.getPatterns()) {
    if (pattern.kind != CanonicalSyncPatternKind::DirectPair ||
        !llvm::all_of(pattern.members, [&](CanonicalSyncMechanismId member) {
          return std::binary_search(outcome.selection.mechanisms.begin(),
                                    outcome.selection.mechanisms.end(), member);
        })) {
      continue;
    }
    ++report.activeDirectPairs;
    const std::size_t extra = pattern.coverage.count();
    if (extra > std::numeric_limits<std::size_t>::max() -
                    report.activeDirectPairExtraCoverage) {
      report.activeDirectPairExtraCoverage =
          std::numeric_limits<std::size_t>::max();
    } else {
      report.activeDirectPairExtraCoverage += extra;
    }
  }
  if (verification.plan) {
    report.predictedSyncInstructions =
        countPredictedSyncInstructions(program, problem, verification.plan);
    report.planSignature = computePlanSignature(problem, verification.plan);
  }
  return report;
}

CanonicalSyncComparisonReport
buildComparisonHeader(const CanonicalSyncProgram &program,
                      const CanonicalSyncPatternProblem &problem,
                      const CanonicalSyncBuildOptions &options,
                      std::uint64_t preparationNanoseconds) {
  CanonicalSyncComparisonReport report;
  report.function = program.getFunction().getSymName().str();
  report.selectionObjective = options.selection.objective;
  report.enabledMechanismFamilies = options.patterns.enabledMechanismFamilies;
  report.directPairsEnabled = options.patterns.enableDirectPairs;
  report.conflictCoreRepairEnabled = options.patterns.enableConflictCoreRepair;
  report.gmAliasPolicy = options.analysis.gmAliasPolicy;
  report.graphNodes = program.getGraph().getNodes().size();
  report.graphEdges = program.getGraph().getEdges().size();
  report.certifiedCompletionFrontiers = static_cast<std::size_t>(std::count_if(
      program.getGraph().getEdges().begin(),
      program.getGraph().getEdges().end(), [](const SyncCoverEdge &edge) {
        return edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier;
      }));
  const CanonicalSyncOwnershipDiscoveryStatistics &ownershipStatistics =
      program.getOwnershipDiscoveryStatistics();
  report.ownershipDiscoveryInspections = ownershipStatistics.inspections;
  report.ownershipCertificatesByKind = ownershipStatistics.certificatesByKind;
  report.ownershipSlots = ownershipStatistics.slots;
  report.ownershipPaths = ownershipStatistics.paths;
  report.ownershipUses = ownershipStatistics.uses;
  report.ownershipNodeReferences = ownershipStatistics.nodeReferences;
  report.ownershipAccessIncidences = ownershipStatistics.accessIncidences;
  report.ownershipDiscoveryTruncated = ownershipStatistics.truncated;
  report.uniqueDemandRows = problem.getObligationDemands().size();
  report.selectionBasisRows = problem.getDemands().size();
  report.basisReducedRows = report.uniqueDemandRows - report.selectionBasisRows;
  report.basisReductionTruncated = problem.wasBasisReductionTruncated();
  for (SyncCoverDemandId demandId : problem.getObligationDemands()) {
    const SyncCoverDemand &demand = program.getGraph().getDemands()[demandId];
    if (demand.distance == 0) {
      ++report.zeroDistanceDemandRows;
    } else {
      ++report.recurrenceDemandRows;
      report.maximumRecurrenceDistance =
          std::max(report.maximumRecurrenceDistance, demand.distance);
    }
    if (program.getGraph().getNodes()[demand.source].resource ==
        program.getGraph().getNodes()[demand.target].resource) {
      ++report.sameResourceDemandRows;
    } else {
      ++report.crossResourceDemandRows;
    }
    for (SyncCoverDemandKind kind : demand.provenanceKinds) {
      switch (kind) {
      case SyncCoverDemandKind::SSA:
        ++report.ssaDemandRows;
        break;
      case SyncCoverDemandKind::MemoryRAW:
        ++report.rawDemandRows;
        break;
      case SyncCoverDemandKind::MemoryWAR:
        ++report.warDemandRows;
        break;
      case SyncCoverDemandKind::MemoryWAW:
        ++report.wawDemandRows;
        break;
      }
    }
    const bool provenanceCountOverflows =
        demand.originalDemandCount >
        std::numeric_limits<std::size_t>::max() - report.demands;
    if (provenanceCountOverflows) {
      report.demands = std::numeric_limits<std::size_t>::max();
      break;
    }
    report.demands += demand.originalDemandCount;
  }
  report.directMechanisms = problem.getMechanisms().size();
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    for (std::size_t origin = 0; origin < kCanonicalSyncMechanismOriginCount;
         ++origin) {
      const auto originKind = static_cast<CanonicalSyncMechanismOrigin>(origin);
      if ((mechanism.originMask &
           canonicalSyncMechanismOriginBit(originKind)) != 0) {
        ++report.candidateMechanismsByOrigin[origin];
      }
    }
  }
  const CanonicalSyncPatternStatistics &statistics =
      problem.getPatternStatistics();
  report.directPairProposals = statistics.directPairProposals;
  report.directPairEvaluations = statistics.directPairEvaluations;
  report.synergisticPairs =
      statistics.get(CanonicalSyncPatternKind::DirectPair).patterns;
  report.pairGenerationTruncated = problem.wasPatternGenerationTruncated();
  report.sourcePrefixInspections = statistics.sourcePrefixInspections;
  report.sourcePrefixCandidates = statistics.sourcePrefixCandidates;
  report.sourcePrefixIncidences = statistics.sourcePrefixIncidences;
  report.sourcePrefixGenerationTruncated =
      statistics.sourcePrefixGenerationTruncated;
  report.loopCarryInspections = statistics.loopCarryInspections;
  report.loopCarryCandidates = statistics.loopCarryCandidates;
  report.loopCarryIncidences = statistics.loopCarryIncidences;
  report.loopCarryGenerationTruncated = statistics.loopCarryGenerationTruncated;
  report.loopBoundaryProtocolInspections =
      statistics.loopBoundaryProtocolInspections;
  report.loopBoundaryProtocolCandidates =
      statistics.loopBoundaryProtocolCandidates;
  report.loopBoundaryProtocolIncidences =
      statistics.loopBoundaryProtocolIncidences;
  report.loopBoundaryProtocolGenerationTruncated =
      statistics.loopBoundaryProtocolGenerationTruncated;
  report.preparationNanoseconds = preparationNanoseconds;
  return report;
}

struct PipeAllFallbackOutcome {
  std::unique_ptr<CanonicalSyncPatternProblem> problem;
  CanonicalSyncVerifiedPlan plan;
  bool deletionTruncated = false;
  std::size_t deletionTrials = 0;
  std::size_t deletionWorkUnits = 0;
};

std::optional<PipeAllFallbackOutcome>
buildLocalizedPipeAllFallback(const CanonicalSyncProgram &program,
                              const CanonicalSyncBuildOptions &options) {
  CanonicalSyncProblemBuildResult built =
      buildCanonicalSyncPipeAllProblem(program, options);
  if (!built) {
    return std::nullopt;
  }
  CanonicalSyncPatternProblem &problem = *built.problem;
  CanonicalSyncSelection selection;
  for (const CanonicalSyncMechanism &mechanism : problem.getMechanisms()) {
    if (mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier) {
      selection.mechanisms.push_back(mechanism.id);
    }
  }
  CanonicalSyncVerifiedPlan plan =
      verifyCanonicalSyncSelection(problem, selection);
  if (!plan) {
    return std::nullopt;
  }
  SyncCoverCoverageWorkBudget cleanupBudget(
      options.maximumBackstopDeletionWorkUnits);
  std::vector<CanonicalSyncMechanismId> deletionOrder;
  std::size_t deletionTrials = 0;
  bool deletionTruncated = false;
  if (!selection.mechanisms.empty()) {
    const bool trialLimitReached = options.maximumBackstopDeletionTrials == 0;
    const bool setupWorkUnavailable =
        !trialLimitReached &&
        !cleanupBudget.consume(selection.mechanisms.size());
    if (trialLimitReached || setupWorkUnavailable) {
      deletionTruncated = true;
    } else {
      deletionOrder = selection.mechanisms;
    }
  }
  for (auto position = deletionOrder.rbegin(); position != deletionOrder.rend();
       ++position) {
    const bool trialLimitReached =
        deletionTrials == options.maximumBackstopDeletionTrials;
    const bool workLimitReached =
        cleanupBudget.exhausted ||
        cleanupBudget.workUnits == cleanupBudget.maximumWorkUnits;
    if (trialLimitReached || workLimitReached) {
      deletionTruncated = true;
      break;
    }
    ++deletionTrials;
    const bool trialSetupUnavailable = !cleanupBudget.consume(
        selection.mechanisms.empty() ? 1 : selection.mechanisms.size());
    if (trialSetupUnavailable) {
      deletionTruncated = true;
      break;
    }
    CanonicalSyncSelection trial = selection;
    if (!cleanupBudget.consume(
            trial.mechanisms.empty() ? 1 : trial.mechanisms.size())) {
      deletionTruncated = true;
      break;
    }
    const auto found = std::lower_bound(trial.mechanisms.begin(),
                                        trial.mechanisms.end(), *position);
    const bool mechanismMissing =
        found == trial.mechanisms.end() || *found != *position;
    if (mechanismMissing) {
      continue;
    }
    const bool eraseWorkUnavailable =
        !cleanupBudget.consume(trial.mechanisms.size());
    if (eraseWorkUnavailable) {
      deletionTruncated = true;
      break;
    }
    trial.mechanisms.erase(found);
    const CanonicalSyncVerifiedPlan verified =
        verifyCanonicalSyncSelection(problem, trial, &cleanupBudget);
    if (cleanupBudget.exhausted) {
      deletionTruncated = true;
      break;
    }
    if (verified) {
      selection = std::move(trial);
      plan = verified;
    }
  }
  return PipeAllFallbackOutcome{std::move(built.problem), std::move(plan),
                                deletionTruncated, deletionTrials,
                                cleanupBudget.workUnits};
}

SelectionOutcome takeFallbackSelection(PipeAllFallbackOutcome fallback,
                                       const SelectionOutcome &failed) {
  SelectionOutcome outcome;
  outcome.ownedProblem = std::move(fallback.problem);
  outcome.selectedProblem = outcome.ownedProblem.get();
  outcome.feasible = true;
  outcome.preciseError = failed.selection.error;
  outcome.preciseSearch = failed.selection.statistics;
  outcome.preciseAllocation = failed.selection.allocation;
  outcome.verifiedPlan = fallback.plan;
  outcome.repairAttempted = failed.repairAttempted;
  outcome.repairSearchExhausted = failed.repairSearchExhausted;
  outcome.repairFrontierTruncated = failed.repairFrontierTruncated;
  outcome.repairBudgetExhausted = failed.repairBudgetExhausted;
  outcome.repairRounds = failed.repairRounds;
  outcome.repairCatalogRebuilds = failed.repairCatalogRebuilds;
  outcome.firstRepairCatalogRebuildWorkUnits =
      failed.firstRepairCatalogRebuildWorkUnits;
  outcome.repairTrials = failed.repairTrials;
  outcome.repairWorkUnits = failed.repairWorkUnits;
  outcome.selection.mechanisms = fallback.plan.mechanisms;
  outcome.selection.allocation = fallback.plan.allocation;
  const std::optional<CanonicalSyncStructuralCost> cost =
      computeCanonicalSyncStructuralCost(outcome.getProblem(),
                                         outcome.selection.mechanisms);
  if (!cost) {
    outcome.selection.error = CanonicalSyncSelectionError::ArithmeticOverflow;
    outcome.feasible = false;
    outcome.fatalConstructionError = true;
  } else {
    outcome.selection.cost = *cost;
  }
  outcome.backstopDeletionTruncated = fallback.deletionTruncated;
  outcome.backstopDeletionTrials = fallback.deletionTrials;
  outcome.backstopDeletionWorkUnits = fallback.deletionWorkUnits;
  outcome.selectionNanoseconds = failed.selectionNanoseconds;
  outcome.repairNanoseconds = failed.repairNanoseconds;
  return outcome;
}

} // namespace

bool mlir::pto::prepareCanonicalSyncRepairTrial(
    CanonicalSyncGreedyOptions &trialOptions,
    ArrayRef<CanonicalSyncMechanismId> allRepairMechanisms,
    ArrayRef<const std::vector<CanonicalSyncMechanismId> *>
        repairMechanismsByOwner,
    ArrayRef<CanonicalSyncMechanismId> collectiveRepairMechanisms,
    ArrayRef<CanonicalSyncMechanismId> owners, bool collective,
    ArrayRef<CanonicalSyncMechanismId> forcedRepairExclusions,
    SyncCoverCoverageWorkBudget *workBudget) {
  bool invalidRepairOrder = false;
  std::optional<CanonicalSyncMechanismId> previousRepair;
  for (CanonicalSyncMechanismId mechanism : allRepairMechanisms) {
    if (workBudget && !workBudget->consume()) {
      return false;
    }
    invalidRepairOrder |= previousRepair && *previousRepair >= mechanism;
    previousRepair = mechanism;
  }
  if (invalidRepairOrder) {
    return false;
  }

  std::vector<CanonicalSyncMechanismId> allowed;
  for (CanonicalSyncMechanismId owner : owners) {
    if (workBudget && !workBudget->consume()) {
      return false;
    }
    const bool ownerHasNoRepairs = owner >= repairMechanismsByOwner.size() ||
                                   !repairMechanismsByOwner[owner];
    if (ownerHasNoRepairs) {
      continue;
    }
    const std::vector<CanonicalSyncMechanismId> &ownerMechanisms =
        *repairMechanismsByOwner[owner];
    if (workBudget && !workBudget->consume(ownerMechanisms.size())) {
      return false;
    }
    allowed.insert(allowed.end(), ownerMechanisms.begin(),
                   ownerMechanisms.end());
  }
  if (collective) {
    if (workBudget && !workBudget->consume(collectiveRepairMechanisms.size())) {
      return false;
    }
    allowed.insert(allowed.end(), collectiveRepairMechanisms.begin(),
                   collectiveRepairMechanisms.end());
  }
  const bool allowedPrepared =
      stableSortAndUniqueRepairValues(allowed, workBudget) &&
      (!workBudget ||
       workBudget->consume(trialOptions.forbiddenMechanisms.size()));
  if (!allowedPrepared) {
    return false;
  }

  std::size_t retainedCapacity = 0;
  const bool retainedCapacityAvailable =
      checkedWorkSum(trialOptions.forbiddenMechanisms.size(),
                     forcedRepairExclusions.size(), retainedCapacity) &&
      (!workBudget || workBudget->consume(forcedRepairExclusions.size()));
  if (!retainedCapacityAvailable) {
    return false;
  }
  std::vector<CanonicalSyncMechanismId> retainedForbidden;
  retainedForbidden.reserve(retainedCapacity);
  for (CanonicalSyncMechanismId mechanism : trialOptions.forbiddenMechanisms) {
    const std::optional<bool> isRepair =
        meteredRepairContains(allRepairMechanisms, mechanism, workBudget);
    if (!isRepair) {
      return false;
    }
    if (!*isRepair) {
      retainedForbidden.push_back(mechanism);
    }
  }
  for (CanonicalSyncMechanismId mechanism : forcedRepairExclusions) {
    const std::optional<bool> isRepair =
        meteredRepairContains(allRepairMechanisms, mechanism, workBudget);
    if (!isRepair || !*isRepair) {
      return false;
    }
    retainedForbidden.push_back(mechanism);
  }
  trialOptions.forbiddenMechanisms = std::move(retainedForbidden);

  const bool lookupWorkAvailable =
      !workBudget || (workBudget->consume(allRepairMechanisms.size()) &&
                      workBudget->consume(allowed.size()));
  if (!lookupWorkAvailable) {
    return false;
  }
  std::size_t allowedPosition = 0;
  for (CanonicalSyncMechanismId mechanism : allRepairMechanisms) {
    bool hasEarlierAllowed = allowedPosition < allowed.size() &&
                             allowed[allowedPosition] < mechanism;
    while (hasEarlierAllowed) {
      ++allowedPosition;
      hasEarlierAllowed = allowedPosition < allowed.size() &&
                          allowed[allowedPosition] < mechanism;
    }
    const bool mechanismAllowed = allowedPosition < allowed.size() &&
                                  allowed[allowedPosition] == mechanism;
    if (!mechanismAllowed) {
      trialOptions.forbiddenMechanisms.push_back(mechanism);
    }
  }
  return stableSortAndUniqueRepairValues(trialOptions.forbiddenMechanisms,
                                         workBudget);
}

LogicalResult mlir::pto::materializeCanonicalSyncPlan(
    const CanonicalSyncProgram &program,
    const CanonicalSyncPatternProblem &problem,
    const CanonicalSyncVerifiedPlan &plan) {
  if (!plan || !plan.allocation.valid || !plan.allocation.feasible) {
    return program.getFunction().emitError(
        "cannot materialize an unverified canonical sync plan");
  }
  const auto groups = stageActions(program, problem, plan);
  if (!groups) {
    return program.getFunction().emitError(
        "canonical sync plan has an invalid physical recipe");
  }
  IRRewriter rewriter(program.getFunction().getContext());
  SmallVector<Operation *> generatedRoots =
      collectGeneratedRoots(program.getFunction());
  for (Operation *operation : llvm::reverse(generatedRoots)) {
    rewriter.eraseOp(operation);
  }
  for (const ActionGroup &group : *groups) {
    if (group.before) {
      rewriter.setInsertionPoint(group.anchor);
    } else {
      rewriter.setInsertionPointAfter(group.anchor);
    }
    for (std::size_t begin = 0; begin < group.actions.size();) {
      const ConcreteAction &action = group.actions[begin];
      if (action.guard == CanonicalSyncActionGuardKind::None) {
        emitPhysicalAction(rewriter, program.getFunction(), action);
        ++begin;
        continue;
      }
      std::size_t end = begin + 1;
      while (end < group.actions.size() &&
             group.actions[end].guard == action.guard &&
             group.actions[end].guardLoop == action.guardLoop) {
        ++end;
      }
      emitGuardedActions(
          rewriter, program.getFunction(),
          ArrayRef<ConcreteAction>(group.actions).slice(begin, end - begin));
      begin = end;
    }
  }
  return success();
}

LogicalResult
mlir::pto::runCanonicalSync(func::FuncOp function,
                            const CanonicalSyncBuildOptions &options) {
  const SteadyClock::time_point preparationStart = SteadyClock::now();
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(function, options.analysis);
  if (failed(program)) {
    return failure();
  }
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(*program, options);
  if (!precise) {
    InFlightDiagnostic diagnostic =
        function.emitError() << "cannot build canonical sync precise "
                                "problem, error="
                             << static_cast<unsigned>(precise.status.error);
    if (precise.problem && precise.status.index &&
        *precise.status.index < precise.problem->getDemands().size()) {
      const SyncCoverDemandId demandId =
          precise.problem->getDemands()[*precise.status.index];
      const SyncCoverDemand &demand =
          program->getGraph().getDemands()[demandId];
      const SyncCoverNode &source =
          program->getGraph().getNodes()[demand.source];
      const SyncCoverNode &target =
          program->getGraph().getNodes()[demand.target];
      const bool sameOperation =
          demand.source < program->getNodeBindings().size() &&
          demand.target < program->getNodeBindings().size() &&
          program->getNodeBindings()[demand.source].operation ==
              program->getNodeBindings()[demand.target].operation;
      diagnostic << ", first_uncovered_row=" << *precise.status.index
                 << ", source_node=" << demand.source
                 << ", target_node=" << demand.target
                 << ", source_resource=" << source.resource
                 << ", target_resource=" << target.resource
                 << ", distance=" << demand.distance
                 << ", same_macro_operation=" << sameOperation;
    }
    return failure();
  }
  CanonicalSyncComparisonReport report =
      buildComparisonHeader(*program, *precise.problem, options,
                            elapsedNanoseconds(preparationStart));
  if (options.analysisOnly || options.compareSelectionStrategies) {
    for (CanonicalSyncSelectionStrategy strategy :
         {CanonicalSyncSelectionStrategy::FixedCover,
          CanonicalSyncSelectionStrategy::ActionAwareSingleton,
          CanonicalSyncSelectionStrategy::PairLookahead}) {
      CanonicalSyncBuildOptions trialOptions = options;
      trialOptions.selection.strategy = strategy;
      SelectionOutcome outcome =
          selectWithBoundedRepair(*program, *precise.problem, trialOptions);
      if (outcome.fatalConstructionError) {
        return failure();
      }
      bool usedLocalizedPipeAll = false;
      if (canUseLocalizedPipeAllBackstop(outcome)) {
        const SteadyClock::time_point fallbackStart = SteadyClock::now();
        std::optional<PipeAllFallbackOutcome> fallback =
            buildLocalizedPipeAllFallback(*program, trialOptions);
        if (!fallback) {
          function.emitError("canonical sync comparison could not construct "
                             "its localized PIPE_ALL backstop");
          return failure();
        }
        outcome = takeFallbackSelection(std::move(*fallback), outcome);
        if (outcome.fatalConstructionError) {
          function.emitError(
              "canonical sync comparison fallback cost overflowed");
          return failure();
        }
        addNanoseconds(outcome.repairNanoseconds,
                       elapsedNanoseconds(fallbackStart));
        usedLocalizedPipeAll = true;
      }
      FreshVerificationResult verification;
      if (outcome.feasible) {
        verification = freshlyVerifySelection(
            *program, outcome.getProblem(), outcome.selection,
            trialOptions.maximumVerificationWorkUnits);
      }
      report.strategies.push_back(buildStrategyReport(
          strategy, *program, outcome, verification, usedLocalizedPipeAll));
    }
    if (options.reportCallback && failed(options.reportCallback(report))) {
      return failure();
    }
    if (options.analysisOnly) {
      return success();
    }
  }

  SelectionOutcome selection =
      selectWithBoundedRepair(*program, *precise.problem, options);
  if (selection.fatalConstructionError) {
    return failure();
  }
  CanonicalSyncProblemBuildResult fullBasis;
  if (selection.feasible) {
    FreshVerificationResult verification = freshlyVerifySelection(
        *program, selection.getProblem(), selection.selection,
        options.maximumVerificationWorkUnits);
    const bool reducedBasis =
        selection.getProblem().getDemands().size() !=
        selection.getProblem().getObligationDemands().size();
    if (!verification.plan && reducedBasis) {
      CanonicalSyncBuildOptions fullOptions = options;
      fullOptions.enableDemandBasisReduction = false;
      fullBasis = buildCanonicalSyncPreciseProblem(*program, fullOptions);
      if (!fullBasis) {
        function.emitError(
            "canonical sync could not rebuild its full demand basis");
        return failure();
      }
      selection =
          selectWithBoundedRepair(*program, *fullBasis.problem, fullOptions);
      if (selection.fatalConstructionError || !selection.feasible) {
        function.emitError(
            "canonical sync full-basis retry failed before materialization");
        return failure();
      }
      verification = freshlyVerifySelection(
          *program, selection.getProblem(), selection.selection,
          fullOptions.maximumVerificationWorkUnits);
      report =
          buildComparisonHeader(*program, selection.getProblem(), fullOptions,
                                elapsedNanoseconds(preparationStart));
    }
    if (!verification.plan) {
      function.emitError() << "canonical sync finalization failed, error="
                           << static_cast<unsigned>(verification.plan.error);
      return failure();
    }
    if (!options.compareSelectionStrategies) {
      report.strategies.push_back(buildStrategyReport(
          options.selection.strategy, *program, selection, verification));
      if (options.reportCallback && failed(options.reportCallback(report))) {
        return failure();
      }
    }
    return materializeCanonicalSyncPlan(*program, selection.getProblem(),
                                        verification.plan);
  }

  if (!canUseLocalizedPipeAllBackstop(selection)) {
    function.emitError()
        << "canonical sync precise planning failed closed, error="
        << static_cast<unsigned>(selection.selection.error);
    return failure();
  }

  for (const CanonicalSyncDomainAllocation &domain :
       selection.selection.allocation.domains) {
    if (domain.required <= domain.available) {
      continue;
    }
    const CanonicalSyncEventDomain &eventDomain =
        selection.getProblem().getDomains()[domain.domain];
    function.emitRemark() << "canonical sync event pressure: source_resource="
                          << eventDomain.sourceResource
                          << ", target_resource=" << eventDomain.targetResource
                          << ", required=" << domain.required
                          << ", available=" << domain.available
                          << ", live_mechanisms="
                          << domain.liveMechanisms.size();
  }

  const SteadyClock::time_point fallbackStart = SteadyClock::now();
  std::optional<PipeAllFallbackOutcome> fallback =
      buildLocalizedPipeAllFallback(*program, options);
  if (!fallback) {
    function.emitError(
        "canonical sync could not construct its localized PIPE_ALL backstop");
    return failure();
  }
  function.emitRemark(
      "canonical sync used localized PIPE_ALL target barriers after bounded "
      "event repair was exhausted");
  SelectionOutcome fallbackOutcome =
      takeFallbackSelection(std::move(*fallback), selection);
  if (fallbackOutcome.fatalConstructionError) {
    return function.emitError("canonical sync fallback cost overflowed");
  }
  addNanoseconds(fallbackOutcome.repairNanoseconds,
                 elapsedNanoseconds(fallbackStart));
  FreshVerificationResult verification = freshlyVerifySelection(
      *program, fallbackOutcome.getProblem(), fallbackOutcome.selection,
      options.maximumVerificationWorkUnits);
  if (!verification.plan) {
    function.emitError() << "canonical sync backstop finalization failed, "
                            "error="
                         << static_cast<unsigned>(verification.plan.error);
    return failure();
  }
  if (!options.compareSelectionStrategies) {
    report.strategies.push_back(buildStrategyReport(options.selection.strategy,
                                                    *program, fallbackOutcome,
                                                    verification, true));
    if (options.reportCallback && failed(options.reportCallback(report))) {
      return failure();
    }
  }
  return materializeCanonicalSyncPlan(*program, fallbackOutcome.getProblem(),
                                      verification.plan);
}
