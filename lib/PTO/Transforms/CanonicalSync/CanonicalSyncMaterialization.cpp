// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
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
  std::map<AllocationKey, std::vector<unsigned>> result;
  for (const CanonicalSyncDomainAllocation &domain : plan.allocation.domains) {
    if (domain.domain >= problem.getDomains().size()) {
      return std::nullopt;
    }
    const unsigned budget = problem.getDomains()[domain.domain].budget;
    for (const CanonicalSyncEventAllocation &use : domain.uses) {
      const AllocationKey key{use.mechanism, use.eventUse};
      const bool invalid =
          use.mechanism >= problem.getMechanisms().size() ||
          use.eventUse >= problem.getMechanisms()[use.mechanism]
                              .descriptor.eventUses.size() ||
          use.ids.empty() ||
          std::any_of(use.ids.begin(), use.ids.end(),
                      [&](unsigned id) { return id >= budget; });
      if (invalid || !result.emplace(key, use.ids).second) {
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

void emitAction(IRRewriter &rewriter, func::FuncOp function,
                const ConcreteAction &action) {
  if (action.guard == CanonicalSyncActionGuardKind::None) {
    emitPhysicalAction(rewriter, function, action);
    return;
  }

  OpBuilder::InsertionGuard insertionGuard(rewriter);
  const Location location = action.anchor->getLoc();
  scf::ForOp guardLoop = action.guardLoop;
  Value condition;
  switch (action.guard) {
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
  emitPhysicalAction(rewriter, function, action);
}

std::uint64_t costValue(const std::vector<std::uint64_t> &profile,
                        std::size_t depth) {
  return depth < profile.size() ? profile[depth] : 0;
}

bool structuralCostLess(const CanonicalSyncStructuralCost &first,
                        const CanonicalSyncStructuralCost &second) {
  const std::size_t depths =
      std::max(first.actionProfile.size(), second.actionProfile.size());
  for (std::size_t reverse = depths; reverse > 0; --reverse) {
    const std::size_t depth = reverse - 1;
    const std::uint64_t firstValue = costValue(first.actionProfile, depth);
    const std::uint64_t secondValue = costValue(second.actionProfile, depth);
    if (firstValue != secondValue) {
      return firstValue < secondValue;
    }
  }
  return std::tie(first.serializationBreadth, first.eventLifetimeArea,
                  first.mechanismCount) < std::tie(second.serializationBreadth,
                                                   second.eventLifetimeArea,
                                                   second.mechanismCount);
}

std::size_t resourceOverflow(const CanonicalSyncSelection &selection) {
  std::size_t result = 0;
  for (const CanonicalSyncDomainAllocation &domain :
       selection.allocation.domains) {
    if (domain.required > domain.available) {
      const std::size_t overflow = domain.required - domain.available;
      const bool accumulationOverflows =
          overflow > std::numeric_limits<std::size_t>::max() - result;
      if (accumulationOverflows) {
        return std::numeric_limits<std::size_t>::max();
      }
      result += overflow;
    }
  }
  return result;
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

std::vector<CanonicalSyncMechanismId>
allConflictCore(const CanonicalSyncSelection &selection) {
  std::vector<CanonicalSyncMechanismId> result;
  for (const CanonicalSyncDomainAllocation &domain :
       selection.allocation.domains) {
    if (domain.required > domain.available) {
      result.insert(result.end(), domain.liveMechanisms.begin(),
                    domain.liveMechanisms.end());
    }
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
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
      : maximumTrials_(maximumTrials), maximumWorkUnits_(maximumWorkUnits) {}

  std::optional<CanonicalSyncSelection>
  run(const CanonicalSyncPatternProblem &problem,
      CanonicalSyncGreedyOptions options) {
    const std::size_t remainingWork = maximumWorkUnits_ - workUnits_;
    if (trials_ == maximumTrials_ || remainingWork == 0) {
      exhausted_ = true;
      return std::nullopt;
    }
    options.maximumWorkUnits =
        std::min(options.maximumWorkUnits, remainingWork);
    ++trials_;
    CanonicalSyncSelection selection =
        selectCanonicalSyncPatterns(problem, std::move(options));
    workUnits_ += std::min(selection.statistics.workUnits, remainingWork);
    exhausted_ |=
        selection.error == CanonicalSyncSelectionError::WorkLimitExceeded;
    return selection;
  }

  bool exhausted() const { return exhausted_; }
  std::size_t trials() const { return trials_; }
  std::size_t workUnits() const { return workUnits_; }

private:
  std::size_t maximumTrials_ = 0;
  std::size_t maximumWorkUnits_ = 0;
  std::size_t trials_ = 0;
  std::size_t workUnits_ = 0;
  bool exhausted_ = false;
};

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

void considerVerifiedRepair(const CanonicalSyncPatternProblem &problem,
                            const CanonicalSyncSelection &selection,
                            std::optional<VerifiedRepairCandidate> &best) {
  if (!selection) {
    return;
  }
  CanonicalSyncVerifiedPlan plan =
      verifyCanonicalSyncSelection(problem, selection);
  if (!plan ||
      (best && !structuralCostLess(selection.cost, best->selection.cost))) {
    return;
  }
  best = VerifiedRepairCandidate{selection, std::move(plan)};
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
  const std::vector<CanonicalSyncMechanismId> initialCore =
      allConflictCore(selection);
  if (initialCore.empty()) {
    outcome.selection = std::move(selection);
    outcome.fatalConstructionError = true;
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    program.getFunction().emitError(
        "canonical sync allocation pressure has no valid conflict core");
    return outcome;
  }
  CanonicalSyncProblemBuildResult repair = buildCanonicalSyncRepairProblem(
      program, problem, options, initialCore, selection.mechanisms);
  if (!repair) {
    outcome.selection = std::move(selection);
    outcome.fatalConstructionError = true;
    outcome.repairNanoseconds = elapsedNanoseconds(repairStart);
    return outcome;
  }
  outcome.ownedProblem = std::move(repair.problem);
  outcome.selectedProblem = outcome.ownedProblem.get();
  outcome.repairAttempted = true;
  outcome.repairFrontierTruncated =
      outcome.ownedProblem->getPatternStatistics().repairFrontierTruncated;
  const auto repairMechanismsByOwner =
      std::move(repair.repairMechanismsByOwner);
  const std::vector<CanonicalSyncMechanismId> collectiveRepairMechanisms =
      std::move(repair.collectiveRepairMechanisms);
  std::vector<CanonicalSyncMechanismId> allRepairMechanisms =
      collectiveRepairMechanisms;
  for (const auto &[owner, mechanisms] : repairMechanismsByOwner) {
    (void)owner;
    allRepairMechanisms.insert(allRepairMechanisms.end(), mechanisms.begin(),
                               mechanisms.end());
  }
  llvm::sort(allRepairMechanisms);
  allRepairMechanisms.erase(
      std::unique(allRepairMechanisms.begin(), allRepairMechanisms.end()),
      allRepairMechanisms.end());
  const auto constrainRepairCatalog =
      [&](CanonicalSyncGreedyOptions &trialOptions,
          ArrayRef<CanonicalSyncMechanismId> owners, bool collective) {
        std::vector<CanonicalSyncMechanismId> allowed;
        for (CanonicalSyncMechanismId owner : owners) {
          const auto position = repairMechanismsByOwner.find(owner);
          if (position != repairMechanismsByOwner.end()) {
            allowed.insert(allowed.end(), position->second.begin(),
                           position->second.end());
          }
        }
        if (collective) {
          allowed.insert(allowed.end(), collectiveRepairMechanisms.begin(),
                         collectiveRepairMechanisms.end());
        }
        llvm::sort(allowed);
        allowed.erase(std::unique(allowed.begin(), allowed.end()),
                      allowed.end());
        trialOptions.forbiddenMechanisms.erase(
            std::remove_if(trialOptions.forbiddenMechanisms.begin(),
                           trialOptions.forbiddenMechanisms.end(),
                           [&](CanonicalSyncMechanismId mechanism) {
                             return std::binary_search(
                                 allRepairMechanisms.begin(),
                                 allRepairMechanisms.end(), mechanism);
                           }),
            trialOptions.forbiddenMechanisms.end());
        for (CanonicalSyncMechanismId mechanism : allRepairMechanisms) {
          if (!std::binary_search(allowed.begin(), allowed.end(), mechanism)) {
            trialOptions.forbiddenMechanisms.push_back(mechanism);
          }
        }
      };
  RepairBudget budget(options.maximumRepairTrials,
                      options.maximumRepairWorkUnits);
  // The repair catalog is never an unrestricted replacement catalog. Preserve
  // the precise pressure result and expose its extra mechanisms only in a
  // trial that forbids at least one live conflicting event.
  CanonicalSyncSelection lastPressure = selection;
  std::vector<CanonicalSyncMechanismId> core = initialCore;
  std::vector<CanonicalSyncMechanismId> forbiddenRepairOwners;

  for (std::size_t round = 1; round <= options.maximumRepairRounds; ++round) {
    outcome.repairRounds = round;
    std::optional<VerifiedRepairCandidate> bestVerified;
    std::optional<CanonicalSyncSelection> bestPressureTrial;
    CanonicalSyncGreedyOptions bestPressureOptions;
    std::vector<CanonicalSyncMechanismId> bestPressureOwners;
    std::vector<CanonicalSyncMechanismId> replaceableCore;
    llvm::copy_if(core, std::back_inserter(replaceableCore),
                  [&](CanonicalSyncMechanismId mechanism) {
                    return consumesEvent(*outcome.ownedProblem, mechanism);
                  });
    // Try owners with a certified repair alternative first. Stable ordering
    // within each class keeps the search deterministic while avoiding work on
    // exclusions that can only reshuffle the same over-pressured event plan.
    llvm::stable_sort(replaceableCore, [&](CanonicalSyncMechanismId first,
                                           CanonicalSyncMechanismId second) {
      const bool firstHasRepair =
          repairMechanismsByOwner.find(first) != repairMechanismsByOwner.end();
      const bool secondHasRepair =
          repairMechanismsByOwner.find(second) != repairMechanismsByOwner.end();
      return firstHasRepair > secondHasRepair;
    });
    const auto considerTrial =
        [&](CanonicalSyncGreedyOptions trialOptions,
            ArrayRef<CanonicalSyncMechanismId> trialOwners) {
          std::optional<CanonicalSyncSelection> trial =
              budget.run(*outcome.ownedProblem, trialOptions);
          if (!trial) {
            return;
          }
          considerVerifiedRepair(*outcome.ownedProblem, *trial, bestVerified);
          const bool improvesPressure =
              trial->error == CanonicalSyncSelectionError::ResourceInfeasible &&
              resourceOverflow(*trial) < resourceOverflow(lastPressure) &&
              (!bestPressureTrial ||
               resourceOverflow(*trial) < resourceOverflow(*bestPressureTrial));
          if (improvesPressure) {
            bestPressureTrial = *trial;
            bestPressureOptions = std::move(trialOptions);
            bestPressureOwners.assign(trialOwners.begin(), trialOwners.end());
          }
        };
    // Required search: forbid one live conflicting event at a time from the
    // current pressure baseline.
    for (CanonicalSyncMechanismId mechanism : replaceableCore) {
      CanonicalSyncGreedyOptions trialOptions = current;
      trialOptions.forbiddenMechanisms.push_back(mechanism);
      std::vector<CanonicalSyncMechanismId> trialOwners = forbiddenRepairOwners;
      trialOwners.push_back(mechanism);
      llvm::sort(trialOwners);
      trialOwners.erase(std::unique(trialOwners.begin(), trialOwners.end()),
                        trialOwners.end());
      constrainRepairCatalog(trialOptions, trialOwners, false);
      llvm::sort(trialOptions.forbiddenMechanisms);
      trialOptions.forbiddenMechanisms.erase(
          std::unique(trialOptions.forbiddenMechanisms.begin(),
                      trialOptions.forbiddenMechanisms.end()),
          trialOptions.forbiddenMechanisms.end());
      considerTrial(std::move(trialOptions), trialOwners);
      // A strictly smaller pressure diagnosis is the next repair baseline,
      // not another candidate to compare exhaustively in this round. Advance
      // immediately so a bounded budget can perform several local removals.
      if (bestPressureTrial || budget.exhausted()) {
        break;
      }
    }
    // Optional acceleration: try replacing the entire live core, but never
    // allow an uncoverable/work-limited collective trial to replace the last
    // valid resource-pressure diagnosis.
    if (!budget.exhausted() &&
        (!bestPressureTrial || round == options.maximumRepairRounds) &&
        !replaceableCore.empty()) {
      CanonicalSyncGreedyOptions collectiveOptions = current;
      std::vector<CanonicalSyncMechanismId> collectiveOwners =
          forbiddenRepairOwners;
      collectiveOwners.insert(collectiveOwners.end(), replaceableCore.begin(),
                              replaceableCore.end());
      llvm::sort(collectiveOwners);
      collectiveOwners.erase(
          std::unique(collectiveOwners.begin(), collectiveOwners.end()),
          collectiveOwners.end());
      collectiveOptions.forbiddenMechanisms.insert(
          collectiveOptions.forbiddenMechanisms.end(), replaceableCore.begin(),
          replaceableCore.end());
      constrainRepairCatalog(collectiveOptions, collectiveOwners, true);
      llvm::sort(collectiveOptions.forbiddenMechanisms);
      collectiveOptions.forbiddenMechanisms.erase(
          std::unique(collectiveOptions.forbiddenMechanisms.begin(),
                      collectiveOptions.forbiddenMechanisms.end()),
          collectiveOptions.forbiddenMechanisms.end());
      considerTrial(std::move(collectiveOptions), collectiveOwners);
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
    lastPressure = *bestPressureTrial;
    current = std::move(bestPressureOptions);
    forbiddenRepairOwners = std::move(bestPressureOwners);
    core = allConflictCore(lastPressure);
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
  for (CanonicalSyncMechanismId mechanismId : outcome.selection.mechanisms) {
    const CanonicalSyncMechanismDescriptor &descriptor =
        problem.getMechanisms()[mechanismId].descriptor;
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
    for (const CanonicalSyncAction &action : descriptor.actions) {
      if (action.kind == CanonicalSyncActionKind::EventSet) {
        ++report.emittedEventSets;
      } else if (action.kind == CanonicalSyncActionKind::EventWait) {
        ++report.emittedEventWaits;
      } else if (action.barrierKind == CanonicalSyncBarrierKind::All) {
        ++report.emittedPipeAllBarriers;
      } else {
        ++report.emittedTargetedBarriers;
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
                      CanonicalSyncGmAliasPolicy gmAliasPolicy,
                      std::uint64_t preparationNanoseconds) {
  CanonicalSyncComparisonReport report;
  report.function = program.getFunction().getSymName().str();
  report.gmAliasPolicy = gmAliasPolicy;
  report.graphNodes = program.getGraph().getNodes().size();
  report.graphEdges = program.getGraph().getEdges().size();
  report.certifiedCompletionFrontiers = static_cast<std::size_t>(std::count_if(
      program.getGraph().getEdges().begin(),
      program.getGraph().getEdges().end(), [](const SyncCoverEdge &edge) {
        return edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier;
      }));
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
  report.slotLifecycleInspections = statistics.slotLifecycleInspections;
  report.slotLifecycleCandidates = statistics.slotLifecycleCandidates;
  report.slotLifecycleConflictIncidences =
      statistics.slotLifecycleConflictIncidences;
  report.slotLifecycleGenerationTruncated =
      statistics.slotLifecycleGenerationTruncated;
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
  outcome.repairTrials = failed.repairTrials;
  outcome.repairWorkUnits = failed.repairWorkUnits;
  outcome.selection.mechanisms = fallback.plan.mechanisms;
  outcome.selection.allocation = fallback.plan.allocation;
  outcome.selection.cost = computeCanonicalSyncStructuralCost(
      outcome.getProblem(), outcome.selection.mechanisms);
  outcome.backstopDeletionTruncated = fallback.deletionTruncated;
  outcome.backstopDeletionTrials = fallback.deletionTrials;
  outcome.backstopDeletionWorkUnits = fallback.deletionWorkUnits;
  outcome.selectionNanoseconds = failed.selectionNanoseconds;
  outcome.repairNanoseconds = failed.repairNanoseconds;
  return outcome;
}

} // namespace

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
    for (const ConcreteAction &action : group.actions) {
      emitAction(rewriter, program.getFunction(), action);
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
  CanonicalSyncComparisonReport report = buildComparisonHeader(
      *program, *precise.problem, options.analysis.gmAliasPolicy,
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
      report = buildComparisonHeader(*program, selection.getProblem(),
                                     options.analysis.gmAliasPolicy,
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
