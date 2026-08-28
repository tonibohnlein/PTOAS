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

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

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
                              ? &block.front()
                              : &block.back();
    if (anchor.kind == SyncCoverAnchorKind::ScopeExit &&
        !position->hasTrait<OpTrait::IsTerminator>()) {
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
    Value ordinal =
        rewriter.create<arith::DivUIOp>(location, offset, loop.getStep());
    Value width = rewriter.create<arith::ConstantIndexOp>(
        location, action.eventIds.size());
    Value lane = rewriter.create<arith::RemUIOp>(location, ordinal, width);
    Value selected = rewriter.create<arith::ConstantIndexOp>(
        location, action.eventIds.front());
    for (std::size_t index = 1; index < action.eventIds.size(); ++index) {
      Value candidate =
          rewriter.create<arith::ConstantIndexOp>(location, index);
      Value matches = rewriter.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::eq, lane, candidate);
      Value event = rewriter.create<arith::ConstantIndexOp>(
          location, action.eventIds[index]);
      selected =
          rewriter.create<arith::SelectOp>(location, matches, event, selected);
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
    condition = rewriter.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::slt, next, guardLoop.getUpperBound());
    break;
  }
  case CanonicalSyncActionGuardKind::None:
    return;
  }
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

struct SelectionOutcome {
  CanonicalSyncSelection selection;
  std::optional<CanonicalSyncVerifiedPlan> verifiedPlan;
  std::unique_ptr<CanonicalSyncPatternProblem> ownedProblem;
  const CanonicalSyncPatternProblem *selectedProblem = nullptr;
  bool feasible = false;
  bool fatalConstructionError = false;
  bool repairFrontierTruncated = false;
  bool repairBudgetExhausted = false;
  bool backstopDeletionTruncated = false;
  std::size_t repairRounds = 0;
  std::size_t repairTrials = 0;
  std::size_t repairWorkUnits = 0;
  std::size_t backstopDeletionTrials = 0;
  std::size_t backstopDeletionWorkUnits = 0;

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
  CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(problem, current);
  SelectionOutcome outcome;
  outcome.selectedProblem = &problem;
  if (selection ||
      selection.error != CanonicalSyncSelectionError::ResourceInfeasible ||
      !options.patterns.enableConflictCoreRepair ||
      options.maximumRepairRounds == 0) {
    outcome.feasible = static_cast<bool>(selection);
    outcome.selection = std::move(selection);
    return outcome;
  }

  const std::vector<CanonicalSyncMechanismId> initialCore =
      firstConflictCore(selection);
  CanonicalSyncProblemBuildResult repair =
      buildCanonicalSyncRepairProblem(program, problem, options, initialCore);
  if (!repair) {
    outcome.selection = std::move(selection);
    outcome.fatalConstructionError = true;
    return outcome;
  }
  outcome.ownedProblem = std::move(repair.problem);
  outcome.selectedProblem = outcome.ownedProblem.get();
  outcome.repairFrontierTruncated =
      outcome.ownedProblem->getPatternStatistics().repairFrontierTruncated;
  RepairBudget budget(options.maximumRepairTrials,
                      options.maximumRepairWorkUnits);
  std::optional<CanonicalSyncSelection> currentSelection =
      budget.run(*outcome.ownedProblem, current);
  std::vector<CanonicalSyncMechanismId> core = initialCore;

  for (std::size_t round = 1; round <= options.maximumRepairRounds; ++round) {
    outcome.repairRounds = round;
    std::optional<VerifiedRepairCandidate> bestVerified;
    std::optional<CanonicalSyncSelection> bestPressureTrial;
    CanonicalSyncGreedyOptions bestPressureOptions;
    if (currentSelection) {
      considerVerifiedRepair(*outcome.ownedProblem, *currentSelection,
                             bestVerified);
      if (currentSelection->error ==
          CanonicalSyncSelectionError::ResourceInfeasible) {
        bestPressureTrial = *currentSelection;
        bestPressureOptions = current;
      }
    }
    for (CanonicalSyncMechanismId mechanism : core) {
      CanonicalSyncGreedyOptions trialOptions = current;
      trialOptions.forbiddenMechanisms.push_back(mechanism);
      std::optional<CanonicalSyncSelection> trial =
          budget.run(*outcome.ownedProblem, trialOptions);
      if (!trial) {
        break;
      }
      considerVerifiedRepair(*outcome.ownedProblem, *trial, bestVerified);
      if (trial->error == CanonicalSyncSelectionError::ResourceInfeasible &&
          (!bestPressureTrial ||
           resourceOverflow(*trial) < resourceOverflow(*bestPressureTrial))) {
        bestPressureTrial = *trial;
        bestPressureOptions = std::move(trialOptions);
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
    currentSelection = std::move(bestPressureTrial);
    current = std::move(bestPressureOptions);
    core = firstConflictCore(*currentSelection);
  }
  if (!outcome.feasible) {
    outcome.selection =
        currentSelection ? std::move(*currentSelection) : std::move(selection);
  }
  outcome.repairBudgetExhausted = budget.exhausted();
  outcome.repairTrials = budget.trials();
  outcome.repairWorkUnits = budget.workUnits();
  return outcome;
}

CanonicalSyncStrategyReport
buildStrategyReport(CanonicalSyncSelectionStrategy strategy,
                    const SelectionOutcome &outcome, bool verified,
                    bool usedLocalizedPipeAll = false) {
  const CanonicalSyncPatternProblem &problem = outcome.getProblem();
  CanonicalSyncStrategyReport report;
  report.strategy = strategy;
  report.error = outcome.selection.error;
  report.verified = verified;
  report.usedLocalizedPipeAll = usedLocalizedPipeAll;
  report.repairFrontierTruncated = outcome.repairFrontierTruncated;
  report.repairBudgetExhausted = outcome.repairBudgetExhausted;
  report.backstopDeletionTruncated = outcome.backstopDeletionTruncated;
  report.repairRounds = outcome.repairRounds;
  report.repairTrials = outcome.repairTrials;
  report.repairWorkUnits = outcome.repairWorkUnits;
  report.backstopDeletionTrials = outcome.backstopDeletionTrials;
  report.backstopDeletionWorkUnits = outcome.backstopDeletionWorkUnits;
  report.cost = outcome.selection.cost;
  report.search = outcome.selection.statistics;
  report.allocation = outcome.selection.allocation;
  for (CanonicalSyncMechanismId mechanismId : outcome.selection.mechanisms) {
    const CanonicalSyncMechanismDescriptor &descriptor =
        problem.getMechanisms()[mechanismId].descriptor;
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
  return report;
}

CanonicalSyncComparisonReport
buildComparisonHeader(const CanonicalSyncPatternProblem &problem) {
  CanonicalSyncComparisonReport report;
  report.demands = problem.getDemands().size();
  report.directMechanisms = problem.getMechanisms().size();
  const CanonicalSyncPatternStatistics &statistics =
      problem.getPatternStatistics();
  report.directPairProposals = statistics.directPairProposals;
  report.directPairEvaluations = statistics.directPairEvaluations;
  report.synergisticPairs =
      statistics.get(CanonicalSyncPatternKind::DirectPair).patterns;
  report.pairGenerationTruncated = problem.wasPatternGenerationTruncated();
  return report;
}

struct PipeAllFallbackOutcome {
  std::unique_ptr<CanonicalSyncPatternProblem> problem;
  CanonicalSyncVerifiedPlan plan;
  bool deletionTruncated = false;
  std::size_t deletionTrials = 0;
  std::size_t deletionWorkUnits = 0;
};

std::size_t saturatedAddSize(std::size_t first, std::size_t second) {
  return second > std::numeric_limits<std::size_t>::max() - first
             ? std::numeric_limits<std::size_t>::max()
             : first + second;
}

std::size_t saturatedMultiplySize(std::size_t first, std::size_t second) {
  return first != 0 && second > std::numeric_limits<std::size_t>::max() / first
             ? std::numeric_limits<std::size_t>::max()
             : first * second;
}

std::size_t
estimateFinalVerificationWork(const CanonicalSyncPatternProblem &problem,
                              ArrayRef<CanonicalSyncMechanismId> mechanisms) {
  const SyncCoverExpansionStatistics expansion =
      problem.getExpansion().getStatistics();
  std::size_t structure =
      saturatedAddSize(expansion.virtualNodes, expansion.virtualEdges);
  structure = saturatedAddSize(structure, problem.getDemands().size());
  structure = saturatedAddSize(structure, problem.getPatterns().size());
  std::size_t supplies = 1;
  for (CanonicalSyncMechanismId mechanism : mechanisms) {
    supplies = saturatedAddSize(
        supplies,
        problem.getMechanisms()[mechanism].descriptor.supplies.size());
  }
  return saturatedMultiplySize(structure, supplies);
}

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
  selection.allocation =
      allocateCanonicalSyncEvents(problem, selection.mechanisms);
  CanonicalSyncVerifiedPlan plan =
      verifyCanonicalSyncSelection(problem, selection);
  if (!plan) {
    return std::nullopt;
  }
  const std::vector<CanonicalSyncMechanismId> deletionOrder =
      selection.mechanisms;
  std::size_t deletionTrials = 0;
  std::size_t deletionWorkUnits = 0;
  bool deletionTruncated = false;
  for (auto position = deletionOrder.rbegin(); position != deletionOrder.rend();
       ++position) {
    CanonicalSyncSelection trial = selection;
    const auto found = std::lower_bound(trial.mechanisms.begin(),
                                        trial.mechanisms.end(), *position);
    const bool mechanismMissing =
        found == trial.mechanisms.end() || *found != *position;
    if (mechanismMissing) {
      continue;
    }
    trial.mechanisms.erase(found);
    const std::size_t work =
        estimateFinalVerificationWork(problem, trial.mechanisms);
    const bool trialLimitReached =
        deletionTrials == options.maximumBackstopDeletionTrials;
    const bool workLimitExceeded =
        work > options.maximumBackstopDeletionWorkUnits ||
        deletionWorkUnits > options.maximumBackstopDeletionWorkUnits - work;
    if (trialLimitReached || workLimitExceeded) {
      deletionTruncated = true;
      break;
    }
    ++deletionTrials;
    deletionWorkUnits += work;
    trial.allocation = allocateCanonicalSyncEvents(problem, trial.mechanisms);
    const CanonicalSyncVerifiedPlan verified =
        verifyCanonicalSyncSelection(problem, trial);
    if (verified) {
      selection = std::move(trial);
      plan = verified;
    }
  }
  return PipeAllFallbackOutcome{std::move(built.problem), std::move(plan),
                                deletionTruncated, deletionTrials,
                                deletionWorkUnits};
}

SelectionOutcome takeFallbackSelection(PipeAllFallbackOutcome fallback,
                                       const SelectionOutcome &failed) {
  SelectionOutcome outcome;
  outcome.ownedProblem = std::move(fallback.problem);
  outcome.selectedProblem = outcome.ownedProblem.get();
  outcome.feasible = true;
  outcome.verifiedPlan = fallback.plan;
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
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(function, options.analysis);
  if (failed(program)) {
    return failure();
  }
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(*program, options);
  const bool preciseUncoverable =
      !precise &&
      precise.status.error == CanonicalSyncProblemError::UncoverableDemand;
  if (!precise && !preciseUncoverable) {
    function.emitError() << "cannot build canonical sync precise problem, "
                            "error="
                         << static_cast<unsigned>(precise.status.error);
    return failure();
  }
  CanonicalSyncComparisonReport report =
      buildComparisonHeader(*precise.problem);
  if (options.analysisOnly || options.compareSelectionStrategies) {
    for (CanonicalSyncSelectionStrategy strategy :
         {CanonicalSyncSelectionStrategy::FixedCover,
          CanonicalSyncSelectionStrategy::ActionAwareSingleton,
          CanonicalSyncSelectionStrategy::PairLookahead}) {
      CanonicalSyncBuildOptions trialOptions = options;
      trialOptions.selection.strategy = strategy;
      SelectionOutcome outcome;
      if (precise) {
        outcome =
            selectWithBoundedRepair(*program, *precise.problem, trialOptions);
      } else {
        outcome.selectedProblem = precise.problem.get();
        outcome.selection.error =
            CanonicalSyncSelectionError::NoCoveringPattern;
      }
      if (outcome.fatalConstructionError) {
        return failure();
      }
      bool usedLocalizedPipeAll = false;
      if (!outcome.feasible) {
        std::optional<PipeAllFallbackOutcome> fallback =
            buildLocalizedPipeAllFallback(*program, trialOptions);
        if (!fallback) {
          function.emitError("canonical sync comparison could not construct "
                             "its localized PIPE_ALL backstop");
          return failure();
        }
        outcome = takeFallbackSelection(std::move(*fallback), outcome);
        usedLocalizedPipeAll = true;
      }
      bool verified = false;
      if (outcome.feasible) {
        verified = outcome.verifiedPlan.has_value() ||
                   static_cast<bool>(verifyCanonicalSyncSelection(
                       outcome.getProblem(), outcome.selection));
      }
      report.strategies.push_back(buildStrategyReport(
          strategy, outcome, verified, usedLocalizedPipeAll));
    }
    if (options.reportCallback && failed(options.reportCallback(report))) {
      return failure();
    }
    if (options.analysisOnly) {
      return success();
    }
  }

  SelectionOutcome selection;
  if (precise) {
    selection = selectWithBoundedRepair(*program, *precise.problem, options);
  } else {
    selection.selectedProblem = precise.problem.get();
    selection.selection.error = CanonicalSyncSelectionError::NoCoveringPattern;
  }
  if (selection.fatalConstructionError) {
    return failure();
  }
  if (selection.feasible) {
    const CanonicalSyncVerifiedPlan plan =
        selection.verifiedPlan
            ? *selection.verifiedPlan
            : verifyCanonicalSyncSelection(selection.getProblem(),
                                           selection.selection);
    if (!plan) {
      function.emitError() << "canonical sync finalization failed, error="
                           << static_cast<unsigned>(plan.error);
      return failure();
    }
    if (!options.compareSelectionStrategies) {
      report.strategies.push_back(
          buildStrategyReport(options.selection.strategy, selection, true));
      if (options.reportCallback && failed(options.reportCallback(report))) {
        return failure();
      }
    }
    return materializeCanonicalSyncPlan(*program, selection.getProblem(), plan);
  }

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
  if (!options.compareSelectionStrategies) {
    report.strategies.push_back(buildStrategyReport(
        options.selection.strategy, fallbackOutcome, true, true));
    if (options.reportCallback && failed(options.reportCallback(report))) {
      return failure();
    }
  }
  return materializeCanonicalSyncPlan(*program, fallbackOutcome.getProblem(),
                                      *fallbackOutcome.verifiedPlan);
}
