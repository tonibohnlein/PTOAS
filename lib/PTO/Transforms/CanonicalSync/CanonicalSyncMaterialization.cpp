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
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit: {
    if (anchor.scope >= program.getScopeBindings().size()) {
      return std::nullopt;
    }
    Operation *owner = program.getScopeBindings()[anchor.scope].owner;
    if (!owner || owner == program.getFunction().getOperation()) {
      return std::nullopt;
    }
    return std::make_pair(owner,
                          anchor.kind == SyncCoverAnchorKind::ScopeEntry);
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
  if (action.guard == CanonicalSyncActionGuardKind::LoopNonEmpty) {
    if (!action.guardScope ||
        *action.guardScope >= program.getScopeBindings().size()) {
      return std::nullopt;
    }
    result.guardLoop = dyn_cast_or_null<scf::ForOp>(
        program.getScopeBindings()[*action.guardScope].owner);
    if (!result.guardLoop) {
      return std::nullopt;
    }
  } else if (action.guard != CanonicalSyncActionGuardKind::None ||
             action.guardScope) {
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
  result.eventId = allocation->second[action.eventLane];
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
  if (!action.guardLoop) {
    emitPhysicalAction(rewriter, function, action);
    return;
  }

  OpBuilder::InsertionGuard insertionGuard(rewriter);
  const Location location = action.anchor->getLoc();
  scf::ForOp guardLoop = action.guardLoop;
  const Value condition = rewriter.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::slt, guardLoop.getLowerBound(),
      guardLoop.getUpperBound());
  scf::IfOp guard = rewriter.create<scf::IfOp>(location, TypeRange{}, condition,
                                               /*withElseRegion=*/false);
  markGenerated(guard, rewriter);
  rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
  emitPhysicalAction(rewriter, function, action);
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
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (failed(problem)) {
    return failure();
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(**problem, options.selection);
  if (!selection) {
    function.emitError() << "canonical sync selection failed, error="
                         << static_cast<unsigned>(selection.error);
    return failure();
  }
  const CanonicalSyncVerifiedPlan plan =
      verifyCanonicalSyncSelection(**problem, selection);
  if (!plan) {
    function.emitError() << "canonical sync final verification failed, error="
                         << static_cast<unsigned>(plan.error);
    return failure();
  }
  return materializeCanonicalSyncPlan(*program, **problem, plan);
}
