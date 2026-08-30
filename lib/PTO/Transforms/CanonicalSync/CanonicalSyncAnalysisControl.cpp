// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncAnalysisInternal.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool checkedTimelineEnd(std::size_t nodeCount, std::size_t &end) {
  const bool overflows =
      nodeCount > (std::numeric_limits<std::size_t>::max() - 1) / 2;
  if (overflows) {
    return false;
  }
  end = nodeCount * 2 + 1;
  return true;
}

} // namespace

LogicalResult ProgramBuilder::buildScopes() {
  std::size_t timelineEnd = 0;
  if (!checkedTimelineEnd(compounds_.size(), timelineEnd)) {
    return function_.emitError("canonical sync timeline overflow");
  }
  scopeBindings_.push_back({function_.getOperation(), &function_.getBody()});
  return addRegion(function_.getBody(), {0, {}}, timelineEnd);
}

LogicalResult ProgramBuilder::addRegion(Region &region,
                                        const RegionContext &context,
                                        std::size_t timelineEnd) {
  if (region.empty()) {
    return success();
  }
  if (!region.hasOneBlock()) {
    return function_.emitError(
        "canonical sync requires single-block structured regions");
  }
  contexts_[&region] = context;
  for (Operation &operation : region.front()) {
    if (isCanonicalSyncOwned(&operation)) {
      continue;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      const bool scopeLimitReached =
          graph_.getScopes().size() >= options_.maximumScopes;
      if (scopeLimitReached) {
        return loop.emitError("canonical sync scope limit exceeded");
      }
      const SyncCoverGraphResult scope = graph_.addScope(
          context.scope, false, SyncCoverTimelineInterval{0, timelineEnd}, true,
          context.guard);
      if (!scope) {
        return loop.emitError("cannot construct canonical sync loop scope");
      }
      loopScopes_.push_back({loop.getOperation(), *scope.index});
      scopeBindings_.push_back({loop.getOperation(), &loop.getRegion()});
      if (failed(addRegion(loop.getRegion(), {*scope.index, context.guard},
                           timelineEnd))) {
        return failure();
      }
      continue;
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      const bool controlLimitReached =
          graph_.getControls().size() >= options_.maximumControls;
      if (controlLimitReached) {
        return conditional.emitError("canonical sync control limit exceeded");
      }
      const std::size_t scopeCount = graph_.getScopes().size();
      const bool scopeLimitExceeded = scopeCount > options_.maximumScopes ||
                                      options_.maximumScopes - scopeCount < 2;
      if (scopeLimitExceeded) {
        return conditional.emitError("canonical sync scope limit exceeded");
      }
      const SyncCoverGraphResult control = graph_.addControl(2, context.scope);
      if (!control) {
        return conditional.emitError(
            "cannot construct canonical sync branch control");
      }
      controlBindings_.push_back({conditional.getOperation()});
      if (failed(addPeriodicControlEvidence(conditional, *control.index,
                                            context.scope)) ||
          failed(addSuccessorControlEvidence(conditional, *control.index,
                                             context.scope))) {
        return failure();
      }
      for (unsigned alternative = 0; alternative < 2; ++alternative) {
        Region &alternativeRegion = conditional->getRegion(alternative);
        SyncCoverGuard alternativeGuard = context.guard;
        alternativeGuard.literals.push_back({*control.index, alternative});
        const SyncCoverGraphResult scope = graph_.addScope(
            context.scope, true, std::nullopt, false, alternativeGuard);
        if (!scope) {
          return conditional.emitError(
              "cannot construct canonical sync branch scope");
        }
        scopeBindings_.push_back(
            {conditional.getOperation(), &alternativeRegion});
        RegionContext alternativeContext{*scope.index,
                                         std::move(alternativeGuard)};
        if (failed(addRegion(alternativeRegion, alternativeContext,
                             timelineEnd))) {
          return failure();
        }
      }
      continue;
    }
    if (!isTransparentRegionOperation(&operation)) {
      continue;
    }
    for (Region &nested : operation.getRegions()) {
      if (failed(addRegion(nested, context, timelineEnd))) {
        return failure();
      }
    }
  }
  return success();
}

std::optional<SyncCoverScopeId>
ProgramBuilder::getNearestLoopScope(SyncCoverScopeId scope) const {
  return graph_.getNearestEnclosingLoop(scope);
}

LogicalResult
ProgramBuilder::addPeriodicControlEvidence(scf::IfOp conditional,
                                           SyncCoverControlId control,
                                           SyncCoverScopeId occurrenceScope) {
  const std::optional<SyncCoverScopeId> loopScope =
      getNearestLoopScope(occurrenceScope);
  if (!loopScope || *loopScope >= scopeBindings_.size()) {
    return success();
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(scopeBindings_[*loopScope].owner);
  auto comparison = conditional.getCondition().getDefiningOp<arith::CmpIOp>();
  const bool unsupportedComparisonKind =
      comparison && comparison.getPredicate() != arith::CmpIPredicate::eq &&
      comparison.getPredicate() != arith::CmpIPredicate::ne;
  if (!loop || !comparison || unsupportedComparisonKind) {
    return success();
  }

  Value remainderValue;
  APInt selectedPhase;
  const auto matchComparisonOperand = [&](Value possibleRemainder,
                                          Value possibleConstant) {
    APInt constant;
    Operation *operation = possibleRemainder.getDefiningOp();
    const bool supportedRemainder =
        operation && isa<arith::RemSIOp, arith::RemUIOp>(operation) &&
        matchPattern(possibleConstant, m_ConstantInt(&constant));
    if (!supportedRemainder) {
      return false;
    }
    remainderValue = possibleRemainder;
    selectedPhase = constant;
    return true;
  };
  const bool unsupportedOperands =
      !matchComparisonOperand(comparison.getLhs(), comparison.getRhs()) &&
      !matchComparisonOperand(comparison.getRhs(), comparison.getLhs());
  if (unsupportedOperands) {
    return success();
  }

  Operation *remainder = remainderValue.getDefiningOp();
  APInt divisor;
  APInt lower;
  APInt step;
  const bool unsupportedPhase =
      remainder->getOperand(0) != loop.getInductionVar() ||
      !matchPattern(remainder->getOperand(1), m_ConstantInt(&divisor)) ||
      !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      !divisor.isStrictlyPositive() || lower.isNegative() ||
      !step.isStrictlyPositive() || divisor.getActiveBits() > 16 ||
      !lower.isSignedIntN(64) || step.getActiveBits() > 63 ||
      !selectedPhase.isSignedIntN(64);
  if (unsupportedPhase) {
    return success();
  }
  const std::size_t phases = divisor.getZExtValue();
  if (phases == 0 || phases > kMaximumSlotCount) {
    return success();
  }
  const std::int64_t selected = selectedPhase.getSExtValue();
  const bool invalidSelected =
      selected < 0 || static_cast<std::size_t>(selected) >= phases;
  if (invalidSelected) {
    return success();
  }

  SyncCoverControlPhaseRelation relation;
  relation.loopScope = *loopScope;
  relation.initialPhase = lower.getZExtValue() % phases;
  relation.nextPhase.resize(phases);
  relation.activeAlternative.resize(phases);
  const std::size_t stepPhase = step.getZExtValue() % phases;
  for (std::size_t phase = 0; phase < phases; ++phase) {
    relation.nextPhase[phase] = (phase + stepPhase) % phases;
    const bool equal = phase == static_cast<std::size_t>(selected);
    const bool takeThen =
        comparison.getPredicate() == arith::CmpIPredicate::eq ? equal : !equal;
    relation.activeAlternative[phase] = takeThen ? 0 : 1;
  }
  if (!graph_.setControlPhaseRelation(control, std::move(relation))) {
    return conditional.emitError(
        "cannot register canonical sync periodic control relation");
  }
  return success();
}

LogicalResult
ProgramBuilder::addSuccessorControlEvidence(scf::IfOp conditional,
                                            SyncCoverControlId control,
                                            SyncCoverScopeId occurrenceScope) {
  const std::optional<SyncCoverScopeId> loopScope =
      getNearestLoopScope(occurrenceScope);
  if (!loopScope || *loopScope >= scopeBindings_.size()) {
    return success();
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(scopeBindings_[*loopScope].owner);
  auto comparison = conditional.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!loop || !comparison ||
      comparison.getPredicate() != arith::CmpIPredicate::slt ||
      comparison.getRhs() != loop.getUpperBound()) {
    return success();
  }
  auto addition = comparison.getLhs().getDefiningOp<arith::AddIOp>();
  if (!addition) {
    return success();
  }
  const bool exactStep = (addition.getLhs() == loop.getInductionVar() &&
                          addition.getRhs() == loop.getStep()) ||
                         (addition.getRhs() == loop.getInductionVar() &&
                          addition.getLhs() == loop.getStep());
  if (!exactStep) {
    return success();
  }
  if (!graph_.setControlSuccessorRelation(control, {*loopScope, 0})) {
    return conditional.emitError(
        "cannot register canonical sync successor control relation");
  }
  return success();
}

LogicalResult ProgramBuilder::validateControlDataflow() {
  auto rejectScheduledControlValue = [&](Operation *owner, Value value,
                                         StringRef role) -> LogicalResult {
    llvm::SetVector<SyncCoverNodeId> producers;
    llvm::DenseSet<Value> visited;
    if (failed(collectScheduledProducers(value, producers, visited))) {
      return failure();
    }
    if (producers.empty()) {
      return success();
    }
    return owner->emitError("canonical sync cannot model asynchronous ")
           << role << " produced by a scheduled pipe operation";
  };

  WalkResult walk = function_.walk([&](Operation *operation) {
    if (isCanonicalSyncOwned(operation)) {
      return WalkResult::skip();
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      if (failed(rejectScheduledControlValue(
              operation, conditional.getCondition(), "scf.if condition"))) {
        return WalkResult::interrupt();
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      const bool unsupportedControl =
          failed(rejectScheduledControlValue(operation, loop.getLowerBound(),
                                             "scf.for lower bound")) ||
          failed(rejectScheduledControlValue(operation, loop.getUpperBound(),
                                             "scf.for upper bound")) ||
          failed(rejectScheduledControlValue(operation, loop.getStep(),
                                             "scf.for step"));
      if (unsupportedControl) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult ProgramBuilder::refineLoopTimelines() {
  for (const auto &[operation, scope] : loopScopes_) {
    const auto nodes = loopNodes_.find(operation);
    const bool emptyLoop = nodes == loopNodes_.end() || nodes->second.empty();
    if (emptyLoop) {
      SyncCoverScopeId parent = graph_.getScopes()[scope].parent;
      while (!graph_.getScopes()[parent].timeline) {
        parent = graph_.getScopes()[parent].parent;
      }
      if (!graph_.setScopeTimeline(scope,
                                   *graph_.getScopes()[parent].timeline)) {
        return operation->emitError(
            "cannot clamp empty canonical sync loop timeline");
      }
      continue;
    }
    const std::optional<SyncCoverTimelinePosition> begin =
        resolveSyncCoverAnchor(
            graph_, {SyncCoverAnchorKind::BeforeNode,
                     graph_.getNodes()[nodes->second.front()].id, 0, 0});
    const std::optional<SyncCoverTimelinePosition> end = resolveSyncCoverAnchor(
        graph_, {SyncCoverAnchorKind::AfterNode,
                 graph_.getNodes()[nodes->second.back()].id, 0, 0});
    if (!begin || !end || !graph_.setScopeTimeline(scope, {*begin, *end})) {
      return operation->emitError("cannot refine canonical sync loop timeline");
    }
  }
  return success();
}

void ProgramBuilder::collectHiddenEventReservations() {
  llvm::DenseSet<Operation *> visited;
  for (const CanonicalSyncNodeBinding &binding : nodeBindings_) {
    if (!visited.insert(binding.operation).second) {
      continue;
    }
    const std::optional<SyncMacroModel> model =
        getSyncMacroModel(binding.operation);
    if (!model) {
      continue;
    }
    for (const SyncMacroHiddenEvent &hidden : model->hiddenEvents) {
      auto &ids =
          eventReservations_[{static_cast<std::uint32_t>(hidden.srcPipe),
                              static_cast<std::uint32_t>(hidden.dstPipe)}];
      ids.insert(ids.end(), hidden.eventIds.begin(), hidden.eventIds.end());
    }
  }
  for (auto &[domain, ids] : eventReservations_) {
    (void)domain;
    llvm::sort(ids);
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }
}

void ProgramBuilder::indexNodesByLoop() {
  llvm::DenseMap<Operation *, SyncCoverScopeId> scopes;
  for (const auto &[operation, scope] : loopScopes_) {
    scopes[operation] = scope;
  }
  for (SyncCoverNodeId node = 0; node < nodeBindings_.size(); ++node) {
    for (Operation *parent = nodeBindings_[node].operation->getParentOp();
         parent != nullptr; parent = parent->getParentOp()) {
      auto scope = scopes.find(parent);
      if (scope != scopes.end()) {
        loopNodes_[parent].push_back(node);
      }
    }
  }
}
