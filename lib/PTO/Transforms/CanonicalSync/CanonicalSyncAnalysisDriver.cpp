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
#include "CanonicalSyncTarget.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool isManualSync(Operation *operation) {
  // Reject operations that consume the intra-core event resources this pass
  // owns. Pipe barriers are modeled as fixed completion supply. Whole-core
  // syncall and fence-barrier-all remain preserved fixed constraints but are
  // not yet credited as cover supply.
  const StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Cases(RecordEventOp::getOperationName(), WaitEventOp::getOperationName(),
             BarrierSyncOp::getOperationName(), true)
      .Cases(SetFlagOp::getOperationName(), WaitFlagOp::getOperationName(),
             SetFlagDynOp::getOperationName(), true)
      .Cases(WaitFlagDynOp::getOperationName(), TSyncOp::getOperationName(),
             true)
      .Cases(GetBufOp::getOperationName(), GetBufDynOp::getOperationName(),
             RlsBufOp::getOperationName(), true)
      .Cases(RlsBufDynOp::getOperationName(),
             DeclareEventIdArrayOp::getOperationName(),
             EventIdArrayGetOp::getOperationName(), true)
      .Cases(EventIdArraySetOp::getOperationName(),
             SyncSetOp::getOperationName(), SyncWaitOp::getOperationName(),
             true)
      .Cases(WaitAsyncEventOp::getOperationName(),
             TestAsyncEventOp::getOperationName(), true)
      .Cases(SetCrossBlockOp::getOperationName(),
             WaitCrossBlockOp::getOperationName(),
             SetIntraBlockOp::getOperationName(), true)
      .Case(WaitIntraBlockOp::getOperationName(), true)
      .Default(false);
}

bool isSupportedOwnedOperation(Operation *operation) {
  const StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Cases(BarrierOp::getOperationName(), SetFlagOp::getOperationName(),
             WaitFlagOp::getOperationName(), true)
      .Cases(SetFlagDynOp::getOperationName(),
             WaitFlagDynOp::getOperationName(), scf::IfOp::getOperationName(),
             true)
      .Cases(arith::AddIOp::getOperationName(),
             arith::SubIOp::getOperationName(),
             arith::DivUIOp::getOperationName(), true)
      .Cases(arith::RemUIOp::getOperationName(),
             arith::CmpIOp::getOperationName(),
             arith::SelectOp::getOperationName(), true)
      .Case(arith::ConstantOp::getOperationName(), true)
      .Default(false);
}

LogicalResult validateOwnedOperationTree(Operation *root) {
  WalkResult walk = root->walk([&](Operation *operation) {
    const bool generatedTerminator =
        operation != root && isa<scf::YieldOp>(operation) &&
        isCanonicalSyncOwned(operation->getParentOp());
    if (generatedTerminator) {
      return WalkResult::advance();
    }
    const bool invalidMarker = !isCanonicalSyncOwned(operation) ||
                               !isSupportedOwnedOperation(operation);
    if (invalidMarker) {
      operation->emitError(
          "canonical sync found malformed pass-owned synchronization IR");
      return WalkResult::interrupt();
    }
    for (Value result : operation->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!isCanonicalSyncOwned(user)) {
          user->emitError("canonical sync pass-owned helper escapes into "
                          "user IR");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

bool isGmArgumentType(Type type) {
  if (auto pointer = dyn_cast<PtrType>(type)) {
    return pointer.getMemorySpace().getAddressSpace() == AddressSpace::GM;
  }
  if (isa<TensorViewType, PartitionTensorViewType>(type)) {
    return true;
  }
  if (auto memory = dyn_cast<MemRefType>(type)) {
    Attribute space = memory.getMemorySpace();
    if (!space) {
      return true;
    }
    auto ptoSpace = dyn_cast<AddressSpaceAttr>(space);
    return ptoSpace && ptoSpace.getAddressSpace() == AddressSpace::GM;
  }
  return false;
}

} // namespace

bool mlir::pto::canonical_sync_detail::isCanonicalSyncOwned(
    Operation *operation) {
  return operation &&
         isa_and_nonnull<UnitAttr>(operation->getAttr("pto.canonical_sync"));
}

bool mlir::pto::canonical_sync_detail::isTransparentRegionOperation(
    Operation *operation) {
  return operation->getName().getStringRef().starts_with("pto.section.");
}

bool mlir::pto::canonical_sync_detail::isCompletionOrdered(
    std::uint32_t resource,
    const CanonicalSyncTargetCapabilities &capabilities) {
  return capabilities.sameResourceCompletionOrdering.supports(resource);
}

bool mlir::pto::canonical_sync_detail::canSignalPrefixCompletion(
    std::uint32_t resource,
    const CanonicalSyncTargetCapabilities &capabilities) {
  // A later set may represent an earlier issued prefix only on resources with
  // an explicit in-order completion contract. Other physical resources may
  // still signal the completion of the immediately preceding operation.
  return isCompletionOrdered(resource, capabilities);
}

ProgramBuilder::ProgramBuilder(func::FuncOp function,
                               const CanonicalSyncAnalysisOptions &options)
    : function_(function), options_(options),
      targetCapabilities_(getCanonicalSyncTargetCapabilities(function)) {}

FailureOr<CanonicalSyncProgram> ProgramBuilder::build() {
  const bool needsTargetCompletionResources =
      options_.discoverTargetCompletionCertificates ||
      options_.discoverBasicOwnershipCertificates;
  const bool failedBarrierConfiguration =
      !graph_.setBlockingTargetedBarrierResources(
          targetCapabilities_.targetedBarrierDrainsSourcePrefix.resources) ||
      !graph_.setCrossResourceTargetedBarrierPairs(
          targetCapabilities_.crossResourceTargetedBarrierCompletion
              .resourcePairs);
  const bool failedTargetCompletionConfiguration =
      needsTargetCompletionResources &&
      targetCapabilities_.targetCompletionResources &&
      !graph_.setTargetCompletionResources(
          *targetCapabilities_.targetCompletionResources);
  const bool failedStage =
      failedBarrierConfiguration || failedTargetCompletionConfiguration ||
      failed(validateInput()) || failed(extract()) || failed(buildScopes()) ||
      failed(buildNodesAndStorage()) || failed(validateControlDataflow()) ||
      failed(addFixedIssueOrder()) ||
      failed(addCertifiedCompletionFrontiers()) ||
      failed(buildStorageConflictIndex()) || failed(addForwardDependencies()) ||
      failed(addRecurrenceDependencies()) ||
      (options_.discoverTargetCompletionCertificates &&
       failed(addTargetCompletionCertificates(targetCapabilities_))) ||
      (options_.discoverBasicOwnershipCertificates &&
       failed(discoverBasicOwnershipCertificates(targetCapabilities_)));
  if (failedBarrierConfiguration) {
    function_.emitError(
        "cannot configure canonical sync blocking-barrier resources");
  }
  if (failedTargetCompletionConfiguration) {
    function_.emitError(
        "cannot configure canonical sync target completion resources");
  }
  if (failedStage) {
    return failure();
  }
  if (!graph_.freezeStructure()) {
    function_.emitError("cannot freeze canonical sync graph");
    return failure();
  }
  std::vector<AddressSpace> storageSpaces(graph_.getStorageDomains().size(),
                                          AddressSpace::Zero);
  for (const auto &[space, domain] : storageDomains_) {
    storageSpaces[domain] = space;
  }
  return CanonicalSyncProgram(
      function_, std::move(graph_), std::move(nodeBindings_),
      std::move(scopeBindings_), std::move(controlBindings_),
      std::move(storageSpaces), std::move(targetCapabilities_),
      ownershipDiscoveryStatistics_, std::move(eventReservations_));
}

LogicalResult ProgramBuilder::validateInput() {
  if (function_.isExternal()) {
    return function_.emitError("canonical sync requires a function body");
  }
  if (options_.maximumPeriodicRecurrenceStates >
      kCanonicalSyncMaximumPeriodicRecurrenceStates) {
    return function_.emitError(
        "canonical sync periodic recurrence state limit exceeds the "
        "supported maximum");
  }
  if (options_.maximumRecurrenceWitnessStates >
      kCanonicalSyncMaximumRecurrenceWitnessStates) {
    return function_.emitError(
        "canonical sync recurrence witness-state limit exceeds the supported "
        "maximum");
  }
  const bool invalidLimits =
      options_.maximumNodes == 0 || options_.maximumScopes == 0 ||
      options_.maximumControls == 0 || options_.maximumStorageAccesses == 0 ||
      options_.maximumPairInspections == 0 ||
      options_.maximumPeriodicRecurrenceStates == 0 ||
      options_.maximumRecurrenceWitnessStates == 0 ||
      options_.maximumBasicOwnershipInspections == 0 ||
      options_.maximumBasicOwnershipCertificates == 0;
  if (invalidLimits) {
    return function_.emitError(
        "canonical sync analysis limits must be positive");
  }
  if (Attribute raw = function_->getAttr("pto.noalias_pairs")) {
    auto pairs = dyn_cast<DenseI64ArrayAttr>(raw);
    const bool invalidPairs = !pairs || pairs.size() % 2 != 0;
    if (invalidPairs) {
      return function_.emitError(
          "expects 'pto.noalias_pairs' to be an even dense i64 array");
    }
    ArrayRef<std::int64_t> values = pairs.asArrayRef();
    for (std::size_t index = 0; index < values.size(); index += 2) {
      const std::int64_t first = values[index];
      const std::int64_t second = values[index + 1];
      const bool invalidPair = first < 0 || second < 0 || first == second ||
                               first >= function_.getNumArguments() ||
                               second >= function_.getNumArguments();
      if (invalidPair) {
        return function_.emitError("invalid 'pto.noalias_pairs' argument pair");
      }
      const bool invalidTypes =
          !isGmArgumentType(function_.getArgument(first).getType()) ||
          !isGmArgumentType(function_.getArgument(second).getType());
      if (invalidTypes) {
        return function_.emitError(
            "expects 'pto.noalias_pairs' to name GM arguments");
      }
      noAliasArguments_.insert(std::minmax(static_cast<unsigned>(first),
                                           static_cast<unsigned>(second)));
    }
  }

  WalkResult walk = function_.walk([&](Operation *operation) {
    const bool hasOwnershipMarker = operation->hasAttr("pto.canonical_sync");
    if (hasOwnershipMarker) {
      if (failed(validateOwnedOperationTree(operation))) {
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    }
    if (isManualSync(operation)) {
      operation->emitError(
          "canonical sync does not accept pre-existing pipe synchronization");
      return WalkResult::interrupt();
    }
    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      const bool insideLoop = barrier->getParentOfType<scf::ForOp>() != nullptr;
      if (insideLoop) {
        barrier.emitError("canonical sync does not yet support fixed pipe "
                          "barriers inside loops");
        return WalkResult::interrupt();
      }
    }
    if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
      const bool hasResults = conditional.getNumResults() != 0;
      if (hasResults) {
        conditional.emitError(
            "canonical sync does not yet support result-carrying scf.if");
        return WalkResult::interrupt();
      }
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      const bool hasIterArgs = !loop.getInitArgs().empty();
      if (hasIterArgs) {
        loop.emitError("canonical sync does not yet support scf.for iter_args");
        return WalkResult::interrupt();
      }
    }
    const bool hasSuccessors = operation->getNumSuccessors() != 0;
    if (hasSuccessors) {
      operation->emitError(
          "canonical sync does not support unstructured control flow");
      return WalkResult::interrupt();
    }
    const bool unsupportedRegion =
        operation->getNumRegions() != 0 &&
        !isa<func::FuncOp, scf::ForOp, scf::IfOp>(operation) &&
        !isTransparentRegionOperation(operation);
    if (unsupportedRegion) {
      operation->emitError("canonical sync cannot model this region operation");
      return WalkResult::interrupt();
    }
    const bool invalidTransparentRegion =
        isTransparentRegionOperation(operation) &&
        operation->getNumRegions() != 1;
    if (invalidTransparentRegion) {
      operation->emitError(
          "canonical sync requires one transparent structured region");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walk.wasInterrupted());
}

LogicalResult ProgramBuilder::extract() {
  PTOIRTranslatorOptions translatorOptions;
  translatorOptions.preciseGmRanges = true;
  translatorOptions.includeExtendedEffects = true;
  translatorOptions.failOnUnmodeledEffects = true;
  PTOIRTranslator translator(syncIR_, bufferMap_, operationMemory_, function_,
                             translatorOptions);
  if (failed(translator.Build())) {
    return failure();
  }
  for (const std::unique_ptr<InstanceElement> &element : syncIR_) {
    auto *compound = dyn_cast<CompoundInstanceElement>(element.get());
    if (!compound || !compound->elementOp) {
      continue;
    }
    const bool fixedOrOwnedSync =
        isCanonicalSyncOwned(compound->elementOp) ||
        isa<BarrierOp, FenceBarrierAllOp, SyncAllOp>(compound->elementOp);
    if (fixedOrOwnedSync) {
      continue;
    }
    compounds_.push_back(compound);
    const bool nodeLimitExceeded = compounds_.size() > options_.maximumNodes;
    if (nodeLimitExceeded) {
      return compound->elementOp->emitError(
          "canonical sync node limit exceeded");
    }
  }
  return success();
}

FailureOr<std::vector<Value>>
ProgramBuilder::getSsaOperands(Operation *operation, int macroPhase) {
  if (macroPhase < 0) {
    return std::vector<Value>(operation->operand_begin(),
                              operation->operand_end());
  }
  const std::optional<SyncMacroModel> model = getSyncMacroModel(operation);
  if (!model) {
    operation->emitError(
        "cannot determine synchronization macro phase operands");
    return failure();
  }
  if (failed(verifySyncMacroModel(operation, *model))) {
    return failure();
  }
  const auto phase = llvm::find_if(model->phases, [&](const auto &candidate) {
    return candidate.phaseId == static_cast<unsigned>(macroPhase);
  });
  if (phase == model->phases.end()) {
    operation->emitError("cannot bind synchronization macro phase operands");
    return failure();
  }
  return std::vector<Value>(phase->useValues.begin(), phase->useValues.end());
}

FailureOr<std::vector<unsigned>>
ProgramBuilder::getCompletedResults(Operation *operation, int macroPhase) {
  std::vector<unsigned> results;
  if (macroPhase < 0) {
    results.reserve(operation->getNumResults());
    for (unsigned result = 0; result < operation->getNumResults(); ++result) {
      results.push_back(result);
    }
    return results;
  }
  const std::optional<SyncMacroModel> model = getSyncMacroModel(operation);
  if (!model || failed(verifySyncMacroModel(operation, *model))) {
    operation->emitError(
        "cannot determine synchronization macro result-completion phases");
    return failure();
  }
  for (unsigned result = 0; result < operation->getNumResults(); ++result) {
    const std::optional<unsigned> completion =
        getSyncMacroResultCompletionPhase(*model, result);
    if (!completion) {
      operation->emitError(
          "cannot bind an SSA result to its synchronization macro phase");
      return failure();
    }
    if (*completion == static_cast<unsigned>(macroPhase)) {
      results.push_back(result);
    }
  }
  return results;
}

LogicalResult ProgramBuilder::buildNodesAndStorage() {
  std::vector<std::uint32_t> resources;
  for (const CompoundInstanceElement *compound : compounds_) {
    const std::uint32_t resource =
        static_cast<std::uint32_t>(compound->kPipeValue);
    if (compound->kPipeValue == PipelineType::PIPE_UNASSIGNED) {
      return compound->elementOp->emitError(
          "canonical sync requires a known issue resource");
    }
    resources.push_back(resource);
  }
  llvm::sort(resources);
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  for (std::uint32_t resource : resources) {
    const SyncCoverEdgeKind kind =
        isCompletionOrdered(resource, targetCapabilities_)
            ? SyncCoverEdgeKind::CompletionPreservingIssueOrder
            : SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
    if (!graph_.setResourceRecurrenceCarryKind(resource, kind)) {
      return function_.emitError(
          "cannot register canonical sync recurrence resource");
    }
  }

  std::map<Operation *, SyncCoverNodeId> physicalAnchors;
  std::size_t nextPhysicalOperation = 0;
  for (std::size_t order = 0; order < compounds_.size(); ++order) {
    CompoundInstanceElement *compound = compounds_[order];
    auto context = contexts_.find(compound->elementOp->getParentRegion());
    if (context == contexts_.end()) {
      return compound->elementOp->emitError(
          "canonical sync lost structured-region context");
    }
    const std::uint32_t resource =
        static_cast<std::uint32_t>(compound->kPipeValue);
    std::vector<std::uint32_t> completionTargets;
    for (std::uint32_t candidate : resources) {
      if (targetCapabilities_.directEventCompletion.supports(resource,
                                                             candidate)) {
        completionTargets.push_back(candidate);
      }
    }
    const auto physicalAnchor = physicalAnchors.find(compound->elementOp);
    const std::optional<SyncCoverNodeId> representative =
        physicalAnchor == physicalAnchors.end()
            ? std::nullopt
            : std::optional<SyncCoverNodeId>(physicalAnchor->second);
    FailureOr<std::vector<unsigned>> completedResults =
        getCompletedResults(compound->elementOp, compound->macroOpInstanceId);
    if (failed(completedResults)) {
      return failure();
    }
    const std::size_t physicalOperation =
        representative ? graph_.getNodes()[*representative].physicalOperation
                       : nextPhysicalOperation++;
    const SyncCoverGraphResult node = graph_.addNode(
        resource, 1, context->second.scope, order, context->second.guard,
        std::move(completionTargets), representative,
        canSignalPrefixCompletion(resource, targetCapabilities_),
        physicalOperation, compound->macroOpInstanceId,
        std::move(*completedResults));
    if (!node) {
      return compound->elementOp->emitError(
          "cannot construct canonical sync operation node");
    }
    FailureOr<std::vector<Value>> ssaOperands =
        getSsaOperands(compound->elementOp, compound->macroOpInstanceId);
    if (failed(ssaOperands)) {
      return failure();
    }
    physicalAnchors.try_emplace(compound->elementOp, *node.index);
    nodeBindings_.push_back({compound->elementOp, compound->macroOpInstanceId,
                             std::move(*ssaOperands)});
    operationNodes_[compound->elementOp].push_back(*node.index);
    nodeAccessIndices_.emplace_back();
    appendAccesses(*node.index, compound->useVec, false);
    appendAccesses(*node.index, compound->defVec, true);
    if (failed(materializeNodeAccesses(*node.index))) {
      return failure();
    }
  }
  for (const auto &[operation, nodes] : operationNodes_) {
    (void)operation;
    if (nodes.empty()) {
      continue;
    }
    for (SyncCoverNodeId node : nodes) {
      if (!graph_.setPhysicalExit(node, nodes.back())) {
        return function_.emitError(
            "cannot register canonical sync physical operation exit");
      }
    }
  }
  if (failed(indexSsaCompletionNodes())) {
    return failure();
  }
  indexNodesByLoop();
  if (failed(refineLoopTimelines())) {
    return failure();
  }
  collectHiddenEventReservations();
  return success();
}

LogicalResult ProgramBuilder::indexSsaCompletionNodes() {
  for (const auto &[operation, nodes] : operationNodes_) {
    const bool hasNoResults = operation->getNumResults() == 0;
    if (hasNoResults) {
      continue;
    }
    for (OpResult result : operation->getResults()) {
      const auto node = llvm::find_if(nodes, [&](SyncCoverNodeId candidate) {
        return llvm::is_contained(graph_.getNodes()[candidate].completedResults,
                                  result.getResultNumber());
      });
      if (node == nodes.end()) {
        return operation->emitError(
            "cannot bind an SSA result to its synchronization macro phase");
      }
      ssaCompletionNodes_[result] = *node;
    }
  }
  return success();
}

FailureOr<CanonicalSyncProgram> mlir::pto::buildCanonicalSyncProgram(
    func::FuncOp function, const CanonicalSyncAnalysisOptions &options) {
  return ProgramBuilder(function, options).build();
}
