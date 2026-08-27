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

#include "llvm/ADT/StringSwitch.h"

#include <algorithm>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

bool isManualSync(Operation *operation) {
  const StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<bool>(name)
      .Cases(RecordEventOp::getOperationName(), WaitEventOp::getOperationName(),
             BarrierSyncOp::getOperationName(), true)
      .Cases(SetFlagOp::getOperationName(), WaitFlagOp::getOperationName(),
             SetFlagDynOp::getOperationName(), true)
      .Cases(WaitFlagDynOp::getOperationName(), BarrierOp::getOperationName(),
             TSyncOp::getOperationName(), true)
      .Cases(GetBufOp::getOperationName(), GetBufDynOp::getOperationName(),
             RlsBufOp::getOperationName(), true)
      .Cases(RlsBufDynOp::getOperationName(),
             DeclareEventIdArrayOp::getOperationName(),
             EventIdArrayGetOp::getOperationName(), true)
      .Cases(EventIdArraySetOp::getOperationName(),
             SyncSetOp::getOperationName(), SyncWaitOp::getOperationName(),
             true)
      .Cases(SyncAllOp::getOperationName(),
             WaitAsyncEventOp::getOperationName(),
             TestAsyncEventOp::getOperationName(), true)
      .Cases(SetCrossBlockOp::getOperationName(),
             WaitCrossBlockOp::getOperationName(),
             SetIntraBlockOp::getOperationName(), true)
      .Cases(WaitIntraBlockOp::getOperationName(),
             FenceBarrierAllOp::getOperationName(), true)
      .Default(false);
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

bool mlir::pto::canonical_sync_detail::isTransparentRegionOperation(
    Operation *operation) {
  return operation->getName().getStringRef().starts_with("pto.section.");
}

bool mlir::pto::canonical_sync_detail::isCompletionOrdered(
    std::uint32_t resource, Operation *operation) {
  const PipelineType pipe = static_cast<PipelineType>(resource);
  return pipe == PipelineType::PIPE_S ||
         (pipe == PipelineType::PIPE_V && isTargetArchA5(operation));
}

bool mlir::pto::canonical_sync_detail::canSignalDirectCompletion(
    std::uint32_t resource) {
  switch (static_cast<PipelineType>(resource)) {
  case PipelineType::PIPE_S:
  case PipelineType::PIPE_V:
  case PipelineType::PIPE_MTE1:
  case PipelineType::PIPE_MTE2:
  case PipelineType::PIPE_MTE3:
  case PipelineType::PIPE_FIX:
    return true;
  case PipelineType::PIPE_M:
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

ProgramBuilder::ProgramBuilder(func::FuncOp function,
                               const CanonicalSyncAnalysisOptions &options)
    : function_(function), options_(options) {}

FailureOr<CanonicalSyncProgram> ProgramBuilder::build() {
  const bool failedStage =
      failed(validateInput()) || failed(extract()) || failed(buildScopes()) ||
      failed(buildNodesAndStorage()) || failed(validateControlDataflow()) ||
      failed(addFixedIssueOrder()) || failed(addForwardDependencies()) ||
      failed(addRecurrenceDependencies());
  if (failedStage) {
    return failure();
  }
  if (!graph_.freezeStructure()) {
    function_.emitError("cannot freeze canonical sync graph");
    return failure();
  }
  return CanonicalSyncProgram(
      function_, std::move(graph_), std::move(nodeBindings_),
      std::move(scopeBindings_), std::move(eventReservations_));
}

LogicalResult ProgramBuilder::validateInput() {
  if (function_.isExternal()) {
    return function_.emitError("canonical sync requires a function body");
  }
  const bool invalidLimits =
      options_.maximumNodes == 0 || options_.maximumScopes == 0 ||
      options_.maximumControls == 0 || options_.maximumStorageAccesses == 0 ||
      options_.maximumPairInspections == 0;
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
    if (isManualSync(operation)) {
      operation->emitError(
          "canonical sync does not accept pre-existing pipe synchronization");
      return WalkResult::interrupt();
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
    compounds_.push_back(compound);
    const bool nodeLimitExceeded = compounds_.size() > options_.maximumNodes;
    if (nodeLimitExceeded) {
      return compound->elementOp->emitError(
          "canonical sync node limit exceeded");
    }
  }
  return success();
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
        isCompletionOrdered(resource, function_.getOperation())
            ? SyncCoverEdgeKind::CompletionPreservingIssueOrder
            : SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
    if (!graph_.setResourceRecurrenceCarryKind(resource, kind)) {
      return function_.emitError(
          "cannot register canonical sync recurrence resource");
    }
  }

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
    if (canSignalDirectCompletion(resource)) {
      for (std::uint32_t candidate : resources) {
        if (candidate != resource) {
          completionTargets.push_back(candidate);
        }
      }
    }
    const SyncCoverGraphResult node =
        graph_.addNode(resource, 1, context->second.scope, order,
                       context->second.guard, std::move(completionTargets));
    if (!node) {
      return compound->elementOp->emitError(
          "cannot construct canonical sync operation node");
    }
    nodeBindings_.push_back({compound->elementOp, compound->macroOpInstanceId});
    operationNodes_[compound->elementOp].push_back(*node.index);
    nodeAccessIndices_.emplace_back();
    appendAccesses(*node.index, compound->useVec, false);
    appendAccesses(*node.index, compound->defVec, true);
    if (failed(materializeNodeAccesses(*node.index))) {
      return failure();
    }
  }
  indexNodesByLoop();
  if (failed(refineLoopTimelines())) {
    return failure();
  }
  collectHiddenEventReservations();
  return success();
}

FailureOr<CanonicalSyncProgram> mlir::pto::buildCanonicalSyncProgram(
    func::FuncOp function, const CanonicalSyncAnalysisOptions &options) {
  return ProgramBuilder(function, options).build();
}
