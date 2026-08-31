// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncVerifier.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOScheduling.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr unsigned kReadBit = 1;
constexpr unsigned kWriteBit = 2;

std::optional<PIPE> convertPipe(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
    return PIPE::PIPE_S;
  case PipelineType::PIPE_V:
    return PIPE::PIPE_V;
  case PipelineType::PIPE_M:
    return PIPE::PIPE_M;
  case PipelineType::PIPE_MTE1:
    return PIPE::PIPE_MTE1;
  case PipelineType::PIPE_MTE2:
    return PIPE::PIPE_MTE2;
  case PipelineType::PIPE_MTE3:
    return PIPE::PIPE_MTE3;
  case PipelineType::PIPE_FIX:
    return PIPE::PIPE_FIX;
  default:
    return std::nullopt;
  }
}

CanonicalAccessMode accessMode(unsigned effects) {
  if (effects == (kReadBit | kWriteBit)) {
    return CanonicalAccessMode::ReadWrite;
  }
  return effects == kWriteBit ? CanonicalAccessMode::Write
                              : CanonicalAccessMode::Read;
}

CanonicalAccessMode accessMode(const VPTOMemoryAccess &access) {
  const bool readsAndWrites = access.reads && access.writes;
  const bool hasNoMode = !access.reads && !access.writes;
  if (access.unknown || readsAndWrites || hasNoMode) {
    return CanonicalAccessMode::ReadWrite;
  }
  return access.writes ? CanonicalAccessMode::Write : CanonicalAccessMode::Read;
}

std::optional<std::pair<int64_t, int64_t>>
getVerifierScalarAccessRange(Operation *operation, Value address) {
  Value offset;
  if (auto load = dyn_cast<LoadScalarOp>(operation)) {
    const Value pointer = load.getPtr();
    if (pointer != address) {
      return std::nullopt;
    }
    offset = load.getOffset();
  } else if (auto store = dyn_cast<StoreScalarOp>(operation)) {
    const Value pointer = store.getPtr();
    if (pointer != address) {
      return std::nullopt;
    }
    offset = store.getOffset();
  } else {
    return std::nullopt;
  }
  APInt constant;
  auto pointer = dyn_cast<PtrType>(address.getType());
  if (!pointer) {
    return std::nullopt;
  }
  const bool isConstant = matchPattern(offset, m_ConstantInt(&constant));
  if (!isConstant) {
    return std::nullopt;
  }
  const bool fitsInt64 = constant.isSignedIntN(64);
  if (!fitsInt64) {
    return std::nullopt;
  }
  const int64_t elementOffset = constant.getSExtValue();
  const std::uint64_t elementBytes =
      getPTOStorageElemByteSize(pointer.getElementType());
  if (elementOffset < 0 || elementBytes == 0 ||
      elementBytes >
          static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
    return std::nullopt;
  }
  int64_t byteOffset;
  if (llvm::MulOverflow(elementOffset, static_cast<int64_t>(elementBytes),
                        byteOffset)) {
    return std::nullopt;
  }
  return std::make_pair(byteOffset, static_cast<int64_t>(elementBytes));
}

std::optional<PIPE> getVerifierNormalizedPipe(Operation *operation) {
  if (auto pipeOperation = dyn_cast<OpPipeInterface>(operation)) {
    const PIPE pipe = pipeOperation.getPipe();
    if (pipe != PIPE::PIPE_ALL && pipe != PIPE::PIPE_UNASSIGNED) {
      return pipe;
    }
    return std::nullopt;
  }
  if (isa<VectorMicroOpInterface>(operation)) {
    return PIPE::PIPE_V;
  }
  if (isa<CubeMicroOpInterface>(operation)) {
    return PIPE::PIPE_M;
  }
  if (isa<SimtOpInterface>(operation)) {
    return PIPE::PIPE_S;
  }
  return std::nullopt;
}

bool isVerifierSyncOperation(Operation *operation) {
  return isa<SetFlagOp, WaitFlagOp, BarrierOp>(operation);
}

LogicalResult
validateVerifierNormalizedEffects(Operation *operation,
                                  const VPTOSchedulingSemantics &semantics) {
  const bool noneWithAccesses =
      semantics.memoryBehavior == VPTOMemoryBehavior::None &&
      !semantics.memoryAccesses.empty();
  const bool explicitWithoutAccesses =
      semantics.memoryBehavior == VPTOMemoryBehavior::Explicit &&
      semantics.memoryAccesses.empty();
  if (noneWithAccesses || explicitWithoutAccesses) {
    return operation->emitError(
        "canonical sync verifier rejects inconsistent normalized memory "
        "semantics");
  }
  for (const VPTOMemoryAccess &access : semantics.memoryAccesses) {
    const bool unknownMode = !access.reads && !access.writes;
    const bool invalidUnknownMode = unknownMode && !access.unknown;
    const bool incompleteRange =
        access.byteOffset.has_value() != access.byteSize.has_value();
    if (invalidUnknownMode || incompleteRange) {
      return operation->emitError(
          "canonical sync verifier rejects an incomplete normalized memory "
          "access");
    }
  }
  for (const VPTOSchedulingEffect &effect : semantics.effects) {
    switch (effect.kind) {
    case VPTOSchedulingEffectKind::PostUpdate:
    case VPTOSchedulingEffectKind::VolatileMemory:
    case VPTOSchedulingEffectKind::AtomicMemory:
      break;
    case VPTOSchedulingEffectKind::ImplicitRead:
    case VPTOSchedulingEffectKind::ImplicitWrite:
    case VPTOSchedulingEffectKind::Barrier:
    case VPTOSchedulingEffectKind::Unknown:
      return operation->emitError(
          "canonical sync verifier does not model this normalized "
          "scheduling effect");
    }
  }
  return success();
}

class VerifierProgramBuilder {
public:
  explicit VerifierProgramBuilder(func::FuncOp function)
      : result(std::make_unique<VerifierProgram>()), aliases(function) {
    result->function = function;
  }

  FailureOr<std::unique_ptr<VerifierProgram>> build();

private:
  std::unique_ptr<VerifierProgram> result;
  AliasAnalysis aliases;
  llvm::DenseSet<Operation *> coveredOperations;

  LogicalResult collectBlock(Block &block);
  LogicalResult collectOperation(Operation *operation);
  LogicalResult addPhase(Operation *operation, PIPE pipe, unsigned phase,
                         ArrayRef<Value> reads, ArrayRef<Value> writes,
                         bool useInterface);
  LogicalResult addNormalizedPhase(Operation *operation, PIPE pipe,
                                   const VPTOSchedulingSemantics &semantics);
  void addCacheActions(CmoCacheInvalidOp operation);
  LogicalResult verifyExtractionCoverage();
  LogicalResult indexSsaDependencies();
  void bindIfResults(scf::IfOp operation);
  void bindForInputs(scf::ForOp operation);
  void bindForResults(scf::ForOp operation);
};

FailureOr<std::unique_ptr<VerifierProgram>> VerifierProgramBuilder::build() {
  func::FuncOp function = result->function;
  if (!function.getBody().hasOneBlock()) {
    return failure();
  }
  if (failed(collectBlock(function.getBody().front()))) {
    return failure();
  }
  if (failed(indexSsaDependencies())) {
    return failure();
  }
  if (failed(verifyExtractionCoverage())) {
    return failure();
  }
  return std::move(result);
}

LogicalResult VerifierProgramBuilder::collectBlock(Block &block) {
  for (Operation &operation : block) {
    if (failed(aliases.observe(&operation))) {
      return failure();
    }
    if (failed(collectOperation(&operation))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VerifierProgramBuilder::collectOperation(Operation *operation) {
  if (auto ifOperation = dyn_cast<scf::IfOp>(operation)) {
    if (failed(collectBlock(ifOperation.getThenRegion().front()))) {
      return failure();
    }
    const bool hasElse = !ifOperation.getElseRegion().empty();
    if (hasElse && failed(collectBlock(ifOperation.getElseRegion().front()))) {
      return failure();
    }
    bindIfResults(ifOperation);
    return success();
  }
  if (auto forOperation = dyn_cast<scf::ForOp>(operation)) {
    bindForInputs(forOperation);
    if (failed(collectBlock(forOperation.getRegion().front()))) {
      return failure();
    }
    bindForResults(forOperation);
    return success();
  }
  if (auto section = dyn_cast<SectionCubeOp>(operation)) {
    return collectBlock(section.getBody().front());
  }
  if (auto section = dyn_cast<SectionVectorOp>(operation)) {
    return collectBlock(section.getBody().front());
  }
  if (operation->hasTrait<OpTrait::IsTerminator>()) {
    return success();
  }
  if (auto cmo = dyn_cast<CmoCacheInvalidOp>(operation)) {
    addCacheActions(cmo);
    coveredOperations.insert(operation);
    return success();
  }
  const bool isFenceAll = isa<FenceBarrierAllOp>(operation);
  const bool isSync = isVerifierSyncOperation(operation);
  if (isFenceAll || isSync) {
    coveredOperations.insert(operation);
    return success();
  }
  if (std::optional<SyncMacroModel> macro = getSyncMacroModel(operation)) {
    for (const SyncMacroPhase &phase : macro->phases) {
      std::optional<PIPE> pipe = convertPipe(phase.pipe);
      if (!pipe || failed(addPhase(operation, *pipe, phase.phaseId,
                                   phase.useValues, phase.defValues, false))) {
        return failure();
      }
    }
    coveredOperations.insert(operation);
    return success();
  }
  const VPTOSchedulingSemantics semantics =
      getVPTOSchedulingSemantics(operation);
  if (isa<MicroOpInterface>(operation)) {
    const bool normalizedPhysical =
        semantics.classificationKnown &&
        semantics.schedulingClass == VPTOSchedulingClass::Schedulable;
    if (!normalizedPhysical) {
      return operation->emitError(
          "canonical sync verifier rejects a raw physical operation without "
          "schedulable normalized VPTOSchedulingSemantics");
    }
    std::optional<PIPE> pipe = getVerifierNormalizedPipe(operation);
    if (!pipe) {
      return operation->emitError(
          "canonical sync verifier cannot resolve a concrete physical pipe "
          "from normalized scheduling semantics");
    }
    if (failed(validateVerifierNormalizedEffects(operation, semantics))) {
      return failure();
    }
    return addNormalizedPhase(operation, *pipe, semantics);
  }
  if (isa<LoadScalarOp, StoreScalarOp>(operation)) {
    return addPhase(operation, PIPE::PIPE_S, 0, {}, {}, true);
  }
  if (auto pipeOperation = dyn_cast<OpPipeInterface>(operation)) {
    return addPhase(operation, pipeOperation.getPipe(), 0, {}, {}, true);
  }
  const bool physicalMarker = isa<VectorMicroOpInterface, CubeMicroOpInterface,
                                  MteOpInterface, SimtOpInterface>(operation);
  if (semantics.classificationKnown &&
      semantics.schedulingClass == VPTOSchedulingClass::Schedulable) {
    std::optional<PIPE> pipe = getVerifierNormalizedPipe(operation);
    if (!pipe) {
      return operation->emitError(
          "canonical sync verifier cannot resolve a concrete physical pipe "
          "from normalized scheduling semantics");
    }
    if (failed(validateVerifierNormalizedEffects(operation, semantics))) {
      return failure();
    }
    return addNormalizedPhase(operation, *pipe, semantics);
  }
  if (semantics.classificationKnown &&
      semantics.schedulingClass == VPTOSchedulingClass::Structural) {
    return success();
  }
  const bool isEffectful = !isPure(operation);
  const bool isUnsupported =
      semantics.schedulingClass == VPTOSchedulingClass::Unsupported;
  if (physicalMarker || isEffectful || isUnsupported) {
    return operation->emitError(
        "canonical sync verifier rejects an unclassified physical or "
        "effectful operation");
  }
  return success();
}

LogicalResult VerifierProgramBuilder::addPhase(Operation *operation, PIPE pipe,
                                               unsigned phaseIndex,
                                               ArrayRef<Value> reads,
                                               ArrayRef<Value> writes,
                                               bool useInterface) {
  FailureOr<CanonicalPhysicalResource> resource =
      resolvePhysicalResource(result->function, operation, pipe);
  if (failed(resource)) {
    return failure();
  }
  llvm::SmallDenseMap<Value, unsigned, 8> effects;
  SmallVector<Value, 8> effectOrder;
  unsigned unknownEffects = 0;
  const auto addEffect = [&](Value value, unsigned effect) {
    const bool firstEffect = effects.find(value) == effects.end();
    if (firstEffect) {
      effectOrder.push_back(value);
    }
    effects[value] |= effect;
  };
  for (Value value : reads) {
    addEffect(value, kReadBit);
  }
  for (Value value : writes) {
    addEffect(value, kWriteBit);
  }
  if (useInterface) {
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface) {
      if (!isPure(operation)) {
        return operation->emitError(
            "canonical sync verifier requires a complete memory-effect "
            "interface for this physical operation");
      }
    } else {
      SmallVector<MemoryEffects::EffectInstance, 8> instances;
      interface.getEffects(instances);
      for (const MemoryEffects::EffectInstance &instance : instances) {
        const bool read = isa<MemoryEffects::Read>(instance.getEffect());
        const bool write = isa<MemoryEffects::Write, MemoryEffects::Allocate,
                               MemoryEffects::Free>(instance.getEffect());
        if (!read && !write) {
          continue;
        }
        Value value = instance.getValue();
        const unsigned bits = (read ? kReadBit : 0) | (write ? kWriteBit : 0);
        if (!value) {
          unknownEffects |= bits;
          continue;
        }
        addEffect(value, bits);
      }
    }
  }
  VerifierPhase phase;
  phase.resource = *resource;
  phase.completion = {
      {operation, phaseIndex, std::numeric_limits<unsigned>::max()}, *resource};
  unsigned accessIndex = 0;
  for (Value value : effectOrder) {
    const auto entry = effects.find(value);
    if (entry == effects.end()) {
      continue;
    }
    SmallVector<AliasFact, 2> facts = aliases.describe(value);
    if (facts.empty()) {
      facts.push_back({});
      facts.back().unknownSpace = true;
    }
    for (const AliasFact &fact : facts) {
      CanonicalAccess access;
      access.mode = accessMode(entry->second);
      access.space = fact.space;
      access.unknownSpace = fact.unknownSpace;
      if (auto range = getVerifierScalarAccessRange(operation, value)) {
        access.addressByteOffset = range->first;
        access.addressByteSize = range->second;
      }
      access.value = value;
      access.aliasRoot = fact.root;
      access.intervals = fact.intervals;
      access.physical = fact.physical;
      access.unknownRange = fact.unknownRange;
      VerifierEffect effect;
      effect.key = {operation, phaseIndex, accessIndex++};
      effect.resource = *resource;
      effect.access = std::move(access);
      phase.effects.push_back(std::move(effect));
    }
  }
  if (unknownEffects != 0) {
    CanonicalAccess access;
    access.mode = accessMode(unknownEffects);
    access.unknownSpace = true;
    access.unknownRange = true;
    VerifierEffect effect;
    effect.key = {operation, phaseIndex, accessIndex++};
    effect.resource = *resource;
    effect.access = std::move(access);
    phase.effects.push_back(std::move(effect));
  }
  result->phases[operation].push_back(std::move(phase));
  coveredOperations.insert(operation);
  return success();
}

LogicalResult VerifierProgramBuilder::addNormalizedPhase(
    Operation *operation, PIPE pipe, const VPTOSchedulingSemantics &semantics) {
  FailureOr<CanonicalPhysicalResource> resource =
      resolvePhysicalResource(result->function, operation, pipe);
  if (failed(resource)) {
    return failure();
  }
  VerifierPhase phase;
  phase.resource = *resource;
  phase.completion = {{operation, 0, std::numeric_limits<unsigned>::max()},
                      *resource};
  unsigned accessIndex = 0;
  const auto appendUnknown = [&]() {
    CanonicalAccess access;
    access.mode = CanonicalAccessMode::ReadWrite;
    access.unknownSpace = true;
    access.unknownRange = true;
    phase.effects.push_back(
        {{operation, 0, accessIndex++}, *resource, std::move(access)});
  };
  if (semantics.memoryBehavior == VPTOMemoryBehavior::Unknown) {
    appendUnknown();
  }
  for (const VPTOMemoryAccess &memory : semantics.memoryAccesses) {
    SmallVector<AliasFact, 2> facts = aliases.describe(memory.address);
    if (facts.empty()) {
      facts.push_back({});
      facts.back().unknownSpace = !memory.addressSpace;
    }
    for (const AliasFact &fact : facts) {
      CanonicalAccess access;
      access.mode = accessMode(memory);
      access.ordered = memory.ordered;
      access.value = memory.address;
      access.aliasRoot = fact.root;
      access.space = fact.space;
      access.unknownSpace = fact.unknownSpace;
      if (auto space =
              dyn_cast_or_null<AddressSpaceAttr>(memory.addressSpace)) {
        access.space = space.getAddressSpace();
        access.unknownSpace = false;
      }
      access.intervals = fact.intervals;
      access.physical = fact.physical;
      access.unknownRange = fact.unknownRange || memory.unknown;
      const bool validRange = memory.byteOffset && memory.byteSize &&
                              *memory.byteOffset >= 0 && *memory.byteSize > 0;
      access.addressByteOffset = memory.byteOffset;
      access.addressByteSize = memory.byteSize;
      if (validRange && !memory.unknown && !fact.physical &&
          fact.root == memory.address) {
        access.intervals = {{static_cast<std::uint64_t>(*memory.byteOffset),
                             static_cast<std::uint64_t>(*memory.byteSize)}};
        access.unknownRange = false;
      }
      phase.effects.push_back(
          {{operation, 0, accessIndex++}, *resource, std::move(access)});
    }
  }
  result->phases[operation].push_back(std::move(phase));
  coveredOperations.insert(operation);
  result->normalizedOperations.insert(operation);
  return success();
}

void VerifierProgramBuilder::addCacheActions(CmoCacheInvalidOp operation) {
  const AddressSpace space = operation.getSpace().getAddressSpace();
  if (space != AddressSpace::GM && space != AddressSpace::Zero) {
    return;
  }
  Value address = operation.getAddr();
  if (!address) {
    VerifierCacheAction action;
    action.operation = operation;
    action.allGm = true;
    action.access.space = AddressSpace::GM;
    action.access.unknownRange = true;
    result->cacheActions[operation].push_back(std::move(action));
    return;
  }
  SmallVector<AliasFact, 2> facts = aliases.describe(address);
  if (facts.empty()) {
    facts.push_back({});
  }
  for (const AliasFact &fact : facts) {
    VerifierCacheAction action;
    action.operation = operation;
    action.access.space = AddressSpace::GM;
    action.access.value = address;
    action.access.aliasRoot = fact.root;
    action.access.intervals = fact.intervals;
    action.access.unknownRange = fact.unknownRange;
    result->cacheActions[operation].push_back(std::move(action));
  }
}

LogicalResult VerifierProgramBuilder::verifyExtractionCoverage() {
  WalkResult walkResult = result->function.walk([&](Operation *operation) {
    const bool isFunction = operation == result->function.getOperation();
    const bool isTerminator = operation->hasTrait<OpTrait::IsTerminator>();
    if (isFunction || isTerminator) {
      return WalkResult::advance();
    }
    const bool hasRegions = operation->getNumRegions() != 0;
    if (hasRegions) {
      const bool modeledControl =
          isa<scf::IfOp, scf::ForOp, SectionCubeOp, SectionVectorOp>(operation);
      if (!modeledControl) {
        operation->emitError(
            "canonical sync verifier extraction coverage found an unmodeled "
            "region-bearing operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    const VPTOSchedulingSemantics semantics =
        getVPTOSchedulingSemantics(operation);
    const bool physical =
        isa<OpPipeInterface, MicroOpInterface, VectorMicroOpInterface,
            CubeMicroOpInterface, MteOpInterface, SimtOpInterface>(operation);
    const bool explicitCanonical =
        getSyncMacroModel(operation).has_value() ||
        isa<LoadScalarOp, StoreScalarOp, CmoCacheInvalidOp, FenceBarrierAllOp>(
            operation) ||
        isVerifierSyncOperation(operation);
    const bool unclassifiedEffectful =
        !semantics.classificationKnown && !isPure(operation);
    const bool normalizedSchedulable =
        semantics.classificationKnown &&
        semantics.schedulingClass == VPTOSchedulingClass::Schedulable;
    const bool mustBeCovered = physical || explicitCanonical ||
                               unclassifiedEffectful || normalizedSchedulable;
    if (mustBeCovered && !coveredOperations.contains(operation)) {
      operation->emitError(
          "canonical sync verifier extraction coverage found an unmodeled "
          "physical or effectful operation");
      return WalkResult::interrupt();
    }
    const bool isMicroOperation = isa<MicroOpInterface>(operation);
    const bool normalizedSemanticsMissing =
        !result->normalizedOperations.contains(operation);
    if (isMicroOperation && normalizedSchedulable &&
        normalizedSemanticsMissing) {
      operation->emitError(
          "canonical sync verifier extraction coverage did not consume raw "
          "physical normalized scheduling semantics");
      return WalkResult::interrupt();
    }
    if (semantics.classificationKnown &&
        semantics.schedulingClass == VPTOSchedulingClass::Schedulable &&
        semantics.memoryBehavior != VPTOMemoryBehavior::None) {
      auto phases = result->phases.find(operation);
      const bool missingEffects =
          phases == result->phases.end() ||
          llvm::all_of(phases->second, [](const VerifierPhase &phase) {
            return phase.effects.empty();
          });
      if (missingEffects) {
        operation->emitError(
            "canonical sync verifier extraction coverage lost normalized "
            "memory effects");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

LogicalResult VerifierProgramBuilder::indexSsaDependencies() {
  llvm::DenseMap<Value, VerifierCompletion> completions;
  for (auto &entry : result->phases) {
    Operation *operation = entry.first;
    if (operation->getResults().empty()) {
      continue;
    }
    const bool hasUniqueCompletion = entry.second.size() == 1;
    if (!hasUniqueCompletion) {
      return operation->emitError(
          "canonical sync verifier cannot bind a multi-phase SSA result");
    }
    for (Value value : operation->getResults()) {
      completions[value] = entry.second.front().completion;
    }
  }

  const auto enqueueStructuredResult =
      [](OpResult resultValue, Operation *definition,
         SmallVectorImpl<Value> &worklist) -> LogicalResult {
    const unsigned index = resultValue.getResultNumber();
    if (auto conditional = dyn_cast<scf::IfOp>(definition)) {
      for (Region *region :
           {&conditional.getThenRegion(), &conditional.getElseRegion()}) {
        if (region->empty()) {
          continue;
        }
        auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yield || index >= yield.getNumOperands()) {
          return conditional.emitError(
              "canonical sync verifier cannot trace an scf.if result");
        }
        worklist.push_back(yield.getOperand(index));
      }
      return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(definition)) {
      auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
      const bool invalidInit = index >= loop.getInitArgs().size();
      const bool invalidYield = !yield || index >= yield.getNumOperands();
      if (invalidInit || invalidYield) {
        return loop.emitError(
            "canonical sync verifier cannot trace an scf.for result");
      }
      worklist.push_back(loop.getInitArgs()[index]);
      worklist.push_back(yield.getOperand(index));
      return success();
    }
    return success();
  };

  for (auto &entry : result->phases) {
    Operation *operation = entry.first;
    std::optional<SyncMacroModel> macro = getSyncMacroModel(operation);
    for (VerifierPhase &phase : entry.second) {
      SmallVector<Value, 8> operands;
      if (macro) {
        const auto found =
            llvm::find_if(macro->phases, [&](const SyncMacroPhase &item) {
              return item.phaseId == phase.completion.key.phase;
            });
        if (found == macro->phases.end()) {
          return operation->emitError(
              "canonical sync verifier cannot find a modeled macro phase");
        }
        operands.append(found->useValues.begin(), found->useValues.end());
      } else {
        operands.append(operation->operand_begin(), operation->operand_end());
      }
      SmallVector<VerifierEffectKey, 8> seenCompletions;
      for (Value seed : operands) {
        SmallVector<Value, 16> worklist{seed};
        llvm::DenseSet<Value> discovered;
        while (!worklist.empty()) {
          Value value = worklist.pop_back_val();
          if (!value || !discovered.insert(value).second) {
            continue;
          }
          if (auto completion = completions.find(value);
              completion != completions.end()) {
            if (!llvm::is_contained(seenCompletions, completion->second.key)) {
              seenCompletions.push_back(completion->second.key);
              phase.ssaSources.push_back(completion->second);
            }
            continue;
          }
          if (auto argument = dyn_cast<BlockArgument>(value)) {
            Block *owner = argument.getOwner();
            if (owner == &result->function.getBody().front()) {
              continue;
            }
            auto loop = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
            if (loop && argument == loop.getInductionVar()) {
              continue;
            }
            return result->function.emitError(
                "canonical sync verifier cannot trace this SSA block argument");
          }
          Operation *definition = value.getDefiningOp();
          if (!definition) {
            continue;
          }
          if (auto resultValue = dyn_cast<OpResult>(value)) {
            if (isa<scf::IfOp, scf::ForOp>(definition)) {
              if (failed(enqueueStructuredResult(resultValue, definition,
                                                 worklist))) {
                return failure();
              }
              continue;
            }
          }
          if (result->phases.contains(definition)) {
            return definition->emitError("canonical sync verifier found an SSA "
                                         "result without a completion phase");
          }
          if (areMemoryLikeTypes(value.getType())) {
            continue;
          }
          const bool hasNestedRegions = definition->getNumRegions() != 0;
          if (hasNestedRegions || !isMemoryEffectFree(definition)) {
            return definition->emitError("canonical sync verifier cannot trace "
                                         "this effectful SSA operation");
          }
          worklist.append(definition->operand_begin(),
                          definition->operand_end());
        }
      }
    }
  }
  return success();
}

void VerifierProgramBuilder::bindIfResults(scf::IfOp operation) {
  for (auto [index, resultValue] : llvm::enumerate(operation.getResults())) {
    SmallVector<AliasFact, 2> facts;
    for (Region *region :
         {&operation.getThenRegion(), &operation.getElseRegion()}) {
      if (region->empty()) {
        continue;
      }
      auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
      if (yield && index < yield.getNumOperands()) {
        SmallVector<AliasFact, 2> branch =
            aliases.describe(yield.getOperand(index));
        facts.append(branch.begin(), branch.end());
      }
    }
    aliases.bind(resultValue, facts);
  }
}

void VerifierProgramBuilder::bindForInputs(scf::ForOp operation) {
  for (auto [argument, initial] :
       llvm::zip(operation.getRegionIterArgs(), operation.getInitArgs())) {
    aliases.bind(argument, aliases.describe(initial));
  }
}

void VerifierProgramBuilder::bindForResults(scf::ForOp operation) {
  auto yield = dyn_cast<scf::YieldOp>(operation.getBody()->getTerminator());
  for (auto [index, resultValue] : llvm::enumerate(operation.getResults())) {
    SmallVector<AliasFact, 2> facts =
        aliases.describe(operation.getInitArgs()[index]);
    if (yield && index < yield.getNumOperands()) {
      SmallVector<AliasFact, 2> carried =
          aliases.describe(yield.getOperand(index));
      facts.append(carried.begin(), carried.end());
    }
    aliases.bind(resultValue, facts);
  }
}

} // namespace

FailureOr<std::unique_ptr<VerifierProgram>>
mlir::pto::canonical_sync_detail::buildVerifierProgram(func::FuncOp function) {
  return VerifierProgramBuilder(function).build();
}
