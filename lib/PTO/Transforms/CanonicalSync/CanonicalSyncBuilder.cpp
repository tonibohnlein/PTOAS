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

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOScheduling.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr unsigned kReadBit = 1;
constexpr unsigned kWriteBit = 2;

std::optional<PIPE> convertMacroPipe(PipelineType pipe) {
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

CanonicalAccessMode accessMode(unsigned bits) {
  if (bits == (kReadBit | kWriteBit)) {
    return CanonicalAccessMode::ReadWrite;
  }
  return bits == kWriteBit ? CanonicalAccessMode::Write
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
getScalarAccessRange(Operation *operation, Value address) {
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

std::optional<PIPE> getNormalizedPipe(Operation *operation) {
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

bool isCanonicalVisibilityOperation(Operation *operation) {
  return isa<CmoCacheInvalidOp, FenceBarrierAllOp>(operation);
}

LogicalResult
validateNormalizedEffects(Operation *operation,
                          const VPTOSchedulingSemantics &semantics) {
  const bool noneWithAccesses =
      semantics.memoryBehavior == VPTOMemoryBehavior::None &&
      !semantics.memoryAccesses.empty();
  const bool explicitWithoutAccesses =
      semantics.memoryBehavior == VPTOMemoryBehavior::Explicit &&
      semantics.memoryAccesses.empty();
  if (noneWithAccesses || explicitWithoutAccesses) {
    return operation->emitError(
        "canonical sync rejects inconsistent normalized memory semantics");
  }
  for (const VPTOMemoryAccess &access : semantics.memoryAccesses) {
    const bool unknownMode = !access.reads && !access.writes;
    const bool invalidUnknownMode = unknownMode && !access.unknown;
    const bool incompleteRange =
        access.byteOffset.has_value() != access.byteSize.has_value();
    if (invalidUnknownMode || incompleteRange) {
      return operation->emitError(
          "canonical sync rejects an incomplete normalized memory access");
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
          "canonical sync does not yet model this normalized scheduling "
          "effect; lower it or provide an explicit canonical contract");
    }
  }
  return success();
}

class ProgramBuilder {
public:
  ProgramBuilder(CanonicalSyncProgram &program,
                 const CanonicalSyncTarget &target)
      : program(program), target(target), function(program.getFunction()),
        aliases(function) {}

  LogicalResult build();

private:
  CanonicalSyncProgram &program;
  const CanonicalSyncTarget &target;
  func::FuncOp function;
  AliasAnalysis aliases;
  unsigned sourceOrder = 0;
  SmallVector<CanonicalControlAtom, 2> controlPath;
  SmallVector<CanonicalRegionId, 2> loopPath;

  LogicalResult buildBlock(Block &block, CanonicalRegionId parent,
                           CanonicalCardinality cardinality, unsigned arm = 0);
  LogicalResult visitOperation(Operation *operation,
                               CanonicalRegionId sequence);
  LogicalResult visitIf(scf::IfOp operation, CanonicalRegionId sequence);
  LogicalResult visitFor(scf::ForOp operation, CanonicalRegionId sequence);
  LogicalResult visitSection(Operation *operation, Region &body,
                             CanonicalRegionId sequence);
  LogicalResult addRegularPhase(Operation *operation,
                                CanonicalRegionId sequence, PIPE pipe);
  LogicalResult addNormalizedPhase(Operation *operation,
                                   CanonicalRegionId sequence, PIPE pipe,
                                   const VPTOSchedulingSemantics &semantics);
  LogicalResult addMacroPhases(Operation *operation, CanonicalRegionId sequence,
                               const SyncMacroModel &model);
  LogicalResult addPhase(Operation *operation, CanonicalRegionId sequence,
                         PIPE pipe, std::optional<unsigned> macroPhase,
                         ArrayRef<Value> reads, ArrayRef<Value> writes,
                         bool useMemoryInterface);
  void addAccesses(CanonicalPhaseId phase, Operation *operation,
                   const llvm::SmallDenseMap<Value, unsigned, 8> &effects,
                   ArrayRef<Value> effectOrder);
  void addNormalizedAccesses(CanonicalPhaseId phase, Operation *operation,
                             const VPTOSchedulingSemantics &semantics);
  void appendAccess(CanonicalPhaseId phase, Operation *operation,
                    CanonicalAccessMode mode, Value value,
                    Attribute addressSpace, std::optional<int64_t> byteOffset,
                    std::optional<int64_t> byteSize, bool forceUnknown,
                    bool ordered = false);
  void bindIfResults(scf::IfOp operation);
  void bindForInputs(scf::ForOp operation);
  void bindForResults(scf::ForOp operation);
};

LogicalResult ProgramBuilder::build() {
  if (!function.getBody().hasOneBlock()) {
    return function.emitError(
        "canonical sync v1 requires a structured single-block function body");
  }
  if (failed(aliases.solve())) {
    return failure();
  }
  CanonicalRegion root;
  root.kind = CanonicalRegionKind::Function;
  root.cardinality = CanonicalCardinality::ExactlyOnce;
  root.operation = function;
  root.parent = kInvalidCanonicalSyncId;
  const CanonicalRegionId rootId = program.appendRegion(std::move(root));
  return buildBlock(function.getBody().front(), rootId,
                    CanonicalCardinality::ExactlyOnce);
}

LogicalResult ProgramBuilder::buildBlock(Block &block, CanonicalRegionId parent,
                                         CanonicalCardinality cardinality,
                                         unsigned arm) {
  CanonicalRegion sequence;
  sequence.parent = parent;
  sequence.kind = CanonicalRegionKind::Sequence;
  sequence.cardinality = cardinality;
  sequence.operation = block.getParentOp();
  sequence.depth = program.getRegion(parent).depth + 1;
  sequence.arm = arm;
  const CanonicalRegionId sequenceId =
      program.appendRegion(std::move(sequence));
  for (Operation &operation : block) {
    if (failed(aliases.observe(&operation))) {
      return failure();
    }
    if (failed(visitOperation(&operation, sequenceId))) {
      return failure();
    }
  }
  return success();
}

LogicalResult ProgramBuilder::visitOperation(Operation *operation,
                                             CanonicalRegionId sequence) {
  if (auto ifOperation = dyn_cast<scf::IfOp>(operation)) {
    return visitIf(ifOperation, sequence);
  }
  if (auto forOperation = dyn_cast<scf::ForOp>(operation)) {
    return visitFor(forOperation, sequence);
  }
  if (isa<scf::WhileOp>(operation)) {
    return operation->emitError(
        "canonical sync v1 does not support effectful scf.while regions");
  }
  if (auto section = dyn_cast<SectionCubeOp>(operation)) {
    return visitSection(operation, section.getBody(), sequence);
  }
  if (auto section = dyn_cast<SectionVectorOp>(operation)) {
    return visitSection(operation, section.getBody(), sequence);
  }
  const bool hasNestedRegions = operation->getNumRegions() != 0;
  if (hasNestedRegions) {
    return operation->emitError(
        "canonical sync v1 does not support this region-bearing operation");
  }
  const bool isTerminator = operation->hasTrait<OpTrait::IsTerminator>();
  const bool isVisibility = isCanonicalVisibilityOperation(operation);
  if (isTerminator || isVisibility) {
    return success();
  }
  if (std::optional<SyncMacroModel> macro = getSyncMacroModel(operation)) {
    return addMacroPhases(operation, sequence, *macro);
  }
  const VPTOSchedulingSemantics semantics =
      getVPTOSchedulingSemantics(operation);
  if (isa<MicroOpInterface>(operation)) {
    const bool normalizedPhysical =
        semantics.classificationKnown &&
        semantics.schedulingClass == VPTOSchedulingClass::Schedulable;
    if (!normalizedPhysical) {
      return operation->emitError(
          "canonical sync rejects a raw physical operation without "
          "schedulable normalized VPTOSchedulingSemantics");
    }
    std::optional<PIPE> pipe = getNormalizedPipe(operation);
    if (!pipe) {
      return operation->emitError(
          "canonical sync normalized scheduling semantics do not identify a "
          "concrete physical pipe");
    }
    if (failed(validateNormalizedEffects(operation, semantics))) {
      return failure();
    }
    return addNormalizedPhase(operation, sequence, *pipe, semantics);
  }
  if (isa<LoadScalarOp, StoreScalarOp>(operation)) {
    return addRegularPhase(operation, sequence, PIPE::PIPE_S);
  }
  if (auto pipeOperation = dyn_cast<OpPipeInterface>(operation)) {
    return addRegularPhase(operation, sequence, pipeOperation.getPipe());
  }
  const bool physicalMarker = isa<VectorMicroOpInterface, CubeMicroOpInterface,
                                  MteOpInterface, SimtOpInterface>(operation);
  if (semantics.classificationKnown &&
      semantics.schedulingClass == VPTOSchedulingClass::Schedulable) {
    std::optional<PIPE> pipe = getNormalizedPipe(operation);
    if (!pipe) {
      return operation->emitError(
          "canonical sync normalized scheduling semantics do not identify a "
          "concrete physical pipe");
    }
    if (failed(validateNormalizedEffects(operation, semantics))) {
      return failure();
    }
    return addNormalizedPhase(operation, sequence, *pipe, semantics);
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
        "canonical sync rejects an unclassified physical or effectful "
        "operation; provide normalized VPTOSchedulingSemantics");
  }
  return success();
}

LogicalResult ProgramBuilder::visitSection(Operation *operation, Region &body,
                                           CanonicalRegionId sequence) {
  if (!body.hasOneBlock()) {
    return operation->emitError(
        "canonical sync requires a single-block physical section");
  }
  CanonicalRegion transparent;
  transparent.parent = sequence;
  transparent.kind = CanonicalRegionKind::Transparent;
  transparent.cardinality = CanonicalCardinality::ExactlyOnce;
  transparent.operation = operation;
  transparent.depth = program.getRegion(sequence).depth + 1;
  const CanonicalRegionId region = program.appendRegion(std::move(transparent));
  return buildBlock(body.front(), region, CanonicalCardinality::ExactlyOnce);
}

LogicalResult ProgramBuilder::visitIf(scf::IfOp operation,
                                      CanonicalRegionId sequence) {
  CanonicalRegion choice;
  choice.parent = sequence;
  choice.kind = CanonicalRegionKind::Choice;
  choice.cardinality = CanonicalCardinality::ExactlyOnce;
  choice.operation = operation;
  choice.depth = program.getRegion(sequence).depth + 1;
  const CanonicalRegionId choiceId = program.appendRegion(std::move(choice));

  controlPath.push_back({choiceId, 0});
  if (failed(buildBlock(operation.getThenRegion().front(), choiceId,
                        CanonicalCardinality::ZeroOrOne, 0))) {
    return failure();
  }
  controlPath.pop_back();
  if (!operation.getElseRegion().empty()) {
    controlPath.push_back({choiceId, 1});
    if (failed(buildBlock(operation.getElseRegion().front(), choiceId,
                          CanonicalCardinality::ZeroOrOne, 1))) {
      return failure();
    }
    controlPath.pop_back();
  }
  bindIfResults(operation);
  return success();
}

void ProgramBuilder::bindIfResults(scf::IfOp operation) {
  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    SmallVector<AliasFact, 2> resultFacts;
    for (Region *region :
         {&operation.getThenRegion(), &operation.getElseRegion()}) {
      if (region->empty()) {
        continue;
      }
      auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
      if (yield && index < yield.getNumOperands()) {
        SmallVector<AliasFact, 2> yielded =
            aliases.describe(yield.getOperand(index));
        resultFacts.append(yielded.begin(), yielded.end());
      }
    }
    aliases.bind(result, resultFacts);
  }
}

LogicalResult ProgramBuilder::visitFor(scf::ForOp operation,
                                       CanonicalRegionId sequence) {
  CanonicalRegion loop;
  loop.parent = sequence;
  loop.kind = CanonicalRegionKind::Loop;
  loop.cardinality = CanonicalCardinality::ZeroOrMore;
  loop.operation = operation;
  loop.depth = program.getRegion(sequence).depth + 1;
  const CanonicalRegionId loopId = program.appendRegion(std::move(loop));
  bindForInputs(operation);
  loopPath.push_back(loopId);
  const LogicalResult result = buildBlock(operation.getRegion().front(), loopId,
                                          CanonicalCardinality::ZeroOrMore);
  loopPath.pop_back();
  if (succeeded(result)) {
    bindForResults(operation);
  }
  return result;
}

void ProgramBuilder::bindForInputs(scf::ForOp operation) {
  for (auto [argument, initial] :
       llvm::zip(operation.getRegionIterArgs(), operation.getInitArgs())) {
    aliases.bind(argument, aliases.describe(initial));
  }
}

void ProgramBuilder::bindForResults(scf::ForOp operation) {
  auto yield = dyn_cast<scf::YieldOp>(operation.getBody()->getTerminator());
  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    SmallVector<AliasFact, 2> resultFacts =
        aliases.describe(operation.getInitArgs()[index]);
    if (yield && index < yield.getNumOperands()) {
      SmallVector<AliasFact, 2> yielded =
          aliases.describe(yield.getOperand(index));
      resultFacts.append(yielded.begin(), yielded.end());
    }
    aliases.bind(result, resultFacts);
  }
}

LogicalResult ProgramBuilder::addRegularPhase(Operation *operation,
                                              CanonicalRegionId sequence,
                                              PIPE pipe) {
  return addPhase(operation, sequence, pipe, std::nullopt, {}, {}, true);
}

LogicalResult
ProgramBuilder::addNormalizedPhase(Operation *operation,
                                   CanonicalRegionId sequence, PIPE pipe,
                                   const VPTOSchedulingSemantics &semantics) {
  if (failed(
          addPhase(operation, sequence, pipe, std::nullopt, {}, {}, false))) {
    return failure();
  }
  CanonicalPhaseId phase = program.getPhases().back().id;
  addNormalizedAccesses(phase, operation, semantics);
  return success();
}

LogicalResult ProgramBuilder::addMacroPhases(Operation *operation,
                                             CanonicalRegionId sequence,
                                             const SyncMacroModel &model) {
  for (const SyncMacroPhase &phase : model.phases) {
    std::optional<PIPE> pipe = convertMacroPipe(phase.pipe);
    if (!pipe) {
      return operation->emitError(
          "canonical sync macro model contains an unsupported physical pipe");
    }
    if (failed(addPhase(operation, sequence, *pipe, phase.phaseId,
                        phase.useValues, phase.defValues, false))) {
      return failure();
    }
  }
  return success();
}

LogicalResult ProgramBuilder::addPhase(Operation *operation,
                                       CanonicalRegionId sequence, PIPE pipe,
                                       std::optional<unsigned> macroPhase,
                                       ArrayRef<Value> reads,
                                       ArrayRef<Value> writes,
                                       bool useMemoryInterface) {
  if (pipe == PIPE::PIPE_UNASSIGNED || pipe == PIPE::PIPE_ALL) {
    return operation->emitError(
        "canonical sync requires a concrete physical execution pipe");
  }
  FailureOr<CanonicalPhysicalResource> resource =
      resolvePhysicalResource(function, operation, pipe);
  if (failed(resource)) {
    return failure();
  }
  if (!target.supportsResource(*resource)) {
    return operation->emitError(
               "physical resource is absent from the verified NPU 2201 model: ")
           << stringifyCanonicalCore(resource->core) << ':'
           << stringifyPIPE(pipe);
  }
  CanonicalPhase phase;
  phase.region = sequence;
  phase.resource = *resource;
  phase.operation = operation;
  phase.sourceOrder = sourceOrder++;
  phase.macroPhase = macroPhase;
  phase.controlPath = controlPath;
  phase.loopPath = loopPath;
  const CanonicalPhaseId phaseId = program.appendPhase(std::move(phase));

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
  if (useMemoryInterface) {
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface) {
      if (isPure(operation)) {
        return success();
      }
      return operation->emitError(
          "canonical sync requires a complete memory-effect interface for "
          "this physical operation");
    }
    SmallVector<MemoryEffects::EffectInstance, 8> instances;
    interface.getEffects(instances);
    for (const MemoryEffects::EffectInstance &instance : instances) {
      Value value = instance.getValue();
      const bool read = isa<MemoryEffects::Read>(instance.getEffect());
      const bool write = isa<MemoryEffects::Write, MemoryEffects::Allocate,
                             MemoryEffects::Free>(instance.getEffect());
      if (!read && !write) {
        continue;
      }
      if (!value) {
        unknownEffects |= (read ? kReadBit : 0) | (write ? kWriteBit : 0);
        continue;
      }
      addEffect(value, (read ? kReadBit : 0) | (write ? kWriteBit : 0));
    }
  }
  addAccesses(phaseId, operation, effects, effectOrder);
  if (unknownEffects != 0) {
    appendAccess(phaseId, operation, accessMode(unknownEffects), Value(),
                 Attribute(), std::nullopt, std::nullopt,
                 /*forceUnknown=*/true);
  }
  return success();
}

void ProgramBuilder::addAccesses(
    CanonicalPhaseId phase, Operation *operation,
    const llvm::SmallDenseMap<Value, unsigned, 8> &effects,
    ArrayRef<Value> effectOrder) {
  for (Value value : effectOrder) {
    const auto entry = effects.find(value);
    if (entry == effects.end()) {
      continue;
    }
    SmallVector<AliasFact, 2> facts = aliases.describe(value);
    if (facts.empty()) {
      appendAccess(phase, operation, accessMode(entry->second), value,
                   Attribute(), std::nullopt, std::nullopt,
                   /*forceUnknown=*/true);
      continue;
    }
    for (const AliasFact &fact : facts) {
      CanonicalAccess access;
      access.phase = phase;
      access.mode = accessMode(entry->second);
      access.space = fact.space;
      access.unknownSpace = fact.unknownSpace;
      if (auto range = getScalarAccessRange(operation, value)) {
        access.addressByteOffset = range->first;
        access.addressByteSize = range->second;
      }
      access.value = value;
      access.aliasRoot = fact.root;
      access.intervals = fact.intervals;
      access.physical = fact.physical;
      access.unknownRange = fact.unknownRange;
      access.slotExpression = fact.slotExpression;
      access.provenance = operation->getName().getStringRef().str();
      program.appendAccess(std::move(access));
    }
  }
}

void ProgramBuilder::addNormalizedAccesses(
    CanonicalPhaseId phase, Operation *operation,
    const VPTOSchedulingSemantics &semantics) {
  if (semantics.memoryBehavior == VPTOMemoryBehavior::Unknown) {
    appendAccess(phase, operation, CanonicalAccessMode::ReadWrite, Value(),
                 Attribute(), std::nullopt, std::nullopt,
                 /*forceUnknown=*/true);
  }
  for (const VPTOMemoryAccess &memory : semantics.memoryAccesses) {
    appendAccess(phase, operation, accessMode(memory), memory.address,
                 memory.addressSpace, memory.byteOffset, memory.byteSize,
                 memory.unknown, memory.ordered);
  }
}

void ProgramBuilder::appendAccess(CanonicalPhaseId phase, Operation *operation,
                                  CanonicalAccessMode mode, Value value,
                                  Attribute addressSpace,
                                  std::optional<int64_t> byteOffset,
                                  std::optional<int64_t> byteSize,
                                  bool forceUnknown, bool ordered) {
  SmallVector<AliasFact, 2> facts = aliases.describe(value);
  if (facts.empty()) {
    facts.push_back({});
    facts.back().unknownSpace = !addressSpace;
  }
  for (const AliasFact &fact : facts) {
    CanonicalAccess access;
    access.phase = phase;
    access.mode = mode;
    access.ordered = ordered;
    access.value = value;
    access.aliasRoot = fact.root;
    access.space = fact.space;
    access.unknownSpace = fact.unknownSpace;
    if (auto space = dyn_cast_or_null<AddressSpaceAttr>(addressSpace)) {
      access.space = space.getAddressSpace();
      access.unknownSpace = false;
    } else if (!value && !addressSpace) {
      access.unknownSpace = true;
    }
    access.intervals = fact.intervals;
    access.physical = fact.physical;
    access.unknownRange = fact.unknownRange || forceUnknown;
    access.slotExpression = fact.slotExpression;
    const bool validRange =
        byteOffset && byteSize && *byteOffset >= 0 && *byteSize > 0;
    access.addressByteOffset = byteOffset;
    access.addressByteSize = byteSize;
    if (validRange && !forceUnknown && !fact.physical && fact.root == value) {
      access.intervals = {{static_cast<std::uint64_t>(*byteOffset),
                           static_cast<std::uint64_t>(*byteSize)}};
      access.unknownRange = false;
    }
    access.provenance = operation->getName().getStringRef().str();
    program.appendAccess(std::move(access));
  }
}

} // namespace

LogicalResult
mlir::pto::canonical_sync_detail::buildCanonicalStructureAndAccesses(
    CanonicalSyncProgram &program, const CanonicalSyncTarget &target) {
  return ProgramBuilder(program, target).build();
}
