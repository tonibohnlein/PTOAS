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
  LogicalResult addMacroPhases(Operation *operation, CanonicalRegionId sequence,
                               const SyncMacroModel &model);
  LogicalResult addPhase(Operation *operation, CanonicalRegionId sequence,
                         PIPE pipe, std::optional<unsigned> macroPhase,
                         ArrayRef<Value> reads, ArrayRef<Value> writes,
                         bool useMemoryInterface);
  void addAccesses(CanonicalPhaseId phase, Operation *operation,
                   const llvm::SmallDenseMap<Value, unsigned, 8> &effects,
                   ArrayRef<Value> effectOrder);
  void bindIfResults(scf::IfOp operation);
  void bindForInputs(scf::ForOp operation);
  void bindForResults(scf::ForOp operation);
};

LogicalResult ProgramBuilder::build() {
  if (!function.getBody().hasOneBlock()) {
    return function.emitError(
        "canonical sync v1 requires a structured single-block function body");
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
  if (std::optional<SyncMacroModel> macro = getSyncMacroModel(operation)) {
    return addMacroPhases(operation, sequence, *macro);
  }
  if (isa<LoadScalarOp, StoreScalarOp>(operation)) {
    return addRegularPhase(operation, sequence, PIPE::PIPE_S);
  }
  if (auto pipeOperation = dyn_cast<OpPipeInterface>(operation)) {
    return addRegularPhase(operation, sequence, pipeOperation.getPipe());
  }
  if (auto call = dyn_cast<func::CallOp>(operation)) {
    if (llvm::any_of(call.getOperandTypes(), areMemoryLikeTypes)) {
      return call.emitError("canonical sync cannot infer the physical effects "
                            "of this helper call");
    }
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
      return success();
    }
    SmallVector<MemoryEffects::EffectInstance, 8> instances;
    interface.getEffects(instances);
    for (const MemoryEffects::EffectInstance &instance : instances) {
      Value value = instance.getValue();
      const bool read = isa<MemoryEffects::Read>(instance.getEffect());
      const bool write = isa<MemoryEffects::Write>(instance.getEffect());
      if (!read && !write) {
        continue;
      }
      if (!value) {
        return operation->emitError(
            "canonical sync requires value-scoped read/write memory effects");
      }
      addEffect(value, read ? kReadBit : kWriteBit);
    }
  }
  addAccesses(phaseId, operation, effects, effectOrder);
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
    for (const AliasFact &fact : facts) {
      CanonicalAccess access;
      access.phase = phase;
      access.mode = accessMode(entry->second);
      access.space = fact.space;
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

} // namespace

LogicalResult
mlir::pto::canonical_sync_detail::buildCanonicalStructureAndAccesses(
    CanonicalSyncProgram &program, const CanonicalSyncTarget &target) {
  return ProgramBuilder(program, target).build();
}
