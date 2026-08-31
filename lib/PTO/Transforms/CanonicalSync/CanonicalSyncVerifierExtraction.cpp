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

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

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

  LogicalResult collectBlock(Block &block);
  LogicalResult collectOperation(Operation *operation);
  LogicalResult addPhase(Operation *operation, PIPE pipe, unsigned phase,
                         ArrayRef<Value> reads, ArrayRef<Value> writes,
                         bool useInterface);
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
  if (std::optional<SyncMacroModel> macro = getSyncMacroModel(operation)) {
    for (const SyncMacroPhase &phase : macro->phases) {
      std::optional<PIPE> pipe = convertPipe(phase.pipe);
      if (!pipe || failed(addPhase(operation, *pipe, phase.phaseId,
                                   phase.useValues, phase.defValues, false))) {
        return failure();
      }
    }
    return success();
  }
  if (isa<LoadScalarOp, StoreScalarOp>(operation)) {
    return addPhase(operation, PIPE::PIPE_S, 0, {}, {}, true);
  }
  if (auto pipeOperation = dyn_cast<OpPipeInterface>(operation)) {
    return addPhase(operation, pipeOperation.getPipe(), 0, {}, {}, true);
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
    if (interface) {
      SmallVector<MemoryEffects::EffectInstance, 8> instances;
      interface.getEffects(instances);
      for (const MemoryEffects::EffectInstance &instance : instances) {
        Value value = instance.getValue();
        if (!value) {
          continue;
        }
        if (isa<MemoryEffects::Read>(instance.getEffect())) {
          addEffect(value, kReadBit);
        }
        if (isa<MemoryEffects::Write>(instance.getEffect())) {
          addEffect(value, kWriteBit);
        }
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
    for (const AliasFact &fact : aliases.describe(value)) {
      CanonicalAccess access;
      access.mode = accessMode(entry->second);
      access.space = fact.space;
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
  result->phases[operation].push_back(std::move(phase));
  return success();
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
