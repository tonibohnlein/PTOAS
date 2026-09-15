// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/OAHS/Plan.h"
#include "PTO/Transforms/OAHS/StorageWitnesses.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncOriginClosure.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <functional>
#include <optional>
#include <limits>

namespace mlir::pto::oahs {
namespace {
std::optional<Pipe> pipe(PIPE value) {
  switch (value) {
  case PIPE::PIPE_S: return Pipe::S;
  case PIPE::PIPE_V: return Pipe::V;
  case PIPE::PIPE_M: return Pipe::M;
  case PIPE::PIPE_MTE1: return Pipe::MTE1;
  case PIPE::PIPE_MTE2: return Pipe::MTE2;
  case PIPE::PIPE_MTE3: return Pipe::MTE3;
  case PIPE::PIPE_FIX: return Pipe::FIX;
  case PIPE::PIPE_ALL:
  case PIPE::PIPE_UNASSIGNED:
  default: return {};
  }
}
PipelineType nativePipe(Pipe value) {
  switch (value) {
  case Pipe::S: return PipelineType::PIPE_S;
  case Pipe::V: return PipelineType::PIPE_V;
  case Pipe::M: return PipelineType::PIPE_M;
  case Pipe::MTE1: return PipelineType::PIPE_MTE1;
  case Pipe::MTE2: return PipelineType::PIPE_MTE2;
  case Pipe::MTE3: return PipelineType::PIPE_MTE3;
  case Pipe::FIX: return PipelineType::PIPE_FIX;
  case Pipe::Count: return PipelineType::PIPE_UNASSIGNED;
  }
  return PipelineType::PIPE_UNASSIGNED;
}
struct Import {
  Program program;
  SmallVector<mlir::Operation *> payload;
};
LogicalResult import(func::FuncOp function, Import &out) {
  if (!function.getBody().hasOneBlock())
    return function.emitError("handoff: function requires one CFG entry block");
  // Audit before production translation, which may otherwise skip an unknown
  // operation. Completeness comes from the operation declaration, never a name.
  WalkResult audited = function.walk([&](mlir::Operation *op) {
    if (op == function.getOperation() ||
        isa<func::ReturnOp, scf::YieldOp, scf::ConditionOp>(op) ||
        isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op))
      return WalkResult::advance();
    if (op->getNumRegions()) {
      op->emitError("handoff: unrepresented structured operation");
      return WalkResult::interrupt();
    }
    if (getSyncMacroModel(op)) {
      if (!isa<MacroSyncOpInterface>(op)) {
        op->emitError("handoff: macro model lacks a completeness declaration");
        return WalkResult::interrupt();
      }
      op->emitError("handoff: complete macro analysis import is not connected");
      return WalkResult::interrupt();
    }
    if (isa<AuthoredSyncOpInterface>(op)) {
      op->emitError("handoff: authored protocol import is not connected");
      return WalkResult::interrupt();
    }
    if (isa<VisibilitySyncOpInterface>(op)) {
      op->emitError("handoff: visibility boundary import is not connected");
      return WalkResult::interrupt();
    }
    if (auto physical = dyn_cast<OpPipeInterface>(op)) {
      auto contract = dyn_cast<SinglePhaseSyncOpInterface>(op);
      if (!contract || !contract.hasCompleteSinglePhaseSyncEffects())
        return op->emitError("handoff: incomplete single-phase semantic declaration"),
               WalkResult::interrupt();
      if (!pipe(physical.getPipe()))
        return op->emitError("handoff: pipeline outside native target contract"),
               WalkResult::interrupt();
      out.payload.push_back(op);
      return WalkResult::advance();
    }
    if (isa<SyncStorageOpInterface>(op) || isMemoryEffectFree(op))
      return WalkResult::advance();
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    SmallVector<MemoryEffects::EffectInstance> entries;
    if (effects) effects.getEffects(entries);
    // Allocation declares storage, not an asynchronous payload phase.
    if (!effects || entries.empty() || llvm::any_of(entries, [&](const auto &e) {
          return !isa<MemoryEffects::Allocate>(e.getEffect()) || !e.getValue() ||
                 e.getValue().getDefiningOp() != op;
        })) {
      op->emitError("handoff: unrepresented operation effects");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (audited.wasInterrupted()) return failure();

  MemoryDependentAnalyzer aliases;
  SyncIRs translated;
  Buffer2MemInfoMap buffers;
  PTOIRTranslator translator(translated, aliases, buffers, function, SyncAnalysisMode::NORMALSYNC);
  if (failed(translator.Build())) return failure();
  if (failed(closeStructuredSyncOrigins(function, translated, buffers)))
    return failure();
  DenseMap<mlir::Operation *, CompoundInstanceElement *> phases;
  for (const auto &entry : translated) {
    if (auto *phase = dyn_cast<CompoundInstanceElement>(entry.get())) {
      if (phase->macroOpInstanceId >= 0 || phases.count(phase->elementOp))
        return function.emitError("handoff: translated phase population mismatch");
      phases[phase->elementOp] = phase;
    } else if (!isa<PlaceHolderInstanceElement, LoopInstanceElement,
                    BranchInstanceElement>(entry.get()))
      return function.emitError("handoff: unexpected translated control");
  }
  if (phases.size() != out.payload.size())
    return function.emitError("handoff: payload population differs from translation");

  DenseMap<mlir::Operation *, unsigned> operationIds;
  for (auto [i, op] : llvm::enumerate(out.payload)) operationIds[op] = i;
  std::function<Region(mlir::Region &)> importRegion =
      [&](mlir::Region &region) -> Region {
    Region sequence;
    sequence.kind = Region::Sequence;
    for (Block &block : region) {
      for (mlir::Operation &op : block) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          Region choice;
          choice.kind = Region::Choice;
          choice.children.push_back(importRegion(ifOp.getThenRegion()));
          choice.children.push_back(importRegion(ifOp.getElseRegion()));
          sequence.children.push_back(std::move(choice));
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          Region loop;
          loop.kind = Region::For;
          loop.zeroTripPossible = true;
          loop.children.push_back(importRegion(forOp.getRegion()));
          sequence.children.push_back(std::move(loop));
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
          Region loop;
          loop.kind = Region::While;
          loop.children.push_back(importRegion(whileOp.getBefore()));
          loop.children.push_back(importRegion(whileOp.getAfter()));
          sequence.children.push_back(std::move(loop));
        } else if (auto found = operationIds.find(&op);
                   found != operationIds.end()) {
          Region phase;
          phase.kind = Region::Operation;
          phase.operation = found->second;
          sequence.children.push_back(std::move(phase));
        }
      }
    }
    return sequence;
  };
  out.program.body = importRegion(function.getBody());

  struct Effect { unsigned operation; const BaseMemInfo *memory; bool write; };
  SmallVector<Effect> physicalEffects;
  // A missing absolute local address is not a distinct-allocation proof. Keep
  // legacy alias behavior unchanged; normalize the handoff import's private
  // records instead. Stable copies also preserve the original SSA effect names.
  SmallVector<std::unique_ptr<BaseMemInfo>> widenedMemory;
  DenseMap<const BaseMemInfo *, const BaseMemInfo *> normalizedMemory;
  auto qualifyMemory = [&](const BaseMemInfo *memory) -> const BaseMemInfo * {
    auto found = normalizedMemory.find(memory);
    if (found != normalizedMemory.end()) return found->second;
    const bool local = memory->scope != AddressSpace::GM &&
                       memory->scope != AddressSpace::Zero;
    const bool overflowing = llvm::any_of(memory->baseAddresses, [&](uint64_t base) {
      return memory->allocateSize > std::numeric_limits<uint64_t>::max() - base;
    });
    if (local && (!memory->hasKnownPhysicalAddresses || !memory->allocateSize ||
                  memory->baseAddresses.empty() || overflowing)) {
      auto copy = memory->clone();
      copy->aliasesUnknownRange = true;
      copy->hasKnownPhysicalAddresses = false;
      widenedMemory.push_back(std::move(copy));
      normalizedMemory[memory] = widenedMemory.back().get();
    } else {
      normalizedMemory[memory] = memory;
    }
    return normalizedMemory.lookup(memory);
  };
  for (unsigned i = 0; i < out.payload.size(); ++i) {
    auto *op = out.payload[i];
    auto *phase = phases.lookup(op);
    auto declared = dyn_cast<MemoryEffectOpInterface>(op);
    if (!phase || !declared || phase->kPipeValue !=
        static_cast<PipelineType>(cast<OpPipeInterface>(op).getPipe()))
      return op->emitError("handoff: translated pipeline or effects mismatch");
    SmallVector<MemoryEffects::EffectInstance> effects;
    declared.getEffects(effects);
    for (const auto &e : effects) {
      if (!e.getValue() || (!isa<MemoryEffects::Read>(e.getEffect()) &&
                            !isa<MemoryEffects::Write>(e.getEffect())))
        return op->emitError("handoff: nonordinary declared memory effect");
      const auto &mapped = isa<MemoryEffects::Write>(e.getEffect()) ? phase->defVec : phase->useVec;
      if (llvm::none_of(mapped, [&](const BaseMemInfo *m) { return m->baseBuffer == e.getValue(); }))
        return op->emitError("handoff: declared effect missing from translation");
    }
    Operation imported;
    imported.pipe = *pipe(cast<OpPipeInterface>(op).getPipe());
    imported.original = i;
    imported.complete = true;
    out.program.operations.push_back(std::move(imported));
    auto add = [&](const auto &memories, bool write) -> LogicalResult {
      for (const BaseMemInfo *memory : memories) {
        if (llvm::none_of(effects, [&](const auto &e) {
              return e.getValue() == memory->baseBuffer &&
                     (write ? isa<MemoryEffects::Write>(e.getEffect()) : isa<MemoryEffects::Read>(e.getEffect()));
            })) return op->emitError("handoff: undeclared translated effect");
        // Admit the same translated storage scopes as production autosync.
        // Visibility remains a separate contract below; ordinary GM byte
        // completion is intentionally governed by production alias behavior.
        physicalEffects.push_back({i, qualifyMemory(memory), write});
      }
      return success();
    };
    if (failed(add(phase->defVec, true)) || failed(add(phase->useVec, false))) return failure();
  }
  // Group only repeated uses of the identical immutable imported record.
  // This is not a may-alias equivalence closure. Origin alternatives and views
  // with distinct records retain separate groups and pairwise overlap tests.
  DenseMap<const BaseMemInfo *, unsigned> groupIds;
  std::vector<const BaseMemInfo *> memories;
  std::vector<FootprintGroup> groups;
  for (const Effect &effect : physicalEffects) {
    auto found = groupIds.find(effect.memory);
    unsigned group;
    if (found == groupIds.end()) {
      group = groups.size(); groupIds[effect.memory] = group;
      memories.push_back(effect.memory);
      const auto *memory = effect.memory;
      FootprintGroup entry;
      entry.description.addressSpace = std::to_string(static_cast<unsigned>(memory->scope));
      entry.description.provenance = "original footprint (including self recurrence)";
      entry.description.unknownRange = memory->aliasesUnknownRange || memory->baseAddresses.empty();
      for (uint64_t base : memory->baseAddresses)
        entry.description.ranges.push_back({base, memory->allocateSize});
      groups.push_back(std::move(entry));
    } else group = found->second;
    groups[group].uses.push_back({effect.operation, !effect.write, effect.write});
  }
  appendStorageWitnesses(out.program, groups, [&](std::size_t a, std::size_t b) {
    return aliases.MemAlias(memories[a], memories[b]);
  });
  // Select the physical core, not a complete graph over unrelated pipelines.
  bool vector = false, cube = false;
  auto kind = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
  if (kind) {
    vector = kind.getKernelKind() == FunctionKernelKind::Vector;
    cube = kind.getKernelKind() == FunctionKernelKind::Cube;
  }
  for (const auto &phase : out.program.operations) {
    vector |= phase.pipe == Pipe::V;
    cube |= phase.pipe == Pipe::M || phase.pipe == Pipe::MTE1 || phase.pipe == Pipe::FIX;
  }
  for (const auto &effect : physicalEffects) {
    auto space = effect.memory->scope;
    vector |= space == AddressSpace::VEC;
    cube |= space == AddressSpace::MAT || space == AddressSpace::LEFT ||
            space == AddressSpace::RIGHT || space == AddressSpace::ACC ||
            space == AddressSpace::BIAS || space == AddressSpace::SCALING;
    if (space == AddressSpace::Zero)
      return function.emitError("handoff: unresolved storage address space");
  }
  if (vector && cube)
    return function.emitError("handoff: mixed physical core context requires explicit sections");
  // A payload-free function has no core-specific obligations. A pure DMA
  // function derives its core from the actual local storage above.
  out.program.target = a3SyncProfile(cube ? SyncCore::Cube : SyncCore::Vector);
  for (const auto &phase : out.program.operations)
    if (!out.program.target.supported[unsigned(phase.pipe)])
      return function.emitError("handoff: operation outside selected core profile");
  out.program.invocation.retirement =
      Program::InvocationContract::DrainAllAtReturn;
  out.program.invocation.alias =
      "production MemoryDependentAnalyzer compatibility contract";
  out.program.invocation.visibility =
      "ordinary GM communication follows production alias behavior; no new hardware guarantee";
  out.program.invocation.boundary =
      "kernel return requires completion of asynchronous physical phases";
  return success();
}

LogicalResult execute(func::FuncOp function,
    llvm::function_ref<void(func::FuncOp)> mutate = {}) {
  Import input;
  if (failed(import(function, input))) return failure();
  // Snapshot every payload and storage declaration, including operands, types,
  // attributes and ordering. Reconstruction must retain the imported contract.
  std::string originalIR;
  llvm::raw_string_ostream originalStream(originalIR);
  function.print(originalStream, OpPrintingFlags().printGenericOpForm());
  originalStream.flush();
  SmallVector<mlir::Operation *> originalOperations;
  function.walk([&](mlir::Operation *op) {
    if (op != function.getOperation()) originalOperations.push_back(op);
  });
  auto result = construct(input.program);
  if (!result.success) return function.emitError("handoff: ") << result.reason;
  // Use production emission directly. Do not run legacy planning, motion,
  // redundancy removal or allocation on the new constructor's chosen plan.
  SyncIRs emission;
  SmallVector<std::unique_ptr<SyncOperation>> storage;
  for (unsigned cut = 0; cut < result.commands.size(); ++cut) {
    auto anchor = std::make_unique<PlaceHolderInstanceElement>(cut, 0);
    anchor->elementOp = cut == input.payload.size()
        ? function.getBody().front().getTerminator() : input.payload[cut];
    for (const auto &command : result.commands[cut]) {
      auto type = command.kind == Command::Publish ? SyncOperation::TYPE::SET_EVENT :
          command.kind == Command::Acquire ? SyncOperation::TYPE::WAIT_EVENT : SyncOperation::TYPE::PIPE_BARRIER;
      auto source = command.kind == Command::BarrierAll ? PipelineType::PIPE_ALL : nativePipe(command.source);
      auto sync = std::make_unique<SyncOperation>(type, source, nativePipe(command.observer),
          storage.size(), cut, std::nullopt);
      sync->eventIds.push_back(command.key);
      anchor->pipeBefore.push_back(sync.get());
      storage.push_back(std::move(sync));
    }
    emission.push_back(std::move(anchor));
  }
  SyncCodegen codegen(emission, function, SyncAnalysisMode::NORMALSYNC);
  codegen.Run();

  // Test hook on the transactional working copy, before any acceptance check.
  if (mutate) mutate(function);
  if (failed(mlir::verify(function))) return failure();
  Commands actual(input.payload.size() + 1);
  llvm::SmallPtrSet<mlir::Operation *, 32> originalSet;
  originalSet.insert(originalOperations.begin(), originalOperations.end());
  DenseMap<mlir::Operation *, unsigned> anchorCuts;
  for (unsigned i = 0; i < input.payload.size(); ++i) anchorCuts[input.payload[i]] = i;
  anchorCuts[function.getBody().front().getTerminator()] = input.payload.size();
  struct Position { mlir::Operation *operation; Block *block; mlir::Operation *nextOriginal; };
  SmallVector<Position> positions;
  SmallVector<mlir::Operation *> generated;
  bool scanFailed = false;
  function.walk([&](mlir::Operation *op) {
    if (scanFailed || op == function.getOperation() || originalSet.contains(op)) return;
    if (!isa<SetFlagOp, WaitFlagOp, BarrierOp>(op)) {
      op->emitError("handoff: unexpected generated operation");
      scanFailed = true; return;
    }
    // Actual block and next original instruction determine participation.
    // A command in an empty arm, at a nested exit, or before a loop owner is
    // not relabelled as the next depth-first payload phase.
    mlir::Operation *next = op->getNextNode();
    while (next && !originalSet.contains(next)) next = next->getNextNode();
    auto found = anchorCuts.find(next);
    if (found == anchorCuts.end() || next->getBlock() != op->getBlock()) {
      op->emitError("handoff: command is outside an original physical cut");
      scanFailed = true; return;
    }
    unsigned cut = found->second;
    auto event = [&](auto e, Command::Kind kind) -> LogicalResult {
      auto a = pipe(e.getSrcPipe().getPipe()), b = pipe(e.getDstPipe().getPipe());
      if (!a || !b) return op->emitError("handoff: emitted event has unsupported pipeline");
      actual[cut].push_back({kind, *a, *b, unsigned(e.getEventId().getEvent())});
      return success();
    };
    if (auto e = dyn_cast<SetFlagOp>(op)) scanFailed = failed(event(e, Command::Publish));
    else if (auto e = dyn_cast<WaitFlagOp>(op)) scanFailed = failed(event(e, Command::Acquire));
    else {
      auto value = cast<BarrierOp>(op).getPipe().getPipe();
      if (value == PIPE::PIPE_ALL) actual[cut].push_back({Command::BarrierAll});
      else if (auto p = pipe(value)) actual[cut].push_back({Command::Barrier, *p});
      else { op->emitError("handoff: emitted barrier has unsupported pipeline"); scanFailed = true; }
    }
    generated.push_back(op);
    positions.push_back({op, op->getBlock(), next});
  });
  if (scanFailed) return failure();
  // Bundle credit was computed for these exact ordered command words. Shared
  // code generation must not silently merge, reorder, or supplement them even
  // when a modified word happens to satisfy the memory checker too.
  if (actual.size() != result.commands.size())
    return function.emitError("handoff: emitted command-cut population changed");
  for (std::size_t cut = 0; cut < actual.size(); ++cut) {
    if (actual[cut].size() != result.commands[cut].size())
      return function.emitError("handoff: emitted command word changed at cut ") << cut;
    for (std::size_t i = 0; i < actual[cut].size(); ++i) {
      const auto &a = actual[cut][i], &b = result.commands[cut][i];
      if (a.kind != b.kind || a.source != b.source ||
          a.observer != b.observer || a.key != b.key)
        return function.emitError("handoff: emitted command order/identity changed at cut ") << cut;
    }
  }
  // Detach generated commands to compare the actual remaining IR with the
  // original imported obligations. Checking a changed reimport against itself
  // would incorrectly accept payload/effect mutations.
  for (auto *op : generated) op->remove();
  std::string reconstructedIR;
  llvm::raw_string_ostream reconstructedStream(reconstructedIR);
  function.print(reconstructedStream, OpPrintingFlags().printGenericOpForm());
  reconstructedStream.flush();
  SmallVector<mlir::Operation *> remainingOperations;
  function.walk([&](mlir::Operation *op) {
    if (op != function.getOperation()) remainingOperations.push_back(op);
  });
  bool unchanged = originalOperations == remainingOperations &&
                   originalIR == reconstructedIR;
  // Restore the exact location recorded before detaching. Verification must
  // never repair a wrong branch/loop position by moving to canonical anchors.
  for (const Position &position : positions)
    position.block->getOperations().insert(position.nextOriginal->getIterator(),
                                           position.operation);
  if (!unchanged)
    return function.emitError("handoff: emission changed original payload or contract");
  auto checked = verify(input.program, actual);
  if (!checked.success) return function.emitError("handoff emitted verification: ") << checked.reason;
  return mlir::verify(function);
}
} // namespace

LogicalResult testing::runHandoffSyncWithMutation(func::FuncOp function,
    llvm::function_ref<void(func::FuncOp)> mutate) {
  if (getTargetArch(function) != PTOArch::A3)
    return function.emitError("handoff: first native target is a3");
  auto parent = function->getParentOfType<ModuleOp>();
  OwningOpRef<ModuleOp> sandbox = ModuleOp::create(function.getLoc());
  if (parent) sandbox->getOperation()->setAttrs(parent->getAttrs());
  auto working = cast<func::FuncOp>(function->clone());
  sandbox->push_back(working);
  if (failed(execute(working, mutate))) return failure();
  function.getBody().takeBody(working.getBody());
  return success();
}

LogicalResult runHandoffSync(func::FuncOp function) {
  return testing::runHandoffSyncWithMutation(function, {});
}

LogicalResult analyzeHandoffSync(func::FuncOp function, NativeAnalysis &result) {
  result = NativeAnalysis{};
  if (getTargetArch(function) != PTOArch::A3)
    return function.emitError("handoff: native analysis target is a3");
  Import input;
  if (failed(import(function, input))) return failure();
  result.analysis = analyze(input.program);
  result.program = std::move(input.program);
  result.phases = std::move(input.payload);
  if (!result.analysis.complete)
    return function.emitError("handoff analysis: ") << result.analysis.reason;
  return success();
}
LogicalResult analyzeHandoffSync(func::FuncOp function) {
  NativeAnalysis result;
  return analyzeHandoffSync(function, result);
}
} // namespace mlir::pto::oahs
