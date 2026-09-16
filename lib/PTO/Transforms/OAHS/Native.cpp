// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/InsertSync/SyncOriginClosure.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include "PTO/Transforms/OAHS/Plan.h"
#include "PTO/Transforms/OAHS/StorageWitnesses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>
#include <limits>
#include <optional>
#include <set>

namespace mlir::pto::oahs {
namespace {
std::optional<Pipe> pipe(PIPE value) {
  switch (value) {
  case PIPE::PIPE_S:
    return Pipe::S;
  case PIPE::PIPE_V:
    return Pipe::V;
  case PIPE::PIPE_M:
    return Pipe::M;
  case PIPE::PIPE_MTE1:
    return Pipe::MTE1;
  case PIPE::PIPE_MTE2:
    return Pipe::MTE2;
  case PIPE::PIPE_MTE3:
    return Pipe::MTE3;
  case PIPE::PIPE_FIX:
    return Pipe::FIX;
  case PIPE::PIPE_ALL:
  case PIPE::PIPE_UNASSIGNED:
  default:
    return {};
  }
}
PipelineType nativePipe(Pipe value) {
  switch (value) {
  case Pipe::S:
    return PipelineType::PIPE_S;
  case Pipe::V:
    return PipelineType::PIPE_V;
  case Pipe::M:
    return PipelineType::PIPE_M;
  case Pipe::MTE1:
    return PipelineType::PIPE_MTE1;
  case Pipe::MTE2:
    return PipelineType::PIPE_MTE2;
  case Pipe::MTE3:
    return PipelineType::PIPE_MTE3;
  case Pipe::FIX:
    return PipelineType::PIPE_FIX;
  case Pipe::Count:
    return PipelineType::PIPE_UNASSIGNED;
  }
  return PipelineType::PIPE_UNASSIGNED;
}
struct Import {
  Program program;
  SmallVector<SyncProtocolModel, 0> protocols;
  SmallVector<mlir::Operation *> payload;
  SmallVector<Value> storageRoots;
  SmallVector<mlir::Operation *> anchors;
  SmallVector<Cut> phaseCuts;
  DenseMap<std::size_t, scf::ForOp> loopOwners;
  std::vector<std::string> observationNotes;
};
// Every actual original instruction is an anchor, including scalar/control
// instructions and region terminators. Synthetic branch/loop decisions have no
// anchor and cannot acquire emitted commands. Payload phases remain unchanged.
LogicalResult importObservedCuts(func::FuncOp function, Import &out) {
  ObservedControl q;
  q.qualification = "MLIR-SCF-original-anchor-control-v1";
  q.scopes.push_back({0, NoControlId, NoControlId});
  DenseMap<mlir::Operation *, std::size_t> ids, phases;
  for (auto [i, op] : llvm::enumerate(out.payload))
    phases[op] = i;
  function.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == function.getOperation())
      return;
    const auto id = q.sites.size();
    ids[op] = id;
    q.sites.emplace_back();
    out.anchors.push_back(op);
    q.sites.back().observation = q.observations.size();
    q.observations.push_back({id, {}, true});
    auto found = phases.find(op);
    if (found != phases.end())
      q.sites.back().operation = found->second;
  });
  out.phaseCuts.resize(out.payload.size());
  for (auto [i, op] : llvm::enumerate(out.payload))
    out.phaseCuts[i] = ids.lookup(op);
  auto scope = [&](unsigned kind, std::size_t parent, std::size_t owner) {
    const auto id = q.scopes.size();
    q.scopes.push_back({kind, parent, owner});
    return id;
  };
  auto entry = [&](mlir::Region &region, std::size_t fallback) {
    return region.empty() ? fallback : ids.lookup(&region.front().front());
  };
  std::function<LogicalResult(mlir::Region &, std::vector<std::size_t>,
                              std::size_t, std::size_t)>
      wire;
  wire = [&](mlir::Region &region, std::vector<std::size_t> exits,
             std::size_t context, std::size_t backedge) -> LogicalResult {
    if (region.empty())
      return success();
    if (!region.hasOneBlock())
      return function.emitError(
          "handoff: observed SCF region requires one original block");
    for (mlir::Operation &op : region.front()) {
      const auto id = ids.lookup(&op);
      q.sites[id].context = context;
      std::vector<std::size_t> next =
          op.getNextNode()
              ? std::vector<std::size_t>{ids.lookup(op.getNextNode())}
              : exits;
      if (auto choice = dyn_cast<scf::IfOp>(op)) {
        if (next.size() != 1)
          return choice.emitError("handoff: malformed choice continuation");
        const auto yes = entry(choice.getThenRegion(), next.front()),
                   no = entry(choice.getElseRegion(), next.front());
        q.sites[id].successors = {yes, no};
        const auto yesScope = scope(1, context, id),
                   noScope = scope(2, context, id);
        if (failed(wire(choice.getThenRegion(), next, yesScope, NoControlId)) ||
            failed(wire(choice.getElseRegion(), next, noScope, NoControlId)))
          return failure();
      } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (next.size() != 1)
          return loop.emitError("handoff: malformed loop continuation");
        const auto header = q.sites.size();
        q.sites.emplace_back();
        out.anchors.push_back(nullptr);
        q.sites[header].context = context;
        q.sites[id].successors = {header};
        q.sites[header].successors = {entry(loop.getRegion(), header),
                                      next.front()};
        if (failed(wire(loop.getRegion(), {header}, scope(3, context, id), id)))
          return failure();
      } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
        if (next.size() != 1)
          return loop.emitError("handoff: malformed while continuation");
        const auto before = entry(loop.getBefore(), NoControlId),
                   after = entry(loop.getAfter(), NoControlId);
        if (before == NoControlId || after == NoControlId)
          return loop.emitError("handoff: empty while region");
        q.sites[id].successors = {before};
        if (failed(wire(loop.getBefore(), {after, next.front()},
                        scope(4, context, id), NoControlId)) ||
            failed(wire(loop.getAfter(), {before}, scope(5, context, id), id)))
          return failure();
      } else {
        q.sites[id].successors = std::move(next);
        if (!op.getNextNode() && backedge != NoControlId)
          q.sites[id].backedgeOwners.assign(q.sites[id].successors.size(),
                                            backedge);
      }
    }
    return success();
  };
  auto *terminator = function.getBody().front().getTerminator();
  if (!isa<func::ReturnOp>(terminator))
    return function.emitError("handoff: missing original return");
  q.entry = ids.lookup(&function.getBody().front().front());
  q.exit = ids.lookup(terminator);
  if (failed(wire(function.getBody(), {}, 0, NoControlId)))
    return failure();
  out.program.observed = std::move(q);
  // Leaf loops are refined independently. No product of unrelated loop periods
  // is formed. Unsupported arithmetic keeps the original sound SCF graph.
  auto integer = [](Value value) -> std::optional<int64_t> {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return {};
    auto attr = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attr || attr.getValue().getBitWidth() > 64)
      return {};
    return attr.getInt();
  };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (auto loop : loops) {
    bool nested = false;
    loop.getRegion().walk([&](mlir::Operation *op) {
      nested |= isa<scf::ForOp, scf::WhileOp>(op);
    });
    if (nested || !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp") ||
        !loop.getInitArgs().empty() ||
        integer(loop.getLowerBound()) != std::optional<int64_t>(0) ||
        integer(loop.getStep()) != std::optional<int64_t>(1))
      continue;
    CountedLoopRegion model;
    model.owner = ids.lookup(loop.getOperation());
    const auto &control = *out.program.observed;
    model.header = control.sites[model.owner].successors.front();
    model.bodyEntry = control.sites[model.header].successors[0];
    model.continuation = control.sites[model.header].successors[1];
    std::vector<bool> seen(control.sites.size());
    std::vector<std::size_t> work{model.bodyEntry};
    while (!work.empty()) {
      const auto at = work.back();
      work.pop_back();
      if (at == model.header || seen[at])
        continue;
      seen[at] = true;
      model.bodySites.push_back(at);
      for (auto next : control.sites[at].successors)
        work.push_back(next);
    }
    std::set<Pipe> participants;
    for (auto site : model.bodySites) {
      const auto physical = control.sites[site].operation;
      if (physical != NoControlId)
        participants.insert(out.program.operations[physical].pipe);
    }
    if (participants.size() < 2)
      continue; // no cross-engine recurrence to refine
    bool periodsAgree = true;
    loop.getRegion().walk([&](scf::IfOp choice) {
      auto cmp = choice.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!cmp || (cmp.getPredicate() != arith::CmpIPredicate::eq &&
                   cmp.getPredicate() != arith::CmpIPredicate::ne))
        return;
      Value expression = cmp.getLhs(), literal = cmp.getRhs();
      if (integer(expression)) {
        std::swap(expression, literal);
      }
      Value lhs, rhs;
      if (auto rem = expression.getDefiningOp<arith::RemUIOp>()) {
        lhs = rem.getLhs();
        rhs = rem.getRhs();
      } else if (auto rem = expression.getDefiningOp<arith::RemSIOp>()) {
        lhs = rem.getLhs();
        rhs = rem.getRhs();
      } else
        return;
      const auto modulus = integer(rhs), residue = integer(literal);
      if (lhs != loop.getInductionVar() || !modulus || !residue ||
          *modulus <= 0 || *residue < 0 || *residue >= *modulus)
        return;
      // Optional observation vocabulary follows one existing modulus and the
      // physical key population, not an arbitrary fixed-point work allowance.
      std::size_t maximumPool = 1;
      for (const auto &row : out.program.target.keys)
        for (const auto &pool : row)
          maximumPool = std::max(maximumPool, pool.size());
      if (uint64_t(*modulus) > maximumPool ||
          (model.period != 1 && uint64_t(model.period) != uint64_t(*modulus))) {
        periodsAgree = false;
        return;
      }
      model.period = unsigned(*modulus);
      model.decisions.push_back(
          {ids.lookup(choice.getOperation()), uint64_t(*modulus),
           uint64_t(*residue), cmp.getPredicate() == arith::CmpIPredicate::eq});
    });
    if (!periodsAgree) {
      out.observationNotes.push_back(
          "kept original SCF control: leaf-loop observation periods exceed the "
          "selected finite vocabulary");
      continue;
    }
    auto refined = refineCountedLoop(out.program, model);
    if (!refined.success) {
      out.observationNotes.push_back("kept original SCF control: " +
                                     refined.reason);
      continue;
    }
    out.program = std::move(refined.program);
    out.loopOwners[model.owner] = loop;
  }
  const auto originals = out.anchors;
  out.anchors.assign(commandCutCount(out.program), nullptr);
  for (Cut at = 0; at < out.anchors.size(); ++at) {
    const auto observation = out.program.observed->sites[at].observation;
    if (observation != NoControlId) {
      const auto anchor =
          out.program.observed->observations[observation].anchor;
      if (anchor >= originals.size() || !originals[anchor])
        return function.emitError("handoff: invalid qualified native anchor");
      out.anchors[at] = originals[anchor];
    }
  }
  std::vector<bool> reached(out.anchors.size());
  std::vector<std::size_t> pending{out.program.observed->entry};
  reached[pending.front()] = true;
  out.phaseCuts.assign(out.payload.size(), NoControlId);
  while (!pending.empty()) {
    const auto at = pending.back();
    pending.pop_back();
    const auto physical = out.program.observed->sites[at].operation;
    if (physical != NoControlId && out.phaseCuts[physical] == NoControlId)
      out.phaseCuts[physical] = at;
    for (auto next : out.program.observed->sites[at].successors)
      if (!reached[next]) {
        reached[next] = true;
        pending.push_back(next);
      }
  }
  return success();
}
LogicalResult import(func::FuncOp function, Import &out) {
  if (!function.getBody().hasOneBlock())
    return function.emitError("handoff: function requires one CFG entry block");
  MemoryDependentAnalyzer aliases;
  SyncIRs translated;
  Buffer2MemInfoMap buffers;
  PTOIRTranslator translator(translated, aliases, buffers, function,
                             SyncAnalysisMode::NORMALSYNC);
  if (failed(translator.Build()))
    return failure();
  if (failed(closeStructuredSyncOrigins(function, translated, buffers)))
    return failure();
  // The shared translator owns semantic qualification. Import the translated
  // phases; do not maintain a second opcode admission/effect extraction path.
  auto semantics = translator.describeSemantics();
  DenseMap<mlir::Operation *, CompoundInstanceElement *> phases;
  for (const auto &record : semantics.operations) {
    if (!record.gap.empty())
      return record.operation->emitError("handoff: shared sync semantics: ")
             << record.gap;
    if (record.kind != SyncSemanticRecord::Ordinary &&
        record.kind != SyncSemanticRecord::Protocol)
      continue;
    if (record.protocol)
      out.protocols.push_back(*record.protocol);
    if (record.phases.empty())
      continue;
    auto *phase = record.phases.front();
    if (!pipe(static_cast<PIPE>(phase->kPipeValue)))
      return record.operation->emitError(
          "handoff: pipeline outside native target contract");
    out.payload.push_back(record.operation);
    phases[record.operation] = phase;
  }

  DenseMap<mlir::Operation *, unsigned> operationIds;
  for (auto [i, op] : llvm::enumerate(out.payload))
    operationIds[op] = i;
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

  struct Effect {
    unsigned operation;
    const BaseMemInfo *memory;
    bool write;
  };
  SmallVector<Effect> physicalEffects;
  // A missing absolute local address is not a distinct-allocation proof. Keep
  // legacy alias behavior unchanged; normalize the handoff import's private
  // records instead. Stable copies also preserve the original SSA effect names.
  SmallVector<std::unique_ptr<BaseMemInfo>> widenedMemory;
  DenseMap<const BaseMemInfo *, const BaseMemInfo *> normalizedMemory;
  auto qualifyMemory = [&](const BaseMemInfo *memory) -> const BaseMemInfo * {
    auto found = normalizedMemory.find(memory);
    if (found != normalizedMemory.end())
      return found->second;
    const bool local = memory->scope != AddressSpace::GM &&
                       memory->scope != AddressSpace::Zero;
    const bool overflowing =
        llvm::any_of(memory->baseAddresses, [&](uint64_t base) {
          return memory->allocateSize >
                 std::numeric_limits<uint64_t>::max() - base;
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
    Operation imported;
    imported.pipe = *pipe(static_cast<PIPE>(phase->kPipeValue));
    imported.original = i;
    imported.complete = true;
    out.program.operations.push_back(std::move(imported));
    auto add = [&](const auto &memories, bool write) -> LogicalResult {
      for (const BaseMemInfo *memory : memories) {
        // Admit the same translated storage scopes as production autosync.
        // Visibility remains a separate contract below; ordinary GM byte
        // completion is intentionally governed by production alias behavior.
        physicalEffects.push_back({i, qualifyMemory(memory), write});
      }
      return success();
    };
    if (failed(add(phase->defVec, true)) || failed(add(phase->useVec, false)))
      return failure();
  }
  // Group only repeated uses of the identical immutable imported record.
  // This is not a may-alias equivalence closure. Origin alternatives and views
  // with distinct records retain separate groups and pairwise overlap tests.
  DenseMap<const BaseMemInfo *, unsigned> groupIds;
  DenseMap<Value, unsigned> storageRoots;
  std::vector<const BaseMemInfo *> memories;
  std::vector<FootprintGroup> groups;
  for (const Effect &effect : physicalEffects) {
    auto found = groupIds.find(effect.memory);
    unsigned group;
    if (found == groupIds.end()) {
      group = groups.size();
      groupIds[effect.memory] = group;
      memories.push_back(effect.memory);
      const auto *memory = effect.memory;
      FootprintGroup entry;
      entry.description.addressSpace =
          std::to_string(static_cast<unsigned>(memory->scope));
      entry.description.provenance =
          "original footprint (including self recurrence)";
      entry.description.unknownRange =
          memory->aliasesUnknownRange || memory->baseAddresses.empty();
      if (memory->rootBuffer) {
          auto inserted = storageRoots.try_emplace(memory->rootBuffer, storageRoots.size());
          if (inserted.second)
              out.storageRoots.push_back(memory->rootBuffer);
          entry.description.storageOrigins.push_back(inserted.first->second);
      }
      for (uint64_t base : memory->baseAddresses)
        entry.description.ranges.push_back({base, memory->allocateSize});
      if (auto coordinates = aliases.storageCoordinates(*memory)) {
          if (coordinates->absolute) {
              entry.description.coordinateSpace = "physical-local";
          } else {
              entry.description.coordinateSpace = "root:" + std::to_string(storageRoots.lookup(coordinates->root));
          }
          entry.description.ranges = {{coordinates->begin, coordinates->size}};
      }
      groups.push_back(std::move(entry));
    } else
      group = found->second;
    groups[group].uses.push_back(
        {effect.operation, !effect.write, effect.write});
  }
  appendCanonicalStorage(
      out.program, groups, [&](std::size_t a, std::size_t b) { return aliases.MemAlias(memories[a], memories[b]); });
  // Select the physical core, not a complete graph over unrelated pipelines.
  bool vector = false, cube = false;
  auto kind =
      function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
  if (kind) {
    vector = kind.getKernelKind() == FunctionKernelKind::Vector;
    cube = kind.getKernelKind() == FunctionKernelKind::Cube;
  }
  for (const auto &phase : out.program.operations) {
    vector |= phase.pipe == Pipe::V;
    cube |= phase.pipe == Pipe::M || phase.pipe == Pipe::MTE1 ||
            phase.pipe == Pipe::FIX;
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
    return function.emitError(
        "handoff: mixed physical core context requires explicit sections");
  // A payload-free function has no core-specific obligations. A pure DMA
  // function derives its core from the actual local storage above.
  out.program.target = a3SyncProfile(cube ? SyncCore::Cube : SyncCore::Vector);
  for (const auto &phase : out.program.operations)
    if (!out.program.target.supported[unsigned(phase.pipe)])
      return function.emitError(
          "handoff: operation outside selected core profile");
  out.program.invocation.retirement =
      Program::InvocationContract::DrainAllAtReturn;
  out.program.invocation.alias =
      "production MemoryDependentAnalyzer compatibility contract";
  out.program.invocation.visibility =
      "ordinary GM communication follows production alias behavior; no new "
      "hardware guarantee";
  out.program.invocation.boundary =
      "kernel return requires completion of asynchronous physical phases";
  if (llvm::any_of(semantics.operations, [](const auto &record) {
        return record.kind == SyncSemanticRecord::Protocol;
      }))
    out.program.invocation.boundary +=
        "; preserved lowering-owned peer matching/participation/progress contracts; "
        "cross-core flags remain owned by the original protocol in their "
        "separate namespace; no local completion credit from peer events or "
        "preserved intrinsic drains";
  return importObservedCuts(function, out);
}

// Emit only total predicates over the original normalized induction variable
// and upper bound. For an active visit 0<=i<N, N-i cannot overflow signed index
// range; no new iteration/history counter or event-state query is introduced.
FailureOr<Value>
emitObservationPredicate(const OriginalObservation &observation,
                         mlir::Operation *anchor, const Import &input,
                         OpBuilder &builder) {
  Value predicate;
  auto loc = anchor->getLoc();
  for (const auto &atom : observation.atoms) {
    auto owner = input.loopOwners.find(atom.owner);
    if (owner == input.loopOwners.end() ||
        !owner->second->isProperAncestor(anchor))
      return anchor->emitError("handoff: original loop values unavailable at "
                               "observed endpoint"),
             failure();
    auto loop = owner->second;
    Value lhs = loop.getInductionVar();
    auto constant = [&](uint64_t value) -> Value {
      return builder.create<arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(value));
    };
    Value part;
    if (atom.kind == ObservationAtom::LoopResidue) {
      lhs = builder.create<arith::RemUIOp>(loc, lhs, constant(atom.parameter));
      part = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs,
                                           constant(atom.value));
    } else if (atom.kind == ObservationAtom::LoopHasPrevious) {
      part = builder.create<arith::CmpIOp>(
          loc,
          atom.value ? arith::CmpIPredicate::sge : arith::CmpIPredicate::slt,
          lhs, constant(atom.parameter));
    } else if (atom.kind == ObservationAtom::LoopHasNext) {
      lhs = builder.create<arith::SubIOp>(loc, loop.getUpperBound(), lhs);
      part = builder.create<arith::CmpIOp>(
          loc,
          atom.value ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sle,
          lhs, constant(atom.parameter));
    } else
      return anchor->emitError("handoff: unsupported native observation atom"),
             failure();
    predicate = predicate
                    ? Value(builder.create<arith::AndIOp>(loc, predicate, part))
                    : part;
  }
  return predicate;
}

// Independent read-back of the fixed emitted arithmetic vocabulary. This does
// not trust the builder's predicate Value or a metadata tag. Original owners,
// signed comparison directions, subtraction operands and literal values must
// agree with the qualified observation before any protocol can be accepted.
LogicalResult checkObservationPredicate(const OriginalObservation &observation,
                                        mlir::Operation *anchor,
                                        const Import &input, Value condition) {
  auto integer = [](Value value) -> std::optional<int64_t> {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return {};
    auto attr = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attr || attr.getValue().getBitWidth() > 64)
      return {};
    return attr.getInt();
  };
  SmallVector<Value> components;
  std::function<void(Value)> flatten = [&](Value value) {
    if (auto both = value.getDefiningOp<arith::AndIOp>()) {
      flatten(both.getLhs());
      flatten(both.getRhs());
    } else
      components.push_back(value);
  };
  flatten(condition);
  if (components.size() != observation.atoms.size())
    return anchor->emitError(
        "handoff: emitted observation has a different predicate population");
  std::vector<bool> used(components.size());
  for (const auto &atom : observation.atoms) {
    auto owner = input.loopOwners.find(atom.owner);
    if (owner == input.loopOwners.end() ||
        !owner->second->isProperAncestor(anchor))
      return anchor->emitError(
          "handoff: emitted predicate lost its original available owner");
    auto loop = owner->second;
    bool matched = false;
    for (std::size_t i = 0; i < components.size() && !matched; ++i) {
      if (used[i])
        continue;
      auto cmp = components[i].getDefiningOp<arith::CmpIOp>();
      if (!cmp)
        continue;
      const auto rhs = integer(cmp.getRhs());
      if (!rhs || *rhs < 0)
        continue;
      if (atom.kind == ObservationAtom::LoopResidue) {
        auto rem = cmp.getLhs().getDefiningOp<arith::RemUIOp>();
        matched =
            rem && cmp.getPredicate() == arith::CmpIPredicate::eq &&
            rem.getLhs() == loop.getInductionVar() &&
            integer(rem.getRhs()) ==
                std::optional<int64_t>(static_cast<int64_t>(atom.parameter)) &&
            uint64_t(*rhs) == atom.value;
      } else if (atom.kind == ObservationAtom::LoopHasPrevious) {
        matched =
            cmp.getLhs() == loop.getInductionVar() &&
            uint64_t(*rhs) == atom.parameter &&
            cmp.getPredicate() == (atom.value ? arith::CmpIPredicate::sge
                                              : arith::CmpIPredicate::slt);
      } else if (atom.kind == ObservationAtom::LoopHasNext) {
        auto remaining = cmp.getLhs().getDefiningOp<arith::SubIOp>();
        matched =
            remaining && remaining.getLhs() == loop.getUpperBound() &&
            remaining.getRhs() == loop.getInductionVar() &&
            uint64_t(*rhs) == atom.parameter &&
            cmp.getPredicate() == (atom.value ? arith::CmpIPredicate::sgt
                                              : arith::CmpIPredicate::sle);
      }
      if (matched)
        used[i] = true;
    }
    if (!matched)
      return anchor->emitError(
          "handoff: emitted guard differs from its original-value observation");
  }
  return success();
}

LogicalResult execute(func::FuncOp function,
                      llvm::function_ref<void(func::FuncOp)> mutate = {}) {
  Import input;
  if (failed(import(function, input)))
    return failure();
  // Snapshot every payload and storage declaration, including operands, types,
  // attributes and ordering. Reconstruction must retain the imported contract.
  std::string originalIR;
  llvm::raw_string_ostream originalStream(originalIR);
  function.print(originalStream, OpPrintingFlags().printGenericOpForm());
  originalStream.flush();
  SmallVector<mlir::Operation *> originalOperations;
  function.walk([&](mlir::Operation *op) {
    if (op != function.getOperation())
      originalOperations.push_back(op);
  });
  auto result = construct(input.program);
  if (!result.success)
    return function.emitError("handoff: ") << result.reason;
  // Use production emission directly. Do not run legacy planning, motion,
  // redundancy removal or allocation on the new constructor's chosen plan.
  SyncIRs emission;
  SmallVector<std::unique_ptr<SyncOperation>> storage;
  struct EmittedWord {
    Cut cut;
    mlir::Operation *anchor;
    scf::IfOp guard;
  };
  SmallVector<EmittedWord> words;
  llvm::SmallPtrSet<mlir::Operation *, 32> originalSet;
  originalSet.insert(originalOperations.begin(), originalOperations.end());
  for (Cut cut = 0; cut < result.commands.size(); ++cut) {
    if (result.commands[cut].empty() ||
        canonicalCommandCut(input.program, cut) != cut)
      continue;
    if (cut >= input.anchors.size() || !input.anchors[cut])
      return function.emitError(
          "handoff: selected command has no original native anchor");
    mlir::Operation *wordAnchor = input.anchors[cut];
    scf::IfOp generatedGuard;
    const auto observation = input.program.observed->sites[cut].observation;
    const auto &descriptor = input.program.observed->observations[observation];
    if (!descriptor.atoms.empty()) {
      OpBuilder builder(wordAnchor);
      auto predicate =
          emitObservationPredicate(descriptor, wordAnchor, input, builder);
      if (failed(predicate))
        return failure();
      generatedGuard =
          builder.create<scf::IfOp>(wordAnchor->getLoc(), *predicate, false);
      wordAnchor = generatedGuard.getThenRegion().front().getTerminator();
    }
    words.push_back({cut, wordAnchor, generatedGuard});
    auto anchor = std::make_unique<PlaceHolderInstanceElement>(cut, 0);
    anchor->elementOp = wordAnchor;
    for (const auto &command : result.commands[cut]) {
      auto type = command.kind == Command::Publish
                      ? SyncOperation::TYPE::SET_EVENT
                  : command.kind == Command::Acquire
                      ? SyncOperation::TYPE::WAIT_EVENT
                      : SyncOperation::TYPE::PIPE_BARRIER;
      auto source = command.kind == Command::BarrierAll
                        ? PipelineType::PIPE_ALL
                        : nativePipe(command.source);
      auto sync = std::make_unique<SyncOperation>(
          type, source, nativePipe(command.observer), storage.size(), cut,
          std::nullopt);
      sync->eventIds.push_back(command.key);
      anchor->pipeBefore.push_back(sync.get());
      storage.push_back(std::move(sync));
    }
    emission.push_back(std::move(anchor));
  }
  SyncCodegen codegen(emission, function, SyncAnalysisMode::NORMALSYNC);
  codegen.Run();
  // Guard expressions were built from the qualified original-value descriptor.
  // Preserve their exact operation/operand/attribute structure, not just a tag
  // asserting a predicate is valid. Mutations are checked before saved pointers
  // into generated regions are dereferenced.
  std::string emittedIR;
  llvm::raw_string_ostream emittedStream(emittedIR);
  function.print(emittedStream, OpPrintingFlags().printGenericOpForm());
  emittedStream.flush();
  SmallVector<mlir::Operation *> emittedPopulation;
  function.walk([&](mlir::Operation *op) { emittedPopulation.push_back(op); });
  if (mutate)
    mutate(function);
  if (failed(mlir::verify(function)))
    return failure();
  std::string currentIR;
  llvm::raw_string_ostream currentStream(currentIR);
  function.print(currentStream, OpPrintingFlags().printGenericOpForm());
  currentStream.flush();
  SmallVector<mlir::Operation *> currentPopulation;
  function.walk([&](mlir::Operation *op) { currentPopulation.push_back(op); });
  if (currentIR != emittedIR || currentPopulation != emittedPopulation)
    return function.emitError(
        "handoff: emitted command/guard structure or original program changed");
  Commands actual(commandCutCount(input.program));
  llvm::SmallPtrSet<mlir::Operation *, 32> readCommands;
  auto read = [&](mlir::Operation *op, Cut cut) -> LogicalResult {
    auto event = [&](auto e, Command::Kind kind) -> LogicalResult {
      auto a = pipe(e.getSrcPipe().getPipe()),
           b = pipe(e.getDstPipe().getPipe());
      if (!a || !b)
        return op->emitError("handoff: emitted event has unsupported pipeline");
      actual[cut].push_back(
          {kind, *a, *b, unsigned(e.getEventId().getEvent())});
      return success();
    };
    if (auto e = dyn_cast<SetFlagOp>(op))
      return event(e, Command::Publish);
    if (auto e = dyn_cast<WaitFlagOp>(op))
      return event(e, Command::Acquire);
    if (auto e = dyn_cast<BarrierOp>(op)) {
      const auto value = e.getPipe().getPipe();
      if (value == PIPE::PIPE_ALL)
        actual[cut].push_back({Command::BarrierAll});
      else if (auto lane = pipe(value))
        actual[cut].push_back({Command::Barrier, *lane});
      else
        return op->emitError(
            "handoff: emitted barrier has unsupported pipeline");
      return success();
    }
    return op->emitError(
        "handoff: unexpected operation in synchronization word");
  };
  for (const auto &word : words) {
    const auto observation =
        input.program.observed->sites[word.cut].observation;
    const auto &descriptor = input.program.observed->observations[observation];
    auto guard = word.guard;
    if (word.guard &&
        failed(checkObservationPredicate(descriptor, input.anchors[word.cut],
                                         input, guard.getCondition())))
      return failure();
    if (bool(word.guard) != !descriptor.atoms.empty())
      return function.emitError("handoff: emitted guard participation changed");
    SmallVector<mlir::Operation *> backwards;
    for (auto *op = word.anchor->getPrevNode();
         op && isa<SetFlagOp, WaitFlagOp, BarrierOp>(op) &&
         !originalSet.contains(op);
         op = op->getPrevNode())
      backwards.push_back(op);
    for (auto *op : llvm::reverse(backwards)) {
      if (!readCommands.insert(op).second || failed(read(op, word.cut)))
        return failure();
    }
  }
  bool extra = false;
  function.walk([&](mlir::Operation *op) {
    if (isa<SetFlagOp, WaitFlagOp, BarrierOp>(op) &&
        !originalSet.contains(op) && !readCommands.contains(op))
      extra = true;
  });
  if (extra)
    return function.emitError("handoff: emitted synchronization outside its "
                              "selected original observation");
  for (Cut cut = 0; cut < actual.size(); ++cut) {
    const auto canonical = canonicalCommandCut(input.program, cut);
    if (canonical != cut)
      actual[cut] = actual[canonical];
  }
  struct Position {
    mlir::Operation *operation;
    Block *block;
    mlir::Operation *nextOriginal;
  };
  SmallVector<Position> positions;
  SmallVector<mlir::Operation *> generated;
  // Remove only generated roots. An if root owns its generated word/yield;
  // original payload is never moved into it. Predicate arithmetic is a root
  // too.
  function.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == function.getOperation() || originalSet.contains(op))
      return;
    auto *parent = op->getParentOp();
    if (parent != function.getOperation() && !originalSet.contains(parent))
      return;
    auto *next = op->getNextNode();
    while (next && !originalSet.contains(next))
      next = next->getNextNode();
    generated.push_back(op);
    positions.push_back({op, op->getBlock(), next});
  });
  for (const auto &position : positions)
    if (!position.nextOriginal)
      return function.emitError(
          "handoff: generated root has no original anchor");
  // Bundle credit was computed for these exact ordered command words. Shared
  // code generation must not silently merge, reorder, or supplement them even
  // when a modified word happens to satisfy the memory checker too.
  if (actual.size() != result.commands.size())
    return function.emitError(
        "handoff: emitted command-cut population changed");
  for (std::size_t cut = 0; cut < actual.size(); ++cut) {
    if (actual[cut].size() != result.commands[cut].size())
      return function.emitError("handoff: emitted command word changed at cut ")
             << cut;
    for (std::size_t i = 0; i < actual[cut].size(); ++i) {
      const auto &a = actual[cut][i], &b = result.commands[cut][i];
      if (a.kind != b.kind || a.source != b.source ||
          a.observer != b.observer || a.key != b.key)
        return function.emitError(
                   "handoff: emitted command order/identity changed at cut ")
               << cut;
    }
  }
  // Detach generated commands to compare the actual remaining IR with the
  // original imported obligations. Checking a changed reimport against itself
  // would incorrectly accept payload/effect mutations.
  for (auto *op : generated)
    op->remove();
  std::string reconstructedIR;
  llvm::raw_string_ostream reconstructedStream(reconstructedIR);
  function.print(reconstructedStream, OpPrintingFlags().printGenericOpForm());
  reconstructedStream.flush();
  SmallVector<mlir::Operation *> remainingOperations;
  function.walk([&](mlir::Operation *op) {
    if (op != function.getOperation())
      remainingOperations.push_back(op);
  });
  bool unchanged = originalOperations == remainingOperations &&
                   originalIR == reconstructedIR;
  // Restore the exact location recorded before detaching. Verification must
  // never repair a wrong branch/loop position by moving to canonical anchors.
  for (const Position &position : positions)
    position.block->getOperations().insert(position.nextOriginal->getIterator(),
                                           position.operation);
  if (!unchanged)
    return function.emitError(
        "handoff: emission changed original payload or contract");
  auto checked = verify(input.program, actual);
  if (!checked.success)
    return function.emitError("handoff emitted verification: ")
           << checked.reason;
  return mlir::verify(function);
}
} // namespace

LogicalResult testing::checkHandoffObservationPredicate(
    const OriginalObservation &observation, mlir::Operation *anchor,
    llvm::ArrayRef<std::pair<std::size_t, mlir::Operation *>> originalLoopOwners,
    Value condition) {
  if (!anchor || !condition)
    return failure();
  Import input;
  for (const auto &[id, operation] : originalLoopOwners) {
    auto loop = dyn_cast_or_null<scf::ForOp>(operation);
    if (!loop || !input.loopOwners.try_emplace(id, loop).second)
      return failure();
  }
  return checkObservationPredicate(observation, anchor, input, condition);
}

LogicalResult testing::runHandoffSyncWithMutation(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate) {
  if (getTargetArch(function) != PTOArch::A3)
    return function.emitError("handoff: first native target is a3");
  auto parent = function->getParentOfType<ModuleOp>();
  OwningOpRef<ModuleOp> sandbox = ModuleOp::create(function.getLoc());
  if (parent)
    sandbox->getOperation()->setAttrs(parent->getAttrs());
  auto working = cast<func::FuncOp>(function->clone());
  sandbox->push_back(working);
  if (failed(execute(working, mutate)))
    return failure();
  function.getBody().takeBody(working.getBody());
  return success();
}

LogicalResult runHandoffSync(func::FuncOp function) {
  return testing::runHandoffSyncWithMutation(function, {});
}

LogicalResult analyzeHandoffSync(func::FuncOp function,
                                 NativeAnalysis &result) {
  result = NativeAnalysis{};
  if (getTargetArch(function) != PTOArch::A3)
    return function.emitError("handoff: native analysis target is a3");
  Import input;
  if (failed(import(function, input)))
    return failure();
  result.analysis = analyze(input.program);
  result.program = std::move(input.program);
  result.protocols = std::move(input.protocols);
  result.phases = std::move(input.payload);
  result.storageRoots = std::move(input.storageRoots);
  result.cuts = std::move(input.anchors);
  result.phaseCuts = std::move(input.phaseCuts);
  result.observationNotes = std::move(input.observationNotes);
  if (!result.analysis.complete)
    return function.emitError("handoff analysis: ") << result.analysis.reason;
  return success();
}
LogicalResult analyzeHandoffSync(func::FuncOp function) {
  NativeAnalysis result;
  return analyzeHandoffSync(function, result);
}
} // namespace mlir::pto::oahs
