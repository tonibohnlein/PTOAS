// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Native.h"
#include "ObservationUnion.h"
#include "NativeFirstUse.h"
#include "NativeFirstConsumer.h"
#include "NativeLastReader.h"
#include "NativeFifoSlots.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/InsertSync/SyncOriginClosure.h"
#include "PTO/Transforms/InsertSync/SyncSlotMapping.h"
#include "PTO/Transforms/InsertSync/SyncAccumulatorOrdering.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include "PTO/Transforms/OAHS/Plan.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
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
  struct SlotLoop {
    unsigned period = 1;
    std::vector<CountedLoopRegion::PeriodicEffects> effects;
    bool enclosing = false;
  };
  DenseMap<mlir::Operation *, SlotLoop> slotLoops;
  std::vector<std::string> observationNotes;
};
enum class ObservationPolicy : unsigned { RefineLeafLoops, QualifiedAccessRoles, ClassInvariantInputs,
                               FirstWriteConsumers, ClassInvariantFirstWrites, FinalReadSourcesBit = 8 };
ObservationPolicy selectedObservationPolicy(bool classInvariant, bool firstWrites, bool finalReads = false) {
  if (finalReads) return ObservationPolicy(unsigned(selectedObservationPolicy(classInvariant, firstWrites)) | 8u);
  if (firstWrites) return classInvariant ? ObservationPolicy::ClassInvariantFirstWrites
                                        : ObservationPolicy::FirstWriteConsumers;
  return classInvariant ? ObservationPolicy::ClassInvariantInputs : ObservationPolicy::QualifiedAccessRoles;
}
// Every actual original instruction is an anchor, including scalar/control
// instructions and region terminators. Synthetic branch/loop decisions have no
// anchor and cannot acquire emitted commands. Payload phases remain unchanged.
LogicalResult importObservedCuts(func::FuncOp function, Import &out,
                                 ObservationPolicy policy) {
  const bool finalReads = (unsigned(policy) & 8u) != 0;
  policy = ObservationPolicy(unsigned(policy) & ~8u);
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
        (!loop.getInitArgs().empty() && !out.slotLoops.count(loop.getOperation())) ||
        integer(loop.getLowerBound()) != std::optional<int64_t>(0) ||
        integer(loop.getStep()) != std::optional<int64_t>(1))
      continue;
    CountedLoopRegion model;
    const auto bound = integer(loop.getUpperBound());
    model.atLeastOnce = bound && *bound > 0;
    auto slots = out.slotLoops.find(loop.getOperation());
    if (slots != out.slotLoops.end()) {
      model.period = slots->second.period;
      model.effects = slots->second.effects;
    }
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
    if (policy != ObservationPolicy::RefineLeafLoops) {
      // Qualify this region from shared physical effects, not operation names.
      // Do not refine unrelated loops merely because a prior loop qualified.
      auto local = refined.program;
      local.observed->loops = {local.observed->loops.back()};
      if (!hasQualifiedRecurringAccesses(local)) {
        out.observationNotes.push_back("kept original SCF control: no qualified recurring access roles");
        continue;
      }
    }
    out.program = std::move(refined.program);
    out.loopOwners[model.owner] = loop;
    if (slots != out.slotLoops.end())
      out.observationNotes.push_back("qualified original carried-slot orbit: period " +
                                     std::to_string(model.period));
  }
  // Compose only physical bank identity with the existing child interface.
  // Do not build enclosing elapsed/remaining or first/tail mode products.
  for (auto loop : loops) {
    const auto slots = out.slotLoops.find(loop.getOperation());
    const bool usableOrbit = slots != out.slotLoops.end() && slots->second.enclosing &&
                             slots->second.period >= 2;
    if (!usableOrbit) {
      continue;
    }
    const auto& control = *out.program.observed;
    CountedLoopRegion model;
    model.owner = ids.lookup(loop.getOperation());
    model.period = slots->second.period;
    model.effects = slots->second.effects;
    const auto bound = integer(loop.getUpperBound());
    model.atLeastOnce = bound && *bound > 0;
    const bool uniqueHeader = control.sites[model.owner].successors.size() == 1;
    if (!uniqueHeader) {
      continue;
    }
    model.header = control.sites[model.owner].successors.front();
    const bool structuredHeader = control.sites[model.header].successors.size() == 2;
    if (!structuredHeader) {
      continue;
    }
    model.bodyEntry = control.sites[model.header].successors[0];
    model.continuation = control.sites[model.header].successors[1];
    std::set<std::size_t> seen;
    std::vector<std::size_t> todo{model.bodyEntry};
    while (!todo.empty()) {
      const auto at = todo.back();
      todo.pop_back();
      if (at == model.header || !seen.insert(at).second) {
        continue;
      }
      model.bodySites.push_back(at);
      for (auto next : control.sites[at].successors) {
        todo.push_back(next);
      }
    }
    auto refined = refineBankOccurrences(out.program, model);
    auto local = refined.program;
    if (refined.success) {
      local.observed->loops = {local.observed->loops.back()};
    }
    if (!refined.success || !hasQualifiedRecurringAccesses(local)) {
      out.observationNotes.push_back("kept original bank interface: " + refined.reason);
      continue;
    }
    out.program = std::move(refined.program);
    out.loopOwners[model.owner] = loop;
    out.observationNotes.push_back("qualified enclosing bank occurrences: period " + std::to_string(model.period));
  }
  // Several analytical residues can map to the same unchanged instruction.
  // Native emission uses original anchors; no payload is duplicated.
  while (out.payload.size() < out.program.operations.size())
    out.payload.push_back(out.payload[out.program.operations[out.payload.size()].original]);
  // Preserve useful entry placement facts even when residue refinement is not
  // admitted (notably tiled loops with a non-unit step). This neither changes
  // the control graph nor makes a physical interval an occurrence certificate.
  for (auto loop : loops) {
    auto &q = *out.program.observed;
    const auto owner = ids.lookup(loop.getOperation());
    if (llvm::any_of(q.loops, [&](const auto &r) { return r.owner == owner; })) continue;
    const auto lower = integer(loop.getLowerBound()), upper = integer(loop.getUpperBound()),
               step = integer(loop.getStep());
    if (!lower || !upper || !step || *step <= 0 || *lower >= *upper ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp")) continue;
    if (q.sites[owner].successors.size() != 1) continue;
    const auto header = q.sites[owner].successors.front();
    if (q.sites[header].successors.size() != 2) continue;
    const auto body = q.sites[header].successors[0], exit = q.sites[header].successors[1];
    ObservedLoop original{owner, owner, exit, {}, body, true};
    std::vector<bool> seen(q.sites.size());
    std::vector<Cut> todo{header};
    while (!todo.empty()) {
      const auto at = todo.back(); todo.pop_back();
      if (at == exit || seen[at]) continue;
      seen[at] = true;
      original.sites.push_back(at);
      for (auto next : q.sites[at].successors) todo.push_back(next);
    }
    q.loops.push_back(std::move(original));
  }
  const bool firstWrites = policy == ObservationPolicy::FirstWriteConsumers ||
                           policy == ObservationPolicy::ClassInvariantFirstWrites;
  std::unique_ptr<StorageFrontierAnalysis> placementStorage;
  if (firstWrites || finalReads) placementStorage = std::make_unique<StorageFrontierAnalysis>(out.program);
  native_detail::importFirstConsumers(out.program, loops, ids, out.loopOwners, out.observationNotes,
      policy == ObservationPolicy::ClassInvariantInputs || policy == ObservationPolicy::ClassInvariantFirstWrites,
      firstWrites, placementStorage.get());
  if (finalReads) native_detail::importFinalReadSources(out.program, loops, ids, out.loopOwners,
      out.observationNotes, *placementStorage);
  native_detail::importLastReaders(out.program, loops, ids, out.loopOwners, out.observationNotes);
  native_detail::importFirstUse(function, out.program, ids, out.observationNotes);
  if (out.loopOwners.empty())
    native_detail::importFifoSlots(function, out.program, out.payload, out.observationNotes);
  while (out.payload.size() < out.program.operations.size())
    out.payload.push_back(out.payload[out.program.operations[out.payload.size()].original]);
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
LogicalResult import(func::FuncOp function, Import &out,
                     ObservationPolicy policy = ObservationPolicy::RefineLeafLoops) {
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
  // Consume the same physical nodes as existing InsertSync. An instruction
  // without a translated node contributes no synchronization effects. The
  // optional describeSemantics audit is not an instruction-admission gate.
  DenseMap<mlir::Operation *, CompoundInstanceElement *> phases;
  for (const auto &entry : translated) {
    auto *phase = dyn_cast<CompoundInstanceElement>(entry.get());
    if (!phase) continue;
    auto *op = phase->elementOp;
    // Every imported phase needs an actual insertion boundary. Never silently
    // discard later phases of a compound instruction at this adapter boundary.
    if (phases.count(op))
      return op->emitError("handoff: multiple translated phases require phase-aware native anchors");
    if (!pipe(static_cast<PIPE>(phase->kPipeValue)))
      return op->emitError(
          "handoff: pipeline outside native target contract");
    out.payload.push_back(op);
    phases[op] = phase;
  }
  // Retain available shared protocol metadata for reports and refinements.
  // Missing/incomplete metadata does not veto the translator's result.
  function.walk([&](mlir::Operation *op) {
    if (auto protocol = getSyncProtocolModel(op); protocol && protocol->complete())
      out.protocols.push_back(*protocol);
  });

  DenseMap<mlir::Operation *, unsigned> operationIds;
  for (auto [i, op] : llvm::enumerate(out.payload))
    operationIds[op] = i;
  std::vector<bool> represented(out.payload.size());
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
          represented[found->second] = true;
          sequence.children.push_back(std::move(phase));
        }
      }
    }
    return sequence;
  };
  out.program.body = importRegion(function.getBody());
  for (auto [i, op] : llvm::enumerate(out.payload))
    if (!represented[i])
      return op->emitError("handoff: translated phase lies outside supported structured control");

  struct Effect {
    unsigned operation;
    const BaseMemInfo *memory;
    bool write;
  };
  SmallVector<Effect> physicalEffects;
  // Qualify scalar slot evolution once, then specialize the SAME shared
  // footprint records. The canonical partition and unknown-alias witnesses
  // still account for every other access, including ACC and unqualified views.
  struct SlotCandidate {
    scf::ForOp loop;
    SyncSlotMapping mapping;
    DenseMap<const BaseMemInfo *, std::vector<const BaseMemInfo *>> memories;
    bool nested = false;
  };
  std::vector<SlotCandidate> slotCandidates;
  SmallVector<std::unique_ptr<BaseMemInfo>> slotMemory;
  // May-footprints do not assert which visit uses a bank. In particular, an
  // outer loop containing children can have a finite address orbit without
  // having a qualified recurring synchronization interface.
  DenseMap<const BaseMemInfo *, const BaseMemInfo *> finiteMemory;
  const auto slotTarget = a3SyncProfile(SyncCore::Cube);
  std::size_t maximumPeriod = 1;
  for (const auto &row : slotTarget.keys)
    for (const auto &pool : row) maximumPeriod = std::max(maximumPeriod, pool.size());
  function.walk([&](scf::ForOp loop) {
    bool nested = false;
    loop.getRegion().walk([&](mlir::Operation *op) { nested |= isa<scf::ForOp, scf::WhileOp>(op); });
    auto mapping = SyncSlotMapping::derive(loop, unsigned(maximumPeriod));
    if (!mapping) return;
    SlotCandidate candidate;
    candidate.loop = loop;
    candidate.nested = nested;
    candidate.mapping = std::move(*mapping);
    for (auto *payload : out.payload) {
      if (!loop->isProperAncestor(payload)) continue;
      auto *phase = phases.lookup(payload);
      auto qualify = [&](const auto &memories) {
        for (const BaseMemInfo *memory : memories) {
          if (candidate.memories.count(memory) || memory->scope == AddressSpace::GM ||
              memory->scope == AddressSpace::Zero || !memory->allocateSize ||
              memory->baseBuffer != memory->rootBuffer) continue;
          auto alloc = memory->rootBuffer.template getDefiningOp<AllocTileOp>();
          if (!alloc || !alloc.getAddr() || !loop->isProperAncestor(alloc) ||
              SyncSlotMapping::literal(alloc.getAddr())) continue;
          std::vector<uint64_t> addresses;
          for (auto &values : candidate.mapping.values) {
            auto address = SyncSlotMapping::evaluate(alloc.getAddr(), values);
            if (!address || memory->allocateSize > std::numeric_limits<uint64_t>::max() - *address) break;
            addresses.push_back(*address);
          }
          if (addresses.size() != candidate.mapping.period) continue;
          auto &variants = candidate.memories[memory];
          for (auto address : addresses) {
            auto copy = memory->clone();
            copy->baseAddresses = {address};
            copy->hasKnownPhysicalAddresses = true;
            copy->aliasesUnknownRange = false;
            variants.push_back(copy.get());
            slotMemory.push_back(std::move(copy));
          }
          auto possible = memory->clone();
          possible->baseAddresses.assign(addresses.begin(), addresses.end());
          possible->hasKnownPhysicalAddresses = true;
          possible->aliasesUnknownRange = false;
          finiteMemory[memory] = possible.get();
          slotMemory.push_back(std::move(possible));
        }
      };
      qualify(phase->defVec);
      qualify(phase->useVec);
    }
    if (!candidate.memories.empty()) {
      slotCandidates.push_back(std::move(candidate));
    }
  });
  DenseMap<const BaseMemInfo *, const std::vector<const BaseMemInfo *> *> bankMemories;
  for (const auto& candidate : slotCandidates) {
    for (const auto& item : candidate.memories) {
      bankMemories[item.first] = &item.second;
    }
  }
  // A missing absolute local address is not a distinct-allocation proof. Keep
  // legacy alias behavior unchanged; normalize the handoff import's private
  // records instead. Stable copies also preserve the original SSA effect names.
  SmallVector<std::unique_ptr<BaseMemInfo>> widenedMemory;
  DenseMap<const BaseMemInfo *, const BaseMemInfo *> normalizedMemory;
  auto qualifyMemory = [&](const BaseMemInfo *memory) -> const BaseMemInfo * {
    if (auto finite = finiteMemory.find(memory); finite != finiteMemory.end())
      return finite->second;
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
  }
  const auto originalPhases = out.program.operations.size();
  struct Variant { unsigned original, residue, operation; SlotCandidate *candidate; };
  std::vector<Variant> variants;
  for (auto &candidate : slotCandidates)
    for (unsigned i = 0; i < out.payload.size(); ++i)
      if (candidate.loop->isProperAncestor(out.payload[i]))
        for (unsigned residue = 0; residue < candidate.mapping.period; ++residue) {
          variants.push_back({i, residue, unsigned(out.program.operations.size()), &candidate});
          out.program.operations.push_back(out.program.operations[i]);
        }
  auto effects = [&](unsigned i, unsigned operation, SlotCandidate *candidate,
                     std::optional<unsigned> residue) -> LogicalResult {
    auto *phase = phases.lookup(out.payload[i]);
    auto add = [&](const auto &memories, bool write) -> LogicalResult {
      for (const BaseMemInfo *memory : memories) {
        // Admit the same translated storage scopes as production autosync.
        // Visibility remains a separate contract below; ordinary GM byte
        // completion is intentionally governed by production alias behavior.
        if (candidate && candidate->memories.count(memory)) {
          const auto &resolved = candidate->memories.find(memory)->second;
          if (residue) physicalEffects.push_back({operation, resolved[*residue], write});
          else for (auto *bank : resolved) physicalEffects.push_back({operation, bank, write});
        } else if (const auto banks = bankMemories.find(memory); banks != bankMemories.end()) {
          for (const auto* bank : *banks->second) {
            physicalEffects.push_back({operation, bank, write});
          }
        } else {
          physicalEffects.push_back({operation, qualifyMemory(memory), write});
        }
      }
      return success();
    };
    if (failed(add(phase->defVec, true)) || failed(add(phase->useVec, false)))
      return failure();
    return success();
  };
  for (unsigned i = 0; i < originalPhases; ++i) {
    SlotCandidate *candidate = nullptr;
    for (auto &slot : slotCandidates) if (slot.loop->isProperAncestor(out.payload[i])) candidate = &slot;
    if (failed(effects(i, i, candidate, {}))) return failure();
  }
  for (const auto &variant : variants)
    if (failed(effects(variant.original, variant.operation, variant.candidate, variant.residue))) return failure();
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
  // Variants are ordered by region, original phase and residue. Consume this
  // sparse population once rather than scanning it for every original phase.
  for (const auto &variant : variants) {
    auto &slot = out.slotLoops[variant.candidate->loop.getOperation()];
    slot.period = variant.candidate->mapping.period;
    slot.enclosing = variant.candidate->nested;
    if (slot.effects.empty() || slot.effects.back().operation != variant.original)
      slot.effects.push_back({variant.original, {}});
    slot.effects.back().residues.push_back(out.program.operations[variant.operation].accesses);
  }
  out.program.operations.resize(originalPhases);
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
  // Require the entire M-access population of an exact ACC atom to obey one
  // native accumulation contract. Unknown aliases, other M instructions,
  // mixed layouts/shapes, mutable valid dimensions and UnitFlag retain the
  // ordinary completion requirement. The actual effects remain unchanged.
  bool mutableDimensions = false;
  function.walk([&](SetValidShapeOp) { mutableDimensions = true; });
  if (cube && !mutableDimensions) {
    std::vector<std::optional<SyncAccumulatorOrder>> common(out.program.cells.size());
    std::vector<bool> invalid(out.program.cells.size());
    // Sparse effect incidences, rather than another cells-by-operations scan.
    for (unsigned i = 0; i < out.program.operations.size(); ++i) {
      const auto &op = out.program.operations[i];
      if (op.pipe != Pipe::M) continue;
      const auto order = syncAccumulatorOrder(out.payload[i]);
      out.program.operations[i].nativeMmadAccumulate = order && isa<TMatmulAccOp>(out.payload[i]);
      for (const auto &access : op.accesses) {
        const auto cell = access.cell;
        const auto &atom = out.program.cells[cell];
        if (invalid[cell] || atom.storage != Cell::Storage::CanonicalInterval || atom.unknownRange ||
            atom.addressSpace != std::to_string(unsigned(AddressSpace::ACC))) continue;
        if (!order || (common[cell] && (common[cell]->signature != order->signature ||
                                       common[cell]->destinationType != order->destinationType)))
          invalid[cell] = true;
        else common[cell] = order;
      }
    }
    for (unsigned cell = 0; cell < out.program.cells.size(); ++cell)
      out.program.cells[cell].nativeMmadAccOrder = !invalid[cell] && common[cell].has_value();
  }
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
  if (!out.protocols.empty())
    out.program.invocation.boundary +=
        "; preserved lowering-owned peer matching/participation/progress contracts; "
        "cross-core flags remain owned by the original protocol in their "
        "separate namespace; no local completion credit from peer events or "
        "preserved intrinsic drains";
  return importObservedCuts(function, out, policy);
}

// Emit only total predicates over the original normalized induction variable
// and upper bound. For an active visit 0<=i<N, N-i cannot overflow signed index
// range; no new iteration/history counter or event-state query is introduced.
struct PredicateCache {
  struct BlockValues {
    std::map<uint64_t, Value> constants;
    std::map<std::pair<std::size_t, uint64_t>, Value> residues;
    std::map<std::size_t, Value> remaining;
    std::map<std::tuple<std::size_t, unsigned, uint64_t, uint64_t>, Value> atoms;
  };
  DenseMap<Block *, BlockValues> blocks;
};
FailureOr<Value>
emitObservationPredicate(const OriginalObservation &observation,
                         mlir::Operation *anchor, const Import &input,
                         OpBuilder &builder, PredicateCache &cache) {
  Value predicate;
  auto loc = anchor->getLoc();
  auto &values = cache.blocks[anchor->getBlock()];
  auto reuse = [&](auto &table, const auto &key, auto build) -> Value {
    auto at = table.find(key);
    if (at != table.end()) {
      auto *definition = at->second.getDefiningOp();
      if (definition && definition->getBlock() == anchor->getBlock() &&
          definition->isBeforeInBlock(anchor))
        return at->second;
    }
    return table[key] = build();
  };
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
      return reuse(values.constants, value, [&]() -> Value {
        return builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(value));
      });
    };
    Value part;
    const auto atomIdentity = detail::atomKey(atom);
    auto prior = values.atoms.find(atomIdentity);
    if (prior != values.atoms.end() &&
        prior->second.getDefiningOp()->isBeforeInBlock(anchor)) {
      part = prior->second;
    } else if (atom.kind == ObservationAtom::LoopResidue) {
      lhs = reuse(values.residues, std::make_pair(atom.owner, atom.parameter), [&]() -> Value {
        return builder.create<arith::RemUIOp>(loc, lhs, constant(atom.parameter));
      });
      part = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs,
                                           constant(atom.value));
    } else if (atom.kind == ObservationAtom::LoopHasPrevious) {
      part = builder.create<arith::CmpIOp>(
          loc,
          atom.value ? arith::CmpIPredicate::sge : arith::CmpIPredicate::slt,
          lhs, constant(uint64_t(*native_detail::firstUseInteger(loop.getLowerBound())) +
                        atom.parameter * uint64_t(*native_detail::firstUseInteger(loop.getStep()))));
    } else if (atom.kind == ObservationAtom::LoopHasNext) {
      lhs = reuse(values.remaining, atom.owner, [&]() -> Value {
        return builder.create<arith::SubIOp>(loc, loop.getUpperBound(), lhs);
      });
      part = builder.create<arith::CmpIOp>(
          loc,
          atom.value ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sle,
          lhs, constant(atom.parameter));
    } else
      return anchor->emitError("handoff: unsupported native observation atom"),
             failure();
    values.atoms[atomIdentity] = part;
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
            uint64_t(*rhs) == uint64_t(*native_detail::firstUseInteger(loop.getLowerBound())) +
                atom.parameter * uint64_t(*native_detail::firstUseInteger(loop.getStep())) &&
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

using NativeConstructor = llvm::function_ref<Result(const Program &)>;
using NativeChecker = llvm::function_ref<Result(const Program &, const Commands &)>;

LogicalResult execute(func::FuncOp function, NativeConstructor constructor,
                      NativeChecker checker,
                      ObservationPolicy policy,
                      llvm::function_ref<void(func::FuncOp)> mutate = {}) {
  Import input;
  if (failed(import(function, input, policy)))
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
  auto result = constructor(input.program);
  if (!result.success) {
    return function.emitError("handoff: ") << result.reason;
  }
  // Use production emission directly. Do not run legacy planning, motion,
  // redundancy removal or allocation on the new constructor's chosen plan.
  SyncIRs emission;
  SmallVector<std::unique_ptr<SyncOperation>> storage;
  struct EmittedWord {
    SmallVector<Cut> cuts;
    mlir::Operation *anchor;
    scf::IfOp guard;
    detail::ObservationUnion predicates;
    bool receiptPrelude = false;
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
    const auto &word = result.commands[cut];
    const bool receiptPrelude = input.program.observed->observations[
        input.program.observed->sites[cut].observation].beforeSharedWord;
    auto sameWord = [&](const auto &candidate) {
      const auto &other = result.commands[candidate.cuts.front()];
      return candidate.anchor == input.anchors[cut] &&
          candidate.receiptPrelude == receiptPrelude && word.size() == other.size() &&
          std::equal(word.begin(), word.end(), other.begin(), [](const auto &a, const auto &b) {
            return a.kind == b.kind && a.source == b.source && a.observer == b.observer && a.key == b.key;
          });
    };
    auto found = llvm::find_if(words, sameWord);
    if (found == words.end()) {
      words.push_back({{}, input.anchors[cut], {}, {}, receiptPrelude});
      found = words.end() - 1;
    }
    found->cuts.push_back(cut);
    const auto observation = input.program.observed->sites[cut].observation;
    found->predicates.push_back(input.program.observed->observations[observation].atoms);
  }
  DenseMap<mlir::Operation *, std::size_t> originalOrder;
  for (auto [index, op] : llvm::enumerate(originalOperations)) originalOrder[op] = index;
  llvm::stable_sort(words, [&](const auto &a, const auto &b) {
    if (a.anchor == b.anchor) return a.receiptPrelude && !b.receiptPrelude;
    return originalOrder.lookup(a.anchor) < originalOrder.lookup(b.anchor);
  });
  PredicateCache predicateCache;
  for (auto &word : words) {
    const auto cut = word.cuts.front();
    mlir::Operation *wordAnchor = word.anchor;
    scf::IfOp generatedGuard;
    word.predicates = detail::simplifyObservationUnion(std::move(word.predicates));
    if (word.predicates.size() != 1 || !word.predicates.front().empty()) {
      OpBuilder builder(wordAnchor);
      Value condition;
      for (const auto &term : word.predicates) {
        OriginalObservation descriptor{0, term, true};
        auto predicate = emitObservationPredicate(descriptor, wordAnchor, input, builder, predicateCache);
        if (failed(predicate)) return failure();
        condition = condition ? Value(builder.create<arith::OrIOp>(wordAnchor->getLoc(), condition, *predicate)) : *predicate;
      }
      generatedGuard =
          builder.create<scf::IfOp>(wordAnchor->getLoc(), condition, false);
      wordAnchor = generatedGuard.getThenRegion().front().getTerminator();
    }
    word.anchor = wordAnchor;
    word.guard = generatedGuard;
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
  SyncCodegen codegen(emission, function, SyncAnalysisMode::NORMALSYNC,
                      /*preserveCommandWords=*/true);
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
    const auto cut = word.cuts.front();
    auto guard = word.guard;
    const bool unconditional = word.predicates.size() == 1 && word.predicates.front().empty();
    if (bool(word.guard) == unconditional)
      return function.emitError("handoff: emitted guard participation changed");
    if (guard) {
      SmallVector<Value> alternatives;
      std::function<void(Value)> flatten = [&](Value value) {
        if (auto either = value.getDefiningOp<arith::OrIOp>()) {
          flatten(either.getLhs()); flatten(either.getRhs());
        } else alternatives.push_back(value);
      };
      flatten(guard.getCondition());
      if (alternatives.size() != word.predicates.size())
        return function.emitError("handoff: emitted observation union changed");
      for (std::size_t i = 0; i < alternatives.size(); ++i) {
        OriginalObservation descriptor{0, word.predicates[i], true};
        if (failed(checkObservationPredicate(descriptor, input.anchors[cut], input, alternatives[i])))
          return failure();
      }
    }
    SmallVector<mlir::Operation *> backwards;
    for (auto *op = word.anchor->getPrevNode();
         op && isa<SetFlagOp, WaitFlagOp, BarrierOp>(op) &&
         !originalSet.contains(op);
         op = op->getPrevNode())
      backwards.push_back(op);
    for (auto *op : llvm::reverse(backwards)) {
      if (!readCommands.insert(op).second || failed(read(op, cut)))
        return failure();
    }
    for (auto member : word.cuts)
      if (member != cut) actual[member] = actual[cut];
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
  auto checked = checker(input.program, actual);
  if (!checked.success) {
    return function.emitError("handoff emitted verification: ")
           << checked.reason;
  }
  return mlir::verify(function);
}

LogicalResult executeTransaction(func::FuncOp function, NativeConstructor constructor,
                                 NativeChecker checker,
                                 llvm::function_ref<void(func::FuncOp)> mutate,
                                 ObservationPolicy policy = ObservationPolicy::RefineLeafLoops) {
  if (getTargetArch(function) != PTOArch::A3) {
    return function.emitError("handoff: first native target is a3");
  }
  auto parent = function->getParentOfType<ModuleOp>();
  OwningOpRef<ModuleOp> sandbox = ModuleOp::create(function.getLoc());
  if (parent) {
    sandbox->getOperation()->setAttrs(parent->getAttrs());
  }
  auto working = cast<func::FuncOp>(function->clone());
  sandbox->push_back(working);
  if (failed(execute(working, constructor, checker, policy, mutate))) {
    return failure();
  }
  function.getBody().takeBody(working.getBody());
  return success();
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

namespace {
LogicalResult executeSelectedHandoffSync(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate, SelectedPlan *report,
    SelectedOptions options = {}) {
  if (report) {
    *report = SelectedPlan{};
  }
  // Choose observations before construction from shared physical access roles.
  // Qualified normalized loops use original first/next-use guards; unrelated
  // loops retain their original SCF graph. Bounding geometry never becomes a
  // full-write certificate, and a construction refusal does not trigger retry.
  return executeTransaction(function,
      [report, options](const Program &program) {
        auto selected = constructSelectedPlan(program, {}, options);
        Result result;
        result.success = selected.success;
        result.reason = selected.reason;
        result.commands = selected.commands;
        if (report) {
          *report = std::move(selected);
        }
        return result;
      },
      [](const Program &program, const Commands &commands) {
        const auto checked = checkCausalFrontier(program, commands);
        Result result;
        result.success = checked.accepted;
        result.reason = checked.reason;
        return result;
      }, mutate, selectedObservationPolicy(options.classInvariantInputs, options.firstWriteConsumers, options.finalReadSources));
}

} // namespace

LogicalResult testing::runHandoffSyncWithMutation(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate) {
  return executeSelectedHandoffSync(function, mutate, nullptr);
}

LogicalResult testing::runSelectedHandoffSyncWithMutation(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate, SelectedPlan *report,
    const SelectedOptions *options) {
  return executeSelectedHandoffSync(function, mutate, report, options ? *options : SelectedOptions{});
}

namespace {
LogicalResult analyzeHandoffSyncWithPolicy(func::FuncOp function,
                                           NativeAnalysis &result,
                                           ObservationPolicy policy) {
  result = NativeAnalysis{};
  if (getTargetArch(function) != PTOArch::A3)
    return function.emitError("handoff: native analysis target is a3");
  Import input;
  if (failed(import(function, input, policy)))
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
} // namespace

LogicalResult testing::analyzeSelectedHandoffSync(func::FuncOp function,
                                                  NativeAnalysis &result, bool classInvariantInputs, bool firstWriteConsumers, bool finalReadSources) {
  return analyzeHandoffSyncWithPolicy(function, result, selectedObservationPolicy(classInvariantInputs, firstWriteConsumers, finalReadSources));
}

LogicalResult runHandoffSync(func::FuncOp function) {
  return executeSelectedHandoffSync(function, {}, nullptr);
}

LogicalResult analyzeHandoffSync(func::FuncOp function,
                                 NativeAnalysis &result) {
  return analyzeHandoffSyncWithPolicy(function, result,
                                      ObservationPolicy::RefineLeafLoops);
}
LogicalResult analyzeHandoffSync(func::FuncOp function) {
  NativeAnalysis result;
  return analyzeHandoffSync(function, result);
}
} // namespace mlir::pto::oahs
