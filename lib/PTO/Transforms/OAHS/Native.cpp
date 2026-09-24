// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Native.h"
#include "ObservationUnion.h"
#include "OriginalReadQueries.h"
#include "SelectedInternal.h"
#include "NativeFirstUse.h"
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
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
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
  mutable DominanceInfo dominance;
  Program program;
  SyncSlotMapping::AnalysisContext scalarFacts;
  uint64_t endpointDiscoveryWork = 0;
  SmallVector<SyncProtocolModel, 0> protocols;
  SmallVector<mlir::Operation *> payload;
  SmallVector<Value> storageRoots;
  SmallVector<mlir::Operation *> anchors;
  SmallVector<Cut> phaseCuts;
  DenseMap<std::size_t, scf::ForOp> loopOwners;
  DenseMap<std::size_t, SyncSlotMapping::LoopDomain> loopDomains;
  struct SlotLoop {
    unsigned period = 1;
    std::vector<CountedLoopRegion::PeriodicEffects> effects;
    bool enclosing = false;
  };
  DenseMap<mlir::Operation *, SlotLoop> slotLoops;
  std::vector<std::pair<mlir::Operation *, PhysicalUseRelation>> physicalUses;
  std::vector<std::string> observationNotes;
};
enum class ObservationPolicy {
  OriginalControl, RefineLeafLoops, QualifiedAccessRolesWithoutFirstUse,
  QualifiedAccessRoles
};
// Every actual original instruction is an anchor, including scalar/control
// instructions and region terminators. Synthetic branch/loop decisions have no
// anchor and cannot acquire emitted commands. Payload phases remain unchanged.
LogicalResult importObservedCuts(func::FuncOp function, Import &out,
                                 ObservationPolicy policy) {
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
  for (auto& [owner, relation] : out.physicalUses) {
    relation.owner = ids.lookup(owner);
    out.program.physicalUses.push_back(std::move(relation));
  }
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
  const auto captured = captureOriginalStructure(out.program);
  if (!captured.success) {
    out.observationNotes.push_back("original structural query unavailable: " + captured.reason);
  }
  if (policy == ObservationPolicy::OriginalControl) {
    return success();
  }
  constexpr std::size_t ObservationSiteBudget = 4096;
  // Leaf loops are refined independently. No product of unrelated loop periods
  // is formed. Unsupported arithmetic keeps the original sound SCF graph.
  auto &constants = out.scalarFacts.constants;
  auto &ranges = out.scalarFacts.ranges;
  auto integer = [&](Value value) { return SyncSlotMapping::evaluateConstant(value, constants); };
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  DenseMap<mlir::Operation *, SyncSlotMapping::LoopDomain> domains;
  for (auto loop : loops) {
    const auto &domain = out.scalarFacts.domain(loop);
    if (domain) {
      domains.try_emplace(loop.getOperation(), *domain);
    }
  }
  // Snapshot original requirements once: later control refinement must not
  // erase another endpoint role or manufacture a new source of causal credit.
  std::map<std::size_t, bool> endpointSeparation;
  if (!loops.empty()) {
    selected::Control control(out.program);
    StorageFrontierAnalysis storage(out.program);
    selected::RequirementFrontiers requirements(out.program, control, storage);
    if (!requirements.complete()) {
      out.observationNotes.push_back("kept original endpoint vocabulary: " + requirements.reason());
    }
    for (auto loop : loops) {
      const auto owner = ids.lookup(loop.getOperation());
      endpointSeparation.emplace(owner, requirements.complete() && requirements.needsOccurrenceSeparation(owner));
    }
    const auto& work = storage.stats();
    out.endpointDiscoveryWork = work.staticSites + work.forwardEvaluations + work.backwardEvaluations +
        work.nearestUseEvaluations + work.useSummarySites + work.useSummaryEdges +
        work.originalReadRegions + work.participationNodes + work.readFrontierNodes +
        work.readCompositionParts + requirements.size() +
        control.loopEntryPreparationSites +
        requirements.endpointClassificationWork() + control.transitionClassificationWork;
  }
  std::map<std::size_t, std::pair<std::vector<std::size_t>, std::vector<std::size_t>>> initialParticipation;
  for (auto loop : loops) {
    bool nested = false;
    loop.getRegion().walk([&](mlir::Operation *op) {
      nested |= isa<scf::ForOp, scf::WhileOp>(op);
    });
    const auto domain = domains.find(loop.getOperation());
    if (nested || domain == domains.end()) {
      continue;
    }
    CountedLoopRegion model;
    model.atLeastOnce = domain->second.atLeastOnce;
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
    const bool physicalResidues = slots != out.slotLoops.end() && slots->second.period > 1;
    if (!physicalResidues && !endpointSeparation.at(model.owner)) {
      continue; // No currently supported occurrence separation was requested.
    }
    struct DecisionRelation {
      unsigned period = 0;
      std::map<uint64_t, std::optional<unsigned>> uniqueResidues;
    };
    DenseMap<Value, DecisionRelation> decisionRelations;
    loop.getRegion().walk([&](scf::IfOp choice) {
      auto cmp = choice.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!cmp || (cmp.getPredicate() != arith::CmpIPredicate::eq &&
                   cmp.getPredicate() != arith::CmpIPredicate::ne))
        return;
      Value expression = cmp.getLhs(), literal = cmp.getRhs();
      if (integer(expression)) {
        std::swap(expression, literal);
      }
      const auto literalValue = integer(literal);
      if (!literalValue) {
        return;
      }
      auto foundRelation = decisionRelations.find(expression);
      if (foundRelation == decisionRelations.end()) {
        DecisionRelation indexed;
        if (auto relation = SyncSlotMapping::derive(loop, expression, out.scalarFacts)) {
          indexed.period = relation->period;
          for (unsigned residue = 0; residue < relation->period; ++residue) {
            const auto value = relation->values[residue].find(expression);
            if (value == relation->values[residue].end()) {
              indexed.period = 0;
              break;
            }
            auto entry = indexed.uniqueResidues.emplace(value->second, residue);
            if (!entry.second) {
              entry.first->second.reset();
            }
          }
          out.endpointDiscoveryWork += relation->period;
        }
        foundRelation = decisionRelations.try_emplace(expression, std::move(indexed)).first;
      }
      const auto &relation = foundRelation->second;
      if (!relation.period || (model.period != 1 && model.period % relation.period != 0)) {
        return; // Keep independent or unrepresented original participation.
      }
      const auto matching = relation.uniqueResidues.find(*literalValue);
      if (matching == relation.uniqueResidues.end() || !matching->second) {
        return; // The current decision interface represents one matching residue.
      }
      if (model.period == 1) {
        model.period = relation.period;
      }
      model.decisions.push_back({ids.lookup(choice.getOperation()), relation.period,
          *matching->second, cmp.getPredicate() == arith::CmpIPredicate::eq});
    });
    // Optional control materialization has a separate work budget. Precise
    // physical relations above remain available when this expansion is skipped.
    if (model.bodySites.size() > ObservationSiteBudget / (2 * model.period)) {
      out.observationNotes.push_back("kept original control: occurrence materialization budget");
      continue;
    }
    auto refined = refineCountedLoop(out.program, model);
    if (!refined.success) {
      out.observationNotes.push_back("kept original SCF control: " +
                                     refined.reason);
      continue;
    }
    // Roles were collected together from the original program, including
    // readers whose producer is outside this child. Successful local private
    // protocol construction is not a prerequisite for preserving those roles.
    out.program = std::move(refined.program);
    if (!model.atLeastOnce) {
      const auto& alternatives = out.program.observed->sites[model.owner].successors;
      initialParticipation.emplace(model.owner, std::make_pair(
          std::vector<std::size_t>{alternatives.front()},
          std::vector<std::size_t>(alternatives.begin() + 1, alternatives.end())));
    }
    out.loopOwners[model.owner] = loop;
    out.loopDomains[model.owner] = domain->second;
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
    const auto domain = domains.find(loop.getOperation());
    if (!usableOrbit || domain == domains.end()) {
      continue;
    }
    const auto& control = *out.program.observed;
    CountedLoopRegion model;
    model.owner = ids.lookup(loop.getOperation());
    model.period = slots->second.period;
    model.effects = slots->second.effects;
    model.atLeastOnce = domain->second.atLeastOnce;
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
    // Bound added materialization, including copied headers, independently of
    // private event feasibility. Physical-use relations remain in the program.
    const bool exceedsSiteBudget = model.bodySites.size() >= ObservationSiteBudget / (model.period - 1);
    if (exceedsSiteBudget) {
      out.observationNotes.push_back("kept original bank control: occurrence materialization budget");
      continue;
    }
    auto refined = refineBankOccurrences(out.program, model);
    if (!refined.success) {
      out.observationNotes.push_back("kept original bank interface: " + refined.reason);
      continue;
    }
    out.program = std::move(refined.program);
    out.loopOwners[model.owner] = loop;
    out.loopDomains[model.owner] = domain->second;
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
    const auto domain = domains.find(loop.getOperation());
    if (domain == domains.end() || !domain->second.atLeastOnce) {
      continue;
    }
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
  // D3 demands, not protocol admission, select the original predicates worth
  // retaining. Each interval is refined independently; no function-wide product.
  if (out.program.originalStructure && !initialParticipation.empty()) {
    storage_detail::OriginalReadQueries reads(out.program);
    const auto demands = reads.participationDemands();
    std::vector<ParticipationRegion> participation;
    std::map<std::size_t, const Region*> originalOwners;
    std::function<void(const Region&)> index = [&](const Region& region) {
      if (region.originalOwner != NoControlId) { originalOwners.emplace(region.originalOwner, &region); }
      for (const auto& child : region.children) { index(child); }
    };
    index(out.program.originalStructure->body);
    auto anchor = [&](const Region& region) -> mlir::Operation* {
      if (region.kind == Region::Operation) {
        return region.operation < out.payload.size() ? out.payload[region.operation] : nullptr;
      }
      return region.originalOwner < out.anchors.size() ? out.anchors[region.originalOwner] : nullptr;
    };
    std::map<std::pair<std::size_t, std::size_t>, bool> intervalAvailability;
    for (const auto& demand : demands) {
      const auto original = originalOwners.find(demand.interval.owner);
      const auto source = initialParticipation.find(demand.predicate.owner);
      const auto subject = out.loopOwners.find(demand.predicate.owner);
      const bool represented = original != originalOwners.end() && source != initialParticipation.end() &&
          subject != out.loopOwners.end();
      if (!represented) {
        continue;
      }
      const auto& region = *original->second;
      const bool sequence = region.children.size() == 1 && region.children.front().kind == Region::Sequence;
      if (!sequence) { continue; }
      const auto& parts = region.children.front().children;
      const auto begin = demand.interval.begin, end = demand.interval.end;
      if (begin >= end || end > parts.size()) { continue; }
      auto* start = anchor(parts[begin]);
      auto ownerLoop = dyn_cast_or_null<scf::ForOp>(out.anchors[demand.interval.owner]);
      if (!ownerLoop || !start) { continue; }
      auto* stop = end == parts.size() ? ownerLoop.getBody()->getTerminator() : anchor(parts[end]);
      if (!stop) { continue; }
      auto child = subject->second;
      const bool boundsAvailable = out.dominance.dominates(child.getLowerBound(), start) &&
          out.dominance.dominates(child.getUpperBound(), start);
      if (!boundsAvailable) {
        out.observationNotes.push_back("kept original participation: sibling bounds unavailable at interval entry");
        continue;
      }
      ParticipationRegion model;
      model.entry = ids.lookup(start); model.exit = ids.lookup(stop);
      model.decision = demand.predicate.owner; model.predicate = demand.predicate;
      model.whenFalse = source->second.first; model.whenTrue = source->second.second;
      model.available = true;
      // Bounds dominating entry are available at every original anchor that
      // entry dominates. Prove this geometry once per interval, independently
      // of the number of sibling predicates demanded there.
      const auto key = std::make_pair(model.entry, model.exit);
      auto prior = intervalAvailability.find(key);
      if (prior == intervalAvailability.end()) {
        std::set<std::size_t> members;
        std::vector<std::size_t> pending{model.entry};
        bool available = true;
        while (!pending.empty()) {
          const bool exhausted = members.size() > ObservationSiteBudget / 2;
          if (exhausted) { break; }
          const auto site = pending.back(); pending.pop_back();
          if (site == model.exit || !members.insert(site).second) { continue; }
          const auto& node = out.program.observed->sites[site];
          if (node.observation != NoControlId) {
            const auto cut = out.program.observed->observations[node.observation].anchor;
            auto* position = cut < out.anchors.size() ? out.anchors[cut] : nullptr;
            available &= position && (position == start || start->isProperAncestor(position) ||
                                       out.dominance.dominates(start, position));
          }
          pending.insert(pending.end(), node.successors.begin(), node.successors.end());
        }
        out.endpointDiscoveryWork += members.size();
        prior = intervalAvailability.emplace(key, available && members.size() <= ObservationSiteBudget / 2).first;
      }
      if (!prior->second) {
        out.observationNotes.push_back("kept original participation: availability or materialization budget unknown");
        continue;
      }
      participation.push_back(std::move(model));
    }
    if (!participation.empty()) {
      auto refined = refineParticipations(out.program, participation);
      out.endpointDiscoveryWork += refined.work;
      for (const auto& reason : refined.refusals) {
        if (!reason.empty()) { out.observationNotes.push_back("kept original participation: " + reason); }
      }
      if (refined.success) {
        if (!refined.accepted.empty()) {
          out.program = std::move(refined.program);
          out.observationNotes.push_back("qualified original read-interval participation");
        }
      } else { out.observationNotes.push_back("kept original participation batch: " + refined.reason); }
    }
    out.endpointDiscoveryWork += reads.evaluations + reads.compositionParts +
        reads.predicateCount() + reads.frontierCount();
  }
  out.endpointDiscoveryWork += constants.evaluations + ranges.evaluations;
  if (policy != ObservationPolicy::QualifiedAccessRolesWithoutFirstUse) {
    native_detail::importFirstUse(function, out.program, ids, out.observationNotes);
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
  // An unreachable canonical site can still name a word executed by a live
  // alias. Only observations with no reachable occurrence lose their anchor.
  const auto& observed = *out.program.observed;
  std::vector<bool> liveObservation(observed.observations.size());
  for (Cut at = 0; at < out.anchors.size(); ++at) {
    if (reached[at] && legalCommandCut(out.program, at)) {
      liveObservation[observed.sites[at].observation] = true;
    }
  }
  for (Cut at = 0; at < out.anchors.size(); ++at) {
    const auto observation = observed.sites[at].observation;
    if (observation == NoControlId || !liveObservation[observation]) {
      out.anchors[at] = nullptr;
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
    if (!phase) {
      continue;
    }
    auto *op = phase->elementOp;
    // Every imported phase needs an actual insertion boundary. Never silently
    // discard later phases of a compound instruction at this adapter boundary.
    if (phases.count(op)) {
      return op->emitError("handoff: multiple translated phases require phase-aware native anchors");
    }
    if (!pipe(static_cast<PIPE>(phase->kPipeValue))) {
      return op->emitError(
          "handoff: pipeline outside native target contract");
    }
    out.payload.push_back(op);
    phases[op] = phase;
  }
  // Retain available shared protocol metadata for reports and refinements.
  // Missing/incomplete metadata does not veto the translator's result.
  function.walk([&](mlir::Operation *op) {
    if (auto protocol = getSyncProtocolModel(op); protocol && protocol->complete()) {
      out.protocols.push_back(*protocol);
    }
  });

  DenseMap<mlir::Operation *, unsigned> operationIds;
  for (auto [i, op] : llvm::enumerate(out.payload))
    operationIds[op] = i;
  std::vector<bool> represented(out.payload.size());
  DenseMap<mlir::Operation*, std::size_t> originalAnchors;
  function.walk<WalkOrder::PreOrder>([&](mlir::Operation* operation) {
    if (operation != function.getOperation()) {
      originalAnchors.try_emplace(operation, originalAnchors.size());
    }
  });
  std::function<Region(mlir::Region &)> importRegion =
      [&](mlir::Region &region) -> Region {
    Region sequence;
    sequence.kind = Region::Sequence;
    for (Block &block : region) {
      for (mlir::Operation &op : block) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          Region choice;
          choice.kind = Region::Choice;
          choice.originalOwner = originalAnchors.lookup(&op);
          choice.children.push_back(importRegion(ifOp.getThenRegion()));
          choice.children.push_back(importRegion(ifOp.getElseRegion()));
          sequence.children.push_back(std::move(choice));
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          Region loop;
          loop.kind = Region::For;
          loop.originalOwner = originalAnchors.lookup(&op);
          const auto& domain = out.scalarFacts.domain(forOp);
          loop.qualifiedCounted = bool(domain);
          loop.zeroTripPossible = !domain || !domain->atLeastOnce;
          loop.children.push_back(importRegion(forOp.getRegion()));
          sequence.children.push_back(std::move(loop));
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
          Region loop;
          loop.kind = Region::While;
          loop.originalOwner = originalAnchors.lookup(&op);
          loop.children.push_back(importRegion(whileOp.getBefore()));
          loop.children.push_back(importRegion(whileOp.getAfter()));
          sequence.children.push_back(std::move(loop));
        } else if (auto found = operationIds.find(&op);
                   found != operationIds.end()) {
          Region phase;
          phase.kind = Region::Operation;
          phase.originalOwner = originalAnchors.lookup(&op);
          phase.operation = found->second;
          represented[found->second] = true;
          sequence.children.push_back(std::move(phase));
        }
      }
    }
    return sequence;
  };
  out.program.body = importRegion(function.getBody());
  for (auto [i, op] : llvm::enumerate(out.payload)) {
    if (!represented[i]) {
      return op->emitError("handoff: translated phase lies outside supported structured control");
    }
  }

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
    unsigned period = 1;
    DenseMap<const BaseMemInfo *, std::vector<const BaseMemInfo *>> memories;
    bool nested = false;
    bool materialize = false;
  };
  std::vector<SlotCandidate> slotCandidates;
  SmallVector<std::unique_ptr<BaseMemInfo>> slotMemory;
  // May-footprints do not assert which visit uses a bank. In particular, an
  // outer loop containing children can have a finite address orbit without
  // having a qualified recurring synchronization interface.
  DenseMap<const BaseMemInfo *, const BaseMemInfo *> finiteMemory;
  function.walk([&](scf::ForOp loop) {
    SlotCandidate candidate;
    candidate.loop = loop;
    loop.getRegion().walk([&](mlir::Operation *op) {
      candidate.nested |= isa<scf::ForOp, scf::WhileOp>(op);
    });
    DenseMap<Value, std::optional<SyncSlotMapping>> relations;
    DenseSet<const BaseMemInfo *> examined;
    for (auto *payload : out.payload) {
      if (!loop->isProperAncestor(payload)) {
        continue;
      }
      auto *phase = phases.lookup(payload);
      auto qualify = [&](const auto &memories) {
        for (const BaseMemInfo *memory : memories) {
          if (!examined.insert(memory).second || memory->scope == AddressSpace::GM ||
              memory->scope == AddressSpace::Zero || !memory->allocateSize ||
              memory->aliasesUnknownRange || memory->baseAddresses.empty()) {
            continue;
          }
          // Rebase the shared translated footprint, including view offsets and
          // conservative envelopes. The effect's immediate SSA spelling is not
          // an allocation-identity proof. Unknown origins remain unknown.
          auto alloc = memory->rootBuffer.template getDefiningOp<AllocTileOp>();
          if (!alloc || !alloc.getAddr() || !loop->isProperAncestor(alloc) ||
              memory->hasKnownPhysicalAddresses) {
            continue;
          }
          auto relation = relations.find(alloc.getAddr());
          if (relation == relations.end()) {
            relation = relations.try_emplace(alloc.getAddr(),
                SyncSlotMapping::derive(loop, alloc.getAddr(), out.scalarFacts)).first;
          }
          if (!relation->second) {
            continue;
          }
          auto &mapping = *relation->second;
          std::vector<SmallVector<uint64_t>> addresses;
          for (auto &values : mapping.values) {
            const auto address = SyncSlotMapping::evaluate(alloc.getAddr(), values);
            SmallVector<uint64_t> footprint;
            for (auto offset : memory->baseAddresses) {
              if (!address || offset > std::numeric_limits<uint64_t>::max() - *address ||
                  memory->allocateSize > std::numeric_limits<uint64_t>::max() - (*address + offset)) {
                break;
              }
              footprint.push_back(*address + offset);
            }
            if (footprint.size() != memory->baseAddresses.size()) {
              break;
            }
            addresses.push_back(std::move(footprint));
          }
          if (addresses.size() != mapping.period) {
            continue;
          }
          auto &variants = candidate.memories[memory];
          auto possible = memory->clone();
          possible->baseAddresses.clear();
          for (const auto &footprint : addresses) {
            auto copy = memory->clone();
            copy->baseAddresses = footprint;
            copy->hasKnownPhysicalAddresses = true;
            copy->aliasesUnknownRange = false;
            variants.push_back(copy.get());
            slotMemory.push_back(std::move(copy));
            possible->baseAddresses.append(footprint.begin(), footprint.end());
          }
          possible->hasKnownPhysicalAddresses = true;
          possible->aliasesUnknownRange = false;
          finiteMemory[memory] = possible.get();
          slotMemory.push_back(std::move(possible));
          // The existing occurrence graph has one phase dimension per owner.
          // Retain every finite relation, but materialize only periods that
          // divide this selected dimension. Never form an unrelated LCM.
          if (candidate.period == 1 || mapping.period % candidate.period == 0) {
            candidate.period = mapping.period;
          }
        }
      };
      qualify(phase->defVec);
      qualify(phase->useVec);
    }
    if (!candidate.memories.empty()) {
      std::set<unsigned> periods;
      for (const auto& relation : candidate.memories) {
        periods.insert(relation.second.size());
      }
      const auto selected = candidate.period;
      for (auto period : periods) {
        auto independent = candidate;
        independent.period = period;
        independent.materialize = period == selected;
        slotCandidates.push_back(std::move(independent));
      }
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
        for (unsigned residue = 0; residue < candidate.period; ++residue) {
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
          if (residue && candidate->period % resolved.size() == 0) {
            physicalEffects.push_back({operation, resolved[*residue % resolved.size()], write});
          } else {
            for (auto *bank : resolved) {
              physicalEffects.push_back({operation, bank, write});
            }
          }
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
  SlotCandidate *previous = nullptr;
  for (const auto &variant : variants) {
    if (previous != variant.candidate) {
      out.physicalUses.push_back({variant.candidate->loop.getOperation(),
                                 {NoControlId, variant.candidate->period, {}}});
      previous = variant.candidate;
    }
    auto& relation = out.physicalUses.back().second;
    if (relation.effects.empty() || relation.effects.back().operation != variant.original) {
      relation.effects.push_back({variant.original, {}});
    }
    relation.effects.back().residues.push_back(out.program.operations[variant.operation].accesses);
    if (variant.candidate->materialize) {
      auto &slot = out.slotLoops[variant.candidate->loop.getOperation()];
      slot.period = relation.period;
      slot.enclosing = variant.candidate->nested;
      if (slot.effects.empty() || slot.effects.back().operation != variant.original) {
        slot.effects.push_back({variant.original, {}});
      }
      slot.effects.back().residues.push_back(out.program.operations[variant.operation].accesses);
    }
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
  // Intern the shared target contract on access incidences. Different or
  // unqualified accesses to the same cell remain independent obligations.
  if (cube) {
    const SyncTileDescriptorState descriptors(function);
    using Contract = std::tuple<std::array<uint64_t, 3>, std::vector<int64_t>, const void*, const void*>;
    std::map<Contract, std::size_t> contracts;
    for (auto& cell : out.program.cells) {
      if (cell.addressSpace == std::to_string(unsigned(AddressSpace::ACC))) {
        cell.domain = Cell::Domain::Accumulator;
      }
    }
    for (unsigned i = 0; i < out.program.operations.size(); ++i) {
      auto& op = out.program.operations[i];
      if (op.pipe != Pipe::M) {
        continue;
      }
      const auto order = syncAccumulatorOrder(out.payload[i], descriptors, buffers);
      op.nativeMmadAccumulate = order && isa<TMatmulAccOp>(out.payload[i]);
      if (!order) {
        continue;
      }
      const auto shape = order->destinationType.getShape();
      const Contract contract{order->signature, {shape.begin(), shape.end()},
          order->destinationType.getElementType().getAsOpaquePointer(),
          order->destinationType.getConfigAttr().getAsOpaquePointer()};
      const auto id = contracts.try_emplace(contract, contracts.size()).first->second;
      for (auto& access : op.accesses) {
        const auto& cell = out.program.cells[access.cell];
        if (cell.domain == Cell::Domain::Accumulator &&
            cell.storage == Cell::Storage::CanonicalInterval && !cell.exclusive && !cell.unknownRange) {
          access.nativeAccumulatorClass = id;
        }
      }
    }
    out.program.nativeAccumulatorClasses = contracts.size();
    // A periodic overlay must retain a proof only for the identical original
    // effect. Moving an address does not move its ACC contract. Unknown or new
    // footprints keep ordinary access obligations until separately proved.
    auto annotate = [&](auto& effects) {
      for (auto& binding : effects) {
        std::map<std::tuple<unsigned, bool, bool, bool>, std::size_t> exact;
        for (const auto& access : out.program.operations[binding.operation].accesses) {
          const auto key = std::make_tuple(access.cell, access.read, access.write, access.definiteWrite);
          const auto inserted = exact.emplace(key, access.nativeAccumulatorClass);
          if (!inserted.second && inserted.first->second != access.nativeAccumulatorClass) {
            inserted.first->second = NoControlId;
          }
        }
        for (auto& residue : binding.residues) {
          for (auto& access : residue) {
            const auto found = exact.find({access.cell, access.read, access.write, access.definiteWrite});
            access.nativeAccumulatorClass = found == exact.end() ? NoControlId : found->second;
          }
        }
      }
    };
    for (auto& relation : out.physicalUses) {
      annotate(relation.second.effects);
    }
    for (auto& slot : out.slotLoops) {
      annotate(slot.second.effects);
    }
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

// Emit total predicates over proved original loop domains. Active visits have
// lower <= IV < upper and ordinal (IV-lower)/step. Remaining distance upper-IV
// is nonnegative and representable; no history counter or event query is added.
struct PredicateCache {
  struct BlockValues {
    std::map<uint64_t, Value> constants;
    std::map<std::pair<std::size_t, uint64_t>, Value> residues;
    std::map<std::size_t, Value> remaining, ordinals;
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
        (atom.kind != ObservationAtom::LoopNonEmpty && !owner->second->isProperAncestor(anchor)))
      return anchor->emitError("handoff: original loop values unavailable at "
                               "observed endpoint"),
             failure();
    auto loop = owner->second;
    const auto domain = input.loopDomains.find(atom.owner);
    if (domain == input.loopDomains.end()) {
      return anchor->emitError("handoff: missing original iteration-domain proof"), failure();
    }
    if (atom.kind == ObservationAtom::LoopNonEmpty &&
        (!input.dominance.dominates(loop.getLowerBound(), anchor) ||
         !input.dominance.dominates(loop.getUpperBound(), anchor))) {
      return anchor->emitError("handoff: original trip bounds unavailable at endpoint"), failure();
    }
    Value lhs = loop.getInductionVar();
    auto constant = [&](uint64_t value) -> Value {
      return reuse(values.constants, value, [&]() -> Value {
        return builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(value));
      });
    };
    auto ordinal = [&]() -> Value {
      return reuse(values.ordinals, atom.owner, [&]() -> Value {
        Value index = loop.getInductionVar();
        if (domain->second.lower) {
          index = builder.create<arith::SubIOp>(loc, index, loop.getLowerBound());
        }
        if (domain->second.step != 1) {
          index = builder.create<arith::DivUIOp>(loc, index, loop.getStep());
        }
        return index;
      });
    };
    Value part;
    const auto atomIdentity = detail::atomKey(atom);
    auto prior = values.atoms.find(atomIdentity);
    if (prior != values.atoms.end() &&
        prior->second.getDefiningOp()->isBeforeInBlock(anchor)) {
      part = prior->second;
    } else if (atom.kind == ObservationAtom::LoopNonEmpty) {
      part = builder.create<arith::CmpIOp>(loc,
          atom.value ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sge,
          loop.getLowerBound(), loop.getUpperBound());
    } else if (atom.kind == ObservationAtom::LoopResidue) {
      lhs = reuse(values.residues, std::make_pair(atom.owner, atom.parameter), [&]() -> Value {
        return builder.create<arith::RemUIOp>(loc, ordinal(), constant(atom.parameter));
      });
      part = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs,
                                           constant(atom.value));
    } else if (atom.kind == ObservationAtom::LoopHasPrevious) {
      part = builder.create<arith::CmpIOp>(
          loc,
          atom.value ? arith::CmpIPredicate::sge : arith::CmpIPredicate::slt,
          ordinal(), constant(atom.parameter));
    } else if (atom.kind == ObservationAtom::LoopHasNext) {
      const auto distance = domain->second.distance(atom.parameter);
      if (!distance) {
        part = builder.create<arith::ConstantIntOp>(loc, atom.value ? 0 : 1, 1);
      } else {
        lhs = reuse(values.remaining, atom.owner, [&]() -> Value {
          return builder.create<arith::SubIOp>(loc, loop.getUpperBound(), loop.getInductionVar());
        });
        part = builder.create<arith::CmpIOp>(
            loc, atom.value ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::sle,
            lhs, constant(*distance));
      }
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
        (atom.kind != ObservationAtom::LoopNonEmpty && !owner->second->isProperAncestor(anchor)))
      return anchor->emitError(
          "handoff: emitted predicate lost its original available owner");
    auto loop = owner->second;
    const auto domain = input.loopDomains.find(atom.owner);
    if (domain == input.loopDomains.end()) {
      return anchor->emitError("handoff: emitted predicate has no original domain proof");
    }
    if (atom.kind == ObservationAtom::LoopNonEmpty &&
        (!input.dominance.dominates(loop.getLowerBound(), anchor) ||
         !input.dominance.dominates(loop.getUpperBound(), anchor))) {
      return anchor->emitError("handoff: emitted predicate uses unavailable original trip bounds");
    }
    auto isOrdinal = [&](Value value) {
      if (domain->second.step != 1) {
        auto divide = value.getDefiningOp<arith::DivUIOp>();
        if (!divide || divide.getRhs() != loop.getStep()) {
          return false;
        }
        value = divide.getLhs();
      }
      if (domain->second.lower) {
        auto subtract = value.getDefiningOp<arith::SubIOp>();
        if (!subtract || subtract.getRhs() != loop.getLowerBound()) {
          return false;
        }
        value = subtract.getLhs();
      }
      return value == loop.getInductionVar();
    };
    bool matched = false;
    for (std::size_t i = 0; i < components.size() && !matched; ++i) {
      if (used[i])
        continue;
      if (atom.kind == ObservationAtom::LoopHasNext && !domain->second.distance(atom.parameter)) {
        auto constant = components[i].getDefiningOp<arith::ConstantOp>();
        auto value = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
        matched = components[i].getType().isInteger(1) && value &&
            (atom.value ? value.getValue().isZero() : value.getValue().isOne());
        used[i] = matched;
        continue;
      }
      auto cmp = components[i].getDefiningOp<arith::CmpIOp>();
      if (!cmp)
        continue;
      if (atom.kind == ObservationAtom::LoopNonEmpty) {
        matched = cmp.getLhs() == loop.getLowerBound() && cmp.getRhs() == loop.getUpperBound() &&
            cmp.getPredicate() == (atom.value ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sge);
        used[i] = matched;
        continue;
      }
      const auto rhs = integer(cmp.getRhs());
      if (!rhs || *rhs < 0)
        continue;
      if (atom.kind == ObservationAtom::LoopResidue) {
        auto rem = cmp.getLhs().getDefiningOp<arith::RemUIOp>();
        matched =
            rem && cmp.getPredicate() == arith::CmpIPredicate::eq &&
            isOrdinal(rem.getLhs()) &&
            integer(rem.getRhs()) ==
                std::optional<int64_t>(static_cast<int64_t>(atom.parameter)) &&
            uint64_t(*rhs) == atom.value;
      } else if (atom.kind == ObservationAtom::LoopHasPrevious) {
        matched =
            isOrdinal(cmp.getLhs()) &&
            uint64_t(*rhs) == atom.parameter &&
            cmp.getPredicate() == (atom.value ? arith::CmpIPredicate::sge
                                              : arith::CmpIPredicate::slt);
      } else if (atom.kind == ObservationAtom::LoopHasNext) {
        auto remaining = cmp.getLhs().getDefiningOp<arith::SubIOp>();
        matched =
            remaining && remaining.getLhs() == loop.getUpperBound() &&
            remaining.getRhs() == loop.getInductionVar() &&
            uint64_t(*rhs) == *domain->second.distance(atom.parameter) &&
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

using NativeConstructor = llvm::function_ref<Result(const Program &, uint64_t)>;
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
  auto result = constructor(input.program, input.endpointDiscoveryWork);
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
    auto sameWord = [&](const auto &candidate) {
      const auto &other = result.commands[candidate.cuts.front()];
      return candidate.anchor == input.anchors[cut] && word.size() == other.size() &&
          std::equal(word.begin(), word.end(), other.begin(), [](const auto &a, const auto &b) {
            return a.kind == b.kind && a.source == b.source && a.observer == b.observer && a.key == b.key;
          });
    };
    auto found = llvm::find_if(words, sameWord);
    if (found == words.end()) {
      words.push_back({{}, input.anchors[cut], {}, {}});
      found = words.end() - 1;
    }
    found->cuts.push_back(cut);
    const auto observation = input.program.observed->sites[cut].observation;
    found->predicates.push_back(input.program.observed->observations[observation].atoms);
  }
  DenseMap<mlir::Operation *, std::size_t> originalOrder;
  for (auto [index, op] : llvm::enumerate(originalOperations)) originalOrder[op] = index;
  llvm::stable_sort(words, [&](const auto &a, const auto &b) {
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
                      SyncCodegen::CommandListPolicy::PreserveOrder);
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
  SyncSlotMapping::ConstantCache constants;
  SyncSlotMapping::RangeCache ranges;
  for (const auto &[id, operation] : originalLoopOwners) {
    auto loop = dyn_cast_or_null<scf::ForOp>(operation);
    if (!loop || !input.loopOwners.try_emplace(id, loop).second) {
      return failure();
    }
    auto domain = SyncSlotMapping::originalLoopDomain(loop, constants, ranges);
    if (!domain) {
      return failure();
    }
    input.loopDomains.try_emplace(id, *domain);
  }
  return checkObservationPredicate(observation, anchor, input, condition);
}

namespace {
LogicalResult executeSelectedAttempt(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate, SelectedPlan *report,
    ObservationPolicy policy, bool *firstUseMaterialized = nullptr) {
  if (report) {
    *report = SelectedPlan{};
  }
  if (firstUseMaterialized) {
    *firstUseMaterialized = false;
  }
  // Choose observations before construction from shared physical access roles.
  // Qualified normalized loops use original first/next-use guards; unrelated
  // loops retain their original SCF graph. Bounding geometry never becomes a
  // full-write certificate. Every attempt owns a fresh import and ledger.
  return executeTransaction(function,
      [report, firstUseMaterialized](const Program &program, uint64_t endpointDiscoveryWork) {
        if (firstUseMaterialized) {
          *firstUseMaterialized = program.observed &&
                                  program.observed->firstUsePrefixes != 0;
        }
        auto selected = constructSelectedPlan(program);
        selected.work.nativeEndpointDiscoveryWork = endpointDiscoveryWork;
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
      }, mutate, policy);
}

LogicalResult executeSelectedHandoffSync(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate, SelectedPlan *report) {
  // Declining optional observation materialization is an admission transaction
  // in the same constructor. It never selects a serializer or InsertSync.
  // Shared peer/queue contracts forbid this local-only retry: the local causal
  // checker cannot establish external progress after changing their ordering.
  bool externalProtocol = false;
  function.walk([&](mlir::Operation *op) {
    externalProtocol |= getSyncProtocolModel(op).has_value();
  });
  SelectedPlan candidate;
  bool firstUseMaterialized = false;
  std::string diagnostics;
  LogicalResult status = failure();
  {
    ScopedDiagnosticHandler capture(function.getContext(), [&](Diagnostic& diagnostic) {
      llvm::raw_string_ostream stream(diagnostics);
      diagnostic.print(stream);
      stream << '\n';
      return success();
    });
    status = executeSelectedAttempt(function, mutate, &candidate,
                                    ObservationPolicy::QualifiedAccessRoles,
                                    &firstUseMaterialized);
  }
  if (succeeded(status) || candidate.success || externalProtocol ||
      candidate.failure != SelectedFailure::EventResource) {
    if (!diagnostics.empty()) {
      llvm::errs() << diagnostics;
    }
    if (report) {
      *report = std::move(candidate);
    }
    return status;
  }
  // First-use prefix splitting is the last optional observation layer. When
  // full-qualified construction cannot realize a key, retain preceding
  // child/storage interfaces through a fresh fixed-policy admission attempt.
  std::optional<DeclinedRecurringAttempt> firstUseDecline;
  std::optional<DeclinedRecurringAttempt> priorDecline;
  if (firstUseMaterialized) {
    firstUseDecline = DeclinedRecurringAttempt{
        candidate.failure, candidate.reason, candidate.cut, candidate.work};
    SelectedPlan prior;
    {
      ScopedDiagnosticHandler capture(function.getContext(), [&](Diagnostic& diagnostic) {
        llvm::raw_string_ostream stream(diagnostics);
        diagnostic.print(stream);
        stream << '\n';
        return success();
      });
      status = executeSelectedAttempt(function, mutate, &prior,
          ObservationPolicy::QualifiedAccessRolesWithoutFirstUse);
    }
    prior.declinedFirstUse = firstUseDecline;
    const bool cannotRetryOriginal = succeeded(status) || prior.success ||
                                     prior.failure != SelectedFailure::EventResource;
    if (cannotRetryOriginal) {
      if (succeeded(status)) {
        function.emitRemark("handoff: declined optional first-use observation after ")
            << candidate.reason;
      } else if (!diagnostics.empty()) {
        llvm::errs() << diagnostics;
      }
      if (report) {
        *report = std::move(prior);
      }
      return status;
    }
    priorDecline = DeclinedRecurringAttempt{
        prior.failure, prior.reason, prior.cut, prior.work};
  }
  // The failed private clone is gone. Retain all immutable physical relations
  // while trying original command words with no speculative observation copies.
  SelectedPlan original;
  status = executeSelectedAttempt(function, mutate, &original, ObservationPolicy::OriginalControl);
  original.declinedFirstUse = firstUseDecline;
  original.declinedObservation = priorDecline.value_or(DeclinedRecurringAttempt{
      candidate.failure, candidate.reason, candidate.cut, candidate.work});
  if (succeeded(status)) {
    function.emitRemark("handoff: declined optional observation materialization after ") << candidate.reason;
  } else if (!diagnostics.empty()) {
    llvm::errs() << diagnostics;
  }
  if (report) {
    *report = std::move(original);
  }
  return status;
}

} // namespace

LogicalResult testing::runHandoffSyncWithMutation(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate) {
  return executeSelectedHandoffSync(function, mutate, nullptr);
}

LogicalResult testing::runSelectedHandoffSyncWithMutation(
    func::FuncOp function, llvm::function_ref<void(func::FuncOp)> mutate, SelectedPlan *report) {
  return executeSelectedHandoffSync(function, mutate, report);
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
  result.endpointDiscoveryWork = input.endpointDiscoveryWork;
  if (!result.analysis.complete)
    return function.emitError("handoff analysis: ") << result.analysis.reason;
  return success();
}
} // namespace

LogicalResult testing::analyzeSelectedHandoffSync(func::FuncOp function,
                                                  NativeAnalysis &result) {
  return analyzeHandoffSyncWithPolicy(function, result,
                                      ObservationPolicy::QualifiedAccessRoles);
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
