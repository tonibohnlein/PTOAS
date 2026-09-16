// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Plan.h"
#include "PTO/Transforms/OAHS/CausalFrontier.h"
#include "BundleQueries.h"
#include "PrefixQueries.h"
#include "StorageFrontierAnalysis.h"
#include "Transfer.h"
#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace mlir::pto::oahs {
namespace {
unsigned lane(Pipe p) { return unsigned(p); }
bool validateRegion(const Program &p, const Region &region,
                    std::vector<unsigned> &seen, std::string &reason) {
  if (region.kind < Region::Sequence || region.kind > Region::Operation) {
    reason = "invalid structured region kind";
    return false;
  }
  if (region.kind == Region::Operation) {
    if (region.operation >= p.operations.size()) {
      reason = "control region references an invalid physical phase";
      return false;
    }
    ++seen[region.operation];
    if (!region.children.empty()) {
      reason = "physical phase region cannot contain children";
      return false;
    }
    return true;
  }
  if (region.kind == Region::Choice && region.children.size() != 2) {
    reason = "choice region requires exactly two alternatives";
    return false;
  }
  if (region.kind == Region::For && region.children.size() != 1) {
    reason = "for region requires exactly one body";
    return false;
  }
  if (region.kind == Region::While && region.children.size() != 2) {
    reason = "while region requires before and after bodies";
    return false;
  }
  for (const Region &child : region.children)
    if (!validateRegion(p, child, seen, reason))
      return false;
  return true;
}
bool validObserved(const Program &p, std::string &reason) {
  if (!p.observed)
    return true;
  const auto &q = *p.observed;
  auto fail = [&](const char *text) {
    reason = text;
    return false;
  };
  if (q.qualification.empty() || q.sites.empty() || q.entry >= q.sites.size() ||
      q.exit >= q.sites.size())
    return fail("missing or malformed observed-control qualification");
  if (q.sites[q.exit].operation != NoControlId ||
      !q.sites[q.exit].successors.empty() ||
      q.sites[q.exit].observation == NoControlId)
    return fail(
        "observed invocation exit requires a legal payload-free terminal cut");
  std::set<std::vector<uint64_t>> unique;
  std::map<std::size_t,
           std::vector<std::tuple<unsigned, std::size_t, uint64_t>>>
      anchorVocabulary;
  std::set<std::size_t> usedObservations;
  for (const auto &site : q.sites)
    if (site.observation != NoControlId)
      usedObservations.insert(site.observation);
  for (std::size_t observationId = 0; observationId < q.observations.size();
       ++observationId) {
    const auto &o = q.observations[observationId];
    if (!o.available)
      return fail("endpoint observation is not available from original values");
    std::set<std::tuple<unsigned, std::size_t, uint64_t>> terms;
    std::vector<std::tuple<unsigned, std::size_t, uint64_t, uint64_t>>
        assignments;
    for (const auto &a : o.atoms) {
      if (a.kind < ObservationAtom::OriginalBoolean ||
          a.kind > ObservationAtom::LoopHasNext)
        return fail("unsupported non-original endpoint observation");
      if (a.kind == ObservationAtom::LoopResidue
              ? (!a.parameter || a.value >= a.parameter)
              : a.value > 1)
        return fail("invalid original-value observation atom");
      if ((a.kind == ObservationAtom::LoopHasPrevious ||
           a.kind == ObservationAtom::LoopHasNext) &&
          !a.parameter)
        return fail("zero-distance recurrence observation");
      if (!terms.emplace(unsigned(a.kind), a.owner, a.parameter).second)
        return fail("duplicate or contradictory observation atom");
      assignments.emplace_back(unsigned(a.kind), a.owner, a.parameter, a.value);
    }
    std::sort(assignments.begin(), assignments.end());
    std::vector<uint64_t> key{o.anchor};
    for (const auto &[kind, owner, parameter, value] : assignments)
      key.insert(key.end(), {kind, owner, parameter, value});
    if (!unique.insert(key).second)
      return fail("duplicate observation identity; share its word instead");
    if (usedObservations.count(observationId)) {
      std::vector<std::tuple<unsigned, std::size_t, uint64_t>> vocabulary(
          terms.begin(), terms.end());
      auto entry = anchorVocabulary.emplace(o.anchor, vocabulary);
      if (!entry.second && entry.first->second != vocabulary)
        return fail("one original anchor requires a common observable feature "
                    "vocabulary");
    }
  }
  for (std::size_t i = 0; i < q.scopes.size(); ++i)
    if (q.scopes[i].kind > AnalysisContext::WhileAfter ||
        (q.scopes[i].parent != NoControlId && q.scopes[i].parent >= i))
      return fail("invalid observed original scope");
  std::vector<unsigned> seen(p.operations.size());
  std::map<std::size_t, std::optional<Pipe>> roles;
  for (std::size_t i = 0; i < q.sites.size(); ++i) {
    const auto &n = q.sites[i];
    if (n.context >= (q.scopes.empty() ? 1 : q.scopes.size()))
      return fail("invalid observed context");
    if (n.operation != NoControlId) {
      if (n.operation >= p.operations.size())
        return fail("invalid observed physical phase");
      ++seen[n.operation];
      if (n.observation == NoControlId)
        return fail("payload needs its original before-issue cut");
    }
    if (n.observation != NoControlId) {
      if (n.observation >= q.observations.size())
        return fail("invalid observation index");
      const std::optional<Pipe> role =
          n.operation == NoControlId
              ? std::nullopt
              : std::optional<Pipe>(p.operations[n.operation].pipe);
      if (roles.count(n.observation) && roles[n.observation] != role)
        return fail("one observed word has incompatible payload roles");
      roles[n.observation] = role;
    }
    if (!n.backedgeOwners.empty() &&
        n.backedgeOwners.size() != n.successors.size())
      return fail("invalid observed edge labels");
    if (i != q.exit && n.successors.empty())
      return fail("observed terminal bypasses invocation exit");
    for (auto target : n.successors)
      if (target >= q.sites.size())
        return fail("invalid observed edge");
  }
  for (auto n : seen)
    if (!n)
      return fail("physical phase absent from observed control");
  return true;
}
bool valid(const Program &p, std::string &reason) {
  if (!detail::phaseModelValid(p, reason)) return false;
  if (p.target.contract.empty()) {
    reason = "missing target contract";
    return false;
  }
  for (const auto& cell : p.cells) {
      if (cell.storage != Cell::Storage::Abstract && cell.storage != Cell::Storage::CanonicalInterval &&
          cell.storage != Cell::Storage::OverlapWitness) {
          reason = "invalid storage identity kind";
          return false;
      }
      if (cell.storage == Cell::Storage::CanonicalInterval &&
          (cell.unknownRange || cell.coordinateSpace.empty() || cell.ranges.size() != 1 || !cell.ranges[0].second ||
           cell.ranges[0].second > std::numeric_limits<uint64_t>::max() - cell.ranges[0].first)) {
          reason = "unqualified canonical storage interval";
          return false;
      }
  }
  for (const auto &op : p.operations) {
    if (!op.complete || lane(op.pipe) >= PipeCount ||
        !p.target.supported[lane(op.pipe)]) {
      reason = "incomplete operation semantics or unsupported pipeline";
      return false;
    }
    for (const auto &a : op.accesses)
        if (a.cell >= p.cells.size() || (!a.read && !a.write) ||
            (a.definiteWrite && (!a.write || p.cells[a.cell].storage == Cell::Storage::OverlapWitness))) {
            reason = "invalid physical effect";
            return false;
        }
    for (const auto &r : op.resources)
      if (r.resource.empty() || (!r.acquire && !r.release) ||
          (r.timing != EffectTiming::Issue &&
           r.timing != EffectTiming::Completion)) {
        reason = "malformed typed resource effect";
        return false;
      }
    for (const auto &v : op.visibility)
      if (v.cell >= p.cells.size() || (!v.publish && !v.acquire) ||
          (v.timing != EffectTiming::Issue &&
           v.timing != EffectTiming::Completion)) {
        reason = "malformed visibility effect";
        return false;
      }
    for (const auto &population : {op.authoredEvents, op.internalTransfers})
      for (const auto &event : population)
        if (lane(event.source) >= PipeCount ||
            lane(event.observer) >= PipeCount ||
            event.source == event.observer) {
          reason = "malformed explicit event effect";
          return false;
        }
  }
  if (!validObserved(p, reason))
    return false;
  if (!p.observed &&
      (!p.body.children.empty() || p.body.kind != Region::Sequence)) {
    std::vector<unsigned> seen(p.operations.size());
    if (!validateRegion(p, p.body, seen, reason))
      return false;
    for (unsigned occurrences : seen)
      if (occurrences != 1) {
        reason =
            occurrences == 0
                ? "physical phase is absent from the control representation"
                : "physical phase occurs more than once in the control "
                  "representation";
        return false;
      }
  }
  return true;
}
bool available(const Target &t, Pipe source, Pipe observer, unsigned key) {
  if (lane(source) >= PipeCount || lane(observer) >= PipeCount ||
      source == observer || !t.supported[lane(source)] ||
      !t.supported[lane(observer)])
    return false;
  const auto &keys = t.keys[lane(source)][lane(observer)];
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

bool reserved(const Program &p, const Command &c) {
  return std::any_of(p.reservations.begin(), p.reservations.end(),
                     [&](const EventIdentity &r) {
                       return r.source == c.source &&
                              r.observer == c.observer && r.key == c.key;
                     });
}
bool supportedEffects(const Program &p, std::string &reason) {
  if (p.finalBlocks && p.target.barrierAll) {
    reason = "phase ALL adapter is not qualified; supply the supported named-prefix vocabulary";
    return false;
  }
  for (const auto &op : p.operations) {
    if (!op.resources.empty() || !op.visibility.empty() ||
        !op.authoredEvents.empty() || !op.internalTransfers.empty()) {
      reason = "typed resource, visibility, or internal/authored transfer "
               "contract is not implemented";
      return false;
    }
  }
  for (const auto &r : p.reservations)
    if (!available(p.target, r.source, r.observer, r.key)) {
      reason = "reservation names an unavailable target key";
      return false;
    }
  return true;
}
bool commandsValid(const Program &p, const Commands &commands,
                   std::string &reason) {
  if (commands.size() != commandCutCount(p)) {
    reason = "command cuts do not match original operations";
    return false;
  }
  std::vector<bool> liveObservation;
  if (p.observed) {
    liveObservation.assign(p.observed->observations.size(), false);
    const auto reached = detail::reachableSites(detail::buildControlGraph(p));
    for (Cut at = 0; at < commands.size(); ++at)
      if (reached[at] && legalCommandCut(p, at))
        liveObservation[p.observed->sites[at].observation] = true;
  }
  for (Cut at = 0; at < commands.size(); ++at) {
    if (!legalCommandCut(p, at) && !commands[at].empty()) {
      reason = "commands at an unavailable original cut";
      return false;
    }
    if (p.observed && !commands[at].empty() &&
        !liveObservation[p.observed->sites[at].observation]) {
      reason = "command word has no reachable original observation";
      return false;
    }
    const auto leader = canonicalCommandCut(p, at);
    if (leader != at) {
      if (commands[at].size() != commands[leader].size()) {
        reason = "nonuniform command word for one original observation";
        return false;
      }
      for (std::size_t i = 0; i < commands[at].size(); ++i) {
        auto a = commands[at][i], b = commands[leader][i];
        if (a.kind != b.kind || a.source != b.source ||
            a.observer != b.observer || a.key != b.key) {
          reason = "nonuniform command word for one original observation";
          return false;
        }
      }
    }
  }
  for (const auto &at : commands)
    for (const auto &c : at) {
      if (c.kind == Command::BarrierAll) {
        if (!p.target.barrierAll) {
          reason = "unsupported all-pipeline barrier";
          return false;
        }
      } else if (c.kind == Command::Barrier) {
        if (lane(c.source) >= PipeCount ||
            !p.target.supported[lane(c.source)] ||
            !p.target.barriers[lane(c.source)]) {
          reason = "unsupported pipeline barrier";
          return false;
        }
      } else if ((c.kind != Command::Publish && c.kind != Command::Acquire) ||
                 !available(p.target, c.source, c.observer, c.key) ||
                 reserved(p, c)) {
        reason = "unsupported or reserved event direction/key";
        return false;
      }
    }
  return true;
}
} // namespace

Result validateProgram(const Program &p) {
  Result result;
  result.success = valid(p, result.reason);
  return result;
}

FrontierCheck checkCausalFrontier(const Program& p, const Commands& commands)
{
    FrontierCheck out;
    const CausalFrontier frontier(p);
    if (!frontier.complete()) {
        out.failure = FrontierFailure::UnsupportedContract;
        out.reason = frontier.reason();
        return out;
    }
    if (!commandsValid(p, commands, out.reason)) {
        out.failure = FrontierFailure::InvalidInput;
        return out;
    }
    for (Cut cut = 0; cut < commands.size(); ++cut) {
        for (const auto& c : commands[cut]) {
            if (c.kind == Command::BarrierAll && cut != invocationExitCut(p)) {
                out.failure = FrontierFailure::UnsupportedContract;
                out.reason = "causal frontier ALL is confined to invocation retirement";
                return out;
            }
        }
    }
    out.complete = true;
    out.keys = frontier.keys();
    const auto graph = detail::buildControlGraph(p);
    std::vector<FrontierCut> states(graph.sites.size());
    states[graph.entry].incoming = frontier.initial();
    std::deque<std::size_t> queue{graph.entry};
    std::vector<bool> queued(graph.sites.size());
    queued[graph.entry] = true;
    auto fail = [&](const FrontierStep& step, Cut cut, std::size_t command) {
        out.failure = step.failure;
        out.reason = step.reason;
        out.residuals = step.residuals;
        out.cut = cut;
        out.command = command;
    };
    while (!queue.empty()) {
        const auto at = queue.front();
        queue.pop_front();
        queued[at] = false;
        ++out.siteEvaluations;
        auto state = states[at].incoming;
        if (at < commands.size())
            for (std::size_t i = 0; i < commands[at].size(); ++i) {
                auto next = frontier.command(state, commands[at][i], {at, i});
                if (!next.applied) {
                    fail(next, at, i);
                    return out;
                }
                state = std::move(next.state);
            }
        states[at].beforeIssue = state;
        if (graph.operations[at] != NoAnalysisId) {
            auto next = frontier.issue(state, graph.operations[at]);
            if (!next.applied) {
                fail(next, at, NoAnalysisId);
                return out;
            }
            state = std::move(next.state);
        }
        if (at == graph.exit) {
            auto next = frontier.exit(state);
            if (!next.applied) {
                fail(next, at, NoAnalysisId);
                return out;
            }
        }
        states[at].outgoing = state;
        for (const auto successor : graph.sites[at].successors) {
            auto merged = frontier.join(states[successor].incoming, state);
            if (!merged.applied) {
                fail(merged, successor, NoAnalysisId);
                return out;
            }
            if (merged.state == states[successor].incoming)
                continue;
            states[successor].incoming = std::move(merged.state);
            if (!queued[successor]) {
                queue.push_back(successor);
                queued[successor] = true;
            }
        }
    }
    out.accepted = true;
    out.cuts.resize(commands.size());
    for (Cut cut = 0; cut < commands.size(); ++cut)
        out.cuts[cut] = std::move(states[cut]);
    return out;
}

namespace {
bool prepareAnalysisProgram(const Program &p, AnalysisResult &out) {
  if (!valid(p, out.reason)) {
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
      const auto &op = p.operations[i];
      if (!op.complete)
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
                                   "missing operation completeness contract"});
      else if (lane(op.pipe) >= PipeCount || !p.target.supported[lane(op.pipe)])
        out.diagnostics.push_back(
            {AnalysisDiagnostic::UnsupportedSemantics, i,
             "operation pipeline outside target contract"});
    }
    if (out.diagnostics.empty())
      out.diagnostics.push_back(
          {AnalysisDiagnostic::InvalidInput, NoAnalysisId, out.reason});
    return false;
  }
  if (!supportedEffects(p, out.reason)) {
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
      const auto &op = p.operations[i];
      for (const auto &resource : op.resources)
        out.diagnostics.push_back(
            {AnalysisDiagnostic::UnsupportedSemantics, i,
             "resource transfer not implemented: " + resource.resource});
      for (const auto &visibility : op.visibility)
        out.diagnostics.push_back(
            {AnalysisDiagnostic::UnsupportedSemantics, i,
             "visibility transfer not implemented for cell " +
                 std::to_string(visibility.cell)});
      if (!op.authoredEvents.empty())
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
                                   "authored event contract not implemented"});
      if (!op.internalTransfers.empty())
        out.diagnostics.push_back(
            {AnalysisDiagnostic::UnsupportedSemantics, i,
             "internal phase transfer contract not implemented"});
    }
    if (out.diagnostics.empty())
      out.diagnostics.push_back(
          {AnalysisDiagnostic::UnsupportedSemantics, NoAnalysisId, out.reason});
    return false;
  }
  return true;
}
bool prepareAnalysisCommands(const Program &p, const Commands &commands,
                             AnalysisResult &out) {
  if (!commandsValid(p, commands, out.reason)) {
    out.diagnostics.push_back(
        {AnalysisDiagnostic::InvalidCommands, NoAnalysisId, out.reason});
    return false;
  }
  if (p.finalBlocks) for (const auto &word : commands) for (const auto &c : word)
    if (c.kind == Command::BarrierAll) {
      out.reason = "phase ALL/retirement adapter is not qualified";
      out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, NoAnalysisId, out.reason});
      return false;
    }
  return true;
}
bool prepareAnalysis(const Program &p, const Commands &commands, AnalysisResult &out) {
  return prepareAnalysisProgram(p, out) && prepareAnalysisCommands(p, commands, out);
}
} // namespace
AnalysisResult analyze(const Program &p, const Commands &commands,
                       AnalysisOptions options) {
  AnalysisResult out;
  if (!prepareAnalysis(p, commands, out)) return out;
  return detail::Transfer(p, commands).inspect(options);
}

struct ReplaySession::Impl {
  const Program program;
  const bool enabled;
  std::optional<detail::TransferCheckpoint> checkpoint;
  Commands previous;
  std::optional<AnalysisResult> previousReport;
  AnalysisOptions previousOptions;
  ReplayStats stats;
  AnalysisResult declarations;
  const bool validProgram;
  Impl(Program p, bool e) : program(std::move(p)), enabled(e),
      validProgram(prepareAnalysisProgram(program, declarations)) {}
};
ReplaySession::ReplaySession(Program p, bool enabled)
    : impl(std::make_unique<Impl>(std::move(p), enabled)) {}
ReplaySession::~ReplaySession() = default;
ReplaySession::ReplaySession(ReplaySession &&) noexcept = default;
ReplaySession &ReplaySession::operator=(ReplaySession &&) noexcept = default;
ReplayStats ReplaySession::lastReplay() const { return impl->stats; }
void ReplaySession::clear() {
  impl->checkpoint.reset(); impl->previousReport.reset();
  impl->previous.clear(); impl->stats = {};
}
AnalysisResult ReplaySession::analyze(const Commands &commands, AnalysisOptions options) {
  auto &self = *impl;
  self.stats = {};
  AnalysisResult out = self.declarations;
  // Program/target/observations are an owned immutable value. Their complete
  // declaration checks run once; actual candidate commands are always checked.
  if (!self.validProgram || !prepareAnalysisCommands(self.program, commands, out)) {
    self.stats.kind = ReplayStats::Invalid;
    return out;
  }
  if (self.enabled && self.previousReport &&
      self.previousOptions.captureStates == options.captureStates &&
      detail::identicalCommands(commands, self.previous)) {
    self.stats.kind = ReplayStats::Unchanged;
    self.stats.reusedSites = self.previousReport->stats.staticSites;
    out = *self.previousReport;
    // Counters describe this invocation, not the original cached computation.
    out.stats.siteEvaluations = out.stats.merges = out.stats.work = 0;
    out.stats.certificationPasses = 0;
    return out;
  }
  if (self.program.finalBlocks || !self.enabled) {
    self.stats.kind = self.enabled ? ReplayStats::PhaseFull : ReplayStats::Disabled;
    out = detail::Transfer(self.program, commands).inspect(options);
    self.checkpoint.reset();
  } else {
    detail::TransferCheckpoint next;
    out = detail::Transfer(self.program, commands).inspect(options,
             self.checkpoint ? &*self.checkpoint : nullptr, &next, &self.stats);
    if (out.complete) self.checkpoint = std::move(next);
  }
  if (out.complete && self.enabled) {
    self.previous = commands; self.previousReport = out;
    self.previousOptions = options;
  }
  return out;
}
AnalysisResult analyze(const Program &p) {
  return analyze(p, Commands(commandCutCount(p)));
}

bool validatePhaseContract(const Program &p, std::string &reason) {
  if (!p.finalBlocks) { reason = "no supplied phase contract"; return false; }
  return valid(p, reason) && supportedEffects(p, reason);
}
bool phaseFragment(const Program &p, std::size_t operation, PhaseFragment &out, std::string &reason) {
  if (!validatePhaseContract(p, reason)) return false;
  if (operation >= p.operations.size()) { reason = "invalid physical operation"; return false; }
  out = detail::buildPhaseFragment(p, operation); return true;
}
PhaseOrderResult checkPhaseOrder(const Program &p, const Commands &commands) {
  PhaseOrderResult out;
  if (!p.finalBlocks) { out.reason = "phase order query requires a supplied profile"; return out; }
  auto checked = analyze(p, commands, {false});
  if (!checked.complete) { out.reason = checked.reason; return out; }
  auto result = detail::phase::collect(p, commands, {false}, true);
  out.complete = result.analysis.complete; out.safe = result.analysis.verified();
  out.stats = result.analysis.stats; out.excess = std::move(result.excess);
  out.exact = out.safe && out.excess.empty(); out.reason = result.analysis.reason;
  if (out.safe && !out.exact) out.reason = "safe with extra original endpoint order";
  return out;
}
PhaseNativeQualification phaseNativeQualification() {
  return {false, {"release-pinned native operation/SDK lowering refinement",
                 "ordered same-role per-block service evidence",
                 "initial writable entry and native progress/queue contract",
                 "complete physical maps/layouts/private events and output visibility",
                 "native phase import/emission reconstruction and device tests"}};
}

Result verify(const Program &p, const Commands &commands) {
  const auto analysis = analyze(p, commands, {false});
  Result result;
  result.success = analysis.verified();
  result.reason = analysis.reason;
  return result;
}

BundleQuery::BundleQuery(Program p, Commands c, AnalysisOptions o, bool incremental)
    : impl(std::make_unique<Impl>(std::move(p), std::move(c), o, incremental)) {}
BundleQuery::~BundleQuery() = default;
BundleQuery::BundleQuery(BundleQuery &&) noexcept = default;
BundleQuery &BundleQuery::operator=(BundleQuery &&) noexcept = default;
const AnalysisResult &BundleQuery::analysis() const { return impl->before; }
BundleEvaluation BundleQuery::evaluate(Commands candidate) const {
  return impl->evaluate(std::move(candidate));
}

PrefixQuery::PrefixQuery(Program p, Commands c)
    : impl(std::make_unique<Impl>(std::move(p), std::move(c))) {}
PrefixQuery::PrefixQuery(Program p) {
  Commands empty(commandCutCount(p));
  impl = std::make_unique<Impl>(std::move(p), std::move(empty));
}
PrefixQuery::~PrefixQuery() = default;
PrefixQuery::PrefixQuery(PrefixQuery &&) noexcept = default;
PrefixQuery &PrefixQuery::operator=(PrefixQuery &&) noexcept = default;
const AnalysisResult &PrefixQuery::analysis() const { return impl->report; }
PrefixQueryStats PrefixQuery::statistics() const { return impl->stats; }
std::vector<CompletionRequirement>
PrefixQuery::consumerRequirements(Cut consumer) const {
  return impl->requirements(consumer);
}
BackwardCutResult PrefixQuery::backwardCuts(Cut consumer) const {
  return impl->backward(consumer);
}
ProspectivePrefix PrefixQuery::inspectPrefix(Pipe source, Cut publication,
                                             Cut consumer) const {
  return impl->inspect(source, publication, consumer, impl->backward(consumer));
}
PrefixCover PrefixQuery::coverByPrefixes(Cut consumer) const {
  std::vector<Cut> allowed;
  for (Cut c = 0; c < commandCutCount(impl->program); ++c)
    if (legalCommandCut(impl->program, c))
      allowed.push_back(c);
  return coverByPrefixes(consumer, allowed);
}
PrefixCover
PrefixQuery::coverByPrefixes(Cut consumer,
                             const std::vector<Cut> &allowed) const {
  PrefixCover out;
  out.consumer = consumer;
  out.backward = impl->backward(consumer);
  if (!out.backward.complete) {
    out.reason = out.backward.reason;
    return out;
  }
  out.requirements = impl->requirements(consumer);
  out.jointUncoveredOperations.assign(impl->program.operations.size(), 1);
  out.complete = true;
  if (out.requirements.empty())
    return out;
  for (const auto &cut : out.backward.cuts)
    if (std::find(allowed.begin(), allowed.end(), cut.cut) != allowed.end())
      for (unsigned source = 0; source < PipeCount; ++source) {
        if (!impl->program.target.supported[source])
          continue;
        if (Pipe(source) ==
                impl->program
                    .operations[operationAtCut(impl->program, consumer)]
                    .pipe &&
            cut.cut != consumer)
          continue;
        // A prospective remainder only GROWS between captures. If its seed
        // leaves every required class outstanding it cannot discharge a
        // component here.
        const auto seed = impl->sourceRemainder(Pipe(source), cut.cut);
        const bool useful =
            std::any_of(out.requirements.begin(), out.requirements.end(),
                        [&](const CompletionRequirement &r) {
                          return !seed[r.demand.producer];
                        });
        if (useful)
          out.candidates.push_back(
              impl->inspect(Pipe(source), cut.cut, consumer, out.backward));
      }
  std::vector<bool> remaining(out.requirements.size(), true);
  while (true) {
    std::size_t best = NoAnalysisId, bestGain = 0;
    for (std::size_t i = 0; i < out.candidates.size(); ++i) {
      const auto &candidate = out.candidates[i];
      if (!candidate.selectable())
        continue;
      std::size_t gain = 0;
      for (auto component : candidate.coveredRequirements)
        gain += remaining[component];
      // Candidates are generated in source-cut/lane order. Equal gain keeps the
      // earlier cut. This is a tie-break, not a structured order-dominance
      // proof.
      if (gain > bestGain) {
        best = i;
        bestGain = gain;
      }
    }
    if (!bestGain)
      break;
    out.selected.push_back(best);
    for (std::size_t op = 0; op < out.jointUncoveredOperations.size(); ++op)
      out.jointUncoveredOperations[op] &=
          out.candidates[best].uncoveredOperations[op];
    for (auto component : out.candidates[best].coveredRequirements)
      remaining[component] = false;
  }
  for (std::size_t i = 0; i < remaining.size(); ++i)
    if (remaining[i])
      out.remaining.push_back(i);
  if (!out.remaining.empty())
    out.reason =
        "no established prospective cover for every residual component";
  return out;
}

SuccessionSummary SuccessionSummary::sequence(const SuccessionSummary &a,
                                              const SuccessionSummary &b) {
  SuccessionSummary out = b;
  out.preservesIncoming = a.preservesIncoming && b.preservesIncoming;
  if (b.preservesIncoming) {
    out.writers = storage_detail::unite(out.writers, a.writers);
    out.readers = storage_detail::unite(out.readers, a.readers);
  }
  return out;
}
SuccessionSummary SuccessionSummary::choice(const SuccessionSummary &a,
                                            const SuccessionSummary &b) {
  return {a.preservesIncoming || b.preservesIncoming,
          storage_detail::unite(a.writers, b.writers),
          storage_detail::unite(a.readers, b.readers)};
}
SuccessionSummary SuccessionSummary::repeat(const SuccessionSummary &a) {
  auto out = a;
  out.preservesIncoming = true;
  return out;
}
std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
SuccessionSummary::apply(const std::vector<std::size_t> &w,
                         const std::vector<std::size_t> &r) const {
  return {preservesIncoming ? storage_detail::unite(writers, w) : writers,
          preservesIncoming ? storage_detail::unite(readers, r) : readers};
}
StorageFrontierAnalysis::StorageFrontierAnalysis(Program p)
    : impl(std::make_unique<Impl>(std::move(p))) {}
StorageFrontierAnalysis::~StorageFrontierAnalysis() = default;
StorageFrontierAnalysis::StorageFrontierAnalysis(
    StorageFrontierAnalysis &&) noexcept = default;
StorageFrontierAnalysis &StorageFrontierAnalysis::operator=(
    StorageFrontierAnalysis &&) noexcept = default;
bool StorageFrontierAnalysis::complete() const { return impl->ok; }
const std::string &StorageFrontierAnalysis::reason() const {
  return impl->error;
}
const StorageFrontierStats &StorageFrontierAnalysis::stats() const {
  return impl->statistics;
}
std::vector<std::size_t>
StorageFrontierAnalysis::sitesForOperation(std::size_t op) const {
  std::vector<std::size_t> out;
  if (impl->ok)
    for (std::size_t s = 0; s < impl->graph.sites.size(); ++s)
      if (impl->operationAt(s) == op && impl->reachable[s])
        out.push_back(s);
  return out;
}
std::vector<StorageOrigin>
StorageFrontierAnalysis::previousWriters(std::size_t s, unsigned c) const {
  return impl->query(s, c, 0);
}
std::vector<StorageOrigin>
StorageFrontierAnalysis::previousReaders(std::size_t s, unsigned c) const {
  return impl->query(s, c, 1);
}
std::vector<StorageOrigin>
StorageFrontierAnalysis::nextWriters(std::size_t s, unsigned c) const {
  return impl->query(s, c, 2);
}
std::vector<StorageOrigin>
StorageFrontierAnalysis::nextReaders(std::size_t s, unsigned c) const {
  return impl->query(s, c, 3);
}
StoragePath StorageFrontierAnalysis::witness(std::size_t s, std::size_t t,
                                             unsigned c) const {
  return impl->path(s, t, c);
}
StorageLifecycle StorageFrontierAnalysis::lifecycleAt(std::size_t s, unsigned c) const { return impl->lifecycle(s, c); }
RequirementProvenance StorageFrontierAnalysis::describeRequirement(const StorageRelationship& r) const
{
    return impl->describe(r);
}
std::vector<StorageRelationship>
StorageFrontierAnalysis::relationshipsAt(std::size_t s) const {
  std::vector<StorageRelationship> out;
  if (!impl->ok || s >= impl->reachable.size() || !impl->reachable[s])
    return out;
  for (unsigned c = 0; c < impl->cells.size(); ++c) {
    const auto &a = impl->cells[c].accessAt[s];
    if (!a.read && !a.write)
      continue;
    auto add = [&](const StorageOrigin &o, StorageRelationship::Kind k) {
      out.push_back({k, c, o, impl->origin(s)});
    };
    for (const auto &o : impl->query(s, c, 0)) {
      if (a.read)
        add(o, StorageRelationship::RAW);
      if (a.write)
        add(o, StorageRelationship::WAW);
    }
    if (a.write)
      for (const auto &o : impl->query(s, c, 1))
        add(o, StorageRelationship::WAR);
  }
  return out;
}
ReaderBoundary StorageFrontierAnalysis::readerBoundary(std::size_t s,
                                                       unsigned c, Pipe pipe,
                                                       bool backward) const {
  ReaderBoundary out;
  if (!impl->ok || s >= impl->reachable.size() || c >= impl->cells.size() ||
      !impl->reachable[s])
    return out;
  const auto &g = impl->graph;
  std::vector<bool> seen(g.sites.size());
  std::vector<std::size_t> todo =
      backward ? impl->predecessors[s] : g.sites[s].successors;
  if (todo.empty())
    out.boundary = true;
  while (!todo.empty()) {
    const auto v = todo.back();
    todo.pop_back();
    if (seen[v] || !impl->reachable[v])
      continue;
    seen[v] = true;
    if ((backward && v == g.entry) ||
        (!backward && g.sites[v].successors.empty()))
      out.boundary = true;
    const auto &a = impl->cells[c].accessAt[v];
    if (a.write) {
      out.boundary = true;
      if (a.definiteWrite)
        continue;
    }
    const auto op = impl->operationAt(v);
    if (a.read && !a.write && op != NoAnalysisId &&
        impl->program.operations[op].pipe == pipe) {
      out.readers.push_back(impl->origin(v));
      continue;
    }
    const auto &next = backward ? impl->predecessors[v] : g.sites[v].successors;
    todo.insert(todo.end(), next.begin(), next.end());
  }
  std::sort(out.readers.begin(), out.readers.end(),
            [](const StorageOrigin &a, const StorageOrigin &b) {
              return a.site < b.site;
            });
  return out;
}
std::vector<Cut> StorageFrontierAnalysis::corridor(std::size_t s, Pipe pipe,
                                                   bool backward,
                                                   bool nearestOnly) const {
  std::vector<Cut> out;
  if (!impl->ok || s >= impl->reachable.size() || !impl->reachable[s])
    return out;
  const auto &g = impl->graph;
  std::vector<bool> seen(g.sites.size());
  std::vector<std::size_t> todo =
      backward ? std::vector<std::size_t>{s} : g.sites[s].successors;
  while (!todo.empty()) {
    const auto v = todo.back();
    todo.pop_back();
    if (seen[v] || !impl->reachable[v])
      continue;
    seen[v] = true;
    const auto op = impl->operationAt(v);
    if (backward && v != s && op != NoAnalysisId &&
        impl->program.operations[op].pipe == pipe)
      continue;
    const auto c = impl->cutAt(v);
    if (c != NoAnalysisId) {
      out.push_back(canonicalCommandCut(impl->program, c));
      if (nearestOnly)
        continue;
    }
    if (!backward && op != NoAnalysisId &&
        impl->program.operations[op].pipe == pipe)
      continue;
    const auto &next = backward ? impl->predecessors[v] : g.sites[v].successors;
    todo.insert(todo.end(), next.begin(), next.end());
  }
  std::sort(out.begin(), out.end(),
            [&](Cut a, Cut b) { return g.cutRanks[a] < g.cutRanks[b]; });
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}
} // namespace mlir::pto::oahs

#include "ObservedFrontend.h"
