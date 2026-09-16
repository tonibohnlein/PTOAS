// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Plan.h"
#include "BundleQueries.h"
#include "PrefixQueries.h"
#include "StorageFrontierAnalysis.h"
#include "Transfer.h"
#include <algorithm>
#include <chrono>
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
  for (const auto &op : p.operations) {
    if (!op.complete || lane(op.pipe) >= PipeCount ||
        !p.target.supported[lane(op.pipe)]) {
      reason = "incomplete operation semantics or unsupported pipeline";
      return false;
    }
    for (const auto &a : op.accesses)
      if (a.cell >= p.cells.size() || (!a.read && !a.write) ||
          (a.definiteWrite && !a.write)) {
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
// Empty structural nodes do not force a different semantic construction path.
bool flatOrder(const Region &region, std::vector<std::size_t> &order) {
  if (region.kind == Region::Operation) {
    order.push_back(region.operation);
    return true;
  }
  if (region.kind == Region::Sequence) {
    for (const auto &child : region.children)
      if (!flatOrder(child, order))
        return false;
    return true;
  }
  // Ignore an entirely payload-free subtree, but never flatten real choices or
  // loops.
  std::vector<std::size_t> contents;
  for (const auto &child : region.children)
    if (!flatOrder(child, contents))
      return false;
  return contents.empty();
}
bool flat(const Program &p) {
  if (p.observed)
    return false;
  if (p.body.kind == Region::Sequence && p.body.children.empty())
    return true;
  std::vector<std::size_t> order;
  if (!flatOrder(p.body, order) || order.size() != p.operations.size())
    return false;
  for (std::size_t i = 0; i < order.size(); ++i)
    if (order[i] != i)
      return false;
  return true;
}
Result conservative(const Program &p, bool scarcity = false) {
  Result result;
  if (!p.target.barrierAll) {
    result.reason = "no qualified conservative completion mechanism";
    return result;
  }
  result.commands.resize(commandCutCount(p));
  const auto reached = detail::reachableSites(detail::buildControlGraph(p));
  std::set<Cut> used;
  for (Cut cut = 0; cut < commandCutCount(p); ++cut) {
    if (!reached[cut] || operationAtCut(p, cut) == NoControlId)
      continue;
    const auto word = canonicalCommandCut(p, cut);
    if (used.insert(word).second) {
      result.commands[word].push_back({Command::BarrierAll});
      ++result.conservativeBarriers;
    }
  }
  for (Cut cut = 0; cut < commandCutCount(p); ++cut) {
    const auto word = canonicalCommandCut(p, cut);
    if (word != cut)
      result.commands[cut] = result.commands[word];
  }
  if (p.invocation.retirement ==
      Program::InvocationContract::DrainAllAtReturn) {
    result.commands[invocationExitCut(p)].push_back({Command::BarrierAll});
    ++result.conservativeBarriers;
  }
  if (scarcity)
    result.scarcityBarriers = result.conservativeBarriers;
  auto checked = verify(p, result.commands);
  result.success = checked.success;
  result.reason = checked.reason;
  return result;
}
// A leg may have several mutually exclusive original publication sites.
// Matching is established by replay of the WHOLE packet population. Individual
// PrefixQuery pairs remain strict and are not weakened for these alternatives.
struct Packet {
  Handoff handoff;
  unsigned key = 0;
  std::optional<unsigned> acknowledgment;
  std::vector<Cut> publications; // empty means the one handoff.publication
  std::vector<Cut> acquisitions; // observed alternatives, same dynamic channel
};
struct PacketBundle {
  std::vector<Packet> legs; // ordered at their common acquisition cut
};
std::vector<Cut> publicationCuts(const Packet &packet) {
  return packet.publications.empty()
             ? std::vector<Cut>{packet.handoff.publication}
             : packet.publications;
}
std::vector<Cut> acquisitionCuts(const Packet &packet) {
  return packet.acquisitions.empty()
             ? std::vector<Cut>{packet.handoff.acquisition}
             : packet.acquisitions;
}
bool relocate(Packet &packet) {
  const auto cuts = publicationCuts(packet);
  const auto targets = acquisitionCuts(packet);
  if (cuts == targets)
    return false;
  packet.publications = targets;
  packet.handoff.publication = packet.handoff.acquisition;
  return true;
}
Commands render(const Program &p, const Commands &barriers,
                const std::vector<PacketBundle> &bundles) {
  Commands result = barriers;
  auto canonical = [&](const std::vector<Cut> &cuts) {
    std::set<Cut> out;
    for (auto c : cuts)
      out.insert(canonicalCommandCut(p, c));
    return out;
  };
  for (const auto &bundle : bundles)
    for (const auto &packet : bundle.legs) {
      const auto &h = packet.handoff;
      auto targets = canonical(acquisitionCuts(packet));
      for (auto c : canonical(publicationCuts(packet)))
        if (!targets.count(c))
          result[c].insert(result[c].begin(), {Command::Publish, h.source,
                                               h.observer, packet.key});
    }
  for (const auto &bundle : bundles)
    for (const auto &packet : bundle.legs) {
      const auto &h = packet.handoff;
      auto sources = canonical(publicationCuts(packet));
      for (auto c : canonical(acquisitionCuts(packet))) {
        auto &at = result[c];
        if (sources.count(c))
          at.push_back({Command::Publish, h.source, h.observer, packet.key});
        at.push_back({Command::Acquire, h.source, h.observer, packet.key});
        if (packet.acknowledgment) {
          at.push_back(
              {Command::Publish, h.observer, h.source, *packet.acknowledgment});
          at.push_back(
              {Command::Acquire, h.observer, h.source, *packet.acknowledgment});
        }
      }
    }
  if (p.invocation.retirement == Program::InvocationContract::DrainAllAtReturn)
    result[invocationExitCut(p)].push_back({Command::BarrierAll});
  if (p.observed)
    for (Cut c = 0; c < result.size(); ++c) {
      const auto leader = canonicalCommandCut(p, c);
      if (leader != c)
        result[c] = result[leader];
    }
  return result;
}
std::vector<unsigned> eligibleKeys(const Program &p, Pipe source,
                                   Pipe observer) {
  std::vector<unsigned> out;
  if (source == observer || !p.target.supported[lane(source)] ||
      !p.target.supported[lane(observer)])
    return out;
  for (unsigned k : p.target.keys[lane(source)][lane(observer)])
    if (!reserved(p, {Command::Publish, source, observer, k}) &&
        std::find(out.begin(), out.end(), k) == out.end())
      out.push_back(k);
  return out;
}
std::optional<unsigned> chooseKey(const Program &p,
                                  const std::vector<PacketBundle> &bundles,
                                  Pipe source, Pipe observer) {
  std::optional<unsigned> reused;
  for (unsigned key : eligibleKeys(p, source, observer)) {
    if (!reused)
      reused = key;
    bool used = false;
    for (const auto &bundle : bundles)
      for (const auto &packet : bundle.legs) {
        const auto &h = packet.handoff;
        used |=
            h.source == source && h.observer == observer && packet.key == key;
        used |= packet.acknowledgment && h.observer == source &&
                h.source == observer && *packet.acknowledgment == key;
      }
    if (!used)
      return key;
  }
  return reused; // proposal only; never a fabricated consumption edge
}
// Retain all shortest topology routes, including lanes with no payload effects.
// This is a finite placement policy, not completeness over all possible routes.
std::vector<std::vector<Pipe>> routes(const Program &p, Pipe source,
                                      Pipe observer) {
  std::array<unsigned, PipeCount> distance;
  distance.fill(PipeCount);
  distance[lane(source)] = 0;
  std::vector<Pipe> queue{source};
  for (std::size_t i = 0; i < queue.size(); ++i)
    for (unsigned b = 0; b < PipeCount; ++b)
      if (!eligibleKeys(p, queue[i], Pipe(b)).empty() &&
          distance[b] == PipeCount) {
        distance[b] = distance[lane(queue[i])] + 1;
        queue.push_back(Pipe(b));
      }
  std::vector<std::vector<Pipe>> out;
  if (distance[lane(observer)] == PipeCount)
    return out;
  std::vector<std::vector<Pipe>> work{{source}};
  while (!work.empty()) {
    auto path = std::move(work.back());
    work.pop_back();
    const auto a = path.back();
    if (a == observer) {
      out.push_back(std::move(path));
      continue;
    }
    for (unsigned b = PipeCount; b-- > 0;)
      if (distance[b] == distance[lane(a)] + 1 &&
          distance[b] <= distance[lane(observer)] &&
          !eligibleKeys(p, a, Pipe(b)).empty()) {
        auto next = path;
        next.push_back(Pipe(b));
        work.push_back(std::move(next));
      }
  }
  return out;
}
// A general backward cut-frontier proposal: the immediately preceding physical
// cuts along every predecessor arm. No matching or credit is assumed. Replay
// may reject it (e.g. a branch's last write needs a not-yet-supported exit
// cut).
std::vector<Cut> predecessorFrontier(const Program &p, Cut consumer) {
  const auto g = detail::buildControlGraph(p);
  std::vector<std::vector<std::pair<std::size_t, bool>>> pred(g.sites.size());
  for (std::size_t a = 0; a < g.sites.size(); ++a)
    for (std::size_t k = 0; k < g.sites[a].successors.size(); ++k)
      pred[g.sites[a].successors[k]].push_back(
          {a, g.sites[a].backedgeOwners[k] != NoAnalysisId});
  std::vector<std::size_t> work{consumer};
  std::vector<bool> seen(g.sites.size());
  std::set<Cut> out;
  while (!work.empty()) {
    auto at = work.back();
    work.pop_back();
    if (seen[at])
      continue;
    seen[at] = true;
    if (at == g.entry)
      return {};
    for (auto [previous, backedge] : pred[at]) {
      if (backedge)
        return {}; // occurrence-dependent frontiers are M4, not guessed
      if (previous < g.legalCuts.size() && g.legalCuts[previous] &&
          previous != consumer)
        out.insert(previous);
      else if (previous == consumer)
        return {};
      else
        work.push_back(previous);
    }
  }
  if (out.size() < 2)
    return {};
  return {out.begin(), out.end()};
}
} // namespace

Result validateProgram(const Program &p) {
  Result result;
  result.success = valid(p, result.reason);
  return result;
}

AnalysisResult analyze(const Program &p, const Commands &commands,
                       AnalysisOptions options) {
  AnalysisResult out;
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
    return out;
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
    return out;
  }
  if (!commandsValid(p, commands, out.reason)) {
    out.diagnostics.push_back(
        {AnalysisDiagnostic::InvalidCommands, NoAnalysisId, out.reason});
    return out;
  }
  if (p.finalBlocks) for (const auto &word : commands) for (const auto &c : word)
    if (c.kind == Command::BarrierAll) {
      out.reason = "phase ALL/retirement adapter is not qualified";
      out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, NoAnalysisId, out.reason});
      return out;
    }
  return detail::Transfer(p, commands).inspect(options);
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

BundleQuery::BundleQuery(Program p, Commands c, AnalysisOptions o)
    : impl(std::make_unique<Impl>(std::move(p), std::move(c), o)) {}
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

namespace {
Result constructAttempt(const Program &p,
                        const StorageFrontierAnalysis *storage,
                        CandidateStage stage) {
  Result result;
  // These counters describe cumulative construction work, not the packets in
  // the final plan. Keep failed-search cost visible when switching to ALL.
  auto fallback = [&]() {
    Result checked;
    if (stage == CandidateStage::Original)
      checked = conservative(p, true);
    else
      checked.reason = "restricted candidate stage requires widening";
    checked.bundleTrials += result.bundleTrials;
    checked.bundleSelections += result.bundleSelections;
    checked.protocolRepairs += result.protocolRepairs;
    return checked;
  };
  if (!valid(p, result.reason) || !supportedEffects(p, result.reason))
    return result;
  Commands barriers(commandCutCount(p));
  std::vector<PacketBundle> packets;
  const bool oneVisit = flat(p);
  const auto lexicalRanks = detail::buildControlGraph(p).cutRanks;
  // Each (consumer, source) has one finite repair slot: absent -> packet or
  // barrier. A bundle has at most PipeCount-1 shortest-route legs. Each leg
  // can relocate once and gain one reply. Keys are fixed at creation.
  // An unchanged/unsupported repair terminates below;
  // no numerical attempt limit or repeated identical candidate is necessary.
  std::map<std::pair<Cut, Pipe>, std::size_t> packetSlots;
  std::set<std::pair<Cut, Pipe>> barrierSlots;
  while (true) {
    Commands actual = render(p, barriers, packets);
    // Construction may temporarily speculate about protocol preconditions.
    // This is private proposal information, never AnalysisResult completion.
    // The public certified analysis is mandatory before final acceptance.
    auto issue = detail::Transfer(p, actual).run(true, false);
    if (issue.kind == detail::Failure::Hazard) {
      const auto consumer = canonicalCommandCut(p, issue.cut);
      const auto consumerOperation = operationAtCut(p, issue.cut);
      const Pipe observer = p.operations[consumerOperation].pipe;
      // Prefer incoming acquired completion before same-lane fences. A real
      // return can cover a same-lane conflict without a separate barrier.
      auto chosen = issue.missing.front();
      for (const auto &d : issue.missing)
        if (p.operations[chosen.producer].pipe == observer &&
            p.operations[d.producer].pipe != observer) {
          chosen = d;
          break;
        }
      const Pipe source = p.operations[chosen.producer].pipe;
      for (const auto &d : issue.missing)
        if (p.operations[d.producer].pipe == source &&
            d.producer > chosen.producer)
          chosen = d;
      result.demands.insert(result.demands.end(), issue.missing.begin(),
                            issue.missing.end());
      const auto slot = std::make_pair(consumer, source);
      if (auto found = packetSlots.find(slot); found != packetSlots.end()) {
        auto &packet = packets[found->second].legs.front();
        // A later discovered generation may not belong to the saved early
        // prefix. Relocate this one packet; never append duplicates forever.
        if (relocate(packet)) {
        } else
          return fallback();
      } else if (source == observer && p.target.barriers[lane(source)]) {
        if (!barrierSlots.insert(slot).second)
          return fallback();
        barriers[consumer].push_back({Command::Barrier, source});
      } else if (source != observer) {
        PrefixQuery query(p, actual);
        std::vector<Cut> allowed;
        if (stage == CandidateStage::Original || !storage) {
          for (Cut c = 0; c < commandCutCount(p); ++c)
            if (legalCommandCut(p, c))
              allowed.push_back(c);
        } else {
          for (auto site : storage->sitesForOperation(consumerOperation))
            for (const auto &relation : storage->relationshipsAt(site)) {
              const auto pipe = p.operations[relation.source.operation].pipe;
              auto cuts = storage->corridor(relation.source.site, pipe, false,
                                            stage == CandidateStage::Nearest);
              allowed.insert(allowed.end(), cuts.begin(), cuts.end());
            }
          std::sort(allowed.begin(), allowed.end());
          allowed.erase(std::unique(allowed.begin(), allowed.end()),
                        allowed.end());
        }
        const auto cover = query.coverByPrefixes(consumer, allowed);
        BundleQuery replay(p, actual);
        const auto frontier = predecessorFrontier(p, consumer);
        struct Proposal {
          Pipe publisher;
          Cut cut;
          std::vector<Cut> alternatives;
          std::vector<Cut> targets = {};
        };
        std::vector<Proposal> proposals;
        for (const auto &candidate : cover.candidates) {
          if (!candidate.selectable() || candidate.source == observer)
            continue;
          bool suppliesSource = std::all_of(
              issue.missing.begin(), issue.missing.end(), [&](const Demand &d) {
                return p.operations[d.producer].pipe != source ||
                       !candidate.uncoveredOperations[d.producer];
              });
          if (suppliesSource)
            proposals.push_back({candidate.source, candidate.publication, {}});
        }
        // A whole alternative-publication bundle can be valid even though none
        // of its individual source/target pairs passes the M2 balance monitor.
        const auto retained = [&](Cut c) {
          return std::find(allowed.begin(), allowed.end(), c) != allowed.end();
        };
        if (!frontier.empty() &&
            std::all_of(frontier.begin(), frontier.end(), retained))
          proposals.insert(proposals.begin(),
                           {source, frontier.front(), frontier});
        if (retained(consumer))
          proposals.push_back(
              {source, consumer, {}}); // ordinary co-located packet
        if (p.observed && storage) {
          // Infer channel frontiers from ORIGINAL reference succession. These
          // are proposals only; whole-word replay establishes participation.
          std::set<Cut> sourceWords, targetWords;
          std::set<Cut> earlyTargets;
          bool firstTarget = true;
          for (auto target : storage->sitesForOperation(consumerOperation)) {
            bool needed = false;
            for (const auto &d : replay.analysis().residuals)
              needed |= d.consumerCut == target &&
                        p.operations[d.demand.producer].pipe == source;
            if (!needed)
              continue;
            targetWords.insert(canonicalCommandCut(p, target));
            auto targetCorridor =
                storage->corridor(target, observer, true, false);
            std::set<Cut> current;
            for (auto c : targetCorridor)
              current.insert(canonicalCommandCut(p, c));
            if (firstTarget)
              earlyTargets = current;
            else {
              std::set<Cut> common;
              std::set_intersection(earlyTargets.begin(), earlyTargets.end(),
                                    current.begin(), current.end(),
                                    std::inserter(common, common.end()));
              earlyTargets = std::move(common);
            }
            firstTarget = false;
            for (const auto &relation : storage->relationshipsAt(target))
              if (p.operations[relation.source.operation].pipe == source) {
                auto candidates = storage->corridor(relation.source.site,
                                                    source, false, true);
                for (auto c : candidates)
                  if (retained(c))
                    sourceWords.insert(canonicalCommandCut(p, c));
              }
          }
          if (!sourceWords.empty() && !targetWords.empty()) {
            std::vector<Cut> sources(sourceWords.begin(), sourceWords.end());
            std::vector<Cut> targets(targetWords.begin(), targetWords.end());
            proposals.insert(proposals.begin(),
                             {source, sources.front(), sources, targets});
            // A legal dominating acquisition can enclose an arbitrary read-only
            // recurrence. No dependence or episode fact itself grants credit.
            if (stage != CandidateStage::Nearest)
              for (auto t : earlyTargets)
                proposals.insert(proposals.begin(),
                                 {source, sources.front(), sources, {t}});
            if (stage != CandidateStage::Nearest) {
              // Search finite source/target corridors together. In a nested
              // recurrence neither isolated endpoint need match the old target.
              // These candidates receive no prefix credit until whole replay.
              for (auto t : earlyTargets)
                for (auto c : allowed) {
                  if (!legalCommandCut(p, c))
                    continue;
                  proposals.insert(
                      proposals.begin(),
                      {source, canonicalCommandCut(p, c), {}, {t}});
                }
            }
          }
        }
        std::optional<PacketBundle> best;
        std::size_t bestCredit = 0, bestCollateral = 0, bestCost = 0;
        // Actual all-source replay is authoritative. Provisional memory
        // progress is only a construction step when recurrence certificates
        // remain open; it is never exported as BundleEvaluation::discharged.
        std::optional<PacketBundle> pending, groupedPending;
        std::size_t groupedTargets = 0;
        for (const auto &proposal : proposals) {
          for (const auto &path : routes(p, proposal.publisher, observer)) {
            if ((stage == CandidateStage::Nearest ||
                 stage == CandidateStage::Corridors) &&
                path.size() != 2)
              continue;
            PacketBundle bundle;
            auto population = packets;
            for (std::size_t j = 0; j + 1 < path.size(); ++j) {
              auto key = chooseKey(p, population, path[j], path[j + 1]);
              if (!key) {
                bundle.legs.clear();
                break;
              }
              Packet leg{
                  {path[j], path[j + 1], j ? consumer : proposal.cut, consumer},
                  *key,
                  {},
                  j ? std::vector<Cut>{} : proposal.alternatives,
                  {}};
              if (!proposal.targets.empty()) {
                leg.acquisitions = proposal.targets;
                leg.handoff.acquisition = proposal.targets.front();
                if (j) {
                  leg.publications = proposal.targets;
                  leg.handoff.publication = proposal.targets.front();
                }
              }
              bundle.legs.push_back(leg);
              // Include helper keys immediately in subsequent resource queries.
              population.push_back({{leg}});
            }
            if (bundle.legs.empty())
              continue;
            auto trial = packets;
            trial.push_back(bundle);
            auto commands = render(p, barriers, trial);
            const auto speculative =
                detail::Transfer(p, commands).run(true, false);
            bool advances = speculative.kind == detail::Failure::None;
            if (speculative.kind == detail::Failure::Hazard) {
              if (canonicalCommandCut(p, speculative.cut) == consumer) {
                advances = speculative.missing.size() < issue.missing.size();
                // The motivating source must really be addressed, not just
                // another source whose component happened to be counted first.
                for (const auto &d : speculative.missing)
                  if (p.operations[d.producer].pipe == source)
                    advances = false;
              } else
                advances =
                    lexicalRanks[speculative.cut] > lexicalRanks[consumer];
            }
            if (!advances && !p.observed)
              continue;
            if (p.observed && advances &&
                proposal.targets.size() > groupedTargets &&
                proposal.targets.size() > 1 &&
                detail::Transfer(p, commands).run(false, true, true).kind ==
                    detail::Failure::None) {
              // Matching is established; rearming can depend on not-yet-built
              // genuine storage returns. Keep this as a finite pending repair,
              // NOT certified BundleEvaluation credit.
              groupedPending = bundle;
              groupedTargets = proposal.targets.size();
            }
            auto evaluated = replay.evaluate(std::move(commands));
            ++result.bundleTrials;
            if (!evaluated.complete || !evaluated.introduced.empty())
              continue;
            std::size_t credit = 0;
            for (const auto &d : evaluated.discharged)
              credit += canonicalCommandCut(p, d.consumerCut) == consumer;
            const auto cost = evaluated.resources.publications +
                              evaluated.resources.acquisitions;
            std::size_t collateral = 0;
            const auto &before = replay.analysis().cuts[consumer].beforeIssue;
            const auto &after = evaluated.analysis.cuts[consumer].beforeIssue;
            if (before && after)
              for (std::size_t a = 0; a < p.operations.size(); ++a) {
                const bool needed = std::any_of(
                    issue.missing.begin(), issue.missing.end(),
                    [&](const Demand &d) { return d.producer == a; });
                collateral += !needed && before->pending[lane(observer)][a] &&
                              !after->pending[lane(observer)][a];
              }
            // Distinct goals: all-source coverage, collateral completion at
            // this consumer, then static commands. This is an explicit local
            // heuristic, NOT a whole-program order-dominance or latency
            // certificate.
            if (evaluated.analysis.protocol.empty() && credit) {
              if (!best || credit > bestCredit ||
                  (credit == bestCredit &&
                   std::make_pair(collateral, cost) <
                       std::make_pair(bestCollateral, bestCost))) {
                best = bundle;
                bestCredit = credit;
                bestCollateral = collateral;
                bestCost = cost;
              }
            } else if (!pending && advances &&
                       (proposal.alternatives.empty() || p.observed))
              pending = bundle;
          }
        }
        if (groupedPending)
          best = std::move(groupedPending);
        if (!best)
          best = std::move(pending);
        if (best) {
          packetSlots.emplace(slot, packets.size());
          packets.push_back(std::move(*best));
          ++result.bundleSelections;
        } else if (p.target.barrierAll && stage == CandidateStage::Original) {
          if (!barrierSlots.insert(slot).second)
            return fallback();
          barriers[consumer].push_back({Command::BarrierAll});
        } else {
          result.reason = "construction policy found no verified completion "
                          "bundle; not a target infeasibility proof";
          return result;
        }
      } else if (p.target.barrierAll && stage == CandidateStage::Original) {
        if (!barrierSlots.insert(slot).second)
          return fallback();
        barriers[consumer].push_back({Command::BarrierAll});
      } else {
        result.reason = "no same-pipeline completion mechanism";
        return result;
      }
      continue;
    }
    if (issue.kind != detail::Failure::None) {
      result.reason = issue.reason;
      return result;
    }
    issue = detail::Transfer(p, actual).run(true, true);
    if (issue.kind == detail::Failure::None) {
      auto checked = verify(p, actual);
      result.success = checked.success;
      result.reason = checked.reason;
      result.commands = std::move(actual);
      for (const auto &bundle : packets)
        for (const auto &packet : bundle.legs) {
          for (Cut publication : publicationCuts(packet))
            for (Cut acquisition : acquisitionCuts(packet)) {
              auto h = packet.handoff;
              h.publication = publication;
              h.acquisition = acquisition;
              result.handoffs.push_back(h);
            }
          if (packet.acknowledgment) {
            const auto &h = packet.handoff;
            result.handoffs.push_back(
                {h.observer, h.source, h.acquisition, h.acquisition});
          }
        }
      return result;
    }
    // Payload transfer is identical in speculative and strict runs. A strict
    // hazard cannot justify retrying the identical candidate without an edit.
    if (issue.kind == detail::Failure::Hazard)
      return fallback();
    bool repaired = false;
    if (issue.kind == detail::Failure::Occupancy) {
      // An early publication can overlap a previous logical generation. Do not
      // repair that by simply assigning more acknowledgment state to the key.
      for (auto &bundle : packets)
        for (auto &packet : bundle.legs) {
          auto &h = packet.handoff;
          if (h.source == issue.endpoint.source &&
              h.observer == issue.endpoint.observer &&
              packet.key == issue.endpoint.key) {
            repaired |= relocate(packet);
          }
        }
    } else if (issue.kind == detail::Failure::Rearm) {
      // Add replies only after the full memory plan fails a consumption proof.
      // Existing storage-release handoffs are already present in that proof.
      for (auto &bundle : packets)
        for (auto &packet : bundle.legs) {
          const auto h = packet.handoff;
          if (h.source != issue.endpoint.source ||
              h.observer != issue.endpoint.observer ||
              packet.key != issue.endpoint.key || packet.acknowledgment)
            continue;
          // On a one-visit word only an earlier consumption can require this
          // return. Do not append an unnecessary final-use acknowledgment.
          if (oneVisit && h.acquisition >= issue.cut)
            continue;
          auto key = chooseKey(p, packets, h.observer, h.source);
          if (key) {
            packet.acknowledgment = key;
            repaired = true;
          }
        }
    }
    if (!repaired)
      return fallback();
    // Helpers are evaluated as actual commands, including zero-memory-credit
    // acknowledgments. Other open recurrence obligations may remain; only final
    // verify accepts. This replay also exposes all-source effects of repairs.
    BundleQuery repairReplay(p, actual, {false});
    const auto repairedState =
        repairReplay.evaluate(render(p, barriers, packets));
    ++result.bundleTrials;
    if (!repairedState.complete) {
      result.reason = repairedState.reason;
      return result;
    }
    ++result.protocolRepairs;
  }
}

} // namespace
Result construct(const Program &p) {
  return construct(p, ConstructionOptions{});
}
Result construct(const Program &p, ConstructionOptions options) {
  std::optional<StorageFrontierAnalysis> storage;
  if (options.storageGuidance)
    storage.emplace(p);
  std::vector<CandidateStage> stages;
  if (storage && storage->complete())
    stages = {CandidateStage::Nearest, CandidateStage::Corridors,
              CandidateStage::Relays};
  stages.push_back(CandidateStage::Original);
  std::vector<StageReport> reports;
  std::size_t trials = 0, selections = 0, repairs = 0;
  for (auto stage : stages) {
    // Fresh candidate state: no key or endpoint commitment leaks from a failed
    // tier.
    const auto start = std::chrono::steady_clock::now();
    auto result = constructAttempt(p, storage ? &*storage : nullptr, stage);
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    reports.push_back({stage, result.success, result.reason,
                       result.bundleTrials, result.bundleSelections,
                       result.protocolRepairs, uint64_t(elapsed)});
    trials += result.bundleTrials;
    selections += result.bundleSelections;
    repairs += result.protocolRepairs;
    if (result.success || stage == CandidateStage::Original) {
      result.bundleTrials = trials;
      result.bundleSelections = selections;
      result.protocolRepairs = static_cast<unsigned>(repairs);
      result.stages = std::move(reports);
      return result;
    }
  }
  std::abort();
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
