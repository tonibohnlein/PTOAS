// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Plan.h"
#include "Transfer.h"
#include "PrefixQueries.h"
#include "BundleQueries.h"
#include <algorithm>
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
    reason = "invalid structured region kind"; return false;
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
    if (!validateRegion(p, child, seen, reason)) return false;
  return true;
}
bool valid(const Program &p, std::string &reason) {
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
      if (a.cell >= p.cells.size() || (!a.read && !a.write)) {
        reason = "invalid physical effect";
        return false;
      }
    for (const auto &r : op.resources)
      if (r.resource.empty() || (!r.acquire && !r.release) ||
          (r.timing != EffectTiming::Issue && r.timing != EffectTiming::Completion)) {
        reason = "malformed typed resource effect"; return false;
      }
    for (const auto &v : op.visibility)
      if (v.cell >= p.cells.size() || (!v.publish && !v.acquire) ||
          (v.timing != EffectTiming::Issue && v.timing != EffectTiming::Completion)) {
        reason = "malformed visibility effect"; return false;
      }
    for (const auto &population : {op.authoredEvents, op.internalTransfers})
      for (const auto &event : population)
        if (lane(event.source) >= PipeCount || lane(event.observer) >= PipeCount ||
            event.source == event.observer) {
          reason = "malformed explicit event effect"; return false;
        }
  }
  if (!p.body.children.empty() || p.body.kind != Region::Sequence) {
    std::vector<unsigned> seen(p.operations.size());
    if (!validateRegion(p, p.body, seen, reason)) return false;
    for (unsigned occurrences : seen)
      if (occurrences != 1) {
        reason = occurrences == 0
            ? "physical phase is absent from the control representation"
            : "physical phase occurs more than once in the control representation";
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
  return std::any_of(p.reservations.begin(), p.reservations.end(), [&](const EventIdentity &r) {
    return r.source == c.source && r.observer == c.observer && r.key == c.key;
  });
}
bool supportedEffects(const Program &p, std::string &reason) {
  for (const auto &op : p.operations) {
    if (!op.resources.empty() || !op.visibility.empty() ||
        !op.authoredEvents.empty() || !op.internalTransfers.empty()) {
      reason = "typed resource, visibility, or internal/authored transfer contract is not implemented";
      return false;
    }
  }
  for (const auto &r : p.reservations)
    if (!available(p.target, r.source, r.observer, r.key)) {
      reason = "reservation names an unavailable target key"; return false;
    }
  return true;
}
bool commandsValid(const Program &p, const Commands &commands, std::string &reason) {
  if (commands.size() != p.operations.size() + 1) {
    reason = "command cuts do not match original operations"; return false;
  }
  for (const auto &at : commands)
    for (const auto &c : at) {
      if (c.kind == Command::BarrierAll) {
        if (!p.target.barrierAll) { reason = "unsupported all-pipeline barrier"; return false; }
      } else if (c.kind == Command::Barrier) {
        if (lane(c.source) >= PipeCount || !p.target.supported[lane(c.source)] ||
            !p.target.barriers[lane(c.source)]) {
          reason = "unsupported pipeline barrier"; return false;
        }
      } else if ((c.kind != Command::Publish && c.kind != Command::Acquire) ||
                 !available(p.target, c.source, c.observer, c.key) || reserved(p, c)) {
        reason = "unsupported or reserved event direction/key"; return false;
      }
    }
  return true;
}
// Empty structural nodes do not force a different semantic construction path.
bool flatOrder(const Region &region, std::vector<std::size_t> &order) {
  if (region.kind == Region::Operation) { order.push_back(region.operation); return true; }
  if (region.kind == Region::Sequence) {
    for (const auto &child : region.children)
      if (!flatOrder(child, order)) return false;
    return true;
  }
  // Ignore an entirely payload-free subtree, but never flatten real choices or loops.
  std::vector<std::size_t> contents;
  for (const auto &child : region.children)
    if (!flatOrder(child, contents)) return false;
  return contents.empty();
}
bool flat(const Program &p) {
  if (p.body.kind == Region::Sequence && p.body.children.empty()) return true;
  std::vector<std::size_t> order;
  if (!flatOrder(p.body, order) || order.size() != p.operations.size()) return false;
  for (std::size_t i = 0; i < order.size(); ++i) if (order[i] != i) return false;
  return true;
}
Result conservative(const Program &p, bool scarcity = false) {
  Result result;
  if (!p.target.barrierAll) {
    result.reason = "no qualified conservative completion mechanism"; return result;
  }
  result.commands.resize(p.operations.size() + 1);
  for (Cut cut = 0; cut < p.operations.size(); ++cut) {
    result.commands[cut].push_back({Command::BarrierAll});
    ++result.conservativeBarriers;
  }
  if (p.invocation.retirement == Program::InvocationContract::DrainAllAtReturn) {
    result.commands.back().push_back({Command::BarrierAll}); ++result.conservativeBarriers;
  }
  if (scarcity) result.scarcityBarriers = result.conservativeBarriers;
  auto checked = verify(p, result.commands);
  result.success = checked.success; result.reason = checked.reason;
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
};
struct PacketBundle {
  std::vector<Packet> legs; // ordered at their common acquisition cut
};
std::vector<Cut> publicationCuts(const Packet &packet) {
  return packet.publications.empty() ? std::vector<Cut>{packet.handoff.publication}
                                     : packet.publications;
}
bool relocate(Packet &packet) {
  const auto cuts = publicationCuts(packet);
  if (cuts.size() == 1 && cuts[0] == packet.handoff.acquisition) return false;
  packet.publications.clear(); packet.handoff.publication = packet.handoff.acquisition;
  return true;
}
Commands render(const Program &p, const Commands &barriers, const std::vector<PacketBundle> &bundles) {
  Commands result = barriers;
  for (const auto &bundle : bundles) for (const auto &packet : bundle.legs) {
    const auto &h = packet.handoff;
    for (const auto publication : publicationCuts(packet))
      if (publication != h.acquisition)
        result[publication].insert(result[publication].begin(),
            {Command::Publish, h.source, h.observer, packet.key});
  }
  for (const auto &bundle : bundles) for (const auto &packet : bundle.legs) {
    const auto &h = packet.handoff;
    auto &at = result[h.acquisition];
    const auto cuts = publicationCuts(packet);
    if (std::find(cuts.begin(), cuts.end(), h.acquisition) != cuts.end())
      at.push_back({Command::Publish, h.source, h.observer, packet.key});
    at.push_back({Command::Acquire, h.source, h.observer, packet.key});
    if (packet.acknowledgment) {
      at.push_back({Command::Publish, h.observer, h.source, *packet.acknowledgment});
      at.push_back({Command::Acquire, h.observer, h.source, *packet.acknowledgment});
    }
  }
  if (p.invocation.retirement == Program::InvocationContract::DrainAllAtReturn)
    result.back().push_back({Command::BarrierAll});
  return result;
}
std::vector<unsigned> eligibleKeys(const Program &p, Pipe source, Pipe observer) {
  std::vector<unsigned> out;
  if (source == observer || !p.target.supported[lane(source)] ||
      !p.target.supported[lane(observer)]) return out;
  for (unsigned k : p.target.keys[lane(source)][lane(observer)])
    if (!reserved(p, {Command::Publish, source, observer, k}) &&
        std::find(out.begin(),out.end(),k)==out.end()) out.push_back(k);
  return out;
}
std::optional<unsigned> chooseKey(const Program &p, const std::vector<PacketBundle> &bundles,
                                Pipe source, Pipe observer) {
  std::optional<unsigned> reused;
  for (unsigned key : eligibleKeys(p, source, observer)) {
    if (!reused) reused = key;
    bool used = false;
    for (const auto &bundle : bundles) for (const auto &packet : bundle.legs) {
      const auto &h = packet.handoff;
      used |= h.source == source && h.observer == observer && packet.key == key;
      used |= packet.acknowledgment && h.observer == source && h.source == observer &&
              *packet.acknowledgment == key;
    }
    if (!used) return key;
  }
  return reused; // proposal only; never a fabricated consumption edge
}
// Retain all shortest topology routes, including lanes with no payload effects.
// This is a finite placement policy, not completeness over all possible routes.
std::vector<std::vector<Pipe>> routes(const Program &p, Pipe source, Pipe observer) {
  std::array<unsigned, PipeCount> distance; distance.fill(PipeCount);
  distance[lane(source)] = 0; std::vector<Pipe> queue{source};
  for (std::size_t i=0;i<queue.size();++i) for (unsigned b=0;b<PipeCount;++b)
    if (!eligibleKeys(p,queue[i],Pipe(b)).empty() && distance[b]==PipeCount) {
      distance[b]=distance[lane(queue[i])]+1; queue.push_back(Pipe(b));
    }
  std::vector<std::vector<Pipe>> out;
  if (distance[lane(observer)]==PipeCount) return out;
  std::vector<std::vector<Pipe>> work{{source}};
  while (!work.empty()) {
    auto path=std::move(work.back()); work.pop_back(); const auto a=path.back();
    if(a==observer) { out.push_back(std::move(path)); continue; }
    for(unsigned b=PipeCount;b-->0;)
      if(distance[b]==distance[lane(a)]+1 && distance[b]<=distance[lane(observer)] &&
         !eligibleKeys(p,a,Pipe(b)).empty()) {
        auto next=path; next.push_back(Pipe(b)); work.push_back(std::move(next));
      }
  }
  return out;
}
// A general backward cut-frontier proposal: the immediately preceding physical
// cuts along every predecessor arm. No matching or credit is assumed. Replay may
// reject it (e.g. a branch's last write needs a not-yet-supported exit cut).
std::vector<Cut> predecessorFrontier(const Program &p, Cut consumer) {
  const auto g=detail::buildControlGraph(p); const auto n=p.operations.size();
  std::vector<std::vector<std::pair<std::size_t,bool>>> pred(g.sites.size());
  for(std::size_t a=0;a<g.sites.size();++a)
    for(std::size_t k=0;k<g.sites[a].successors.size();++k)
      pred[g.sites[a].successors[k]].push_back({a,g.sites[a].backedgeOwners[k]!=NoAnalysisId});
  std::vector<std::size_t> work{consumer}; std::vector<bool> seen(g.sites.size());
  std::set<Cut> out;
  while(!work.empty()) {
    auto at=work.back();work.pop_back(); if(seen[at]) continue;seen[at]=true;
    if(at==g.entry) return {};
    for(auto [previous,backedge]:pred[at]) {
      if(backedge) return {}; // occurrence-dependent frontiers are M4, not guessed
      if(previous<n && previous!=consumer) out.insert(previous);
      else if(previous==consumer) return {};
      else work.push_back(previous);
    }
  }
  if(out.size()<2) return {};
  return {out.begin(),out.end()};
}
} // namespace

Result validateProgram(const Program &p) {
  Result result;
  result.success = valid(p, result.reason);
  return result;
}

AnalysisResult analyze(const Program &p, const Commands &commands, AnalysisOptions options) {
  AnalysisResult out;
  if (!valid(p, out.reason)) {
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
      const auto &op = p.operations[i];
      if (!op.complete)
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
                                   "missing operation completeness contract"});
      else if (lane(op.pipe) >= PipeCount || !p.target.supported[lane(op.pipe)])
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
                                   "operation pipeline outside target contract"});
    }
    if (out.diagnostics.empty())
      out.diagnostics.push_back({AnalysisDiagnostic::InvalidInput, NoAnalysisId, out.reason});
    return out;
  }
  if (!supportedEffects(p, out.reason)) {
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
      const auto &op = p.operations[i];
      for (const auto &resource : op.resources)
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
            "resource transfer not implemented: " + resource.resource});
      for (const auto &visibility : op.visibility)
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
            "visibility transfer not implemented for cell " + std::to_string(visibility.cell)});
      if (!op.authoredEvents.empty())
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
                                  "authored event contract not implemented"});
      if (!op.internalTransfers.empty())
        out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, i,
                                  "internal phase transfer contract not implemented"});
    }
    if (out.diagnostics.empty())
      out.diagnostics.push_back({AnalysisDiagnostic::UnsupportedSemantics, NoAnalysisId, out.reason});
    return out;
  }
  if (!commandsValid(p, commands, out.reason)) {
    out.diagnostics.push_back({AnalysisDiagnostic::InvalidCommands, NoAnalysisId, out.reason});
    return out;
  }
  return detail::Transfer(p, commands).inspect(options);
}
AnalysisResult analyze(const Program &p) {
  return analyze(p, Commands(p.operations.size() + 1));
}
Result verify(const Program &p, const Commands &commands) {
  const auto analysis = analyze(p, commands, {false});
  Result result;
  result.success = analysis.verified(); result.reason = analysis.reason;
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

PrefixQuery::PrefixQuery(Program p, Commands c) : impl(std::make_unique<Impl>(std::move(p), std::move(c))) {}
PrefixQuery::PrefixQuery(Program p) {
  Commands empty(p.operations.size() + 1);
  impl = std::make_unique<Impl>(std::move(p), std::move(empty));
}
PrefixQuery::~PrefixQuery() = default;
PrefixQuery::PrefixQuery(PrefixQuery &&) noexcept = default;
PrefixQuery &PrefixQuery::operator=(PrefixQuery &&) noexcept = default;
const AnalysisResult &PrefixQuery::analysis() const { return impl->report; }
std::vector<CompletionRequirement> PrefixQuery::consumerRequirements(Cut consumer) const {
  return impl->requirements(consumer);
}
BackwardCutResult PrefixQuery::backwardCuts(Cut consumer) const { return impl->backward(consumer); }
ProspectivePrefix PrefixQuery::inspectPrefix(Pipe source, Cut publication, Cut consumer) const {
  return impl->inspect(source, publication, consumer, impl->backward(consumer));
}
PrefixCover PrefixQuery::coverByPrefixes(Cut consumer) const {
  PrefixCover out; out.consumer = consumer; out.backward = impl->backward(consumer);
  if (!out.backward.complete) { out.reason = out.backward.reason; return out; }
  out.requirements = impl->requirements(consumer);
  out.jointUncoveredOperations.assign(impl->program.operations.size(), 1);
  out.complete = true;
  if (out.requirements.empty()) return out;
  for (const auto &cut : out.backward.cuts)
    for (unsigned source = 0; source < PipeCount; ++source) {
      if (!impl->program.target.supported[source]) continue;
      if (Pipe(source) == impl->program.operations[consumer].pipe && cut.cut != consumer) continue;
      // A prospective remainder only GROWS between captures. If its seed leaves
      // every required class outstanding it cannot discharge a component here.
      const auto seed = impl->sourceRemainder(Pipe(source), cut.cut);
      const bool useful = std::any_of(out.requirements.begin(), out.requirements.end(),
                           [&](const CompletionRequirement &r) { return !seed[r.demand.producer]; });
      if (useful) out.candidates.push_back(impl->inspect(Pipe(source), cut.cut, consumer, out.backward));
    }
  std::vector<bool> remaining(out.requirements.size(), true);
  while (true) {
    std::size_t best = NoAnalysisId, bestGain = 0;
    for (std::size_t i = 0; i < out.candidates.size(); ++i) {
      const auto &candidate = out.candidates[i];
      if (!candidate.selectable()) continue;
      std::size_t gain = 0;
      for (auto component : candidate.coveredRequirements) gain += remaining[component];
      // Candidates are generated in source-cut/lane order. Equal gain keeps the
      // earlier cut. This is a tie-break, not a structured order-dominance proof.
      if (gain > bestGain) { best = i; bestGain = gain; }
    }
    if (!bestGain) break;
    out.selected.push_back(best);
    for (std::size_t op = 0; op < out.jointUncoveredOperations.size(); ++op)
      out.jointUncoveredOperations[op] &= out.candidates[best].uncoveredOperations[op];
    for (auto component : out.candidates[best].coveredRequirements) remaining[component] = false;
  }
  for (std::size_t i = 0; i < remaining.size(); ++i) if (remaining[i]) out.remaining.push_back(i);
  if (!out.remaining.empty()) out.reason = "no established prospective cover for every residual component";
  return out;
}

Result construct(const Program &p) {
  Result result;
  if (!valid(p, result.reason) || !supportedEffects(p, result.reason)) return result;
  Commands barriers(p.operations.size() + 1);
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
      const auto consumer = issue.cut;
      const Pipe observer = p.operations[consumer].pipe;
      // Prefer incoming acquired completion before same-lane fences. A real
      // return can cover a same-lane conflict without a separate barrier.
      auto chosen = issue.missing.front();
      for (const auto &d : issue.missing)
        if (p.operations[chosen.producer].pipe == observer &&
            p.operations[d.producer].pipe != observer) { chosen = d; break; }
      const Pipe source = p.operations[chosen.producer].pipe;
      for (const auto &d : issue.missing)
        if (p.operations[d.producer].pipe == source && d.producer > chosen.producer) chosen = d;
      result.demands.insert(result.demands.end(), issue.missing.begin(), issue.missing.end());
      const auto slot = std::make_pair(consumer, source);
      if (auto found = packetSlots.find(slot); found != packetSlots.end()) {
        auto &packet = packets[found->second].legs.front();
        // A later discovered generation may not belong to the saved early
        // prefix. Relocate this one packet; never append duplicates forever.
        if (relocate(packet)) {}
        else return conservative(p, true);
      } else if (source == observer && p.target.barriers[lane(source)]) {
        if (!barrierSlots.insert(slot).second) return conservative(p, true);
        barriers[consumer].push_back({Command::Barrier, source});
      } else if (source != observer) {
        PrefixQuery query(p, actual);
        const auto cover = query.coverByPrefixes(consumer);
        BundleQuery replay(p, actual);
        const auto frontier = predecessorFrontier(p, consumer);
        struct Proposal { Pipe publisher; Cut cut; std::vector<Cut> alternatives; };
        std::vector<Proposal> proposals;
        for (const auto &candidate : cover.candidates) {
          if (!candidate.selectable() || candidate.source == observer) continue;
          bool suppliesSource = std::all_of(issue.missing.begin(),issue.missing.end(),[&](const Demand &d) {
            return p.operations[d.producer].pipe!=source || !candidate.uncoveredOperations[d.producer];
          });
          if (suppliesSource) proposals.push_back({candidate.source,candidate.publication,{}});
        }
        // A whole alternative-publication bundle can be valid even though none
        // of its individual source/target pairs passes the M2 balance monitor.
        if (!frontier.empty()) proposals.insert(proposals.begin(), {source,frontier.front(),frontier});
        proposals.push_back({source,consumer,{}}); // ordinary co-located packet
        std::optional<PacketBundle> best;
        std::size_t bestCredit=0, bestCollateral=0, bestCost=0;
        // Actual all-source replay is authoritative. Provisional memory progress
        // is only a construction step when recurrence certificates remain open;
        // it is never exported as BundleEvaluation::discharged.
        std::optional<PacketBundle> pending;
        for (const auto &proposal : proposals) {
          for (const auto &path : routes(p,proposal.publisher,observer)) {
            PacketBundle bundle; auto population=packets;
            for(std::size_t j=0;j+1<path.size();++j) {
              auto key=chooseKey(p,population,path[j],path[j+1]);
              if(!key) { bundle.legs.clear(); break; }
              Packet leg{{path[j],path[j+1],j?consumer:proposal.cut,consumer},*key,{},
                         j?std::vector<Cut>{}:proposal.alternatives};
              bundle.legs.push_back(leg);
              // Include helper keys immediately in subsequent resource queries.
              population.push_back({{leg}});
            }
            if(bundle.legs.empty()) continue;
            auto trial=packets;trial.push_back(bundle);
            auto commands=render(p,barriers,trial);
            const auto speculative=detail::Transfer(p,commands).run(true,false);
            bool advances=speculative.kind==detail::Failure::None;
            if(speculative.kind==detail::Failure::Hazard) {
              if(speculative.cut==consumer) {
                advances=speculative.missing.size()<issue.missing.size();
                // The motivating source must really be addressed, not just
                // another source whose component happened to be counted first.
                for(const auto &d:speculative.missing)
                  if(p.operations[d.producer].pipe==source) advances=false;
              } else advances=lexicalRanks[speculative.cut]>lexicalRanks[consumer];
            }
            if(!advances) continue;
            auto evaluated=replay.evaluate(std::move(commands)); ++result.bundleTrials;
            if(!evaluated.complete || !evaluated.introduced.empty()) continue;
            std::size_t credit=0;
            for(const auto &d:evaluated.discharged) credit+=d.demand.consumer==consumer;
            const auto cost=evaluated.resources.publications+evaluated.resources.acquisitions;
            std::size_t collateral=0;
            const auto &before=replay.analysis().cuts[consumer].beforeIssue;
            const auto &after=evaluated.analysis.cuts[consumer].beforeIssue;
            if(before && after) for(std::size_t a=0;a<p.operations.size();++a) {
              const bool needed=std::any_of(issue.missing.begin(),issue.missing.end(),
                  [&](const Demand &d) { return d.producer==a; });
              collateral+=!needed && before->pending[lane(observer)][a] &&
                           !after->pending[lane(observer)][a];
            }
            // Distinct goals: all-source coverage, collateral completion at this
            // consumer, then static commands. This is an explicit local heuristic,
            // NOT a whole-program order-dominance or latency certificate.
            if(evaluated.analysis.protocol.empty() && credit) {
              if(!best || credit>bestCredit || (credit==bestCredit &&
                  std::make_pair(collateral,cost)<std::make_pair(bestCollateral,bestCost))) {
                best=bundle;bestCredit=credit;bestCollateral=collateral;bestCost=cost;
              }
            } else if(!pending && proposal.alternatives.empty()) pending=bundle;
          }
        }
        if(!best) best=std::move(pending);
        if(best) {
          packetSlots.emplace(slot,packets.size()); packets.push_back(std::move(*best));
          ++result.bundleSelections;
        } else if(p.target.barrierAll) {
          if(!barrierSlots.insert(slot).second) return conservative(p,true);
          barriers[consumer].push_back({Command::BarrierAll});
        } else {
          result.reason="construction policy found no verified completion bundle; not a target infeasibility proof";
          return result;
        }
      } else if (p.target.barrierAll) {
        if (!barrierSlots.insert(slot).second) return conservative(p, true);
        barriers[consumer].push_back({Command::BarrierAll});
      } else { result.reason = "no same-pipeline completion mechanism"; return result; }
      continue;
    }
    if (issue.kind != detail::Failure::None) {
      result.reason = issue.reason; return result;
    }
    issue = detail::Transfer(p, actual).run(true, true);
    if (issue.kind == detail::Failure::None) {
      auto checked = verify(p, actual);
      result.success = checked.success; result.reason = checked.reason;
      result.commands = std::move(actual);
      for (const auto &bundle : packets) for (const auto &packet : bundle.legs) {
        for(Cut publication : publicationCuts(packet)) {
          auto h=packet.handoff;h.publication=publication;result.handoffs.push_back(h);
        }
        if (packet.acknowledgment) {
          const auto &h = packet.handoff;
          result.handoffs.push_back({h.observer, h.source, h.acquisition, h.acquisition});
        }
      }
      return result;
    }
    // Payload transfer is identical in speculative and strict runs. A strict
    // hazard cannot justify retrying the identical candidate without an edit.
    if (issue.kind == detail::Failure::Hazard) return conservative(p, true);
    bool repaired = false;
    if (issue.kind == detail::Failure::Occupancy) {
      // An early publication can overlap a previous logical generation. Do not
      // repair that by simply assigning more acknowledgment state to the key.
      for (auto &bundle : packets) for (auto &packet : bundle.legs) {
        auto &h = packet.handoff;
        if (h.source == issue.endpoint.source && h.observer == issue.endpoint.observer &&
            packet.key == issue.endpoint.key) {
          repaired |= relocate(packet);
        }
      }
    } else if (issue.kind == detail::Failure::Rearm) {
      // Add replies only after the full memory plan fails a consumption proof.
      // Existing storage-release handoffs are already present in that proof.
      for (auto &bundle : packets) for (auto &packet : bundle.legs) {
        const auto h = packet.handoff;
        if (h.source != issue.endpoint.source || h.observer != issue.endpoint.observer ||
            packet.key != issue.endpoint.key || packet.acknowledgment) continue;
        // On a one-visit word only an earlier consumption can require this
        // return. Do not append an unnecessary final-use acknowledgment.
        if (oneVisit && h.acquisition >= issue.cut) continue;
        auto key = chooseKey(p, packets, h.observer, h.source);
        if (key) { packet.acknowledgment = key; repaired = true; }
      }
    }
    if (!repaired) return conservative(p, true);
    // Helpers are evaluated as actual commands, including zero-memory-credit
    // acknowledgments. Other open recurrence obligations may remain; only final
    // verify accepts. This replay also exposes all-source effects of repairs.
    BundleQuery repairReplay(p, actual, {false});
    const auto repairedState = repairReplay.evaluate(render(p, barriers, packets));
    ++result.bundleTrials;
    if (!repairedState.complete) { result.reason = repairedState.reason; return result; }
    ++result.protocolRepairs;
  }
}
} // namespace mlir::pto::oahs
