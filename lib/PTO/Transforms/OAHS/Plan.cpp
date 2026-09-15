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
struct Packet {
  Handoff handoff;
  unsigned key = 0;
  std::optional<unsigned> acknowledgment;
};
Commands render(const Program &p, const Commands &barriers, const std::vector<Packet> &packets) {
  Commands result = barriers;
  // Early publications precede commands at the following source cut. They
  // capture only knowledge actually established at that cut on reconstruction.
  for (const auto &packet : packets) {
    const auto &h = packet.handoff;
    if (h.publication != h.acquisition)
      result[h.publication].insert(result[h.publication].begin(),
          {Command::Publish, h.source, h.observer, packet.key});
  }
  for (const auto &packet : packets) {
    const auto &h = packet.handoff;
    auto &at = result[h.acquisition];
    if (h.publication == h.acquisition)
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
std::optional<unsigned> chooseKey(const Program &p, const std::vector<Packet> &packets,
                                  Pipe source, Pipe observer) {
  std::optional<unsigned> reused;
  for (unsigned key : p.target.keys[lane(source)][lane(observer)]) {
    if (reserved(p, {Command::Publish, source, observer, key})) continue;
    if (!reused) reused = key;
    bool used = false;
    for (const auto &packet : packets) {
      const auto &h = packet.handoff;
      used |= h.source == source && h.observer == observer && packet.key == key;
      used |= packet.acknowledgment && h.observer == source && h.source == observer &&
              *packet.acknowledgment == key;
    }
    if (!used) return key;
  }
  // Reuse is a proposal, not a fact. All keys, including these, are checked by
  // the finite protocol transfer before any emitted program can be committed.
  return reused;
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
  std::vector<Packet> packets;
  const bool oneVisit = flat(p);
  // Each (consumer, source) has one finite repair slot: absent -> packet or
  // barrier. A packet may move to its consumer once and gain one reply. Keys
  // are fixed at creation. An unchanged/unsupported repair terminates below;
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
        auto &h = packets[found->second].handoff;
        // A later discovered generation may not belong to the saved early
        // prefix. Relocate this one packet; never append duplicates forever.
        if (h.publication != h.acquisition) h.publication = h.acquisition;
        else return conservative(p, true);
      } else if (source == observer && p.target.barriers[lane(source)]) {
        if (!barrierSlots.insert(slot).second) return conservative(p, true);
        barriers[consumer].push_back({Command::Barrier, source});
      } else if (source != observer) {
        // Query only established M1 facts for source-prefix proposals. The
        // session owns this exact candidate and is discarded after every edit.
        // Realization/reuse can still fail; no coverage record bypasses verify.
        PrefixQuery query(p, actual);
        const auto cover = query.coverByPrefixes(consumer);
        const ProspectivePrefix *best = nullptr;
        for (const auto &candidate : cover.candidates) {
          if (!candidate.selectable() || candidate.source == observer) continue;
          const bool suppliesSource = std::all_of(issue.missing.begin(), issue.missing.end(),
            [&](const Demand &d) {
              return p.operations[d.producer].pipe != source ||
                     !candidate.uncoveredOperations[d.producer];
            });
          if (!suppliesSource) continue;
          if (!best || candidate.coveredRequirements.size() > best->coveredRequirements.size()) best = &candidate;
        }
        const Pipe publisher = best ? best->source : source;
        auto key = chooseKey(p, packets, publisher, observer);
        if (key) {
          // A missing precise query does not remove the ordinary source cut.
          // The co-located source/target packet is checked like every other plan.
          const Cut publication = best ? best->publication : consumer;
          packetSlots.emplace(slot, packets.size());
          packets.push_back({{publisher, observer, publication, consumer}, *key, {}});
        } else if (p.target.barrierAll) {
          if (!barrierSlots.insert(slot).second) return conservative(p, true);
          barriers[consumer].push_back({Command::BarrierAll});
        } else { result.reason = "no eligible completion route"; return result; }
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
      for (const auto &packet : packets) {
        result.handoffs.push_back(packet.handoff);
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
      for (auto &packet : packets) {
        auto &h = packet.handoff;
        if (h.source == issue.endpoint.source && h.observer == issue.endpoint.observer &&
            packet.key == issue.endpoint.key && h.publication != h.acquisition) {
          h.publication = h.acquisition; repaired = true;
        }
      }
    } else if (issue.kind == detail::Failure::Rearm) {
      // Add replies only after the full memory plan fails a consumption proof.
      // Existing storage-release handoffs are already present in that proof.
      for (std::size_t i = 0; i < packets.size(); ++i) {
        const auto h = packets[i].handoff;
        if (h.source != issue.endpoint.source || h.observer != issue.endpoint.observer ||
            packets[i].key != issue.endpoint.key || packets[i].acknowledgment) continue;
        // On a one-visit word only an earlier consumption can require this
        // return. Do not append an unnecessary final-use acknowledgment.
        if (oneVisit && h.acquisition >= issue.cut) continue;
        auto key = chooseKey(p, packets, h.observer, h.source);
        if (key) { packets[i].acknowledgment = key; repaired = true; }
      }
    }
    if (!repaired) return conservative(p, true);
    ++result.protocolRepairs;
  }
}
} // namespace mlir::pto::oahs
