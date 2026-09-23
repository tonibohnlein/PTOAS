// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_TRANSFER_H
#define PTO_OAHS_TRANSFER_H

#include "Control.h"
#include "PhaseTransfer.h"
#include "PTO/Transforms/OAHS/Analysis.h"
#include "PTO/Transforms/OAHS/Replay.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace mlir::pto::oahs::detail {
using Key = std::tuple<Pipe, Pipe, unsigned>;
using Bits = std::vector<uint8_t>;
constexpr uint8_t AllLanes = (1u << PipeCount) - 1;
struct Token {
  // Logical reference balance, not wall-clock occupancy. May-union at joins.
  uint8_t occupancy = 1; // 1 empty, 2 full
  bool valid = false;
  Bits remainder, acknowledgments;
  uint8_t consumedAt = AllLanes;
  bool operator==(const Token &b) const {
    return occupancy == b.occupancy && valid == b.valid &&
           remainder == b.remainder && acknowledgments == b.acknowledgments &&
           consumedAt == b.consumedAt;
  }
};
struct State {
  std::array<Bits, PipeCount> pending;
  std::vector<Token> tokens;
  State(std::size_t operations, std::size_t keys) : tokens(keys) {
    if (operations > Bits().max_size() || keys > Bits().max_size())
      std::abort(); // impossible container extent; compatible with no-EH builds
    for (auto &p : pending)
      p.resize(operations);
    for (auto &t : tokens) {
      t.remainder.assign(operations, 1);
      t.acknowledgments.resize(keys);
    }
  }
  bool operator==(const State &b) const {
    return pending == b.pending && tokens == b.tokens;
  }
  bool join(const State &b) {
    bool changed = false;
    for (unsigned q = 0; q < PipeCount; ++q)
      for (std::size_t i = 0; i < pending[q].size(); ++i)
        if (b.pending[q][i] && !pending[q][i]) {
          pending[q][i] = 1;
          changed = true;
        }
    for (std::size_t e = 0; e < tokens.size(); ++e) {
      auto &a = tokens[e];
      const auto &other = b.tokens[e];
      const uint8_t occupancy = a.occupancy | other.occupancy;
      const uint8_t consumedAt = a.consumedAt & other.consumedAt;
      const bool valid = a.valid && other.valid && occupancy == 2;
      changed |= occupancy != a.occupancy || consumedAt != a.consumedAt ||
                 valid != a.valid;
      a.occupancy = occupancy;
      a.consumedAt = consumedAt;
      a.valid = valid;
      for (std::size_t i = 0; i < a.remainder.size(); ++i) {
        const uint8_t value = valid ? (a.remainder[i] | other.remainder[i]) : 1;
        changed |= value != a.remainder[i];
        a.remainder[i] = value;
      }
      for (std::size_t f = 0; f < a.acknowledgments.size(); ++f) {
        const uint8_t value =
            valid && a.acknowledgments[f] && other.acknowledgments[f];
        changed |= value != a.acknowledgments[f];
        a.acknowledgments[f] = value;
      }
    }
    return changed;
  }
};
// Private checkpoint: these are unsuppressed discovery states, not certified
// completion facts. Only a ReplaySession tied to the same immutable Program
// may reuse them. A changed key numbering forces a full solve.
struct TransferCheckpoint {
  std::map<Key, unsigned> keyIds;
  Commands commands;
  std::vector<std::shared_ptr<State>> incoming;
};
inline bool identicalWord(const std::vector<Command> &a,
                          const std::vector<Command> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::tie(a[i].kind, a[i].source, a[i].observer, a[i].key) !=
        std::tie(b[i].kind, b[i].source, b[i].observer, b[i].key)) return false;
  return true;
}
inline bool identicalCommands(const Commands &a, const Commands &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!identicalWord(a[i], b[i])) return false;
  return true;
}
// One finite transfer semantics for both candidate analysis and final checking.
// Static operation classes represent all their dynamic visits. New visits add
// the class to every live remainder, so an old receipt cannot complete them.
// The public inspection service removes failed primitive certificates, invalidates their
// dependent facts, and exports only recomputed conservative snapshots. Final
// checking uses that service, with no unresolved obligations permitted.
class Transfer {
  const Program &p;
  const Commands &commands;
  std::map<Key, unsigned> keyIds;
  std::vector<std::vector<std::pair<std::size_t, Access>>> byCell;
  uint64_t work = 0;
  // Inspection monotonically removes
  // primitive certificates that fail at stabilized invariants, then recomputes
  // all dependent receipts. These masks never escape as a valid emitted plan.
  std::vector<std::vector<bool>> suppressed;
  std::vector<std::shared_ptr<State>> incoming;
  std::vector<AnalysisContext> contexts;
  std::vector<std::size_t> cutContexts, siteOperations;
  std::size_t exitSite = 0;
  void havocKey(State &s, unsigned e) {
    auto &token = s.tokens[e];
    token.occupancy = 3;
    token.valid = false;
    token.consumedAt = 0;
    std::fill(token.remainder.begin(), token.remainder.end(), 1);
    std::fill(token.acknowledgments.begin(), token.acknowledgments.end(), 0);
    for (auto &other : s.tokens)
      other.acknowledgments[e] = 0;
  }
  // Diagnostic work only. It cannot change acceptance or select serialization.
  void charge(uint64_t amount) {
    work += std::min(amount, std::numeric_limits<uint64_t>::max() - work);
  }
  void command(State &s, Cut cut, std::size_t index) {
    const auto &c = commands[cut][index];
    charge(1 + p.operations.size() + keyIds.size());
    if (c.kind == Command::BarrierAll) {
      for (auto &q : s.pending)
        std::fill(q.begin(), q.end(), 0);
      // Completion of prior commands is not consumption of a full event.
      // No extra consumption-knowledge credit is needed by this contract.
      return;
    }
    const auto source = unsigned(c.source), observer = unsigned(c.observer);
    if (c.kind == Command::Barrier) {
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == source)
          s.pending[source][i] = 0;
      return;
    }
    unsigned e = keyIds.at({c.source, c.observer, c.key});
    auto &token = s.tokens[e];
    if (!suppressed.empty() && suppressed[cut][index]) {
      // An unresolved endpoint is not a reset or an acquisition certificate.
      // Preserve payload obligations, lose knowledge of this key, and
      // invalidate every cached acknowledgment of its generation. Later valid
      // independent primitives can still be analyzed, with all original
      // obligations retained.
      havocKey(s, e);
      return;
    }
    if (c.kind == Command::Publish) {
      token.remainder = s.pending[source];
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == source)
          token.remainder[i] = 0;
      for (unsigned f = 0; f < s.tokens.size(); ++f)
        token.acknowledgments[f] =
            (s.tokens[f].consumedAt & (1u << source)) != 0;
      token.occupancy = 2;
      token.valid = true;
    } else {
      const bool usable = token.occupancy == 2 && token.valid;
      Bits facts = token.acknowledgments;
      if (usable) {
        for (std::size_t i = 0; i < p.operations.size(); ++i)
          s.pending[observer][i] &= token.remainder[i];
        for (unsigned f = 0; f < s.tokens.size(); ++f)
          if (facts[f])
            s.tokens[f].consumedAt |= 1u << observer;
      }
      // Latest consumption supersedes all earlier receipts of this key.
      for (auto &t : s.tokens)
        t.acknowledgments[e] = 0;
      token.consumedAt = 1u << observer;
      token.occupancy = 1;
      token.valid = false;
      std::fill(token.remainder.begin(), token.remainder.end(), 1);
      std::fill(token.acknowledgments.begin(), token.acknowledgments.end(), 0);
    }
    return;
  }
  void operation(State &s, std::size_t at) {
    for (std::size_t i = 0; i < commands[at].size(); ++i)
      command(s, at, i);
    const auto id = siteOperations[at];
    if (id != NoAnalysisId) {
      charge(1 + keyIds.size() + p.operations[id].accesses.size());
      issue(s, id);
    }
  }
  void issue(State &s, std::size_t id) {
    const auto observer = unsigned(p.operations[id].pipe);
    for (auto &q : s.pending)
      q[id] = 1;
    for (auto &token : s.tokens)
      if (token.valid)
        token.remainder[id] = 1;
    if (p.target.synchronous[observer])
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == observer)
          s.pending[observer][i] = 0;
  }
  std::vector<ControlSite> sites;
  uint64_t evaluations = 0, merges = 0;

  std::size_t buildControl() {
    auto graph = buildControlGraph(p);
    exitSite = graph.exit;
    siteOperations = std::move(graph.operations);
    sites = std::move(graph.sites);
    contexts = std::move(graph.contexts);
    cutContexts = std::move(graph.cutContexts);
    return graph.entry;
  }

  void solve(std::size_t root, const TransferCheckpoint *checkpoint = nullptr,
             ReplayStats *replay = nullptr) {
    // A null state is unreachable bottom, not fresh quiescent input.
    std::vector<bool> dirty(sites.size(), true);
    const bool compatible = checkpoint && checkpoint->keyIds == keyIds &&
        checkpoint->commands.size() == commands.size() &&
        checkpoint->incoming.size() == sites.size();
    if (compatible) {
      incoming = checkpoint->incoming;
      std::fill(dirty.begin(), dirty.end(), false);
      std::deque<std::size_t> affected;
      for (Cut c = 0; c < commands.size(); ++c)
        if (!identicalWord(commands[c], checkpoint->commands[c])) {
          dirty[c] = true;
          affected.push_back(c);
          if (replay) ++replay->changedCuts;
        }
      // Includes loop headers and earlier lexical cuts reached by backedges.
      // No old receipt, return or consumption fact survives inside this cone.
      while (!affected.empty()) {
        const auto c = affected.front(); affected.pop_front();
        for (auto next : sites[c].successors) if (!dirty[next]) {
          dirty[next] = true; affected.push_back(next);
        }
      }
      for (std::size_t i = 0; i < incoming.size(); ++i)
        if (dirty[i]) incoming[i].reset();
      if (replay) replay->kind = ReplayStats::Incremental;
    } else {
      incoming.assign(sites.size(), nullptr);
      if (replay && checkpoint) replay->kind = ReplayStats::KeyLayoutChanged;
    }
    if (replay) for (std::size_t i = 0; i < sites.size(); ++i) {
      replay->invalidatedSites += dirty[i];
      replay->reusedSites += !dirty[i] && bool(incoming[i]);
    }
    std::deque<std::size_t> queue;
    std::vector<bool> queued(sites.size());
    auto merge = [&](std::size_t target, const State &next) {
      ++merges;
      bool changed;
      if (!incoming[target]) {
        incoming[target] = std::make_shared<State>(next); changed = true;
      } else {
        // Checkpoints share only immutable states. Any destination that is
        // changed obtains its own copy before a join; unchanged prefixes are
        // not deep-copied merely to analyze another candidate.
        if (!incoming[target].unique())
          incoming[target] = std::make_shared<State>(*incoming[target]);
        changed = incoming[target]->join(next);
      }
      if (changed && !queued[target]) {
        queue.push_back(target); queued[target] = true;
      }
    };
    if (dirty[root]) merge(root, State(p.operations.size(), keyIds.size()));
    if (compatible) {
      // Unchanged predecessor states remain exact. Recompute their outgoing
      // contribution to the dirty cone; never seed dirty states with the OLD
      // joined state (which could retain a removed edge's completion credit).
      for (std::size_t i = 0; i < sites.size(); ++i) {
        if (dirty[i] || !incoming[i] ||
            std::none_of(sites[i].successors.begin(), sites[i].successors.end(),
                         [&](std::size_t next) { return dirty[next]; })) continue;
        State next = *incoming[i];
        ++evaluations; charge(1);
        if (replay) ++replay->boundaryEvaluations;
        if (i < commands.size()) operation(next, i);
        for (auto target : sites[i].successors)
          if (dirty[target]) merge(target, next);
      }
    }
    while (!queue.empty()) {
      const auto id = queue.front(); queue.pop_front(); queued[id] = false;
      State next = *incoming[id];
      ++evaluations; charge(1);
      if (id < commands.size()) operation(next, id);
      for (auto target : sites[id].successors) merge(target, next);
    }
  }
  BoundaryFacts snapshot(const State &s) const {
    BoundaryFacts out;
    out.pending = s.pending;
    for (const auto &t : s.tokens)
      out.events.push_back(
          {t.occupancy, t.valid, t.remainder, t.acknowledgments, t.consumedAt});
    return out;
  }
  std::vector<ProtocolObligation> preconditions(const State &s, Cut at,
                                                std::size_t index) const {
    const auto &c = commands[at][index];
    std::vector<ProtocolObligation> out;
    if (c.kind != Command::Publish && c.kind != Command::Acquire)
      return out;
    const auto &t = s.tokens[keyIds.at({c.source, c.observer, c.key})];
    auto add = [&](ProtocolObligation::Kind kind, const char *why) {
      out.push_back({kind,
                     at,
                     index,
                     cutContexts[at],
                     {c.source, c.observer, c.key, false},
                     why});
    };
    if (c.kind == Command::Publish) {
      if (t.occupancy != 1)
        add(ProtocolObligation::PublicationNotEmpty,
            "publication is not empty on every represented path");
      if (!(t.consumedAt & (1u << unsigned(c.source))))
        add(ProtocolObligation::ConsumptionNotEstablished,
            "event rearm lacks causal consumption");
    } else {
      if (t.occupancy != 2)
        add(ProtocolObligation::AcquisitionNotFull,
            "acquisition has no full publication on every represented path");
      if (!t.valid)
        add(ProtocolObligation::ReceiptNotEstablished,
            "acquisition receipt is not established on every represented path");
    }
    return out;
  }
  void requirements(const State &s, Cut at, AnalysisResult &out) const {
    const auto id = siteOperations[at];
    const auto observer = unsigned(p.operations[id].pipe);
    std::set<std::tuple<std::size_t, unsigned, CompletionRequirement::Kind>>
        seen;
    for (const auto &access : p.operations[id].accesses)
      for (const auto &other : byCell[access.cell]) {
        if (access.nativeAccumulatorClass != NoControlId &&
            access.nativeAccumulatorClass == other.second.nativeAccumulatorClass &&
            p.operations[id].nativeMmadAccumulate &&
            p.operations[id].pipe == Pipe::M && p.operations[other.first].pipe == Pipe::M)
          continue;
        if (!s.pending[observer][other.first])
          continue;
        auto add = [&](CompletionRequirement::Kind kind, Property property) {
          if (!seen.emplace(other.first, access.cell, kind).second)
            return;
          out.residuals.push_back(
              {kind,
               {other.first, id, access.cell, property},
               other.second,
               access,
               p.observed ? NoAnalysisId : cutContexts[other.first],
               cutContexts[at],
               at});
          if (property == Property::ResourceExclusion)
              out.residuals.back().reasons = TypedOrControl;
        };
        if (other.second.write && access.read)
          add(CompletionRequirement::RAW, Property::ByteCompletion);
        if (other.second.read && access.write)
          add(CompletionRequirement::WAR, Property::ByteCompletion);
        if (other.second.write && access.write)
          add(CompletionRequirement::WAW, Property::ByteCompletion);
        if (p.cells[access.cell].exclusive)
          add(CompletionRequirement::ExclusiveResource,
              Property::ResourceExclusion);
      }
  }

public:
  Transfer(const Program &program, const Commands &actual)
      : p(program), commands(actual), byCell(p.cells.size()) {
    for (const auto &at : commands)
      for (const auto &c : at)
        if (c.kind == Command::Publish || c.kind == Command::Acquire) {
          Key key{c.source, c.observer, c.key};
          if (!keyIds.count(key))
            keyIds[key] = keyIds.size();
        }
    for (std::size_t i = 0; i < p.operations.size(); ++i)
      for (const auto &a : p.operations[i].accesses)
        byCell[a.cell].push_back({i, a});
  }
  std::size_t siteCount() const { return sites.size(); }
  uint64_t evaluationCount() const { return evaluations; }
  uint64_t mergeCount() const { return merges; }
  uint64_t workCount() const { return work; }

  AnalysisResult inspect(AnalysisOptions options,
                         const TransferCheckpoint *checkpoint = nullptr,
                         TransferCheckpoint *save = nullptr,
                         ReplayStats *replay = nullptr) {
    if (p.finalBlocks) return phase::collect(p, commands, options).analysis;
    AnalysisResult out;
    work = evaluations = merges = 0;
    const auto root = buildControl();
    const auto count = commands.size();
    suppressed.resize(commands.size());
    for (std::size_t at = 0; at < commands.size(); ++at)
      suppressed[at].assign(commands[at].size(), false);
    std::map<std::pair<Cut, std::size_t>, std::vector<ProtocolObligation>>
        rejected;
    // Proof discovery is provisional. Remove certificates whose preconditions
    // fail at the invariant, poison only their key, and invalidate/recompute
    // all affected completion. Each repeating pass disables at least one NEW
    // command: at most event-command-count + 1 solves; no numerical work
    // allowance.
    while (true) {
      ++out.stats.certificationPasses;
      const bool firstPass = out.stats.certificationPasses == 1;
      solve(root, firstPass ? checkpoint : nullptr,
            firstPass ? replay : nullptr);
      if (firstPass && save) {
        save->keyIds = keyIds;
        save->commands = commands;
        save->incoming = incoming;
      }
      bool changed = false;
      for (Cut at = 0; at < count; ++at) {
        if (!incoming[at])
          continue;
        State state = *incoming[at];
        for (std::size_t i = 0; i < commands[at].size(); ++i) {
          if (!suppressed[at][i]) {
            auto problems = preconditions(state, at, i);
            if (!problems.empty()) {
              suppressed[at][i] = true;
              changed = true;
              rejected[{at, i}] = std::move(problems);
              ++out.stats.suppressedCommands;
            }
          }
          command(state, at, i);
        }
      }
      if (!changed)
        break;
    }
    out.contexts = contexts;
    out.keys.resize(keyIds.size());
    for (const auto &item : keyIds)
      out.keys[item.second] = {std::get<0>(item.first), std::get<1>(item.first),
                               std::get<2>(item.first), false};
    for (Cut at = 0; at < count; ++at) {
      CutFacts facts;
      facts.cut = at;
      facts.context = cutContexts[at];
      facts.reachable = bool(incoming[at]);
      if (!facts.reachable) {
        out.cuts.push_back(std::move(facts));
        continue;
      }
      State state = *incoming[at];
      if (options.captureStates)
        facts.incoming = snapshot(state);
      for (std::size_t i = 0; i < commands[at].size(); ++i) {
        out.commands.push_back(
            {at, i, cutContexts[at], commands[at][i], !suppressed[at][i]});
        if (suppressed[at][i]) {
          // Prefer diagnostics at the final conservative invariant. Retain the
          // invalidated certificate's cause if it is no longer locally visible.
          auto problems = preconditions(state, at, i);
          if (problems.empty())
            problems = rejected.at({at, i});
          out.protocol.insert(out.protocol.end(), problems.begin(),
                              problems.end());
        }
        command(state, at, i);
      }
      if (options.captureStates)
        facts.beforeIssue = snapshot(state);
      if (siteOperations[at] != NoAnalysisId) {
        requirements(state, at, out);
        issue(state, siteOperations[at]);
      } else if (at == exitSite) {
        for (const auto &item : keyIds) {
          if (state.tokens[item.second].occupancy != 1)
            out.protocol.push_back(
                {ProtocolObligation::UnconsumedAtExit, at, NoAnalysisId, 0,
                 out.keys[item.second],
                 "event is not empty at required invocation exit"});
        }
        if (p.invocation.retirement ==
            Program::InvocationContract::DrainAllAtReturn)
          for (std::size_t i = 0; i < p.operations.size(); ++i) {
            auto q = unsigned(p.operations[i].pipe);
            if (state.pending[q][i])
              out.retirement.push_back({i, Pipe(q)});
          }
      }
      if (options.captureStates)
        facts.outgoing = snapshot(state);
      out.cuts.push_back(std::move(facts));
    }
    out.stats.staticSites = sites.size();
    out.stats.siteEvaluations = evaluations;
    out.stats.merges = merges;
    out.stats.work = work;
    out.complete = true;
    if (!out.protocol.empty())
      out.reason = out.protocol.front().reason;
    else if (!out.residuals.empty())
      out.reason = "uncovered original completion requirements";
    else if (!out.retirement.empty())
      out.reason = "outstanding payload at required retirement";
    return out;
  }


};
} // namespace mlir::pto::oahs::detail
#endif
