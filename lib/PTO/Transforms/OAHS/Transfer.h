// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_TRANSFER_H
#define PTO_OAHS_TRANSFER_H

#include "PTO/Transforms/OAHS/Analysis.h"
#include <algorithm>
#include <array>
#include <map>
#include <deque>
#include <optional>
#include <limits>
#include <cstdlib>
#include <tuple>
#include <set>
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
    for (auto &p : pending) p.resize(operations);
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
          pending[q][i] = 1; changed = true;
        }
    for (std::size_t e = 0; e < tokens.size(); ++e) {
      auto &a = tokens[e]; const auto &other = b.tokens[e];
      const uint8_t occupancy = a.occupancy | other.occupancy;
      const uint8_t consumedAt = a.consumedAt & other.consumedAt;
      const bool valid = a.valid && other.valid && occupancy == 2;
      changed |= occupancy != a.occupancy || consumedAt != a.consumedAt || valid != a.valid;
      a.occupancy = occupancy; a.consumedAt = consumedAt; a.valid = valid;
      for (std::size_t i = 0; i < a.remainder.size(); ++i) {
        const uint8_t value = valid ? (a.remainder[i] | other.remainder[i]) : 1;
        changed |= value != a.remainder[i]; a.remainder[i] = value;
      }
      for (std::size_t f = 0; f < a.acknowledgments.size(); ++f) {
        const uint8_t value = valid && a.acknowledgments[f] && other.acknowledgments[f];
        changed |= value != a.acknowledgments[f]; a.acknowledgments[f] = value;
      }
    }
    return changed;
  }
};
struct Failure {
  enum Kind { None, Hazard, Rearm, Occupancy, Invalid, Retirement } kind = None;
  Cut cut = 0;
  std::size_t command = 0;
  Command endpoint;
  std::vector<Demand> missing;
  std::string reason;
};

// One finite transfer semantics for both candidate analysis and final checking.
// Static operation classes represent all their dynamic visits. New visits add
// the class to every live remainder, so an old receipt cannot complete them.
// Private construction queries may assume protocol preconditions while proposing
// repairs. They never expose authoritative completion. The public inspection
// service removes failed primitive certificates, invalidates their dependent
// facts, and exports only recomputed conservative snapshots. Final checking uses
// that service, with no unresolved obligations permitted.
class Transfer {
  const Program &p;
  const Commands &commands;
  std::map<Key, unsigned> keyIds;
  std::vector<std::vector<std::pair<std::size_t, Access>>> byCell;
  uint64_t work = 0;
  Failure failure;
  // Empty in speculative construction queries. Inspection monotonically removes
  // primitive certificates that fail at stabilized invariants, then recomputes
  // all dependent receipts. These masks never escape as a valid emitted plan.
  std::vector<std::vector<bool>> suppressed;
  std::vector<std::optional<State>> incoming;
  std::vector<AnalysisContext> contexts;
  std::vector<std::size_t> cutContexts;
  void havocKey(State &s, unsigned e) {
    auto &token = s.tokens[e];
    token.occupancy = 3; token.valid = false; token.consumedAt = 0;
    std::fill(token.remainder.begin(), token.remainder.end(), 1);
    std::fill(token.acknowledgments.begin(), token.acknowledgments.end(), 0);
    for (auto &other : s.tokens) other.acknowledgments[e] = 0;
  }
  // Diagnostic work only. It cannot change acceptance or select serialization.
  void charge(uint64_t amount) {
    work += std::min(amount, std::numeric_limits<uint64_t>::max() - work);
  }
  bool command(State &s, Cut cut, std::size_t index, bool checkProtocol) {
    const auto &c = commands[cut][index];
    charge(1 + p.operations.size() + keyIds.size());
    if (c.kind == Command::BarrierAll) {
      for (auto &q : s.pending) std::fill(q.begin(), q.end(), 0);
      // Completion of prior commands is not consumption of a full event.
      // No extra consumption-knowledge credit is needed by this contract.
      return true;
    }
    const auto source = unsigned(c.source), observer = unsigned(c.observer);
    if (c.kind == Command::Barrier) {
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == source) s.pending[source][i] = 0;
      return true;
    }
    unsigned e = keyIds.at({c.source, c.observer, c.key});
    auto &token = s.tokens[e];
    if (!suppressed.empty() && suppressed[cut][index]) {
      // An unresolved endpoint is not a reset or an acquisition certificate.
      // Preserve payload obligations, lose knowledge of this key, and invalidate
      // every cached acknowledgment of its generation. Later valid independent
      // primitives can still be analyzed, with all original obligations retained.
      havocKey(s, e);
      return true;
    }
    auto fail = [&](Failure::Kind kind, const char *reason) {
      failure.kind = kind; failure.cut = cut; failure.command = index;
      failure.endpoint = c; failure.reason = reason; return false;
    };
    if (checkProtocol) {
      const auto problems = preconditions(s, cut, index);
      if (!problems.empty()) {
        const auto &problem = problems.front();
        return fail(problem.kind == ProtocolObligation::ConsumptionNotEstablished
                        ? Failure::Rearm : Failure::Occupancy,
                    problem.reason.c_str());
      }
    }
    if (c.kind == Command::Publish) {
      token.remainder = s.pending[source];
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == source) token.remainder[i] = 0;
      for (unsigned f = 0; f < s.tokens.size(); ++f)
        token.acknowledgments[f] = (s.tokens[f].consumedAt & (1u << source)) != 0;
      token.occupancy = 2; token.valid = true;
    } else {
      const bool usable = token.occupancy == 2 && token.valid;
      Bits facts = token.acknowledgments;
      if (usable) {
        for (std::size_t i = 0; i < p.operations.size(); ++i)
          s.pending[observer][i] &= token.remainder[i];
        for (unsigned f = 0; f < s.tokens.size(); ++f)
          if (facts[f]) s.tokens[f].consumedAt |= 1u << observer;
      }
      // Latest consumption supersedes all earlier receipts of this key.
      for (auto &t : s.tokens) t.acknowledgments[e] = 0;
      token.consumedAt = 1u << observer;
      token.occupancy = 1; token.valid = false;
      std::fill(token.remainder.begin(), token.remainder.end(), 1);
      std::fill(token.acknowledgments.begin(), token.acknowledgments.end(), 0);
    }
    return true;
  }
  bool cut(State &s, Cut id, bool checkProtocol) {
    for (std::size_t i = 0; i < commands[id].size(); ++i)
      if (!command(s, id, i, checkProtocol)) return false;
    return true;
  }
  bool operation(State &s, std::size_t id, bool checkPayload, bool checkProtocol) {
    if (!cut(s, id, checkProtocol)) return false;
    unsigned observer = unsigned(p.operations[id].pipe);
    charge(1 + keyIds.size() + p.operations[id].accesses.size());
    if (checkPayload) {
      std::vector<Demand> missing;
      for (const auto &a : p.operations[id].accesses)
        for (const auto &other : byCell[a.cell]) {
          charge(1);
          if (!s.pending[observer][other.first]) continue;
          if (a.write || other.second.write)
            missing.push_back({other.first, id, a.cell, Property::ByteCompletion});
          else if (p.cells[a.cell].exclusive)
            missing.push_back({other.first, id, a.cell, Property::ResourceExclusion});
        }
      if (!missing.empty()) {
        failure.kind = Failure::Hazard; failure.cut = id;
        failure.missing = std::move(missing);
        failure.reason = "uncovered original access at operation " + std::to_string(id);
        return false;
      }
    }
    issue(s, id);
    return true;
  }
  void issue(State &s, std::size_t id) {
    const auto observer = unsigned(p.operations[id].pipe);
    for (auto &q : s.pending) q[id] = 1;
    for (auto &token : s.tokens)
      if (token.valid) token.remainder[id] = 1;
    if (p.target.synchronous[observer])
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == observer) s.pending[observer][i] = 0;
  }
  struct Site {
    // IDs [0,N) are physical phases; N is the original invocation exit.
    // Remaining sites are ONLY original control edges, never occurrences.
    std::vector<std::size_t> successors;
  };
  std::vector<Site> sites;
  uint64_t evaluations = 0, merges = 0;

  std::size_t buildControl() {
    const std::size_t exit = p.operations.size();
    sites.clear(); sites.resize(exit + 1);
    contexts = {AnalysisContext{}}; cutContexts.assign(exit + 1, 0);
    auto entryFor = [&](const Region &r) {
      if (r.kind == Region::Operation) return r.operation;
      const std::size_t id = sites.size(); sites.emplace_back(); return id;
    };
    if (p.body.kind == Region::Sequence && p.body.children.empty()) {
      for (std::size_t i = 0; i < exit; ++i) sites[i].successors = {i+1};
      return std::size_t(0);
    }
    struct Task { const Region *region; std::size_t entry, next, context; };
    const std::size_t root = entryFor(p.body);
    std::vector<Task> tasks{{&p.body, root, exit, 0}};
    auto contextFor = [&](AnalysisContext::Kind kind, const Task &task) {
      const auto id = contexts.size();
      contexts.push_back({kind, task.context, task.entry}); return id;
    };
    // Build once with an explicit stack. No recursive nested-loop solving.
    while (!tasks.empty()) {
      const auto task = tasks.back(); tasks.pop_back();
      const auto &r = *task.region;
      if (r.kind == Region::Operation) {
        sites[task.entry].successors = {task.next};
        cutContexts[task.entry] = task.context;
      } else if (r.kind == Region::Sequence) {
        std::vector<std::size_t> children;
        for (const auto &child : r.children) children.push_back(entryFor(child));
        sites[task.entry].successors = {children.empty() ? task.next : children.front()};
        for (std::size_t i = 0; i < children.size(); ++i)
          tasks.push_back({&r.children[i], children[i],
                           i+1 < children.size() ? children[i+1] : task.next, task.context});
      } else if (r.kind == Region::Choice) {
        const auto yes = entryFor(r.children[0]), no = entryFor(r.children[1]);
        sites[task.entry].successors = {yes, no};
        tasks.push_back({&r.children[0], yes, task.next, contextFor(AnalysisContext::ThenArm, task)});
        tasks.push_back({&r.children[1], no, task.next, contextFor(AnalysisContext::ElseArm, task)});
      } else if (r.kind == Region::For) {
        const auto body = entryFor(r.children[0]);
        sites[task.entry].successors = {body, task.next}; // preserve zero trips
        tasks.push_back({&r.children[0], body, task.entry, contextFor(AnalysisContext::ForBody, task)});
      } else {
        const auto before = entryFor(r.children[0]), after = entryFor(r.children[1]);
        const auto decision = sites.size(); sites.emplace_back();
        sites[task.entry].successors = {before}; // before ALWAYS executes
        sites[decision].successors = {after, task.next};
        tasks.push_back({&r.children[0], before, decision, contextFor(AnalysisContext::WhileBefore, task)});
        tasks.push_back({&r.children[1], after, before, contextFor(AnalysisContext::WhileAfter, task)});
      }
    }
    return root;
  }

  bool solve(std::size_t root) {
    const auto exit = p.operations.size();
    // nullopt is unreachable bottom; it is NOT fresh quiescent input.
    incoming.clear(); incoming.resize(sites.size());
    incoming[root].emplace(p.operations.size(), keyIds.size());
    std::deque<std::size_t> queue{root};
    std::vector<bool> queued(sites.size()); queued[root] = true;
    while (!queue.empty()) {
      const auto id = queue.front(); queue.pop_front(); queued[id] = false;
      State next = *incoming[id]; ++evaluations; charge(1);
      // Discovery computes the total conservative transfer. Acceptance checks
      // run only after the entire static worklist has stabilized.
      if (id < exit) {
        if (!operation(next, id, false, false)) return false;
      } else if (id == exit) {
        if (!cut(next, exit, false)) return false;
      }
      for (auto target : sites[id].successors) {
        ++merges;
        bool changed;
        if (!incoming[target]) { incoming[target] = next; changed = true; }
        else changed = incoming[target]->join(next);
        if (changed && !queued[target]) { queue.push_back(target); queued[target] = true; }
      }
    }
    return true;
  }
  BoundaryFacts snapshot(const State &s) const {
    BoundaryFacts out; out.pending = s.pending;
    for (const auto &t : s.tokens)
      out.events.push_back({t.occupancy, t.valid, t.remainder,
                            t.acknowledgments, t.consumedAt});
    return out;
  }
  std::vector<ProtocolObligation> preconditions(const State &s, Cut at,
                                               std::size_t index) const {
    const auto &c = commands[at][index];
    std::vector<ProtocolObligation> out;
    if (c.kind != Command::Publish && c.kind != Command::Acquire) return out;
    const auto &t = s.tokens[keyIds.at({c.source, c.observer, c.key})];
    auto add = [&](ProtocolObligation::Kind kind, const char *why) {
      out.push_back({kind, at, index, cutContexts[at],
                     {c.source, c.observer, c.key, false}, why});
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
  void requirements(const State &s, Cut id, AnalysisResult &out) const {
    const auto observer = unsigned(p.operations[id].pipe);
    std::set<std::tuple<std::size_t, unsigned, CompletionRequirement::Kind>> seen;
    for (const auto &access : p.operations[id].accesses)
      for (const auto &other : byCell[access.cell]) {
        if (!s.pending[observer][other.first]) continue;
        auto add = [&](CompletionRequirement::Kind kind, Property property) {
          if (!seen.emplace(other.first, access.cell, kind).second) return;
          out.residuals.push_back({kind, {other.first, id, access.cell, property},
              other.second, access, cutContexts[other.first], cutContexts[id]});
        };
        if (other.second.write && access.read)
          add(CompletionRequirement::RAW, Property::ByteCompletion);
        if (other.second.read && access.write)
          add(CompletionRequirement::WAR, Property::ByteCompletion);
        if (other.second.write && access.write)
          add(CompletionRequirement::WAW, Property::ByteCompletion);
        if (p.cells[access.cell].exclusive)
          add(CompletionRequirement::ExclusiveResource, Property::ResourceExclusion);
      }
  }

public:
  Transfer(const Program &program, const Commands &actual)
      : p(program), commands(actual), byCell(p.cells.size()) {
    for (const auto &at : commands)
      for (const auto &c : at)
        if (c.kind == Command::Publish || c.kind == Command::Acquire) {
          Key key{c.source, c.observer, c.key};
          if (!keyIds.count(key)) keyIds[key] = keyIds.size();
        }
    for (std::size_t i = 0; i < p.operations.size(); ++i)
      for (const auto &a : p.operations[i].accesses) byCell[a.cell].push_back({i, a});
  }
  std::size_t siteCount() const { return sites.size(); }
  uint64_t evaluationCount() const { return evaluations; }
  uint64_t mergeCount() const { return merges; }
  uint64_t workCount() const { return work; }

  AnalysisResult inspect(AnalysisOptions options) {
    AnalysisResult out;
    failure = {}; work = evaluations = merges = 0;
    const auto root = buildControl(), exit = p.operations.size();
    suppressed.resize(commands.size());
    for (std::size_t at = 0; at < commands.size(); ++at)
      suppressed[at].assign(commands[at].size(), false);
    std::map<std::pair<Cut, std::size_t>, std::vector<ProtocolObligation>> rejected;
    // Proof discovery is provisional. Remove certificates whose preconditions
    // fail at the invariant, poison only their key, and invalidate/recompute all
    // affected completion. Each repeating pass disables at least one NEW command:
    // at most event-command-count + 1 solves; no numerical work allowance.
    while (true) {
      ++out.stats.certificationPasses;
      if (!solve(root)) { out.reason = failure.reason; return out; }
      bool changed = false;
      for (Cut at = 0; at <= exit; ++at) {
        if (!incoming[at]) continue;
        State state = *incoming[at];
        for (std::size_t i = 0; i < commands[at].size(); ++i) {
          if (!suppressed[at][i]) {
            auto problems = preconditions(state, at, i);
            if (!problems.empty()) {
              suppressed[at][i] = true; changed = true;
              rejected[{at, i}] = std::move(problems);
              ++out.stats.suppressedCommands;
            }
          }
          (void)command(state, at, i, false);
        }
      }
      if (!changed) break;
    }
    out.contexts = contexts;
    out.keys.resize(keyIds.size());
    for (const auto &item : keyIds)
      out.keys[item.second] = {std::get<0>(item.first), std::get<1>(item.first),
                              std::get<2>(item.first), false};
    for (Cut at = 0; at <= exit; ++at) {
      CutFacts facts; facts.cut = at; facts.context = cutContexts[at];
      facts.reachable = bool(incoming[at]);
      if (!facts.reachable) { out.cuts.push_back(std::move(facts)); continue; }
      State state = *incoming[at];
      if (options.captureStates) facts.incoming = snapshot(state);
      for (std::size_t i = 0; i < commands[at].size(); ++i) {
        out.commands.push_back({at, i, cutContexts[at], commands[at][i], !suppressed[at][i]});
        if (suppressed[at][i]) {
          // Prefer diagnostics at the final conservative invariant. Retain the
          // invalidated certificate's cause if it is no longer locally visible.
          auto problems = preconditions(state, at, i);
          if (problems.empty()) problems = rejected.at({at, i});
          out.protocol.insert(out.protocol.end(), problems.begin(), problems.end());
        }
        (void)command(state, at, i, false);
      }
      if (options.captureStates) facts.beforeIssue = snapshot(state);
      if (at < exit) {
        requirements(state, at, out);
        // Issue without executing the cut twice: same fresh-effect primitive.
        issue(state, at);
      } else {
        for (const auto &item : keyIds) {
          if (state.tokens[item.second].occupancy != 1)
            out.protocol.push_back({ProtocolObligation::UnconsumedAtExit, at,
                NoAnalysisId, 0, out.keys[item.second],
                "event is not empty at required invocation exit"});
        }
        if (p.invocation.retirement == Program::InvocationContract::DrainAllAtReturn)
          for (std::size_t i = 0; i < exit; ++i) {
            auto q = unsigned(p.operations[i].pipe);
            if (state.pending[q][i]) out.retirement.push_back({i, Pipe(q)});
          }
      }
      if (options.captureStates) facts.outgoing = snapshot(state);
      out.cuts.push_back(std::move(facts));
    }
    out.stats.staticSites = sites.size(); out.stats.siteEvaluations = evaluations;
    out.stats.merges = merges; out.stats.work = work;
    out.complete = true;
    if (!out.protocol.empty()) out.reason = out.protocol.front().reason;
    else if (!out.residuals.empty()) out.reason = "uncovered original completion requirements";
    else if (!out.retirement.empty()) out.reason = "outstanding payload at required retirement";
    return out;
  }

  Failure run(bool checkPayload = true, bool checkProtocol = true) {
    suppressed.clear();
    failure = {}; work = evaluations = merges = 0;
    const std::size_t root = buildControl(), exit = p.operations.size();
    if (!solve(root)) return failure;
    // Each original phase is checked once at its invariant. Checking is not a
    // second recursive solve, and no optimistic discovery result is accepted.
    for (std::size_t id = 0; id < exit; ++id) {
      if (!incoming[id]) continue;
      State state = *incoming[id];
      if (!operation(state, id, checkPayload, checkProtocol)) return failure;
    }
    if (!incoming[exit]) {
      failure.kind = Failure::Invalid;
      failure.reason = "original invocation exit is unreachable in structural control";
      return failure;
    }
    State state = *incoming[exit];
    if (!cut(state, exit, checkProtocol)) return failure;
    if (checkProtocol)
      for (const auto &t : state.tokens)
        if (t.occupancy != 1) {
          failure.kind = Failure::Occupancy; failure.cut = exit;
          failure.reason = "unconsumed event at invocation exit"; return failure;
        }
    if (p.invocation.retirement == Program::InvocationContract::DrainAllAtReturn)
      for (unsigned q = 0; q < PipeCount; ++q)
        for (std::size_t i = 0; i < p.operations.size(); ++i)
          if (unsigned(p.operations[i].pipe) == q && state.pending[q][i]) {
            failure.kind = Failure::Retirement; failure.cut = exit;
            failure.reason = "outstanding payload at required retirement"; return failure;
          }
    return failure;
  }
};
} // namespace mlir::pto::oahs::detail
#endif
