// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_TRANSFER_H
#define PTO_OAHS_TRANSFER_H

#include "PTO/Transforms/OAHS/Plan.h"
#include <algorithm>
#include <array>
#include <map>
#include <deque>
#include <optional>
#include <limits>
#include <cstdlib>
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
// The construction-only mode may ignore protocol preconditions while proposing
// repairs, but final checking always enables all preconditions at invariants.
class Transfer {
  const Program &p;
  const Commands &commands;
  std::map<Key, unsigned> keyIds;
  std::vector<std::vector<std::pair<std::size_t, Access>>> byCell;
  uint64_t work = 0;
  Failure failure;
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
    auto fail = [&](Failure::Kind kind, const char *reason) {
      failure.kind = kind; failure.cut = cut; failure.command = index;
      failure.endpoint = c; failure.reason = reason; return false;
    };
    if (c.kind == Command::Publish) {
      if (checkProtocol && token.occupancy != 1)
        return fail(Failure::Occupancy, "publication is not empty on every represented path");
      if (checkProtocol && !(token.consumedAt & (1u << source)))
        return fail(Failure::Rearm, "event rearm lacks causal consumption");
      token.remainder = s.pending[source];
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == source) token.remainder[i] = 0;
      for (unsigned f = 0; f < s.tokens.size(); ++f)
        token.acknowledgments[f] = (s.tokens[f].consumedAt & (1u << source)) != 0;
      token.occupancy = 2; token.valid = true;
    } else {
      const bool usable = token.occupancy == 2 && token.valid;
      if (checkProtocol && !usable)
        return fail(Failure::Occupancy, "acquisition lacks one live matching publication on every path");
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
    for (auto &q : s.pending) q[id] = 1;
    for (auto &token : s.tokens)
      if (token.valid) token.remainder[id] = 1;
    if (p.target.synchronous[observer])
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (unsigned(p.operations[i].pipe) == observer) s.pending[observer][i] = 0;
    return true;
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
    auto entryFor = [&](const Region &r) {
      if (r.kind == Region::Operation) return r.operation;
      const std::size_t id = sites.size(); sites.emplace_back(); return id;
    };
    if (p.body.kind == Region::Sequence && p.body.children.empty()) {
      for (std::size_t i = 0; i < exit; ++i) sites[i].successors = {i+1};
      return std::size_t(0);
    }
    struct Task { const Region *region; std::size_t entry, next; };
    const std::size_t root = entryFor(p.body);
    std::vector<Task> tasks{{&p.body, root, exit}};
    // Build once with an explicit stack. No recursive nested-loop solving.
    while (!tasks.empty()) {
      const auto task = tasks.back(); tasks.pop_back();
      const auto &r = *task.region;
      if (r.kind == Region::Operation) {
        sites[task.entry].successors = {task.next};
      } else if (r.kind == Region::Sequence) {
        std::vector<std::size_t> children;
        for (const auto &child : r.children) children.push_back(entryFor(child));
        sites[task.entry].successors = {children.empty() ? task.next : children.front()};
        for (std::size_t i = 0; i < children.size(); ++i)
          tasks.push_back({&r.children[i], children[i],
                           i+1 < children.size() ? children[i+1] : task.next});
      } else if (r.kind == Region::Choice) {
        const auto yes = entryFor(r.children[0]), no = entryFor(r.children[1]);
        sites[task.entry].successors = {yes, no};
        tasks.push_back({&r.children[0], yes, task.next});
        tasks.push_back({&r.children[1], no, task.next});
      } else if (r.kind == Region::For) {
        const auto body = entryFor(r.children[0]);
        sites[task.entry].successors = {body, task.next}; // preserve zero trips
        tasks.push_back({&r.children[0], body, task.entry});
      } else {
        const auto before = entryFor(r.children[0]), after = entryFor(r.children[1]);
        const auto decision = sites.size(); sites.emplace_back();
        sites[task.entry].successors = {before}; // before ALWAYS executes
        sites[decision].successors = {after, task.next};
        tasks.push_back({&r.children[0], before, decision});
        tasks.push_back({&r.children[1], after, before});
      }
    }
    return root;
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

  Failure run(bool checkPayload = true, bool checkProtocol = true) {
    failure = {}; work = evaluations = merges = 0;
    const std::size_t root = buildControl(), exit = p.operations.size();
    // nullopt is unreachable bottom; it is NOT fresh quiescent input.
    std::vector<std::optional<State>> incoming(sites.size());
    incoming[root].emplace(p.operations.size(), keyIds.size());
    std::deque<std::size_t> queue{root};
    std::vector<bool> queued(sites.size()); queued[root] = true;
    while (!queue.empty()) {
      const auto id = queue.front(); queue.pop_front(); queued[id] = false;
      State next = *incoming[id]; ++evaluations; charge(1);
      // Discovery computes the total conservative transfer. Acceptance checks
      // run only after the entire static worklist has stabilized.
      if (id < exit) {
        if (!operation(next, id, false, false)) return failure;
      } else if (id == exit) {
        if (!cut(next, exit, false)) return failure;
      }
      for (auto target : sites[id].successors) {
        ++merges;
        bool changed;
        if (!incoming[target]) { incoming[target] = next; changed = true; }
        else changed = incoming[target]->join(next);
        if (changed && !queued[target]) { queue.push_back(target); queued[target] = true; }
      }
    }
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
