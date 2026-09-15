// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-2.0
#ifndef PTO_OAHS_TRANSFER_H
#define PTO_OAHS_TRANSFER_H

#include "PTO/Transforms/OAHS/Plan.h"
#include <algorithm>
#include <array>
#include <map>
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
    for (auto &p : pending) p.resize(operations);
    for (auto &t : tokens) {
      t.remainder.assign(operations, 1);
      t.acknowledgments.resize(keys);
    }
  }
  bool operator==(const State &b) const {
    return pending == b.pending && tokens == b.tokens;
  }
  void join(const State &b) {
    for (unsigned q = 0; q < PipeCount; ++q)
      for (std::size_t i = 0; i < pending[q].size(); ++i)
        pending[q][i] |= b.pending[q][i];
    for (std::size_t e = 0; e < tokens.size(); ++e) {
      auto &a = tokens[e]; const auto &other = b.tokens[e];
      a.occupancy |= other.occupancy;
      a.consumedAt &= other.consumedAt;
      a.valid = a.valid && other.valid && a.occupancy == 2;
      for (std::size_t i = 0; i < a.remainder.size(); ++i)
        a.remainder[i] = a.valid ? (a.remainder[i] | other.remainder[i]) : 1;
      for (std::size_t f = 0; f < a.acknowledgments.size(); ++f)
        a.acknowledgments[f] = a.valid && a.acknowledgments[f] && other.acknowledgments[f];
    }
  }
};
struct Failure {
  enum Kind { None, Hazard, Rearm, Occupancy, Invalid, Budget, Retirement } kind = None;
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
  bool charge(uint64_t amount) {
    if (amount > p.verificationWorkLimit - std::min(work, p.verificationWorkLimit)) {
      failure.kind = Failure::Budget;
      failure.reason = "finite completion/protocol analysis allowance exhausted";
      return false;
    }
    work += amount;
    return true;
  }
  bool command(State &s, Cut cut, std::size_t index, bool checkProtocol) {
    const auto &c = commands[cut][index];
    if (!charge(1 + p.operations.size() + keyIds.size())) return false;
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
    if (!charge(1 + keyIds.size() + p.operations[id].accesses.size())) return false;
    if (checkPayload) {
      std::vector<Demand> missing;
      for (const auto &a : p.operations[id].accesses)
        for (const auto &other : byCell[a.cell]) {
          if (!charge(1)) return false;
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
  bool region(const Region &r, State &s, bool checkPayload, bool checkProtocol) {
    if (!charge(1)) return false;
    if (r.kind == Region::Operation)
      return operation(s, r.operation, checkPayload, checkProtocol);
    if (r.kind == Region::Sequence) {
      for (const auto &child : r.children)
        if (!region(child, s, checkPayload, checkProtocol)) return false;
      return true;
    }
    if (r.kind == Region::Choice) {
      State other = s;
      if (!region(r.children[0], s, checkPayload, checkProtocol) ||
          !region(r.children[1], other, checkPayload, checkProtocol)) return false;
      s.join(other); return true;
    }
    const State entry = s;
    State head = entry;
    // Discover an ascending invariant with no optimistic acceptance. Primitive
    // and payload preconditions are checked below in the stabilized invariant.
    while (true) {
      State next = head;
      if (!region(r.children[0], next, false, false)) return false;
      if (r.kind == Region::While && !region(r.children[1], next, false, false)) return false;
      next.join(head); next.join(entry);
      if (next == head) break;
      head = std::move(next);
    }
    State body = head;
    if (!region(r.children[0], body, checkPayload, checkProtocol)) return false;
    if (r.kind == Region::While) {
      State after = body;
      if (!region(r.children[1], after, checkPayload, checkProtocol)) return false;
      s = std::move(body); // condition-false exit is after the before-region
    } else {
      body.join(entry); s = std::move(body); // includes original zero-trip path
    }
    return true;
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
  Failure run(bool checkPayload = true, bool checkProtocol = true) {
    State state(p.operations.size(), keyIds.size());
    if (p.body.kind == Region::Sequence && p.body.children.empty()) {
      for (std::size_t i = 0; i < p.operations.size(); ++i)
        if (!operation(state, i, checkPayload, checkProtocol)) return failure;
    } else if (!region(p.body, state, checkPayload, checkProtocol)) return failure;
    if (!cut(state, p.operations.size(), checkProtocol)) return failure;
    if (checkProtocol)
      for (const auto &t : state.tokens)
        if (t.occupancy != 1) {
          failure.kind = Failure::Occupancy; failure.cut = p.operations.size();
          failure.reason = "unconsumed event at invocation exit"; return failure;
        }
    if (p.invocation.retirement == Program::InvocationContract::DrainAllAtReturn)
      for (unsigned q = 0; q < PipeCount; ++q)
        for (std::size_t i = 0; i < p.operations.size(); ++i)
          if (unsigned(p.operations[i].pipe) == q && state.pending[q][i]) {
            failure.kind = Failure::Retirement; failure.cut = p.operations.size();
            failure.reason = "outstanding payload at required retirement"; return failure;
          }
    return failure;
  }
};
} // namespace mlir::pto::oahs::detail
#endif
