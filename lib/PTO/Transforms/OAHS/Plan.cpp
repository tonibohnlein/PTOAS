// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Plan.h"
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace mlir::pto::oahs {
namespace {
using Frontier = std::array<std::size_t, PipeCount>;
using Knowledge = std::array<Frontier, PipeCount>;
using Key = std::tuple<Pipe, Pipe, unsigned>;
using Outstanding = std::vector<std::set<std::size_t>>;
unsigned lane(Pipe p) { return unsigned(p); }
void join(Frontier &to, const Frontier &from) {
  for (unsigned p = 0; p < PipeCount; ++p)
    to[p] = std::max(to[p], from[p]);
}
bool validateRegion(const Program &p, const Region &region,
                    std::vector<unsigned> &seen, std::string &reason) {
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
bool straightLine(const Program &p) {
  if (p.body.kind == Region::Sequence && p.body.children.empty()) return true;
  if (p.body.kind != Region::Sequence ||
      p.body.children.size() != p.operations.size()) return false;
  for (std::size_t i = 0; i < p.body.children.size(); ++i)
    if (p.body.children[i].kind != Region::Operation ||
        p.body.children[i].operation != i)
      return false;
  return true;
}
void joinOutstanding(Outstanding &to, const Outstanding &from) {
  for (std::size_t cell = 0; cell < to.size(); ++cell)
    to[cell].insert(from[cell].begin(), from[cell].end());
}
bool conflicts(const Program &p, std::size_t producer,
               std::size_t consumer, unsigned cell) {
  bool producerRead = false, producerWrite = false;
  bool consumerRead = false, consumerWrite = false;
  for (const Access &access : p.operations[producer].accesses)
    if (access.cell == cell) {
      producerRead |= access.read;
      producerWrite |= access.write;
    }
  for (const Access &access : p.operations[consumer].accesses)
    if (access.cell == cell) {
      consumerRead |= access.read;
      consumerWrite |= access.write;
    }
  return (producerRead || producerWrite) && (consumerRead || consumerWrite) &&
         (producerWrite || consumerWrite || p.cells[cell].exclusive);
}
bool writesCell(const Program &p, std::size_t operation, unsigned cell) {
  for (const Access &access : p.operations[operation].accesses)
    if (access.cell == cell && access.write) return true;
  return false;
}
Outstanding transferRegion(const Program &p, const Region &region,
                           Outstanding state, std::vector<Demand> &result) {
  if (region.kind == Region::Operation) {
    const Operation &operation = p.operations[region.operation];
    std::set<unsigned> cells;
    for (const Access &access : operation.accesses) cells.insert(access.cell);
    for (unsigned cell : cells) {
      for (std::size_t producer : state[cell])
        if (conflicts(p, producer, region.operation, cell)) {
          if (writesCell(p, producer, cell) ||
              writesCell(p, region.operation, cell))
            result.push_back({producer, region.operation, cell,
                              Property::ByteCompletion});
          if (p.cells[cell].exclusive)
            result.push_back({producer, region.operation, cell,
                              Property::ResourceExclusion});
        }
      state[cell].insert(region.operation);
    }
    return state;
  }
  if (region.kind == Region::Sequence) {
    for (const Region &child : region.children)
      state = transferRegion(p, child, std::move(state), result);
    return state;
  }
  if (region.kind == Region::Choice) {
    Outstanding merged(p.cells.size());
    for (const Region &child : region.children)
      joinOutstanding(merged, transferRegion(p, child, state, result));
    return merged;
  }
  if (region.kind == Region::For) {
    const Outstanding entry = state;
    Outstanding head = entry;
    while (true) {
      Outstanding body = transferRegion(p, region.children.front(), head, result);
      Outstanding next = entry;
      joinOutstanding(next, body);
      if (next == head) return next;
      head = std::move(next);
    }
  }
  // While executes before once, then either exits or executes after and returns
  // to before. The finite producer sets make this fixed point terminate.
  const Outstanding entry = state;
  Outstanding head = entry;
  while (true) {
    Outstanding before =
        transferRegion(p, region.children[0], head, result);
    Outstanding after =
        transferRegion(p, region.children[1], before, result);
    Outstanding next = entry;
    joinOutstanding(next, after);
    if (next == head) return before;
    head = std::move(next);
  }
}
std::vector<Demand> structuralDemands(const Program &p) {
  std::vector<Demand> result;
  (void)transferRegion(p, p.body, Outstanding(p.cells.size()), result);
  std::sort(result.begin(), result.end(), [](const Demand &a, const Demand &b) {
    return std::tie(a.consumer, a.producer, a.cell, a.property) <
           std::tie(b.consumer, b.producer, b.cell, b.property);
  });
  result.erase(std::unique(result.begin(), result.end(),
      [](const Demand &a, const Demand &b) {
        return std::tie(a.consumer, a.producer, a.cell, a.property) ==
               std::tie(b.consumer, b.producer, b.cell, b.property);
      }), result.end());
  return result;
}
bool hasUnimplementedEffects(const Program &p) {
  if (!p.reservations.empty()) return true;
  for (const Operation &operation : p.operations)
    if (!operation.resources.empty() || !operation.visibility.empty() ||
        !operation.authoredEvents.empty() ||
        !operation.internalTransfers.empty())
      return true;
  return false;
}
bool valid(const Program &p, std::string &reason) {
  if (p.target.contract.empty()) {
    reason = "missing target contract";
    return false;
  }
  if (p.conservativeCompletion && p.conservativeReason.empty()) {
    reason = "conservative completion requires an explicit analysis reason";
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
std::vector<Demand> demands(const Program &p) {
  std::vector<Demand> result;
  // Reduce each witness to the minimal ordered conflict frontier: the latest
  // writer and every reader since it. A later write must follow all of those
  // readers; a later read must follow only the latest writer. This keeps the
  // representation linear apart from genuine multi-reader release state.
  struct Use { std::size_t operation; bool read; bool write; };
  std::vector<std::vector<Use>> uses(p.cells.size());
  for (std::size_t i = 0; i < p.operations.size(); ++i)
    for (const auto &a : p.operations[i].accesses) {
      auto &entries = uses[a.cell];
      if (!entries.empty() && entries.back().operation == i) {
        entries.back().read |= a.read;
        entries.back().write |= a.write;
      } else {
        entries.push_back({i, a.read, a.write});
      }
    }
  for (unsigned cell = 0; cell < uses.size(); ++cell) {
    const auto &entries = uses[cell];
    std::optional<std::size_t> lastWrite;
    std::vector<std::size_t> readers;
    std::optional<std::size_t> lastExclusive;
    for (const Use &entry : entries) {
      auto add = [&](std::size_t producer, Property property) {
        if (producer != entry.operation)
          result.push_back({producer, entry.operation, cell, property});
      };
      if (p.cells[cell].exclusive && lastExclusive)
        add(*lastExclusive, Property::ResourceExclusion);
      if (entry.write) {
        if (lastWrite) add(*lastWrite, Property::ByteCompletion);
        for (std::size_t reader : readers)
          add(reader, Property::ByteCompletion);
        readers.clear();
        lastWrite = entry.operation;
      } else if (entry.read) {
        if (lastWrite) add(*lastWrite, Property::ByteCompletion);
        readers.push_back(entry.operation);
      }
      if (p.cells[cell].exclusive) lastExclusive = entry.operation;
    }
  }
  std::sort(result.begin(), result.end(), [](const Demand &a, const Demand &b) {
    return std::tie(a.consumer, a.producer, a.cell, a.property) <
           std::tie(b.consumer, b.producer, b.cell, b.property);
  });
  result.erase(std::unique(result.begin(), result.end(), [](const Demand &a, const Demand &b) {
    return std::tie(a.consumer, a.producer, a.cell, a.property) ==
           std::tie(b.consumer, b.producer, b.cell, b.property);
  }), result.end());
  return result;
}
bool available(const Target &t, Pipe source, Pipe observer, unsigned key) {
  if (lane(source) >= PipeCount || lane(observer) >= PipeCount ||
      source == observer || !t.supported[lane(source)] ||
      !t.supported[lane(observer)])
    return false;
  const auto &keys = t.keys[lane(source)][lane(observer)];
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

// Adapted from the old PrefixState immutable-receipt semantics. A frontier uses
// original occurrence IDs here (one straight-line visit), preserving generation
// identity directly. No SET advances the source's issue/completion knowledge.
struct State {
  Knowledge known{};
  Frontier issued{};
  struct Token { Frontier receipt{}; bool live = false, used = false; };
  std::map<Key, Token> tokens;
  // Consumption knowledge is separate from payload completion. This verifier
  // supports causal key reuse, although the first allocator uses one-shot keys.
  std::array<std::set<Key>, PipeCount> consumed;
  std::map<Key, std::set<Key>> acknowledgments;

  bool command(const Target &t, const Command &c, std::string &reason) {
    if (c.kind == Command::BarrierAll) {
      if (!t.barrierAll) { reason = "unsupported all-pipeline barrier"; return false; }
      for (auto &k : known) join(k, issued);
      // A full barrier is not a token consumption operation.
      return true;
    }
    if (lane(c.source) >= PipeCount || !t.supported[lane(c.source)]) {
      reason = "unsupported command pipeline"; return false;
    }
    auto source = lane(c.source);
    if (c.kind == Command::Barrier) {
      if (!t.barriers[source]) { reason = "unsupported pipeline barrier"; return false; }
      known[source][source] = issued[source];
      return true;
    }
    if ((c.kind != Command::Publish && c.kind != Command::Acquire) ||
        !available(t, c.source, c.observer, c.key)) {
      reason = "unsupported event direction or key"; return false;
    }
    Key key{c.source, c.observer, c.key};
    auto &token = tokens[key];
    if (c.kind == Command::Publish) {
      if (token.live || (token.used && !consumed[source].count(key))) {
        reason = "event rearm lacks causal consumption"; return false;
      }
      token.receipt = known[source];
      token.receipt[source] = issued[source];
      token.live = token.used = true;
      acknowledgments[key] = consumed[source];
    } else {
      if (!token.live) { reason = "unmatched acquisition"; return false; }
      join(known[lane(c.observer)], token.receipt);
      token.live = false;
      // Invalidate stale consumption knowledge for the previous generation.
      for (auto &k : consumed) k.erase(key);
      for (auto &entry : acknowledgments) entry.second.erase(key);
      auto &observer = consumed[lane(c.observer)];
      observer.insert(acknowledgments[key].begin(), acknowledgments[key].end());
      observer.insert(key);
    }
    return true;
  }
  void issue(const Program &p, std::size_t i) {
    auto source = lane(p.operations[i].pipe);
    issued[source] = i + 1;
    if (p.target.synchronous[source]) known[source][source] = i + 1;
  }
};
} // namespace

Result analyze(const Program &p) {
  Result result;
  result.success = valid(p, result.reason);
  return result;
}

Result verify(const Program &p, const Commands &commands) {
  Result result;
  if (!valid(p, result.reason)) return result;
  if (hasUnimplementedEffects(p)) {
    result.reason = "resource, visibility, or authored effects are not implemented";
    return result;
  }
  if (p.conservativeCompletion) {
    if (commands.size() != p.operations.size() + 1) {
      result.reason = "command anchors do not match conservative phases";
      return result;
    }
    if (!p.target.barrierAll) {
      result.reason = "analysis-budget completion is unsupported by the target";
      return result;
    }
    for (const auto &cut : commands)
      for (const Command &command : cut)
        if (command.kind != Command::BarrierAll) {
          result.reason = "command is outside the verified analysis-budget realization";
          return result;
        }
    for (std::size_t cut = 1; cut < p.operations.size(); ++cut)
      if (commands[cut].empty()) {
        result.reason = "missing conservative completion before physical phase " +
                        std::to_string(cut);
        return result;
      }
    if (p.invocation.retirement ==
            Program::InvocationContract::DrainAllAtReturn &&
        commands.back().empty()) {
      result.reason = "missing conservative completion at required retirement";
      return result;
    }
    result.success = true;
    return result;
  }
  if (!straightLine(p)) {
    if (commands.size() != p.operations.size() + 1) {
      result.reason = "command anchors do not match original physical phases";
      return result;
    }
    if (!p.target.barrierAll) {
      result.reason = "structured conservative completion is unsupported by the target";
      return result;
    }
    for (const auto &cut : commands)
      for (const Command &command : cut)
        if (command.kind != Command::BarrierAll) {
          result.reason = "structured command is outside the verified conservative realization";
          return result;
        }
    for (const Demand &demand : structuralDemands(p))
      if (std::none_of(commands[demand.consumer].begin(),
                       commands[demand.consumer].end(), [](const Command &command) {
            return command.kind == Command::BarrierAll;
          })) {
        result.reason = "uncovered structured demand at physical phase " +
                        std::to_string(demand.consumer);
        return result;
      }
    if (p.invocation.retirement ==
            Program::InvocationContract::DrainAllAtReturn &&
        std::none_of(commands.back().begin(), commands.back().end(),
                     [](const Command &command) {
          return command.kind == Command::BarrierAll;
        })) {
      result.reason = "outstanding structured payload at required retirement";
      return result;
    }
    result.success = true;
    return result;
  }
  if (commands.size() != p.operations.size() + 1) {
    result.reason = "command cuts do not match original operations"; return result;
  }
  const auto obligations = demands(p);
  State state;
  for (Cut cut = 0; cut < commands.size(); ++cut) {
    for (const auto &c : commands[cut])
      if (!state.command(p.target, c, result.reason)) return result;
    if (cut == p.operations.size()) break;
    auto observer = lane(p.operations[cut].pipe);
    for (const auto &d : obligations)
      if (d.consumer == cut &&
          state.known[observer][lane(p.operations[d.producer].pipe)] < d.producer + 1) {
        result.reason = "uncovered original demand at operation " + std::to_string(cut);
        return result;
      }
    state.issue(p, cut);
  }
  for (const auto &entry : state.tokens)
    if (entry.second.live) { result.reason = "unconsumed event at lifetime exit"; return result; }
  if (p.invocation.retirement ==
      Program::InvocationContract::DrainAllAtReturn)
    for (unsigned source = 0; source < PipeCount; ++source)
      if (state.known[source][source] < state.issued[source]) {
        result.reason = "outstanding payload at required retirement"; return result;
      }
  result.success = true;
  return result;
}

Result construct(const Program &p) {
  Result result;
  if (!valid(p, result.reason)) return result;
  if (hasUnimplementedEffects(p)) {
    result.reason = "resource, visibility, or authored effects are not implemented";
    return result;
  }
  if (p.conservativeCompletion) {
    if (!p.target.barrierAll) {
      result.reason = "analysis-budget completion is unsupported by the target";
      return result;
    }
    result.commands.resize(p.operations.size() + 1);
    for (std::size_t cut = 1; cut < p.operations.size(); ++cut) {
      result.commands[cut].push_back({Command::BarrierAll});
      ++result.conservativeBarriers;
    }
    if (p.invocation.retirement ==
        Program::InvocationContract::DrainAllAtReturn) {
      result.commands.back().push_back({Command::BarrierAll});
      ++result.conservativeBarriers;
    }
    Result checked = verify(p, result.commands);
    result.success = checked.success;
    result.reason = checked.reason;
    return result;
  }
  if (!straightLine(p)) {
    if (!p.target.barrierAll) {
      result.reason = "structured conservative completion is unsupported by the target";
      return result;
    }
    result.demands = structuralDemands(p);
    result.commands.resize(p.operations.size() + 1);
    std::set<std::size_t> consumers;
    for (const Demand &demand : result.demands)
      consumers.insert(demand.consumer);
    for (std::size_t consumer : consumers) {
      result.commands[consumer].push_back({Command::BarrierAll});
      ++result.conservativeBarriers;
    }
    if (p.invocation.retirement ==
        Program::InvocationContract::DrainAllAtReturn)
      result.commands.back().push_back({Command::BarrierAll});
    Result checked = verify(p, result.commands);
    result.success = checked.success;
    result.reason = checked.reason;
    return result;
  }
  result.demands = demands(p);
  result.commands.resize(p.operations.size() + 1);
  Knowledge known{};
  // Snapshot of the source's completion prefix after each original operation.
  // Later source issue cannot extend an earlier immutable publication receipt.
  std::vector<Frontier> prefixes;
  Frontier issued{};
  for (Cut cut = 0; cut < p.operations.size(); ++cut) {
    auto observer = lane(p.operations[cut].pipe);
    Frontier required{};
    for (const auto &d : result.demands)
      if (d.consumer == cut) {
        auto source = lane(p.operations[d.producer].pipe);
        required[source] = std::max(required[source], d.producer + 1);
      }
    for (unsigned source = 0; source < PipeCount; ++source) {
      if (known[observer][source] >= required[source]) continue;
      if (source == observer) {
        if (p.target.barriers[source]) {
          result.commands[cut].push_back({Command::Barrier, Pipe(source)});
          known[source][source] = issued[source];
        } else if (p.target.barrierAll) {
          result.commands[cut].push_back({Command::BarrierAll});
          for (auto &k : known) join(k, issued);
        } else { result.reason = "no same-pipeline completion mechanism"; return result; }
      } else if (!p.target.keys[source][observer].empty()) {
        result.handoffs.push_back({Pipe(source), Pipe(observer), required[source], cut});
        join(known[observer], prefixes[required[source] - 1]);
      } else if (p.target.barrierAll) {
        result.commands[cut].push_back({Command::BarrierAll});
        for (auto &k : known) join(k, issued);
      } else { result.reason = "no completion route"; return result; }
    }
    issued[observer] = cut + 1;
    auto receipt = known[observer];
    receipt[observer] = issued[observer];
    prefixes.push_back(receipt);
    if (p.target.synchronous[observer]) known[observer][observer] = cut + 1;
  }
  // Global one-shot allocation: never invent consumption causality from source
  // order. Scarcity first tries deterministic key reuse and accepts it only
  // when the reconstructed verifier proves consumption-before-rearm. Remaining
  // scarcity uses an explicitly supported full barrier at the demand cut.
  std::map<std::pair<Pipe, Pipe>, std::set<unsigned>> used;
  std::vector<Handoff> pending;
  for (const auto &h : result.handoffs) {
    auto &population = used[{h.source, h.observer}];
    const auto &keys = p.target.keys[lane(h.source)][lane(h.observer)];
    auto key = std::find_if(keys.begin(), keys.end(), [&](unsigned k) { return !population.count(k); });
    if (key == keys.end()) {
      if (!p.target.barrierAll) { result.reason = "event allocation exhausted"; return result; }
      unsigned marker = ~unsigned(pending.size());
      result.commands[h.acquisition].push_back(
          {Command::BarrierAll, Pipe::S, Pipe::S, marker});
      pending.push_back(h);
      ++result.scarcityBarriers;
      continue;
    }
    population.insert(*key);
    // Publish immediately after its producer, before commands at that next cut.
    result.commands[h.publication].insert(result.commands[h.publication].begin(),
        {Command::Publish, h.source, h.observer, *key});
    result.commands[h.acquisition].push_back({Command::Acquire, h.source, h.observer, *key});
  }
  if (p.invocation.retirement ==
      Program::InvocationContract::DrainAllAtReturn)
    result.commands.back().push_back({Command::BarrierAll});
  for (unsigned i = 0; i < pending.size(); ++i) {
    const Handoff &handoff = pending[i];
    unsigned marker = ~i;
    const auto &keys = p.target.keys[lane(handoff.source)][lane(handoff.observer)];
    for (unsigned key : keys) {
      Commands trial = result.commands;
      auto &atConsumer = trial[handoff.acquisition];
      auto barrier = std::find_if(atConsumer.begin(), atConsumer.end(),
          [&](const Command &command) {
            return command.kind == Command::BarrierAll && command.key == marker;
          });
      if (barrier == atConsumer.end()) break;
      atConsumer.erase(barrier);
      trial[handoff.publication].insert(trial[handoff.publication].begin(),
          {Command::Publish, handoff.source, handoff.observer, key});
      atConsumer.push_back(
          {Command::Acquire, handoff.source, handoff.observer, key});
      if (!verify(p, trial).success) continue;
      result.commands = std::move(trial);
      --result.scarcityBarriers;
      break;
    }
  }
  auto checked = verify(p, result.commands);
  result.success = checked.success;
  result.reason = checked.reason;
  return result;
}
} // namespace mlir::pto::oahs
