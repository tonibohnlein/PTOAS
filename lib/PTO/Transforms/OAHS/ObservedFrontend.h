// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_OBSERVED_FRONTEND_H
#define PTO_OAHS_OBSERVED_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <functional>
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::oahs {
namespace observed_detail {
inline uint64_t addMod(uint64_t a, uint64_t b, uint64_t m) {
  return a >= m - b ? a - (m - b) : a + b;
}
inline uint64_t mulMod(uint64_t a, uint64_t b, uint64_t m) {
  a %= m;
  uint64_t out = 0;
  while (b) {
    if (b & 1)
      out = addMod(out, a, m);
    b >>= 1;
    if (b)
      a = addMod(a, a, m);
  }
  return out;
}
inline bool same(const OriginalObservation &a, const OriginalObservation &b) {
  if (a.anchor != b.anchor || a.available != b.available ||
      a.atoms.size() != b.atoms.size())
    return false;
  for (std::size_t i = 0; i < a.atoms.size(); ++i) {
    const auto &x = a.atoms[i], &y = b.atoms[i];
    if (x.kind != y.kind || x.owner != y.owner || x.parameter != y.parameter ||
        x.value != y.value)
      return false;
  }
  return true;
}
} // namespace observed_detail
ObservedImport makePeriodicLoop(const Program &body, unsigned period,
                                const std::vector<PeriodicAccess> &bindings) {
  ObservedImport out;
  auto fail = [&](const char *why) {
    out.reason = why;
    return out;
  };
  const auto declared = validateProgram(body);
  if (!declared.success) {
    out.reason = declared.reason;
    return out;
  }
  if (!period || period == std::numeric_limits<unsigned>::max() ||
      body.observed || !body.body.children.empty() ||
      body.body.kind != Region::Sequence || body.operations.empty())
    return fail("periodic frontend requires a nonempty flat normalized body "
                "and a positive explicit period");
  if (body.operations.size() > body.operations.max_size() / period)
    return fail(
        "periodic phase population exceeds intrinsic container capacity");
  std::set<std::pair<std::size_t, std::size_t>> bound;
  for (const auto &b : bindings) {
    if (b.operation >= body.operations.size() ||
        b.access >= body.operations[b.operation].accesses.size() ||
        b.cells.empty() || !bound.emplace(b.operation, b.access).second)
      return fail("invalid or duplicate periodic effect binding");
    for (auto c : b.cells)
      if (c >= body.cells.size() || body.cells[c].unknownRange)
        return fail("periodic binding requires declared exact cell identities");
    if (observed_detail::mulMod(period, b.stride, b.cells.size()) != 0)
      return fail("supplied period does not preserve a selector");
  }
  out.program = body;
  out.program.operations.clear();
  out.program.body = {};
  for (unsigned residue = 0; residue < period; ++residue)
    for (std::size_t i = 0; i < body.operations.size(); ++i) {
      auto op = body.operations[i];
      op.original = i;
      for (const auto &b : bindings)
        if (b.operation == i) {
          const auto k = observed_detail::addMod(
              observed_detail::mulMod(residue, b.stride, b.cells.size()),
              b.offset % b.cells.size(), b.cells.size());
          op.accesses[b.access].cell = b.cells[k];
        }
      out.program.operations.push_back(std::move(op));
      out.originalPhases.push_back(i);
    }
  ObservedControl q;
  q.qualification =
      "normalized-nonnegative-counted-loop/exact-affine-modular-effects-v1";
  q.scopes = {{0, NoControlId, NoControlId}, {3, 0, 1}};
  q.observations = {{body.operations.size() + 1, {}, true},
                    {body.operations.size() + 2, {}, true}};
  q.sites.resize(2);
  q.entry = 1;
  q.exit = 0;
  q.sites[0].observation = 0;
  q.sites[1].observation = 1;
  using State =
      std::tuple<unsigned, unsigned,
                 unsigned>; // residue, capped elapsed, capped remaining
  std::map<State, std::size_t> headers;
  std::vector<State> work;
  auto header = [&](State state) {
    auto found = headers.find(state);
    if (found != headers.end())
      return found->second;
    const auto id = q.sites.size();
    q.sites.emplace_back();
    headers[state] = id;
    work.push_back(state);
    return id;
  };
  for (unsigned remaining = 0; remaining <= period + 1; ++remaining) {
    const auto target = header({0, 0, remaining});
    q.sites[q.entry].successors.push_back(target);
  }
  std::map<std::tuple<std::size_t, unsigned, bool, bool>, std::size_t>
      observations;
  for (std::size_t index = 0; index < work.size(); ++index) {
    const auto [residue, elapsed, remaining] = work[index];
    const auto at = headers.at(work[index]);
    if (!remaining) {
      q.sites[at].successors = {q.exit};
      continue;
    }
    std::vector<std::size_t> chain;
    for (std::size_t j = 0; j <= body.operations.size(); ++j) {
      const auto key =
          std::make_tuple(j, residue, elapsed >= period, remaining > period);
      auto it = observations.find(key);
      std::size_t observation;
      if (it == observations.end()) {
        observation = q.observations.size();
        observations[key] = observation;
        q.observations.push_back(
            {j,
             {{ObservationAtom::LoopResidue, 0, period, residue},
              {ObservationAtom::LoopHasPrevious, 0, period,
               uint64_t(elapsed >= period)},
              {ObservationAtom::LoopHasNext, 0, period,
               uint64_t(remaining > period)}},
             true});
      } else
        observation = it->second;
      const auto node = q.sites.size();
      q.sites.emplace_back();
      chain.push_back(node);
      q.sites[node].observation = observation;
      q.sites[node].context = 1;
      if (j < body.operations.size())
        q.sites[node].operation = residue * body.operations.size() + j;
    }
    q.sites[at].successors = {chain.front()};
    for (std::size_t j = 0; j + 1 < chain.size(); ++j)
      q.sites[chain[j]].successors = {chain[j + 1]};
    const auto nextResidue = (residue + 1) % period,
               nextElapsed = std::min(elapsed + 1, period);
    const auto one = header({nextResidue, nextElapsed,
                             remaining == period + 1 ? period : remaining - 1});
    q.sites[chain.back()].successors.push_back(one);
    q.sites[chain.back()].backedgeOwners.push_back(1);
    if (remaining == period + 1) {
      const auto many = header({nextResidue, nextElapsed, period + 1});
      q.sites[chain.back()].successors.push_back(many);
      q.sites[chain.back()].backedgeOwners.push_back(1);
    }
  }
  ObservedLoop loop{0, q.entry, q.exit, {}};
  for (std::size_t site = 2; site < q.sites.size(); ++site)
    loop.sites.push_back(site);
  q.loops.push_back(std::move(loop));
  out.program.observed = std::move(q);
  const auto valid = validateProgram(out.program);
  out.success = valid.success;
  out.reason = valid.reason;
  return out;
}
ObservedImport addStructuredBoundaryCuts(const Program &input) {
  ObservedImport out;
  auto valid = validateProgram(input);
  if (!valid.success) {
    out.reason = valid.reason;
    return out;
  }
  if (input.observed) {
    out.reason =
        "structured boundary frontend expects the original region tree";
    return out;
  }
  out.program = input;
  ObservedControl q;
  q.qualification = "qualified-original-region-entry-exit-cuts-v1";
  q.scopes = {{0, NoControlId, NoControlId}};
  auto node = [&](std::size_t operation, std::size_t context, bool legal) {
    const auto id = q.sites.size();
    q.sites.emplace_back();
    q.sites[id].operation = operation;
    q.sites[id].context = context;
    if (legal) {
      q.sites[id].observation = q.observations.size();
      q.observations.push_back({id, {}, true});
    }
    return id;
  };
  auto scope = [&](unsigned kind, std::size_t parent, std::size_t owner) {
    const auto id = q.scopes.size();
    q.scopes.push_back({kind, parent, owner});
    return id;
  };
  std::function<std::pair<std::size_t, std::size_t>(const Region &,
                                                    std::size_t)>
      build;
  build = [&](const Region &r,
              std::size_t context) -> std::pair<std::size_t, std::size_t> {
    const auto begin = node(
        r.kind == Region::Operation ? r.operation : NoControlId, context, true);
    if (r.kind == Region::Operation)
      return {begin, begin};
    if (r.kind == Region::Sequence) {
      auto last = begin;
      for (const auto &child : r.children) {
        auto block = build(child, context);
        q.sites[last].successors = {block.first};
        last = block.second;
      }
      const auto end = node(NoControlId, context, true);
      q.sites[last].successors = {end};
      return {begin, end};
    }
    if (r.kind == Region::Choice) {
      const auto yes = build(r.children[0], scope(1, context, begin)),
                 no = build(r.children[1], scope(2, context, begin));
      const auto end = node(NoControlId, context, true);
      q.sites[begin].successors = {yes.first, no.first};
      q.sites[yes.second].successors = {end};
      q.sites[no.second].successors = {end};
      return {begin, end};
    }
    if (r.kind == Region::For) {
      const auto header = node(NoControlId, context, false);
      const auto body = build(r.children[0], scope(3, context, begin));
      const auto end = node(NoControlId, context, true);
      q.sites[begin].successors = {header};
      q.sites[header].successors = {body.first, end};
      q.sites[body.second].successors = {header};
      q.sites[body.second].backedgeOwners = {begin};
      return {begin, end};
    }
    const auto before = build(r.children[0], scope(4, context, begin));
    const auto decision = node(NoControlId, context, false);
    const auto after = build(r.children[1], scope(5, context, begin));
    const auto end = node(NoControlId, context, true);
    q.sites[begin].successors = {before.first};
    q.sites[before.second].successors = {decision};
    q.sites[decision].successors = {after.first, end};
    q.sites[after.second].successors = {before.first};
    q.sites[after.second].backedgeOwners = {begin};
    return {begin, end};
  };
  Region tree = input.body;
  if (tree.kind == Region::Sequence && tree.children.empty())
    for (std::size_t i = 0; i < input.operations.size(); ++i)
      tree.children.push_back({Region::Operation, {}, i});
  const auto root = build(tree, 0);
  q.entry = root.first;
  q.exit = node(NoControlId, 0, true);
  q.sites[root.second].successors = {q.exit};
  out.program.observed = std::move(q);
  for (std::size_t i = 0; i < input.operations.size(); ++i)
    out.originalPhases.push_back(i);
  valid = validateProgram(out.program);
  out.success = valid.success;
  out.reason = valid.reason;
  return out;
}
ObservedImport refineCountedLoop(const Program &input,
                                 const CountedLoopRegion &loop) {
  ObservedImport out;
  auto fail = [&](const char *why) {
    out.reason = why;
    return out;
  };
  auto valid = validateProgram(input);
  if (!valid.success) {
    out.reason = valid.reason;
    return out;
  }
  if (!input.observed || !loop.period ||
      loop.period == std::numeric_limits<unsigned>::max())
    return fail("normalized loop refinement requires a valid finite period and "
                "observed control");
  const auto &old = *input.observed;
  const auto size = old.sites.size();
  if (loop.owner >= size || loop.header >= size || loop.bodyEntry >= size ||
      loop.continuation >= size || loop.bodySites.empty())
    return fail("invalid normalized loop boundary");
  if (old.sites[loop.owner].successors !=
          std::vector<std::size_t>{loop.header} ||
      old.sites[loop.header].operation != NoControlId ||
      old.sites[loop.header].observation != NoControlId ||
      old.sites[loop.header].successors !=
          std::vector<std::size_t>{loop.bodyEntry, loop.continuation})
    return fail("normalized loop control is not the supplied "
                "preheader/header/body/exit");
  std::set<std::size_t> members(loop.bodySites.begin(), loop.bodySites.end());
  if (members.size() != loop.bodySites.size() ||
      !members.count(loop.bodyEntry) || members.count(loop.owner) ||
      members.count(loop.header) || members.count(loop.continuation))
    return fail("invalid or overlapping loop body population");
  for (auto site : members) {
    if (site >= size)
      return fail("invalid loop body site");
    for (auto next : old.sites[site].successors)
      if (!members.count(next) && next != loop.header)
        return fail("body has an unqualified control escape");
    if (old.sites[site].observation != NoControlId)
      for (const auto &atom :
           old.observations[old.sites[site].observation].atoms)
        if (atom.owner == loop.owner)
          return fail("duplicate refinement of one original loop scope");
  }
  // Refinement redirects the initializer and tombstones the old body. Every
  // external entry must therefore execute that initializer. The graph's entry
  // has no predecessor edge, so check it separately from ordinary incoming
  // edges. Otherwise the unrefined header can still reach a tombstoned body.
  if (old.entry == loop.header || members.count(old.entry))
    return fail("original entry bypasses normalized loop initialization");
  for (std::size_t site = 0; site < size; ++site) {
    if (members.count(site))
      continue;
    for (auto next : old.sites[site].successors) {
      if (site != loop.header && members.count(next))
        return fail("unqualified external entry to loop body");
      if (site != loop.owner && next == loop.header)
        return fail("unqualified external entry to loop header");
    }
  }
  std::map<std::size_t, ResidueDecision> decisions;
  for (const auto &d : loop.decisions) {
    if (!members.count(d.site) || !d.modulus || d.residue >= d.modulus ||
        loop.period % d.modulus || old.sites[d.site].successors.size() != 2 ||
        !decisions.emplace(d.site, d).second)
      return fail("invalid original residue decision");
  }
  out.program = input;
  auto &q = *out.program.observed;
  std::map<std::size_t, std::vector<std::size_t>> phases;
  for (std::size_t i = 0; i < input.operations.size(); ++i)
    out.originalPhases.push_back(i);
  for (const auto &binding : loop.effects) {
    if (binding.operation >= input.operations.size() ||
        binding.residues.size() != loop.period || phases.count(binding.operation))
      return fail("invalid periodic original-effect binding");
    for (std::size_t site = 0; site < size; ++site)
      if (old.sites[site].operation == binding.operation && !members.count(site))
        return fail("periodic effect escapes its qualified loop");
    const auto original = input.operations[binding.operation];
    auto &variants = phases[binding.operation];
    for (unsigned residue = 0; residue < loop.period; ++residue) {
      auto operation = original;
      operation.accesses = binding.residues[residue];
      for (const auto &access : operation.accesses)
        if (access.cell >= input.cells.size() || access.definiteWrite)
          return fail("periodic native effect needs existing conservative cells");
      const auto id = residue ? out.program.operations.size() : binding.operation;
      if (residue) {
        out.program.operations.push_back(std::move(operation));
        out.originalPhases.push_back(binding.operation);
      } else out.program.operations[id] = std::move(operation);
      variants.push_back(id);
    }
  }
  using Mode = std::tuple<unsigned, unsigned, unsigned>;
  std::map<Mode, std::size_t> headers;
  std::vector<Mode> work;
  auto header = [&](Mode mode) {
    auto it = headers.find(mode);
    if (it != headers.end())
      return it->second;
    const auto id = q.sites.size();
    q.sites.emplace_back();
    q.sites.back().context = old.sites[loop.header].context;
    headers.emplace(mode, id);
    work.push_back(mode);
    return id;
  };
  q.sites[loop.owner].successors.clear();
  for (unsigned remaining = loop.atLeastOnce ? 1 : 0; remaining <= loop.period + 1; ++remaining) {
    const auto id = header({0, 0, remaining});
    q.sites[loop.owner].successors.push_back(id);
  }
  std::map<std::tuple<std::size_t, unsigned, bool, bool, bool, bool>, std::size_t>
      observations;
  for (std::size_t modeIndex = 0; modeIndex < work.size(); ++modeIndex) {
    const auto [residue, elapsed, remaining] = work[modeIndex];
    const auto at = headers.at(work[modeIndex]);
    if (!remaining) {
      q.sites[at].successors = {loop.continuation};
      continue;
    }
    std::map<std::size_t, std::size_t> clone;
    for (auto site : members) {
      const auto id = q.sites.size();
      clone.emplace(site, id);
      q.sites.push_back(old.sites[site]);
      auto phase = phases.find(old.sites[site].operation);
      if (phase != phases.end()) q.sites[id].operation = phase->second[residue];
      q.sites[id].successors.clear();
      q.sites[id].backedgeOwners.clear();
      const auto oldObservation = old.sites[site].observation;
      if (oldObservation == NoControlId)
        continue;
      const auto key =
          std::make_tuple(oldObservation, residue, elapsed >= loop.period,
                          remaining > loop.period, elapsed != 0,
                          remaining > 1);
      auto found = observations.find(key);
      if (found == observations.end()) {
        auto observation = old.observations[oldObservation];
        observation.atoms.insert(
            observation.atoms.end(),
            {{ObservationAtom::LoopResidue, loop.owner, loop.period, residue},
             {ObservationAtom::LoopHasPrevious, loop.owner, loop.period,
              uint64_t(elapsed >= loop.period)},
             {ObservationAtom::LoopHasNext, loop.owner, loop.period,
              uint64_t(remaining > loop.period)}});
        // A containing storage-use cycle and original first-iteration guards
        // need the first child visit independently of the same-bank distance.
        // For period one the ordinary previous-use atom is already identical.
        if (loop.period > 1)
          observation.atoms.push_back(
              {ObservationAtom::LoopHasPrevious, loop.owner, 1,
               uint64_t(elapsed != 0)});
        // Enclosing producer/reader cycles need the child's first and final
        // visits at the actual reader frontier, not only at its backedge.
        // These remain predicates over the original IV and upper bound.
        if (loop.period > 1)
          observation.atoms.push_back(
              {ObservationAtom::LoopHasNext, loop.owner, 1, uint64_t(remaining > 1)});
        const auto index = q.observations.size();
        q.observations.push_back(std::move(observation));
        observations.emplace(key, index);
        q.sites[id].observation = index;
      } else
        q.sites[id].observation = found->second;
    }
    q.sites[at].successors = {clone.at(loop.bodyEntry)};
    const unsigned nextResidue = (residue + 1) % loop.period,
                   nextElapsed = std::min(elapsed + 1, loop.period);
    std::vector<std::size_t> back;
    back.push_back(
        header({nextResidue, nextElapsed,
                remaining == loop.period + 1 ? loop.period : remaining - 1}));
    if (remaining == loop.period + 1)
      back.push_back(header({nextResidue, nextElapsed, loop.period + 1}));
    for (auto site : members) {
      const auto id = clone.at(site);
      // Retain the original owner of an internal backedge. A nested loop
      // keeps its own recurrence within each outer mode; only edges to the
      // refined header acquire the new outer-mode destinations below.
      std::vector<std::pair<std::size_t, std::size_t>> next;
      const auto &original = old.sites[site];
      for (std::size_t edge = 0; edge < original.successors.size(); ++edge)
        next.emplace_back(original.successors[edge],
                          original.backedgeOwners.empty()
                              ? NoControlId
                              : original.backedgeOwners[edge]);
      auto decision = decisions.find(site);
      if (decision != decisions.end()) {
        const auto &d = decision->second;
        const bool yes = ((residue % d.modulus) == d.residue) == d.equal;
        next = {next[yes ? 0 : 1]};
      }
      for (const auto &[target, owner] : next) {
        if (target == loop.header) {
          q.sites[id].successors.insert(q.sites[id].successors.end(),
                                        back.begin(), back.end());
          q.sites[id].backedgeOwners.insert(q.sites[id].backedgeOwners.end(),
                                            back.size(), loop.owner);
        } else {
          q.sites[id].successors.push_back(clone.at(target));
          q.sites[id].backedgeOwners.push_back(owner);
        }
      }
    }
  }
  // The unrefined body has been replaced in the ANALYSIS graph. Leaving its
  // unconditional observation attached to the original anchor could let a
  // conservative emitter add a word not represented by any refined visit.
  // Keep stable site IDs as effect-free, unavailable, unreachable tombstones.
  for (auto site : members) {
    q.sites[site].operation = NoControlId;
    q.sites[site].observation = NoControlId;
  }
  ObservedLoop refinedLoop{loop.owner, loop.owner, loop.continuation, {}};
  for (std::size_t site = size; site < q.sites.size(); ++site)
    refinedLoop.sites.push_back(site);
  q.loops.push_back(std::move(refinedLoop));
  q.qualification += "; normalized-counted-first-tail-v1";
  valid = validateProgram(out.program);
  out.success = valid.success;
  out.reason = valid.reason;
  return out;
}
namespace observed_detail {
// Exact marginal origin transfer along reference control, after event legality
// has been verified. No origin bit creates a completion edge or a runtime
// guard.
inline std::vector<ObservedCorrespondence>
correspondence(const Program &p, const Commands &commands) {
  using Origins = std::set<ObservedEndpoint>;
  using Key = std::tuple<Pipe, Pipe, unsigned>;
  std::map<Key, std::size_t> keys;
  for (const auto &word : commands)
    for (const auto &c : word)
      if (c.kind == Command::Publish || c.kind == Command::Acquire)
        keys.emplace(Key{c.source, c.observer, c.key}, keys.size());
  const auto &q = *p.observed;
  using Facts = std::vector<Origins>;
  std::vector<std::optional<Facts>> incoming(q.sites.size());
  incoming[q.entry] = Facts(keys.size());
  std::deque<std::size_t> queue{q.entry};
  std::vector<bool> queued(q.sites.size());
  queued[q.entry] = true;
  auto step = [&](std::size_t site, Facts state, auto visit) {
    const auto observation = q.sites[site].observation;
    for (std::size_t i = 0; i < commands[site].size(); ++i) {
      const auto &c = commands[site][i];
      if (c.kind != Command::Publish && c.kind != Command::Acquire)
        continue;
      auto &origins = state[keys.at(Key{c.source, c.observer, c.key})];
      if (c.kind == Command::Publish)
        origins = {{observation, i}};
      else {
        visit(ObservedEndpoint{observation, i}, origins);
        origins.clear();
      }
    }
    return state;
  };
  while (!queue.empty()) {
    const auto at = queue.front();
    queue.pop_front();
    queued[at] = false;
    auto next = step(at, *incoming[at], [](const auto &, const auto &) {});
    for (auto target : q.sites[at].successors) {
      bool changed = false;
      if (!incoming[target]) {
        incoming[target] = next;
        changed = true;
      } else
        for (std::size_t key = 0; key < keys.size(); ++key)
          for (const auto &origin : next[key])
            changed =
                incoming[target]->at(key).insert(origin).second || changed;
      if (changed && !queued[target]) {
        queued[target] = true;
        queue.push_back(target);
      }
    }
  }
  std::map<ObservedEndpoint, Origins> found;
  for (std::size_t at = 0; at < q.sites.size(); ++at)
    if (incoming[at])
      (void)step(at, *incoming[at],
                 [&](const auto &endpoint, const auto &origins) {
                   found[endpoint].insert(origins.begin(), origins.end());
                 });
  std::vector<ObservedCorrespondence> out;
  for (const auto &[endpoint, origins] : found)
    out.push_back({endpoint, {origins.begin(), origins.end()}});
  return out;
}
} // namespace observed_detail
ObservedSchema exportObservedSchema(const Program &p,
                                    const Commands &commands) {
  ObservedSchema out;
  if (!p.observed) {
    out.reason = "observed schema requires an original observation model";
    return out;
  }
  const auto report = analyze(p, commands, {false});
  if (!report.verified()) {
    out.reason = report.reason;
    return out;
  }
  std::set<std::size_t> emitted;
  for (Cut c = 0; c < commands.size(); ++c) {
    const auto o = p.observed->sites[c].observation;
    if (o == NoControlId || commands[c].empty() || !emitted.insert(o).second)
      continue;
    out.words.push_back({p.observed->observations[o], commands[c]});
  }
  out.correspondence = observed_detail::correspondence(p, commands);
  out.complete = true;
  return out;
}
Result reconstructObservedSchema(const Program &p,
                                 const ObservedSchema &schema) {
  Result out;
  if (!p.observed || !schema.complete) {
    out.reason = "incomplete observed schema";
    return out;
  }
  const auto declared = validateProgram(p);
  if (!declared.success) {
    out.reason = declared.reason;
    return out;
  }
  const auto reachable = detail::reachableSites(detail::buildControlGraph(p));
  out.commands.resize(commandCutCount(p));
  std::set<std::size_t> used;
  for (const auto &word : schema.words) {
    auto found =
        std::find_if(p.observed->observations.begin(),
                     p.observed->observations.end(), [&](const auto &o) {
                       return observed_detail::same(o, word.observation);
                     });
    if (found == p.observed->observations.end()) {
      out.reason = "emitted observation is absent from original cut vocabulary";
      return out;
    }
    const auto id = std::size_t(found - p.observed->observations.begin());
    if (!used.insert(id).second) {
      out.reason = "duplicate emitted observed word";
      return out;
    }
    bool represented = false;
    for (Cut c = 0; c < out.commands.size(); ++c)
      represented |= reachable[c] && p.observed->sites[c].observation == id;
    if (!represented) {
      out.reason = "emitted word has no reachable original observation";
      return out;
    }
    for (Cut c = 0; c < out.commands.size(); ++c)
      if (p.observed->sites[c].observation == id)
        out.commands[c] = word.commands;
  }
  auto checked = verify(p, out.commands);
  out.success = checked.success;
  out.reason = checked.reason;
  if (out.success) {
    const auto actual = observed_detail::correspondence(p, out.commands);
    bool same = actual.size() == schema.correspondence.size();
    if (same)
      for (std::size_t i = 0; i < actual.size(); ++i)
        same &= actual[i].acquisition == schema.correspondence[i].acquisition &&
                actual[i].publications == schema.correspondence[i].publications;
    if (!same) {
      out.success = false;
      out.reason = "emitted endpoint correspondence differs from "
                   "original-control replay";
    }
  }
  return out;
}
} // namespace mlir::pto::oahs
#endif
