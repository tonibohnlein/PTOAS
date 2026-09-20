// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_NATIVE_FIRST_CONSUMER_H
#define PTO_OAHS_NATIVE_FIRST_CONSUMER_H
#include "NativeFirstUse.h"
#include <limits>

namespace mlir::pto::oahs::native_detail {
// Split only a straight first-iteration prefix. The original positive bounds
// prove entry; the backedge still admits arbitrarily many later visits. Only
// invariant-input consumers get distinct first/later command observations.
// Other original words, physical operations and all completion histories stay
// shared. This creates vocabulary, not synchronization or completion credit.
inline void importFirstConsumers(
    Program &program, const SmallVector<scf::ForOp> &loops,
    const DenseMap<mlir::Operation *, std::size_t> &ids,
    DenseMap<std::size_t, scf::ForOp> &owners,
    std::vector<std::string> &notes) {
  std::map<unsigned, std::set<Pipe>> writers;
  for (const auto &op : program.operations) for (auto access : op.accesses)
    if (access.write) writers[access.cell].insert(op.pipe);
  std::vector<std::vector<std::size_t>> predecessors(program.observed->sites.size());
  for (std::size_t at = 0; at < predecessors.size(); ++at)
    for (auto next : program.observed->sites[at].successors) predecessors[next].push_back(at);
  for (auto loop : loops) {
    const auto owner = ids.lookup(loop.getOperation());
    if (owners.count(owner)) continue;
    bool nested = false;
    loop.walk([&](mlir::Operation *op) {
      nested |= op != loop.getOperation() && isa<scf::ForOp, scf::WhileOp>(op);
    });
    const auto lower = firstUseInteger(loop.getLowerBound());
    const auto upper = firstUseInteger(loop.getUpperBound());
    const auto step = firstUseInteger(loop.getStep());
    if (nested || !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp") ||
        !lower || !upper || !step || *lower < 0 || *step <= 0 || *lower >= *upper ||
        *lower > std::numeric_limits<int64_t>::max() - *step) continue;
    const auto &old = *program.observed;
    if (old.sites[owner].successors.size() != 1) continue;
    const auto header = old.sites[owner].successors.front();
    if (old.sites[header].successors.size() != 2) continue;
    const auto body = old.sites[header].successors[0];
    const auto exit = old.sites[header].successors[1];
    std::set<std::size_t> members;
    std::vector<std::size_t> todo{body};
    while (!todo.empty()) {
      auto at = todo.back(); todo.pop_back();
      if (at == header || at == exit || !members.insert(at).second) continue;
      for (auto next : old.sites[at].successors) todo.push_back(next);
    }
    std::set<unsigned> written;
    std::set<Pipe> bodyPipes;
    for (auto at : members) {
      auto op = old.sites[at].operation;
      if (op != NoControlId) {
        bodyPipes.insert(program.operations[op].pipe);
        for (auto a : program.operations[op].accesses)
          if (a.write) written.insert(a.cell);
      }
    }
    // Admission is useful only when a late publication would observe other
    // source work. Inspect the original straight incoming corridor once; do
    // not reserve channels merely to reduce repeated command counts.
    std::set<unsigned> earlyInputs, seenWrites;
    std::set<Pipe> laterPipes;
    std::set<std::size_t> incoming;
    for (auto at = owner; at < predecessors.size() && predecessors[at].size() == 1;) {
      at = predecessors[at].front();
      if (!incoming.insert(at).second || old.sites[at].successors.size() != 1) break;
      const auto op = old.sites[at].operation;
      if (op == NoControlId) continue;
      const auto &operation = program.operations[op];
      for (auto access : operation.accesses)
        if (access.write && seenWrites.insert(access.cell).second && laterPipes.count(operation.pipe))
          earlyInputs.insert(access.cell);
      laterPipes.insert(operation.pipe);
    }
    // A repeated parent may own several invariant inputs consumed by this
    // child. Their separate first consumers are also needed to seal reuse
    // cycles, even when the final producer has no later sibling load.
    if (auto parent = loop->getParentOfType<scf::ForOp>()) {
      std::map<Pipe, std::set<unsigned>> parentWrites, reusableInputs;
      parent.walk([&](mlir::Operation *operation) {
        if (operation == loop.getOperation() || loop->isAncestor(operation)) return;
        const auto found = ids.find(operation);
        if (found == ids.end()) return;
        const auto source = old.sites[found->second].operation;
        if (source == NoControlId) return;
        for (auto access : program.operations[source].accesses)
          if (access.write) parentWrites[program.operations[source].pipe].insert(access.cell);
      });
      for (auto at : members) {
        const auto op = old.sites[at].operation;
        if (op == NoControlId) continue;
        for (auto access : program.operations[op].accesses) {
          if (!access.read || written.count(access.cell)) continue;
          for (auto writer : writers[access.cell])
            if (writer != program.operations[op].pipe && !bodyPipes.count(writer) &&
                parentWrites[writer].count(access.cell))
              reusableInputs[writer].insert(access.cell);
        }
      }
      for (const auto &[pipe, cells] : reusableInputs) {
        (void)pipe;
        if (cells.size() > 1) earlyInputs.insert(cells.begin(), cells.end());
      }
    }
    if (earlyInputs.empty()) continue;
    std::vector<std::size_t> prefix;
    std::set<std::size_t> consumers, seen;
    std::set<Pipe> issued;
    std::set<std::pair<unsigned, Pipe>> readInputs;
    std::size_t last = 0;
    for (auto at = body; members.count(at) && seen.insert(at).second;) {
      const auto &site = old.sites[at];
      if (site.observation == NoControlId ||
          !old.observations[site.observation].atoms.empty()) break;
      prefix.push_back(at);
      const auto op = site.operation;
      if (op != NoControlId) {
        const auto &operation = program.operations[op];
        bool invariant = false;
        for (auto a : operation.accesses) if (a.read && earlyInputs.count(a.cell) && !written.count(a.cell) &&
            !program.cells[a.cell].unknownRange && !program.cells[a.cell].exclusive &&
            readInputs.insert({a.cell, operation.pipe}).second)
          for (auto writer : writers[a.cell])
            invariant |= writer != operation.pipe && !bodyPipes.count(writer);
        // Entry acquisition already handles the first observer operation.
        if (invariant && issued.count(operation.pipe)) {
          consumers.insert(at);
          last = prefix.size();
        }
        issued.insert(operation.pipe);
      }
      if (site.successors.size() != 1 ||
          (!site.backedgeOwners.empty() && site.backedgeOwners[0] != NoControlId)) break;
      at = site.successors[0];
    }
    if (!last) continue;
    // Include the command anchor immediately after the final copied payload.
    // Joining at that anchor would otherwise erase its source publication cut.
    if (prefix.size() <= last) continue;
    prefix.resize(last + 1);
    auto candidate = program;
    auto &q = *candidate.observed;
    std::map<std::size_t, std::size_t> clones;
    for (auto at : prefix) {
      clones[at] = q.sites.size();
      q.sites.push_back(old.sites[at]);
      // If the prefix reaches the terminator, its first transition to the
      // repeated header is forward. Only the original later visits backedge.
      for (auto &edgeOwner : q.sites.back().backedgeOwners)
        if (edgeOwner == owner) edgeOwner = NoControlId;
      if (consumers.count(at)) {
        for (unsigned later = 0; later != 2; ++later) {
          auto observation = old.observations[old.sites[at].observation];
          observation.atoms.push_back({ObservationAtom::LoopHasPrevious, owner, 1, later});
          q.sites[later ? at : clones[at]].observation = q.observations.size();
          q.observations.push_back(std::move(observation));
        }
      }
    }
    for (auto at : prefix) for (auto &next : q.sites[clones[at]].successors)
      if (clones.count(next)) next = clones[next];
    q.sites[owner].successors = {clones.at(body)};
    for (auto &region : q.loops) {
      bool contains = region.owner == owner ||
          std::find(region.sites.begin(), region.sites.end(), owner) != region.sites.end();
      if (contains) for (auto at : prefix) region.sites.push_back(clones[at]);
      if (region.owner == owner) {
        region.bodyEntry = clones.at(body);
        for (auto at : prefix) region.firstVisitPrefix.push_back(clones[at]);
      }
    }
    q.qualification += "; first-consumer-prefix-v1";
    selected::Control control(candidate);
    if (!control.complete) continue;
    program = std::move(candidate);
    owners[owner] = loop;
    notes.push_back("qualified first-consumer prefix: " + std::to_string(prefix.size()) + " sites");
  }
}
} // namespace mlir::pto::oahs::native_detail
#endif
