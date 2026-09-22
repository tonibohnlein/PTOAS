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
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <limits>
#include <memory>

namespace mlir::pto::oahs::native_detail {
// Collect first/final endpoint roles before refining one original owner.
// The first-only fallback splits a straight prefix. Original positive bounds
// prove entry; the backedge still admits arbitrarily many later visits. Only
// invariant-input consumers and first writes conflicting with source-inactive
// readers get distinct first/later command observations.
// Other original words, physical operations and all completion histories stay
// shared. This creates vocabulary, not synchronization or completion credit.
inline void importFirstConsumers(
    Program &program, const SmallVector<scf::ForOp> &loops,
    const DenseMap<mlir::Operation *, std::size_t> &ids,
    DenseMap<std::size_t, scf::ForOp> &owners,
    std::vector<std::string> &notes, bool classInvariant = false, bool firstWrites = false,
    const StorageFrontierAnalysis *storageView = nullptr) {
  // Snapshot the original storage view before cloning occurrence vocabulary.
  // Marginal prior-reader witnesses only admit useful candidates; they grant
  // no completion or rearming credit to the selected constructor.
  std::unique_ptr<StorageFrontierAnalysis> ownedStorage;
  if (firstWrites && !storageView) ownedStorage = std::make_unique<StorageFrontierAnalysis>(program);
  const auto *storage = storageView ? storageView : ownedStorage.get();
  std::map<unsigned, std::set<Pipe>> writers;
  for (const auto &op : program.operations) for (auto access : op.accesses) {
    if (access.write) writers[access.cell].insert(op.pipe);
  }
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
            if (writer != program.operations[op].pipe && (classInvariant || !bodyPipes.count(writer)) &&
                parentWrites[writer].count(access.cell))
              reusableInputs[writer].insert(access.cell);
        }
      }
      for (const auto &[pipe, cells] : reusableInputs) {
        (void)pipe;
        if (cells.size() > 1) earlyInputs.insert(cells.begin(), cells.end());
      }
    }
    // Reader completion is invariant when its source pipe does not issue in
    // this region, even though the receiving pipe repeatedly overwrites the
    // cell. This is a candidate occurrence boundary only: source-time history,
    // completion, balanced participation and rearming remain selected queries.
    std::set<std::pair<unsigned, Pipe>> firstWriteRoles;
    if (firstWrites && storage && storage->complete()) for (auto at : members) {
      const auto op = old.sites[at].operation;
      if (op == NoControlId) continue;
      const auto &operation = program.operations[op];
      for (auto access : operation.accesses) {
        const auto &cell = program.cells[access.cell];
        if (!access.write || cell.unknownRange || cell.exclusive) continue;
        for (auto original : storage->sitesForOperation(op))
          for (const auto &origin : storage->previousReaders(original, access.cell)) {
            const auto reader = program.operations[origin.operation].pipe;
            if (reader != operation.pipe && !bodyPipes.count(reader))
              firstWriteRoles.insert({access.cell, operation.pipe});
          }
      }
    }
    if (earlyInputs.empty() && firstWriteRoles.empty()) continue;
    std::vector<std::size_t> prefix;
    std::set<std::size_t> consumers, firstWriters, seen;
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
            invariant |= writer != operation.pipe && (classInvariant || !bodyPipes.count(writer));
        bool firstWrite = false;
        for (auto a : operation.accesses)
          if (a.write) firstWrite |= firstWriteRoles.erase({a.cell, operation.pipe}) != 0;
        // A first write needs its own deadline even when it is the observer's
        // first payload: an earlier outward publication can forbid entry
        // placement. Preserve that publication's original position.
        if (invariant && issued.count(operation.pipe)) {
          consumers.insert(at);
          last = prefix.size();
        }
        if (firstWrite) {
          firstWriters.insert(at);
          last = prefix.size();
        }
        issued.insert(operation.pipe);
      }
      if (site.successors.size() != 1 ||
          (!site.backedgeOwners.empty() && site.backedgeOwners[0] != NoControlId)) break;
      at = site.successors[0];
    }
    if (!firstWriters.empty() && prefix.size() != members.size()) {
      // Keep the initial first-write extension to a whole straight child.
      // A truncated clone can discard another bank's post-read boundary.
      firstWriters.clear();
      last = 0;
      for (std::size_t i = 0; i < prefix.size(); ++i)
        if (consumers.count(prefix[i])) last = i + 1;
    }
    if (!last) continue;
    // Collect the other end of the lifetime on the SAME original owner,
    // before either observation refinement claims it. The straight prefix may
    // be followed by arbitrary acyclic choices; every suffix access is checked.
    std::set<std::size_t> lastPublications;
    if (loop->getParentOfType<scf::ForOp>() &&
        *lower + ((*upper - *lower - 1) / *step) * *step <=
            std::numeric_limits<int64_t>::max() - *step) {
      std::map<unsigned, std::set<Pipe>> readers;
      std::map<unsigned, std::size_t> lastRead;
      std::set<unsigned> suffixReads;
      const std::set<std::size_t> prefixSites(prefix.begin(), prefix.end());
      for (auto site : members) {
        const auto op = old.sites[site].operation;
        if (op == NoControlId) continue;
        for (auto a : program.operations[op].accesses) if (a.read) {
          readers[a.cell].insert(program.operations[op].pipe);
          if (!prefixSites.count(site)) suffixReads.insert(a.cell);
        }
      }
      for (std::size_t i = 0; i < prefix.size(); ++i) {
        const auto op = old.sites[prefix[i]].operation;
        if (op != NoControlId) for (auto a : program.operations[op].accesses)
          if (a.read) lastRead[a.cell] = i;
      }
      for (const auto &[cell, index] : lastRead) {
        if (written.count(cell) || suffixReads.count(cell) || readers[cell].size() != 1 ||
            writers[cell].size() != 1 || bodyPipes.count(*writers[cell].begin()) ||
            program.cells[cell].exclusive || program.cells[cell].unknownRange ||
            index + 1 >= prefix.size() || consumers.count(prefix[index + 1])) continue;
        const auto reader = *readers[cell].begin();
        bool trailing = false;
        for (auto i = index + 1; i < prefix.size(); ++i) {
          const auto op = old.sites[prefix[i]].operation;
          trailing |= op != NoControlId && program.operations[op].pipe == reader;
        }
        if (trailing) lastPublications.insert(prefix[index + 1]);
      }
    }
    if (!lastPublications.empty() && firstWriters.empty()) {
      ReaderVisitRegion request;
      request.owner = owner;
      request.firstConsumers.assign(consumers.begin(), consumers.end());
      request.lastPublications.assign(lastPublications.begin(), lastPublications.end());
      request.step = uint64_t(*step);
      request.singleVisit = *upper - *lower <= *step;
      auto refined = refineReaderVisits(program, request);
      if (refined.success) {
        selected::Control control(refined.program);
        if (control.complete) {
          program = std::move(refined.program);
          owners[owner] = loop;
          notes.push_back("qualified joint first/final reader prefix: step " + std::to_string(*step));
          continue;
        }
      }
    }
    // Include the command anchor immediately after the final copied payload.
    // Joining at that anchor would otherwise erase its source publication cut.
    if (prefix.size() <= last) continue;
    if (firstWriters.empty()) prefix.resize(last + 1);
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
    std::map<std::size_t, std::size_t> receipts;
    for (auto at : firstWriters) {
      auto observation = old.observations[old.sites[at].observation];
      observation.atoms.push_back({ObservationAtom::LoopHasPrevious, owner, 1, 0});
      observation.beforeSharedWord = true;
      const auto boundary = q.sites.size();
      receipts[clones.at(at)] = boundary;
      q.sites.push_back({NoControlId, q.observations.size(), {clones.at(at)},
                         {NoControlId}, old.sites[at].context});
      q.observations.push_back(std::move(observation));
    }
    for (auto at : prefix) for (auto &next : q.sites[clones.at(at)].successors)
      if (receipts.count(next)) next = receipts.at(next);
    q.sites[owner].successors = {receipts.count(clones.at(body)) ?
        receipts.at(clones.at(body)) : clones.at(body)};
    for (auto &region : q.loops) {
      bool contains = region.owner == owner ||
          std::find(region.sites.begin(), region.sites.end(), owner) != region.sites.end();
      if (contains) {
        for (auto at : prefix) region.sites.push_back(clones[at]);
        for (const auto &[consumer, boundary] : receipts) region.sites.push_back(boundary);
      }
      if (region.owner == owner) {
        region.bodyEntry = q.sites[owner].successors.front();
        for (const auto &[consumer, boundary] : receipts)
          region.firstWriteFrontiers.emplace_back(boundary, consumer);
        for (auto at : prefix) region.firstVisitPrefix.push_back(clones[at]);
      }
    }
    if (!refreshLoopOccurrences(q)) {
      notes.push_back("first-consumer candidate lacks a closed child occurrence interface");
      continue;
    }
    q.qualification += "; first-consumer-prefix-v1";
    selected::Control control(candidate);
    if (!control.complete) {
      notes.push_back("first-consumer candidate declined: " + control.reason);
      continue;
    }
    program = std::move(candidate);
    owners[owner] = loop;
    notes.push_back("qualified first-consumer prefix: " + std::to_string(prefix.size()) + " sites");
  }
}
} // namespace mlir::pto::oahs::native_detail
#endif
