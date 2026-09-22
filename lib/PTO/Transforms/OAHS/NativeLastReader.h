// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_NATIVE_LAST_READER_H
#define PTO_OAHS_NATIVE_LAST_READER_H
#include "NativeFirstUse.h"
namespace mlir::pto::oahs::native_detail {
// Original physical effects qualify final sources independently of whether
// the original suffix has reader-pipe payloads: selected waits can also broaden
// a later release. No completion or event credit is produced here.
inline void importFinalReadSources(
    Program &program, const SmallVector<scf::ForOp> &loops,
    const DenseMap<mlir::Operation *, std::size_t> &ids,
    DenseMap<std::size_t, scf::ForOp> &owners, std::vector<std::string> &notes,
    const StorageFrontierAnalysis &storage) {
  std::map<unsigned, std::set<Pipe>> writers;
  for (const auto &op : program.operations) for (auto a : op.accesses)
    if (a.write) writers[a.cell].insert(op.pipe);
  if (!storage.complete()) return;
  std::set<std::pair<std::size_t, unsigned>> requiredReturns;
  for (std::size_t op = 0; op < program.operations.size(); ++op)
    for (auto access : program.operations[op].accesses) if (access.write)
      for (auto site : storage.sitesForOperation(op))
        for (const auto &reader : storage.previousReaders(site, access.cell))
          if (program.operations[reader.operation].pipe != program.operations[op].pipe)
            requiredReturns.emplace(reader.operation, access.cell);
  for (auto loop : loops) {
    const auto owner = ids.lookup(loop.getOperation());
    const auto lower = firstUseInteger(loop.getLowerBound());
    const auto upper = firstUseInteger(loop.getUpperBound());
    const auto step = firstUseInteger(loop.getStep());
    if (owners.count(owner) || !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp") ||
        !lower || !upper || !step || *lower < 0 || *step <= 0 || *lower >= *upper ||
        *lower + ((*upper - *lower - 1) / *step) * *step >
          std::numeric_limits<int64_t>::max() - *step) continue;
    const auto &q = *program.observed;
    auto found = std::find_if(q.loops.begin(), q.loops.end(),
        [&](const auto &r) { return r.owner == owner; });
    if (found == q.loops.end() || !found->atLeastOnce ||
        found->bodyEntry == NoControlId || q.sites[owner].successors.size() != 1) continue;
    const auto header = q.sites[owner].successors.front();
    std::set<std::size_t> members;
    std::vector<std::size_t> todo{found->bodyEntry}, prefix;
    while (!todo.empty()) {
      auto at = todo.back(); todo.pop_back();
      if (at == header || at == found->exit || !members.insert(at).second) continue;
      for (auto next : q.sites[at].successors) todo.push_back(next);
    }
    std::set<std::size_t> seen;
    for (auto at = found->bodyEntry; members.count(at) && seen.insert(at).second;) {
      if (q.sites[at].successors.size() != 1) break;
      prefix.push_back(at); at = q.sites[at].successors.front();
    }
    const std::set<std::size_t> prefixSites(prefix.begin(), prefix.end());
    std::set<unsigned> written, suffixReads;
    std::set<Pipe> issued;
    std::map<unsigned, std::set<Pipe>> readers;
    std::map<unsigned, std::size_t> last;
    for (auto site : members) {
      auto op = q.sites[site].operation;
      if (op == NoControlId) continue;
      issued.insert(program.operations[op].pipe);
      for (auto a : program.operations[op].accesses) {
        if (a.write) written.insert(a.cell);
        if (a.read) {
          readers[a.cell].insert(program.operations[op].pipe);
          if (!prefixSites.count(site)) suffixReads.insert(a.cell);
        }
      }
    }
    for (std::size_t i = 0; i < prefix.size(); ++i) {
      auto op = q.sites[prefix[i]].operation;
      if (op != NoControlId) for (auto a : program.operations[op].accesses)
        if (a.read) last[a.cell] = i;
    }
    std::set<std::size_t> gaps;
    for (const auto &[cell, i] : last) {
      if (!requiredReturns.count({q.sites[prefix[i]].operation, cell}) ||
          i + 1 >= prefix.size() || written.count(cell) || suffixReads.count(cell) ||
          readers[cell].size() != 1 || writers[cell].size() != 1 ||
          issued.count(*writers[cell].begin()) || program.cells[cell].exclusive ||
          program.cells[cell].unknownRange) continue;
      gaps.insert(prefix[i + 1]);
    }
    if (gaps.empty()) continue;
    ReaderVisitRegion request;
    request.owner = owner; request.step = uint64_t(*step);
    request.singleVisit = *upper - *lower <= *step;
    request.finalSourceGaps = true;
    request.lastPublications.assign(gaps.begin(), gaps.end());
    auto refined = refineReaderVisits(program, request);
    if (!refined.success) continue;
    selected::Control control(refined.program);
    if (!control.complete) continue;
    program = std::move(refined.program); owners[owner] = loop;
    notes.push_back("qualified final-read source gaps: " + std::to_string(gaps.size()));
  }
}
// Expose final visits only when an invariant cross-engine input has trailing
// work on its reader engine. This is an original-control qualification, not a
// promise to select a channel. Existing refined owners retain their vocabulary.
inline void importLastReaders(
    Program &program, const SmallVector<scf::ForOp> &loops,
    const DenseMap<mlir::Operation *, std::size_t> &ids,
    DenseMap<std::size_t, scf::ForOp> &owners, std::vector<std::string> &notes) {
  std::map<unsigned, std::set<Pipe>> writers;
  for (const auto &op : program.operations) for (auto access : op.accesses)
    if (access.write) writers[access.cell].insert(op.pipe);
  for (auto loop : loops) {
    const auto owner = ids.lookup(loop.getOperation());
    const auto lower = firstUseInteger(loop.getLowerBound());
    const auto upper = firstUseInteger(loop.getUpperBound());
    const auto step = firstUseInteger(loop.getStep());
    // Unit step keeps LoopHasNext's native remaining-distance contract exact.
    // Nonempty nonnegative constant bounds also make upper-IV overflow-free.
    if (owners.count(owner) || !loop->getParentOfType<scf::ForOp>() ||
        !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp") ||
        !lower || !upper || !step || *step != 1 || *lower < 0 || *lower >= *upper) continue;
    const auto &q = *program.observed;
    const auto found = std::find_if(q.loops.begin(), q.loops.end(),
        [&](const auto &region) { return region.owner == owner; });
    if (found == q.loops.end() || found->bodyEntry == NoControlId ||
        q.sites[owner].successors.size() != 1) continue;
    const auto header = q.sites[owner].successors.front();
    std::vector<std::size_t> body;
    std::set<std::size_t> seen;
    auto at = found->bodyEntry;
    while (at != header && at != found->exit && seen.insert(at).second) {
      if (q.sites[at].successors.size() != 1) break;
      body.push_back(at);
      at = q.sites[at].successors.front();
    }
    if (at != header) continue;
    std::set<unsigned> written;
    std::set<Pipe> issued;
    std::map<unsigned, std::size_t> lastRead;
    std::map<unsigned, std::set<Pipe>> readers;
    for (std::size_t i = 0; i < body.size(); ++i) {
      const auto op = q.sites[body[i]].operation;
      if (op == NoControlId) continue;
      const auto &operation = program.operations[op];
      issued.insert(operation.pipe);
      for (auto access : operation.accesses) {
        if (access.write) written.insert(access.cell);
        if (access.read) { lastRead[access.cell] = i; readers[access.cell].insert(operation.pipe); }
      }
    }
    std::set<std::size_t> anchors;
    for (const auto &[cell, index] : lastRead) {
      if (written.count(cell) || readers[cell].size() != 1 || writers[cell].size() != 1 ||
          issued.count(*writers[cell].begin()) || program.cells[cell].exclusive ||
          program.cells[cell].unknownRange || index + 1 >= body.size()) continue;
      const auto reader = *readers[cell].begin();
      bool trailing = false;
      for (auto i = index + 1; i < body.size(); ++i) {
        const auto op = q.sites[body[i]].operation;
        trailing |= op != NoControlId && program.operations[op].pipe == reader;
      }
      if (trailing) anchors.insert(body[index + 1]);
    }
    if (anchors.empty()) continue;
    auto refined = refineLastVisit(program, owner, {anchors.begin(), anchors.end()});
    if (!refined.success) continue;
    selected::Control control(refined.program);
    if (!control.complete) continue;
    program = std::move(refined.program);
    owners[owner] = loop;
    notes.push_back("qualified last-reader visit: " + std::to_string(body.size()) + " copied sites");
  }
}
} // namespace mlir::pto::oahs::native_detail
#endif
