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
