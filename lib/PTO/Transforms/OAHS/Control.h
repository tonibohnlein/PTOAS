// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_CONTROL_H
#define PTO_OAHS_CONTROL_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include <vector>

namespace mlir::pto::oahs::detail {
struct ControlSite {
  std::vector<std::size_t> successors;
  // Parallel to successors. A value names the ORIGINAL loop whose backedge
  // this edge crosses. It is a static witness, not an iteration distance.
  std::vector<std::size_t> backedgeOwners;
};
struct ControlGraph {
  std::vector<ControlSite> sites;
  std::vector<AnalysisContext> contexts;
  std::vector<std::size_t> cutContexts, cutRanks;
  std::size_t entry = 0;
};
// Shared by completion solving and source-cut queries. Caller has validated the
// original tree. No duplicated occurrence graph or new program predicates.
inline ControlGraph buildControlGraph(const Program &p) {
  ControlGraph g;
  const auto exit = p.operations.size();
  g.sites.resize(exit + 1);
  g.contexts = {AnalysisContext{}}; g.cutContexts.assign(exit + 1, 0);
  g.cutRanks.resize(exit + 1); g.cutRanks[exit] = exit;
  auto edge = [&](std::size_t from, std::size_t to, std::size_t loop = NoAnalysisId) {
    g.sites[from].successors.push_back(to);
    g.sites[from].backedgeOwners.push_back(loop);
  };
  auto entryFor = [&](const Region &r) {
    if (r.kind == Region::Operation) return r.operation;
    const auto id = g.sites.size(); g.sites.emplace_back(); return id;
  };
  if (p.body.kind == Region::Sequence && p.body.children.empty()) {
    for (std::size_t i = 0; i < exit; ++i) { edge(i, i+1); g.cutRanks[i] = i; }
    return g;
  }
  struct Task {
    const Region *region;
    std::size_t entry, next, context, nextBackedge;
  };
  g.entry = entryFor(p.body);
  std::vector<Task> tasks{{&p.body, g.entry, exit, 0, NoAnalysisId}};
  auto contextFor = [&](AnalysisContext::Kind kind, const Task &task) {
    const auto id = g.contexts.size();
    g.contexts.push_back({kind, task.context, task.entry}); return id;
  };
  while (!tasks.empty()) {
    const auto task = tasks.back(); tasks.pop_back();
    const auto &r = *task.region;
    if (r.kind == Region::Operation) {
      edge(task.entry, task.next, task.nextBackedge);
      g.cutContexts[task.entry] = task.context;
    } else if (r.kind == Region::Sequence) {
      std::vector<std::size_t> children;
      for (const auto &child : r.children) children.push_back(entryFor(child));
      edge(task.entry, children.empty() ? task.next : children.front(),
           children.empty() ? task.nextBackedge : NoAnalysisId);
      for (std::size_t i = 0; i < children.size(); ++i)
        tasks.push_back({&r.children[i], children[i],
                         i+1 < children.size() ? children[i+1] : task.next,
                         task.context, i+1 < children.size() ? NoAnalysisId : task.nextBackedge});
    } else if (r.kind == Region::Choice) {
      const auto yes = entryFor(r.children[0]), no = entryFor(r.children[1]);
      edge(task.entry, yes); edge(task.entry, no);
      tasks.push_back({&r.children[0], yes, task.next,
                      contextFor(AnalysisContext::ThenArm, task), task.nextBackedge});
      tasks.push_back({&r.children[1], no, task.next,
                      contextFor(AnalysisContext::ElseArm, task), task.nextBackedge});
    } else if (r.kind == Region::For) {
      const auto body = entryFor(r.children[0]);
      edge(task.entry, body); edge(task.entry, task.next, task.nextBackedge);
      tasks.push_back({&r.children[0], body, task.entry,
                      contextFor(AnalysisContext::ForBody, task), task.entry});
    } else {
      const auto before = entryFor(r.children[0]), after = entryFor(r.children[1]);
      const auto decision = g.sites.size(); g.sites.emplace_back();
      edge(task.entry, before); edge(decision, after);
      edge(decision, task.next, task.nextBackedge);
      tasks.push_back({&r.children[0], before, decision,
                      contextFor(AnalysisContext::WhileBefore, task), NoAnalysisId});
      tasks.push_back({&r.children[1], after, before,
                      contextFor(AnalysisContext::WhileAfter, task), task.entry});
    }
  }
  std::vector<const Region *> order{&p.body};
  std::size_t rank = 0;
  while (!order.empty()) {
    const auto *r = order.back(); order.pop_back();
    if (r->kind == Region::Operation) g.cutRanks[r->operation] = rank++;
    else for (auto i = r->children.rbegin(); i != r->children.rend(); ++i) order.push_back(&*i);
  }
  return g;
}
} // namespace mlir::pto::oahs::detail
#endif
