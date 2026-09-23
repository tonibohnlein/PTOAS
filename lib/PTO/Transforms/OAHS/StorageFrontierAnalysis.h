// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_STORAGE_FRONTIER_ANALYSIS_H
#define PTO_OAHS_STORAGE_FRONTIER_ANALYSIS_H
#include "ControlComponents.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <tuple>
#include <set>

namespace mlir::pto::oahs {
namespace storage_detail {
inline std::vector<std::size_t> unite(std::vector<std::size_t> a,
                                      const std::vector<std::size_t> &b) {
  a.insert(a.end(), b.begin(), b.end());
  std::sort(a.begin(), a.end());
  a.erase(std::unique(a.begin(), a.end()), a.end());
  return a;
}
struct Matrix {
  std::size_t width = 0;
  std::vector<uint64_t> words;
  Matrix() = default;
  Matrix(std::size_t sites, std::size_t origins)
      : width(origins / 64 + (origins % 64 != 0)) {
    if (width && sites > words.max_size() / width)
      std::abort();
    words.resize(sites * width);
  }
  bool merge(std::size_t to, const std::vector<uint64_t> &bits) {
    bool changed = false;
    for (std::size_t i = 0; i < width; ++i) {
      auto &w = words[to * width + i];
      const auto n = w | bits[i];
      changed |= n != w;
      w = n;
    }
    return changed;
  }
  std::vector<uint64_t> row(std::size_t site) const {
    return {words.begin() + site * width, words.begin() + (site + 1) * width};
  }
  bool test(std::size_t site, std::size_t bit) const {
    return (words[site * width + bit / 64] >> (bit % 64)) & 1;
  }
};
} // namespace storage_detail
struct StorageFrontierAnalysis::Impl {
  Program program;
  detail::ControlGraph graph;
  std::vector<std::vector<std::size_t>> predecessors;
  std::vector<bool> reachable;
  std::set<std::size_t> originalOwners;
  struct CellInfo {
    std::vector<StorageOrigin> origins;
    std::vector<std::size_t> writerOrigins;
    std::vector<std::size_t> originAt;
    std::vector<Access> accessAt;
    storage_detail::Matrix fw, fr, bw, br;
    mutable std::vector<bool> withoutFullWriter;
    mutable std::map<std::size_t, std::size_t> uniqueWriters;
    using SummaryKey = std::tuple<std::vector<std::size_t>, bool, bool>;
    mutable std::map<SummaryKey, std::map<std::size_t, unsigned>> useSummaries;
  };
  std::vector<CellInfo> cells;
  using UseQuery = std::tuple<unsigned, std::vector<std::size_t>,
                              std::vector<std::size_t>, bool>;
  std::map<UseQuery, PhysicalUseFrontier> useFrontiers;
  mutable StorageFrontierStats statistics;
  mutable std::vector<bool> repeatedSites;
  bool ok = false;
  std::string error;
  std::size_t operationAt(std::size_t s) const { return graph.operations[s]; }
  Cut cutAt(std::size_t s) const {
    return s < graph.legalCuts.size() && graph.legalCuts[s] ? s : NoAnalysisId;
  }
  StorageOrigin origin(std::size_t s) const {
    const auto c = cutAt(s);
    return {s, operationAt(s), c,
            c < graph.cutContexts.size() ? graph.cutContexts[c] : NoAnalysisId};
  }
  explicit Impl(Program p) : program(std::move(p)) {
    const auto valid = validateProgram(program);
    if (!valid.success) {
      error = valid.reason;
      return;
    }
    graph = detail::buildControlGraph(program);
    for (const auto& context : graph.contexts) { originalOwners.insert(context.ownerSite); }
    if (program.observed) {
      for (const auto& loop : program.observed->loops) { originalOwners.insert(loop.owner); }
    }
    const auto n = graph.sites.size();
    statistics.staticSites = n;
    predecessors.resize(n);
    reachable.assign(n, false);
    for (std::size_t s = 0; s < n; ++s)
      for (auto t : graph.sites[s].successors)
        predecessors[t].push_back(s);
    std::vector<std::size_t> todo{graph.entry};
    reachable[graph.entry] = true;
    while (!todo.empty()) {
      const auto s = todo.back();
      todo.pop_back();
      for (auto t : graph.sites[s].successors)
        if (!reachable[t]) {
          reachable[t] = true;
          todo.push_back(t);
        }
    }
    cells.resize(program.cells.size());
    for (unsigned c = 0; c < cells.size(); ++c) {
      auto &info = cells[c];
      info.originAt.assign(n, NoAnalysisId);
      info.accessAt.resize(n);
      for (std::size_t s = 0; s < n; ++s) {
        const auto op = operationAt(s);
        if (op == NoAnalysisId || !reachable[s])
          continue;
        Access a;
        a.cell = c;
        for (const auto &x : program.operations[op].accesses)
          if (x.cell == c) {
            a.read |= x.read;
            a.write |= x.write;
            a.definiteWrite |= x.write && x.definiteWrite;
          }
        info.accessAt[s] = a;
        if (a.read || a.write) {
          info.originAt[s] = info.origins.size();
          info.origins.push_back(origin(s));
          if (a.write) {
              info.writerOrigins.push_back(info.originAt[s]);
          }
        }
      }
      const auto count = info.origins.size();
      statistics.originIncidences += count;
      info.fw = {n, count};
      info.fr = {n, count};
      info.bw = {n, count};
      info.br = {n, count};
      statistics.storageWords += 4 * info.fw.words.size();
      solve(info, false);
      solve(info, true);
    }
    ok = true;
  }
  void solve(CellInfo &c, bool backward) {
    auto &w = backward ? c.bw : c.fw;
    auto &r = backward ? c.br : c.fr;
    std::deque<std::size_t> queue;
    std::vector<bool> queued = reachable;
    for (std::size_t s = 0; s < reachable.size(); ++s)
      if (reachable[s])
        queue.push_back(s);
    while (!queue.empty()) {
      const auto s = queue.front();
      queue.pop_front();
      queued[s] = false;
      if (backward)
        ++statistics.backwardEvaluations;
      else
        ++statistics.forwardEvaluations;
      auto ws = w.row(s), rs = r.row(s);
      const auto a = c.accessAt[s];
      if (a.write && a.definiteWrite) {
        std::fill(ws.begin(), ws.end(), 0);
        std::fill(rs.begin(), rs.end(), 0);
      }
      const auto bit = c.originAt[s];
      if (bit != NoAnalysisId) {
        if (a.write)
          ws[bit / 64] |= uint64_t(1) << (bit % 64);
        else if (a.read)
          rs[bit / 64] |= uint64_t(1) << (bit % 64);
      }
      const auto &targets =
          backward ? predecessors[s] : graph.sites[s].successors;
      for (auto t : targets)
        if (reachable[t]) {
          bool changed = w.merge(t, ws);
          changed = r.merge(t, rs) || changed;
          if (changed && !queued[t]) {
            queued[t] = true;
            queue.push_back(t);
          }
        }
    }
  }
  unsigned summarizeUses(unsigned cell, const std::vector<std::size_t>& starts,
      const std::vector<std::size_t>& stops, bool backward, bool includeStops) const {
    const auto& info = cells[cell];
    auto& cached = info.useSummaries[std::make_tuple(stops, backward, includeStops)];
    struct Node {
      unsigned roles = 0;
      bool terminal = false, queued = false;
      std::vector<std::size_t> dependents;
    };
    // Only complete nearest-access closures enter cached. A cached site cannot
    // depend on a newly discovered node, so independent demand expansion never
    // invalidates an earlier answer (including an access-free closed cycle).
    std::map<std::size_t, Node> nodes;
    std::vector<std::size_t> discover;
    auto add = [&](std::size_t site) {
      const bool unseen = reachable[site] && !cached.count(site);
      const bool added = unseen && nodes.try_emplace(site).second;
      if (added) {
        discover.push_back(site);
      }
    };
    for (auto site : starts) { add(site); }
    if (!discover.empty()) { ++statistics.useSummarySolves; }
    for (std::size_t at = 0; at < discover.size(); ++at) {
      const auto site = discover[at];
      auto& node = nodes.at(site);
      ++statistics.useSummarySites;
      const bool stop = std::binary_search(stops.begin(), stops.end(), site);
      const auto& access = info.accessAt[site];
      const auto& next = backward ? predecessors[site] : graph.sites[site].successors;
      if (stop && !includeStops) {
        node.roles = PhysicalUseSummary::IntervalStop;
      } else if (access.read && access.write) {
        node.roles = PhysicalUseSummary::ReadWrite;
      } else if (access.read) {
        node.roles = PhysicalUseSummary::Read;
      } else if (access.write) {
        node.roles = access.definiteWrite ? PhysicalUseSummary::FullWrite : PhysicalUseSummary::PartialWrite;
      } else if (stop) {
        node.roles = PhysicalUseSummary::IntervalStop;
      } else if (backward && site == graph.entry) {
        node.roles = PhysicalUseSummary::Entry;
      } else if (next.empty()) {
        node.roles = backward ? PhysicalUseSummary::Entry : PhysicalUseSummary::Exit;
      }
      node.terminal = node.roles != 0;
      if (node.terminal) { continue; }
      for (auto target : next) {
        ++statistics.useSummaryEdges;
        if (!reachable[target]) { continue; }
        const auto known = cached.find(target);
        if (known != cached.end()) {
          node.roles |= known->second;
        } else {
          add(target);
          nodes.at(target).dependents.push_back(site);
        }
      }
    }
    std::deque<std::size_t> pending;
    for (auto& [site, node] : nodes) {
      if (node.roles) { pending.push_back(site); node.queued = true; }
    }
    while (!pending.empty()) {
      const auto site = pending.front();
      pending.pop_front();
      auto& node = nodes.at(site);
      node.queued = false;
      ++statistics.useSummarySites;
      for (auto next : node.dependents) {
        ++statistics.useSummaryEdges;
        auto& dependent = nodes.at(next);
        const auto joined = dependent.roles | node.roles;
        if (joined == dependent.roles) { continue; }
        dependent.roles = joined;
        if (!dependent.queued) { pending.push_back(next); dependent.queued = true; }
      }
    }
    for (const auto& [site, node] : nodes) { cached.emplace(site, node.roles); }
    unsigned result = 0;
    for (auto site : starts) {
      const auto known = cached.find(site);
      if (known != cached.end()) { result |= known->second; }
    }
    return result;
  }
  std::vector<StorageOrigin> query(std::size_t s, unsigned cell,
                                   unsigned kind) const {
    std::vector<StorageOrigin> out;
    if (!ok || s >= reachable.size() || cell >= cells.size() || !reachable[s])
      return out;
    const auto &c = cells[cell];
    const auto &m = kind == 0   ? c.fw
                    : kind == 1 ? c.fr
                    : kind == 2 ? c.bw
                                : c.br;
    for (std::size_t i = 0; i < c.origins.size(); ++i)
      if (m.test(s, i))
        out.push_back(c.origins[i]);
    return out;
  }
  StoragePath path(std::size_t s, std::size_t t, unsigned cell) const {
    StoragePath out;
    if (!ok || s >= reachable.size() || t >= reachable.size() ||
        cell >= cells.size() || !reachable[s] || !reachable[t])
      return out;
    ++statistics.witnessQueries;
    std::vector<std::size_t> parent(reachable.size(), NoAnalysisId);
    std::deque<std::size_t> queue;
    for (auto v : graph.sites[s].successors)
      if (parent[v] == NoAnalysisId) {
        parent[v] = s;
        queue.push_back(v);
      }
    bool found = false;
    while (!queue.empty()) {
      const auto v = queue.front();
      queue.pop_front();
      ++statistics.witnessSites;
      if (v == t) {
        found = true;
        break;
      }
      const auto a = cells[cell].accessAt[v];
      if (a.write && a.definiteWrite)
        continue;
      for (auto u : graph.sites[v].successors)
        if (parent[u] == NoAnalysisId) {
          parent[u] = v;
          queue.push_back(u);
        }
    }
    if (!found)
      return out;
    out.exists = out.definiteWriteFree = true;
    std::size_t v = t;
    out.sites.push_back(v);
    do {
      v = parent[v];
      out.sites.push_back(v);
    } while (v != s);
    std::reverse(out.sites.begin(), out.sites.end());
    std::set<std::size_t> loops;
    for (std::size_t i = 0; i + 1 < out.sites.size(); ++i) {
      const auto a = out.sites[i], b = out.sites[i + 1];
      const auto &node = graph.sites[a];
      for (std::size_t j = 0; j < node.successors.size(); ++j)
        if (node.successors[j] == b && node.backedgeOwners[j] != NoAnalysisId)
          loops.insert(node.backedgeOwners[j]);
      if (i && cells[cell].accessAt[a].write)
        out.definiteWriteFree = false;
    }
    out.crossedLoopOwners.assign(loops.begin(), loops.end());
    return out;
  }
  // Graph queries deliberately distinguish zero-length reachability from a
  // repeated visit. They traverse original control, including enclosing edges.
  template <typename Stop>
  bool reaches(std::vector<std::size_t> starts, std::size_t target, Stop stop) const
  {
      std::vector<bool> seen(graph.sites.size());
      while (!starts.empty()) {
          const auto s = starts.back();
          starts.pop_back();
          if (seen[s] || !reachable[s] || stop(s))
              continue;
          seen[s] = true;
          if (s == target)
              return true;
          for (auto next : graph.sites[s].successors)
              starts.push_back(next);
      }
      return false;
  }
  bool repeated(std::size_t s) const
  {
      return reaches(graph.sites[s].successors, s, [](std::size_t) { return false; });
  }
  bool cyclicSite(std::size_t site) const
  {
      if (repeatedSites.empty()) {
          std::vector<std::size_t> membership;
          const auto groups = detail::strongComponents(graph, predecessors, reachable, membership);
          repeatedSites.resize(graph.sites.size());
          statistics.classificationSites += graph.sites.size();
          for (const auto& group : groups) {
              const bool cyclic = group.size() > 1 ||
                  std::find(graph.sites[group.front()].successors.begin(),
                            graph.sites[group.front()].successors.end(), group.front()) !=
                      graph.sites[group.front()].successors.end();
              for (auto at : group) {
                  repeatedSites[at] = cyclic;
              }
          }
      }
      return repeatedSites[site];
  }
  bool uninitialized(std::size_t site, unsigned cell) const
  {
      auto& absent = cells[cell].withoutFullWriter;
      if (absent.empty()) {
          absent.resize(graph.sites.size());
          std::vector<std::size_t> pending{graph.entry};
          absent[graph.entry] = true;
          while (!pending.empty()) {
              const auto at = pending.back();
              pending.pop_back();
              ++statistics.classificationSites;
              // This is the incoming fact: a write at the target cannot
              // initialize its own read. Only outgoing paths are stopped.
              if (cells[cell].accessAt[at].definiteWrite) {
                  continue;
              }
              for (auto next : graph.sites[at].successors) {
                  if (!absent[next]) {
                      absent[next] = true;
                      pending.push_back(next);
                  }
              }
          }
      }
      return absent[site];
  }
  std::size_t uniqueWriter(std::size_t site, unsigned cell) const
  {
      const auto& info = cells[cell];
      const auto cached = info.uniqueWriters.find(site);
      if (cached != info.uniqueWriters.end()) {
          return cached->second;
      }
      auto writer = NoAnalysisId;
      for (auto i : info.writerOrigins) {
          ++statistics.classificationOrigins;
          if (info.fw.test(site, i)) {
              if (writer != NoAnalysisId) {
                  writer = NoAnalysisId;
                  break;
              }
              writer = info.origins[i].site;
          }
      }
      info.uniqueWriters.emplace(site, writer);
      return writer;
  }
  unsigned classify(const StorageRelationship& relationship) const
  {
      ++statistics.classificationQueries;
      const auto s = relationship.source.site, t = relationship.target.site;
      const auto cell = relationship.cell;
      unsigned flags = AdditionalOverlap;
      if (!ok || s >= reachable.size() || t >= reachable.size() || cell >= cells.size() ||
          !reachable[s] || !reachable[t]) {
          return flags;
      }
      const auto& info = cells[cell];
      const auto& source = info.accessAt[s];
      const auto& target = info.accessAt[t];
      const auto index = info.originAt[s];
      // Original incoming provenance encodes positive-length paths with no
      // intervening definite overwrite. RMW origins occupy the writer row.
      if (index == NoAnalysisId || !(source.write ? info.fw : info.fr).test(t, index)) {
          return flags;
      }
      const bool raw = relationship.kind == StorageRelationship::RAW && source.write && target.read;
      const bool reuse = target.write && ((relationship.kind == StorageRelationship::WAR && source.read) ||
                                         (relationship.kind == StorageRelationship::WAW && source.write));
      if (reuse && program.cells[cell].storage == Cell::Storage::CanonicalInterval) {
          flags |= KnownReuse;
      }
      // A unique full writer dominates exactly when there is no uninitialized
      // entry path: every other path contributes its last full writer to fw.
      if (raw && source.definiteWrite && s != t && uniqueWriter(t, cell) == s &&
          !uninitialized(t, cell) && !cyclicSite(s) && !cyclicSite(t)) {
          flags |= KnownReadiness;
      }
      return flags;
  }
  StorageLifecycle lifecycle(std::size_t s, unsigned cell) const
  {
      StorageLifecycle out;
      out.cell = cell;
      if (!ok || s >= reachable.size() || cell >= cells.size() || !reachable[s])
          return out;
      out.reachable = true;
      out.access = origin(s);
      out.previousWriters = query(s, cell, 0);
      out.previousReaders = query(s, cell, 1);
      out.nextWriters = query(s, cell, 2);
      out.nextReaders = query(s, cell, 3);
      auto context = out.access.context;
      while (context != NoAnalysisId && context < graph.contexts.size()) {
          const auto& scope = graph.contexts[context];
          if (scope.kind == AnalysisContext::ForBody || scope.kind == AnalysisContext::WhileBefore ||
              scope.kind == AnalysisContext::WhileAfter)
              out.enclosingLoops.push_back(scope.ownerSite);
          if (context == 0 || scope.parent == context)
              break;
          context = scope.parent;
      }
      for (const auto& o : cells[cell].origins)
          out.participantEngines |= 1u << unsigned(program.operations[o.operation].pipe);
      out.mayHaveNoPriorFullWrite =
          reaches({graph.entry}, s, [&](std::size_t at) { return at != s && cells[cell].accessAt[at].definiteWrite; });
      out.mayExitWithoutFurtherAccess = reaches(graph.sites[s].successors, graph.exit, [&](std::size_t at) {
          const auto& a = cells[cell].accessAt[at];
          return a.read || a.write;
      });
      return out;
  }
  RequirementProvenance describe(const StorageRelationship& r) const
  {
      RequirementProvenance out;
      const auto s = r.source.site, t = r.target.site;
      if (!ok || s >= reachable.size() || t >= reachable.size() || r.cell >= cells.size() || !reachable[s] ||
          !reachable[t])
          return out;
      const auto witness = path(s, t, r.cell);
      if (!witness.exists)
          return out;
      out.occurrence.crossedLoopOwners = witness.crossedLoopOwners;
      if (program.observed) {
          out.occurrence.sourceObservation = program.observed->sites[s].observation;
          out.occurrence.targetObservation = program.observed->sites[t].observation;
      }
      const bool dominates = !reaches({graph.entry}, t, [&](std::size_t at) { return at == s; });
      if (s != t && dominates && !repeated(s) && !repeated(t))
          out.occurrence.kind = OccurrenceQualification::SingleVisit;
      out.reasons = classify(r);
      return out;
  }
};
} // namespace mlir::pto::oahs
#endif
