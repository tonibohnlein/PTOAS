// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_PREFIX_QUERIES_H
#define PTO_OAHS_PREFIX_QUERIES_H
#include "Control.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include <algorithm>
#include <deque>
#include <set>
#include <map>
#include <tuple>

namespace mlir::pto::oahs {
struct PrefixQuery::Impl {
  const Program program;
  const Commands commands;
  const AnalysisResult report;
  detail::ControlGraph graph;
  struct Predecessor {
    std::size_t site, loop;
  };
  std::vector<std::vector<Predecessor>> predecessors;
  std::vector<std::vector<std::size_t>> requirementsAt;
  std::vector<Cut> canonical;
  std::vector<std::vector<Cut>> members;
  mutable PrefixQueryStats stats;
  mutable std::map<Cut, BackwardCutResult> backwardCache;
  mutable std::map<std::pair<Pipe, Cut>, AnalysisBits> sourceCache;
  struct PairInfo {
    bool reachable = false, repeated = false, available = false, liveExit = false;
    AnalysisBits intervening;
    std::size_t evaluations = 0;
  };
  mutable std::map<std::pair<Cut, Cut>, PairInfo> pairCache;
  Impl(Program p, Commands c)
      : program(std::move(p)), commands(std::move(c)),
        report(analyze(program, commands)) {
    if (!report.complete)
      return;
    graph = detail::buildControlGraph(program);
    const auto count = commandCutCount(program);
    requirementsAt.resize(count); canonical.resize(count); members.resize(count);
    std::map<std::size_t, Cut> leaders;
    for (Cut c = 0; c < count; ++c) {
      canonical[c] = c;
      if (program.observed && legalCommandCut(program, c)) {
        const auto observation = program.observed->sites[c].observation;
        canonical[c] = leaders.emplace(observation, c).first->second;
      }
      members[canonical[c]].push_back(c);
    }
    std::vector<std::vector<std::size_t>> perSite(count);
    for (std::size_t i = 0; i < report.residuals.size(); ++i)
      perSite[report.residuals[i].consumerCut].push_back(i);
    // Preserve the original per-cut ordering, including collecting phase reports
    // whose residuals may have been discovered in a different worklist order.
    for (Cut c = 0; c < count; ++c)
      requirementsAt[canonical[c]].insert(requirementsAt[canonical[c]].end(),
                                         perSite[c].begin(), perSite[c].end());
    predecessors.resize(graph.sites.size());
    for (std::size_t from = 0; from < graph.sites.size(); ++from) {
      const auto &site = graph.sites[from];
      for (std::size_t i = 0; i < site.successors.size(); ++i)
        predecessors[site.successors[i]].push_back(
            {from, site.backedgeOwners[i]});
    }
  }
  bool physicalCut(Cut cut) const { return legalCommandCut(program, cut); }
  bool sameObservation(Cut a, Cut b) const {
    return a < canonical.size() && b < canonical.size() &&
           canonical[a] == canonical[b];
  }
  std::vector<CompletionRequirement> requirements(Cut consumer) const {
    std::vector<CompletionRequirement> result;
    if (consumer < requirementsAt.size())
      for (auto i : requirementsAt[canonical[consumer]])
        result.push_back(report.residuals[i]);
    return result;
  }
  BackwardCutResult backward(Cut consumer) const {
    BackwardCutResult out;
    out.consumer = consumer;
    if (!report.complete) {
      out.reason = report.reason;
      return out;
    }
    if (!physicalCut(consumer) ||
        operationAtCut(program, consumer) == NoControlId) {
      out.reason = "consumer is not an original physical cut";
      return out;
    }
    const auto cached = backwardCache.find(canonical[consumer]);
    if (cached != backwardCache.end()) {
      ++stats.backwardHits;
      out = cached->second; out.consumer = consumer; return out;
    }
    ++stats.backwardTraversals;
    std::vector<std::array<bool, 2>> seen(graph.sites.size());
    std::deque<std::pair<std::size_t, bool>> work;
    for (Cut c : members[canonical[consumer]]) {
      work.push_back({c, false}); seen[c][0] = true;
    }
    std::set<std::size_t> loops;
    while (!work.empty()) {
      const auto [site, across] = work.front();
      work.pop_front();
      ++out.visitedStates;
      for (const auto &pred : predecessors[site]) {
        const bool next = across || pred.loop != NoAnalysisId;
        if (pred.loop != NoAnalysisId)
          loops.insert(pred.loop);
        if (!seen[pred.site][next]) {
          seen[pred.site][next] = true;
          work.push_back({pred.site, next});
        }
      }
    }
    for (Cut cut = 0; cut < commandCutCount(program); ++cut) {
      if (!physicalCut(cut) || canonical[cut] != cut)
        continue;
      bool without = false, through = false;
      for (Cut member : members[cut])
        if (report.cuts[member].reachable) {
          without |= seen[member][0];
          through |= seen[member][1];
        }
      if (without || through)
        out.cuts.push_back({cut, graph.cutContexts[cut], without, through});
    }
    std::sort(out.cuts.begin(), out.cuts.end(),
              [&](const BackwardCut &a, const BackwardCut &b) {
                return graph.cutRanks[a.cut] < graph.cutRanks[b.cut];
              });
    out.crossedLoopOwners.assign(loops.begin(), loops.end());
    out.complete = true;
    backwardCache.emplace(canonical[consumer], out);
    return out;
  }
  AnalysisBits sourceRemainder(Pipe source, Cut at) const {
    const auto key = std::make_pair(source, canonical[at]);
    auto cached = sourceCache.find(key);
    if (cached != sourceCache.end()) { ++stats.sourceHits; return cached->second; }
    ++stats.sourceSnapshots;
    AnalysisBits bits(program.operations.size());
    for (Cut c : members[canonical[at]])
      if (report.cuts[c].incoming)
        for (std::size_t i = 0; i < bits.size(); ++i)
          bits[i] |= report.cuts[c].incoming->pending[unsigned(source)][i];
    for (std::size_t i = 0; i < bits.size(); ++i)
      if (program.operations[i].pipe == source)
        bits[i] = 0;
    sourceCache.emplace(key, bits);
    return bits;
  }
  bool eligible(Pipe source, Pipe observer, Cut publication,
                Cut consumer) const {
    const auto a = unsigned(source), b = unsigned(observer);
    if (a == b)
      return publication == consumer && program.target.barriers[a];
    for (auto key : program.target.keys[a][b]) {
      const bool reserved = std::any_of(
          program.reservations.begin(), program.reservations.end(),
          [&](const EventIdentity &r) {
            return r.source == source && r.observer == observer && r.key == key;
          });
      if (!reserved)
        return true;
    }
    return false;
  }
  const PairInfo &correspondence(Cut publication, Cut consumer) const {
    const auto key = std::make_pair(canonical[publication], canonical[consumer]);
    const auto cached = pairCache.find(key);
    if (cached != pairCache.end()) { ++stats.correspondenceHits; return cached->second; }
    ++stats.correspondenceTraversals;
    PairInfo info;
    info.intervening.resize(program.operations.size());
    std::vector<uint8_t> incoming(graph.sites.size());
    incoming[graph.entry] = 1;
    std::deque<std::size_t> queue{graph.entry};
    std::vector<bool> queued(graph.sites.size()); queued[graph.entry] = true;
    while (!queue.empty()) {
      const auto id = queue.front(); queue.pop_front(); queued[id] = false;
      auto next = incoming[id]; ++info.evaluations;
      if (sameObservation(id, publication)) next = 2;
      if (sameObservation(id, consumer)) next = 1;
      for (auto successor : graph.sites[id].successors) {
        const auto joined = uint8_t(incoming[successor] | next);
        if (joined != incoming[successor]) {
          incoming[successor] = joined;
          if (!queued[successor]) { queued[successor] = true; queue.push_back(successor); }
        }
      }
    }
    bool sourceReachable = false, targetReachable = false;
    info.available = true;
    for (Cut c : members[key.first]) if (incoming[c]) {
      sourceReachable = true; info.repeated |= incoming[c] != 1;
    }
    for (Cut c : members[key.second]) if (incoming[c]) {
      targetReachable = true;
      info.available &= (sameObservation(c, publication) ? 2 : incoming[c]) == 2;
    }
    info.reachable = sourceReachable && targetReachable && incoming[graph.exit];
    info.liveExit = incoming[graph.exit] != 1;
    if (info.reachable && info.available) {
      std::vector<bool> seen(graph.sites.size());
      std::vector<std::size_t> todo;
      for (Cut c : members[key.first]) if (incoming[c]) todo.push_back(c);
      while (!todo.empty()) {
        const auto id = todo.back(); todo.pop_back();
        if (sameObservation(id, consumer) || seen[id]) continue;
        seen[id] = true; ++info.evaluations;
        if (graph.operations[id] != NoAnalysisId) info.intervening[graph.operations[id]] = 1;
        for (auto next : graph.sites[id].successors) todo.push_back(next);
      }
    }
    return pairCache.emplace(key, std::move(info)).first->second;
  }
  // Reference alternation is a two-bit finite monitor, independent of the
  // receipt's payload bit set. Freshness is then a may-access reachability
  // query between these two cuts. We do not propagate N-bit ghost states at
  // every site for every candidate, and never allocate a physical key for the
  // query.
  ProspectivePrefix inspect(Pipe source, Cut publication, Cut consumer,
                            const BackwardCutResult &backward) const {
    ProspectivePrefix out;
    out.source = source;
    out.publication = publication;
    out.acquisition = consumer;
    if (!report.complete) {
      out.reason = report.reason;
      return out;
    }
    if (!physicalCut(publication) || !physicalCut(consumer) ||
        operationAtCut(program, consumer) == NoControlId ||
        unsigned(source) >= PipeCount ||
        !program.target.supported[unsigned(source)]) {
      out.reason = "invalid prospective source or physical cut";
      return out;
    }
    out.complete = true;
    out.observer = program.operations[operationAtCut(program, consumer)].pipe;
    out.sourceContext = graph.cutContexts[publication];
    out.targetContext = graph.cutContexts[consumer];
    out.routeAvailable = eligible(source, out.observer, publication, consumer);
    out.uncoveredOperations.assign(program.operations.size(), 1);
    auto where = std::find_if(
        backward.cuts.begin(), backward.cuts.end(),
        [&](const BackwardCut &c) { return c.cut == publication; });
    if (where == backward.cuts.end()) {
      out.reason =
          "publication has no backward predecessor path to the consumer";
      return out;
    }
    out.crossesBackedge = where->throughBackedge;
    const auto seed = sourceRemainder(source, publication);
    const auto &info = correspondence(publication, consumer);
    out.siteEvaluations = info.evaluations;
    if (!info.reachable) {
      out.reason = "prospective endpoint or invocation exit is unreachable";
      return out;
    }
    out.availableAtEveryAcquisition = info.available;
    out.repeatedPublication = info.repeated;
    out.unconsumedAtExit = info.liveExit;
    out.matchingEstablished = !info.repeated && info.available && !info.liveExit;
    if (info.available) {
      out.uncoveredOperations = seed;
      for (std::size_t i = 0; i < seed.size(); ++i)
        out.uncoveredOperations[i] |= info.intervening[i];
    }
    if (out.matchingEstablished) {
      const auto required = requirements(consumer);
      for (std::size_t i = 0; i < required.size(); ++i)
        if (!out.uncoveredOperations[required[i].demand.producer])
          out.coveredRequirements.push_back(i);
    }
    if (!out.availableAtEveryAcquisition)
      out.reason = "source participation/preceding occurrence is not "
                   "established at every acquisition";
    else if (out.repeatedPublication)
      out.reason =
          "source can execute twice without its corresponding acquisition";
    else if (out.unconsumedAtExit)
      out.reason = "source has an unmatched final or bypass publication";
    else if (!out.routeAvailable)
      out.reason =
          "no eligible directional key or supported consumer-cut fence";
    else if (out.coveredRequirements.empty())
      out.reason = "prospective prefix covers no residual component";
    return out;
  }
};

} // namespace mlir::pto::oahs
#endif
