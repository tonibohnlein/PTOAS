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
  Impl(Program p, Commands c)
      : program(std::move(p)), commands(std::move(c)),
        report(analyze(program, commands)) {
    if (!report.complete)
      return;
    graph = detail::buildControlGraph(program);
    requirementsAt.resize(commandCutCount(program));
    for (std::size_t i = 0; i < report.residuals.size(); ++i)
      requirementsAt[report.residuals[i].consumerCut].push_back(i);
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
    return a < commandCutCount(program) && b < commandCutCount(program) &&
           canonicalCommandCut(program, a) == canonicalCommandCut(program, b);
  }
  std::vector<CompletionRequirement> requirements(Cut consumer) const {
    std::vector<CompletionRequirement> result;
    if (consumer < requirementsAt.size())
      for (Cut c = 0; c < requirementsAt.size(); ++c)
        if (sameObservation(c, consumer))
          for (auto i : requirementsAt[c])
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
    std::vector<std::array<bool, 2>> seen(graph.sites.size());
    std::deque<std::pair<std::size_t, bool>> work;
    for (Cut c = 0; c < commandCutCount(program); ++c)
      if (sameObservation(c, consumer)) {
        work.push_back({c, false});
        seen[c][0] = true;
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
      if (!physicalCut(cut) || canonicalCommandCut(program, cut) != cut)
        continue;
      bool without = false, through = false;
      for (Cut member = 0; member < commandCutCount(program); ++member)
        if (sameObservation(cut, member) && report.cuts[member].reachable) {
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
    return out;
  }
  AnalysisBits sourceRemainder(Pipe source, Cut at) const {
    AnalysisBits bits(program.operations.size());
    for (Cut c = 0; c < commandCutCount(program); ++c)
      if (sameObservation(c, at) && report.cuts[c].incoming)
        for (std::size_t i = 0; i < bits.size(); ++i)
          bits[i] |= report.cuts[c].incoming->pending[unsigned(source)][i];
    for (std::size_t i = 0; i < bits.size(); ++i)
      if (program.operations[i].pipe == source)
        bits[i] = 0;
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
    const auto n = graph.exit;
    // 0 unreachable, 1 empty, 2 full; join is union. Every site can gain each
    // balance bit at most once. This proves reference matching, not safe rearm.
    std::vector<uint8_t> incoming(graph.sites.size());
    incoming[graph.entry] = 1;
    std::deque<std::size_t> work{graph.entry};
    std::vector<bool> queued(graph.sites.size());
    queued[graph.entry] = true;
    while (!work.empty()) {
      const auto id = work.front();
      work.pop_front();
      queued[id] = false;
      auto next = incoming[id];
      ++out.siteEvaluations;
      if (sameObservation(id, publication))
        next = 2;
      if (sameObservation(id, consumer))
        next = 1;
      for (auto successor : graph.sites[id].successors) {
        const auto joined = uint8_t(incoming[successor] | next);
        if (joined != incoming[successor]) {
          incoming[successor] = joined;
          if (!queued[successor]) {
            queued[successor] = true;
            work.push_back(successor);
          }
        }
      }
    }
    bool sourceReachable = false, targetReachable = false;
    for (Cut at = 0; at < commandCutCount(program); ++at)
      if (incoming[at]) {
        sourceReachable |= sameObservation(at, publication);
        targetReachable |= sameObservation(at, consumer);
      }
    if (!sourceReachable || !targetReachable || !incoming[n]) {
      out.reason = "prospective endpoint or invocation exit is unreachable";
      return out;
    }
    out.availableAtEveryAcquisition = true;
    for (Cut c = 0; c < commandCutCount(program); ++c)
      if (incoming[c]) {
        if (sameObservation(c, publication))
          out.repeatedPublication |= incoming[c] != 1;
        if (sameObservation(c, consumer))
          out.availableAtEveryAcquisition &=
              (sameObservation(c, publication) ? 2 : incoming[c]) == 2;
      }
    out.unconsumedAtExit = incoming[n] != 1;
    out.matchingEstablished = !out.repeatedPublication &&
                              out.availableAtEveryAcquisition &&
                              !out.unconsumedAtExit;
    if (out.availableAtEveryAcquisition) {
      out.uncoveredOperations = seed;
      // Every original effect issued after capture and before the acquisition
      // enters the remainder, even a new visit of the same static source phase.
      // Stop BEFORE the consumer payload. Existing events/fences cannot enlarge
      // this immutable receipt retroactively. Revisited cuts need no unrolling.
      std::vector<bool> seen(graph.sites.size());
      std::vector<std::size_t> todo;
      for (Cut c = 0; c < commandCutCount(program); ++c)
        if (sameObservation(c, publication) && incoming[c])
          todo.push_back(c);
      while (!todo.empty()) {
        const auto id = todo.back();
        todo.pop_back();
        if (sameObservation(id, consumer) || seen[id])
          continue;
        seen[id] = true;
        ++out.siteEvaluations;
        if (graph.operations[id] != NoAnalysisId)
          out.uncoveredOperations[graph.operations[id]] = 1;
        for (auto next : graph.sites[id].successors)
          todo.push_back(next);
      }
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
