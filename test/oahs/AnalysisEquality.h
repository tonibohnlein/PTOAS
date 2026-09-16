// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_TEST_ANALYSIS_EQUALITY_H
#define PTO_OAHS_TEST_ANALYSIS_EQUALITY_H
#include "PTO/Transforms/OAHS/Analysis.h"
#include <sstream>
#include <iomanip>
namespace oahs_test {
// Stable complete public-report serialization, deliberately excluding only
// work counters. This is a TEST comparison, not a cache key or proof input.
inline std::string report(const mlir::pto::oahs::AnalysisResult &r) {
  namespace o = mlir::pto::oahs;
  std::ostringstream s;
  auto event = [&](const o::EventIdentity &e) {
    s << unsigned(e.source) << ',' << unsigned(e.observer) << ',' << e.key << ',' << e.privateIdentity << ';';
  };
  auto bits = [&](const o::AnalysisBits &b) {
    s << b.size() << ':'; for (auto x : b) s << unsigned(x); s << ';';
  };
  auto access = [&](const o::Access &a) { s << a.cell << ',' << a.read << ',' << a.write << ',' << a.definiteWrite << ';'; };
  auto boundary = [&](const std::optional<o::BoundaryFacts> &b) {
    s << bool(b) << '{';
    if (b) {
      for (const auto &p : b->pending) bits(p);
      s << b->events.size() << ':';
      for (const auto &e : b->events) {
        s << unsigned(e.possibleOccupancy) << ',' << e.receiptValidOnEveryPath << ',' << unsigned(e.consumptionKnownAt) << ';';
        bits(e.uncoveredOperations); bits(e.carriedConsumptions);
      }
      s << b->phaseResources.size() << ':';
      for (const auto &p : b->phaseResources) {
        s << p.cell << ',' << p.possiblePermission << ';';
        bits(p.completedOperations); bits(p.carriedConsumptions);
      }
    }
    s << '}';
  };
  s << r.complete << ',' << r.verified() << ',' << std::quoted(r.reason);
  s << "D" << r.diagnostics.size();
  for (const auto &d : r.diagnostics) s << ':' << d.kind << ',' << d.operation << ',' << std::quoted(d.reason);
  s << "C" << r.contexts.size();
  for (const auto &c : r.contexts) s << ':' << c.kind << ',' << c.parent << ',' << c.ownerSite;
  s << "K" << r.keys.size(); for (const auto &k : r.keys) event(k);
  s << "B" << r.cuts.size();
  for (const auto &c : r.cuts) {
    s << c.cut << ',' << c.context << ',' << c.reachable << ';';
    boundary(c.incoming); boundary(c.beforeIssue); boundary(c.outgoing);
  }
  s << "E" << r.commands.size();
  for (const auto &c : r.commands) s << c.cut << ',' << c.command << ',' << c.context << ',' << c.endpoint.kind << ','
    << unsigned(c.endpoint.source) << ',' << unsigned(c.endpoint.observer) << ',' << c.endpoint.key << ',' << c.preconditionsEstablished << ';';
  s << "R" << r.residuals.size();
  for (const auto &c : r.residuals) {
    s << c.kind << ',' << c.demand.producer << ',' << c.demand.consumer << ',' << c.demand.cell << ',' << unsigned(c.demand.property) << ';';
    access(c.producerAccess); access(c.consumerAccess);
    s << c.producerContext << ',' << c.consumerContext << ',' << c.consumerCut << ',' << c.producerEndpoint << ',' << c.consumerEndpoint << ';';
  }
  s << "P" << r.protocol.size();
  for (const auto &p : r.protocol) { s << p.kind << ',' << p.cut << ',' << p.command << ',' << p.context << ';'; event(p.event); s << std::quoted(p.reason); }
  s << "T" << r.retirement.size(); for (const auto &t : r.retirement) s << t.operation << ',' << unsigned(t.observer) << ';';
  s << "F" << r.phaseResources.size(); for (const auto &p : r.phaseResources) s << p.kind << ',' << p.cut << ',' << p.operation << ',' << p.cell << ',' << std::quoted(p.reason);
  return s.str();
}
}
#endif
