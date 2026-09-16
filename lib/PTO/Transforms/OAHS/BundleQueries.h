// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_BUNDLE_QUERIES_H
#define PTO_OAHS_BUNDLE_QUERIES_H
#include "PTO/Transforms/OAHS/Bundles.h"
#include <map>
#include <tuple>

namespace mlir::pto::oahs {
namespace detail {
inline auto requirementKey(const CompletionRequirement &r) {
  return std::make_tuple(r.demand.producer, r.demand.consumer, r.demand.cell,
                         r.kind, r.demand.property, r.consumerCut,
                         r.consumerContext);
}
inline auto protocolKey(const ProtocolObligation &r) {
  return std::make_tuple(r.cut, r.event.source, r.event.observer, r.event.key,
                         r.kind);
}
inline auto retirementKey(const RetirementRequirement &r) {
  return std::make_pair(r.operation, r.observer);
}
template <class T, class KeyFn>
std::vector<T> difference(const std::vector<T> &a, const std::vector<T> &b,
                          KeyFn key) {
  using Key = decltype(key(std::declval<T>()));
  std::map<Key, std::size_t> counts;
  for (const auto &v : b)
    ++counts[key(v)];
  std::vector<T> result;
  for (const auto &v : a) {
    auto found = counts.find(key(v));
    if (found == counts.end() || !found->second)
      result.push_back(v);
    else
      --found->second;
  }
  return result;
}
inline bool sameCommands(const std::vector<Command> &a,
                         const std::vector<Command> &b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::tie(a[i].kind, a[i].source, a[i].observer, a[i].key) !=
        std::tie(b[i].kind, b[i].source, b[i].observer, b[i].key))
      return false;
  return true;
}
} // namespace detail
struct BundleQuery::Impl {
  const Program program;
  const Commands commands;
  const AnalysisOptions options;
  const AnalysisResult before;
  Impl(Program p, Commands c, AnalysisOptions o)
      : program(std::move(p)), commands(std::move(c)), options(o),
        before(analyze(program, commands, options)) {}
  BundleEvaluation evaluate(Commands candidate) const {
    BundleEvaluation out;
    out.commands = std::move(candidate);
    if (!before.complete) {
      out.reason = "invalid base plan: " + before.reason;
      return out;
    }
    out.analysis = analyze(program, out.commands, options);
    out.complete = out.analysis.complete;
    out.reason = out.analysis.reason;
    if (!out.complete)
      return out;
    out.discharged = detail::difference(
        before.residuals, out.analysis.residuals, detail::requirementKey);
    out.introduced = detail::difference(
        out.analysis.residuals, before.residuals, detail::requirementKey);
    out.resolvedProtocol = detail::difference(
        before.protocol, out.analysis.protocol, detail::protocolKey);
    out.introducedProtocol = detail::difference(
        out.analysis.protocol, before.protocol, detail::protocolKey);
    out.retired = detail::difference(before.retirement, out.analysis.retirement,
                                     detail::retirementKey);
    out.introducedRetirement = detail::difference(
        out.analysis.retirement, before.retirement, detail::retirementKey);
    out.resources.keys = out.analysis.keys;
    for (Cut cut = 0; cut < out.commands.size(); ++cut) {
      if (!detail::sameCommands(commands[cut], out.commands[cut]))
        out.changedCuts.push_back(cut);
      if (canonicalCommandCut(program, cut) == cut)
        for (const auto &c : out.commands[cut]) {
          out.resources.publications += c.kind == Command::Publish;
          out.resources.acquisitions += c.kind == Command::Acquire;
          out.resources.fences += c.kind == Command::Barrier;
          out.resources.allFences += c.kind == Command::BarrierAll;
        }
    }
    return out;
  }
};
} // namespace mlir::pto::oahs
#endif
