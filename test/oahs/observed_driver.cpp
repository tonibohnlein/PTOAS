// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "ObservedFixtures.h"
#include "PTO/Transforms/OAHS/Analysis.h"
#include <iostream>
#include <string>
namespace o = mlir::pto::oahs;
int main(int argc, char **argv) {
  if (argc < 2)
    return 2;
  const std::string name = argv[1];
  o::ObservedImport input;
  if (name == "ring")
    input = observed_fixtures::ring(argc > 2 ? std::stoul(argv[2]) : 2);
  else if (name == "refined")
    input = observed_fixtures::refinedRing(argc > 2 ? std::stoul(argv[2]) : 2);
  else if (name == "stride")
    input = observed_fixtures::ring(2, true);
  else if (name == "nested")
    input = observed_fixtures::nested();
  else if (name == "mixed")
    input = observed_fixtures::mixed();
  else if (name == "independent")
    input = observed_fixtures::independent();
  else
    return 2;
  if (!input.success)
    return 3;
  const auto &p = input.program;
  auto plan = o::construct(p);
  if (!plan.success)
    return 4;
  std::size_t selectedSlots = 0;
  const auto mutation = argc > 3 ? std::stoul(argv[3]) : 0;
  for (o::Cut c = 0; c < plan.commands.size(); ++c)
    if (o::canonicalCommandCut(p, c) == c) {
      for (std::size_t j = 0; j < plan.commands[c].size(); ++j) {
        ++selectedSlots;
        if (mutation == selectedSlots)
          for (o::Cut d = 0; d < plan.commands.size(); ++d)
            if (o::canonicalCommandCut(p, d) == c)
              plan.commands[d].erase(plan.commands[d].begin() + j);
        if (mutation == selectedSlots)
          break;
      }
    }
  auto report = o::analyze(p, plan.commands);
  std::cout << "{\"slots\":" << selectedSlots
            << ",\"verified\":" << (report.verified() ? "true" : "false")
            << ",\"case\":{\"name\":\"" << name << "\",\"lanes\":[";
  bool comma = false;
  for (unsigned i = 0; i < o::PipeCount; ++i)
    if (p.target.supported[i]) {
      if (comma)
        std::cout << ',';
      comma = true;
      std::cout << i;
    }
  std::cout << "],\"cells\":" << p.cells.size() << ",\"ops\":[";
  for (unsigned i = 0; i < p.operations.size(); ++i) {
    if (i)
      std::cout << ',';
    const auto &a = p.operations[i];
    std::cout << "{\"pipe\":" << unsigned(a.pipe) << ",\"accesses\":[";
    for (unsigned j = 0; j < a.accesses.size(); ++j) {
      if (j)
        std::cout << ',';
      auto x = a.accesses[j];
      std::cout << '[' << x.cell << ',' << x.read << ',' << x.write << ']';
    }
    std::cout << "]}";
  }
  std::cout << "],\"observed\":{\"entry\":" << p.observed->entry
            << ",\"exit\":" << p.observed->exit << ",\"sites\":[";
  for (unsigned i = 0; i < p.observed->sites.size(); ++i) {
    if (i)
      std::cout << ',';
    auto &s = p.observed->sites[i];
    std::cout << "{\"operation\":";
    if (s.operation == o::NoControlId)
      std::cout << "null";
    else
      std::cout << s.operation;
    std::cout << ",\"observation\":";
    if (s.observation == o::NoControlId)
      std::cout << "null";
    else
      std::cout << s.observation;
    std::cout << ",\"successors\":[";
    for (unsigned j = 0; j < s.successors.size(); ++j) {
      if (j)
        std::cout << ',';
      std::cout << s.successors[j];
    }
    std::cout << "]}";
  }
  std::cout << "],\"observations\":[";
  for (unsigned i = 0; i < p.observed->observations.size(); ++i) {
    if (i)
      std::cout << ',';
    const auto &x = p.observed->observations[i];
    std::cout << "{\"anchor\":" << x.anchor
              << ",\"available\":" << (x.available ? "true" : "false")
              << ",\"atoms\":[";
    for (unsigned j = 0; j < x.atoms.size(); ++j) {
      if (j)
        std::cout << ',';
      const auto &a = x.atoms[j];
      std::cout << '[' << unsigned(a.kind) << ',' << a.owner << ','
                << a.parameter << ',' << a.value << ']';
    }
    std::cout << "]}";
  }
  std::cout << "]}},\"commands\":[";
  for (unsigned c = 0; c < plan.commands.size(); ++c) {
    if (c)
      std::cout << ',';
    std::cout << '[';
    for (unsigned j = 0; j < plan.commands[c].size(); ++j) {
      if (j)
        std::cout << ',';
      auto x = plan.commands[c][j];
      std::cout << '[' << unsigned(x.kind) << ',' << unsigned(x.source) << ','
                << unsigned(x.observer) << ',' << x.key << ']';
    }
    std::cout << ']';
  }
  std::cout << "]}\n";
  return 0;
}
