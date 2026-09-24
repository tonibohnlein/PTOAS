// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Corpus runner for the existing Phase A query boundary. Success means that
// preparation and queries ran, not that unresolved facts or Phase B are complete.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace {
namespace fs = mlir::pto::frontiersynch;

std::string printIR(mlir::Operation *operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation->print(stream);
  return result;
}

struct Counts {
  std::array<std::size_t, 3> hazards{};
  std::size_t interpreted = 0;
  std::size_t unknownOccurrence = 0;
  std::size_t decodedUnresolved = 0;
  std::size_t readers = 0;
};

using Demand = std::tuple<std::size_t, std::size_t, fs::FactoredUseNode::Hazard>;

std::vector<std::size_t> origins(const fs::FactoredUseResult &result, std::size_t id,
                                 const std::map<std::size_t, bool> &choices) {
  const auto &node = result.nodes.at(id);
  switch (node.kind) {
  case fs::FactoredUseNode::Kind::Empty:
    return {};
  case fs::FactoredUseNode::Kind::Incoming:
    return {fs::NoControlId};
  case fs::FactoredUseNode::Kind::Access:
    return {node.operation};
  case fs::FactoredUseNode::Kind::Both: {
    auto left = origins(result, node.left, choices);
    auto right = origins(result, node.right, choices);
    left.insert(left.end(), right.begin(), right.end());
    return left;
  }
  case fs::FactoredUseNode::Kind::Choose:
    return origins(result, choices.at(node.owner) ? node.left : node.right, choices);
  default:
    return {};
  }
}

std::vector<Demand> demands(const fs::FactoredUseResult &result, std::size_t id,
                            const std::map<std::size_t, bool> &choices) {
  const auto &node = result.nodes.at(id);
  switch (node.kind) {
  case fs::FactoredUseNode::Kind::Empty:
    return {};
  case fs::FactoredUseNode::Kind::Demand: {
    std::vector<Demand> found;
    for (auto source : origins(result, node.left, choices)) {
      if (source != fs::NoControlId) {
        found.emplace_back(source, node.operation, node.hazard);
      }
    }
    return found;
  }
  case fs::FactoredUseNode::Kind::Both: {
    auto left = demands(result, node.left, choices);
    auto right = demands(result, node.right, choices);
    left.insert(left.end(), right.begin(), right.end());
    return left;
  }
  case fs::FactoredUseNode::Kind::Choose:
    return demands(result, choices.at(node.owner) ? node.left : node.right, choices);
  default:
    return {};
  }
}

fs::Region factoredOperation(std::size_t id) {
  fs::Region result;
  result.kind = fs::Region::Operation;
  result.operation = id;
  return result;
}

fs::Region factoredChoice(std::size_t owner, fs::Region yes) {
  fs::Region result;
  result.kind = fs::Region::Choice;
  result.originalOwner = owner;
  result.children.push_back(std::move(yes));
  result.children.emplace_back();
  return result;
}

bool checkFactoredProvenance() {
  fs::OriginalStructure original;
  original.cells.resize(1);
  original.operations.resize(6);
  auto effect = [&](std::size_t op, bool read, bool write) {
    fs::Access access;
    access.cell = 0;
    access.read = read;
    access.write = write;
    access.definiteWrite = write;
    original.operations[op].accesses.push_back(access);
  };
  effect(0, false, true);  // W0
  effect(1, true, false);  // A
  effect(2, true, true);   // Conditional read-modify-write W1
  effect(3, true, false);  // B
  effect(4, true, false);  // Optional R
  effect(5, false, true);  // W2
  for (std::size_t op : {0, 1}) {
    original.body.children.push_back(factoredOperation(op));
  }
  original.body.children.push_back(factoredChoice(10, factoredOperation(2)));
  original.body.children.push_back(factoredOperation(3));
  original.body.children.push_back(factoredChoice(11, factoredOperation(4)));
  original.body.children.push_back(factoredOperation(5));
  fs::FactoredProvenance analysis(original, 0);
  const auto &result = analysis.get();
  if (!result.complete) {
    return false;
  }
  for (bool g : {false, true}) {
    for (bool h : {false, true}) {
      const std::map<std::size_t, bool> choices{{10, g}, {11, h}};
      std::set<Demand> actual;
      for (auto demand : demands(result, result.demands, choices)) {
        actual.insert(demand);
      }
      std::set<Demand> expected{
          {0, 1, fs::FactoredUseNode::Hazard::RAW},
          {g ? 2UL : 0UL, 3, fs::FactoredUseNode::Hazard::RAW},
          {g ? 2UL : 0UL, 5, fs::FactoredUseNode::Hazard::WAW},
          {3, 5, fs::FactoredUseNode::Hazard::WAR}};
      if (g) {
        expected.emplace(0, 2, fs::FactoredUseNode::Hazard::RAW);
        expected.emplace(0, 2, fs::FactoredUseNode::Hazard::WAW);
        expected.emplace(1, 2, fs::FactoredUseNode::Hazard::WAR);
      } else {
        expected.emplace(1, 5, fs::FactoredUseNode::Hazard::WAR);
      }
      if (h) {
        expected.emplace(g ? 2UL : 0UL, 4, fs::FactoredUseNode::Hazard::RAW);
        expected.emplace(4, 5, fs::FactoredUseNode::Hazard::WAR);
      }
      if (actual != expected ||
          origins(result, result.priorWriters[3], choices) != std::vector<std::size_t>{g ? 2UL : 0UL} ||
          origins(result, result.nextWriters[1], choices) != std::vector<std::size_t>{g ? 2UL : 5UL}) {
        return false;
      }
    }
  }
  fs::OriginalStructure many;
  many.cells.resize(1);
  many.operations.resize(66);
  many.operations[0].accesses.push_back({0, false, true, true});
  many.body.children.push_back(factoredOperation(0));
  for (std::size_t i = 1; i <= 64; ++i) {
    many.operations[i].accesses.push_back({0, true, false, false});
    many.body.children.push_back(factoredChoice(i, factoredOperation(i)));
  }
  many.operations[65].accesses.push_back({0, false, true, true});
  many.body.children.push_back(factoredOperation(65));
  fs::FactoredProvenance sharedAnalysis(many, 0);
  const auto &shared = sharedAnalysis.get();
  const bool sharedSizeValid = shared.complete && shared.nodes.size() < 64 * 12;
  if (!sharedSizeValid) {
    return false;
  }
  fs::OriginalStructure partial;
  partial.cells.resize(1);
  partial.operations.resize(5);
  partial.operations[0].accesses.push_back({0, false, true, true});
  partial.operations[1].accesses.push_back({0, true, false, false});
  partial.operations[2].accesses.push_back({0, false, true, false});
  partial.operations[3].accesses.push_back({0, true, false, false});
  partial.operations[4].accesses.push_back({0, false, true, true});
  for (std::size_t i = 0; i < partial.operations.size(); ++i) {
    partial.body.children.push_back(factoredOperation(i));
  }
  fs::FactoredProvenance partialAnalysis(partial, 0);
  const auto &weak = partialAnalysis.get();
  const std::map<std::size_t, bool> noChoices;
  return weak.complete &&
         origins(weak, weak.priorWriters[3], noChoices) == std::vector<std::size_t>({0, 2}) &&
         origins(weak, weak.priorReaders[4], noChoices) == std::vector<std::size_t>({1, 3});
}

mlir::LogicalResult inspectRequests(const fs::ProgramAnalysis &analysis, Counts &counts) {
  for (std::size_t site = 0; site < analysis.structure().operations.size(); ++site) {
    const auto &requests = analysis.requirementsAt(site);
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto &request = requests[index];
      if (request.source.operation >= analysis.structure().operations.size()) {
        return mlir::failure();
      }
      bool subscribed = false;
      for (const auto &subscription : analysis.subscriptionsAt(request.source.operation)) {
        subscribed |= subscription.deadlineOperation == site && subscription.requirementIndex == index;
      }
      if (!subscribed) {
        return mlir::failure();
      }
      const auto &answer = analysis.interpretAt(site, index);
      ++counts.interpreted;
      ++counts.hazards.at(static_cast<std::size_t>(request.relationship.kind));
      counts.unknownOccurrence += answer.occurrence.status == fs::OriginalOccurrenceInterpretation::Status::Unknown;
      counts.decodedUnresolved += !answer.decoded->unresolved.empty();
      counts.readers += answer.decoded->hasReaderFrontier;
    }
  }
  return mlir::success();
}

mlir::LogicalResult analyze(mlir::func::FuncOp function) {
  mlir::pto::SyncInput input;
  if (mlir::failed(input.build(function))) {
    return mlir::failure();
  }
  fs::OriginalStructure original;
  if (mlir::failed(fs::importOriginalStructure(function, input, original))) {
    return mlir::failure();
  }
  fs::ProgramAnalysis analysis(input, std::move(original));
  if (!analysis.complete()) {
    return function.emitError(analysis.reason());
  }
  Counts counts;
  if (mlir::failed(inspectRequests(analysis, counts))) {
    return function.emitError("missing original source subscription");
  }
  const auto &structure = analysis.structure();
  llvm::outs() << function.getName() << ": phases=" << structure.operations.size()
               << " cells=" << structure.cells.size() << " bank-relations=" << structure.physicalAddresses.size()
               << "\n  RAW=" << counts.hazards[0] << " WAR=" << counts.hazards[1] << " WAW=" << counts.hazards[2]
               << " interpreted=" << counts.interpreted << " readers=" << counts.readers
               << "\n  unknown-occurrence=" << counts.unknownOccurrence
               << " decoded-unresolved=" << counts.decodedUnresolved << "\n";
  return mlir::success();
}

mlir::LogicalResult runFile(llvm::StringRef path, mlir::MLIRContext &context, bool verifyOnly) {
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
  if (!module || mlir::failed(mlir::verify(*module))) {
    return mlir::failure();
  }
  if (verifyOnly) {
    std::size_t localSync = 0, crossSync = 0;
    module->walk([&](mlir::Operation *operation) {
      const auto name = operation->getName().getStringRef();
      localSync += name == "pto.set_flag_dyn" || name == "pto.wait_flag_dyn";
      crossSync += name == "pto.sync.set" || name == "pto.sync.wait";
    });
    llvm::outs() << "verified: local-sync=" << localSync << " cross-sync=" << crossSync << "\n";
    return mlir::success();
  }
  const auto before = printIR(module->getOperation());
  for (auto function : module->getOps<mlir::func::FuncOp>()) {
    if (mlir::failed(analyze(function))) {
      return mlir::failure();
    }
  }
  if (before != printIR(module->getOperation())) {
    return module->emitError("Phase A changed the original IR");
  }
  llvm::outs() << "original-ir-unchanged; construction-not-run\n";
  return mlir::success();
}
} // namespace

int main(int argc, char **argv) {
  const bool selfTest = argc == 2 && llvm::StringRef(argv[1]) == "--factored-self-test";
  if (selfTest) {
    if (!checkFactoredProvenance()) {
      llvm::errs() << "factored original-provenance check failed\n";
      return 1;
    }
    llvm::outs() << "factored original-provenance check passed\n";
    return 0;
  }
  const bool verifyOnly = argc == 3 && llvm::StringRef(argv[1]) == "--verify-only";
  if (argc != 2 && !verifyOnly) {
    llvm::errs() << "usage: pto-frontier-analysis-test [--verify-only] input.pto\n";
    return 1;
  }
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();
  mlir::MLIRContext context(dialects);
  context.disableMultithreading();
  return mlir::failed(runFile(argv[verifyOnly ? 2 : 1], context, verifyOnly));
}
