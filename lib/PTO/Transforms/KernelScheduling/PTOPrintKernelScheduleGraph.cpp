// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/KernelScheduleGraph.h"
#include "PTO/Transforms/KernelScheduling/PTOISADuration.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PRINTKERNELSCHEDULEGRAPH
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PlacementReuseEdge {
  pto::KernelScheduleGraph::VertexIdx source = 0;
  pto::KernelScheduleGraph::VertexIdx target = 0;
  pto::ScheduleDependencyKind kind = pto::ScheduleDependencyKind::PlacementReuseRAW;
  std::string provenance;
};

std::optional<pto::ScheduleDependencyKind>
parsePlacementReuseKind(llvm::StringRef kind) {
  return llvm::StringSwitch<std::optional<pto::ScheduleDependencyKind>>(kind)
      .Case("raw", pto::ScheduleDependencyKind::PlacementReuseRAW)
      .Case("war", pto::ScheduleDependencyKind::PlacementReuseWAR)
      .Case("waw", pto::ScheduleDependencyKind::PlacementReuseWAW)
      .Default(std::nullopt);
}

FailureOr<llvm::SmallVector<PlacementReuseEdge, 8>>
loadPlacementReuseEdges(llvm::StringRef path, llvm::StringRef function) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return failure();
  }
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getInteger("schema_version") != 1) {
    return failure();
  }
  if (std::optional<llvm::StringRef> namedFunction = root->getString("function");
      namedFunction && *namedFunction != function) {
    return llvm::SmallVector<PlacementReuseEdge, 8>{};
  }
  llvm::json::Array *rawEdges = root->getArray("edges");
  if (!rawEdges) {
    return failure();
  }
  llvm::SmallVector<PlacementReuseEdge, 8> edges;
  for (llvm::json::Value &rawEdge : *rawEdges) {
    llvm::json::Object *edge = rawEdge.getAsObject();
    if (!edge) {
      return failure();
    }
    std::optional<int64_t> source = edge->getInteger("source_node");
    std::optional<int64_t> target = edge->getInteger("target_node");
    std::optional<llvm::StringRef> kindText = edge->getString("kind");
    std::optional<llvm::StringRef> provenance = edge->getString("provenance");
    std::optional<pto::ScheduleDependencyKind> kind =
        kindText ? parsePlacementReuseKind(*kindText) : std::nullopt;
    if (!source || !target || *source < 0 || *target < 0 || !kind ||
        !provenance || provenance->empty()) {
      return failure();
    }
    edges.push_back({static_cast<pto::KernelScheduleGraph::VertexIdx>(*source),
                     static_cast<pto::KernelScheduleGraph::VertexIdx>(*target),
                     *kind, provenance->str()});
  }
  return edges;
}

struct PrintKernelScheduleGraphPass
    : public pto::impl::PrintKernelScheduleGraphBase<
          PrintKernelScheduleGraphPass> {
  using pto::impl::PrintKernelScheduleGraphBase<
      PrintKernelScheduleGraphPass>::PrintKernelScheduleGraphBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    if (format != "text" && format != "dot") {
      func.emitError() << "unsupported kernel schedule graph format '" << format
                       << "'; expected 'text' or 'dot'";
      signalPassFailure();
      return;
    }
    if (requireExactDurations && durationTable.empty()) {
      func.emitError("--require-exact-durations requires --duration-table");
      signalPassFailure();
      return;
    }

    std::optional<pto::PTOISADurationTable> durationTableData;
    if (!durationTable.empty()) {
      FailureOr<pto::PTOISADurationTable> loaded =
          pto::PTOISADurationTable::loadFromFile(durationTable);
      if (failed(loaded)) {
        func.emitError() << "failed to load PTO-ISA duration table '"
                         << durationTable << "'";
        signalPassFailure();
        return;
      }
      durationTableData = std::move(*loaded);
    }
    pto::KernelScheduleGraphBuildOptions options;
    options.durationTable = durationTableData ? &*durationTableData : nullptr;
    options.requireExactDurations = requireExactDurations;
    FailureOr<pto::KernelScheduleGraph> graph =
        pto::buildKernelScheduleGraph(func, options);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    if (!placementReuseEdges.empty()) {
      FailureOr<llvm::SmallVector<PlacementReuseEdge, 8>> edges =
          loadPlacementReuseEdges(placementReuseEdges, func.getSymName());
      if (failed(edges)) {
        func.emitError() << "invalid placement-reuse edge file '"
                         << placementReuseEdges << "'";
        signalPassFailure();
        return;
      }
      for (const PlacementReuseEdge &edge : *edges) {
        if (edge.source >= graph->getNodes().size() ||
            edge.target >= graph->getNodes().size()) {
          func.emitError("placement-reuse edge refers to an unknown schedule node");
          signalPassFailure();
          return;
        }
        graph->addDependency(edge.source, edge.target, edge.kind,
                             /*iterationDistance=*/0,
                             /*recurrenceLoop=*/nullptr,
                             reuseSyncLatencyCycles, edge.provenance);
      }
    }
    if (format == "dot") {
      pto::printKernelScheduleGraphDot(llvm::outs(), func, *graph);
    } else {
      pto::printKernelScheduleGraph(llvm::outs(), func, *graph);
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPrintKernelScheduleGraphPass() {
  return std::make_unique<PrintKernelScheduleGraphPass>();
}
