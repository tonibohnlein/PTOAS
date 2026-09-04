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
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
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
  unsigned iterationDistance = 0;
  unsigned recurrenceLoopDepth = 0;
  std::string provenance;
};

struct ResolvedNodeDuration {
  pto::KernelScheduleGraph::VertexIdx node = 0;
  unsigned pyptoAccessOrder = 0;
  std::string operation;
  uint64_t cycles = 0;
  std::string provenance;
};

struct ResolvedNodeDurationDocument {
  std::string graphSha256;
  llvm::SmallVector<ResolvedNodeDuration, 32> nodes;
};

std::string getGraphSha256(func::FuncOp func,
                           const pto::KernelScheduleGraph &graph) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  pto::printKernelScheduleGraph(stream, func, graph);
  stream.flush();
  const auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(text));
  return llvm::toHex(digest, /*LowerCase=*/true);
}

std::optional<pto::ScheduleDependencyKind>
parsePlacementReuseKind(llvm::StringRef kind) {
  return llvm::StringSwitch<std::optional<pto::ScheduleDependencyKind>>(kind)
      .Case("raw", pto::ScheduleDependencyKind::PlacementReuseRAW)
      .Case("war", pto::ScheduleDependencyKind::PlacementReuseWAR)
      .Case("waw", pto::ScheduleDependencyKind::PlacementReuseWAW)
      .Default(std::nullopt);
}

FailureOr<llvm::SmallVector<PlacementReuseEdge, 8>>
loadPlacementReuseEdges(llvm::StringRef path, llvm::StringRef function,
                        llvm::StringRef graphSha256) {
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
  std::optional<llvm::StringRef> namedFunction = root->getString("function");
  if (!namedFunction || *namedFunction != function) {
    return failure();
  }
  if (std::optional<llvm::StringRef> contract = root->getString("contract")) {
    if (*contract != "ptoas_placement_reuse_topology_v1" ||
        root->getBoolean("topology_only") != true ||
        root->getString("ptoas_graph_sha256") != graphSha256) {
      return failure();
    }
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
    const int64_t iterationDistance =
        edge->getInteger("iteration_distance").value_or(0);
    const int64_t recurrenceLoopDepth =
        edge->getInteger("recurrence_loop_depth").value_or(0);
    std::optional<pto::ScheduleDependencyKind> kind =
        kindText ? parsePlacementReuseKind(*kindText) : std::nullopt;
    if (!source || !target || *source < 0 || *target < 0 || !kind ||
        iterationDistance < 0 || recurrenceLoopDepth < 0 ||
        ((iterationDistance == 0) != (recurrenceLoopDepth == 0)) ||
        !provenance || provenance->empty()) {
      return failure();
    }
    edges.push_back({static_cast<pto::KernelScheduleGraph::VertexIdx>(*source),
                     static_cast<pto::KernelScheduleGraph::VertexIdx>(*target),
                     *kind, static_cast<unsigned>(iterationDistance),
                     static_cast<unsigned>(recurrenceLoopDepth),
                     provenance->str()});
  }
  return edges;
}

static unsigned getLoopDepth(Operation *loop) {
  unsigned depth = 1;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      ++depth;
    }
  }
  return depth;
}

static Operation *findCommonRecurrenceLoop(Operation *source,
                                           Operation *target,
                                           unsigned expectedDepth) {
  for (Operation *sourceParent = source->getParentOp(); sourceParent;
       sourceParent = sourceParent->getParentOp()) {
    if (!isa<LoopLikeOpInterface>(sourceParent) ||
        getLoopDepth(sourceParent) != expectedDepth) {
      continue;
    }
    for (Operation *targetParent = target->getParentOp(); targetParent;
         targetParent = targetParent->getParentOp()) {
      if (targetParent == sourceParent) {
        return sourceParent;
      }
    }
  }
  return nullptr;
}

FailureOr<ResolvedNodeDurationDocument>
loadResolvedNodeDurations(llvm::StringRef path, llvm::StringRef function,
                          llvm::StringRef graphSha256,
                          const pto::KernelScheduleGraph &graph) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return failure();
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getInteger("schema_version") != 1 ||
      root->getString("contract") != "ptoas_resolved_node_durations_v1" ||
      root->getString("function") != function ||
      root->getString("ptoas_graph_sha256") != graphSha256 ||
      !root->getString("schedule_semantic_sha256") ||
      !root->getObject("duration_model")) {
    return failure();
  }
  llvm::json::Object *shape = root->getObject("graph_shape");
  if (!shape || shape->getInteger("node_count") !=
                    static_cast<int64_t>(graph.getNodes().size()) ||
      shape->getInteger("dag_edge_count") !=
          static_cast<int64_t>(graph.getGraph().NumEdges()) ||
      shape->getInteger("dependency_count") !=
          static_cast<int64_t>(graph.getDependencies().size())) {
    return failure();
  }
  llvm::json::Array *rawNodes = root->getArray("nodes");
  if (!rawNodes || rawNodes->size() != graph.getNodes().size()) {
    return failure();
  }

  ResolvedNodeDurationDocument document;
  document.graphSha256 = graphSha256.str();
  llvm::SmallVector<bool, 32> seen(graph.getNodes().size(), false);
  for (llvm::json::Value &rawNode : *rawNodes) {
    llvm::json::Object *node = rawNode.getAsObject();
    if (!node) {
      return failure();
    }
    std::optional<int64_t> id = node->getInteger("node_id");
    std::optional<int64_t> access = node->getInteger("pypto_access_order");
    std::optional<int64_t> cycles = node->getInteger("cycles");
    std::optional<llvm::StringRef> operation = node->getString("op_name");
    std::optional<llvm::StringRef> source = node->getString("source");
    std::optional<llvm::StringRef> detail = node->getString("detail");
    if (!id || *id < 0 || static_cast<uint64_t>(*id) >= seen.size() ||
        seen[*id] || !access || *access < 0 || !cycles || *cycles <= 0 ||
        !operation || operation->empty() || !source || source->empty() ||
        !detail) {
      return failure();
    }
    const auto &graphNode = graph.getNode(static_cast<unsigned>(*id));
    if (!graphNode.pyptoAccessOrder ||
        *graphNode.pyptoAccessOrder != static_cast<unsigned>(*access) ||
        graphNode.operation->getName().getStringRef() != *operation) {
      return failure();
    }
    seen[*id] = true;
    document.nodes.push_back(
        {static_cast<unsigned>(*id), static_cast<unsigned>(*access),
         operation->str(), static_cast<uint64_t>(*cycles), source->str()});
  }
  return document;
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
    if (!durationTable.empty() && !nodeDurations.empty()) {
      func.emitError("--duration-table and --node-durations are mutually exclusive");
      signalPassFailure();
      return;
    }
    if (requireExactDurations && durationTable.empty() &&
        nodeDurations.empty()) {
      func.emitError(
          "--require-exact-durations requires --duration-table or --node-durations");
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
    options.requireExactDurations =
        requireExactDurations && nodeDurations.empty();
    FailureOr<pto::KernelScheduleGraph> graph =
        pto::buildKernelScheduleGraph(func, options);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    const std::string baseGraphSha256 = getGraphSha256(func, *graph);
    if (!nodeDurations.empty()) {
      FailureOr<ResolvedNodeDurationDocument> durations =
          loadResolvedNodeDurations(nodeDurations, func.getSymName(),
                                    baseGraphSha256, *graph);
      if (failed(durations)) {
        func.emitError() << "invalid or mismatched resolved node-duration file '"
                         << nodeDurations << "'";
        signalPassFailure();
        return;
      }
      for (const ResolvedNodeDuration &duration : durations->nodes) {
        if (failed(graph->setNodeDuration(duration.node, duration.cycles,
                                          duration.provenance))) {
          func.emitError("failed to apply an external node duration");
          signalPassFailure();
          return;
        }
      }
    }
    if (requireExactDurations &&
        llvm::any_of(graph->getNodes(), [](const pto::KernelScheduleNode &node) {
          return !node.hasExactDuration;
        })) {
      func.emitError("kernel schedule graph has unresolved node durations");
      signalPassFailure();
      return;
    }
    if (!placementReuseEdges.empty()) {
      FailureOr<llvm::SmallVector<PlacementReuseEdge, 8>> edges =
          loadPlacementReuseEdges(placementReuseEdges, func.getSymName(),
                                  baseGraphSha256);
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
        Operation *recurrenceLoop = nullptr;
        if (edge.iterationDistance != 0) {
          recurrenceLoop = findCommonRecurrenceLoop(
              graph->getNode(edge.source).operation,
              graph->getNode(edge.target).operation,
              edge.recurrenceLoopDepth);
          if (!recurrenceLoop) {
            func.emitError("placement-reuse recurrence cannot be joined to a common loop");
            signalPassFailure();
            return;
          }
        }
        graph->addDependency(edge.source, edge.target, edge.kind,
                             edge.iterationDistance, recurrenceLoop,
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
