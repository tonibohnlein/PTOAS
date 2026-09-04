// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- ScheduleGraph.h - OSP-compatible adjacency DAG -----------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_SCHEDULEGRAPH_H
#define MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_SCHEDULEGRAPH_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

/// A small, dependency-free computational DAG. Its public query API satisfies
/// OneStopParallel's ComputationalDagTypedVerticesEdgeDesc concept; OSP adds
/// its default edge descriptor and edge views on top of these adjacency lists.
class ScheduleGraph {
public:
  using VertexIdx = std::size_t;
  using VertexWorkWeightType = std::uint64_t;
  using VertexCommWeightType = std::uint64_t;
  using VertexMemWeightType = std::uint64_t;
  using VertexTypeType = std::uint32_t;

  VertexIdx AddVertex(VertexWorkWeightType workWeight,
                      VertexCommWeightType commWeight,
                      VertexMemWeightType memWeight,
                      VertexTypeType vertexType = 0) {
    const VertexIdx id = vertices_.size();
    vertices_.push_back(id);
    children_.emplace_back();
    parents_.emplace_back();
    workWeights_.push_back(workWeight);
    commWeights_.push_back(commWeight);
    memWeights_.push_back(memWeight);
    vertexTypes_.push_back(vertexType);
    numVertexTypes_ = std::max(numVertexTypes_, vertexType + 1);
    return id;
  }

  bool AddEdge(VertexIdx source, VertexIdx target) {
    const bool invalidSource = source >= NumVertices();
    const bool invalidTarget = target >= NumVertices();
    if (invalidSource || invalidTarget || source == target) {
      return false;
    }
    const auto &out = children_[source];
    const bool duplicate =
        std::find(out.begin(), out.end(), target) != out.end();
    if (duplicate) {
      return false;
    }
    children_[source].push_back(target);
    parents_[target].push_back(source);
    ++numEdges_;
    return true;
  }

  const std::vector<VertexIdx> &Vertices() const { return vertices_; }
  VertexIdx NumVertices() const { return vertices_.size(); }
  VertexIdx NumEdges() const { return numEdges_; }
  const std::vector<VertexIdx> &Parents(VertexIdx vertex) const {
    return parents_[vertex];
  }
  const std::vector<VertexIdx> &Children(VertexIdx vertex) const {
    return children_[vertex];
  }
  VertexIdx InDegree(VertexIdx vertex) const { return parents_[vertex].size(); }
  VertexIdx OutDegree(VertexIdx vertex) const {
    return children_[vertex].size();
  }

  VertexWorkWeightType VertexWorkWeight(VertexIdx vertex) const {
    return workWeights_[vertex];
  }
  bool SetVertexWorkWeight(VertexIdx vertex, VertexWorkWeightType workWeight) {
    if (vertex >= NumVertices()) {
      return false;
    }
    workWeights_[vertex] = workWeight;
    return true;
  }
  VertexCommWeightType VertexCommWeight(VertexIdx vertex) const {
    return commWeights_[vertex];
  }
  VertexMemWeightType VertexMemWeight(VertexIdx vertex) const {
    return memWeights_[vertex];
  }
  VertexTypeType VertexType(VertexIdx vertex) const {
    return vertexTypes_[vertex];
  }
  VertexTypeType NumVertexTypes() const { return numVertexTypes_; }

  bool IsAcyclic() const {
    std::vector<VertexIdx> indegrees;
    indegrees.reserve(NumVertices());
    std::vector<VertexIdx> ready;
    ready.reserve(NumVertices());
    for (VertexIdx vertex : Vertices()) {
      indegrees.push_back(InDegree(vertex));
      const bool isReady = indegrees.back() == 0;
      if (isReady) {
        ready.push_back(vertex);
      }
    }

    VertexIdx visited = 0;
    for (std::size_t index = 0; index < ready.size(); ++index) {
      const VertexIdx vertex = ready[index];
      ++visited;
      for (VertexIdx child : Children(vertex)) {
        if (--indegrees[child] == 0) {
          ready.push_back(child);
        }
      }
    }
    return visited == NumVertices();
  }

private:
  std::vector<VertexIdx> vertices_;
  std::vector<std::vector<VertexIdx>> children_;
  std::vector<std::vector<VertexIdx>> parents_;
  std::vector<VertexWorkWeightType> workWeights_;
  std::vector<VertexCommWeightType> commWeights_;
  std::vector<VertexMemWeightType> memWeights_;
  std::vector<VertexTypeType> vertexTypes_;
  VertexIdx numEdges_ = 0;
  VertexTypeType numVertexTypes_ = 0;
};

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_SCHEDULEGRAPH_H
