// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"

#include <algorithm>
#include <deque>
#include <limits>

using namespace mlir::pto;

SyncConflictGraph::Vertex SyncConflictGraph::addVertex() {
  const Vertex vertex = neighbors_.size();
  neighbors_.emplace_back();
  return vertex;
}

bool SyncConflictGraph::addEdge(Vertex first, Vertex second) {
  if (first >= size() || second >= size() || first == second ||
      adjacent(first, second)) {
    return false;
  }
  neighbors_[first].push_back(second);
  neighbors_[second].push_back(first);
  return true;
}

bool SyncConflictGraph::adjacent(Vertex first, Vertex second) const {
  if (first >= size() || second >= size()) {
    return false;
  }
  const auto &neighbors = neighbors_[first];
  return std::find(neighbors.begin(), neighbors.end(), second) !=
         neighbors.end();
}

namespace {

std::vector<std::size_t>
maximumCardinalityOrder(const SyncConflictGraph &graph) {
  std::vector<unsigned> weights(graph.size(), 0);
  std::vector<bool> selected(graph.size(), false);
  std::vector<std::size_t> order;
  order.reserve(graph.size());
  for (std::size_t step = 0; step < graph.size(); ++step) {
    std::size_t best = graph.size();
    for (std::size_t vertex = 0; vertex < graph.size(); ++vertex) {
      if (selected[vertex]) {
        continue;
      }
      if (best == graph.size() || weights[vertex] > weights[best] ||
          (weights[vertex] == weights[best] && vertex < best)) {
        best = vertex;
      }
    }
    selected[best] = true;
    order.push_back(best);
    for (std::size_t neighbor : graph.neighbors(best)) {
      if (!selected[neighbor]) {
        ++weights[neighbor];
      }
    }
  }
  return order;
}

bool isPerfectEliminationOrder(const SyncConflictGraph &graph,
                               const std::vector<std::size_t> &order) {
  std::vector<std::size_t> position(graph.size(), 0);
  for (std::size_t index = 0; index < order.size(); ++index) {
    position[order[index]] = index;
  }
  for (std::size_t index = 0; index < order.size(); ++index) {
    const std::size_t vertex = order[index];
    std::size_t firstLater = graph.size();
    for (std::size_t neighbor : graph.neighbors(vertex)) {
      if (position[neighbor] > index &&
          (firstLater == graph.size() ||
           position[neighbor] < position[firstLater])) {
        firstLater = neighbor;
      }
    }
    if (firstLater == graph.size()) {
      continue;
    }
    for (std::size_t neighbor : graph.neighbors(vertex)) {
      if (neighbor != firstLater && position[neighbor] > index &&
          !graph.adjacent(firstLater, neighbor)) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<int>>
componentsWithoutClosedNeighborhood(const SyncConflictGraph &graph) {
  std::vector<std::vector<int>> components(graph.size(),
                                           std::vector<int>(graph.size(), -1));
  for (std::size_t excluded = 0; excluded < graph.size(); ++excluded) {
    std::vector<bool> removed(graph.size(), false);
    removed[excluded] = true;
    for (std::size_t neighbor : graph.neighbors(excluded)) {
      removed[neighbor] = true;
    }
    int component = 0;
    for (std::size_t root = 0; root < graph.size(); ++root) {
      if (removed[root] || components[excluded][root] != -1) {
        continue;
      }
      std::deque<std::size_t> ready{root};
      components[excluded][root] = component;
      while (!ready.empty()) {
        const std::size_t vertex = ready.front();
        ready.pop_front();
        for (std::size_t neighbor : graph.neighbors(vertex)) {
          if (!removed[neighbor] && components[excluded][neighbor] == -1) {
            components[excluded][neighbor] = component;
            ready.push_back(neighbor);
          }
        }
      }
      ++component;
    }
  }
  return components;
}

bool hasAsteroidalTriple(const SyncConflictGraph &graph) {
  const auto components = componentsWithoutClosedNeighborhood(graph);
  for (std::size_t first = 0; first < graph.size(); ++first) {
    for (std::size_t second = first + 1; second < graph.size(); ++second) {
      if (graph.adjacent(first, second)) {
        continue;
      }
      for (std::size_t third = second + 1; third < graph.size(); ++third) {
        if (graph.adjacent(first, third) || graph.adjacent(second, third)) {
          continue;
        }
        const bool connectedWithoutFirst =
            components[first][second] == components[first][third];
        const bool connectedWithoutSecond =
            components[second][first] == components[second][third];
        const bool connectedWithoutThird =
            components[third][first] == components[third][second];
        if (connectedWithoutFirst && connectedWithoutSecond &&
            connectedWithoutThird) {
          return true;
        }
      }
    }
  }
  return false;
}

SyncColoring greedyColor(const SyncConflictGraph &graph,
                         const std::vector<std::size_t> &order) {
  SyncColoring result;
  result.colors.assign(graph.size(), SyncColoring::kUncolored);
  std::vector<bool> unavailable(graph.size(), false);
  for (std::size_t vertex : order) {
    std::fill(unavailable.begin(), unavailable.end(), false);
    for (std::size_t neighbor : graph.neighbors(vertex)) {
      const unsigned color = result.colors[neighbor];
      if (color != SyncColoring::kUncolored) {
        unavailable[color] = true;
      }
    }
    unsigned color = 0;
    while (color < unavailable.size() && unavailable[color]) {
      ++color;
    }
    result.colors[vertex] = color;
    result.colorCount = std::max(result.colorCount, color + 1);
  }
  return result;
}

} // namespace

IntervalColoring mlir::pto::colorIntervalGraph(const SyncConflictGraph &graph) {
  IntervalColoring result;
  std::vector<std::size_t> selection = maximumCardinalityOrder(graph);
  std::vector<std::size_t> peo(selection.rbegin(), selection.rend());
  if (!isPerfectEliminationOrder(graph, peo) || hasAsteroidalTriple(graph)) {
    return result;
  }
  result.isInterval = true;
  result.coloring = greedyColor(graph, selection);
  return result;
}

SyncColoring mlir::pto::colorDsatur(const SyncConflictGraph &graph) {
  SyncColoring result;
  result.colors.assign(graph.size(), SyncColoring::kUncolored);
  for (std::size_t step = 0; step < graph.size(); ++step) {
    std::size_t best = graph.size();
    unsigned bestSaturation = 0;
    for (std::size_t vertex = 0; vertex < graph.size(); ++vertex) {
      if (result.colors[vertex] != SyncColoring::kUncolored) {
        continue;
      }
      std::vector<bool> seen(graph.size(), false);
      unsigned saturation = 0;
      for (std::size_t neighbor : graph.neighbors(vertex)) {
        const unsigned color = result.colors[neighbor];
        if (color != SyncColoring::kUncolored && !seen[color]) {
          seen[color] = true;
          ++saturation;
        }
      }
      if (best == graph.size() || saturation > bestSaturation ||
          (saturation == bestSaturation &&
           graph.neighbors(vertex).size() > graph.neighbors(best).size()) ||
          (saturation == bestSaturation &&
           graph.neighbors(vertex).size() == graph.neighbors(best).size() &&
           vertex < best)) {
        best = vertex;
        bestSaturation = saturation;
      }
    }
    std::vector<bool> unavailable(graph.size(), false);
    for (std::size_t neighbor : graph.neighbors(best)) {
      const unsigned color = result.colors[neighbor];
      if (color != SyncColoring::kUncolored) {
        unavailable[color] = true;
      }
    }
    unsigned color = 0;
    while (color < unavailable.size() && unavailable[color]) {
      ++color;
    }
    result.colors[best] = color;
    result.colorCount = std::max(result.colorCount, color + 1);
  }
  return result;
}

SyncColoring mlir::pto::colorSmallestLast(const SyncConflictGraph &graph) {
  std::vector<bool> removed(graph.size(), false);
  std::vector<unsigned> degree(graph.size(), 0);
  std::vector<std::size_t> removalOrder;
  removalOrder.reserve(graph.size());
  for (std::size_t vertex = 0; vertex < graph.size(); ++vertex) {
    degree[vertex] = graph.neighbors(vertex).size();
  }
  for (std::size_t step = 0; step < graph.size(); ++step) {
    std::size_t best = graph.size();
    for (std::size_t vertex = 0; vertex < graph.size(); ++vertex) {
      if (!removed[vertex] &&
          (best == graph.size() || degree[vertex] < degree[best] ||
           (degree[vertex] == degree[best] && vertex < best))) {
        best = vertex;
      }
    }
    removed[best] = true;
    removalOrder.push_back(best);
    for (std::size_t neighbor : graph.neighbors(best)) {
      if (!removed[neighbor]) {
        --degree[neighbor];
      }
    }
  }
  std::reverse(removalOrder.begin(), removalOrder.end());
  return greedyColor(graph, removalOrder);
}

bool mlir::pto::isValidColoring(const SyncConflictGraph &graph,
                                const SyncColoring &coloring) {
  if (coloring.colors.size() != graph.size()) {
    return false;
  }
  for (std::size_t vertex = 0; vertex < graph.size(); ++vertex) {
    if (coloring.colors[vertex] == SyncColoring::kUncolored ||
        coloring.colors[vertex] >= coloring.colorCount) {
      return false;
    }
    for (std::size_t neighbor : graph.neighbors(vertex)) {
      if (coloring.colors[vertex] == coloring.colors[neighbor]) {
        return false;
      }
    }
  }
  return true;
}
