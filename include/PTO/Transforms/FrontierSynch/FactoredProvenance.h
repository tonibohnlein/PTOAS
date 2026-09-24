// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FACTOREDPROVENANCE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FACTOREDPROVENANCE_H

#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include <string>

namespace mlir::pto::frontiersynch {

// A shared expression for one cell in an acyclic original-control projection.
// Node IDs are immutable; a Choice always names the same original condition
// occurrence in every provenance and demand expression that refers to it.
struct FactoredUseNode {
  enum class Kind { Empty, Incoming, Access, Both, Choose, Demand } kind = Kind::Empty;
  enum class Sort { Origins, Demands } sort = Sort::Origins;
  enum class Hazard { RAW, WAR, WAW } hazard = Hazard::RAW;
  std::size_t left = 0, right = 0, owner = NoControlId, operation = NoControlId;
};

struct FactoredUseResult {
  // Completeness here means the original acyclic syntax was formed into an
  // expression. It does not qualify geometry, guards at endpoints, or an
  // occurrence continuation for exact placement.
  bool complete = false;
  std::size_t cell = NoControlId;
  std::string reason;
  std::size_t demands = 0, finalWriters = 0, finalReaders = 0;
  std::vector<FactoredUseNode> nodes;
  // Each access queries its incoming state before the original operation's
  // combined read/write transfer. These are useful even when a later write
  // replaces the outgoing state.
  std::vector<std::size_t> priorWriters, priorReaders;
  // Backward expressions stop at this projection's exit. Empty is not a proof
  // that a larger enclosing continuation has no further access.
  std::vector<std::size_t> nextWriters, nextReaders;
  // An operation leaf retains every shared translated-effect incidence of
  // its read and write roles. The leaf does not merge those effect witnesses.
  std::vector<std::vector<std::size_t>> readIncidences, writeIncidences;
};

class FactoredProvenance {
public:
  explicit FactoredProvenance(const OriginalStructure &original, std::size_t cell)
      : original(original), cell(cell) {
    result.cell = cell;
    result.nodes.push_back({});
    FactoredUseNode incoming;
    incoming.kind = FactoredUseNode::Kind::Incoming;
    result.nodes.push_back(incoming);
    result.priorWriters.assign(original.operations.size(), 0);
    result.priorReaders.assign(original.operations.size(), 0);
    result.nextWriters.assign(original.operations.size(), 0);
    result.nextReaders.assign(original.operations.size(), 0);
    result.readIncidences.resize(original.operations.size());
    result.writeIncidences.resize(original.operations.size());
    if (cell >= original.cells.size()) {
      result.reason = "invalid original physical cell";
      return;
    }
    if (hasRepetition(original.body)) {
      result.reason = "factored occurrence projection has an unresolved repetition";
      return;
    }
    State forward{1, 0, 0};
    forward = transfer(original.body, forward, false);
    result.demands = forward.demands;
    result.finalWriters = forward.writers;
    result.finalReaders = forward.readers;
    State backward{0, 0, 0};
    transfer(original.body, backward, true);
    result.complete = true;
  }

  const FactoredUseResult &get() const { return result; }

private:
  struct State {
    std::size_t writers = 0, readers = 0, demands = 0;
  };

  bool hasRepetition(const Region &region) const {
    if (region.kind == Region::For || region.kind == Region::While) {
      return true;
    }
    for (const auto &child : region.children) {
      if (hasRepetition(child)) {
        return true;
      }
    }
    return false;
  }

  std::size_t append(FactoredUseNode node) {
    const auto id = result.nodes.size();
    result.nodes.push_back(node);
    return id;
  }

  std::size_t access(std::size_t operation) {
    FactoredUseNode node;
    node.kind = FactoredUseNode::Kind::Access;
    node.operation = operation;
    return append(node);
  }

  std::size_t both(std::size_t a, std::size_t b, FactoredUseNode::Sort sort) {
    if (!a || a == b) {
      return b;
    }
    if (!b) {
      return a;
    }
    FactoredUseNode node;
    node.kind = FactoredUseNode::Kind::Both;
    node.sort = sort;
    node.left = a;
    node.right = b;
    return append(node);
  }

  std::size_t choose(std::size_t owner, std::size_t yes, std::size_t no,
                     FactoredUseNode::Sort sort) {
    if (yes == no) {
      return yes;
    }
    FactoredUseNode node;
    node.kind = FactoredUseNode::Kind::Choose;
    node.sort = sort;
    node.owner = owner;
    node.left = yes;
    node.right = no;
    return append(node);
  }

  std::size_t demand(std::size_t sources, std::size_t target, FactoredUseNode::Hazard hazard) {
    if (!sources) {
      return 0;
    }
    FactoredUseNode node;
    node.kind = FactoredUseNode::Kind::Demand;
    node.sort = FactoredUseNode::Sort::Demands;
    node.hazard = hazard;
    node.left = sources;
    node.operation = target;
    return append(node);
  }

  State transfer(const Region &region, State state, bool backward) {
    if (region.kind == Region::Operation) {
      if (region.operation >= original.operations.size()) {
        return state;
      }
      bool reads = false, writes = false, fullWrite = false;
      for (std::size_t incidence = 0; incidence < original.operations[region.operation].accesses.size();
           ++incidence) {
        const auto &effect = original.operations[region.operation].accesses[incidence];
        if (effect.cell == cell) {
          reads |= effect.read;
          writes |= effect.write;
          fullWrite |= effect.write && effect.definiteWrite;
          if (!backward) {
            if (effect.read) {
              result.readIncidences[region.operation].push_back(incidence);
            }
            if (effect.write) {
              result.writeIncidences[region.operation].push_back(incidence);
            }
          }
        }
      }
      if (!backward) {
        result.priorWriters[region.operation] = state.writers;
        result.priorReaders[region.operation] = state.readers;
        if (reads) {
          state.demands = both(state.demands,
                               demand(state.writers, region.operation, FactoredUseNode::Hazard::RAW),
                               FactoredUseNode::Sort::Demands);
        }
        if (writes) {
          state.demands = both(state.demands,
                               demand(state.writers, region.operation, FactoredUseNode::Hazard::WAW),
                               FactoredUseNode::Sort::Demands);
          state.demands = both(state.demands,
                               demand(state.readers, region.operation, FactoredUseNode::Hazard::WAR),
                               FactoredUseNode::Sort::Demands);
        }
      } else {
        result.nextWriters[region.operation] = state.writers;
        result.nextReaders[region.operation] = state.readers;
      }
      if (fullWrite) {
        state.writers = access(region.operation);
        state.readers = backward && reads ? access(region.operation) : 0;
      } else {
        if (writes) {
          state.writers = both(state.writers, access(region.operation),
                               FactoredUseNode::Sort::Origins);
        }
        if (reads) {
          state.readers = both(state.readers, access(region.operation),
                               FactoredUseNode::Sort::Origins);
        }
      }
      return state;
    }
    const bool binaryChoice = region.kind == Region::Choice && region.children.size() == 2;
    if (binaryChoice) {
      auto yes = transfer(region.children[0], state, backward);
      auto no = transfer(region.children[1], state, backward);
      return {choose(region.originalOwner, yes.writers, no.writers,
                     FactoredUseNode::Sort::Origins),
              choose(region.originalOwner, yes.readers, no.readers,
                     FactoredUseNode::Sort::Origins),
              choose(region.originalOwner, yes.demands, no.demands,
                     FactoredUseNode::Sort::Demands)};
    }
    if (!backward) {
      for (const auto &child : region.children) {
        state = transfer(child, state, false);
      }
    } else {
      for (auto child = region.children.rbegin(); child != region.children.rend(); ++child) {
        state = transfer(*child, state, true);
      }
    }
    return state;
  }

  const OriginalStructure &original;
  std::size_t cell;
  FactoredUseResult result;
};

} // namespace mlir::pto::frontiersynch
#endif
