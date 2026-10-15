// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDMANDATORYTRANSPORT_H
#define PTO_FRONTIERSYNCH_BOUNDEDMANDATORYTRANSPORT_H
#include "llvm/ADT/BitVector.h"
#include <limits>
#include <map>
#include <utility>
#include <vector>
namespace mlir::pto::frontiersynch::detail {
// Every edge is supplied by the bounded producer's mandatory-path certificate.
// The complete list covers every physical residue. A missing path is unknown,
// not proof of nonreachability. This directory contains no expression IDs or
// runtime state; cached rows survive expression transactions unchanged.
class BoundedMandatoryTransport {
public:
    using Graph = std::vector<std::vector<unsigned>>;
    explicit BoundedMandatoryTransport(std::vector<Graph> graphs)
        : graphs(std::move(graphs)), nodes(this->graphs.empty() ? 0 : this->graphs.front().size()),
          valid(validate()) {}
    bool alwaysReaches(unsigned source, unsigned target)
    {
        if (!valid || source >= nodes || target >= nodes) { return false; }
        auto found = rows.find(source);
        if (found != rows.end()) { return found->second.test(target); }
        llvm::BitVector common(nodes, true);
        for (const auto& graph : graphs) {
            llvm::BitVector reached(nodes);
            std::vector<unsigned> pending{source};
            while (!pending.empty()) {
                const auto next = pending.back(); pending.pop_back();
                if (reached.test(next)) { continue; }
                reached.set(next);
                pending.insert(pending.end(), graph[next].begin(), graph[next].end());
            }
            common &= reached;
        }
        return rows.emplace(source, std::move(common)).first->second.test(target);
    }
    std::size_t retainedRows() const { return rows.size(); }
private:
    const std::vector<Graph> graphs;
    const std::size_t nodes;
    const bool valid;
    std::map<unsigned, llvm::BitVector> rows;
    bool validate() const
    {
        if (!nodes || nodes > std::numeric_limits<unsigned>::max()) { return false; }
        for (const auto& graph : graphs) {
            if (graph.size() != nodes) { return false; }
            for (const auto& edges : graph) {
                for (auto target : edges) {
                    if (target >= nodes) { return false; }
                }
            }
        }
        return true;
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
