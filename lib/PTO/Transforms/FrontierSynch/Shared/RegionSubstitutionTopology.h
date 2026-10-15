// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Arena-wide structural reverse adjacency; no per-root closures or rewritten IDs.
#ifndef PTO_FRONTIER_REGIONSUBSTITUTIONTOPOLOGY_H
#define PTO_FRONTIER_REGIONSUBSTITUTIONTOPOLOGY_H
#include "RegionIntegerExpressionsInternal.h"
#include "llvm/ADT/ScopeExit.h"
#include <set>
namespace mlir::pto::frontiersynch {
struct RegionExpressions::SubstitutionTopology {
    // The indexed prefix may be shorter than the arena; targets and newly
    // rebuilt results need no index. Ordered entries support an old-root
    // prefix query and release each rolled-back edge without allocating.
    std::vector<std::multiset<Id>> parents;
    static void count(uint64_t* work)
    {
        if (work && *work != UINT64_MAX) { ++*work; }
    }
    template<class Visitor>
    static void operands(const Node& node, Visitor&& visit)
    {
        // Match appendOperands exactly without a temporary buffer. Opaque
        // evaluation provenance is separate metadata and is not traversed.
        for (auto operand : {node.a, node.b, node.c}) {
            if (operand != invalid) { visit(operand); }
        }
        if (node.integer) {
            for (auto operand : node.integer->inputs) { visit(operand); }
        }
    }
    void extend(const RegionExpressions& arena, Id root, uint64_t* work)
    {
        while (parents.size() <= root) {
            const auto position = static_cast<Id>(parents.size());
            const auto& node = arena.nodes[position];
            count(work);
            parents.emplace_back();
            std::size_t inserted = 0;
            // Preserve a complete indexed prefix if an edge allocation throws.
            // This cleanup traverses original operands and only erases storage.
            auto rollback = llvm::make_scope_exit([&] {
                std::size_t current = 0;
                operands(node, [&](Id operand) {
                    if (current++ < inserted) { parents[operand].erase(parents[operand].find(position)); }
                });
                parents.pop_back();
            });
            operands(node, [&](Id operand) {
                count(work);
                parents[operand].insert(position);
                ++inserted;
            });
            rollback.release();
        }
    }
    void truncate(const RegionExpressions& arena, std::size_t size)
    {
        // Called before node rollback. Erasing one entry for each operand
        // handles repeated edges and releases all temporary fan-out storage.
        // No cleanup allocation occurs in the noexcept transaction destructor.
        while (parents.size() > size) {
            const auto id = static_cast<Id>(parents.size() - 1);
            operands(arena.nodes[id], [&](Id operand) {
                parents[operand].erase(parents[operand].find(id));
            });
            parents.pop_back();
        }
    }
    void appendParents(Id source, Id root, SmallVectorImpl<Id>& pending, uint64_t* work) const
    {
        const auto& users = parents[source];
        // Charge the container lookup and each traversed edge. Internal tree
        // comparisons, like existing hash probes, are not separate node visits.
        count(work);
        const auto end = users.upper_bound(root);
        for (auto it = users.begin(); it != end; ++it) {
            count(work); pending.push_back(*it);
        }
    }
};
} // namespace mlir::pto::frontiersynch
#endif
