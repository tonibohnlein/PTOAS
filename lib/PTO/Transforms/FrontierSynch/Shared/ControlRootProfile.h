// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared, typed expression DAG for exact regional queries and endpoint recipes.
// Diagnostic-only union/dependency accounting. Discover topology once, then
// propagate root bitsets in consumer-to-operand order; never rescan a whole DAG
// independently for each root. Categories overlap; the union counts each node once.
#ifndef PTO_FRONTIER_CONTROLROOTPROFILE_H
#define PTO_FRONTIER_CONTROLROOTPROFILE_H
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <array>
#include <vector>
#include <map>
namespace mlir::pto::frontiersynch {
class FamilyEndpointRecipe;
llvm::json::Array profileFamilyControlRoots(const FamilyEndpointRecipe& recipe);
} // namespace mlir::pto::frontiersynch
namespace mlir::pto::frontiersynch::detail {
template <class Key>
class ControlRootProfile {
    struct Root {
        Key value;
        unsigned role;
    };
    struct Node {
        Key value;
        llvm::SmallVector<std::size_t> operands;
        unsigned users = 0;
    };
    llvm::SmallVector<Root> roots;

public:
    llvm::DenseMap<Key, unsigned> roleMasks;
    std::array<Key, 3> largestRoots{};
    // Roles: exact guard, logical selector, physical ID. A repeated use is not
    // another distinct expression root, but remains visible in root_uses.
    void add(Key value, unsigned role) { roots.push_back({value, role}); }
    template <class Dependencies, class Countable>
    llvm::json::Object report(Dependencies dependencies, Countable countable)
    {
        llvm::DenseMap<Key, std::size_t> index;
        std::vector<Node> nodes;
        auto intern = [&](Key value) {
            auto inserted = index.try_emplace(value, nodes.size());
            if (inserted.second) {
                nodes.push_back({value, {}, 0});
            }
            return inserted.first->second;
        };
        std::map<std::pair<std::size_t, unsigned>, unsigned> unique;
        for (auto root : roots) {
            unique.try_emplace({intern(root.value), root.role}, unique.size());
        }
        uint64_t edges = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            auto values = dependencies(nodes[i].value);
            for (auto value : values) {
                const auto operand = intern(value);
                nodes[i].operands.push_back(operand);
                ++nodes[operand].users;
                ++edges;
            }
        }
        std::vector<unsigned> roles(unique.size());
        std::vector<std::size_t> rootNodes(unique.size());
        for (const auto& entry : unique) {
            rootNodes[entry.second] = entry.first.first;
            roles[entry.second] = entry.first.second;
        }
        llvm::SmallVector<std::size_t> ready, order;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (!nodes[i].users) {
                ready.push_back(i);
            }
        }
        while (!ready.empty()) {
            const auto i = ready.pop_back_val();
            order.push_back(i);
            for (auto operand : nodes[i].operands) {
                if (!--nodes[operand].users) {
                    ready.push_back(operand);
                }
            }
        }
        if (order.size() != nodes.size()) {
            return llvm::json::Object{{"error", "cyclic control dependency"}};
        }
        std::vector<uint64_t> sizes(unique.size()), bits(nodes.size());
        std::vector<unsigned> masksByNode(nodes.size()), rootCounts(nodes.size()), guardCounts(nodes.size());
        uint64_t rootMemberships = 0;
        // Reuse one topology and one word per node. Exact largest-root and
        // sharing counts require root reachability work, but never N*R storage.
        for (std::size_t begin = 0; begin < unique.size(); begin += 64) {
            std::fill(bits.begin(), bits.end(), 0);
            std::array<uint64_t, 3> roleBits{};
            const auto end = std::min(begin + 64, unique.size());
            for (auto root = begin; root < end; ++root) {
                const uint64_t bit = uint64_t(1) << (root - begin);
                bits[rootNodes[root]] |= bit;
                roleBits[roles[root]] |= bit;
            }
            for (auto i : order) {
                for (auto operand : nodes[i].operands) {
                    bits[operand] |= bits[i];
                }
            }
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                if (!countable(nodes[i].value)) {
                    continue;
                }
                auto value = bits[i];
                rootCounts[i] = std::min(2U, rootCounts[i] + unsigned(llvm::popcount(value)));
                guardCounts[i] = std::min(2U, guardCounts[i] + unsigned(llvm::popcount(value & roleBits[0])));
                for (unsigned role = 0; role < 3; ++role) {
                    if (value & roleBits[role]) {
                        masksByNode[i] |= 1U << role;
                    }
                }
                while (value) {
                    const auto bit = llvm::countr_zero(value);
                    ++sizes[begin + bit];
                    ++rootMemberships;
                    value &= value - 1;
                }
            }
        }
        std::array<uint64_t, 8> masks{};
        uint64_t shared = 0, guardShared = 0, total = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const auto mask = masksByNode[i];
            roleMasks[nodes[i].value] = mask;
            if (mask) {
                ++masks[mask];
                ++total;
            }
            shared += rootCounts[i] > 1;
            guardShared += guardCounts[i] > 1;
        }
        std::array<uint64_t, 3> totals{}, largest{}, sums{}, counts{};
        for (unsigned mask = 1; mask < masks.size(); ++mask) {
            for (unsigned role = 0; role < 3; ++role) {
                if (mask & (1U << role)) {
                    totals[role] += masks[mask];
                }
            }
        }
        for (unsigned i = 0; i < sizes.size(); ++i) {
            auto role = roles[i];
            ++counts[role];
            sums[role] += sizes[i];
            if (counts[role] == 1 || sizes[i] > largest[role]) {
                largest[role] = sizes[i];
                largestRoots[role] = nodes[rootNodes[i]].value;
            }
        }
        llvm::json::Array categories, overlaps;
        const char* names[]{"guards", "selectors", "ids"};
        for (unsigned role = 0; role < 3; ++role) {
            categories.push_back(
                llvm::json::Object{
                    {"role", names[role]},
                    {"roots", counts[role]},
                    {"nodes", totals[role]},
                    {"exclusive_nodes", masks[1U << role]},
                    {"largest_root", largest[role]},
                    {"sum_root_nodes", sums[role]}});
        }
        for (unsigned mask = 1; mask < masks.size(); ++mask) {
            overlaps.push_back(llvm::json::Object{{"role_mask", mask}, {"nodes", masks[mask]}});
        }
        return llvm::json::Object{
            {"root_uses", roots.size()},
            {"union_nodes", total},
            {"shared_nodes", shared},
            {"shared_guard_nodes", guardShared},
            {"topology_nodes", nodes.size()},
            {"topology_edges", edges},
            {"root_batches", (unique.size() + 63) / 64},
            {"root_memberships", rootMemberships},
            {"scratch_node_words", nodes.size()},
            {"categories", std::move(categories)},
            {"overlaps", std::move(overlaps)}};
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
