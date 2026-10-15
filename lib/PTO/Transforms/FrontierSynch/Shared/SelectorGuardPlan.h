// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Plan conditional execution of owned scalar groups without constructing SSA.
#ifndef PTO_FRONTIER_SELECTORGUARDPLAN_H
#define PTO_FRONTIER_SELECTORGUARDPLAN_H
#include "ScalarConstruction.h"
#include "llvm/ADT/DenseSet.h"
#include <set>
namespace mlir::pto::frontiersynch::detail {
struct SelectorUse {
    uint64_t value = 0, guard = 0; // A zero guard denotes an unguarded/unsupported use.
    Operation* cut = nullptr;
};
struct SelectorGuardGroup {
    Operation* cut = nullptr;
    SmallVector<uint64_t> guards, outputs;
    SmallVector<Type> types;
    SmallVector<std::size_t> instructions;
};
// Each instruction is owned, topological, region-free scalar preparation.
// Terminals include every non-owned use; predicate uses always have guard zero.
// Unsupported users poison their complete producer closure. Cross-group edges
// consequently go from a superset of guards to a subset, preventing cycles.
inline SmallVector<SelectorGuardGroup, 0> planSelectorGuards(
    ArrayRef<ScalarInstruction> instructions, ArrayRef<SelectorUse> terminals)
{
    struct Demand { bool blocked = false; std::set<uint64_t> guards; };
    std::vector<Demand> demand(instructions.size());
    llvm::DenseMap<uint64_t, std::size_t> definitions;
    llvm::DenseMap<uint64_t, SmallVector<std::size_t>> users;
    llvm::DenseSet<uint64_t> exposed;
    for (auto entry : llvm::enumerate(instructions)) {
        for (auto id : entry.value().results) { definitions[id] = entry.index(); }
        for (auto id : entry.value().operands) { users[id].push_back(entry.index()); }
    }
    for (auto use : terminals) {
        auto found = definitions.find(use.value);
        if (found == definitions.end()) { continue; }
        auto& target = demand[found->second];
        target.blocked |= !use.guard || instructions[found->second].cut != use.cut;
        if (use.guard) { target.guards.insert(use.guard); }
        exposed.insert(use.value);
    }
    for (std::size_t i = instructions.size(); i-- > 0;) {
        // An unused operation can still be partial. If it remains eager, its
        // operands cannot become guarded fallback values merely because its
        // results have no consumers. The same rule covers unsupported types.
        // Literal identities also carry structural provenance for later
        // allocation. Keep their literal SSA shape rather than yield fallback
        // results; ordinary exclusive sinking can still move the constant.
        demand[i].blocked = demand[i].blocked || demand[i].guards.empty() ||
            instructions[i].name.getValue() == arith::ConstantOp::getOperationName() ||
            llvm::any_of(instructions[i].types, [](Type type) { return !type.isIntOrIndex(); });
        for (auto operand : instructions[i].operands) {
            auto found = definitions.find(operand);
            if (found == definitions.end()) { continue; }
            auto& producer = demand[found->second];
            producer.blocked |= demand[i].blocked ||
                (!demand[i].guards.empty() && instructions[found->second].cut != instructions[i].cut);
            producer.guards.insert(demand[i].guards.begin(), demand[i].guards.end());
        }
    }
    SmallVector<SelectorGuardGroup, 0> groups;
    std::map<std::pair<Operation*, std::set<uint64_t>>, std::size_t> directory;
    std::vector<std::size_t> membership(instructions.size(), SIZE_MAX);
    for (auto entry : llvm::enumerate(instructions)) {
        const auto i = entry.index();
        if (demand[i].blocked || demand[i].guards.empty() ||
            llvm::any_of(entry.value().types, [](Type type) { return !type.isIntOrIndex(); })) { continue; }
        auto [found, added] = directory.emplace(std::make_pair(entry.value().cut, demand[i].guards), groups.size());
        if (added) {
            SelectorGuardGroup group;
            group.cut = entry.value().cut;
            llvm::append_range(group.guards, demand[i].guards);
            groups.push_back(std::move(group));
        }
        membership[i] = found->second;
        groups[found->second].instructions.push_back(i);
    }
    std::vector<bool> active(groups.size(), false);
    for (std::size_t i = 0; i < groups.size(); ++i) { active[i] = groups[i].guards.size() > 1; }
    // Any owned consumer of a moved group must also execute after its yielded
    // inputs. Include single-guard descendant groups rather than leave an eager
    // consumer before the newly conditional producer.
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        if (membership[i] == SIZE_MAX) { continue; }
        for (auto operand : instructions[i].operands) {
            auto found = definitions.find(operand);
            if (found != definitions.end() && membership[found->second] != SIZE_MAX) {
                active[membership[i]] = active[membership[i]] || active[membership[found->second]];
            }
        }
    }
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        if (membership[i] == SIZE_MAX) { continue; }
        auto& group = groups[membership[i]];
        for (auto result : llvm::enumerate(instructions[i].results)) {
            const auto id = result.value();
            if (exposed.count(id) || llvm::any_of(users.lookup(id), [&](std::size_t user) {
                    return membership[user] != membership[i];
                })) {
                group.outputs.push_back(id);
                group.types.push_back(instructions[i].types[result.index()]);
            }
        }
    }
    SmallVector<SelectorGuardGroup, 0> selected;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (active[i]) { selected.push_back(std::move(groups[i])); }
    }
    groups = std::move(selected);
    // A producer's guard set contains its consumer's set. Strict cardinality
    // therefore orders distinct dependent groups; equal sets were grouped above.
    llvm::stable_sort(groups, [](const auto& a, const auto& b) { return a.guards.size() > b.guards.size(); });
    return groups;
}
// Both consumers execute this exact scalar schedule, without constant folding
// or cache reuse. Result-bearing SCF itself is not a scalar operation.
template<class Consumer>
void constructSelectorGuardGroup(const SelectorGuardGroup& group, Consumer& consumer)
{
    auto guard = consumer.lookup(group.guards.front());
    for (auto id : llvm::drop_begin(group.guards)) { guard = consumer.lor(guard, consumer.lookup(id)); }
    consumer.begin(group, guard);
    for (auto type : group.types) { consumer.zero(type); }
    consumer.finish(group);
}
} // namespace mlir::pto::frontiersynch::detail
#endif
