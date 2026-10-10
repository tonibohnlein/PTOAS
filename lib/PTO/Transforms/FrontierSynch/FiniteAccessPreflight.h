// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Count only storage pairs that can appear in the complete reference order.
// This bounds construction work; it never changes a mathematical class.
#ifndef PTO_FRONTIERSYNCH_FINITEACCESSPREFLIGHT_H
#define PTO_FRONTIERSYNCH_FINITEACCESSPREFLIGHT_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "llvm/ADT/MapVector.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch::detail {
struct FiniteAccessCounts {
    uint64_t reads = 0, writes = 0;
};
inline bool addPairProduct(uint64_t left, uint64_t right, uint64_t limit, uint64_t& total)
{
    if (total > limit || (right && left > (limit - total) / right)) {
        return false;
    }
    total += left * right;
    return true;
}
inline std::optional<uint64_t> finiteAccessPairEstimate(const ArithmeticProgram& program, uint64_t limit)
{
    using Counts = llvm::MapVector<std::pair<AddressSpace, Value>, FiniteAccessCounts>;
    std::map<std::size_t, Counts> sites;
    std::set<std::pair<std::size_t, std::size_t>> order;
    for (const auto& relation : program.primitives.relations) {
        if (relation.pieces.empty()) {
            continue;
        }
        if (relation.kind == PrimitiveKind::Order && relation.sourceSite && relation.targetSite) {
            order.emplace(*relation.sourceSite, *relation.targetSite);
        }
        const bool read = relation.kind == PrimitiveKind::Reads;
        if (!read && relation.kind != PrimitiveKind::Writes) {
            continue;
        }
        if (!relation.sourceSite || !relation.storageSpace) {
            return std::nullopt;
        }
        auto& count = sites[*relation.sourceSite][{*relation.storageSpace, relation.storageBase}];
        auto& value = read ? count.reads : count.writes;
        if (relation.pieces.size() > UINT64_MAX - value) {
            return std::nullopt;
        }
        value += relation.pieces.size();
    }
    uint64_t total = 0;
    for (auto [source, target] : order) {
        auto left = sites.find(source), right = sites.find(target);
        if (left == sites.end() || right == sites.end()) {
            continue;
        }
        for (const auto& [storage, a] : left->second) {
            auto found = right->second.find(storage);
            if (found == right->second.end()) {
                continue;
            }
            const auto& b = found->second;
            if (!addPairProduct(a.writes, b.writes, limit, total) || !addPairProduct(a.writes, b.reads, limit, total) ||
                !addPairProduct(a.reads, b.writes, limit, total)) {
                return std::nullopt;
            }
        }
    }
    return total;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
