// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Proof-only ranges follow the immutable machine-bit DAG. Unknown inputs and
// unsupported operations retain full ranges; no caller fact is inferred from
// SSA names, tensor extents, or representative executions. LLVM ConstantRange
// preserves wrapping arithmetic and signed comparison boundaries.
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instructions.h"
#include <optional>
namespace mlir::pto::frontiersynch {
std::map<RegionExpressions::Id, std::pair<uint64_t, uint64_t>>
RegionExpressions::unsignedProofBounds(ArrayRef<Id> order) const
{
    using Range = llvm::ConstantRange;
    using Predicate = llvm::CmpInst::Predicate;
    const Range full(64, true), boolean(APInt(64, 0), APInt(64, 2));
    std::map<Id, Range> ranges;
    auto lookup = [&](Id id) {
        auto found = ranges.find(id);
        return found == ranges.end() ? full : found->second;
    };
    auto predicate = [&](Kind kind) -> std::optional<Predicate> {
        switch (kind) {
        case Kind::Eq: return llvm::CmpInst::ICMP_EQ;
        case Kind::Lt: return llvm::CmpInst::ICMP_ULT;
        case Kind::Le: return llvm::CmpInst::ICMP_ULE;
        case Kind::SLt: return llvm::CmpInst::ICMP_SLT;
        case Kind::SLe: return llvm::CmpInst::ICMP_SLE;
        default: return std::nullopt;
        }
    };
    auto arm = [&](const Node& select, Id value, bool yes) {
        const auto& comparison = nodes[select.a];
        auto pred = predicate(comparison.kind);
        if (!pred) { return lookup(value); }
        auto other = comparison.b;
        if (comparison.b == value) {
            other = comparison.a;
            *pred = llvm::CmpInst::getSwappedPredicate(*pred);
        } else if (comparison.a != value) { return lookup(value); }
        if (!constantValue(other)) { return lookup(value); }
        if (!yes) { *pred = llvm::CmpInst::getInversePredicate(*pred); }
        return lookup(value).intersectWith(Range::makeAllowedICmpRegion(*pred, lookup(other)));
    };
    for (auto id : order) {
        const auto& node = nodes[id];
        auto range = node.boolean ? boolean : full;
        if (node.kind == Kind::Constant) { range = Range(APInt(64, node.literal)); }
        else if (node.kind == Kind::Select) { range = arm(node, node.b, true).unionWith(arm(node, node.c, false)); }
        else if (node.kind == Kind::Add) { range = lookup(node.a).add(lookup(node.b)); }
        else if (node.kind == Kind::Sub) { range = lookup(node.a).sub(lookup(node.b)); }
        else if ((node.kind == Kind::Div || node.kind == Kind::Rem) && !lookup(node.b).contains(APInt(64, 0))) {
            range = node.kind == Kind::Div ? lookup(node.a).udiv(lookup(node.b)) : lookup(node.a).urem(lookup(node.b));
        }
        ranges.emplace(id, std::move(range));
    }
    std::map<Id, std::pair<uint64_t, uint64_t>> result;
    for (const auto& [id, range] : ranges) {
        // A wrapped set's unsigned hull may be full. Losing precision here is
        // safe; narrowing it to a signed half would create a false premise.
        result.emplace(id, range.isEmptySet() ? std::make_pair(uint64_t(0), UINT64_MAX) :
            std::make_pair(range.getUnsignedMin().getZExtValue(), range.getUnsignedMax().getZExtValue()));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
