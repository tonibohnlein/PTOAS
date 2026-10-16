// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_REGIONTOTALITY_H
#define PTO_FRONTIER_REGIONTOTALITY_H
#include "RegionConstruction.h"
namespace mlir::pto::frontiersynch::detail {
inline bool RegionConstructionAccess::totalScalarDag(
    const RegionExpressions& arena, ArrayRef<RegionExpressions::Id> roots, uint64_t& visited)
{
    bool total = llvm::all_of(roots, [&](auto root) { return root < arena.size(); });
    visited = arena.visitReferenced(roots, [&](auto id) {
        // All other current constructors are total on defined atomic inputs:
        // unsigned arithmetic wraps and division has a positive literal divisor.
        // Opaque evaluation provenance is atomic, never traversed or rewritten.
        using Kind = RegionExpressions::Kind;
        switch (arena.nodes[id].kind) {
            case Kind::Constant: case Kind::Input: case Kind::Add: case Kind::Sub:
            case Kind::Div: case Kind::Rem: case Kind::Lt: case Kind::Le: case Kind::Eq:
            case Kind::SLt: case Kind::SLe: case Kind::And: case Kind::Or: case Kind::Not:
            case Kind::Select: return;
            case Kind::Integer: break;
        }
        total = false;
    });
    return total;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
