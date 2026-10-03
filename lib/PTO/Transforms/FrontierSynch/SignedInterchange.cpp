// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SignedInternal.h"
#include <limits>
namespace mlir::pto::frontiersynch {
using namespace signed_detail;
FailureOr<SymbolicPrimitive> SignedRelation::toSymbolic() const
{
    const auto schema = owner->schema();
    const auto space = *schema->space(source, target);
    auto result = presburger::PresburgerRelation::getEmpty(space);
    for (const auto& piece : data) {
        const auto shape = layout(owner, source, target, piece).value;
        if (std::uint64_t(space.getNumVars()) + shape.free() >= std::numeric_limits<unsigned>::max()) {
            return failure();
        }
        auto localSpace = space;
        localSpace.insertVar(presburger::VarKind::Local, 0, shape.free());
        presburger::IntegerRelation relation(localSpace);
        SmallVector<unsigned> original;
        auto tuple = [&](SymbolicTuple role, const SignedTag& tag, unsigned offset, unsigned active) {
            if (role == SymbolicTuple::Occurrence || role == SymbolicTuple::Event) {
                relation.addBound(presburger::BoundType::EQ, offset, BigInt(std::int64_t(*tag.site)));
                for (unsigned i = 0; i < active; ++i) {
                    original.push_back(offset + i + 1);
                }
                for (unsigned i = active; i < schema->coordinateDepth(); ++i) {
                    relation.addBound(presburger::BoundType::EQ, offset + i + 1, BigInt(0));
                }
                if (role == SymbolicTuple::Event) {
                    relation.addBound(
                        presburger::BoundType::EQ, offset + schema->coordinateDepth() + 1,
                        BigInt(*tag.kind == PeriodicEventKind::Start ? 0 : 1));
                }
            } else {
                for (unsigned i = 0; i < active; ++i) {
                    original.push_back(offset + i);
                }
            }
        };
        tuple(source, piece.domain, 0, shape.domain);
        tuple(target, piece.range, *schema->arity(source), shape.range);
        for (unsigned i = 0; i < shape.parameters; ++i) {
            original.push_back(space.getNumDimVars() + i);
        }
        for (unsigned i = 0; i < shape.free(); ++i) {
            SmallVector<BigInt> row(relation.getNumCols(), BigInt(0));
            row[original[i]] = BigInt(1);
            row[space.getNumVars() + i] = -owner->period();
            row.back() = -piece.residues[i];
            relation.addEquality(row);
        }
        for (const auto& [direction, bound] : decode(owner, source, target, piece).value.bounds) {
            SmallVector<BigInt> row(relation.getNumCols(), BigInt(0));
            for (auto term : {direction.first, direction.second}) {
                if (term) {
                    row[space.getNumVars() + (term > 0 ? term : -term) - 1] -= BigInt(term > 0 ? 1 : -1);
                }
            }
            row.back() = bound;
            relation.addInequality(row);
        }
        result.unionInPlace(relation);
    }
    return SymbolicPrimitive::import(schema, source, target, schema->parameters(), result);
}
} // namespace mlir::pto::frontiersynch
