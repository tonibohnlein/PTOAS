// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Synthetic declarations test the common representation independently of op semantics.
#ifndef PTO_TEST_SYNC_EXPLICIT_ACCESS_CHECKS_H
#define PTO_TEST_SYNC_EXPLICIT_ACCESS_CHECKS_H
#include "PTO/IR/PTOAccessRegion.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "../../lib/PTO/Transforms/InsertSync/SyncEffectRanges.h"

inline bool checkExplicitAccessContracts(mlir::func::FuncOp function, const mlir::pto::SyncInput& input)
{
    using namespace mlir;
    using namespace mlir::pto;
    if (function->hasAttr("test.pointer_v2")) {
        auto op = *function.getOps<PTOLoadOp>().begin();
        auto* context = function.getContext();
        auto contract = makeByteAccessRegion(op.getPtrMutable(), AffineMap::get(0, 1, {}, context),
            AffineMap::get(0, 1, {getAffineSymbolExpr(0, context) * 4}, context), 4, {1});
        auto region = resolveSelectedRegion(input, op.getPtr(), op, contract);
        return region && region->base == op.getPtr() && region->elementBytes == 4 &&
            region->extents.empty() && region->symbols.size() == 1;
    }
    if (!function->hasAttr("test.access_v2")) {
        return true;
    }
    auto effects = input.accesses().effects();
    if (effects.size() != 1) {
        return false;
    }
    const auto& effect = effects.front();
    auto* op = effect.phase->elementOp;
    auto& operand = op->getOpOperand(0);
    auto* context = function.getContext();
    auto c = [&](int64_t v) { return getAffineConstantExpr(v, context); };
    auto d = getAffineDimExpr(0, context);
    auto s = getAffineSymbolExpr(0, context);
    auto singleton = AffineMap::get(0, 0, {}, context);
    auto oneOffset = [&](int64_t v) { return AffineMap::get(0, 0, {c(v)}, context); };
    auto apply = [&](DictionaryAttr contract) {
        auto copy = effect;
        SyncMemoryEffect declared(MemoryEffects::Read::get(), &operand, contract);
        mlir::pto::detail::applyAccessCoverage(input, copy, {declared});
        return copy;
    };
    auto point = makeByteAccessRegion(operand, singleton, oneOffset(12), 4);
    auto result = apply(point);
    if (result.precision != SyncAccessPrecision::Exact || !result.exactRanges || result.ranges.size() != 1 ||
        result.ranges.front().begin != 1036 || result.ranges.front().end != 1040) {
        return false;
    }
    auto stride = makeByteAccessRegion(operand, oneOffset(3), AffineMap::get(1, 0, {d * 8}, context), 4);
    result = apply(stride);
    if (!result.exactRanges || result.ranges.size() != 3 || result.ranges.back().begin != 1040) {
        return false;
    }
    auto empty = makeByteAccessRegion(operand, oneOffset(0), AffineMap::get(1, 0, {d}, context), 4);
    result = apply(empty);
    if (result.precision != SyncAccessPrecision::Exact || !result.exactRanges || !result.ranges.empty()) {
        return false;
    }
    auto symbolic = makeByteAccessRegion(operand, AffineMap::get(0, 1, {}, context),
        AffineMap::get(0, 1, {s * 4}, context), 4, {1});
    result = apply(symbolic);
    if (result.precision != SyncAccessPrecision::Exact || result.exactRanges ||
        !result.region || result.region->symbols.size() != 1) {
        return false;
    }
    auto symbolicExtent = makeByteAccessRegion(operand, AffineMap::get(0, 1, {s}, context),
        AffineMap::get(1, 1, {d * 4}, context), 4, {1});
    result = apply(symbolicExtent);
    if (result.precision != SyncAccessPrecision::Exact || result.exactRanges || !result.region) {
        return false;
    }
    auto coordinates = makeExplicitAccessRegion(operand, singleton, AffineMap::get(0, 0, {c(1), c(2)}, context));
    result = apply(coordinates);
    if (!result.exactRanges || result.ranges.size() != 1 || result.ranges.front().begin != 1064) {
        return false;
    }
    Builder builder(context);
    for (auto [key, bad] : SmallVector<std::pair<StringRef, Attribute>>{
             {"byte_width", builder.getI64IntegerAttr(0)},
             {"byte_width", builder.getI64IntegerAttr(-1)},
             {"addressing", builder.getStringAttr("invalid")},
             {"symbol_operands", builder.getDenseI64ArrayAttr({99})},
             {"extents", AffineMapAttr::get(oneOffset(-1))},
             {"coordinates", AffineMapAttr::get(AffineMap::get(1, 0, {d}, context))},
             {"coordinates", AffineMapAttr::get(oneOffset(INT64_MAX))}}) {
        NamedAttrList invalid(point);
        invalid.set(key, bad);
        result = apply(invalid.getDictionary(context));
        if (result.precision == SyncAccessPrecision::Exact || result.region) {
            return false;
        }
    }
    SyncMemoryEffect first(MemoryEffects::Read::get(), &operand, point);
    SyncMemoryEffect duplicate(MemoryEffects::Read::get(), &operand, point);
    result = effect;
    mlir::pto::detail::applyAccessCoverage(input, result, {first, duplicate});
    if (result.precision != SyncAccessPrecision::Exact || result.regions.size() != 2) {
        return false;
    }
    SyncMemoryEffect unresolved(MemoryEffects::Read::get(), &operand);
    mlir::pto::detail::applyAccessCoverage(input, result, {first, unresolved});
    return result.precision == SyncAccessPrecision::UpperBound && !result.hasDefiniteWrites();
}
#endif
