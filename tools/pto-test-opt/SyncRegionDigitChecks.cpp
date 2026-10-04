// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compare generic byte-map materialization against explicit coordinate images.
#include "../../lib/PTO/Transforms/InsertSync/SyncRegionDigits.h"
#include "../../lib/PTO/Transforms/InsertSync/SyncEffectRanges.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
using namespace mlir::pto;
namespace {
int64_t evaluate(AffineExpr expression, ArrayRef<int64_t> coordinates)
{
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        return constant.getValue();
    }
    if (auto dim = dyn_cast<AffineDimExpr>(expression)) {
        return coordinates[dim.getPosition()];
    }
    auto binary = cast<AffineBinaryOpExpr>(expression);
    const auto left = evaluate(binary.getLHS(), coordinates), right = evaluate(binary.getRHS(), coordinates);
    switch (expression.getKind()) {
        case AffineExprKind::Add: return left + right;
        case AffineExprKind::Mul: return left * right;
        case AffineExprKind::FloorDiv: return left / right;
        case AffineExprKind::Mod: return left % right;
        default: llvm_unreachable("test map grammar");
    }
}
SyncAccessRegion region(AffineExpr expression, ArrayRef<int64_t> shape, unsigned bytes)
{
    SyncAccessRegion result;
    result.byteOffset = expression;
    result.elementBytes = bytes;
    for (auto size : shape) {
        result.extents.push_back(getAffineConstantExpr(size, expression.getContext()));
    }
    return result;
}
std::set<uint64_t> image(const SyncAccessRegion& access)
{
    SmallVector<int64_t> shape, coordinates(access.extents.size(), 0);
    uint64_t count = 1;
    for (auto extent : access.extents) {
        const auto size = cast<AffineConstantExpr>(extent).getValue();
        shape.push_back(size);
        count *= size;
    }
    std::set<uint64_t> result;
    for (uint64_t i = 0; i < count; ++i) {
        auto index = i;
        for (unsigned axis = 0; axis < shape.size(); ++axis) {
            coordinates[axis] = index % shape[axis];
            index /= shape[axis];
        }
        const auto start = static_cast<uint64_t>(evaluate(access.byteOffset, coordinates));
        for (unsigned byte = 0; byte < access.elementBytes; ++byte) {
            result.insert(start + byte);
        }
    }
    return result;
}
bool check(StringRef name, const SyncAccessRegion& access, std::optional<std::size_t> expectedRanges = std::nullopt)
{
    SmallVector<SyncStorageCell> actual;
    if (!mlir::pto::detail::materializeDigitRegion(access, AddressSpace::MAT, actual) ||
        (expectedRanges && actual.size() != *expectedRanges)) {
        llvm::errs() << "digit materialization failed: " << name << "\n";
        return false;
    }
    std::set<uint64_t> bytes;
    uint64_t previousEnd = 0;
    for (const auto& range : actual) {
        const bool valid = range.space == AddressSpace::MAT && range.begin < range.end &&
            (bytes.empty() || previousEnd < range.begin);
        if (!valid) {
            return false;
        }
        for (uint64_t byte = range.begin; byte < range.end; ++byte) {
            bytes.insert(byte);
        }
        previousEnd = range.end;
    }
    if (bytes != image(access)) {
        llvm::errs() << "digit byte image differs: " << name << "\n";
        return false;
    }
    SmallVector<SyncStorageCell> viaShared;
    if (!mlir::pto::detail::materializeRegion(access, AddressSpace::MAT, viaShared) ||
        viaShared.size() != actual.size()) {
        return false;
    }
    for (auto [a, b] : llvm::zip(actual, viaShared)) {
        if (a.space != b.space || a.begin != b.begin || a.end != b.end) {
            return false;
        }
    }
    return true;
}
bool reject(const SyncAccessRegion& access, bool alsoShared = false)
{
    SmallVector<SyncStorageCell> result{{AddressSpace::GM, 123, 127}};
    const bool accepted = mlir::pto::detail::materializeDigitRegion(access, AddressSpace::MAT, result);
    const bool preserved = !accepted && result.size() == 1 && result.front().space == AddressSpace::GM &&
        result.front().begin == 123 && result.front().end == 127;
    if (!preserved || !alsoShared) {
        return preserved;
    }
    return !mlir::pto::detail::materializeRegion(access, AddressSpace::MAT, result) &&
        result.size() == 1 && result.front().space == AddressSpace::GM &&
        result.front().begin == 123 && result.front().end == 127;
}
} // namespace
bool checkSyncRegionDigits(func::FuncOp function)
{
    if (!function->hasAttr("test.digit_regions")) {
        return true;
    }
    auto* context = function.getContext();
    auto d0 = getAffineDimExpr(0, context), d1 = getAffineDimExpr(1, context);
    auto c = [&](int64_t value) { return getAffineConstantExpr(value, context); };
    const auto blocked = d0.floorDiv(16) * 512 + d1.floorDiv(16) * 4096 + (d0 % 16) * 32 + (d1 % 16) * 2;
    const bool basic = check("row-major", region(d0 * 12 + d1 * 4, {2, 3}, 4), 1) &&
        check("permutation", region(d0 * 4 + d1 * 12, {3, 2}, 4), 1) &&
        check("overlapping-strides", region(d0 * 2 + d1 * 3, {3, 2}, 4), 1) &&
        check("negative-stride", region(c(100) - d0 * 8 + d1 * 2, {4, 3}, 2), 4) &&
        check("zero-stride", region(d1 * 4, {10000, 8}, 4), 1) &&
        check("digit-tail", region(d0.floorDiv(4) * 32 + (d0 % 4) * 2, {7}, 2), 2) &&
        check("two-tails", region(d0.floorDiv(4) * 128 + (d0 % 4) * 2 +
                                  d1.floorDiv(3) * 512 + (d1 % 3) * 16, {7, 5}, 2)) &&
        check("linear-plus-digits", region(d0 * 2 + d0.floorDiv(4) * 16 + (d0 % 4) * 2, {9}, 2)) &&
        check("negative-digits", region(c(80) - d0.floorDiv(4) * 24 - (d0 % 4) * 2, {7}, 2), 2) &&
        check("empty", region(d0, {0}, 1), 0) && check("scalar", region(c(9), {}, 4), 1);
    const bool layouts = check("blocked-full", region(blocked, {128, 128}, 2), 1) &&
        check("partial-eight-intervals", region(d0.floorDiv(16) * 8192 +
                  (d0 % 16) * 128 + d1 * 2, {128, 64}, 2), 8) &&
        check("full-A", region(blocked, {128, 256}, 2), 1) &&
        check("full-B", region(blocked, {128, 512}, 2), 1) &&
        check("local-A", region(d0.floorDiv(16) * 512 + d1.floorDiv(16) * 2048 +
              (d0 % 16) * 32 + (d1 % 16) * 2, {64, 128}, 2), 1) &&
        check("local-B", region(d0.floorDiv(16) * 512 + d1.floorDiv(16) * 2048 +
              (d0 % 16) * 32 + (d1 % 16) * 2, {64, 256}, 2), 1) &&
        check("accumulator", region(d0.floorDiv(16) * 512 + d1.floorDiv(8) * 4096 +
              (d0 % 16) * 32 + (d1 % 8) * 4, {128, 256}, 4), 1) &&
        check("large-contiguous", region(d0.floorDiv(16) * 512 + (d0 % 16) * 32,
                                         {8192}, 32), 1) &&
        check("sparse-limit", region(d0 * 4, {4096}, 2), 4096);
    const bool failures = reject(region(d0 * 4, {4097}, 2), true) &&
        reject(region(c(INT64_MAX) + d0, {2}, 1), true) &&
        reject(region(d0 * INT64_MAX, {3}, 1), true) &&
        reject(region(d0 * INT64_MIN, {2}, 1), true) &&
        reject(region(c(-1) + d0, {2}, 1)) &&
        reject(region(d0.floorDiv(4) + d0.floorDiv(3), {8}, 1)) &&
        reject(region((d0 + d1).floorDiv(4), {8, 8}, 1)) &&
        reject(region(d0.floorDiv(2) * 8 + d1 * 16384, {3, 3000}, 1));
    if (!basic || !layouts || !failures) {
        llvm::errs() << "generic fixed-layout checks failed\n";
        return false;
    }
    llvm::outs() << "generic fixed-layout checks passed\n";
    return true;
}
