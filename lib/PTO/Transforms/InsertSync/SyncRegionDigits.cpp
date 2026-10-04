// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Separate each bounded coordinate into quotient/remainder digits, then merge
// dimensions only when their translated byte intervals touch or overlap.
#include "SyncRegionDigits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::detail {
namespace {
struct Axis {
    int64_t linear = 0;
    int64_t quotient = 0;
    int64_t remainder = 0;
    int64_t divisor = 0;
};
struct Digit { int64_t stride; int64_t extent; };
struct Box {
    int64_t origin = 0;
    SmallVector<Digit> digits;
};
struct Form {
    int64_t origin = 0;
    SmallVector<Axis> axes;
    unsigned visited = 0;
};
bool addTo(int64_t& target, int64_t value)
{
    int64_t sum = 0;
    if (llvm::AddOverflow(target, value, sum)) {
        return false;
    }
    target = sum;
    return true;
}
bool parse(AffineExpr expression, int64_t scale, Form& form, unsigned depth = 0)
{
    if (!expression || depth > 64 || ++form.visited > maxMaterializedSlices) {
        return false;
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        int64_t value = 0;
        return !llvm::MulOverflow(constant.getValue(), scale, value) && addTo(form.origin, value);
    }
    if (auto dim = dyn_cast<AffineDimExpr>(expression)) {
        return dim.getPosition() < form.axes.size() && addTo(form.axes[dim.getPosition()].linear, scale);
    }
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary) {
        return false;
    }
    if (expression.getKind() == AffineExprKind::Add) {
        return parse(binary.getLHS(), scale, form, depth + 1) && parse(binary.getRHS(), scale, form, depth + 1);
    }
    auto constant = dyn_cast<AffineConstantExpr>(binary.getRHS());
    if (expression.getKind() == AffineExprKind::Mul && constant) {
        int64_t product = 0;
        return !llvm::MulOverflow(scale, constant.getValue(), product) &&
            parse(binary.getLHS(), product, form, depth + 1);
    }
    auto dim = dyn_cast<AffineDimExpr>(binary.getLHS());
    const bool digit = expression.getKind() == AffineExprKind::FloorDiv ||
        expression.getKind() == AffineExprKind::Mod;
    const bool valid = digit && dim && constant && constant.getValue() > 0 && dim.getPosition() < form.axes.size();
    if (!valid) {
        return false;
    }
    auto& axis = form.axes[dim.getPosition()];
    if (axis.divisor && axis.divisor != constant.getValue()) {
        return false;
    }
    axis.divisor = constant.getValue();
    return addTo(expression.getKind() == AffineExprKind::FloorDiv ? axis.quotient : axis.remainder, scale);
}
bool splitAxis(const Axis& axis, int64_t extent, SmallVectorImpl<Box>& boxes)
{
    if (!axis.divisor) {
        for (auto& box : boxes) {
            box.digits.push_back({axis.linear, extent});
        }
        return true;
    }
    int64_t quotientStride = 0, remainderStride = axis.linear;
    if (llvm::MulOverflow(axis.linear, axis.divisor, quotientStride) ||
        !addTo(quotientStride, axis.quotient) || !addTo(remainderStride, axis.remainder)) {
        return false;
    }
    const int64_t full = extent / axis.divisor, tail = extent % axis.divisor;
    const unsigned count = static_cast<unsigned>(full != 0) + static_cast<unsigned>(tail != 0);
    if (!count || boxes.size() > maxMaterializedSlices / count) {
        return false;
    }
    SmallVector<Box> next;
    for (const auto& box : boxes) {
        if (full) {
            auto fullBox = box;
            fullBox.digits.push_back({quotientStride, full});
            fullBox.digits.push_back({remainderStride, axis.divisor});
            next.push_back(std::move(fullBox));
        }
        if (tail) {
            auto tailBox = box;
            int64_t shift = 0;
            if (llvm::MulOverflow(quotientStride, full, shift) || !addTo(tailBox.origin, shift)) {
                return false;
            }
            tailBox.digits.push_back({remainderStride, tail});
            next.push_back(std::move(tailBox));
        }
    }
    boxes.assign(next.begin(), next.end());
    return true;
}
bool normalize(Box& box)
{
    for (auto& digit : box.digits) {
        if (digit.extent <= 1) {
            digit.stride = 0;
        } else if (digit.stride < 0) {
            int64_t shift = 0, positive = 0;
            if (llvm::MulOverflow(digit.stride, digit.extent - 1, shift) ||
                !addTo(box.origin, shift) || llvm::SubOverflow(int64_t{0}, digit.stride, positive)) {
                return false;
            }
            digit.stride = positive;
        }
    }
    int64_t maximum = box.origin;
    for (const auto& digit : box.digits) {
        int64_t span = 0;
        if (llvm::MulOverflow(digit.stride, digit.extent - 1, span) || !addTo(maximum, span)) {
            return false;
        }
    }
    llvm::sort(box.digits, [](const Digit& a, const Digit& b) { return a.stride < b.stride; });
    return box.origin >= 0;
}
bool emitBox(Box box, unsigned elementBytes, AddressSpace space, SmallVectorImpl<SyncStorageCell>& pending)
{
    if (!normalize(box)) {
        return false;
    }
    uint64_t width = elementBytes;
    unsigned sparse = 0;
    for (; sparse < box.digits.size(); ++sparse) {
        const auto& digit = box.digits[sparse];
        if (static_cast<uint64_t>(digit.stride) > width) {
            break;
        }
        // normalize proved every dimension span and their sum fit int64_t.
        const auto span = static_cast<uint64_t>(digit.stride * (digit.extent - 1));
        if (span > UINT64_MAX - width) {
            return false;
        }
        width += span;
    }
    uint64_t count = 1;
    for (unsigned i = sparse; i < box.digits.size(); ++i) {
        const auto extent = static_cast<uint64_t>(box.digits[i].extent);
        if (extent > (maxMaterializedSlices - pending.size()) / count) {
            return false;
        }
        count *= extent;
    }
    if (count > maxMaterializedSlices - pending.size()) {
        return false;
    }
    for (uint64_t linear = 0; linear < count; ++linear) {
        uint64_t index = linear, begin = static_cast<uint64_t>(box.origin);
        for (unsigned i = sparse; i < box.digits.size(); ++i) {
            const auto& digit = box.digits[i];
            begin += (index % digit.extent) * static_cast<uint64_t>(digit.stride);
            index /= digit.extent;
        }
        if (width > UINT64_MAX - begin) {
            return false;
        }
        pending.push_back({space, begin, begin + width});
    }
    return true;
}
void mergeRanges(SmallVectorImpl<SyncStorageCell>& ranges)
{
    llvm::sort(ranges, [](const SyncStorageCell& a, const SyncStorageCell& b) { return a.begin < b.begin; });
    std::size_t used = 0;
    for (const auto range : ranges) {
        if (used && range.begin <= ranges[used - 1].end) {
            ranges[used - 1].end = std::max(ranges[used - 1].end, range.end);
        } else {
            ranges[used++] = range;
        }
    }
    ranges.resize(used);
}
} // namespace
bool materializeDigitRegion(const SyncAccessRegion& region, AddressSpace space,
                            SmallVectorImpl<SyncStorageCell>& result)
{
    if (region.base || !region.symbols.empty() || !region.elementBytes || !region.byteOffset) {
        return false;
    }
    Form form;
    form.axes.resize(region.extents.size());
    if (!parse(region.byteOffset, 1, form)) {
        return false;
    }
    SmallVector<Box> boxes{{form.origin, {}}};
    for (auto [i, expression] : llvm::enumerate(region.extents)) {
        auto extent = dyn_cast<AffineConstantExpr>(expression);
        if (!extent || extent.getValue() < 0) {
            return false;
        }
        if (!extent.getValue()) {
            result.clear();
            return true;
        }
        if (!splitAxis(form.axes[i], extent.getValue(), boxes)) {
            return false;
        }
    }
    SmallVector<SyncStorageCell> pending;
    for (auto& box : boxes) {
        if (!emitBox(std::move(box), region.elementBytes, space, pending)) {
            return false;
        }
    }
    mergeRanges(pending);
    result.assign(pending.begin(), pending.end());
    return true;
}
} // namespace mlir::pto::detail
