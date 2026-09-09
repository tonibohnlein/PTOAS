// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/SyncGlobalOccurrences.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {
struct AffineSlice {
    uint64_t stride = 0, bias = 0, extent = 0;
    uint64_t safeUpper = 0;
};
inline std::optional<uint64_t> partitionLimit(
    uint64_t stride, uint64_t bias, uint64_t extent, uint64_t arithmeticMax = 2147483647)
{
    if (!stride || !extent || bias >= stride || extent > stride - bias || bias > arithmeticMax ||
        extent > arithmeticMax - bias) {
        return std::nullopt;
    }
    return (arithmeticMax - bias - extent) / stride + 1;
}
inline bool disjointOccurrencePartition(
    const AffineSlice& a, const AffineSlice& b, bool sameSite, bool exclusiveWithinIteration)
{
    if (!a.safeUpper || !b.safeUpper || a.stride != b.stride || !a.stride || a.bias >= a.stride || b.bias >= b.stride ||
        !a.extent || !b.extent || a.extent > a.stride - a.bias || b.extent > b.stride - b.bias) {
        return false;
    }
    bool disjointWithin = a.bias + a.extent <= b.bias || b.bias + b.extent <= a.bias;
    return disjointWithin || sameSite || exclusiveWithinIteration;
}

struct BoundGuard { Value upper; uint64_t limit; };
struct GlobalSlice { Value root; Operation *carrier = nullptr; AffineSlice affine; std::optional<BoundGuard> guard; bool known = false; };
struct OccurrenceProof { bool disjoint = false; SmallVector<BoundGuard> guards; };

constexpr uint64_t maxArithmetic = 2147483647; // Conservative for 32/64-bit index lowering.
std::optional<int64_t> literal(Value v)
{
    IntegerAttr a;
    if (
        !v || !matchPattern(v, m_Constant(&a)) || !a.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return a.getValue().getSExtValue();
}
struct Affine {
    uint64_t scale = 0, bias = 0, safeUpper = maxArithmetic;
};
std::optional<Affine> finish(Affine a)
{
    if (
        a.scale > maxArithmetic || a.bias > maxArithmetic) {
        return std::nullopt;
    }
    if (
        a.scale) {
        a.safeUpper = std::min(a.safeUpper, (maxArithmetic - a.bias) / a.scale + 1);
    }
    return a;
}
std::optional<Affine> plus(Affine a, Affine b)
{
    if (
        a.scale > maxArithmetic - b.scale || a.bias > maxArithmetic - b.bias) {
        return std::nullopt;
    }
    return finish({a.scale + b.scale, a.bias + b.bias, std::min(a.safeUpper, b.safeUpper)});
}
std::optional<Affine> times(Affine a, uint64_t b)
{
    if (
        b > maxArithmetic || (a.scale && b > maxArithmetic / a.scale) || (a.bias && b > maxArithmetic / a.bias)) {
        return std::nullopt;
    }
    // Keep restrictions of every intermediate, even multiplication by zero.
    return finish({a.scale * b, a.bias * b, a.safeUpper});
}
std::optional<Affine> affine(Value v, Value iv, unsigned depth = 0)
{
    if (
        !v || depth > 24 || !v.getType().isIndex()) {
        return std::nullopt;
    }
    if (
        v == iv && iv) {
        return Affine{1, 0, maxArithmetic};
    }
    if (
        auto c = literal(v)) {
        if (
            *c < 0 || uint64_t(*c) > maxArithmetic) {
            return std::nullopt;
        }
        return Affine{0, uint64_t(*c), maxArithmetic};
    }
    if (
        auto add = v.getDefiningOp<arith::AddIOp>()) {
        auto a = affine(add.getLhs(), iv, depth + 1), b = affine(add.getRhs(), iv, depth + 1);
        return a && b ? plus(*a, *b) : std::nullopt;
    }
    if (
        auto mul = v.getDefiningOp<arith::MulIOp>()) {
        auto a = affine(mul.getLhs(), iv, depth + 1), b = affine(mul.getRhs(), iv, depth + 1);
        if (
            !a || !b || (a->scale && b->scale)) {
            return std::nullopt;
        }
        auto result = b->scale ? times(*b, a->bias) : times(*a, b->bias);
        if (
            result) {
            result->safeUpper = std::min({result->safeUpper, a->safeUpper, b->safeUpper});
        }
        return result;
    }
    return std::nullopt;
}

struct GuardLiteral {
    Value expression;
    int64_t constant;
    bool equal;
};
std::optional<GuardLiteral> equality(Value condition, bool thenArm)
{
    auto compare = condition.getDefiningOp<arith::CmpIOp>();
    if (!compare ||
        (compare.getPredicate() != arith::CmpIPredicate::eq && compare.getPredicate() != arith::CmpIPredicate::ne)) {
        return std::nullopt;
    }
    bool equal = (compare.getPredicate() == arith::CmpIPredicate::eq) == thenArm;
    if (
        auto c = literal(compare.getRhs())) {
        return GuardLiteral{compare.getLhs(), *c, equal};
    }
    if (
        auto c = literal(compare.getLhs())) {
        return GuardLiteral{compare.getRhs(), *c, equal};
    }
    return std::nullopt;
}
bool exclusiveInIteration(Operation* a, Operation* b, Operation* carrier)
{
    SmallVector<std::pair<scf::IfOp, bool>, 4> ga, gb;
    auto collect = [&](Operation* op, auto& out) {
        Operation* child = op;
        for (Operation* parent = op->getParentOp(); parent && parent != carrier;
             child = parent, parent = parent->getParentOp()) {
            if (
                auto choice = dyn_cast<scf::IfOp>(parent)) {
                out.push_back({choice, child->getParentRegion() == &choice.getThenRegion()});
            }
        }
    };
    collect(a, ga);
    collect(b, gb);
    for (auto [ia, ta] : ga) {
        for (auto [ib, tb] : gb) {
            if (
                ia == ib && ta != tb) {
                return true;
            }
            if (
                ia.getCondition() == ib.getCondition() && ta != tb) {
                return true;
            }
            auto ca = equality(ia.getCondition(), ta), cb = equality(ib.getCondition(), tb);
            if (
                !ca || !cb || ca->expression != cb->expression) {
                continue;
            }
            if (
                ca->constant == cb->constant && ca->equal != cb->equal) {
                return true;
            }
            if (
                ca->equal && cb->equal && ca->constant != cb->constant) {
                return true;
            }
        }
    }
    return false;
}

GlobalSlice recoverGlobalSlice(
    Value value, Operation* access, func::FuncOp function)
{
    GlobalSlice result;
    auto partition = value.getDefiningOp<PartitionViewOp>();
    if (
        !partition) {
        return result;
    }
    auto view = partition.getSource().getDefiningOp<MakeTensorViewOp>();
    auto type = dyn_cast<PartitionTensorViewType>(value.getType());
    if (
        !view || !type || partition.getSizes().size() != 2 || partition.getOffsets().size() != 2 ||
        view.getStrides().size() != 2) {
        return result;
    }
    // Exact GM extent is currently qualified only for a plain full-tile UB
    // store to a contiguous row-major view. A view descriptor alone is not an
    // instruction footprint (padding/conversion/FIX modes remain conservative).
    if (
        !isa<TStoreOp>(access) || access->getNumOperands() != 2 ||
        (view.getLayoutAttr() && view.getLayoutAttr().getLayout() != Layout::ND)) {
        return result;
    }
    if (auto layout = access->getAttrOfType<LayoutAttr>("layout")) {
        if (layout.getLayout() != Layout::ND) return result;
    }
    Value payload = access->getOperand(0);
    auto payloadType = dyn_cast<TileBufType>(payload.getType());
    auto allocation = payload.getDefiningOp<AllocTileOp>();
    auto payloadSpace =
        payloadType ? dyn_cast_or_null<AddressSpaceAttr>(payloadType.getMemorySpace()) : AddressSpaceAttr();
    auto rows = literal(partition.getSizes()[0]), cols = literal(partition.getSizes()[1]);
    auto rowStride = literal(view.getStrides()[0]), colStride = literal(view.getStrides()[1]);
    if (
        !payloadType || !allocation || !payloadSpace || payloadSpace.getAddressSpace() != AddressSpace::VEC || !rows ||
        !cols || *rows <= 0 || *cols <= 0 || !rowStride || !colStride || *rowStride != *cols || *colStride != 1 ||
        payloadType.getShape().size() != 2 || payloadType.getValidShape().size() != 2 ||
        payloadType.getShape() != payloadType.getValidShape() || payloadType.getShape()[0] != *rows ||
        payloadType.getShape()[1] != *cols || payloadType.getElementType() != type.getElementType() ||
        payloadType.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor) ||
        payloadType.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox) ||
        payloadType.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne)) {
        return result;
    }
    if (
        !payloadType.getElementType().isF16() && !payloadType.getElementType().isF32() &&
        !payloadType.getElementType().isInteger(32)) {
        return result;
    }
    const uint64_t elementBytes = getPTOStorageElemByteSize(type.getElementType());
    if (
        !elementBytes || uint64_t(*cols) > maxArithmetic / elementBytes || (uint64_t(*cols) * elementBytes) % 32) {
        return result;
    }
    if (
        (allocation.getValidRow() && literal(allocation.getValidRow()) != rows) ||
        (allocation.getValidCol() && literal(allocation.getValidCol()) != cols)) {
        return result;
    }
    for (Operation* user : payload.getUsers()) {
        if (
            !isa<TLoadOp, TStoreOp, TAbsOp, TAddOp>(user)) {
            return result;
        }
    }
    result.root = view.getPtr();
    auto root = dyn_cast<BlockArgument>(result.root);
    if (
        !root || root.getOwner() != &function.getBody().front()) {
        return result;
    }
    scf::ForOp carrier;
    for (Operation* p = access->getParentOp(); p && p != function.getOperation(); p = p->getParentOp()) {
        if (
            auto loop = dyn_cast<scf::ForOp>(p)) {
            if (
                carrier) {
                return result; // No invented flattened ordinal across nested resets.
            }
            carrier = loop;
        } else if (!isa<scf::IfOp>(p)) {
            return result;
        }
    }
    Value iv;
    if (
        carrier) {
        auto lower = literal(carrier.getLowerBound()), step = literal(carrier.getStep());
        if (
            !lower || *lower < 0 || !step || *step != 1 || carrier->hasAttr("unsignedCmp") ||
            carrier->hasAttr("unsigned_cmp")) {
            return result;
        }
        iv = carrier.getInductionVar();
    }
    auto total = std::optional<Affine>(Affine{});
    uint64_t span = 1;
    for (unsigned i = 0; i < 2; ++i) {
        auto stride = literal(view.getStrides()[i]);
        auto extent = literal(partition.getSizes()[i]);
        auto offset = affine(partition.getOffsets()[i], iv);
        if (
            !stride || !extent || *stride <= 0 || *extent <= 0 || !offset || uint64_t(*stride) > maxArithmetic ||
            uint64_t(*extent) > maxArithmetic) {
            return result;
        }
        if (
            uint64_t(*extent - 1) > (maxArithmetic - span) / uint64_t(*stride)) {
            return result;
        }
        span += uint64_t(*extent - 1) * uint64_t(*stride);
        auto scaled = times(*offset, uint64_t(*stride));
        if (
            !scaled || !total) {
            return result;
        }
        total = plus(*total, *scaled);
    }
    uint64_t bytes = getPTOStorageElemByteSize(type.getElementType());
    if (
        !bytes || span > maxArithmetic / bytes || !total) {
        return result;
    }
    total = times(*total, bytes);
    if (
        !total) {
        return result;
    }
    result.affine = {total->scale, total->bias, span * bytes, total->safeUpper};
    if (
        !total->scale) {
        if (
            total->bias > maxArithmetic - span * bytes) {
            return result;
        }
        result.known = true;
        return result;
    }
    if (
        !carrier) {
        return result;
    }
    auto limit = partitionLimit(total->scale, total->bias, span * bytes, maxArithmetic);
    if (
        !limit) {
        return result;
    }
    uint64_t safe = std::min(*limit, total->safeUpper);
    if (
        auto upper = literal(carrier.getUpperBound())) {
        if (
            *upper > 0 && uint64_t(*upper) > safe) {
            return result;
        }
    } else {
        auto upperArg = dyn_cast<BlockArgument>(carrier.getUpperBound());
        if (
            !upperArg || upperArg.getOwner() != &function.getBody().front()) {
            return result;
        }
        result.guard = BoundGuard{upperArg, safe};
    }
    result.carrier = carrier.getOperation();
    result.affine.safeUpper = safe;
    result.known = true;
    return result;
}

OccurrenceProof compareOccurrences(
    const GlobalSlice& a, Operation* source, const GlobalSlice& b, Operation* target)
{
    OccurrenceProof result;
    if (
        !a.known || !b.known || a.root != b.root) {
        return result;
    }
    if (
        !a.affine.stride && !b.affine.stride) {
        result.disjoint =
            a.affine.bias + a.affine.extent <= b.affine.bias || b.affine.bias + b.affine.extent <= a.affine.bias;
        return result;
    }
    if (
        !a.carrier || a.carrier != b.carrier) {
        return result;
    }
    result.disjoint = disjointOccurrencePartition(
        a.affine, b.affine, source == target, exclusiveInIteration(source, target, a.carrier));
    if (
        result.disjoint) {
        if (
            a.guard) {
            result.guards.push_back(*a.guard);
        }
        if (
            b.guard) {
            result.guards.push_back(*b.guard);
        }
    }
    return result;
}

} // namespace

bool mlir::pto::disjointInsertSyncGlobalOccurrences(
    const BaseMemInfo *a, Operation *source, const BaseMemInfo *b,
    Operation *target, func::FuncOp function) {
  if (!a || !b || !source || !target || a->scope != AddressSpace::GM || b->scope != AddressSpace::GM)
    return false;
  auto first = recoverGlobalSlice(a->baseBuffer, source, function);
  auto second = recoverGlobalSlice(b->baseBuffer, target, function);
  auto proof = compareOccurrences(first, source, second, target);
  return proof.disjoint && proof.guards.empty();
}
