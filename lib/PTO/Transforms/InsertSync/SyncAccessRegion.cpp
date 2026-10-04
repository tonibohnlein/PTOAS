// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Recover buffer descriptor geometry from types, addresses and view operations.
// A descriptor is not a claim about an instruction's actual read/write region.
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "SyncRegionArithmetic.h"
#include "SyncScalarEvolution.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::pto {
namespace {
class RegionBuilder {
public:
    RegionBuilder(const SyncInput& input, Operation* access)
        : input(input), context(access->getContext()), access(access), scalars(context) {}
    SyncAccessRegion region;
    AffineExpr number(int64_t value) { return getAffineConstantExpr(value, context); }
    AffineExpr value(Value v)
    {
        return scalars.value(v, [&](Value symbol) {
            auto entry = symbolIds.try_emplace(symbol, region.symbols.size());
            if (entry.second) {
                region.symbols.push_back(symbol);
            }
            return getAffineSymbolExpr(entry.first->second, context);
        });
    }

    AffineExpr add(AffineExpr a, AffineExpr b) { return detail::checkedAdd(a, b); }
    AffineExpr mul(AffineExpr a, AffineExpr b) { return detail::checkedMul(a, b); }
    AffineExpr scale(AffineExpr a, int64_t factor) { return mul(a, number(factor)); }
    bool extents(Value operand);
    bool selection(Value operand, DictionaryAttr contract);
    AffineExpr map(Value operand, ArrayRef<AffineExpr> coordinates);

private:
    const SyncInput& input;
    MLIRContext* context;
    Operation* access;
    DenseMap<Value, TileBufType> boxedDescriptors;
    detail::ScalarEvolution scalars;
    DenseMap<Value, unsigned> symbolIds;
    AffineExpr pointer(Value operand);
    AffineExpr tile(Value operand, TileBufType type, ArrayRef<AffineExpr> coordinates);
    AffineExpr tileBase(Value operand);
    TileBufType boxedPhysicalType(Value operand);
    bool boxedPointerStrides(TileBufType type, int64_t& row, int64_t& col);
    bool currentShape(Value operand, Value& row, Value& col);
};

unsigned bytes(Type type)
{
    if (!type.isIntOrFloat() || type.getIntOrFloatBitWidth() % 8 != 0) {
        return 0;
    }
    return type.getIntOrFloatBitWidth() / 8;
}

bool RegionBuilder::currentShape(Value operand, Value& row, Value& col)
{
    // Metadata is attached to the descriptor, not to its physical allocation.
    // Resolve updates in the access's block in execution order. A relevant
    // update behind structured control requires a reaching-metadata analysis;
    // it must never be replaced by the allocation's initial shape.
    bool unresolved = false;
    for (auto* user : operand.getUsers()) {
        if (auto update = dyn_cast<SetValidShapeOp>(user);
            update && update->getBlock() != access->getBlock()) {
            unresolved = true;
        }
    }
    if (unresolved) {
        return false;
    }
    bool precedingUpdate = false;
    for (auto& operation : *access->getBlock()) {
        if (&operation == access) {
            break;
        }
        if (auto update = dyn_cast<SetValidShapeOp>(operation);
            update && update.getSource() == operand) {
            precedingUpdate = true;
            row = update.getValidRow();
            col = update.getValidCol();
        }
    }
    // A later update can reach the same access on a loop backedge. Initial
    // metadata is usable only if the descriptor is recreated in this block or
    // a preceding update establishes its state on every iteration.
    auto* definition = operand.getDefiningOp();
    const bool recreated = definition && definition->getBlock() == access->getBlock() &&
                           definition->isBeforeInBlock(access);
    if (!precedingUpdate && !recreated) {
        bool repeated = false;
        for (auto* parent = access->getParentOp(); parent; parent = parent->getParentOp()) {
            repeated |= isa<LoopLikeOpInterface>(parent);
        }
        if (repeated) {
            for (auto* user : operand.getUsers()) {
                if (auto update = dyn_cast<SetValidShapeOp>(user);
                    update && update->getBlock() == access->getBlock()) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool RegionBuilder::extents(Value operand)
{
    if (auto type = dyn_cast<TileBufType>(operand.getType())) {
        auto shape = type.getValidShape();
        if (shape.size() != 2) {
            return false;
        }
        Value row, col;
        if (auto alloc = operand.getDefiningOp<AllocTileOp>()) {
            row = alloc.getValidRow();
            col = alloc.getValidCol();
        } else if (auto view = operand.getDefiningOp<SubViewOp>()) {
            row = view.getValidRow();
            col = view.getValidCol();
        } else if (auto get = operand.getDefiningOp<MultiTileGetOp>()) {
            if (auto alloc = get.getSource().getDefiningOp<AllocMultiTileOp>()) {
                row = alloc.getValidRow();
                col = alloc.getValidCol();
            }
        }
        if (type.hasDynamicValid() && !currentShape(operand, row, col)) {
            return false;
        }
        for (unsigned i = 0; i < 2; ++i) {
            auto extent = shape[i] >= 0 ? number(shape[i]) : value(i == 0 ? row : col);
            if (!extent) {
                return false;
            }
            region.extents.push_back(extent);
        }
        return true;
    }
    ValueRange sizes;
    if (auto view = operand.getDefiningOp<MakeTensorViewOp>()) {
        sizes = view.getShape();
    } else if (auto view = operand.getDefiningOp<PartitionViewOp>()) {
        sizes = view.getSizes();
    } else {
        return false;
    }
    for (auto size : sizes) {
        region.extents.push_back(value(size));
    }
    return true;
}

AffineExpr RegionBuilder::pointer(Value operand)
{
    if (auto add = operand.getDefiningOp<AddPtrOp>()) {
        auto base = pointer(add.getPtr());
        auto width = bytes(cast<PtrType>(add.getPtr().getType()).getElementType());
        return base && width ? this->add(base, scale(value(add.getOffset()), width)) : AffineExpr{};
    }
    if (auto castOp = operand.getDefiningOp<CastPtrOp>()) {
        if (isa<PtrType>(castOp.getInput().getType())) {
            return pointer(castOp.getInput());
        }
        // Integer addresses have no disjoint-root provenance. Keeping the
        // integer value as an absolute symbolic address still describes them.
        if (castOp.getInput().getType().isIntOrIndex()) {
            return value(castOp.getInput());
        }
        return {};
    }
    if (!isa<PtrType>(operand.getType())) {
        return {};
    }
    region.base = operand;
    return number(0);
}

AffineExpr RegionBuilder::tileBase(Value operand)
{
    if (auto view = operand.getDefiningOp<SubViewOp>()) {
        auto source = boxedPhysicalType(view.getSource());
        int64_t rowStride = 0, colStride = 0;
        auto base = tileBase(view.getSource());
        if (!source || !base || view.getOffsets().size() != 2 ||
            !boxedPointerStrides(source, rowStride, colStride)) {
            return {};
        }
        auto offset = add(scale(value(view.getOffsets()[0]), rowStride),
                          scale(value(view.getOffsets()[1]), colStride));
        return add(base, scale(offset, bytes(source.getElementType())));
    }
    if (auto get = operand.getDefiningOp<MultiTileGetOp>()) {
        auto found = input.buffers().find(get.getSource());
        if (found == input.buffers().end() || found->second.size() != 1) {
            return {};
        }
        const auto& memory = *found->second.front();
        if (!memory.hasKnownPhysicalAddresses || memory.baseAddresses.empty()) {
            return {};
        }
        const auto& addresses = memory.baseAddresses;
        auto slot = value(get.getSlot());
        if (auto selected = dyn_cast<AffineConstantExpr>(slot)) {
            if (selected.getValue() < 0 || static_cast<uint64_t>(selected.getValue()) >= addresses.size() ||
                addresses[selected.getValue()] > INT64_MAX) {
                return {};
            }
            return number(addresses[selected.getValue()]);
        }
        uint64_t stride = addresses.size() > 1 ? addresses[1] - addresses[0] : 0;
        for (auto [i, address] : llvm::enumerate(addresses)) {
            if (address > INT64_MAX || address < addresses[0] ||
                (i && (address - addresses[0]) / i != stride) ||
                (i && (address - addresses[0]) % i != 0)) {
                return {};
            }
        }
        if (stride > INT64_MAX) {
            return {};
        }
        return add(number(addresses[0]), scale(value(get.getSlot()), static_cast<int64_t>(stride)));
    }
    if (auto view = operand.getDefiningOp<BitcastOp>()) {
        return map(view.getSrc(), {number(0), number(0)});
    }
    if (auto view = operand.getDefiningOp<TReshapeOp>()) {
        return map(view.getSrc(), {number(0), number(0)});
    }
    if (auto alloc = operand.getDefiningOp<AllocTileOp>()) {
        return value(alloc.getAddr());
    }
    return {};
}

bool RegionBuilder::boxedPointerStrides(TileBufType type, int64_t& row, int64_t& col)
{
    const auto width = type ? bytes(type.getElementType()) : 0;
    if (!type || !width || type.getShape().size() != 2 || type.getShape()[0] <= 0 ||
        type.getShape()[1] <= 0 || type.getCompactModeI32() != static_cast<int>(CompactMode::Null)) {
        return false;
    }
    int64_t ir = 16, ic = 16;
    bool innerRow = type.getSLayoutValueI32() == static_cast<int>(SLayout::RowMajor);
    auto fractal = type.getSFractalSizeI32();
    if (fractal == 512 && 32 % width == 0) {
        ir = innerRow ? 16 : 32 / width;
        ic = innerRow ? 32 / width : 16;
    } else if (fractal == 32) {
        ic = 2;
    } else if (fractal != 1024) {
        return false;
    }
    if (type.getBLayoutValueI32() == static_cast<int>(BLayout::ColMajor)) {
        if (!innerRow) {
            return false;
        }
        row = ic;
        col = type.getShape()[0];
    } else {
        row = type.getShape()[1];
        col = ir;
    }
    return true;
}

TileBufType RegionBuilder::boxedPhysicalType(Value operand)
{
    auto cached = boxedDescriptors.find(operand);
    if (cached != boxedDescriptors.end()) {
        return cached->second;
    }
    auto recover = [&]() -> TileBufType {
        auto type = dyn_cast<TileBufType>(operand.getType());
        int64_t row = 0, col = 0;
        if (!type || type.getSLayoutValueI32() == static_cast<int>(SLayout::NoneBox) ||
            !boxedPointerStrides(type, row, col)) {
            return {};
        }
        if (auto view = operand.getDefiningOp<SubViewOp>()) {
            auto source = boxedPhysicalType(view.getSource());
            int64_t sourceRow = 0, sourceCol = 0;
            if (!source || !boxedPointerStrides(source, sourceRow, sourceCol)) {
                return {};
            }
            auto shape = row == sourceRow && col == sourceCol ? type.getShape() : source.getShape();
            return TileBufType::get(context, shape, type.getElementType(), type.getMemorySpace(),
                                    type.getValidShape(), type.getConfigAttr());
        }
        return type;
    };
    auto type = recover();
    // Cache unavailable descriptors as well; pointer recovery revisits the
    // same ancestry and must not repeat a full walk at each nested view.
    boxedDescriptors.try_emplace(operand, type);
    return type;
}

AffineExpr RegionBuilder::tile(Value operand, TileBufType type, ArrayRef<AffineExpr> coordinates)
{
    if (coordinates.size() != 2 || type.getShape().size() != 2) {
        return {};
    }
    auto shape = type.getShape();
    auto base = tileBase(operand);
    const auto width = bytes(type.getElementType());
    if (!base || !width || shape[0] <= 0 || shape[1] <= 0) {
        return {};
    }
    int64_t elements = 0, capacity = 0;
    if (llvm::MulOverflow(shape[0], shape[1], elements) ||
        llvm::MulOverflow(elements, static_cast<int64_t>(width), capacity)) {
        return {};
    }
    region.elementBytes = width;
    auto r = coordinates[0], c = coordinates[1];
    bool rowMajor = type.getBLayoutValueI32() == static_cast<int>(BLayout::RowMajor);
    if (type.getSLayoutValueI32() == static_cast<int>(SLayout::NoneBox)) {
        int64_t stride = rowMajor ? shape[1] : shape[0];
        AffineExpr physicalStride = number(stride);
        if (type.getCompactModeI32() == static_cast<int>(CompactMode::RowPlusOne)) {
            if (stride == INT64_MAX) {
                return {};
            }
            physicalStride = number(stride + 1);
        } else if (type.getCompactModeI32() == static_cast<int>(CompactMode::Normal)) {
            auto saved = region.extents.size();
            if (!extents(operand)) {
                return {};
            }
            physicalStride = region.extents[saved + (rowMajor ? 1 : 0)];
            region.extents.resize(saved);
        }
        return add(base, scale(rowMajor ? add(mul(r, physicalStride), c) : add(mul(c, physicalStride), r), width));
    }
    const bool compact = type.getCompactModeI32() != static_cast<int>(CompactMode::Null);
    if (compact) {
        return {};
    }
    int64_t ir = 16, ic = 16;
    auto fractal = type.getSFractalSizeI32();
    bool innerRow = type.getSLayoutValueI32() == static_cast<int>(SLayout::RowMajor);
    if (fractal == 512 && 32 % width == 0) {
        ir = innerRow ? 16 : 32 / width;
        ic = innerRow ? 32 / width : 16;
    } else if (fractal == 32) {
        ic = 2;
    } else if (fractal != 1024) {
        return {};
    }
    int64_t nr = shape[0] / ir + (shape[0] % ir != 0);
    int64_t nc = shape[1] / ic + (shape[1] % ic != 0);
    auto physicalRows = number(nr);
    auto physicalCols = number(nc);
    auto outer = rowMajor ? add(mul(r.floorDiv(ir), physicalCols), c.floorDiv(ic)) :
                            add(mul(c.floorDiv(ic), physicalRows), r.floorDiv(ir));
    auto inner = innerRow ? add(scale(r % ir, ic), c % ic) : add(scale(c % ic, ir), r % ir);
    return add(base, scale(add(scale(outer, ir * ic), inner), width));
}

AffineExpr RegionBuilder::map(Value operand, ArrayRef<AffineExpr> coordinates)
{
    if (auto view = operand.getDefiningOp<PartitionViewOp>()) {
        if (coordinates.size() != view.getOffsets().size()) {
            return {};
        }
        SmallVector<AffineExpr> shifted;
        for (auto [coordinate, offset] : llvm::zip(coordinates, view.getOffsets())) {
            auto shiftedCoordinate = add(coordinate, value(offset));
            if (!shiftedCoordinate) {
                return {};
            }
            shifted.push_back(shiftedCoordinate);
        }
        return map(view.getSource(), shifted);
    }
    if (auto view = operand.getDefiningOp<SubViewOp>()) {
        auto parentType = dyn_cast<TileBufType>(view.getSource().getType());
        if (!parentType) {
            return {};
        }
        if (parentType.getSLayoutValueI32() != static_cast<int>(SLayout::NoneBox)) {
            // ResolveBufferSelect adds linear pointer strides, then selects
            // the child shape only when its pointer strides are unchanged.
            auto physical = boxedPhysicalType(operand);
            return physical ? tile(operand, physical, coordinates) : AffineExpr{};
        }
        if (coordinates.size() != view.getOffsets().size()) {
            return {};
        }
        SmallVector<AffineExpr> shifted;
        for (auto [coordinate, offset] : llvm::zip(coordinates, view.getOffsets())) {
            auto shiftedCoordinate = add(coordinate, value(offset));
            if (!shiftedCoordinate) {
                return {};
            }
            shifted.push_back(shiftedCoordinate);
        }
        return map(view.getSource(), shifted);
    }
    if (auto view = operand.getDefiningOp<MakeTensorViewOp>()) {
        auto offset = pointer(view.getPtr());
        const auto width = bytes(cast<PtrType>(view.getPtr().getType()).getElementType());
        if (!offset || !width || coordinates.size() != view.getStrides().size()) {
            return {};
        }
        region.elementBytes = width;
        for (auto [coordinate, stride] : llvm::zip(coordinates, view.getStrides())) {
            offset = add(offset, scale(mul(coordinate, value(stride)), width));
            if (!offset) {
                return {};
            }
        }
        return offset;
    }
    if (auto type = dyn_cast<TileBufType>(operand.getType())) {
        return tile(operand, type, coordinates);
    }
    return {};
}

bool RegionBuilder::selection(Value operand, DictionaryAttr contract)
{
    auto version = contract.getAs<IntegerAttr>("pto.access_region");
    auto shapeIndex = contract.getAs<IntegerAttr>("shape_operand");
    auto capacity = contract.getAs<BoolAttr>("capacity");
    auto coordinates = contract.getAs<AffineMapAttr>("coordinates");
    auto indices = contract.getAs<DenseI64ArrayAttr>("symbol_operands");
    if (contract.size() != 5 || !version || !version.getType().isInteger(64) || version.getInt() != 1 ||
        !shapeIndex || !shapeIndex.getType().isInteger(64) ||
        !capacity || !coordinates || !indices || shapeIndex.getInt() < 0 ||
        static_cast<uint64_t>(shapeIndex.getInt()) >= access->getNumOperands()) {
        return false;
    }
    auto shape = access->getOperand(shapeIndex.getInt());
    if (capacity.getValue()) {
        auto type = dyn_cast<TileBufType>(shape.getType());
        if (!type || llvm::any_of(type.getShape(), [](int64_t size) { return size < 0; })) {
            return false;
        }
        for (auto size : type.getShape()) {
            region.extents.push_back(number(size));
        }
    } else if (!extents(shape)) {
        return false;
    }
    auto transform = coordinates.getValue();
    if (transform.getNumDims() != region.extents.size() || transform.getNumSymbols() != indices.size()) {
        return false;
    }
    SmallVector<AffineExpr> dimensions, symbols, selected;
    for (unsigned i = 0; i < region.extents.size(); ++i) {
        dimensions.push_back(getAffineDimExpr(i, context));
    }
    for (int64_t index : indices.asArrayRef()) {
        if (index < 0 || static_cast<uint64_t>(index) >= access->getNumOperands()) {
            return false;
        }
        Value symbol = access->getOperand(index);
        if (!symbol.getType().isIntOrIndex()) {
            return false;
        }
        symbols.push_back(value(symbol));
    }
    for (auto expression : transform.getResults()) {
        auto coordinate = detail::substitute(expression, dimensions, symbols);
        if (!coordinate) {
            return false;
        }
        selected.push_back(coordinate);
    }
    region.byteOffset = map(operand, selected);
    return static_cast<bool>(region.byteOffset);
}

// Failed scalar expansions may have visited intermediate SSA values. Export
// only symbols used by the final address or extents, keeping maps compact.
void compactSymbols(SyncAccessRegion& region)
{
    SmallVector<bool> used(region.symbols.size(), false);
    auto mark = [&](AffineExpr expression) {
        expression.walk([&](AffineExpr part) {
            if (auto symbol = dyn_cast<AffineSymbolExpr>(part)) {
                used[symbol.getPosition()] = true;
            }
        });
    };
    mark(region.byteOffset);
    for (auto extent : region.extents) {
        mark(extent);
    }
    SmallVector<Value> symbols;
    SmallVector<AffineExpr> replacement;
    for (auto [i, original] : llvm::enumerate(region.symbols)) {
        replacement.push_back(used[i] ? getAffineSymbolExpr(symbols.size(), region.byteOffset.getContext()) :
                                       getAffineConstantExpr(0, region.byteOffset.getContext()));
        if (used[i]) {
            symbols.push_back(original);
        }
    }
    region.byteOffset = region.byteOffset.replaceSymbols(replacement);
    for (auto& extent : region.extents) {
        extent = extent.replaceSymbols(replacement);
    }
    region.symbols = std::move(symbols);
}

} // namespace

std::optional<SyncAccessRegion> resolveSelectedRegion(const SyncInput& input, Value operand,
                                                     Operation* at, DictionaryAttr contract)
{
    if (!operand || !at || !contract) {
        return std::nullopt;
    }
    RegionBuilder builder(input, at);
    for (auto* parent = at->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            builder.region.iterations.push_back({loop.getInductionVar(), loop.getLowerBound(),
                                                 loop.getUpperBound(), loop.getStep()});
        }
    }
    std::reverse(builder.region.iterations.begin(), builder.region.iterations.end());
    if (!builder.selection(operand, contract)) {
        return std::nullopt;
    }
    compactSymbols(builder.region);
    return std::move(builder.region);
}

std::optional<SyncAccessRegion> resolveBufferRegion(const SyncInput& input, Value operand, Operation* at)
{
    if (!operand || !at) {
        return std::nullopt;
    }
    RegionBuilder builder(input, at);
    for (auto* parent = at->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            builder.region.iterations.push_back({loop.getInductionVar(), loop.getLowerBound(),
                                                 loop.getUpperBound(), loop.getStep()});
        }
    }
    std::reverse(builder.region.iterations.begin(), builder.region.iterations.end());
    if (!builder.extents(operand)) {
        return std::nullopt;
    }
    SmallVector<AffineExpr> coordinates;
    for (unsigned i = 0; i < builder.region.extents.size(); ++i) {
        coordinates.push_back(getAffineDimExpr(i, at->getContext()));
    }
    builder.region.byteOffset = builder.map(operand, coordinates);
    if (!builder.region.byteOffset) {
        return std::nullopt;
    }
    compactSymbols(builder.region);
    return std::move(builder.region);
}

bool SyncAccessRegion::empty() const
{
    return llvm::any_of(extents, [](AffineExpr extent) {
        auto constant = dyn_cast<AffineConstantExpr>(extent);
        return constant && constant.getValue() == 0;
    });
}
} // namespace mlir::pto
