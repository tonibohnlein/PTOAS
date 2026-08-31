// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

std::optional<std::uint64_t> checkedShapeBytes(ArrayRef<int64_t> shape,
                                               Type elementType) {
  std::uint64_t bytes = getPTOStorageElemByteSize(elementType);
  if (bytes == 0) {
    return std::nullopt;
  }
  for (int64_t dimension : shape) {
    if (dimension < 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::uint64_t>::max() / bytes) {
      return std::nullopt;
    }
    bytes *= static_cast<std::uint64_t>(dimension);
  }
  return bytes;
}

std::optional<std::uint64_t> getTypeBytes(Type type) {
  if (auto tile = dyn_cast<TileBufType>(type)) {
    return checkedShapeBytes(tile.getShape(), tile.getElementType());
  }
  if (auto multi = dyn_cast<MultiTileBufType>(type)) {
    return getTypeBytes(multi.getSlotType());
  }
  if (auto view = dyn_cast<TensorViewType>(type)) {
    return checkedShapeBytes(view.getShape(), view.getElementType());
  }
  if (auto view = dyn_cast<PartitionTensorViewType>(type)) {
    return checkedShapeBytes(view.getShape(), view.getElementType());
  }
  if (auto memref = dyn_cast<MemRefType>(type)) {
    return checkedShapeBytes(memref.getShape(), memref.getElementType());
  }
  return std::nullopt;
}

std::optional<AddressSpace> getTypeSpace(Type type) {
  if (auto tile = dyn_cast<TileBufType>(type)) {
    if (auto space =
            dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace())) {
      return space.getAddressSpace();
    }
  }
  if (auto multi = dyn_cast<MultiTileBufType>(type)) {
    return getTypeSpace(multi.getSlotType());
  }
  if (auto space = getPTOAddressSpaceAttr(type)) {
    return space.getAddressSpace();
  }
  if (auto memref = dyn_cast<BaseMemRefType>(type)) {
    if (auto space =
            dyn_cast_or_null<AddressSpaceAttr>(memref.getMemorySpace())) {
      return space.getAddressSpace();
    }
    return std::nullopt;
  }
  if (isa<PtrType, TensorViewType, PartitionTensorViewType>(type)) {
    return AddressSpace::GM;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> getConstantUnsigned(Value value) {
  IntegerAttr attribute;
  if (!value) {
    return std::nullopt;
  }
  const bool isConstant = matchPattern(value, m_Constant(&attribute));
  if (!isConstant || attribute.getValue().isNegative()) {
    return std::nullopt;
  }
  return attribute.getValue().getZExtValue();
}

AliasFact makeRootFact(Value value) {
  AliasFact fact;
  fact.root = value;
  if (std::optional<AddressSpace> space = getTypeSpace(value.getType())) {
    fact.space = *space;
  } else {
    fact.unknownSpace = true;
  }
  if (std::optional<std::uint64_t> bytes = getTypeBytes(value.getType())) {
    fact.intervals.push_back({0, *bytes});
    fact.unknownRange = false;
  }
  return fact;
}

void offsetFacts(SmallVectorImpl<AliasFact> &facts, std::uint64_t offset) {
  for (AliasFact &fact : facts) {
    for (CanonicalByteInterval &interval : fact.intervals) {
      const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
      if (offset > maximum - interval.begin) {
        fact.intervals.clear();
        fact.unknownRange = true;
        break;
      }
      interval.begin += offset;
    }
  }
}

} // namespace

bool mlir::pto::canonical_sync_detail::areMemoryLikeTypes(Type type) {
  return isa<PtrType, TileBufType, MultiTileBufType, TensorViewType,
             PartitionTensorViewType, BaseMemRefType>(type);
}

AliasAnalysis::AliasAnalysis(func::FuncOp function) : function(function) {
  initializeArguments();
}

void AliasAnalysis::initializeArguments() {
  for (Value argument : function.getArguments()) {
    if (areMemoryLikeTypes(argument.getType())) {
      facts[argument].push_back(makeRootFact(argument));
    }
  }
}

ArrayRef<AliasFact> AliasAnalysis::lookup(Value value) const {
  auto found = facts.find(value);
  return found == facts.end() ? ArrayRef<AliasFact>()
                              : ArrayRef<AliasFact>(found->second);
}

SmallVector<AliasFact, 2> AliasAnalysis::describe(Value value) const {
  ArrayRef<AliasFact> known = lookup(value);
  if (!known.empty()) {
    return SmallVector<AliasFact, 2>(known.begin(), known.end());
  }
  if (value && areMemoryLikeTypes(value.getType())) {
    return {makeRootFact(value)};
  }
  return {};
}

void AliasAnalysis::bind(Value value, ArrayRef<AliasFact> aliases) {
  auto &destination = facts[value];
  destination.append(aliases.begin(), aliases.end());
}

void AliasAnalysis::bindAlias(Value result, Value source) {
  SmallVector<AliasFact, 2> aliases = describe(source);
  bind(result, aliases);
}

LogicalResult AliasAnalysis::bindAllocation(Operation *operation) {
  if (auto alloc = dyn_cast<AllocTileOp>(operation)) {
    AliasFact fact = makeRootFact(alloc.getResult());
    if (std::optional<std::uint64_t> address =
            getConstantUnsigned(alloc.getAddr())) {
      fact.intervals = {
          {*address, fact.intervals.empty() ? 0 : fact.intervals[0].size}};
      fact.unknownRange = fact.intervals[0].size == 0;
      fact.physical = !fact.unknownSpace && fact.space != AddressSpace::GM &&
                      fact.space != AddressSpace::Zero;
    } else if (!fact.unknownSpace && fact.space != AddressSpace::GM &&
               fact.space != AddressSpace::Zero) {
      // Distinct local SSA allocations are not disjoint after physical memory
      // planning. A dynamic or absent address therefore aliases every range in
      // its local address space until a physical interval is known.
      fact.intervals.clear();
      fact.unknownRange = true;
    }
    facts[alloc.getResult()].push_back(std::move(fact));
    return success();
  }
  auto multi = dyn_cast<AllocMultiTileOp>(operation);
  if (!multi) {
    return success();
  }
  AliasFact fact = makeRootFact(multi.getResult());
  const std::uint64_t slotBytes =
      fact.intervals.empty() ? 0 : fact.intervals[0].size;
  fact.intervals.clear();
  if (auto planned = operation->getAttrOfType<DenseI64ArrayAttr>(
          kPtoMultiBufferAddrsAttrName)) {
    for (int64_t address : planned.asArrayRef()) {
      if (address < 0) {
        return multi.emitError(
            "canonical sync requires nonnegative planned addresses");
      }
      fact.intervals.push_back(
          {static_cast<std::uint64_t>(address), slotBytes});
    }
    fact.physical = true;
  } else if (std::optional<std::uint64_t> base =
                 getConstantUnsigned(multi.getAddr())) {
    for (std::uint32_t slot = 0; slot < multi.getResult().getType().getCount();
         ++slot) {
      if (slotBytes != 0 &&
          slot >
              (std::numeric_limits<std::uint64_t>::max() - *base) / slotBytes) {
        return multi.emitError(
            "canonical sync multi-buffer address overflows uint64_t");
      }
      fact.intervals.push_back({*base + slot * slotBytes, slotBytes});
    }
    fact.physical = true;
  }
  fact.unknownRange = fact.intervals.empty() || slotBytes == 0;
  facts[multi.getResult()].push_back(std::move(fact));
  return success();
}

LogicalResult AliasAnalysis::observe(Operation *operation) {
  if (isa<AllocTileOp, AllocMultiTileOp>(operation)) {
    return bindAllocation(operation);
  }
  if (auto view = dyn_cast<MakeTensorViewOp>(operation)) {
    bindAlias(view.getResult(), view.getPtr());
  } else if (auto view = dyn_cast<PartitionViewOp>(operation)) {
    bindAlias(view.getResult(), view.getSource());
  } else if (auto view = dyn_cast<SubViewOp>(operation)) {
    bindAlias(view.getResult(), view.getSource());
  } else if (auto view = dyn_cast<MultiTileGetOp>(operation)) {
    SmallVector<AliasFact, 2> aliases = describe(view.getSource());
    IntegerAttr slotAttribute;
    std::optional<int64_t> constantSlot;
    if (matchPattern(view.getSlot(), m_Constant(&slotAttribute))) {
      constantSlot = slotAttribute.getInt();
    }
    for (AliasFact &fact : aliases) {
      fact.slotExpression = view.getSlot();
      if (constantSlot && *constantSlot >= 0 &&
          static_cast<std::uint64_t>(*constantSlot) < fact.intervals.size()) {
        const CanonicalByteInterval selected =
            fact.intervals[static_cast<size_t>(*constantSlot)];
        fact.intervals = {selected};
      }
    }
    bind(view.getResult(), aliases);
  } else if (auto add = dyn_cast<AddPtrOp>(operation)) {
    SmallVector<AliasFact, 2> aliases = describe(add.getPtr());
    std::optional<std::uint64_t> elements =
        getConstantUnsigned(add.getOffset());
    auto pointerType = dyn_cast<PtrType>(add.getPtr().getType());
    const std::uint64_t elementBytes =
        pointerType ? getPTOStorageElemByteSize(pointerType.getElementType())
                    : 0;
    if (!elements || elementBytes == 0 ||
        *elements > std::numeric_limits<std::uint64_t>::max() / elementBytes) {
      for (AliasFact &fact : aliases) {
        fact.intervals.clear();
        fact.unknownRange = true;
      }
    } else {
      offsetFacts(aliases, *elements * elementBytes);
    }
    bind(add.getResult(), aliases);
  } else if (auto cast = dyn_cast<CastPtrOp>(operation)) {
    bindAlias(cast.getResult(), cast.getInput());
  } else if (auto ptrToInt = dyn_cast<PtrToIntOp>(operation)) {
    bindAlias(ptrToInt.getResult(), ptrToInt.getPtr());
  } else if (auto intToPtr = dyn_cast<IntToPtrOp>(operation)) {
    if (auto ptrToInt = intToPtr.getAddr().getDefiningOp<PtrToIntOp>()) {
      bindAlias(intToPtr.getResult(), ptrToInt.getPtr());
    } else {
      AliasFact fact = makeRootFact(intToPtr.getResult());
      fact.intervals.clear();
      fact.unknownRange = true;
      facts[intToPtr.getResult()].push_back(std::move(fact));
    }
  } else if (auto reshape = dyn_cast<TReshapeOp>(operation)) {
    bindAlias(reshape.getResult(), reshape.getSrc());
  } else if (auto bitcast = dyn_cast<BitcastOp>(operation)) {
    bindAlias(bitcast.getResult(), bitcast.getSrc());
  } else if (auto select = dyn_cast<arith::SelectOp>(operation)) {
    bindAlias(select.getResult(), select.getTrueValue());
    bindAlias(select.getResult(), select.getFalseValue());
  }
  return success();
}
