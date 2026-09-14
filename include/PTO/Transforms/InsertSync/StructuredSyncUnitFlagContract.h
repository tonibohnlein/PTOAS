// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCUNITFLAGCONTRACT_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCUNITFLAGCONTRACT_H

#include "PTO/Transforms/InsertSync/StructuredSyncMatrixContract.h"

namespace mlir::pto::structured_sync {
struct UnitFlagInfo {
    enum Kind : uint8_t { Unknown, ProducerFinal, StoreFinal } kind = Unknown;
    uint64_t base = 0, bytes = 0, rows = 0, cols = 0;
    explicit operator bool() const { return kind != Unknown; }
};

// Exact native lowering witness for the first paired update/update ownership
// contract.  This does not reuse MmadInfo: ownership and accumulator ordering
// are independently selected hardware facts.
inline UnitFlagInfo unitFlagLoweringFacts(const CompoundInstanceElement& phase)
{
    using matrix_contract_detail::exactAccFootprint;
    using matrix_contract_detail::exactMatrixValid;
    UnitFlagInfo result;
    if (auto producer = dyn_cast<TMatmulOp>(phase.elementOp)) {
        if (producer.getAccPhase() != AccPhase::Final ||
            phase.kPipeValue != static_cast<PipelineType>(PIPE::PIPE_M))
            return result;
        auto lhs = dyn_cast<TileBufType>(producer.getLhs().getType());
        auto rhs = dyn_cast<TileBufType>(producer.getRhs().getType());
        auto dst = dyn_cast<TileBufType>(producer.getDst().getType());
        auto space = [](TileBufType type, AddressSpace expected) {
            auto attr = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
            return attr && attr.getAddressSpace() == expected;
        };
        if (!lhs || !rhs || !dst || !dst.getElementType().isF32() ||
            lhs.getElementType() != rhs.getElementType() ||
            (!lhs.getElementType().isF16() && !lhs.getElementType().isBF16()) ||
            !space(lhs, AddressSpace::LEFT) || !space(rhs, AddressSpace::RIGHT) ||
            !space(dst, AddressSpace::ACC) || lhs.getBLayoutValueI32() != 0 ||
            lhs.getSLayoutValueI32() != 1 || lhs.getSFractalSizeI32() != 512 ||
            rhs.getBLayoutValueI32() != 0 || rhs.getSLayoutValueI32() != 2 ||
            rhs.getSFractalSizeI32() != 512 || dst.getBLayoutValueI32() != 1 ||
            dst.getSLayoutValueI32() != 1 || dst.getSFractalSizeI32() != 1024)
            return result;
        auto a = exactMatrixValid(producer.getLhs());
        auto b = exactMatrixValid(producer.getRhs());
        auto c = exactMatrixValid(producer.getDst());
        auto footprint = exactAccFootprint(producer.getDst(), phase, true);
        if (!a || !b || !c || !footprint || a->second != b->first || a->first != c->first ||
            b->second != c->second || !c->first || !c->second || !a->second || c->first > 4095 ||
            c->second > 4095 || a->second > 4095 || (c->first % 16) || (c->second % 16) ||
            (a->second % 16) || footprint->first % 1024 ||
            footprint->second != 4 * c->first * c->second)
            return result;
        result = {UnitFlagInfo::ProducerFinal, footprint->first, footprint->second, c->first, c->second};
        return result;
    }
    auto store = dyn_cast<TStoreOp>(phase.elementOp);
    if (!store || store.getStPhase() != STPhase::Final ||
        phase.kPipeValue != static_cast<PipelineType>(PIPE::PIPE_FIX) || store.getFp() ||
        store.getPreQuantScalar() || store.getAtomicType() != AtomicType::AtomicNone ||
        store.getReluPreMode() != ReluPreMode::NoRelu)
        return result;
    auto src = dyn_cast<TileBufType>(store.getSrc().getType());
    auto dst = dyn_cast<PartitionTensorViewType>(store.getDst().getType());
    auto srcSpace = src ? dyn_cast_or_null<AddressSpaceAttr>(src.getMemorySpace()) : AddressSpaceAttr{};
    auto valid = exactMatrixValid(store.getSrc());
    auto footprint = exactAccFootprint(store.getSrc(), phase, false);
    if (!src || !dst || !srcSpace || srcSpace.getAddressSpace() != AddressSpace::ACC ||
        !src.getElementType().isF32() || !dst.getElementType().isF32() || dst.getRank() != 2 ||
        !valid || !footprint || dst.getShape()[0] != int64_t(valid->first) ||
        dst.getShape()[1] != int64_t(valid->second) || footprint->first % 1024 ||
        footprint->second != 4 * valid->first * valid->second)
        return result;
    auto partition = store.getDst().getDefiningOp<PartitionViewOp>();
    if (!partition || partition.getOffsets().size() != 2 || partition.getSizes().size() != 2)
        return result;
    for (unsigned i = 0; i < 2; ++i) {
        auto offset = matrix_contract_detail::literal(partition.getOffsets()[i]);
        auto size = matrix_contract_detail::literal(partition.getSizes()[i]);
        if (!offset || *offset != 0 || !size || *size != dst.getShape()[i])
            return result;
    }
    auto view = partition.getSource().getDefiningOp<MakeTensorViewOp>();
    if (!view || view.getStrides().size() != 2 ||
        matrix_contract_detail::literal(view.getStrides()[0]) != std::optional<int64_t>(dst.getShape()[1]) ||
        matrix_contract_detail::literal(view.getStrides()[1]) != std::optional<int64_t>(1))
        return result;
    result = {UnitFlagInfo::StoreFinal, footprint->first, footprint->second, valid->first, valid->second};
    return result;
}
} // namespace mlir::pto::structured_sync
#endif
