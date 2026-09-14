// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCMATRIXCONTRACT_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCMATRIXCONTRACT_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/IR/Matchers.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::pto::structured_sync {
namespace matrix_contract_detail {
inline std::optional<int64_t> literal(Value value) {
    IntegerAttr a;
    if (!value || !matchPattern(value,m_Constant(&a)) || !a.getValue().isSignedIntN(64)) return {};
    if (value.getType().isInteger(1)) return int64_t(a.getValue().getZExtValue());
    return a.getValue().getSExtValue();
}

// S7's lowering witness is deliberately stricter than may-access extraction.
// An allocation's maximum shape alone does not establish mad's effective m/n/k.
// Only direct immutable descriptors with constant FULL valid dimensions enter
// this first rule. Unqualified geometry retains ordinary completion demands.
inline std::optional<std::pair<uint64_t,uint64_t>> exactMatrixValid(Value value) {
    auto type=dyn_cast<TileBufType>(value.getType());
    auto alloc=value.getDefiningOp<AllocTileOp>();
    if (!type || !alloc || type.getRank()!=2 || type.getCompactModeI32()!=0)
        return {};
    for (Operation *user:value.getUsers())
        if (isa<SetValidShapeOp>(user)) return {};
    auto shape=type.getShape(), valid=type.getValidShape();
    if (shape.size()!=2 || valid.size()!=2 || shape[0]<=0 || shape[1]<=0)
        return {};
    std::optional<int64_t> rows=valid[0]>=0?std::optional<int64_t>(valid[0]):literal(alloc.getValidRow());
    std::optional<int64_t> cols=valid[1]>=0?std::optional<int64_t>(valid[1]):literal(alloc.getValidCol());
    if (!rows || !cols || *rows!=shape[0] || *cols!=shape[1]) return {};
    // Explicit operands, when present, must agree as well; do not trust a
    // static type over a conflicting runtime descriptor argument.
    if ((alloc.getValidRow() && literal(alloc.getValidRow())!=rows) ||
        (alloc.getValidCol() && literal(alloc.getValidCol())!=cols)) return {};
    return std::make_pair(uint64_t(*rows),uint64_t(*cols));
}
inline std::optional<std::pair<uint64_t,uint64_t>> exactAccFootprint(
    Value value,const CompoundInstanceElement &phase,bool write) {
    const auto &entries=write?phase.defVec:phase.useVec;
    const BaseMemInfo *match=nullptr;
    for (auto *entry:entries) if (entry && entry->baseBuffer==value) {
        if (match) return {};
        match=entry;
    }
    if (!match || match->scope!=AddressSpace::ACC || !match->hasKnownPhysicalAddresses ||
        match->aliasesUnknownRange || match->baseAddresses.size()!=1 || !match->allocateSize || match->allocateSize>uint64_t(INT64_MAX) ||
        match->baseAddresses[0]>uint64_t(INT64_MAX)-match->allocateSize) return {};
    return std::make_pair(match->baseAddresses[0],match->allocateSize);
}
} // namespace matrix_contract_detail

// Extract the existing native lowering witness from a translated physical
// phase. Unknown retains ordinary synchronization requirements. A known result
// is only lowering data: callers still need the selected hardware contract and
// intrinsicAccumulatorOrder's storage, geometry and complete M-stream proof.
inline MmadInfo matrixLoweringFacts(const CompoundInstanceElement &phase) {
    using matrix_contract_detail::exactMatrixValid;
    using matrix_contract_detail::exactAccFootprint;
    MmadInfo result;
    Operation *op=phase.elementOp;
    Value a,b,dst,input;
    if (auto init=dyn_cast<TMatmulOp>(op)) {
        if (auto accPhase=init.getAccPhaseAttr())
            if (accPhase.getValue()!=AccPhase::Unspecified) return result;
        a=init.getLhs(); b=init.getRhs(); dst=init.getDst();
    } else if (auto update=dyn_cast<TMatmulAccOp>(op)) {
        if (auto accPhase=update.getAccPhaseAttr())
            if (accPhase.getValue()!=AccPhase::Unspecified) return result;
        a=update.getLhs(); b=update.getRhs(); dst=update.getDst(); input=update.getAccIn();
    } else return result; // no GEMV, bias, MX, partial implicit lowering or UnitFlag claim
    if (phase.kPipeValue!=static_cast<PipelineType>(PIPE::PIPE_M)) return result;
    auto at=dyn_cast<TileBufType>(a.getType()),bt=dyn_cast<TileBufType>(b.getType()),
         ct=dyn_cast<TileBufType>(dst.getType());
    if (!at || !bt || !ct || !ct.getElementType().isF32() ||
        at.getElementType()!=bt.getElementType() ||
        (!at.getElementType().isF16() && !at.getElementType().isBF16())) return result;
    auto space=[](TileBufType t,AddressSpace expected) {
        auto s=dyn_cast_or_null<AddressSpaceAttr>(t.getMemorySpace());
        return s && s.getAddressSpace()==expected;
    };
    if (!space(at,AddressSpace::LEFT) || !space(bt,AddressSpace::RIGHT) ||
        !space(ct,AddressSpace::ACC) ||
        at.getBLayoutValueI32()!=0 || at.getSLayoutValueI32()!=1 || at.getSFractalSizeI32()!=512 ||
        bt.getBLayoutValueI32()!=0 || bt.getSLayoutValueI32()!=2 || bt.getSFractalSizeI32()!=512 ||
        ct.getBLayoutValueI32()!=1 || ct.getSLayoutValueI32()!=1 || ct.getSFractalSizeI32()!=1024)
        return result;
    auto av=exactMatrixValid(a),bv=exactMatrixValid(b),cv=exactMatrixValid(dst);
    auto footprint=exactAccFootprint(dst,phase,true);
    if (!av || !bv || !cv || !footprint || av->second!=bv->first ||
        av->first!=cv->first || bv->second!=cv->second) return result;
    if (input && (input.getType()!=dst.getType() || exactMatrixValid(input)!=cv ||
                  exactAccFootprint(input,phase,false)!=footprint)) return result;
    // These plain PTO-ISA overloads derive m/k from LEFT valid shape and n from
    // RIGHT valid shape. No descriptor/source-independent m/n estimate is used.
    if (cv->first>4095 || cv->second>4095 || av->second>4095 ||
        footprint->second!=4*cv->first*cv->second) return result;
    result.kind=input?MmadInfo::Accumulate:MmadInfo::Initialize;
    result.accumulatorBase=footprint->first; result.accumulatorBytes=footprint->second;
    result.m=cv->first; result.n=cv->second; result.k=av->second;
    result.input=at.getElementType().isF16()?MmadInfo::F16:MmadInfo::BF16;
    return result;
}

} // namespace mlir::pto::structured_sync

#endif
