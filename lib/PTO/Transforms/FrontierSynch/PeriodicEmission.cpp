// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Constant-offset endpoint preparation for certified fixed-body quotients.
#include "DirectEmissionInternal.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
unsigned periodicEndpointWidth(scf::ForOp loop)
{
    auto width = DataLayout::closest(loop).getTypeSizeInBits(loop.getInductionVar().getType()).getFixedValue();
    // Endpoint additions need two spare bits. Use a scalar width supported by
    // PTO's C++ lowering instead of an arbitrary width (e.g. i66 -> int64_t).
    for (unsigned supported : {8U, 16U, 32U, 64U, 128U}) {
        if (width <= supported - 2) { return supported; }
    }
    return IntegerType::kMaxWidth + 1;
}
LogicalResult emitPeriodicDemands(
    const TraceDemandAnalysis& trace, const SelectedAnalysis& selected, IRMapping& mapping,
    DirectEmissionResult& result)
{
    auto loop = selected.loop;
    auto iv = loop.getInductionVar();
    unsigned bits = periodicEndpointWidth(loop);
    if (bits > IntegerType::kMaxWidth) {
        result.reason = "periodic endpoint arithmetic is not representable";
        return failure();
    }
    auto arithmetic = IntegerType::get(loop.getContext(), bits);
    auto cast = [&](OpBuilder& builder, Location loc, Value value) -> Value {
        value = mapping.lookup(value);
        if (isa<IndexType>(value.getType())) {
            return builder.create<arith::IndexCastOp>(loc, arithmetic, value);
        }
        return builder.create<arith::ExtSIOp>(loc, arithmetic, value);
    };
    auto emit = [&](const PeriodicPrerequisite& edge, bool outgoing) {
        auto source = trace.sites()[selected.sites[edge.source]].phase->kPipeValue;
        auto target = trace.sites()[selected.sites[edge.consumer]].phase->kPipeValue;
        auto site = selected.sites[outgoing ? edge.source : edge.consumer];
        auto* anchor = mapping.lookup(trace.sites()[site].anchor);
        OpBuilder builder(anchor);
        if (outgoing) {
            builder.setInsertionPointAfter(anchor);
        }
        auto loc = anchor->getLoc();
        if (source == target) {
            // Local offsets need no partner coordinate or widened arithmetic.
            // Unit positive progression makes distance one equivalent to iv>lb.
            if (edge.distance == 0) {
                builder.create<pto::BarrierOp>(loc,
                    pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(target)));
            } else if (edge.distance == 1) {
                auto enabled = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                    mapping.lookup(iv), mapping.lookup(loop.getLowerBound()));
                auto guard = builder.create<scf::IfOp>(loc, enabled, false);
                guard.getThenBodyBuilder().create<pto::BarrierOp>(loc,
                    pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(target)));
            } else {
                return failure();
            }
            ++result.barriers;
            return success();
        }
        Value coordinate = cast(builder, loc, iv);
        Value enabled = builder.create<arith::ConstantIntOp>(loc, 1, 1);
        if (edge.distance != 0) {
            // This producer's shared-model certificate uses only offsets 0/1.
            // Larger numerical offsets need a proved arithmetic-width bound.
            if (edge.distance != 1) {
                return failure();
            }
            auto step = cast(builder, loc, loop.getStep());
            auto partner = outgoing ? Value(builder.create<arith::AddIOp>(loc, coordinate, step)) :
                                      Value(builder.create<arith::SubIOp>(loc, coordinate, step));
            auto bound = cast(builder, loc, outgoing ? loop.getUpperBound() : loop.getLowerBound());
            enabled = builder.create<arith::CmpIOp>(
                loc, outgoing ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sge, partner, bound);
            if (!outgoing) {
                coordinate = partner;
            }
        }
        auto guard = builder.create<scf::IfOp>(loc, enabled, false);
        auto body = guard.getThenBodyBuilder();
        if (source == target) {
            body.create<pto::BarrierOp>(loc, pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(target)));
            ++result.barriers;
            return success();
        }
        auto identity = body.create<arith::IndexCastOp>(loc, body.getIndexType(), coordinate);
        auto key =
            body.getI64IntegerAttr(selected.sites[edge.source] * trace.sites().size() + selected.sites[edge.consumer]);
        auto src = pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(source));
        auto dst = pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(target));
        if (outgoing) {
            body.create<pto::LogicalSetOp>(loc, src, dst, key, ValueRange{identity});
            ++result.sets;
        } else {
            body.create<pto::LogicalWaitOp>(loc, src, dst, key, ValueRange{identity});
            ++result.waits;
        }
        return success();
    };
    // Install local marks first, then incoming waits, then outgoing publications.
    for (bool local : {true, false}) {
        for (auto id : selected.periodic.retained()) {
            const auto& edge = selected.periodic.generators()[id].edge;
            bool same = trace.sites()[selected.sites[edge.source]].phase->kPipeValue ==
                        trace.sites()[selected.sites[edge.consumer]].phase->kPipeValue;
            if (same == local && failed(emit(edge, false))) {
                return failure();
            }
        }
    }
    for (auto id : selected.periodic.retained()) {
        const auto& edge = selected.periodic.generators()[id].edge;
        if (trace.sites()[selected.sites[edge.source]].phase->kPipeValue !=
                trace.sites()[selected.sites[edge.consumer]].phase->kPipeValue &&
            failed(emit(edge, true))) {
            return failure();
        }
    }
    result.privateSelectors = !selected.periodic.retained().empty();
    return success();
}
} // namespace mlir::pto::frontiersynch
