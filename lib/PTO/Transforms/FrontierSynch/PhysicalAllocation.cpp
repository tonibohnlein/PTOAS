// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preflight the whole logical plan, then lower IDs without moving commands.
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
struct Endpoint {
    Operation* operation = nullptr;
    const PhysicalRecordAllocation* allocation = nullptr;
    bool publish = false;
};
FailureOr<SmallVector<Endpoint>> preflight(func::FuncOp function, const PhysicalAllocationPlan& plan)
{
    llvm::DenseMap<int64_t, const PhysicalRecordAllocation*> records;
    llvm::DenseMap<int64_t, std::pair<unsigned, unsigned>> counts;
    for (const auto& record : plan.records) {
        records[record.record] = &record;
    }
    SmallVector<Endpoint> endpoints;
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        const bool publish = isa<LogicalSetOp>(op), consume = isa<LogicalWaitOp>(op);
        if (!publish && !consume) {
            if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(op)) {
                op->emitError("physical allocation cannot share its supplied IDs with existing synchronization");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        }
        auto id = op->getAttrOfType<IntegerAttr>("plan_id");
        auto record = op->getAttrOfType<IntegerAttr>("record_id");
        auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
        const auto* allocation = record ? records.lookup(record.getInt()) : nullptr;
        if (!id || id.getInt() != plan.planId || !allocation || !source || !target ||
            static_cast<uint32_t>(source.getPipe()) != allocation->sourcePipe ||
            static_cast<uint32_t>(target.getPipe()) != allocation->targetPipe ||
            op->getNumOperands() != 1 || !op->getOperand(0).getType().isIndex()) {
            op->emitError("logical endpoint does not match its cyclic allocation certificate");
            return WalkResult::interrupt();
        }
        auto& count = counts[allocation->record];
        unsigned& occurrences = publish ? count.first : count.second;
        if (++occurrences != 1) {
            op->emitError("cyclic allocation expects one static SET and WAIT per record");
            return WalkResult::interrupt();
        }
        endpoints.push_back({op, allocation, publish});
        return WalkResult::advance();
    });
    if (walked.wasInterrupted()) {
        return failure();
    }
    for (const auto& record : plan.records) {
        if (counts.lookup(record.record) != std::make_pair(1U, 1U)) {
            return function.emitError("cyclic allocation record has missing logical endpoints"), failure();
        }
    }
    return endpoints;
}
Value physicalId(OpBuilder& builder, Location location, Value ordinal,
                 const PhysicalRecordAllocation& record, Value phaseOffset)
{
    auto constant = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value));
    };
    const uint64_t capacity = record.ids.size();
    Value modulus = constant(capacity);
    // Reduce BEFORE multiplication: native capacities are at most eight, so
    // intermediates are bounded by 56 even for a full-width source ordinal.
    Value phase = builder.create<arith::RemUIOp>(location, ordinal, modulus);
    phase = builder.create<arith::MulIOp>(location, phase, constant(record.stride % capacity));
    auto offset = phaseOffset ? phaseOffset : constant(record.phase % capacity);
    phase = builder.create<arith::AddIOp>(location, phase, offset);
    phase = builder.create<arith::RemUIOp>(location, phase, modulus);
    bool contiguous = true;
    for (std::size_t i = 0; i < record.ids.size(); ++i) {
        contiguous &= record.ids[i] == record.ids.front() + static_cast<int64_t>(i);
    }
    if (contiguous) {
        return builder.create<arith::AddIOp>(location, phase, constant(record.ids.front()));
    }
    Value id = constant(record.ids.front());
    for (std::size_t i = 1; i < record.ids.size(); ++i) {
        Value selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, phase, constant(i));
        id = builder.create<arith::SelectOp>(location, selected, constant(record.ids[i]), id);
    }
    return id;
}
} // namespace
void emitAllocatedCommand(OpBuilder& builder, Operation* op, const PhysicalRecordAllocation& record,
                          Value phaseOffset)
{
    const bool publish = isa<LogicalSetOp>(op);
    auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
    if (record.ids.size() == 1) {
        auto id = EventAttr::get(op->getContext(), *symbolizeEVENT(static_cast<uint32_t>(record.ids.front())));
        if (publish) {
            builder.create<SetFlagOp>(op->getLoc(), source, target, id);
        } else {
            builder.create<WaitFlagOp>(op->getLoc(), source, target, id);
        }
    } else {
        Value id = physicalId(builder, op->getLoc(), op->getOperand(0), record, phaseOffset);
        if (publish) {
            builder.create<SetFlagDynOp>(op->getLoc(), source, target, id);
        } else {
            builder.create<WaitFlagDynOp>(op->getLoc(), source, target, id);
        }
    }
}
LogicalResult allocatePhysicalEventIds(func::FuncOp function, ArrayRef<int64_t> eligibleIds)
{
    if (!function) {
        return failure();
    }
    auto plan = decodeCyclicAllocation(function, eligibleIds);
    if (failed(plan)) {
        return failure();
    }
    auto endpoints = preflight(function, *plan);
    if (failed(endpoints)) {
        return failure();
    }
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) {
        return function.emitError("physical allocation requires 64-bit occurrence indices");
    }
    auto compacted = compactAllocatedEndpoints(function, *plan);
    llvm::DenseSet<Operation*> erased(compacted.begin(), compacted.end());
    for (const auto& endpoint : *endpoints) {
        if (!erased.contains(endpoint.operation)) {
            OpBuilder builder(endpoint.operation);
            emitAllocatedCommand(builder, endpoint.operation, *endpoint.allocation);
            endpoint.operation->erase();
        }
    }
    function.walk([](Operation* op) { op->removeAttr("pto.endpoint_cut"); });
    function->removeAttr(CyclicAllocationAttr);
    return success();
}
} // namespace mlir::pto::frontiersynch
namespace mlir::pto {
#define GEN_PASS_DEF_PTOFRONTIERALLOCATE
#include "PTO/Transforms/Passes.h.inc"
namespace {
class PTOFrontierAllocatePass : public impl::PTOFrontierAllocateBase<PTOFrontierAllocatePass> {
public:
    using Base = impl::PTOFrontierAllocateBase<PTOFrontierAllocatePass>;
    PTOFrontierAllocatePass() = default;
    explicit PTOFrontierAllocatePass(const PTOFrontierAllocateOptions& options) : Base(options) {}
    void runOnOperation() override
    {
        if (failed(frontiersynch::allocatePhysicalEventIds(getOperation(), eligibleIds))) {
            signalPassFailure();
        }
    }
};
} // namespace
std::unique_ptr<Pass> createPTOFrontierAllocatePass()
{
    return std::make_unique<PTOFrontierAllocatePass>();
}
std::unique_ptr<Pass> createPTOFrontierAllocatePass(const PTOFrontierAllocateOptions& options)
{
    return std::make_unique<PTOFrontierAllocatePass>(options);
}
} // namespace mlir::pto
