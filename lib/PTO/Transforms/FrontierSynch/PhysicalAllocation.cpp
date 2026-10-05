// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preflight logical families, preserve certified member-to-phase maps, and lower
// commands in place. Allocation does not inspect or reconstruct endpoint guards.
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
#include <optional>
namespace mlir::pto::frontiersynch {
namespace {
struct Family {
    SmallVector<const PhysicalRecordAllocation*> members;
    std::optional<int64_t> sourceCut, targetCut;
};
using Records = llvm::DenseMap<int64_t, const PhysicalRecordAllocation*>;
using Families = std::map<int64_t, Family>;
struct Endpoint {
    Operation* operation = nullptr;
    const PhysicalRecordAllocation* allocation = nullptr;
    SmallVector<uint64_t> phases;
};
std::optional<int64_t> number(DictionaryAttr dictionary, StringRef name)
{
    auto attr = dictionary.getAs<IntegerAttr>(name);
    if (!attr || !attr.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return attr.getInt();
}
LogicalResult readFamily(func::FuncOp function, DictionaryAttr item, const Records& records,
                         Families& families, llvm::DenseSet<int64_t>& seen)
{
    auto id = number(item, "id"), source = number(item, "source_pipe"), target = number(item, "target_pipe");
    auto sourceCut = number(item, "source_cut"), targetCut = number(item, "target_cut");
    auto local = item.getAs<BoolAttr>("local");
    auto members = item.getAs<ArrayAttr>("members");
    if (!id || *id < 0 || !source || !target || !sourceCut || !targetCut || *targetCut < 0 ||
        !local || local.getValue() != (*source == *target) || (!local.getValue() && *sourceCut < 0) ||
        !members || members.empty() || families.count(*id)) {
        return function.emitError("malformed or duplicate endpoint-family allocation identity");
    }
    Family family;
    family.sourceCut = sourceCut;
    family.targetCut = targetCut;
    for (auto attr : members) {
        auto member = dyn_cast<DictionaryAttr>(attr);
        auto record = member ? number(member, "record") : std::nullopt;
        const auto* allocation = record ? records.lookup(*record) : nullptr;
        if (!record || *record < 0 || !seen.insert(*record).second ||
            (local.getValue() ? allocation != nullptr : allocation == nullptr)) {
            return function.emitError("endpoint-family member is missing, repeated, or absent from its allocation");
        }
        if (!allocation) {
            continue;
        }
        if (allocation->sourcePipe != *source || allocation->targetPipe != *target ||
            (!family.members.empty() && (allocation->stride != family.members.front()->stride ||
                                        allocation->ids != family.members.front()->ids))) {
            return function.emitError("endpoint-family members have incompatible cyclic allocations");
        }
        family.members.push_back(allocation);
    }
    // Keep local family identities occupied, but no notification can name one.
    families.emplace(*id, std::move(family));
    return success();
}
FailureOr<Families> readFamilies(func::FuncOp function, const PhysicalAllocationPlan& plan, const Records& records)
{
    Families result;
    auto attribute = function->getAttr("pto.endpoint_families");
    auto metadata = dyn_cast_or_null<DictionaryAttr>(attribute);
    if (attribute && !metadata) {
        return function.emitError("malformed endpoint-family metadata"), failure();
    }
    auto version = metadata ? number(metadata, "version") : std::optional<int64_t>(1);
    auto planId = metadata ? number(metadata, "plan") : std::optional<int64_t>(plan.planId);
    if (!version || (*version != 1 && *version != 2) || !planId || *planId != plan.planId) {
        return function.emitError("unsupported endpoint-family metadata version or plan identity"), failure();
    }
    if (*version == 1) {
        // Version 1 stored provenance for per-record logical commands; adapt
        // those commands as singleton families without parsing their guards.
        for (const auto& record : plan.records) {
            result.emplace(record.record, Family{{&record}, {}, {}});
        }
        return result;
    }
    auto families = metadata.getAs<ArrayAttr>("families");
    if (!families) {
        return function.emitError("endpoint-family metadata is missing families"), failure();
    }
    llvm::DenseSet<int64_t> seen;
    for (auto attr : families) {
        auto item = dyn_cast<DictionaryAttr>(attr);
        if (!item) {
            return function.emitError("malformed endpoint-family entry"), failure();
        }
        if (failed(readFamily(function, item, records, result, seen))) {
            return failure();
        }
    }
    for (const auto& record : plan.records) {
        if (!seen.contains(record.record)) {
            return function.emitError("allocation record is missing from the endpoint-family partition"), failure();
        }
    }
    return result;
}
bool validMember(Operation* op, const Family& family)
{
    if (op->getNumOperands() < 1 || op->getNumOperands() > 2 || !op->getOperand(0).getType().isIndex()) {
        return false;
    }
    if (op->getNumOperands() == 1) {
        return family.members.size() == 1;
    }
    auto member = op->getOperand(1);
    if (!member.getType().isIndex()) {
        return false;
    }
    APInt value;
    if (matchPattern(member, m_ConstantInt(&value))) {
        return !value.isNegative() && value.getActiveBits() <= 64 && value.getZExtValue() < family.members.size();
    }
    return family.sourceCut.has_value(); // Dynamic members require the grouped-family producer's certificate.
}
FailureOr<SmallVector<Endpoint>> preflight(func::FuncOp function, const PhysicalAllocationPlan& plan)
{
    Records records;
    llvm::DenseMap<int64_t, std::pair<unsigned, unsigned>> counts;
    for (const auto& record : plan.records) {
        records[record.record] = &record;
    }
    auto families = readFamilies(function, plan, records);
    if (failed(families)) {
        return failure();
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
        auto found = record ? families->find(record.getInt()) : families->end();
        if (!id || id.getInt() != plan.planId || found == families->end() || found->second.members.empty()) {
            op->emitError("logical endpoint does not match its cyclic allocation certificate");
            return WalkResult::interrupt();
        }
        const auto& family = found->second;
        const auto* allocation = family.members.front();
        if (!source || !target || static_cast<uint32_t>(source.getPipe()) != allocation->sourcePipe ||
            static_cast<uint32_t>(target.getPipe()) != allocation->targetPipe || !validMember(op, family)) {
            op->emitError("logical endpoint has an invalid family member or pipe assignment");
            return WalkResult::interrupt();
        }
        auto expectedCut = publish ? family.sourceCut : family.targetCut;
        auto cut = op->getParentOp()->getAttrOfType<IntegerAttr>("pto.endpoint_cut");
        if (expectedCut && (!cut || cut.getInt() != *expectedCut)) {
            op->emitError("logical endpoint does not match its certified family cut");
            return WalkResult::interrupt();
        }
        Endpoint endpoint{op, allocation, {}};
        for (const auto* member : family.members) {
            auto& count = counts[member->record];
            unsigned& occurrences = publish ? count.first : count.second;
            if (++occurrences != 1) {
                op->emitError("cyclic allocation expects one static SET and WAIT per family member");
                return WalkResult::interrupt();
            }
            endpoint.phases.push_back(member->phase % member->ids.size());
        }
        endpoints.push_back(std::move(endpoint));
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
Value memberPhase(OpBuilder& builder, const Endpoint& endpoint)
{
    auto location = endpoint.operation->getLoc();
    auto number = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value));
    };
    const auto& phases = endpoint.phases;
    if (phases.size() == 1) {
        return number(phases.front());
    }
    Value member = endpoint.operation->getOperand(1);
    const uint64_t capacity = endpoint.allocation->ids.size();
    const uint64_t coefficient = (phases[1] + capacity - phases[0]) % capacity;
    bool affine = true;
    for (std::size_t i = 0; i < phases.size(); ++i) {
        affine &= phases[i] == (phases[0] + (i % capacity) * coefficient) % capacity;
    }
    if (affine) {
        if (!coefficient) {
            return number(phases.front());
        }
        Value modulus = number(capacity);
        Value term = builder.create<arith::RemUIOp>(location, member, modulus);
        if (coefficient == 1 && !phases.front()) {
            return term;
        }
        if (coefficient != 1) {
            term = builder.create<arith::MulIOp>(location, term, number(coefficient));
        }
        if (phases.front()) {
            term = builder.create<arith::AddIOp>(location, number(phases.front()), term);
        }
        return builder.create<arith::RemUIOp>(location, term, modulus);
    }
    Value result = number(phases.front());
    // An irregular finite phase map remains one command with a shared decision
    // expression. It is never expanded back into independently guarded commands.
    for (std::size_t i = 1; i < phases.size(); ++i) {
        if (phases[i] != phases.front()) {
            auto selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, member, number(i));
            result = builder.create<arith::SelectOp>(location, selected, number(phases[i]), result);
        }
    }
    return result;
}
Value physicalId(OpBuilder& builder, Location location, Value ordinal,
                 const PhysicalRecordAllocation& record, Value phaseOffset)
{
    auto constant = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value));
    };
    const uint64_t capacity = record.ids.size();
    const uint64_t stride = record.stride % capacity;
    // memberPhase and the singleton constant are already in [0, capacity).
    // A zero cyclic stride needs no ordinal arithmetic or second remainder.
    Value phase = phaseOffset ? phaseOffset : constant(record.phase % capacity);
    if (stride) {
        Value modulus = constant(capacity);
        Value term = builder.create<arith::RemUIOp>(location, ordinal, modulus);
        if (stride != 1) {
            term = builder.create<arith::MulIOp>(location, term, constant(stride));
        }
        // Reduce before multiplying; native capacity bounds these intermediates.
        phase = builder.create<arith::AddIOp>(location, term, phase);
        phase = builder.create<arith::RemUIOp>(location, phase, modulus);
    }
    bool contiguous = true;
    for (std::size_t i = 0; i < record.ids.size(); ++i) {
        contiguous &= record.ids[i] == record.ids.front() + static_cast<int64_t>(i);
    }
    if (contiguous) {
        if (record.ids.front()) {
            return builder.create<arith::AddIOp>(location, phase, constant(record.ids.front()));
        }
        return phase;
    }
    Value id = constant(record.ids.front());
    for (std::size_t i = 1; i < record.ids.size(); ++i) {
        Value selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, phase, constant(i));
        id = builder.create<arith::SelectOp>(location, selected, constant(record.ids[i]), id);
    }
    return id;
}
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
} // namespace
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
    for (const auto& endpoint : *endpoints) {
        OpBuilder builder(endpoint.operation);
        Value phase;
        if (endpoint.allocation->ids.size() != 1) {
            phase = memberPhase(builder, endpoint);
        }
        emitAllocatedCommand(builder, endpoint.operation, *endpoint.allocation, phase);
        endpoint.operation->erase();
    }
    function.walk([](Operation* op) {
        op->removeAttr("pto.endpoint_cut");
        op->removeAttr("pto.family_loop");
    });
    function->removeAttr("pto.endpoint_families");
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
