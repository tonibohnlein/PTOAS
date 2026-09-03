// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncSemanticMemory.cpp - ProtocolSync memory summaries -----------===//

#include "SyncSemanticsInternal.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

SyncVisibilityClass visibilityFor(AddressSpace space)
{
    if (space == AddressSpace::GM) {
        return SyncVisibilityClass::Global;
    }
    return space == AddressSpace::Zero ? SyncVisibilityClass::Unknown : SyncVisibilityClass::Local;
}

std::optional<SyncSlotExpression> getSlotExpression(Value value)
{
    Value selector = findMultiTileSlotExpr(value);
    if (!selector) {
        return std::nullopt;
    }
    return SyncSlotExpression{selector, getSyncSlotDepth(value).value_or(0)};
}

void recordSlotStatistics(const std::optional<SyncSlotExpression>& slot, ProtocolSyncStatistics* statistics)
{
    if (!statistics || !slot) {
        return;
    }
    if (slot->depth != 0) {
        ++statistics->slotExpressionsKnown;
    } else {
        ++statistics->slotExpressionsUnknown;
    }
}

} // namespace

std::optional<std::uint32_t> mlir::pto::protocol_sync::getSyncSlotDepth(Value value)
{
    Value current = value;
    for (unsigned hops = 0; current && hops < pto::kValue32; ++hops) {
        Operation* definition = current.getDefiningOp();
        if (!definition) {
            break;
        }
        if (auto get = dyn_cast<MultiTileGetOp>(definition)) {
            return get.getSource().getType().getCount();
        }
        if (auto subview = dyn_cast<SubViewOp>(definition)) {
            current = subview.getSource();
        } else if (auto reshape = dyn_cast<TReshapeOp>(definition)) {
            current = reshape.getSrc();
        } else if (auto bitcast = dyn_cast<BitcastOp>(definition)) {
            current = bitcast.getSrc();
        } else {
            break;
        }
    }
    return std::nullopt;
}

SyncPhysicalCore mlir::pto::protocol_sync::detail::resolveCore(Operation* operation, PIPE pipe)
{
    for (Operation* parent = operation; parent; parent = parent->getParentOp()) {
        if (isa<SectionCubeOp>(parent)) {
            return SyncPhysicalCore::Cube;
        }
        if (isa<SectionVectorOp>(parent)) {
            return SyncPhysicalCore::Vector;
        }
    }
    if (pipe == PIPE::PIPE_M || pipe == PIPE::PIPE_MTE1) {
        return SyncPhysicalCore::Cube;
    }
    if (pipe == PIPE::PIPE_FIX) {
        return SyncPhysicalCore::Cube;
    }
    if (pipe == PIPE::PIPE_V) {
        return SyncPhysicalCore::Vector;
    }
    if (pipe == PIPE::PIPE_S || pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_MTE3) {
        func::FuncOp function = operation->getParentOfType<func::FuncOp>();
        auto kind = function ? function->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name) :
                               FunctionKernelKindAttr();
        if (kind) {
            return kind.getKernelKind() == FunctionKernelKind::Cube ? SyncPhysicalCore::Cube : SyncPhysicalCore::Vector;
        }
    }
    return SyncPhysicalCore::Unknown;
}

void mlir::pto::protocol_sync::detail::setFailure(SyncOpSummary& summary, SyncFailureReason reason, StringRef message)
{
    if (summary.failure == SyncFailureReason::None) {
        summary.failure = reason;
        summary.failureDetail = message.str();
    }
}

void mlir::pto::protocol_sync::detail::setPhysicalResource(
    Operation* operation, PIPE pipe, SyncPhysicalPhase& phase, SyncOpSummary& summary)
{
    phase.pipe = pipe;
    phase.core = resolveCore(operation, pipe);
    if (phase.core == SyncPhysicalCore::Unknown) {
        setFailure(summary, SyncFailureReason::MissingPhysicalCore, "operation has no physical core assignment");
    }
}

void mlir::pto::protocol_sync::detail::appendValueEffects(
    SyncPhysicalPhase& phase, Value value, SyncAccessMode mode, const SyncSemanticContext& context,
    SyncOpSummary& summary, ProtocolSyncStatistics* statistics)
{
    ArrayRef<SyncStorageProvenance> storage = context.lookupStorage(value);
    std::optional<SyncSlotExpression> slot = getSlotExpression(value);
    if (storage.empty()) {
        SyncStorageProvenance unknown;
        unknown.value = value;
        unknown.root = value;
        unknown.aliasesUnknownRange = true;
        if (auto space = getPTOAddressSpaceAttr(value.getType())) {
            unknown.space = space.getAddressSpace();
        }
        const SyncVisibilityClass visibility = visibilityFor(unknown.space);
        recordSlotStatistics(slot, statistics);
        phase.effects.push_back({value, mode, std::move(unknown), visibility, slot});
        setFailure(summary, SyncFailureReason::MissingStorageProvenance, "memory effect has no storage provenance");
        return;
    }
    for (const SyncStorageProvenance& provenance : storage) {
        recordSlotStatistics(slot, statistics);
        phase.effects.push_back({value, mode, provenance, visibilityFor(provenance.space), slot});
    }
}

void mlir::pto::protocol_sync::detail::addPipelinePhase(
    Operation* operation, PIPE pipe, const SyncSemanticContext& context, SyncOpSummary& summary,
    ProtocolSyncStatistics* statistics)
{
    if (pipe == PIPE::PIPE_UNASSIGNED || pipe == PIPE::PIPE_ALL) {
        setFailure(summary, SyncFailureReason::MissingPipeline, "operation has no concrete physical pipeline");
        return;
    }
    SyncPhysicalPhase phase;
    setPhysicalResource(operation, pipe, phase, summary);
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface) {
        if (!isMemoryEffectFree(operation)) {
            setFailure(
                summary, SyncFailureReason::UnsupportedEffectfulOperation,
                "physical operation has no MemoryEffectOpInterface");
        }
        summary.phases.push_back(std::move(phase));
        return;
    }
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    interface.getEffects(effects);
    for (const MemoryEffects::EffectInstance& effect : effects) {
        Value value = effect.getValue();
        if (!value) {
            continue;
        }
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
            appendValueEffects(phase, value, SyncAccessMode::Read, context, summary, statistics);
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
            appendValueEffects(phase, value, SyncAccessMode::Write, context, summary, statistics);
        }
    }
    summary.phases.push_back(std::move(phase));
}
