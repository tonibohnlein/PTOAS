// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncSemanticProviders.cpp - ProtocolSync summary providers -------===//

#include "SyncSemanticsInternal.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

std::optional<PIPE> convertLegacyPipe(PipelineType pipe)
{
    switch (pipe) {
        case PipelineType::PIPE_S:
            return PIPE::PIPE_S;
        case PipelineType::PIPE_V:
            return PIPE::PIPE_V;
        case PipelineType::PIPE_M:
            return PIPE::PIPE_M;
        case PipelineType::PIPE_MTE1:
            return PIPE::PIPE_MTE1;
        case PipelineType::PIPE_MTE2:
            return PIPE::PIPE_MTE2;
        case PipelineType::PIPE_MTE3:
            return PIPE::PIPE_MTE3;
        case PipelineType::PIPE_FIX:
            return PIPE::PIPE_FIX;
        default:
            return std::nullopt;
    }
}

func::FuncOp lookupSyncHelper(func::CallOp call)
{
    ModuleOp module = call->getParentOfType<ModuleOp>();
    if (!module || call.getCallee().empty()) {
        return {};
    }
    func::FuncOp callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    const bool isHelper =
        callee && (callee->hasAttr("pto.tileop.helper") || callee->hasAttr("pto.ptodsl.subkernel_helper"));
    if (!isHelper) {
        return {};
    }
    return callee;
}

std::optional<PIPE> getSyncHelperPipe(func::FuncOp callee)
{
    if (auto role = callee->getAttrOfType<StringAttr>("pto.ptodsl.subkernel_helper")) {
        return llvm::StringSwitch<std::optional<PIPE>>(role.getValue())
            .Case("cube", PIPE::PIPE_M)
            .Case("simd", PIPE::PIPE_V)
            .Default(std::nullopt);
    }
    auto kind = callee->getAttrOfType<StringAttr>("pto.tileop.kind");
    return kind ? llvm::StringSwitch<std::optional<PIPE>>(kind.getValue())
                      .Case("cube", PIPE::PIPE_M)
                      .Case("vector", PIPE::PIPE_V)
                      .Default(std::nullopt) :
                  std::nullopt;
}

bool isSyncHelperMemoryOperand(Type type)
{
    return isa<PtrType, TileBufType, TensorViewType, PartitionTensorViewType>(type);
}

std::optional<std::uint32_t> getQueueDepth(Value handle)
{
    Operation* definition = handle ? handle.getDefiningOp() : nullptr;
    auto depth = definition ? definition->getAttrOfType<IntegerAttr>("slot_num") : IntegerAttr();
    const std::int64_t rawDepth = depth ? depth.getInt() : 0;
    const bool validDepth = rawDepth > 0 && rawDepth <= std::numeric_limits<std::uint32_t>::max();
    if (!validDepth) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(rawDepth);
}

void populateQueueResources(Value handle, SyncQueueSemantics& queue)
{
    Operation* definition = handle ? handle.getDefiningOp() : nullptr;
    if (auto init = dyn_cast_or_null<InitializeL2G2LPipeOp>(definition)) {
        queue.directionMask = init.getDirMask();
        queue.localSlotCount =
            init.getLocalSlotNumAttr() ?
                std::optional<std::uint32_t>(static_cast<std::uint32_t>(init.getLocalSlotNumAttr().getInt())) :
                std::nullopt;
        queue.flagBase = init.getFlagBaseAttr() ?
                             std::optional<std::uint32_t>(static_cast<std::uint32_t>(init.getFlagBaseAttr().getInt())) :
                             std::nullopt;
        queue.endpoint = init.getLocalAddr();
        queue.peerEndpoint = init.getPeerLocalAddr();
    } else if (auto init = dyn_cast_or_null<InitializeL2LPipeOp>(definition)) {
        queue.directionMask = init.getDirMask();
        queue.flagBase = init.getFlagBaseAttr() ?
                             std::optional<std::uint32_t>(static_cast<std::uint32_t>(init.getFlagBaseAttr().getInt())) :
                             std::nullopt;
        queue.endpoint = init.getLocalAddr();
        queue.peerEndpoint = init.getPeerLocalAddr();
    }
}

SyncQueueSemantics makeQueue(SyncQueueRole role, Value handle)
{
    SyncQueueSemantics queue{role, handle, getQueueDepth(handle)};
    populateQueueResources(handle, queue);
    return queue;
}

std::optional<SyncQueueSemantics> getQueueSemanticsImpl(Operation* operation)
{
    if (isa<InitializeL2G2LPipeOp, InitializeL2LPipeOp>(operation)) {
        Value handle = operation->getResult(0);
        return makeQueue(SyncQueueRole::Initialize, handle);
    }
    if (auto op = dyn_cast<TAllocOp>(operation)) {
        return makeQueue(SyncQueueRole::ProducerAcquire, op.getPipeHandle());
    }
    if (auto op = dyn_cast<TPushOp>(operation)) {
        return makeQueue(SyncQueueRole::ProducerPublish, op.getPipeHandle());
    }
    if (auto op = dyn_cast<TPopOp>(operation)) {
        return makeQueue(SyncQueueRole::ConsumerAcquire, op.getPipeHandle());
    }
    if (auto op = dyn_cast<TFreeOp>(operation)) {
        return makeQueue(SyncQueueRole::ConsumerRelease, op.getPipeHandle());
    }
    return std::nullopt;
}

StringRef getFixedProtocolKind(Operation* operation)
{
    if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(operation)) {
        return "explicit-event";
    }
    if (isa<GetBufOp, GetBufDynOp, RlsBufOp, RlsBufDynOp>(operation)) {
        return "explicit-buffer-id";
    }
    if (isa<SyncSetOp, SyncWaitOp, SetCrossBlockOp, WaitCrossBlockOp>(operation)) {
        return "explicit-cross-core";
    }
    if (isa<SetIntraBlockOp, WaitIntraBlockOp>(operation)) {
        return "explicit-intra-block";
    }
    if (isa<BarrierOp, BarrierSyncOp, FenceBarrierAllOp, TSyncOp, SyncAllOp>(operation)) {
        return "explicit-barrier";
    }
    if (isa<CmoCacheInvalidOp>(operation)) {
        return "explicit-visibility";
    }
    if (isa<WaitAsyncEventOp>(operation)) {
        return "explicit-async-completion";
    }
    return {};
}

bool isSyncStructuralOperation(Operation* operation)
{
    return isa<
        AllocTileOp, AllocMultiTileOp, DeclareTileOp, DeclareGlobalOp, MakeTensorViewOp, PartitionViewOp, AddPtrOp,
        PtrToIntOp, IntToPtrOp, CastPtrOp, SubViewOp, MultiTileGetOp, TReshapeOp, BitcastOp, DeclareEventIdArrayOp,
        EventIdArrayGetOp, EventIdArraySetOp, DeclareLocalArrayOp, LocalArrayGetOp, LocalArraySetOp, DeclareStructOp,
        StructGetOp, StructSetOp, arith::SelectOp>(operation);
}

void markProvider(SyncOpSummary& summary, SyncSummaryProvider provider, ProtocolSyncStatistics* statistics)
{
    summary.provider = provider;
    if (statistics) {
        ++statistics->providerHits;
        ++statistics->operationsSummarized;
    }
}

void addMacroSummary(
    Operation* operation, const SyncMacroModel& model, const SyncSemanticContext& context, SyncOpSummary& summary,
    ProtocolSyncStatistics* statistics)
{
    summary.completion.kind = SyncCompletionKind::MacroInternal;
    for (const SyncMacroPhase& macroPhase : model.phases) {
        std::optional<PIPE> pipe = convertLegacyPipe(macroPhase.pipe);
        if (!pipe) {
            protocol_sync::detail::setFailure(
                summary, SyncFailureReason::MissingPipeline, "macro phase has no supported physical pipeline");
            continue;
        }
        SyncPhysicalPhase phase;
        phase.phaseIndex = macroPhase.phaseId;
        protocol_sync::detail::setPhysicalResource(operation, *pipe, phase, summary);
        phase.completion.kind = SyncCompletionKind::MacroInternal;
        for (Value value : macroPhase.defValues) {
            protocol_sync::detail::appendValueEffects(
                phase, value, SyncAccessMode::Write, context, summary, statistics);
        }
        for (Value value : macroPhase.useValues) {
            protocol_sync::detail::appendValueEffects(phase, value, SyncAccessMode::Read, context, summary, statistics);
        }
        summary.phases.push_back(std::move(phase));
    }
    for (const SyncMacroHiddenEvent& event : model.hiddenEvents) {
        std::optional<PIPE> source = convertLegacyPipe(event.srcPipe);
        std::optional<PIPE> target = convertLegacyPipe(event.dstPipe);
        if (!source || !target) {
            protocol_sync::detail::setFailure(
                summary, SyncFailureReason::MissingPipeline, "hidden event has no supported physical pipeline");
            continue;
        }
        SyncEventReservation reservation;
        reservation.source = *source;
        reservation.target = *target;
        reservation.eventIds.assign(event.eventIds.begin(), event.eventIds.end());
        summary.eventReservations.push_back(std::move(reservation));
    }
    if (!summary.eventReservations.empty()) {
        summary.suppliedProtocols.push_back({"macro-internal-events"});
    }
}

void addHelperSummary(
    func::CallOp call, func::FuncOp callee, const SyncSemanticContext& context, SyncOpSummary& summary,
    ProtocolSyncStatistics* statistics)
{
    std::optional<PIPE> pipe = getSyncHelperPipe(callee);
    if (!pipe) {
        protocol_sync::detail::setFailure(
            summary, SyncFailureReason::MissingPipeline, "helper has no recognized physical role");
        return;
    }
    SyncPhysicalPhase phase;
    protocol_sync::detail::setPhysicalResource(call.getOperation(), *pipe, phase, summary);
    auto effects = callee->getAttrOfType<ArrayAttr>("pto.tileop.effects");
    bool precise = effects && effects.size() == call.getNumOperands();
    for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
        if (!isSyncHelperMemoryOperand(operand.getType())) {
            continue;
        }
        StringRef effect = "readwrite";
        if (precise) {
            if (auto value = dyn_cast<StringAttr>(effects[index])) {
                effect = value.getValue();
            }
        }
        if (effect == "read" || effect == "readwrite") {
            protocol_sync::detail::appendValueEffects(
                phase, operand, SyncAccessMode::Read, context, summary, statistics);
        }
        if (effect == "write" || effect == "readwrite") {
            protocol_sync::detail::appendValueEffects(
                phase, operand, SyncAccessMode::Write, context, summary, statistics);
        }
    }
    if (!phase.effects.empty()) {
        summary.phases.push_back(std::move(phase));
    }
}

} // namespace

std::optional<SyncQueueSemantics> mlir::pto::protocol_sync::getSyncQueueSemantics(Operation* operation)
{
    return getQueueSemanticsImpl(operation);
}

SyncOpSummary SyncSemanticExtractor::summarize(Operation* operation) const
{
    SyncOpSummary summary;
    summary.operation = operation;
    auto tryProvider = [&]() {
        if (statistics) {
            ++statistics->providerLookupAttempts;
        }
    };

    tryProvider();
    if (StringRef protocol = getFixedProtocolKind(operation); !protocol.empty()) {
        markProvider(summary, SyncSummaryProvider::FixedSynchronization, statistics);
        summary.suppliedProtocols.push_back({protocol.str()});
        if (statistics) {
            ++statistics->fixedSuppliedProtocols;
        }
        return summary;
    }
    tryProvider();
    if (auto queue = getSyncQueueSemantics(operation)) {
        markProvider(summary, SyncSummaryProvider::Queue, statistics);
        summary.queue = *queue;
        if (auto pipe = dyn_cast<OpPipeInterface>(operation)) {
            protocol_sync::detail::addPipelinePhase(operation, pipe.getPipe(), context, summary, statistics);
        }
        return summary;
    }
    tryProvider();
    if (auto model = getSyncMacroModel(operation)) {
        markProvider(summary, SyncSummaryProvider::Macro, statistics);
        addMacroSummary(operation, *model, context, summary, statistics);
        if (statistics) {
            statistics->hiddenEventReservations += summary.eventReservations.size();
        }
        return summary;
    }
    tryProvider();
    if (auto call = dyn_cast<func::CallOp>(operation)) {
        if (func::FuncOp callee = lookupSyncHelper(call)) {
            markProvider(summary, SyncSummaryProvider::Helper, statistics);
            addHelperSummary(call, callee, context, summary, statistics);
            return summary;
        }
    }
    tryProvider();
    if (auto pipe = dyn_cast<OpPipeInterface>(operation)) {
        markProvider(summary, SyncSummaryProvider::Pipeline, statistics);
        protocol_sync::detail::addPipelinePhase(operation, pipe.getPipe(), context, summary, statistics);
        return summary;
    }
    if (isa<LoadScalarOp, StoreScalarOp>(operation)) {
        markProvider(summary, SyncSummaryProvider::Pipeline, statistics);
        protocol_sync::detail::addPipelinePhase(operation, PIPE::PIPE_S, context, summary, statistics);
        return summary;
    }
    tryProvider();
    const bool isTerminator = operation->hasTrait<OpTrait::IsTerminator>();
    const bool isStructural = isSyncStructuralOperation(operation);
    const bool hasNoMemoryEffects = isMemoryEffectFree(operation);
    if (isTerminator || isStructural || hasNoMemoryEffects) {
        markProvider(summary, SyncSummaryProvider::Structural, statistics);
        return summary;
    }
    summary.failure = SyncFailureReason::UnsupportedEffectfulOperation;
    summary.failureDetail = "effectful operation has no synchronization summary";
    return summary;
}
