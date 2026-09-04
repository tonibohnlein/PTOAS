// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- ProtocolSyncTarget.cpp - Shared ProtocolSync target facts --------===//
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool moduleNamesA3Target(func::FuncOp function, std::string& reason)
{
    ModuleOp module = function->getParentOfType<ModuleOp>();
    if (!module) {
        reason = "ProtocolSync emission requires an enclosing module with an explicit A3 target";
        return false;
    }
    auto targetArch = module->getAttrOfType<StringAttr>("pto.target_arch");
    if (!targetArch) {
        reason = "ProtocolSync emission requires explicit pto.target_arch = 'a3'";
        return false;
    }
    const StringRef arch = targetArch.getValue();
    if (!arch.equals_insensitive("a3")) {
        reason = (llvm::Twine("target profile '") + arch + "' is not the explicit ProtocolSync A3 target").str();
        return false;
    }
    if (Attribute attribute = module->getAttr("pto.device-spec")) {
        auto deviceSpec = dyn_cast<StringAttr>(attribute);
        if (!deviceSpec) {
            reason = "pto.device-spec must be a string when ProtocolSync emission is requested";
            return false;
        }
        reason =
            (llvm::Twine("device profile '") + deviceSpec.getValue() + "' has no ProtocolSync A3 qualification record")
                .str();
        return false;
    }
    return true;
}

} // namespace

ProtocolSyncTarget ProtocolSyncTarget::resolve(func::FuncOp function)
{
    ProtocolSyncTarget target;
    if (!moduleNamesA3Target(function, target.unsupportedReason)) {
        return target;
    }

    target.kind = ProtocolSyncTargetKind::Npu2201A3;
    target.name = "npu2201-a3-protocol-sync-v1";
    target.unsupportedReason.clear();
    target.compilerEventIds = {0, 1, 2, 3, 4, 5};

    const auto resource = [](SyncPhysicalCore core, PIPE pipe) { return ProtocolSyncResource{core, pipe}; };
    const auto addEvent = [&](SyncPhysicalCore core, PIPE source, PIPE destination) {
        target.eventPairs.push_back({resource(core, source), resource(core, destination)});
    };

    for (PIPE pipe : {PIPE::PIPE_M, PIPE::PIPE_MTE1, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3, PIPE::PIPE_FIX}) {
        target.barrierResources.push_back(resource(SyncPhysicalCore::Cube, pipe));
    }
    for (PIPE pipe : {PIPE::PIPE_V, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3}) {
        target.barrierResources.push_back(resource(SyncPhysicalCore::Vector, pipe));
    }

    addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_M, PIPE::PIPE_MTE1);
    addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_M, PIPE::PIPE_MTE2);
    addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_M, PIPE::PIPE_FIX);
    for (PIPE destination : {PIPE::PIPE_M, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3}) {
        addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_MTE1, destination);
    }
    for (PIPE destination : {PIPE::PIPE_M, PIPE::PIPE_MTE1, PIPE::PIPE_MTE3, PIPE::PIPE_FIX}) {
        addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_MTE2, destination);
    }
    for (PIPE destination : {PIPE::PIPE_MTE1, PIPE::PIPE_MTE2}) {
        addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_MTE3, destination);
    }
    for (PIPE destination : {PIPE::PIPE_M, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3}) {
        addEvent(SyncPhysicalCore::Cube, PIPE::PIPE_FIX, destination);
    }

    constexpr PIPE vectorPipes[] = {PIPE::PIPE_S, PIPE::PIPE_V, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3};
    for (PIPE source : vectorPipes) {
        for (PIPE destination : vectorPipes) {
            if (source != destination) {
                addEvent(SyncPhysicalCore::Vector, source, destination);
            }
        }
    }
    return target;
}

bool ProtocolSyncTarget::supportsPipeBarrier(ProtocolSyncResource resource) const
{
    return isSupported() && llvm::is_contained(barrierResources, resource);
}

bool ProtocolSyncTarget::supportsEvent(ProtocolSyncResource source, ProtocolSyncResource target) const
{
    return isSupported() && source.core == target.core &&
           llvm::is_contained(eventPairs, std::make_pair(source, target));
}

bool ProtocolSyncTarget::supportsReadyRelease(SyncPhysicalCore core, PIPE producer, PIPE consumer) const
{
    return supportsEvent({core, producer}, {core, consumer}) && supportsEvent({core, consumer}, {core, producer});
}

StringRef mlir::pto::protocol_sync::stringifyProtocolSyncTargetKind(ProtocolSyncTargetKind kind)
{
    switch (kind) {
        case ProtocolSyncTargetKind::Npu2201A3:
            return "npu2201-a3";
        case ProtocolSyncTargetKind::Unsupported:
            return "unsupported";
    }
    return "unsupported";
}
