// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/IR/PTO.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
namespace mlir::pto {
namespace {
SyncMechanismFact fact(SyncMechanismAvailability value, StringRef obligation = {})
{
    return {value, "R2 NPU2201 AIC/AIV table; H0 pinned content", obligation};
}
bool classic(StringRef architecture)
{
    return architecture == "a2" || architecture == "a3" || architecture == "a2a3";
}
} // namespace
SyncPhysicalCore recoverSyncPhysicalCore(Operation* anchor)
{
    SyncPhysicalCore result = SyncPhysicalCore::Unknown;
    auto merge = [&](SyncPhysicalCore core) {
        result = result == SyncPhysicalCore::Unknown || result == core ? core : SyncPhysicalCore::Conflict;
    };
    for (Operation* op = anchor; op; op = op->getParentOp()) {
        if (isa<FunctionOpInterface, ModuleOp>(op)) {
            if (auto raw = op->getAttr(FunctionKernelKindAttr::name)) {
                if (auto kind = dyn_cast<FunctionKernelKindAttr>(raw)) {
                    merge(kind.getKernelKind() == FunctionKernelKind::Cube ?
                              SyncPhysicalCore::AIC : SyncPhysicalCore::AIV);
                } else {
                    result = SyncPhysicalCore::Conflict;
                }
            }
        }
        if (isa<SectionCubeOp>(op)) {
            merge(SyncPhysicalCore::AIC);
        } else if (isa<SectionVectorOp>(op)) {
            merge(SyncPhysicalCore::AIV);
        }
    }
    return result;
}
std::optional<bool> getSyncPipelinePresence(StringRef architecture, SyncPhysicalCore core, PIPE pipe)
{
    if (!classic(architecture) || (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV)) {
        return std::nullopt;
    }
    if (pipe == PIPE::PIPE_S || pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_MTE3) {
        return true;
    }
    if (pipe == PIPE::PIPE_M || pipe == PIPE::PIPE_MTE1 || pipe == PIPE::PIPE_FIX) {
        return core == SyncPhysicalCore::AIC;
    }
    if (pipe == PIPE::PIPE_V) {
        return core == SyncPhysicalCore::AIV;
    }
    return std::nullopt;
}
SyncMechanismFact getSyncBarrierFact(StringRef architecture, SyncPhysicalCore core, PIPE pipe)
{
    if (pipe == PIPE::PIPE_S) {
        return classic(architecture) ?
            fact(SyncMechanismAvailability::Nonexistent, "R2 excludes PIPE_S local PipeBarrier") :
            fact(SyncMechanismAvailability::RecipeUnmet, "PIPE_S local PipeBarrier has no qualified target recipe");
    }
    if (!classic(architecture)) {
        return fact(SyncMechanismAvailability::Unqualified, "target barrier contract not qualified");
    }
    if (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV) {
        return fact(SyncMechanismAvailability::Unqualified, "actual core context missing/conflicting");
    }
    auto exists = getSyncPipelinePresence(architecture, core, pipe);
    if (!exists) {
        return fact(SyncMechanismAvailability::Unqualified, "virtual/other pipe needs a separate recipe");
    }
    if (!*exists) {
        return fact(SyncMechanismAvailability::Nonexistent, "pipe absent from this physical core");
    }
    return fact(SyncMechanismAvailability::Documented);
}
SyncMechanismFact getSyncEventFact(StringRef architecture, SyncPhysicalCore core, PIPE source, PIPE target)
{
    if (source == PIPE::PIPE_ALL || target == PIPE::PIPE_ALL ||
        source == PIPE::PIPE_UNASSIGNED || target == PIPE::PIPE_UNASSIGNED) {
        return fact(SyncMechanismAvailability::RecipeUnmet, "flag endpoint is not a concrete directed pipe");
    }
    if (!classic(architecture) || (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV)) {
        return fact(SyncMechanismAvailability::Unqualified, "target/core event contract not qualified");
    }
    auto a = getSyncPipelinePresence(architecture, core, source);
    auto b = getSyncPipelinePresence(architecture, core, target);
    if (!a || !b) {
        return fact(SyncMechanismAvailability::Unqualified, "virtual/other pipe needs a separate event recipe");
    }
    if (!*a || !*b || source == target) {
        return fact(SyncMechanismAvailability::Nonexistent, "no directed event between these core-local pipes");
    }
    if (core == SyncPhysicalCore::AIV) {
        return fact(SyncMechanismAvailability::Documented);
    }
    if (source == PIPE::PIPE_S || target == PIPE::PIPE_S ||
        (source == PIPE::PIPE_M && target == PIPE::PIPE_MTE3) ||
        (source == PIPE::PIPE_MTE3 && target == PIPE::PIPE_M)) {
        return fact(SyncMechanismAvailability::Nonexistent, "R2 marks this AIC direction hardware-nonexistent");
    }
    return fact(SyncMechanismAvailability::Documented);
}
} // namespace mlir::pto

