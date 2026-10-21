// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Qualified standalone native prefix order. Storage-pair protection alone does not
// imply this stronger event relation; the supplied scalar serialization contract
// is scoped below to synchronous, noncompound vector-buffer accesses and scalar GM pointer loads.
#ifndef PTO_FRONTIERSYNCH_PREFIXNATIVEORDER_H
#define PTO_FRONTIERSYNCH_PREFIXNATIVEORDER_H
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
namespace mlir::pto::frontiersynch::prefix {
enum class NativeOrder { Unknown, Overlapping, Serialized };
inline NativeOrder nativeOrder(
    const mlir::pto::CompoundInstanceElement& phase, llvm::StringRef architecture, mlir::pto::SyncPhysicalCore core)
{
    using namespace mlir;
    namespace pto = mlir::pto;
    auto pipe = static_cast<pto::PIPE>(phase.kPipeValue);
    auto present = pto::getSyncPipelinePresence(architecture, core, pipe);
    if (!present || !*present) {
        return NativeOrder::Unknown;
    }
    if (pipe != pto::PIPE::PIPE_S) {
        return NativeOrder::Overlapping;
    }
    if (architecture != "a3" || core != pto::SyncPhysicalCore::AIV || phase.macroOpInstanceId >= 0 ||
        !phase.elementOp) {
        return NativeOrder::Unknown;
    }
    Value memory;
    if (auto get = dyn_cast<pto::TGetValOp>(phase.elementOp)) {
        if (pto::resultAvailability(phase, get.getDst()) != pto::SyncResultAvailability::SynchronousScalar) {
            return NativeOrder::Unknown;
        }
        memory = get.getSrc();
    } else if (auto set = dyn_cast<pto::TSetValOp>(phase.elementOp)) {
        memory = set.getDst();
    } else if (auto load = dyn_cast<pto::PTOLoadOp>(phase.elementOp)) {
        auto pointer = dyn_cast<pto::PtrType>(load.getPtr().getType());
        if (!pointer || pointer.getMemorySpace().getAddressSpace() != pto::AddressSpace::GM ||
            pto::resultAvailability(phase, load.getValue()) != pto::SyncResultAvailability::SynchronousScalar) {
            return NativeOrder::Unknown;
        }
        // Existing scalar-result availability states completion of this access
        // before its result can be used. The supplied S serialization premise
        // applies across mixed admitted scalar phases, not just equal opcodes.
        return NativeOrder::Serialized;
    } else {
        return NativeOrder::Unknown;
    }
    auto tile = dyn_cast<pto::TileBufType>(memory.getType());
    auto space = tile ? dyn_cast_or_null<pto::AddressSpaceAttr>(tile.getMemorySpace()) : pto::AddressSpaceAttr{};
    const auto scalar = static_cast<uint32_t>(pto::PIPE::PIPE_S);
    if (!space || space.getAddressSpace() != pto::AddressSpace::VEC ||
        !pto::frontiersynch::ptoStorageProtection().protectsScalar(scalar, scalar)) {
        return NativeOrder::Unknown;
    }
    // Supplied premise: each admitted scalar memory operation completes before
    // the next admitted scalar operation starts. This adds C_i -> I_next;
    // it does not discharge vector/transfer visibility into a scalar read.
    return NativeOrder::Serialized;
}
} // namespace mlir::pto::frontiersynch::prefix
#endif
