// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test the generic consumer with synthetic MLIR effect declarations. These
// declarations do not assert native coverage for the fixture's PTO operations.
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "../../lib/PTO/Transforms/InsertSync/SyncEffectRanges.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::pto;

int runSyncRegionContractChecks(func::FuncOp function, const SyncInput& input)
{
    SyncStorageEffects storage;
    if (failed(storage.build(input))) {
        return 1;
    }
    unsigned checked = 0;
    for (const auto& effect : storage.effects()) {
        if (!effect.descriptorRegion) {
            continue;
        }
        OpOperand* operand = nullptr;
        for (auto& candidate : effect.phase->elementOp->getOpOperands()) {
            if (candidate.get() == effect.memory->baseBuffer) {
                operand = &candidate;
                break;
            }
        }
        if (!operand) {
            return 1;
        }
        auto kind = effect.mode == SyncAccessMode::Read ?
            static_cast<MemoryEffects::Effect*>(MemoryEffects::Read::get()) : MemoryEffects::Write::get();
        SyncMemoryEffect full(kind, operand, 0, true);
        auto copy = effect;
        mlir::pto::detail::applyAccessCoverage(input, copy, {full});
        if (copy.precision != SyncAccessPrecision::Exact || !copy.region ||
            (copy.region->symbols.empty() && !copy.region->base && !copy.exactRanges)) {
            return 1;
        }
        if (copy.exactRanges) {
            uint64_t bytes = 0;
            for (const auto& range : copy.ranges) {
                bytes += range.end - range.begin;
            }
            if (auto expected = function->getAttrOfType<IntegerAttr>("test.region_bytes");
                expected && bytes != expected.getValue().getZExtValue()) {
                return 1;
            }
        } else if (!copy.ranges.empty()) {
            return 1;
        }
        SyncMemoryEffect partial(kind, operand);
        mlir::pto::detail::applyAccessCoverage(input, copy, {full, partial});
        if (copy.precision != SyncAccessPrecision::Unknown || copy.region || copy.exactRanges || !copy.ranges.empty()) {
            return 1;
        }
        SyncMemoryEffect parameters(kind, operand, StringAttr::get(function.getContext(), "opaque"), 0, true);
        mlir::pto::detail::applyAccessCoverage(input, copy, {parameters});
        if (copy.precision != SyncAccessPrecision::Unknown) {
            return 1;
        }
        mlir::pto::detail::applyAccessCoverage(input, copy, input.effectsFor(*effect.phase));
        if (copy.precision != effect.precision) {
            return 1;
        }
        ++checked;
    }
    if ((!checked) != function->hasAttr("test.no_region")) {
        return 1;
    }
    llvm::outs() << "synthetic-region-contract " << function.getSymName() << ": " << checked << " passed\n";
    return 0;
}
