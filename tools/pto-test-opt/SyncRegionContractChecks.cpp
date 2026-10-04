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
#include "PTO/IR/PTOAccessRegion.h"
#include "../../lib/PTO/Transforms/InsertSync/SyncEffectRanges.h"
#include "llvm/Support/raw_ostream.h"
#include "SyncScalarEvolutionChecks.h"
using namespace mlir;
using namespace mlir::pto;

int runSyncRegionContractChecks(func::FuncOp function, const SyncInput& input)
{
    if (!checkScalarEvolution(function)) {
        return 1;
    }
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
        if (checked == 0) {
            if (auto contract = function->getAttrOfType<DictionaryAttr>("test.selected_region")) {
                SyncMemoryEffect selection(kind, operand, contract);
                auto selectedCopy = effect;
                mlir::pto::detail::applyAccessCoverage(input, selectedCopy, {selection});
                auto begin = function->getAttrOfType<IntegerAttr>("test.selected_begin");
                auto bytes = function->getAttrOfType<IntegerAttr>("test.selected_bytes");
                if (!begin || !bytes || selectedCopy.precision != SyncAccessPrecision::Exact ||
                    !selectedCopy.exactRanges || selectedCopy.ranges.size() != 1 ||
                    selectedCopy.ranges[0].begin != begin.getValue().getZExtValue() ||
                    selectedCopy.ranges[0].end - selectedCopy.ranges[0].begin != bytes.getValue().getZExtValue()) {
                    return 1;
                }
            }
        }
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
        auto selected = makeAccessRegion(*operand, false,
            AffineMap::getMultiDimIdentityMap(effect.descriptorRegion->extents.size(), function.getContext()));
        SyncMemoryEffect identity(kind, operand, selected);
        mlir::pto::detail::applyAccessCoverage(input, copy, {identity});
        if (copy.precision != SyncAccessPrecision::Exact || !copy.region ||
            copy.region->byteOffset != effect.descriptorRegion->byteOffset ||
            copy.region->extents != effect.descriptorRegion->extents) {
            return 1;
        }
        NamedAttrList malformed(selected);
        malformed.set("shape_operand", IntegerAttr::get(IntegerType::get(function.getContext(), 64), -1));
        SyncMemoryEffect invalid(kind, operand, malformed.getDictionary(function.getContext()));
        mlir::pto::detail::applyAccessCoverage(input, copy, {invalid});
        if (copy.precision != SyncAccessPrecision::Unknown || copy.region) {
            return 1;
        }
        mlir::pto::detail::applyAccessCoverage(input, copy, {full, identity});
        if (copy.precision != SyncAccessPrecision::Unknown || copy.region) {
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
