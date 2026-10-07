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
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/IR/PTOAccessRegion.h"
#include "../../lib/PTO/Transforms/InsertSync/SyncEffectRanges.h"
#include "llvm/Support/raw_ostream.h"
#include "SyncScalarEvolutionChecks.h"
#include "../../lib/PTO/Transforms/FrontierSynch/CountedLoop.h"
#include "SyncExplicitAccessChecks.h"
using namespace mlir;
using namespace mlir::pto;

bool checkSyncRegionDigits(func::FuncOp function);

int runSyncRegionContractChecks(func::FuncOp function, const SyncInput& input)
{
    if (!checkScalarEvolution(function) || !checkExplicitAccessContracts(function, input) ||
        !checkSyncRegionDigits(function)) {
        return 1;
    }
    auto dimensions = function.walk([&](Operation* operation) -> WalkResult {
        auto expected = operation->getAttrOfType<DenseI64ArrayAttr>("test.tile_valid_shapes");
        if (!expected) {
            return WalkResult::advance();
        }
        if (expected.size() % 3) {
            return WalkResult::interrupt();
        }
        for (int64_t i = 0; i < expected.size(); i += 3) {
            auto operand = expected[i];
            if (operand < 0 || static_cast<uint64_t>(operand) >= operation->getNumOperands()) {
                return WalkResult::interrupt();
            }
            auto shape = resolveConstantTileValidShape(operation->getOperand(operand), operation);
            if (expected[i + 1] == -1 ? bool(shape) :
                (!shape || (*shape)[0] != expected[i + 1] || (*shape)[1] != expected[i + 2])) {
                operation->emitError("unexpected effective valid dimensions");
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    if (dimensions.wasInterrupted()) {
        return 1;
    }
    auto counted = function.walk([&](scf::ForOp loop) -> WalkResult {
        auto samples = loop->getAttrOfType<DenseI64ArrayAttr>("test.counted_samples");
        if (!samples) { return WalkResult::advance(); }
        auto domain = frontiersynch::CountedLoop::get(loop);
        if (!domain || samples.size() % 3) { return WalkResult::interrupt(); }
        frontiersynch::RegionExpressions arena;
        auto expression = domain->trips(arena);
        for (int64_t i = 0; i < samples.size(); i += 3) {
            frontiersynch::RegionExpressions::Substitution values({
                {arena.input(loop.getLowerBound()), arena.constant(samples[i])},
                {arena.input(loop.getUpperBound()), arena.constant(samples[i + 1])}});
            auto actual = arena.constantValue(arena.substitute(expression, values));
            if (!actual || *actual != static_cast<uint64_t>(samples[i + 2])) {
                loop.emitError("counted trip circuit differs from supplied execution count");
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    if (counted.wasInterrupted()) { return 1; }
    const auto groups = frontiersynch::structuredProtection(input.accesses());
    DenseMap<int64_t, uint64_t> expectedGroups;
    DenseMap<uint64_t, int64_t> actualGroups;
    for (auto* phase : input.instructions()) {
        auto expected = phase->elementOp->getAttrOfType<IntegerAttr>("test.protection_group");
        if (!expected) { continue; }
        const auto label = expected.getInt();
        const auto group = groups.at(phase) & ~frontiersynch::protectionResetBit;
        if (label < 0 || bool(label) != bool(group) ||
            expectedGroups.try_emplace(label, group).first->second != group ||
            actualGroups.try_emplace(group, label).first->second != label) {
            phase->elementOp->emitError("unexpected structured protection group");
            return 1;
        }
    }
    const auto& storage = input.accesses();
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
                if (!begin || !bytes || !selectedCopy.rangesMaterialized || selectedCopy.ranges.size() != 1 ||
                    selectedCopy.ranges[0].begin != begin.getValue().getZExtValue() ||
                    selectedCopy.ranges[0].end - selectedCopy.ranges[0].begin != bytes.getValue().getZExtValue()) {
                    return 1;
                }
            }
        }
        SyncMemoryEffect full(kind, operand, 0, true);
        auto copy = effect;
        mlir::pto::detail::applyAccessCoverage(input, copy, {full});
        if (!copy.region ||
            (copy.region->symbols.empty() && !copy.region->base && !copy.rangesMaterialized)) {
            return 1;
        }
        if (copy.rangesMaterialized) {
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
        if (!copy.region ||
            copy.region->byteOffset != effect.descriptorRegion->byteOffset ||
            copy.region->extents != effect.descriptorRegion->extents) {
            return 1;
        }
        auto bound = effect;
        SyncMemoryEffect boundDeclaration(kind, operand);
        mlir::pto::detail::applyAccessCoverage(input, bound, {boundDeclaration});
        auto matchesBound = [&]() {
            return copy.rangesMaterialized == bound.rangesMaterialized && copy.regions.size() == bound.regions.size() &&
                copy.ranges.size() == bound.ranges.size() && llvm::all_of(llvm::zip(copy.ranges, bound.ranges),
                    [](auto pair) {
                        const auto& a = std::get<0>(pair);
                        const auto& b = std::get<1>(pair);
                        return sameStorageDomain(a, b) && a.begin == b.begin && a.end == b.end;
                    });
        };
        NamedAttrList malformed(selected);
        malformed.set("shape_operand", IntegerAttr::get(IntegerType::get(function.getContext(), 64), -1));
        SyncMemoryEffect invalid(kind, operand, malformed.getDictionary(function.getContext()));
        mlir::pto::detail::applyAccessCoverage(input, copy, {invalid});
        if (!matchesBound()) {
            return 1;
        }
        mlir::pto::detail::applyAccessCoverage(input, copy, {full, identity});
        if (copy.regions.size() != 2) {
            return 1;
        }
        SyncMemoryEffect partial(kind, operand);
        mlir::pto::detail::applyAccessCoverage(input, copy, {full, partial});
        if (!matchesBound()) {
            return 1;
        }
        SyncMemoryEffect parameters(kind, operand, StringAttr::get(function.getContext(), "opaque"), 0, true);
        mlir::pto::detail::applyAccessCoverage(input, copy, {parameters});
        if (!matchesBound()) {
            return 1;
        }
        mlir::pto::detail::applyAccessCoverage(input, copy, input.effectsFor(*effect.phase));
        if (copy.rangesMaterialized != effect.rangesMaterialized || copy.regions.size() != effect.regions.size()) {
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
