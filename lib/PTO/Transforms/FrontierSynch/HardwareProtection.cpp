// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A2/A3 MMAD accumulator access ordering; physical effects remain unchanged.
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include <limits>
#include <optional>
namespace mlir::pto::frontiersynch {
namespace {
struct MatrixFacts {
    Type accumulator;
    bool initializes = false;
};
std::optional<MatrixFacts> facts(Operation* operation)
{
    if (!operation || !isTargetArchA3(operation)) {
        return std::nullopt;
    }
    Value lhs, rhs, dst;
    bool initializes = false;
    if (auto init = dyn_cast<TMatmulOp>(operation)) {
        lhs = init.getLhs();
        rhs = init.getRhs();
        dst = init.getDst();
        initializes = true;
    } else if (auto acc = dyn_cast<TMatmulAccOp>(operation)) {
        lhs = acc.getLhs();
        rhs = acc.getRhs();
        dst = acc.getDst();
        if (acc.getAccIn() != dst) {
            return std::nullopt; // The native lowering accumulates into dst.
        }
    } else {
        return std::nullopt;
    }
    auto a = dyn_cast<TileBufType>(lhs.getType());
    auto b = dyn_cast<TileBufType>(rhs.getType());
    auto c = dyn_cast<TileBufType>(dst.getType());
    if (!a || !b || !c) {
        return std::nullopt;
    }
    auto aShape = resolveConstantTileValidShape(lhs, operation);
    auto bShape = resolveConstantTileValidShape(rhs, operation);
    auto cShape = resolveConstantTileValidShape(dst, operation);
    if (!aShape || !bShape || !cShape) {
        return std::nullopt;
    }
    const auto& av = *aShape;
    const auto& bv = *bShape;
    const auto& cv = *cShape;
    if (av[0] <= 0 || av[0] > 4095 || av[1] <= 0 || av[1] > 4095 ||
        bv[1] <= 0 || bv[1] > 4095 || av[1] != bv[0] || av[0] != cv[0] || bv[1] != cv[1]) {
        return std::nullopt;
    }
    // A2/A3-compatible Mmad API, synchronization optimization:
    // https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.html
    // The bounded native dimensions above also make this product overflow-safe.
    constexpr int64_t block = 16;
    constexpr int64_t threshold = 10;
    if ((av[0] / block) * (bv[1] / block) < threshold) {
        return std::nullopt;
    }
    return MatrixFacts{c, initializes};
}
bool allWritten(const ExplicitEffects& occurrence, llvm::ArrayRef<uint32_t> atoms)
{
    std::unordered_set<uint32_t> writes;
    for (const auto& access : occurrence.accesses) {
        if (access.write) {
            writes.insert(access.atom);
        }
    }
    for (auto atom : atoms) {
        if (!writes.count(atom)) {
            return false;
        }
    }
    return !atoms.empty();
}
} // namespace
void HardwareProtectionBuilder::endScope()
{
    activeGroup = 0;
    accumulatorType = {};
    activeAtoms.clear();
}
void HardwareProtectionBuilder::observe(Operation* operation, ExplicitEffects& occurrence,
                                        llvm::ArrayRef<uint32_t> accumulatorAtoms)
{
    for (auto& access : occurrence.accesses) {
        access.protectionGroup = 0;
    }
    auto matrix = facts(operation);
    const auto matrixPipe = static_cast<uint32_t>(PipelineType::PIPE_M);
    if (!matrix || occurrence.pipe != matrixPipe || !allWritten(occurrence, accumulatorAtoms)) {
        for (const auto& access : occurrence.accesses) {
            if (activeAtoms.count(access.atom)) {
                endScope();
                break;
            }
        }
        if (occurrence.pipe == matrixPipe) {
            endScope();
        }
        return;
    }
    std::unordered_set<uint32_t> atoms(accumulatorAtoms.begin(), accumulatorAtoms.end());
    if (matrix->initializes || matrix->accumulator != accumulatorType || atoms != activeAtoms) {
        endScope();
        if (!matrix->initializes || nextGroup == invocationProtectionBit - 1) {
            return;
        }
        activeGroup = ++nextGroup;
        accumulatorType = matrix->accumulator;
        activeAtoms = std::move(atoms);
    }
    for (auto& access : occurrence.accesses) {
        if (activeAtoms.count(access.atom)) {
            access.protectionGroup = activeGroup;
        }
    }
}
std::vector<uint64_t> modeledProtectionGroups(const SyncInput& input,
    llvm::ArrayRef<const CompoundInstanceElement*> phases)
{
    const auto effects = input.accesses().effects();
    std::vector<uint64_t> groups(effects.size(), 0);
    HardwareProtectionBuilder builder;
    std::optional<std::size_t> active;
    for (const auto* phase : phases) {
        SmallVector<std::size_t> accumulators;
        for (auto id : input.accesses().effectsFor(phase)) {
            if (effects[id].memory->scope == AddressSpace::ACC) { accumulators.push_back(id); }
        }
        // An unrelated pipe can break the chain by accessing this accumulator.
        if (static_cast<uint32_t>(phase->kPipeValue) != static_cast<uint32_t>(PipelineType::PIPE_M)) {
            if (active && llvm::any_of(input.accesses().effectsFor(phase), [&](auto id) {
                    return input.accesses().mayOverlap(*active, id);
                })) { builder.endScope(); active.reset(); }
            continue;
        }
        if (accumulators.empty()) { builder.endScope(); active.reset(); continue; }
        const auto first = accumulators.front();
        const auto& candidate = effects[first];
        const bool oneOperand = candidate.sharedProvenanceComplete &&
            llvm::all_of(accumulators, [&](auto id) {
                return effects[id].sharedProvenanceComplete &&
                    effects[id].memory->baseBuffer == candidate.memory->baseBuffer;
            });
        if (!oneOperand) { builder.endScope(); active.reset(); continue; }
        if (!active || effects[*active].memory->baseBuffer != candidate.memory->baseBuffer) {
            builder.endScope();
        }
        ExplicitEffects occurrence;
        occurrence.pipe = static_cast<uint32_t>(phase->kPipeValue);
        for (auto id : accumulators) {
            occurrence.accesses.push_back({0, effects[id].mode == SyncAccessMode::Read,
                effects[id].mode == SyncAccessMode::Write});
        }
        builder.observe(phase->elementOp, occurrence, {0});
        const auto group = occurrence.accesses.front().protectionGroup;
        for (auto id : accumulators) { groups[id] = group; }
        active = group ? std::optional<std::size_t>(first) : std::nullopt;
    }
    return groups;
}
bool mayHaveHardwareProtectedPair(const SyncInput& input,
                                 llvm::ArrayRef<const CompoundInstanceElement*> phases)
{
    if (phases.size() > UINT32_MAX / 2 || llvm::any_of(phases, [](const auto* phase) { return !phase; })) {
        return true;
    }
    std::vector<ExplicitEffects> candidates(phases.size());
    for (uint32_t i = 0; i < phases.size(); ++i) {
        auto& candidate = candidates[i];
        candidate.payload = i; candidate.pipe = static_cast<uint32_t>(phases[i]->kPipeValue);
        bool read = false, write = false;
        for (auto id : input.accesses().effectsFor(phases[i])) {
            const auto& effect = input.accesses().effects()[id];
            if (effect.memory && effect.memory->scope == AddressSpace::ACC) {
                read |= effect.mode == SyncAccessMode::Read;
                write |= effect.mode == SyncAccessMode::Write;
            }
        }
        if (write) { candidate.accesses.push_back({0, read, true}); }
    }
    // Test potential writer pairs through the common target rule. Collapsing
    // accumulator atoms and skipping intermediate sites overapproximates
    // possible protection; rejecting it is conservative. In particular an
    // inactive intervening site cannot hide a newly protected pair. Check
    // both orders and repeated sites, since pairs may cross iteration ends.
    for (std::size_t a = 0; a < candidates.size(); ++a) {
        if (candidates[a].accesses.empty()) { continue; }
        for (std::size_t b = 0; b < candidates.size(); ++b) {
            if (candidates[b].accesses.empty()) { continue; }
            auto first = candidates[a], second = candidates[b];
            second.payload = static_cast<uint32_t>(candidates.size() + b);
            HardwareProtectionBuilder protection;
            protection.observe(phases[a]->elementOp, first, {0});
            protection.observe(phases[b]->elementOp, second, {0});
            if (hardwareProtectsConflict(first.pipe, first.accesses.front().protectionGroup,
                                        second.pipe, second.accesses.front().protectionGroup)) {
                return true;
            }
        }
    }
    return false;
}
} // namespace mlir::pto::frontiersynch
