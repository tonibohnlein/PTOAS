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
    auto av = a.hasValidShape() ? a.getValidShape() : a.getShape();
    auto bv = b.hasValidShape() ? b.getValidShape() : b.getShape();
    auto cv = c.hasValidShape() ? c.getValidShape() : c.getShape();
    if (av.size() != 2 || bv.size() != 2 || cv.size() != 2 ||
        av[0] <= 0 || av[0] > 4095 || av[1] <= 0 || av[1] > 4095 ||
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
        if (!matrix->initializes || nextGroup == std::numeric_limits<uint64_t>::max()) {
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
} // namespace mlir::pto::frontiersynch
