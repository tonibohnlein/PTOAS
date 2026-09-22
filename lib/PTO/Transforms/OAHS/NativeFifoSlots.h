// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_NATIVE_FIFO_SLOTS_H
#define PTO_OAHS_NATIVE_FIFO_SLOTS_H
#include "PTO/IR/SyncProtocolModel.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto::oahs::native_detail {
// One invocation-owned FIFO only. No product with other qualified occurrences,
// no assumption that separate SSA roots/directions imply separate GM storage.
inline void importFifoSlots(
    func::FuncOp function, Program& p, llvm::ArrayRef<mlir::Operation*> payload, std::vector<std::string>& notes)
{
    if (!p.observed || p.alternatingSlots || p.staticFifoSlots)
        return;
    for (const auto& observation : p.observed->observations)
        if (!observation.atoms.empty())
            return;
    Value handle;
    AlternatingSlotRegion region;
    unsigned cell = unsigned(-1);
    for (std::size_t op = 0; op < payload.size(); ++op) {
        auto model = getSyncProtocolModel(payload[op]);
        if (!model || (model->kind != SyncProtocolModel::Receive && model->kind != SyncProtocolModel::Send))
            continue;
        if (!model->complete() || model->globalSlots != 2 || (handle && model->handle != handle))
            return;
        handle = model->handle;
        region.slotBytes = model->globalSlotBytes;
        auto& population = model->kind == SyncProtocolModel::Receive ? region.reads : region.writes;
        population.push_back(op);
        unsigned found = 0;
        for (auto a : p.operations[op].accesses) {
            if (p.cells[a.cell].addressSpace != std::to_string(unsigned(AddressSpace::GM)))
                continue;
            if ((cell != unsigned(-1) && cell != a.cell) || ++found != 1)
                return;
            cell = a.cell;
        }
        if (!found)
            return;
    }
    if (!handle || cell == unsigned(-1))
        return;
    auto init = handle.getDefiningOp<InitializeL2G2LPipeOp>();
    if (!init || init->getBlock() != &function.getBody().front())
        return;
    const auto& root = p.cells[cell];
    if (root.storageOrigins.size() != 1 || root.nativeMmadAccOrder)
        return;
    // Preserve every alias witness. Decline a root shared with other imported
    // storage records rather than splitting only one half of an overlap.
    for (unsigned other = 0; other < p.cells.size(); ++other) {
        if (other == cell)
            continue;
        for (auto origin : p.cells[other].storageOrigins)
            if (origin == root.storageOrigins.front())
                return;
    }
    region.cell = cell;
    auto refined = refineStaticSlots(p, region);
    bool exact = refined.success;
    if (exact) {
        const auto& slots = *refined.program.staticFifoSlots;
        for (const auto* population : {&slots.reads, &slots.writes}) {
            for (auto op : *population) {
                unsigned count = 0;
                for (auto access : refined.program.operations[op].accesses) {
                    count += llvm::is_contained(slots.cells, access.cell);
                }
                exact &= count == 1;
            }
        }
    }
    if (exact) {
        p = std::move(refined.program);
        notes.push_back("qualified static FIFO cursor slots; original graph unchanged");
        return;
    }
    auto alternating = refineAlternatingSlots(p, region);
    if (alternating.success) {
        p = std::move(alternating.program);
        notes.push_back("qualified alternating two-slot FIFO uses; shared original words");
        return;
    }
    if (!refined.success) {
        notes.push_back("kept pooled FIFO effects: " + refined.reason);
        return;
    }
    p = std::move(refined.program);
    notes.push_back("retained partial FIFO cursor facts; ambiguous accesses touch both slots");
}
} // namespace mlir::pto::oahs::native_detail
#endif
