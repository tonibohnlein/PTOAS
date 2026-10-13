// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_SEQUENCESTORAGEINDEX_H
#define PTO_FRONTIERSYNCH_SEQUENCESTORAGEINDEX_H
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
// Index one finalized partition. Enumerate only cells covered by a selection;
// unrelated storage domains and empty child/cell combinations are never visited.
class SequenceStorageIndex {
public:
    explicit SequenceStorageIndex(ArrayRef<SyncStorageCell> cells) : cells(cells)
    {
        for (std::size_t id = 0; id < cells.size(); ++id) {
            domains[cells[id].space][cells[id].base].push_back(id);
        }
        for (auto& [space, bases] : domains) {
            for (auto& [base, ids] : bases) {
                std::sort(ids.begin(), ids.end(), [&](auto a, auto b) { return cells[a].begin < cells[b].begin; });
            }
        }
    }
    template <typename Visit> void covered(const SyncStorageCell& range, Visit visit) const
    {
        auto space = domains.find(range.space);
        if (space == domains.end() || range.begin >= range.end) { return; }
        auto base = space->second.find(range.base);
        if (base == space->second.end()) { return; }
        const auto& ids = base->second;
        auto it = std::lower_bound(ids.begin(), ids.end(), range.begin,
            [&](auto id, auto begin) { return cells[id].begin < begin; });
        for (; it != ids.end() && cells[*it].begin < range.end; ++it) {
            if (cells[*it].end <= range.end) { visit(*it); }
        }
    }
private:
    ArrayRef<SyncStorageCell> cells;
    std::map<AddressSpace, llvm::MapVector<Value, std::vector<std::size_t>>> domains;
};
} // namespace mlir::pto::frontiersynch
#endif
