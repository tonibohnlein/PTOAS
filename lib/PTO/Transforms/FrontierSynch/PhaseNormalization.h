// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_PHASENORMALIZATION_H
#define PTO_FRONTIERSYNCH_PHASENORMALIZATION_H
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/InsertSync/SyncAccessRegion.h"
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
class PhaseNormalization {
public:
    PhaseNormalization(scf::ForOp outer, const PhaseIndex& index, RegionExpressions& arena)
        : outer(outer), index(index), arena(arena) {}
    bool independent(Value value);
    // Exact scalar modulo spelling, including a contiguous low-bit mask.
    static std::optional<uint64_t> modulus(Value value);
    // Whole-value invariance, unlike residue(), which proves only a congruence.
    bool periodic(Value value, uint64_t period);
    bool periodic(const SyncAccessRegion& region, uint64_t period);
    std::optional<RegionExpressions::Id> atPhase(Value value, uint64_t phase, uint64_t period);
    std::optional<RegionExpressions::Id> residue(Value value, uint64_t phase, uint64_t period);
private:
    scf::ForOp outer;
    const PhaseIndex& index;
    RegionExpressions& arena;
    DenseMap<Value, bool> independence;
    DenseMap<Value, std::map<uint64_t, bool>> periodicValues;
    DenseMap<Value, std::map<std::pair<uint64_t, uint64_t>, std::optional<RegionExpressions::Id>>> phaseValues;
    DenseMap<Value, std::map<std::pair<uint64_t, uint64_t>, std::optional<RegionExpressions::Id>>> residues;
};
} // namespace mlir::pto::frontiersynch
#endif
