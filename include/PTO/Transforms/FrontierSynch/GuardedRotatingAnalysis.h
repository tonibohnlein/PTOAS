// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGANALYSIS_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
namespace mlir::pto::frontiersynch {
// Normalized conditional accesses used by finite physical-boundary exporters.
// Offset is totalized on inactive arms. Read/write predicates already merge
// coincident aliases at one payload, so an RMW remains a writer in summaries.
struct GuardedRotatingFragment {
    uint32_t payload = 0;
    std::size_t effect = 0;
    std::pair<uint64_t, uint64_t> slotAtom;
    uint64_t slots = 0, divisor = 0, refresh = 0, inverseStride = 0;
    RegionExpressions::Id offset = RegionExpressions::invalid;
    RegionExpressions::Id read = RegionExpressions::invalid, write = RegionExpressions::invalid;
    std::optional<SyncStorageCell> firstPhysicalSlot;
    uint64_t physicalSlotStride = 0;
};
struct GuardedRotatingAnalysis {
    std::string error;
    scf::ForOp loop;
    std::shared_ptr<RegionExpressions> expressions;
    SmallVector<const CompoundInstanceElement*> phases;
    std::vector<GuardedPeriodicPayload> payloads;
    std::vector<GuardedPeriodicRecord> generators;
    std::vector<GuardedRotatingFragment> fragments;
    SmallVector<std::size_t> dischargedEffects;
    GuardedPeriodicQuotient periodic;
    uint64_t refreshBound = 0;
};
// Immutable presence and offset parameters, exact fixed within-slot atoms, and
// one potential payload skeleton. No iterations, slot values or guard valuations
// are enumerated. Conditional alias normalization and strict writer selection
// cost O(A^2) circuit operations, excluding encoded arithmetic bit costs.
// Queries/retention are supplied by the shared parameterized quotient. Insertion
// must separately establish legal matching cuts. Local barriers precede their consumers.
// guardBindings replace only predicates certified invariant on the analyzed
// slice, in the supplied expression arena; their original cuts are unchanged.
// Potential target-protected accumulator writer pairs are unsupported here;
// use the numerical protected route until conditional protection is represented
// in the required graph. Ordinary unprotected ACC effects are supported.
GuardedRotatingAnalysis analyzeGuardedRotating(scf::ForOp loop, const SyncInput& input,
    const GuardedRecognition& recognition, std::shared_ptr<RegionExpressions> expressions = {},
    const DenseMap<Value, RegionExpressions::Id>& guardBindings = DenseMap<Value, RegionExpressions::Id>());
// Session adapter: reuse the unchanged invocation's prerequisite index.
GuardedRotatingAnalysis analyzeGuardedRotating(scf::ForOp loop, const SyncInput& input,
    const GuardedRecognition& recognition, const PhaseIndex& index,
    std::shared_ptr<RegionExpressions> expressions = {},
    const DenseMap<Value, RegionExpressions::Id>& guardBindings = DenseMap<Value, RegionExpressions::Id>());
} // namespace mlir::pto::frontiersynch
#endif
