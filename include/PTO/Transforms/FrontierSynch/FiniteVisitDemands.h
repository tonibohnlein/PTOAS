// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEVISITDEMANDS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEVISITDEMANDS_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
namespace mlir::pto::frontiersynch {
// TRUSTED storage producer boundary: these selectors describe exactly the
// common persistent cells and shared read-only effects of the original type.
// Omitted effects are certified disjoint visit-owned storage, or shared reads
// disjoint from every writer of every type. The certificate includes cross-type
// and cross-visit disjointness, not just disjointness within one type. It also
// certifies that residual relationships are represented by the supplied cells
// or by the explicit adjacent prerequisite templates. The original internal
// graph and recipes are never projected. owner retains the geometric proof.
struct FiniteVisitStorageProjection {
    std::vector<RegionalStorageBoundary> storageBoundary;
    std::shared_ptr<const void> owner;
};
struct FiniteVisitType {
    RegionalAnalysis body;
    std::optional<FiniteVisitStorageProjection> projectedStorage;
};
struct FiniteVisitPrerequisite {
    uint32_t sourceType = 0, targetType = 0;
    // Completion -> Start in the two distinct visits; body-local coordinates.
    RegionalEvent source, target;
    RegionExpressions::Id guard = RegionExpressions::invalid;
    bool native = false;
};
struct FiniteVisitInput {
    std::vector<FiniteVisitType> types;
    std::vector<FiniteVisitPrerequisite> prerequisites;
    std::vector<scf::ForOp> enclosing;
    // Immutable invocation context. Invalid selects true. Type selection is
    // NOT this context: bodies are already specialized within their own type.
    RegionExpressions::Id context = RegionExpressions::invalid;
};
struct FiniteVisitBoundary {
    uint32_t sourceType = 0, targetType = 0;
    std::vector<RepeatedCrossing> demands, native;
};
struct FiniteVisitCost {
    uint64_t qualificationChecks = 0, pairReductions = 0;
    RegionalCost regional;
};
struct FiniteVisitDemands {
    std::string error;
    // Preserved even on qualification failure. Original IR/shared effects are
    // borrowed unchanged, as in RegionalAnalysis; typed owners stay alive here.
    std::shared_ptr<const FiniteVisitInput> original;
    // Row-major sourceType*h+targetType. Every record has displacement one.
    std::vector<FiniteVisitBoundary> boundaries;
    // Potential common support; selectors/records carry immutable-parameter presence.
    std::vector<SyncStorageCell> persistentCells;
    std::vector<uint32_t> pipes;
    FiniteVisitCost cost;
};
// Caller certifies exact internal covers/queries/selectors and within-type
// invariance of their predicates, coordinates and physical maps. All symbolic
// inputs are shared immutable invocation parameters; visit-varying values must
// be specialized or independently bound BEFORE this constructor. Reusing one
// unbound branch predicate for two different visits violates this contract.
// Supplied prerequisites cover ALL additional inter-visit requirements and
// connect adjacent visits only. Native pipe crossings are generated here.
//
// Validates pipe presence and persistent refresh in every type whenever any type
// contains that pipe or writer under the immutable invocation context,
// then performs h^2 existing exact two-region crossing reductions. Work is those
// reductions plus charged qualification and descriptor scans, independent of T.
// This is a demand-template library, not an arbitrary-word all-event query,
// whole-region storage selector, endpoint-availability or allocation certificate.
// A consumer selects internal type recipes and boundary (sigma[t],sigma[t+1])
// recipes; both endpoints must use the same source visit and original record.
FiniteVisitDemands buildFiniteVisitDemands(func::FuncOp function, FiniteVisitInput input);
} // namespace mlir::pto::frontiersynch
#endif
