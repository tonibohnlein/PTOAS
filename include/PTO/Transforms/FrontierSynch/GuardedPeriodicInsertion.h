// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDPERIODICINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDPERIODICINSERTION_H
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
// Borrowed exact periodic graph with original endpoint cuts. Periodic ordinal
// is floor(original loop ordinal / period); residues select each original cut.
// Empty residues mean period=1. Payload order must match residue/site reference
// order, and the finite invocation is a prefix of that invariant skeleton.
// Local demands use a barrier immediately before the consumer. For nonadjacent
// local endpoints this may strengthen the order without changing the supplied
// minimum-demand result; no order-equality certificate is implied.
struct GuardedPeriodicEndpointInput {
    scf::ForOp loop;
    std::shared_ptr<RegionExpressions> expressions;
    ArrayRef<const CompoundInstanceElement*> phases;
    ArrayRef<GuardedPeriodicPayload> payloads;
    ArrayRef<GuardedPeriodicRecord> generators;
    const GuardedPeriodicQuotient* periodic = nullptr;
    uint64_t period = 1;
    ArrayRef<uint64_t> residues;
};
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedPeriodicEndpoints(
    func::FuncOp function, const GuardedPeriodicEndpointInput& input, std::string& error,
    const RegionalDemandFilter& filter = {});
} // namespace mlir::pto::frontiersynch
#endif
