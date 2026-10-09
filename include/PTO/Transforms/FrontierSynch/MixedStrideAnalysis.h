// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Analysis-only phase expansion of fixed mixed-stride accesses; original payload IR stays intact.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_MIXEDSTRIDEANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_MIXEDSTRIDEANALYSIS_H
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
namespace mlir::pto::frontiersynch {
struct MixedStrideExpansion {
    std::string error;
    uint64_t period = 1;
    std::vector<PeriodicPayload> payloads;
    std::vector<RotatingFragment> fragments;
};
// The numerical period is lcm(b/gcd(s-t,b)) over accesses in each family.
// Site r*m+a denotes original site a in iteration p*n+r. Storage and work
// expand with p, not with runtime trips. Limits are representation limits;
// rejection supplies no exclusion from the mixed-stride mathematical class.
// Protection confined to one original visit is kept separate in each phase.
MixedStrideExpansion expandMixedStride(ArrayRef<PeriodicPayload> payloads,
    ArrayRef<RotatingFragment> fragments,
    uint64_t maximumExpandedPayloads = NumericTemplateLimits{}.payloads,
    uint64_t maximumExpandedFragments = NumericTemplateLimits{}.fragments);
// Lift an original periodic relation to the same phase-expanded skeleton.
// Both endpoint phases and the superperiod displacement are preserved exactly.
std::optional<std::vector<PeriodicRecord>> expandMixedStrideRecords(
    ArrayRef<PeriodicRecord> records, uint64_t payloadCount, uint64_t period);
struct MixedStrideDemands {
    scf::ForOp loop;
    std::vector<const CompoundInstanceElement*> phases;
    uint64_t period = 1;
    PeriodicAnalysis periodic;
};
FailureOr<std::shared_ptr<MixedStrideDemands>> analyzeMixedStrideFunction(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program,
    const PhaseIndex& index, std::string& error);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareMixedStrideLogicalInsertion(
    func::FuncOp function, const MixedStrideDemands& demands, std::string& error);
DictionaryAttr mixedStrideAllocationCertificate(
    const MixedStrideDemands& demands, int64_t plan, MLIRContext* context);
// Whole-function fixed-body adapter. Only the common-stride diagnostic may be
// discharged by expansion; every other shared recognition obligation remains.
// Endpoint guards check the original source and consumer against the original
// finite prefix independently, including incomplete final superperiods.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareMixedStrideInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
