// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEREGIONAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEREGIONAL_H
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplate.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Finite/LifetimeScanInput.h"
namespace mlir::pto::frontiersynch {
// Finite body order and extrema, without retained effect-to-atom incidences.
// The accompanying NumericTemplate owns the compact source memberships.
struct NumericBodyAnalysis {
    SmallVector<const CompoundInstanceElement*> phases;
    StorageScanResult scan;
    StorageScanWork scanWork;
    ExplicitReduction reduction;
    std::vector<ExplicitCellBoundary> storageBoundary;
};
struct NumericBodyMathematics {
    NumericTemplate body;
    std::shared_ptr<const NumericBodyAnalysis> demands;
    std::shared_ptr<const SyncInput> inputOwner;
    std::shared_ptr<const PhaseIndex> indexOwner;
    std::shared_ptr<const NormalizedControlDescription> normalized;
};
std::shared_ptr<const NumericBodyAnalysis> analyzeNumericBody(const NumericTemplate& body, std::string& error);
FailureOr<RegionalAnalysis> exportNumericBody(func::FuncOp function,
    const SyncInput& input, const NumericBodyMathematics& mathematics,
    std::shared_ptr<RegionExpressions> expressions, ArrayRef<scf::ForOp> enclosing, std::string& error);
// One specialized body, with all bounded inner coordinates retained at their
// original cuts. No loop is inferred or repeated by this adapter.
FailureOr<RegionalAnalysis> numericBodyRegionalResult(func::FuncOp function,
    const SyncInput& input, const NumericTemplate& body, std::shared_ptr<RegionExpressions> expressions,
    ArrayRef<scf::ForOp> enclosing, std::string& error);
// Normalize one original explicit run under a caller-certified geometry phase.
// No loop is expanded. The caller owns speculative arena/callback lifetimes;
// unsupported geometry can fall back to another regional provider.
FailureOr<RegionalAnalysis> specializedExplicitRunRegional(func::FuncOp function, scf::ForOp outer,
    ArrayRef<Operation*> operations, const PhaseIndex& index, const SyncInput& input,
    std::shared_ptr<RegionExpressions> expressions, ArrayRef<scf::ForOp> enclosing,
    TemplateGeometryConstant geometry, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
