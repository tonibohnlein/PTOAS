// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Optional uniform Presburger allocation of an already selected logical plan.
#ifndef PTO_FRONTIERSYNCH_GENERAL_PHYSICAL_EMISSION_H
#define PTO_FRONTIERSYNCH_GENERAL_PHYSICAL_EMISSION_H
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
#include <cstddef>
#include <string>
namespace mlir::pto {
class SyncInput;
namespace frontiersynch {
class TraceDemandAnalysis;
class CostLedger;
struct DirectEmissionResult;
// Relations use the same original schema/context and Event -> Event tuples.
// Production supplies sealed functional covers; this helper additionally checks
// strict ordered matching and uniform consumption before fixed-capacity reuse.
LogicalResult certifyGeneralReuse(SymbolicSchemaHandle schema,
    const presburger::PresburgerRelation& demand, const presburger::PresburgerRelation& native,
    const presburger::PresburgerRelation& readiness, PipelineType source, PipelineType target,
    std::size_t capacity, std::string& reason);
LogicalResult emitGeneralPhysical(const SyncInput&, const TraceDemandAnalysis&,
                                 DirectEmissionResult&, CostLedger&);
} // namespace frontiersynch
} // namespace mlir::pto
#endif
