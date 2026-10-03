// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private import of a fixed executed body under its admitted regional context.
// Coordinates qualify bounds only. Static hazard equivalence establishes the
// selected shared model, never physical exact effects or a native-order fact.
#ifndef PTO_FRONTIERSYNCH_STATIONARYCELLS_H
#define PTO_FRONTIERSYNCH_STATIONARYCELLS_H
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CostLedger.h"
#include <memory>
#include <string>
namespace mlir::pto { class SyncInput; }
namespace mlir::pto::frontiersynch {
class StationaryCellInput {
public:
    static FailureOr<std::shared_ptr<const StationaryCellInput>> build(
        const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases,
        CostLedger& costs, std::string& reason);
    const RotatingFootprintAnalysis& storage() const { return normalized; }
    ArrayRef<std::size_t> cells() const { return sharedCells; }
    std::size_t sitePairChecks() const { return checkedPairs; }
private:
    StationaryCellInput() = default;
    RotatingFootprintAnalysis normalized;
    SmallVector<std::size_t> sharedCells;
    std::size_t checkedPairs = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
