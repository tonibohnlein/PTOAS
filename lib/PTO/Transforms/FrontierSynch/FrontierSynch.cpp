// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/FrontierSynch.h"
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"

namespace mlir::pto::frontiersynch {
LogicalResult run(func::FuncOp function, const SyncInput& input)
{
    OriginalStructure original;
    if (failed(importOriginalStructure(function, input, original))) {
        return failure();
    }
    ProgramAnalysis analysis(input, std::move(original));
    if (!analysis.complete()) {
        return function.emitError("frontier-synch: original lifetime analysis failed: ") << analysis.reason();
    }
    // Query the frozen original obligation universe, not a materialized list
    // of every possible source/target pair. Placement and selected completion
    // remain Phase B tasks; this diagnostic does not instantiate either.
    const auto& obligations = analysis.obligations().stats();
    return function.emitError("frontier-synch: imported ")
           << input.instructions().size() << " instruction phases through InsertSync; analyzed "
           << analysis.structure().cells.size() << " storage cells; indexed "
           << obligations.factoredFamilies + obligations.marginalFamilies << " original storage obligation families ("
           << obligations.factoredFamilies << " factored, " << obligations.marginalFamilies
           << " conservative marginal), " << obligations.typedFamilies << " control-value obligation families, and "
           << analysis.lifetimes().stats().accessIncidences << " physical access incidences; "
           << "construction is not implemented yet";
}
} // namespace mlir::pto::frontiersynch
