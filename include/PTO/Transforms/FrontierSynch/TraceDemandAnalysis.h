// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared source sites and modeled conflict queries; control remains in MLIR.
// Selected demand reduction belongs to the regional analysis backends.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_TRACEDEMANDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_TRACEDEMANDANALYSIS_H
#include "PTO/Transforms/FrontierSynch/StructuredInputAdapter.h"

namespace mlir::pto::frontiersynch {
class TraceDemandAnalysis {
public:
    // Site records borrow SyncInput phases and MLIR anchors. Their owners must
    // outlive this analysis and any copies; queries require unchanged input IR.
    static FailureOr<TraceDemandAnalysis> build(func::FuncOp function, const SyncInput& input);
    ArrayRef<StructuredSite> sites() const { return records; }
    const std::optional<SmallVector<std::size_t>>& sequence() const { return explicitSequence; }
    ArrayRef<PipelineType> pipes() const { return columns; }
    // Invocation-local lazy cache; queries are single-threaded and borrow the
    // unchanged SyncInput just like phase/anchor records. Copies/moves retain
    // the same borrowing obligation.
    ArrayRef<SmallVector<std::size_t>> conflicts() const;
    bool conflictsPrepared() const { return conflictsReady; }

private:
    TraceDemandAnalysis() = default;
    std::optional<SmallVector<std::size_t>> explicitSequence;
    SmallVector<StructuredSite> records;
    SmallVector<PipelineType> columns;
    mutable SmallVector<SmallVector<std::size_t>> incoming;
    const SyncInput* sharedInput = nullptr;
    mutable bool conflictsReady = false;
};
} // namespace mlir::pto::frontiersynch
#endif
