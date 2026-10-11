// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One session-owned diagnostic ledger. Outcomes retain this owner and request
// identity; reports materialize snapshots only at the reporting boundary.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISLEDGER_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISLEDGER_H
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include <vector>
#include <utility>
namespace mlir::pto::frontiersynch {
struct AnalysisStageRecord {
    uint64_t id = 0, request = 0, parent = 0, parentRequest = 0;
    std::size_t region = 0;
    std::string stage, method;
    uint64_t elapsedNanoseconds = 0, exclusiveNanoseconds = 0;
    EstimatedCount constructions;
    uint64_t cacheHits = 0, firstDispatches = 0;
};
struct AnalysisLedger {
    std::vector<AnalysisCostRecord> estimates;
    std::vector<AnalysisStageRecord> stages;
    // Profiling is opt-in. IDs name entries within this immutable-input session.
    bool profiling = false;
    uint64_t nextRequest = 0, currentRequest = 0;
    std::vector<std::size_t> activeStages;
    uint64_t snapshotRecordsCopied = 0;
    void recordEstimate(AnalysisCostRecord record)
    {
        record.id = estimates.size() + 1;
        record.requestId = currentRequest;
        estimates.push_back(std::move(record));
    }
};
} // namespace mlir::pto::frontiersynch
#endif
