// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Session cache entries retain demands independently of capability failures.
// An instance belongs to one unchanged SyncInput, alias/hardware policy and
// original occurrence tree. initialize/invalidate destroys every entry when
// that context changes. Specialized parameter/phase views do not use these
// original-region keys; their private arenas and substitutions remain separate.
#ifndef PTO_FRONTIERSYNCH_ANALYSISSESSIONINTERNAL_H
#define PTO_FRONTIERSYNCH_ANALYSISSESSIONINTERNAL_H
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include <set>
namespace mlir::pto::frontiersynch {
struct BackendAttempt {
    bool produced = false;
    std::shared_ptr<const MathematicalResult> mathematical;
    std::string demandError;
    std::optional<bool> endpoints;
    // The first successful synchronization request owns its detached fragment
    // until PrepareLogical consumes it. Later preparations instantiate afresh.
    std::unique_ptr<PreparedLogicalPlan> pendingLogical;
    std::shared_ptr<const NumericTemplateEndpoints> periodicEndpoints;
    // Only the allocation capability is memoized here, including unavailable exports.
    std::map<int64_t, DictionaryAttr> allocation;
    std::map<std::pair<std::size_t, std::size_t>, int64_t> arithmeticRecords;
};
struct AnalysisSessionState {
    std::map<std::size_t, std::map<AnalysisBackend, BackendAttempt>> attempts;
    // Canonical original-loop construction is independent of root/regional exports.
    std::map<std::pair<std::size_t, AnalysisBackend>, BackendAttempt> loopAttempts;
    std::shared_ptr<RegionExpressions> expressions = std::make_shared<RegionExpressions>();
    std::set<AnalysisBackend> wholeRegionEvidence;
    std::vector<std::size_t> activeRegions;
};
} // namespace mlir::pto::frontiersynch
#endif
