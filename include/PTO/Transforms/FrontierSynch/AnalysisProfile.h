// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_ANALYSISPROFILE_H
#define PTO_FRONTIERSYNCH_ANALYSISPROFILE_H
#include "PTO/Transforms/FrontierSynch/AnalysisLedger.h"
#include "llvm/ADT/StringRef.h"
#include <chrono>
#include <memory>
namespace mlir::pto::frontiersynch {
// LIFO scopes charge child elapsed time out of their parent exclusive time.
// Disabled profiling neither reads a clock nor retains stage records.
class AnalysisProfileScope {
public:
    AnalysisProfileScope(std::shared_ptr<AnalysisLedger> ledger, llvm::StringRef stage,
        std::size_t region = 0, llvm::StringRef method = {}, bool newRequest = false, uint64_t originatingRequest = 0);
    ~AnalysisProfileScope();
    AnalysisProfileScope(const AnalysisProfileScope&) = delete;
    AnalysisProfileScope& operator=(const AnalysisProfileScope&) = delete;
    uint64_t requestId() const;
    void constructed();
    void dispatched();
    void cacheHit();
    void recordWork(std::vector<std::pair<std::string, uint64_t>> work);
private:
    std::shared_ptr<AnalysisLedger> ledger;
    std::size_t index = 0;
    uint64_t request = 0, previousRequest = 0;
    bool ownsRequest = false, timed = false;
    std::chrono::steady_clock::time_point started;
};
} // namespace mlir::pto::frontiersynch
#endif
