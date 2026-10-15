// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Attempt-owned diagnostics; no generated values or admission policy.
#ifndef PTO_FRONTIERSYNCH_PREFIXEXPRESSIONPROFILE_H
#define PTO_FRONTIERSYNCH_PREFIXEXPRESSIONPROFILE_H
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
namespace mlir {
class Operation;
namespace pto::frontiersynch {
class PrefixExpressionProfile {
public:
    struct Entry {
        uint64_t arena = 0, root = 0, requests = 0, emissions = 0, attempts = 0;
        uint64_t directFailures = 0, inactiveAttempts = 0, historyAttempts = 0;
        uint64_t constructed = 0, retained = 0, maximumRetained = 0;
        uint64_t inactiveSuccesses = 0, preparationNodes = 0, preparationInputs = 0, sharedReturns = 0,
                 bindingNodes = 0;
        uint64_t thresholdDiscovery = 0, thresholdSubstitution = 0, proofVisits = 0, historySubstitution = 0;
        uint64_t thresholdDependencies = 0, thresholdReused = 0, thresholdSkipped = 0, thresholdPreparationHits = 0;
        uint64_t distance = 0;
        std::string cut, location, producer;
    };
    Entry& request(const void* arena, uint32_t root, Operation* cut,
                   const std::string& producer = "regional", uint64_t distance = 0);
    void resetEmission();
    llvm::json::Object report() const;
private:
    using Key = std::tuple<const void*, uint32_t, Operation*, std::string, uint64_t>;
    std::map<Key, Entry> entries;
    std::map<const void*, uint64_t> arenas;
};
} // namespace pto::frontiersynch
} // namespace mlir
#endif
