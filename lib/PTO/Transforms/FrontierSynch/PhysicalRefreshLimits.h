// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Representation preflight only; failure does not exclude a mathematical class.
#ifndef PTO_FRONTIERSYNCH_PHYSICALREFRESHLIMITS_H
#define PTO_FRONTIERSYNCH_PHYSICALREFRESHLIMITS_H
#include <cstdint>
#include <initializer_list>
namespace mlir::pto::frontiersynch::detail {
constexpr uint64_t physicalRefreshFragmentLimit = 65536;
inline bool physicalRefreshWindowFits(uint64_t span, uint64_t payloads, uint64_t guards, uint64_t accesses)
{
    if (!span || span >= physicalRefreshFragmentLimit) {
        return false;
    }
    uint64_t perVisit = 0;
    for (auto count : {payloads, guards, accesses}) {
        if (count > physicalRefreshFragmentLimit - perVisit) {
            return false;
        }
        perVisit += count;
    }
    return perVisit <= physicalRefreshFragmentLimit / (span + 1);
}
} // namespace mlir::pto::frontiersynch::detail
#endif
