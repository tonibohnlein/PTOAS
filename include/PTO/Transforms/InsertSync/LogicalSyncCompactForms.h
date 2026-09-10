// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCCOMPACTFORMS_H
#define PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCCOMPACTFORMS_H

#include <cstddef>
#include <vector>

namespace mlir::pto::logical_sync::compact {

// Preserve deterministic key order, but do not build a larger event family
// merely to save a key when an unused one exists. Occupancy is a snapshot for
// one immutable assignment step; the caller must still prove recurrence.
template <typename Occupied>
std::vector<unsigned> unusedFirstKeys(unsigned capacity, Occupied occupied)
{
    std::vector<unsigned> free, used;
    free.reserve(capacity);
    used.reserve(capacity);
    for (unsigned k = 0; k < capacity; ++k)
        (occupied(k) ? used : free).push_back(k);
    free.insert(free.end(), used.begin(), used.end());
    return free;
}

// Preserve the existing sharing-first search in a genuinely over-capacity
// domain. Otherwise early dedicated assignments could occupy all keys and
// prevent a later stream from using a key that earlier sharing would free.
template <typename Occupied>
std::vector<unsigned> eventKeyTrialOrder(unsigned capacity, size_t streamCount, Occupied occupied)
{
    if (streamCount <= capacity) return unusedFirstKeys(capacity, occupied);
    std::vector<unsigned> keys;
    keys.reserve(capacity);
    for (unsigned k = 0; k < capacity; ++k) keys.push_back(k);
    return keys;
}

} // namespace mlir::pto::logical_sync::compact
#endif
