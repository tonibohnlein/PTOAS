// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSLOTMAPPING_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSLOTMAPPING_H
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncPlanningPrimitives.h"

namespace mlir::pto {
// An unequal selector proves non-overlap only if every unequal-index pair is
// physically disjoint. Equal capacity or equal root names alone are insufficient.
// This same certificate is required when selectors index recurring event keys.
inline bool haveCompatibleInsertSyncSlots(const BaseMemInfo* first, const BaseMemInfo* second)
{
    if (!first || !second || first->scope != second->scope || first->scope == AddressSpace::GM ||
        first->aliasesUnknownRange || second->aliasesUnknownRange || first->baseAddresses.size() < 2 ||
        first->baseAddresses.size() > kMaxMultiBufferCount ||
        first->baseAddresses.size() != second->baseAddresses.size() || !first->allocateSize || !second->allocateSize) {
        return false;
    }
    if (!(first->hasKnownPhysicalAddresses && second->hasKnownPhysicalAddresses) &&
        (first->rootBuffer != second->rootBuffer ||
         first->hasKnownPhysicalAddresses != second->hasKnownPhysicalAddresses)) {
        return false;
    }
    for (size_t a = 0; a < first->baseAddresses.size(); ++a) {
        for (size_t b = 0; b < second->baseAddresses.size(); ++b) {
            if (a != b &&
                insert_sync_detail::addressRangesMayOverlap(
                    first->baseAddresses[a], first->allocateSize, second->baseAddresses[b], second->allocateSize)) {
                return false;
            }
        }
    }
    return true;
}
} // namespace mlir::pto
#endif
