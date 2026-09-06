// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCPLANNINGPRIMITIVES_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCPLANNINGPRIMITIVES_H

#include <cstdint>
#include <limits>

namespace mlir::pto::insert_sync_detail {

/// These stages control dependency *insertion*, not physical execution order.
/// Both stages query the same already-built logical handoffs. Neither assigns
/// event IDs or moves an instruction. Combined retains the legacy visit order.
enum class RepairStage : std::uint8_t { Combined, CrossPipe, SamePipe };

constexpr bool shouldAnalyzePair(RepairStage stage, bool samePipe) {
  return stage == RepairStage::Combined ||
         (stage == RepairStage::SamePipe ? samePipe : !samePipe);
}

template <typename Callback>
void forEachRepairStage(bool deferSamePipe, Callback &&callback) {
  if (deferSamePipe) {
    callback(RepairStage::CrossPipe);
    callback(RepairStage::SamePipe);
  } else {
    callback(RepairStage::Combined);
  }
}

/// On failure the output is left unchanged. Address overflow is unknown
/// provenance; wrapping it to a small integer must never prove disjointness.
inline bool checkedAddressAdd(std::uint64_t base, std::uint64_t offset,
                              std::uint64_t &result) {
  if (offset > std::numeric_limits<std::uint64_t>::max() - base)
    return false;
  result = base + offset;
  return true;
}

/// BaseMemInfo uses a zero allocation size for an unknown extent. Thus this
/// helper is intentionally NOT a generic mathematical empty-interval predicate.
/// A nonrepresentable exclusive end also conservatively returns may-alias.
inline bool addressRangesMayOverlap(std::uint64_t aStart, std::uint64_t aSize,
                                   std::uint64_t bStart, std::uint64_t bSize) {
  if (aSize == 0 || bSize == 0)
    return true;
  std::uint64_t aEnd = 0;
  std::uint64_t bEnd = 0;
  if (!checkedAddressAdd(aStart, aSize, aEnd) ||
      !checkedAddressAdd(bStart, bSize, bEnd))
    return true;
  return aStart < bEnd && bStart < aEnd;
}

} // namespace mlir::pto::insert_sync_detail

#endif // PTO_TRANSFORMS_INSERTSYNC_SYNCPLANNINGPRIMITIVES_H
