// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_REGIONALSELECTORPARTITION_H
#define PTO_FRONTIERSYNCH_REGIONALSELECTORPARTITION_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
std::optional<std::vector<SyncStorageCell>> selectorSupportRanges(
    const RegionalAnalysis& region, uint64_t& visits);
std::optional<std::vector<SyncStorageCell>> selectorDomainSupportRanges(
    const RegionalAnalysis& region, AddressSpace space, Value base, uint64_t& visits);
// Uniformity evidence only; unavailable is distinct from an absent footprint.
std::optional<std::vector<uint64_t>> uniformSelectorCuts(
    const RegionalAnalysis& region, SyncStorageCell range, uint64_t& visits);
// Existing bounded per-byte compatibility path when no uniform proof exists.
std::optional<std::vector<uint64_t>> finiteSelectorCuts(
    const RegionalAnalysis& region, SyncStorageCell range, uint64_t& visits);
bool normalizeSelectorSupport(SmallVector<SyncStorageCell>& cells);
std::optional<std::vector<uint64_t>> commonSelectorCuts(
    ArrayRef<RegionalAnalysis> children, SyncStorageCell range, uint64_t& visits);
} // namespace mlir::pto::frontiersynch
#endif
