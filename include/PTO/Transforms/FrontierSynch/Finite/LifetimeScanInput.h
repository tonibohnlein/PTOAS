// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Borrowed synchronous input for the unchanged explicit lifetime scan.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LIFETIMESCANINPUT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LIFETIMESCANINPUT_H
#include "PTO/Transforms/FrontierSynch/Finite/LifetimeScan.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>
namespace mlir::pto::frontiersynch {
struct StorageOccurrenceMetadata {
    uint32_t payload = 0;
    uint32_t pipe = 0;
};
using StorageAccessConsumer = llvm::function_ref<bool(const CellAccess&)>;
// All callbacks and their backing owners must remain alive through the scan.
// They are invoked synchronously and never retained. Access modes are still
// consolidated per occurrence; sharing avoids copies, not O(c) incidence work.
struct StorageScanInput {
    std::size_t size;
    llvm::function_ref<StorageOccurrenceMetadata(std::size_t)> occurrence;
    llvm::function_ref<bool(std::size_t, StorageAccessConsumer)> accesses;
};
struct StorageScanWork {
    uint64_t incidenceVisits = 0;
    uint64_t consolidatedVisits = 0;
    uint64_t hazardWitnesses = 0;
    uint64_t retainedWitnesses = 0;
    bool counterOverflow = false;
};
StorageScanResult scanStorageLifetimes(
    const StorageScanInput& input, llvm::ArrayRef<StorageGenerator> prerequisites = {},
    StorageProtectionPolicy protection = {}, bool retainWitnesses = true, StorageScanWork* work = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
