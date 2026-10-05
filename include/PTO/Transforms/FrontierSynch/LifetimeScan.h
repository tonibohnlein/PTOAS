// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sparse storage generators from explicit effects in reference order.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LIFETIMESCAN_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LIFETIMESCAN_H
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
struct CellAccess {
    uint32_t atom = 0;
    bool read = false;
    bool write = false;
    uint64_t protectionGroup = 0; // Certified same-pipe writer group; zero is unprotected.
};
struct ExplicitEffects {
    uint32_t payload = 0;
    uint32_t pipe = 0;
    std::vector<CellAccess> accesses;
};
struct StorageGenerator {
    uint32_t source = 0;
    uint32_t target = 0;
};
enum class StorageHazard { RAW, WAR, WAW, Supplied };
struct StorageWitness {
    uint32_t generator = 0;
    uint32_t atom = 0; // Unused for Supplied prerequisites.
    StorageHazard hazard = StorageHazard::RAW;
};
struct StorageScanResult {
    std::string error;
    std::vector<StorageGenerator> generators;
    std::vector<StorageWitness> witnesses;
    uint64_t protectedHazards = 0;
};
// IDs must be unique; input order is reference order. Modes for the same atom
// are consolidated before reading old state, including read-modify-write.
// Native completion order on each pipe justifies retaining its latest reader.
// Nonzero protection groups must certify hardware protection for EVERY ordered
// pair of their writers to each grouped atom. This is a caller-supplied proof,
// not an arbitrary pairwise exemption. IDs are local to one scan: remap them
// before merging independently built inputs or repeating a dynamic scope.
StorageScanResult scanStorageLifetimes(llvm::ArrayRef<ExplicitEffects> occurrences,
                                      llvm::ArrayRef<StorageGenerator> prerequisites = {});
} // namespace mlir::pto::frontiersynch
#endif
