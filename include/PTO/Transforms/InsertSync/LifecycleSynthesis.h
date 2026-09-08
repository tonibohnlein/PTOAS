// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LIFECYCLESYNTHESIS_H
#define PTO_TRANSFORMS_INSERTSYNC_LIFECYCLESYNTHESIS_H

#include "PTO/Transforms/InsertSync/LifecycleProtocol.h"
#include "PTO/Transforms/InsertSync/StorageFrontierAnalysis.h"
#include "PTO/Transforms/InsertSync/InsertSyncOptions.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::pto {
// Read-only structural import using R5's NativeGraph/guard/residue logic, without
// demanding that an as-yet UNSYNCHRONIZED function already has completion supply.
// No new IR attributes, dialect operations, or caller promises are needed.
struct InsertSyncLifecycleStructure {
    StorageFrontierSnapshot::Status status = StorageFrontierSnapshot::Status::Unsupported;
    std::string reason;
    bool cube = false;
    Operation *lifetimeScope = nullptr;
    insert_sync_frontier::Program program;
    std::vector<Operation *> anchors;
    std::vector<const CompoundInstanceElement *> phases;
};
InsertSyncLifecycleStructure buildInsertSyncLifecycleStructure(
    func::FuncOp function, const SyncIRs &syncIR, insert_sync_frontier::Budget &budget);

class InsertSyncLifecyclePlan {
public:
    struct Slice {
        AddressSpace space = AddressSpace::Zero;
        uint64_t begin = 0, bytes = 0;
        bool operator==(const Slice &other) const {
            return space == other.space && begin == other.begin && bytes == other.bytes;
        }
    };
    struct Channel {
        insert_sync_frontier::LogicalLifecycle logical;
        SmallVector<Slice, 2> members;
    };
    std::vector<Channel> channels;
    llvm::DenseMap<Operation *, unsigned> phaseIds;
    std::vector<Operation *> phases;
    bool cube = false;
    Operation *lifetimeScope = nullptr;
    mutable uint64_t suppliedPairs = 0;

    // Used BEFORE legacy insertion. Removes only exact covered access pairs,
    // never marks a whole pipe complete or hides mixed residual dependencies.
    void removeSuppliedDependencies(CompoundInstanceElement *source,
                                   CompoundInstanceElement *target,
                                   DepBaseMemInfoPairVec &pairs) const;
};

struct InsertSyncLifecycleResult {
    enum class Status { Applied, Unchanged, InputError, InternalError };
    Status status = Status::Unchanged;
    std::string reason;
    unsigned attempted = 0, selected = 0, logicalStreams = 0;
    uint64_t suppliedPairs = 0;
    bool combinedEventAuditProved = false;
};
// Integrated optional construction path. It clones the original, derives
// lifecycles before insertion, lets the EXISTING analyzer repair residuals,
// allocates disjoint concrete keys, emits both plans and verifies the protocol
// delta. Unsupported/capacity outcomes return the original for ordinary insertion.
InsertSyncLifecycleResult tryInsertSyncLifecycleSynthesis(
    func::FuncOp function, const InsertSyncOptions &options);
} // namespace mlir::pto
#endif
