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

#include "PTO/Transforms/InsertSync/LifecycleBoundaryProtocol.h"
#include "PTO/Transforms/InsertSync/LifecycleCompletion.h"
#include "PTO/Transforms/InsertSync/StorageFrontierAnalysis.h"
#include "PTO/Transforms/InsertSync/InsertSyncOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::pto {
struct LifecycleIterationClass {
    Operation *loop = nullptr;
    bool first = false, last = false;
};
struct InsertSyncLifecycleStructure {
    StorageFrontierSnapshot::Status status = StorageFrontierSnapshot::Status::Unsupported;
    std::string reason;
    bool cube = false;
    Operation *lifetimeScope = nullptr;
    insert_sync_frontier::Program program;
    std::vector<Operation *> anchors;
    std::vector<const CompoundInstanceElement *> phases;
    // Actual R5 guard-product bindings. No generated/input attribute is trusted
    // as a predicate. Entries are indexed by the same guarded Program node.
    std::vector<insert_sync_frontier::GuardEnvironment> guards;
    std::vector<StorageFrontierGuardRecord> guardDomains;
    std::vector<std::vector<LifecycleIterationClass>> iterations;
    std::vector<Operation *> loopExits;
    std::vector<Block *> blockExits;
};
InsertSyncLifecycleStructure buildInsertSyncLifecycleStructure(
    func::FuncOp function, const SyncIRs &syncIR, insert_sync_frontier::Budget &budget);

struct LifecycleGuardTest {
    enum class Kind { First, Last, Empty, ValueEquals };
    Kind kind = Kind::Empty;
    Operation *loop = nullptr;
    Value expression;
    int64_t value = 0;
    unsigned variable = insert_sync_frontier::kInvalid;
};
struct LifecyclePlacement {
    enum class Role { AcquireFree, PublishReady, AcquireReady, PublishFree, BypassReady };
    Role role = Role::AcquireFree;
    Operation *anchor = nullptr;
    Block *blockEnd = nullptr;
    bool after = false;
    std::vector<LifecycleGuardTest> tests;
    insert_sync_frontier::GuardedRole guard;
};
struct LifecycleDiagnostic {
    std::string identity;
    std::string stage;
    std::string reason;
    bool accepted = false;
};
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
        std::string identity;
        std::vector<LifecyclePlacement> placements;
        unsigned guardedActions = 0, consumerRegions = 0;
    };
    std::vector<Channel> channels;
    llvm::DenseMap<Operation *, unsigned> phaseIds;
    std::vector<Operation *> phases;
    bool cube = false;
    Operation *lifetimeScope = nullptr;
    mutable uint64_t suppliedPairs = 0;
    // A read-only selected-plan projection, initialized before residual insertion.
    // It contains no candidate barriers and no concrete hardware event IDs.
    insert_sync_frontier::Program completionStructure;
    struct CompletionRequirement {
        Operation *source = nullptr, *target = nullptr;
        Value sourceAccess, targetAccess;
    };
    mutable insert_sync_frontier::LifecycleCompletionSupply completionSupply;
    mutable std::vector<CompletionRequirement> completionRequirements;
    mutable std::set<std::pair<unsigned, unsigned>> completionRequirementKeys;
    mutable uint64_t completionSuppliedPairs = 0, completionWork = 0;
    mutable unsigned importedResidualHandoffs = 0;
    mutable bool completionInternalError = false;
    mutable std::string completionReason;
    // Called at the start of each repair stage, before new same-pipe repairs.
    // The staged walk can include supported cross-pipe residual handoffs from
    // its first walk; the combined walk still sees the complete protocol supply.
    void refreshCompletionSupply(const SyncIRs& ir, const SyncOperations& syncs) const;
    void removeSuppliedDependencies(CompoundInstanceElement *source,
                                   CompoundInstanceElement *target,
                                   DepBaseMemInfoPairVec &pairs) const;
};

std::optional<bool> classifyInsertSyncLifecycleContinuation(
    Value condition, scf::ForOp loop, bool first, bool last);

bool qualifyInsertSyncLifecycleBoundaries(
    const InsertSyncLifecycleStructure &structure, InsertSyncLifecyclePlan::Channel &channel,
    func::FuncOp function, insert_sync_frontier::Budget &budget, std::string &reason);

struct InsertSyncLifecycleResult {
    enum class Status { Applied, Unchanged, InputError, InternalError };
    Status status = Status::Unchanged;
    std::string reason;
    unsigned attempted = 0, selected = 0, logicalStreams = 0;
    unsigned planningAttempts = 0, retries = 0, guardedActions = 0, consumerRegions = 0;
    uint64_t suppliedPairs = 0;
    bool combinedEventAuditProved = false;
    std::vector<LifecycleDiagnostic> diagnostics;
};
InsertSyncLifecycleResult tryInsertSyncLifecycleSynthesis(
    func::FuncOp function, const InsertSyncOptions &options);
} // namespace mlir::pto
#endif
