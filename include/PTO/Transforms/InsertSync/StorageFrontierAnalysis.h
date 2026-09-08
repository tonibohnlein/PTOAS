// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERANALYSIS_H
#define PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERANALYSIS_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "PTO/Transforms/InsertSync/StorageFrontierControl.h"
#include "PTO/Transforms/InsertSync/StorageFrontierQueries.h"
#include "PTO/Transforms/InsertSync/BufferGenerationAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncRequirements.h"
#include <string>
#include <cstdint>

namespace mlir::pto {
struct InsertSyncStorageFlow {
    insert_sync_frontier::BufferGenerationFlow facts;
    llvm::DenseMap<Operation*, unsigned> phases;
    std::vector<Operation*> operations;
    std::optional<SmallVector<Operation*>> immediatePredecessors(Operation* target) const {
        auto t = phases.find(target);
        if (facts.status != insert_sync_frontier::BufferGenerationFlow::Status::Complete || t == phases.end())
            return std::nullopt;
        const auto& previous = facts.precedingOnLane[t->second];
        if (previous.empty() || previous.test(operations.size())) return std::nullopt;
        SmallVector<Operation*> result;
        for (unsigned p = 0; p < operations.size(); ++p)
            if (previous.test(p)) result.push_back(operations[p]);
        return result;
    }
    bool provesUnordered(Operation* source, Operation* target) const {
        auto s = phases.find(source), t = phases.find(target);
        return facts.status == insert_sync_frontier::BufferGenerationFlow::Status::Complete &&
               s != phases.end() && t != phases.end() && !facts.orderedPhases[s->second].test(t->second);
    }
};
// Snapshot users must invalidate it after any physical/synchronization/control
// mutation. References point into the supplied function; no concrete IDs are
// inferred from metadata. Unsupported is a query result, never admission policy.
struct StorageFrontierAccessRecord {
    unsigned identity = 0, phase = 0;
    Value handle, root;
    AddressSpace space = AddressSpace::Zero;
    SmallVector<uint64_t> addresses;
    uint64_t upperBoundBytes = 0;
    bool write = false, conservative = true;
};
struct StorageFrontierGuardRecord {
    unsigned identity = 0;
    Value expression;
    Operation* invocationScope = nullptr;
    // A loop-shape variable has no SSA value. Its domain is empty/one/many.
    Operation* tripShapeOf = nullptr;
    std::vector<int64_t> possibleValues;
};
struct StorageFrontierSnapshot {
    enum class Status { Complete, Unsupported, AnalysisLimit, InternalError };
    Status status = Status::Unsupported;
    std::string reason;
    Operation* physicalOwner = nullptr;
    bool cubeContext = false;
    std::vector<StorageFrontierGuardRecord> guardDomains;
    insert_sync_frontier::Program program;
    std::vector<Operation*> anchors;
    std::vector<insert_sync_frontier::GuardEnvironment> guards;
    std::vector<StorageFrontierAccessRecord> accesses;
    std::vector<insert_sync_frontier::RequirementWitness> witnesses;
    std::vector<insert_sync_frontier::Requirement> requirements;
    std::vector<insert_sync_frontier::Atom> atoms;
    struct PhysicalAtom {
        AddressSpace space;
        uint64_t begin, end;
    };
    std::vector<PhysicalAtom> physicalAtoms;
    std::vector<insert_sync_frontier::LaneProjection> lanes;
    insert_sync_frontier::LifecycleResult lifecycle;
    insert_sync_frontier::GenerationFrontiers generations;
    insert_sync_frontier::BufferGenerationFlow storageFlow;
    insert_sync_frontier::CompletionResult supply;
};
// Conservative occurrence and payload checks: this entry never assumes a
// runtime arithmetic guard that is not present in the supplied IR.
StorageFrontierSnapshot analyzeInsertSyncStorageFrontiers(
    func::FuncOp function, const SyncIRs& syncIR, bool useMmadChains, insert_sync_frontier::Budget& budget,
    bool allowSingleSection = false);

struct StorageFrontierRefinementResult {
    unsigned removed = 0;
    unsigned guarded = 0;
    unsigned atoms = 0;
    unsigned requirements = 0;
    unsigned occurrenceProofs = 0;
    unsigned generations = 0;
    unsigned signalsAdvanced = 0;
    unsigned waitsDelayed = 0;
    unsigned boundaryHandoffs = 0;
    unsigned placementAttempts = 0;
    uint64_t work = 0;
    bool internalError = false;
    std::string reason;
};
// Only explicitly owned barriers are candidates. Events and payload stay fixed.
// Reimported physical completion must establish every removed prefix/drain.
StorageFrontierRefinementResult refineInsertSyncCompletion(
    func::FuncOp function, const SyncIRs& syncIR, ArrayRef<Operation*> ownedBarriers,
    insert_sync_frontier::Budget& budget);
bool disjointInsertSyncGlobalOccurrences(const BaseMemInfo* sourceAccess, Operation* source,
                                         const BaseMemInfo* targetAccess, Operation* target,
                                         func::FuncOp function);
bool recheckInsertSyncGenerationRequirements(func::FuncOp function, const SyncRequirements& requirements);
// Uses the existing translated accesses. No new admission gate or speculative
// effect classification. Static events and physical operation order are retained.
// The analysis is optional; unproved/budget outcomes leave the function intact.
// Placement preserves each event key and its dynamic participation, but may
// introduce mutually exclusive static first/last/zero-trip sites.
StorageFrontierRefinementResult refineInsertSyncStorageFrontiers(
    func::FuncOp function, const SyncIRs& syncIR, ArrayRef<Operation*> candidates, bool useMmadChains = false,
    bool placeFrontiers = false, ArrayRef<Operation*> ownedEvents = {}, bool fixedProtocols = false,
    const SyncRequirements* requirements = nullptr);
} // namespace mlir::pto
#endif
