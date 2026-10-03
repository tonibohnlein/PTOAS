// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared target identity and qualification, borrowed from original MLIR context.
// Native order and effects come from shared SyncIR; this stores placement
// context, source identities and event-pool resources for physical emission.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCTARGETINFO_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCTARGETINFO_H
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include <utility>
#include <string>
#include <memory>
#include <optional>
namespace mlir::pto {
class MemoryDependentAnalyzer;
// Classic EventID identity includes physical core and the directed template.
// Equal numeric IDs in different directions are distinct, not an aggregate pool.
struct SyncEventPool {
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE source = PIPE::PIPE_S;
    PIPE target = PIPE::PIPE_S;
    SmallVector<unsigned> eligibleIds;
    SmallVector<unsigned> reservedIds;
    std::string namespaceSource;
};
// Original mechanism provenance, borrowed from unchanged shared source IR.
// Native completion proof uses the original drain alone; readiness/release
// memory pairs separately certify its stronger ordering as model-redundant.
enum class SyncBarrierRequirementDomain { Unqualified, SharedUnion };
// Shared original scalar guard provenance. Comparison leaves are an immutable
// Index argument and typed constant; no translated Boolean expression is stored.
enum class SyncActivationOrigin { None, Argument, ComparisonResult };
struct SyncActivationWitness {
    SyncActivationOrigin origin = SyncActivationOrigin::None;
    Value argument;
    Operation* comparison = nullptr;
    Operation* constantSource = nullptr;
    IntegerAttr constant;
    int64_t predicate = -1;
    bool argumentFirst = true;
};
// An arm is a view of the original predicate, not a separately negated guard.
struct SyncParticipationArm {
    Region* region = nullptr;
    bool outcome = true;
};
// One immutable shared capture independent of any native barrier certificate.
// Raw handles borrow unchanged source IR; copied target records share this owner.
struct SyncParticipation {
    Operation* owner = nullptr;
    Value condition;
    SyncActivationWitness witness;
    std::optional<SyncParticipationArm> armFor(Operation* child) const;
    bool thenOnly() const;
};
enum class SyncBarrierHazard { ReadAfterWrite, WriteAfterRead, WriteAfterWrite };
struct SyncBarrierLexicalWitness {
    const CompoundInstanceElement* source = nullptr;
    const CompoundInstanceElement* consumer = nullptr;
    SyncBarrierHazard kind = SyncBarrierHazard::ReadAfterWrite;
    SmallVector<std::pair<const BaseMemInfo*, const BaseMemInfo*>> pairs;
};
// Original rectangular occurrence frames, with invariant Index bounds and
// pure scalar-only nesting. Neither routine recovers memory effects.
FailureOr<SmallVector<Operation*>> recoverSyncLoopFrames(func::FuncOp function, Operation* innermost);
// Optional work count records unique original SSA values, including rejected leaves.
bool syncFramesNonempty(Value condition, ArrayRef<Operation*> frames, unsigned* visitedValues = nullptr);
struct SyncOriginalBarrierChain {
    Operation* barrier = nullptr;
    Operation* loop = nullptr;
    SmallVector<Operation*> frames;
    const CompoundInstanceElement* producer = nullptr;
    const CompoundInstanceElement* consumer = nullptr;
    // consumer is the actual first compute; releaseProducer is the actual last.
    const CompoundInstanceElement* releaseProducer = nullptr;
    SmallVector<const CompoundInstanceElement*> computes;
    SmallVector<SyncBarrierLexicalWitness> lexical;
    SmallVector<std::pair<const BaseMemInfo*, const BaseMemInfo*>> ready, release, finalReads;
    SyncBarrierRequirementDomain requirementDomain = SyncBarrierRequirementDomain::Unqualified;
    // Optional once-executed same-pipe prefix. Its complete shared effects
    // remain in the invocation; first source barrier drains this occurrence.
    const CompoundInstanceElement* prefix = nullptr;
    SmallVector<std::pair<const BaseMemInfo*, const BaseMemInfo*>> prefixWrites;
    // Common original participation, never a generated flag condition.
    std::shared_ptr<const SyncParticipation> participationContext;

};
struct SyncCoreContext {
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    SmallVector<Region*> regions;
};
SyncCoreContext recoverSyncCoreContext(Operation* anchor);
struct SyncPhaseTarget {
    const CompoundInstanceElement* phase = nullptr;
    SyncCoreContext context;
    unsigned anchorPhaseCount = 0;
    // Only external MLIR cuts are represented. Internal macro cuts are unmet.
    Operation* before = nullptr;
    Operation* after = nullptr;
    SmallVector<SyncMacroHiddenEvent> hiddenEvents;
};
// Original straight-line drain, including its unavoidable C->I baseline.
// This proves native C order independently of any selected/generated handoff.
struct SyncFiniteDrain {
    Operation* barrier = nullptr;
    const CompoundInstanceElement* previous = nullptr;
    const CompoundInstanceElement* consumer = nullptr;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
};
class SyncTargetInfo {
public:
    // All source operations/regions/values and shared phases must remain unchanged
    // and alive. Attribute serialization contains stable IDs, never raw pointers.
    void reset();
    void build(func::FuncOp function, ArrayRef<const CompoundInstanceElement*> phases,
               const MemoryDependentAnalyzer& memory);
    ArrayRef<SyncPhaseTarget> phases() const { return records; }
    ArrayRef<SyncFiniteDrain> finiteDrains() const { return originalFiniteDrains; }
    const SyncOriginalBarrierChain& originalBarrierChain() const { return originalChain; }
    // Original H0 namespace; no traversal or generated/source ID substitution.
    std::optional<int64_t> sourceIdentity(Operation* op) const;
    DictionaryAttr attribute() const;
    DictionaryAttr activationAttribute() const;
    const SyncParticipation& participation() const
    {
        static const SyncParticipation empty;
        return commonParticipation ? *commonParticipation : empty;
    }
    // Detached speculative clones use original enclosing source context; never
    // reconstruct missing module/core facts from pipeline names or fake attrs.
    LogicalResult preflightMechanisms(func::FuncOp pending, const IRMapping& mapping, std::string& reason) const;
    SyncMechanismFact eventFact(SyncPhysicalCore core, PIPE source, PIPE target) const;
    SyncMechanismFact barrierFact(SyncPhysicalCore core, PIPE pipe) const;
    // Classic static A2/A3 event IDs, excluding shared hidden macro usage.
    // Unsupported target mechanisms are distinct from event-capacity exhaustion.
    FailureOr<SyncEventPool> eventPool(
        SyncPhysicalCore core, PIPE source, PIPE target, std::string& reason) const;
    FailureOr<SmallVector<unsigned>> eligibleEventIds(
        SyncPhysicalCore core, PIPE source, PIPE target, std::string& reason) const;
    // Separate from frontier site/loop/value namespaces: explicit MLIR postorder.
    LogicalResult retainSourceIds(func::FuncOp clone, const IRMapping& mapping) const;
    bool valueAvailable(Value value, Operation* anchor, bool after) const;
    StringRef architecture() const { return arch; }

private:
    SmallVector<SyncFiniteDrain> originalFiniteDrains;
    SyncOriginalBarrierChain originalChain;
    std::shared_ptr<const SyncParticipation> commonParticipation;
    func::FuncOp function;
    DenseMap<Operation*, int64_t> sourceIds;
    std::string arch;
    SmallVector<SyncPhaseTarget, 0> records;
    mutable DominanceInfo dominance;
};
} // namespace mlir::pto
#endif
