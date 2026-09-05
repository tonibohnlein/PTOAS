// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LocalMemoryLoopFlow.cpp - Sparse dynamic-occurrence requirements ----===//
// A single straight-line scf.for shares atoms with its prefix and suffix.
// Replaying the body twice exposes distance-one outstanding frontiers; it is
// NOT a bounded-trip correctness proof. Boundary relations quantify over every
// executing iteration, including read-only atoms whose readers never retire.
// This analysis is opt-in and never marks recurring accesses covered.

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

enum class Slice { Prefix, Body, Suffix };

struct PhaseEffect {
    SyncPhaseId phase = kInvalidSyncId;
    SyncAccessId read = kInvalidSyncId;
    SyncAccessId write = kInvalidSyncId;
    Slice slice = Slice::Prefix;
};

struct Occurrence {
    SyncAccessId access = kInvalidSyncId;
    Slice slice = Slice::Prefix;
    unsigned iteration = 0;
};

struct Frontier {
    Occurrence writer;
    SmallVector<Occurrence, 4> readers;
};

const SyncRegion* findCarrier(const StructuredSyncIR& schedule)
{
    const SyncRegion* carrier = nullptr;
    for (const SyncRegion& region : schedule.getRegions()) {
        if (region.kind == SyncRegionKind::Choice || region.kind == SyncRegionKind::PhysicalSection) {
            return nullptr;
        }
        if (region.kind != SyncRegionKind::Loop) {
            continue;
        }
        auto loop = dyn_cast_or_null<scf::ForOp>(region.operation);
        const bool invalidLoop = carrier || !loop || loop.getNumRegionIterArgs() != 0 || !region.guard.empty();
        if (invalidLoop) {
            return nullptr;
        }
        IntegerAttr step;
        const bool positiveStep =
            matchPattern(loop.getStep(), m_Constant(&step)) && step.getValue().isStrictlyPositive();
        const bool validParent = loop->getBlock() == &schedule.getFunction().getBody().front();
        if (!positiveStep || !validParent) {
            return nullptr;
        }
        for (Operation& operation : *loop.getBody()) {
            const bool structuredBody = operation.getNumRegions() != 0;
            if (structuredBody) {
                return nullptr;
            }
        }
        carrier = &region;
    }
    return carrier;
}

bool hasSupportedPhases(const StructuredSyncIR& schedule, const SyncRegion& carrier)
{
    auto loop = cast<scf::ForOp>(carrier.operation);
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool ordinary = phase.operation && phase.core == SyncPhysicalCore::Vector && !phase.macroPhase &&
                              phase.guard.empty() && phase.completion == SyncCompletionKind::PhaseEnd;
        if (!ordinary) {
            return false;
        }
        const bool inBody = phase.operation->getBlock() == loop.getBody();
        if (inBody) {
            if (phase.iterationDomain.loops != SmallVector<SyncRegionId, 2>{carrier.id}) {
                return false;
            }
        } else if (phase.operation->getBlock() != loop->getBlock() || !phase.iterationDomain.loops.empty()) {
            return false;
        }
    }
    return true;
}

SyncIterationRelation relationFor(const Occurrence& source, const Occurrence& target, SyncRegionId carrier, bool bypass)
{
    if (bypass && source.slice == Slice::Prefix && target.slice == Slice::Suffix) {
        return {SyncIterationRelationKind::LoopBypass, 0, carrier};
    }
    if (source.slice == Slice::Prefix && target.slice == Slice::Body) {
        return {SyncIterationRelationKind::LoopEntry, 0, carrier};
    }
    if (source.slice == Slice::Body && target.slice == Slice::Suffix) {
        return {SyncIterationRelationKind::LoopExit, 0, carrier};
    }
    if (source.slice == Slice::Body && target.slice == Slice::Body && source.iteration != target.iteration) {
        return {SyncIterationRelationKind::LoopCarried, 1, carrier};
    }
    return {SyncIterationRelationKind::SameIteration, 0};
}

using RequirementKey = std::tuple<SyncObligationKind, SyncAccessId, SyncAccessId, SyncIterationRelationKind>;

class LoopTransfer {
public:
    LoopTransfer(const StructuredSyncIR& schedule, const SyncRegion& carrier, std::size_t maximumEntries)
        : schedule(schedule), carrier(carrier), remaining(maximumEntries)
    {}

    LogicalResult run(const SyncLocalStorageAtom& atom)
    {
        auto loop = cast<scf::ForOp>(carrier.operation);
        std::map<SyncPhaseId, PhaseEffect> effects;
        for (SyncAccessId id : atom.accesses) {
            const SyncAccess* access = schedule.findAccess(id);
            const SyncPhase* phase = access ? schedule.findPhase(access->phase) : nullptr;
            if (!phase || !phase->operation) {
                return failure();
            }
            PhaseEffect& effect = effects[phase->id];
            effect.phase = phase->id;
            effect.slice = phase->operation->getBlock() == loop.getBody() ? Slice::Body :
                           phase->operation->isBeforeInBlock(loop)        ? Slice::Prefix :
                                                                            Slice::Suffix;
            if (access->mode != SyncAccessMode::Write) {
                effect.read = std::min(effect.read, id);
            }
            if (access->mode != SyncAccessMode::Read) {
                effect.write = std::min(effect.write, id);
            }
        }
        Frontier incoming;
        if (failed(transfer(effects, Slice::Prefix, 0, false, atom.id, incoming))) {
            return failure();
        }
        Frontier outgoing = incoming;
        // With a body write, the outstanding writer/readers stabilize after
        // one traversal up to iteration renaming. Without a write, readers
        // stay outstanding and LoopExit quantifies over every iteration.
        for (unsigned iteration = 0; iteration < 2; ++iteration) {
            if (failed(transfer(effects, Slice::Body, iteration, false, atom.id, outgoing))) {
                return failure();
            }
        }
        if (failed(transfer(effects, Slice::Suffix, 0, false, atom.id, outgoing))) {
            return failure();
        }
        return transfer(effects, Slice::Suffix, 0, true, atom.id, incoming);
    }

    SmallVector<SyncResidualObligation, 16> requirements;
    bool limitExceeded = false;

private:
    LogicalResult append(SyncObligationKind kind, Occurrence source, Occurrence target, std::uint32_t atom, bool bypass)
    {
        if (source.access == kInvalidSyncId) {
            return success();
        }
        const SyncIterationRelation relation = relationFor(source, target, carrier.id, bypass);
        const RequirementKey key{kind, source.access, target.access, relation.kind};
        auto found = indices.find(key);
        const bool duplicate = found != indices.end() && llvm::is_contained(requirements[found->second].atoms, atom);
        if (duplicate) {
            return success();
        }
        if (remaining == 0) {
            limitExceeded = true;
            return failure();
        }
        --remaining;
        if (found != indices.end()) {
            requirements[found->second].atoms.push_back(atom);
            return success();
        }
        const bool idsExhausted = requirements.size() >= kInvalidSyncId;
        if (idsExhausted) {
            limitExceeded = true;
            return failure();
        }
        const SyncAccess* first = schedule.findAccess(source.access);
        const SyncAccess* second = schedule.findAccess(target.access);
        if (!first || !second) {
            return failure();
        }
        SyncResidualObligation obligation;
        obligation.id = static_cast<SyncObligationId>(requirements.size());
        obligation.localRequirement = obligation.id;
        obligation.kind = kind;
        obligation.source = first->phase;
        obligation.target = second->phase;
        obligation.sourceAccess = first->id;
        obligation.targetAccess = second->id;
        obligation.control = SyncControlRelation::MustExecute;
        obligation.iteration = relation;
        obligation.precision = SyncRegionPrecision::Conservative;
        obligation.atoms.push_back(atom);
        obligation.detail = "canonical single-loop outstanding-access ordering; diagnostic-only";
        indices.emplace(key, obligation.id);
        requirements.push_back(std::move(obligation));
        return success();
    }

    LogicalResult transfer(
        const std::map<SyncPhaseId, PhaseEffect>& effects, Slice slice, unsigned iteration, bool bypass,
        std::uint32_t atom, Frontier& frontier)
    {
        for (const auto& entry : effects) {
            const PhaseEffect& effect = entry.second;
            if (effect.slice != slice) {
                continue;
            }
            const bool writes = effect.write != kInvalidSyncId;
            Occurrence target{writes ? effect.write : effect.read, slice, iteration};
            if (failed(append(SyncObligationKind::Completion, frontier.writer, target, atom, bypass))) {
                return failure();
            }
            if (writes) {
                for (Occurrence reader : frontier.readers) {
                    if (failed(append(SyncObligationKind::Reclamation, reader, target, atom, bypass))) {
                        return failure();
                    }
                }
                frontier.writer = target;
                frontier.readers.clear();
            } else {
                frontier.readers.push_back(target);
            }
        }
        return success();
    }

    const StructuredSyncIR& schedule;
    const SyncRegion& carrier;
    std::size_t remaining;
    std::map<RequirementKey, unsigned> indices;
};

} // namespace

LogicalResult mlir::pto::protocol_sync::analyzeLocalLoopFlow(
    const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result, std::size_t maximumEntries)
{
    const bool invalidInput =
        !schedule.isFrozen() || !result.requirements.empty() || !result.states.empty() || result.coveredAccesses.any();
    if (invalidInput) {
        return failure();
    }
    result.loopStatus = SyncLocalLoopStatus::Unsupported;
    result.boundary = "single-loop occurrence analysis does not authorize recurring emission";
    if (!llvm::hasSingleElement(schedule.getFunction().getBody())) {
        return success();
    }
    const SyncRegion* carrier = findCarrier(schedule);
    const bool supported = carrier && hasSupportedPhases(schedule, *carrier) && schedule.getFailures().empty();
    if (!supported) {
        return success();
    }
    LoopTransfer transfer(schedule, *carrier, maximumEntries);
    for (const SyncLocalStorageAtom& atom : result.atoms) {
        if (failed(transfer.run(atom))) {
            if (!transfer.limitExceeded) {
                return failure();
            }
            result.loopStatus = SyncLocalLoopStatus::LimitExceeded;
            return success();
        }
    }
    result.loopCarrier = carrier->id;
    result.requirements = std::move(transfer.requirements);
    result.loopStatus = SyncLocalLoopStatus::Complete;
    return success();
}
