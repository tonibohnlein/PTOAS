// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LocalMemoryRegionFlow.cpp - Compositional per-atom transfer ----------===//

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool sameGuard(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& lhs, const auto& rhs) {
               return lhs.choice == rhs.choice && lhs.arm == rhs.arm;
           });
}

struct PhaseAccess {
    SyncPhaseId phase = kInvalidSyncId;
    SyncAccessId read = kInvalidSyncId;
    SyncAccessId write = kInvalidSyncId;
};

using RequirementKey = std::tuple<SyncObligationKind, SyncAccessId, SyncAccessId>;

LogicalResult appendRequirement(
    const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result, std::map<RequirementKey, unsigned>& indices,
    SyncObligationKind kind, SyncAccessId source, SyncAccessId target, std::uint32_t atom)
{
    if (source == kInvalidSyncId) {
        return success();
    }
    const auto key = std::make_tuple(kind, source, target);
    auto found = indices.find(key);
    if (found != indices.end()) {
        result.requirements[found->second].atoms.push_back(atom);
        return success();
    }
    const bool requirementIdsExhausted = result.requirements.size() >= kInvalidSyncId;
    if (requirementIdsExhausted) {
        return failure();
    }
    const SyncAccess* sourceAccess = schedule.findAccess(source);
    const SyncAccess* targetAccess = schedule.findAccess(target);
    if (!sourceAccess || !targetAccess) {
        return failure();
    }
    const SyncPhase* phase = schedule.findPhase(sourceAccess->phase);
    if (!phase) {
        return failure();
    }
    SyncResidualObligation obligation;
    obligation.id = static_cast<SyncObligationId>(result.requirements.size());
    obligation.localRequirement = obligation.id;
    obligation.kind = kind;
    obligation.source = sourceAccess->phase;
    obligation.target = targetAccess->phase;
    obligation.sourceAccess = source;
    obligation.targetAccess = target;
    obligation.atoms.push_back(atom);
    obligation.precision = SyncRegionPrecision::Conservative;
    const SyncPhase* targetPhase = schedule.findPhase(targetAccess->phase);
    if (!targetPhase) {
        return failure();
    }
    obligation.control = !sameGuard(phase->guard, targetPhase->guard) ? SyncControlRelation::Unknown :
                         phase->guard.empty()                         ? SyncControlRelation::MustExecute :
                                                                        SyncControlRelation::SameGuard;
    obligation.iteration = {SyncIterationRelationKind::SameIteration, 0};
    obligation.detail = kind == SyncObligationKind::Reclamation ? "canonical local outstanding read before overwrite" :
                                                                  "canonical local outstanding write before access";
    indices.emplace(key, obligation.id);
    result.requirements.push_back(std::move(obligation));
    return success();
}

template <typename T>
void unionInto(SmallVectorImpl<T>& target, ArrayRef<T> source)
{
    for (const T& value : source) {
        if (!llvm::is_contained(target, value)) {
            target.push_back(value);
        }
    }
    llvm::sort(target);
}

SyncLocalAtomFlow joinFlows(ArrayRef<SyncLocalAtomFlow> alternatives)
{
    SyncLocalAtomFlow joined;
    if (alternatives.empty()) {
        return joined;
    }
    joined.mustDefinitions = alternatives.front().mustDefinitions;
    joined.mayHaveLiveIn = false;
    for (const SyncLocalAtomFlow& flow : alternatives) {
        joined.mayHaveLiveIn |= flow.mayHaveLiveIn;
        unionInto(joined.mayDefinitions, ArrayRef(flow.mayDefinitions));
        unionInto(joined.outstandingWrites, ArrayRef(flow.outstandingWrites));
        unionInto(joined.outstandingReaders, ArrayRef(flow.outstandingReaders));
        llvm::erase_if(
            joined.mustDefinitions, [&](std::uint32_t id) { return !llvm::is_contained(flow.mustDefinitions, id); });
    }
    return joined;
}

class AtomRegionTransfer {
public:
    AtomRegionTransfer(
        const StructuredSyncIR& schedule, const SyncLocalStorageAtom& atom, SyncLocalMemoryAnalysis& result,
        std::map<RequirementKey, unsigned>& indices, std::size_t& stateEntries, SyncLocalFlowOptions options,
        bool& limitExceeded)
        : schedule(schedule),
          atom(atom),
          result(result),
          indices(indices),
          stateEntries(stateEntries),
          options(options),
          limitExceeded(limitExceeded)
    {}

    LogicalResult run()
    {
        for (SyncAccessId id : atom.accesses) {
            const SyncAccess* access = schedule.findAccess(id);
            if (!access) {
                return failure();
            }
            PhaseAccess& phase = phases[access->phase];
            phase.phase = access->phase;
            const SyncPhase* physical = schedule.findPhase(access->phase);
            if (!physical) {
                return failure();
            }
            const SyncRegion* owner = schedule.findRegion(physical->region);
            while (owner && relevantRegions.insert(owner->id).second) {
                owner = schedule.findRegion(owner->parent);
            }
            if (access->mode != SyncAccessMode::Write) {
                phase.read = std::min(phase.read, id);
            }
            if (access->mode != SyncAccessMode::Read) {
                phase.write = std::min(phase.write, id);
            }
        }
        SyncLocalAtomFlow incoming;
        if (!options.expandState) {
            // Keep the established linear transfer sparse. Expanded histories
            // and region recursion are not needed for production requirements.
            for (const auto& entry : phases) {
                if (failed(transferPhase(entry.second, incoming))) {
                    return failure();
                }
            }
            return success();
        }
        if (schedule.getRegions().empty()) {
            return failure();
        }
        return success(succeeded(analyzeRegion(schedule.getRegions().front(), incoming, 0)));
    }

private:
    bool charge(std::size_t entries)
    {
        if (!options.expandState) {
            return true;
        }
        if (entries > options.maximumExpandedEntries - stateEntries) {
            limitExceeded = true;
            return false;
        }
        stateEntries += entries;
        return true;
    }

    bool chargeFlow(const SyncLocalAtomFlow& flow)
    {
        return charge(
            flow.mayDefinitions.size() + flow.mustDefinitions.size() + flow.outstandingWrites.size() +
            flow.outstandingReaders.size() + 1);
    }

    LogicalResult require(SyncObligationKind kind, SyncAccessId source, SyncAccessId target)
    {
        if (failed(appendRequirement(schedule, result, indices, kind, source, target, atom.id))) {
            return failure();
        }
        auto found = indices.find({kind, source, target});
        if (found == indices.end()) {
            return failure();
        }
        touchedRequirements.push_back(found->second);
        return success();
    }

    LogicalResult transferPhase(const PhaseAccess& access, SyncLocalAtomFlow& flow)
    {
        const bool writes = access.write != kInvalidSyncId;
        const SyncAccessId target = writes ? access.write : access.read;
        for (SyncAccessId writer : flow.outstandingWrites) {
            if (failed(require(SyncObligationKind::Completion, writer, target))) {
                return failure();
            }
        }
        if (writes) {
            for (SyncAccessId reader : flow.outstandingReaders) {
                if (failed(require(SyncObligationKind::Reclamation, reader, access.write))) {
                    return failure();
                }
            }
        }
        const SyncPhase* phase = schedule.findPhase(access.phase);
        // Bound the explicit may-definition snapshots as well as recursion.
        const std::size_t entries = flow.mayDefinitions.size() + flow.mustDefinitions.size() + 1;
        const bool withinBudget = charge(entries);
        const bool invalidState = !phase || !withinBudget || result.states.size() >= kInvalidSyncId;
        if (invalidState) {
            return failure();
        }
        SyncLocalMemoryState state;
        state.id = static_cast<std::uint32_t>(result.states.size());
        state.atom = atom.id;
        state.phase = access.phase;
        const bool singleFrontier = flow.outstandingWrites.size() == 1 && !flow.mayDefinitions.empty();
        state.previousDefinition =
            options.expandState ? (singleFrontier ? flow.mayDefinitions.back() : kInvalidSyncId) : previousDefinition;
        state.reads = access.read != kInvalidSyncId;
        state.writes = writes;
        state.precision = SyncRegionPrecision::Conservative;
        state.mayDefinitions = flow.mayDefinitions;
        state.mustDefinitions = flow.mustDefinitions;
        state.region = phase->region;
        state.guard = phase->guard;
        state.iterationDomain = phase->iterationDomain;
        state.mayHaveLiveIn = flow.mayHaveLiveIn;
        if (writes) {
            if (options.expandState) {
                flow.mayDefinitions.push_back(state.id);
            }
            previousDefinition = state.id;
            flow.outstandingWrites.assign(1, access.write);
            flow.outstandingReaders.clear();
        } else {
            flow.outstandingReaders.push_back(access.read);
        }
        result.states.push_back(std::move(state));
        return success();
    }

    FailureOr<SyncLocalAtomFlow> analyzeRegion(
        const SyncRegion& region, const SyncLocalAtomFlow& incoming, unsigned depth)
    {
        const bool snapshotBudget = chargeFlow(incoming);
        const bool transferBudget = chargeFlow(incoming);
        if (!snapshotBudget || !transferBudget) {
            return failure();
        }
        SyncLocalRegionSummary summary;
        summary.region = region.id;
        summary.atom = atom.id;
        summary.incoming = incoming;
        if (!relevantRegions.contains(region.id)) {
            summary.outgoing = incoming;
            summary.complete = true;
            result.regionSummaries.push_back(std::move(summary));
            return incoming;
        }
        if (depth >= 64) {
            limitExceeded = true;
            return failure();
        }
        if (region.kind == SyncRegionKind::Loop) {
            return failure();
        }
        const std::size_t firstRequirement = touchedRequirements.size();
        SyncLocalAtomFlow outgoing = incoming;
        SmallVector<SyncLocalAtomFlow, 2> alternatives;
        bool sawWrite = false;
        bool allAlternativesWrite = !region.elements.empty();
        for (const SyncRegionElement& element : region.elements) {
            if (element.kind == SyncRegionElement::Kind::Phase) {
                auto found = phases.find(element.phase);
                if (found == phases.end()) {
                    continue;
                }
                const PhaseAccess& access = found->second;
                if (!sawWrite) {
                    summary.firstAccesses.push_back(access.write != kInvalidSyncId ? access.write : access.read);
                    sawWrite = access.write != kInvalidSyncId;
                }
                if (failed(transferPhase(access, outgoing))) {
                    return failure();
                }
            } else if (element.kind == SyncRegionElement::Kind::ChildRegion) {
                const SyncRegion* child = schedule.findRegion(element.child);
                if (!child) {
                    return failure();
                }
                const bool choice = region.kind == SyncRegionKind::Choice;
                auto transferred = analyzeRegion(*child, choice ? incoming : outgoing, depth + 1);
                if (failed(transferred)) {
                    return failure();
                }
                const auto& childSummary = result.regionSummaries.back();
                if (!sawWrite || choice) {
                    unionInto(summary.firstAccesses, ArrayRef(childSummary.firstAccesses));
                }
                if (choice) {
                    allAlternativesWrite &= childSummary.writesOnEveryPath;
                    alternatives.push_back(std::move(*transferred));
                } else {
                    sawWrite |= childSummary.writesOnEveryPath;
                    outgoing = std::move(*transferred);
                }
            }
        }
        if (region.kind == SyncRegionKind::Choice) {
            outgoing = joinFlows(alternatives);
            sawWrite = allAlternativesWrite;
        }
        if (!chargeFlow(outgoing)) {
            return failure();
        }
        summary.outgoing = outgoing;
        summary.writesOnEveryPath = sawWrite;
        unionInto(summary.requirements, ArrayRef(touchedRequirements).drop_front(firstRequirement));
        summary.complete = true;
        result.regionSummaries.push_back(std::move(summary));
        return outgoing;
    }

    const StructuredSyncIR& schedule;
    const SyncLocalStorageAtom& atom;
    SyncLocalMemoryAnalysis& result;
    std::map<RequirementKey, unsigned>& indices;
    std::map<SyncPhaseId, PhaseAccess> phases;
    llvm::DenseSet<SyncRegionId> relevantRegions;
    SmallVector<SyncObligationId, 16> touchedRequirements;
    std::size_t& stateEntries;
    SyncLocalFlowOptions options;
    bool& limitExceeded;
    std::uint32_t previousDefinition = kInvalidSyncId;
};

} // namespace

LogicalResult mlir::pto::protocol_sync::analyzeLocalRegionFlow(
    const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result, SyncLocalFlowOptions options)
{
    const bool freshTransfer = result.states.empty() && result.regionSummaries.empty() && result.requirements.empty();
    const bool validInput = schedule.isFrozen() && freshTransfer;
    if (!validInput) {
        return failure();
    }
    std::map<RequirementKey, unsigned> indices;
    std::size_t stateEntries = 0;
    bool limitExceeded = false;
    SyncLocalMemoryAnalysis transferred;
    for (const SyncLocalStorageAtom& atom : result.atoms) {
        if (failed(
                AtomRegionTransfer(schedule, atom, transferred, indices, stateEntries, options, limitExceeded).run())) {
            if (limitExceeded) {
                result.expandedStateStatus = SyncLocalExpandedStateStatus::LimitExceeded;
                return success();
            }
            return failure();
        }
    }
    result.states = std::move(transferred.states);
    result.regionSummaries = std::move(transferred.regionSummaries);
    result.requirements = std::move(transferred.requirements);
    result.expandedStateStatus =
        options.expandState ? SyncLocalExpandedStateStatus::Complete : SyncLocalExpandedStateStatus::NotRequested;
    return success();
}
