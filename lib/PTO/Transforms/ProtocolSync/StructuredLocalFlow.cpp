// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredLocalFlow.cpp - Guarded conservative outstanding histories ===//
// Shared physical atoms; branch arms receive identical incoming state and join
// by union. Conservative footprints never kill incoming generations. This first
// native choice baseline retains all relevant histories, not sparse retirement.
// Two symbolic loop passes expose every body/body relation at ANY positive
// distance, including skipped writes. This is not a distance-one assumption.
#include "PTO/Transforms/ProtocolSync/StructuredFrontier.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
struct Reference {
    SyncAccessId access;
    bool carried = false;
    bool operator==(const Reference& other) const { return access == other.access && carried == other.carried; }
};
struct History {
    SmallVector<Reference, 8> effects;
    SmallVector<std::uint32_t, 8> definitions;
};
template <typename T>
void merge(SmallVectorImpl<T>& a, ArrayRef<T> b)
{
    for (const auto& value : b) {
        if (!llvm::is_contained(a, value)) {
            a.push_back(value);
        }
    }
}
void merge(History& a, const History& b)
{
    merge(a.effects, ArrayRef(b.effects));
    merge(a.definitions, ArrayRef(b.definitions));
}
using Key = std::tuple<SyncObligationKind, SyncAccessId, SyncAccessId, SyncIterationRelationKind, SyncRegionId>;

class Transfer {
public:
    Transfer(const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result, std::size_t budget)
        : schedule(schedule), result(result), remaining(budget)
    {}

    bool run(const SyncLocalStorageAtom& storage)
    {
        atom = &storage;
        states.clear();
        History incoming;
        return region(schedule.getRegions().front(), incoming, 0);
    }
    bool limited = false;

private:
    bool charge(std::size_t count)
    {
        if (count > remaining) {
            limited = true;
            return false;
        }
        remaining -= count;
        return true;
    }
    SyncLocalAtomFlow snapshot(const History& history)
    {
        SyncLocalAtomFlow flow;
        flow.mayDefinitions.assign(history.definitions.begin(), history.definitions.end());
        for (const auto& ref : history.effects) {
            const auto* access = schedule.findAccess(ref.access);
            if (access->mode != SyncAccessMode::Read) {
                merge(flow.outstandingWrites, ArrayRef<SyncAccessId>(access->id));
            }
            if (access->mode != SyncAccessMode::Write) {
                merge(flow.outstandingReaders, ArrayRef<SyncAccessId>(access->id));
            }
        }
        return flow;
    }
    bool requirement(const Reference& prior, const SyncAccess& access)
    {
        if (!charge(1)) {
            return false;
        }
        const auto* source = schedule.findAccess(prior.access);
        const auto* a = schedule.findPhase(source->phase);
        const auto* b = schedule.findPhase(access.phase);
        const bool noHazard = (!prior.carried && a->id == b->id) ||
                              (source->mode == SyncAccessMode::Read && access.mode == SyncAccessMode::Read);
        if (noHazard) {
            return true;
        }
        SyncIterationRelation relation = structuredIterationRelation(*a, *b);
        const bool carried =
            prior.carried && !b->iterationDomain.loops.empty() && a->iterationDomain.loops == b->iterationDomain.loops;
        if (carried) {
            relation = {SyncIterationRelationKind::LoopCarriedAny, 0, b->iterationDomain.loops.front()};
        }
        const auto participation = classifySyncParticipation(*a, *b, relation);
        if (participation == SyncParticipationRelation::MutuallyExclusive) {
            return true;
        }
        if (participation == SyncParticipationRelation::Unknown) {
            return false;
        }
        const auto kind =
            source->mode == SyncAccessMode::Read ? SyncObligationKind::Reclamation : SyncObligationKind::Completion;
        const Key key{kind, source->id, access.id, relation.kind, relation.carrier};
        auto found = indices.find(key);
        if (found != indices.end()) {
            auto& masks = result.requirements[found->second].atoms;
            if (!llvm::is_contained(masks, atom->id)) {
                masks.push_back(atom->id);
            }
            return true;
        }
        const bool exhaustedRequirements = result.requirements.size() >= kInvalidSyncId;
        if (exhaustedRequirements) {
            return false;
        }
        SyncResidualObligation obligation;
        obligation.id = result.requirements.size();
        obligation.localRequirement = obligation.id;
        obligation.kind = kind;
        obligation.source = a->id;
        obligation.target = b->id;
        obligation.sourceAccess = source->id;
        obligation.targetAccess = access.id;
        obligation.atoms.push_back(atom->id);
        obligation.precision = SyncRegionPrecision::Conservative;
        obligation.participation = participation;
        obligation.control =
            a->guard.empty() && b->guard.empty() ? SyncControlRelation::MustExecute : SyncControlRelation::Unknown;
        obligation.iteration = relation;
        obligation.detail = "canonical guarded outstanding local effect";
        indices.emplace(key, obligation.id);
        result.requirements.push_back(std::move(obligation));
        return true;
    }
    bool phase(const SyncPhase& physical, History& history)
    {
        SmallVector<const SyncAccess*, 4> accesses;
        for (auto id : atom->accesses) {
            const auto* access = schedule.findAccess(id);
            if (access->phase == physical.id) {
                accesses.push_back(access);
            }
        }
        if (accesses.empty()) {
            return true;
        }
        for (const auto* access : accesses) {
            for (const auto& prior : history.effects) {
                if (!requirement(prior, *access)) {
                    return false;
                }
            }
        }
        const bool exhaustedHistory = !charge(history.effects.size() + history.definitions.size() + 1);
        if (exhaustedHistory) {
            return false;
        }
        auto found = states.find(physical.id);
        if (found == states.end()) {
            const bool exhaustedStates = result.states.size() >= kInvalidSyncId;
            if (exhaustedStates) {
                return false;
            }
            SyncLocalMemoryState state;
            state.id = result.states.size();
            state.atom = atom->id;
            state.phase = physical.id;
            state.region = physical.region;
            state.guard = physical.guard;
            state.iterationDomain = physical.iterationDomain;
            state.precision = SyncRegionPrecision::Conservative;
            found = states.emplace(physical.id, state.id).first;
            result.states.push_back(std::move(state));
        }
        auto& state = result.states[found->second];
        merge(state.mayDefinitions, ArrayRef(history.definitions));
        for (const auto* access : accesses) {
            state.reads |= access->mode != SyncAccessMode::Write;
            state.writes |= access->mode != SyncAccessMode::Read;
            const Reference ref{access->id, false};
            if (!llvm::is_contained(history.effects, ref)) {
                history.effects.push_back(ref);
            }
        }
        if (state.writes && !llvm::is_contained(history.definitions, state.id)) {
            history.definitions.push_back(state.id);
        }
        return true;
    }
    bool elements(const SyncRegion& owner, History& history, unsigned depth)
    {
        for (const auto& element : owner.elements) {
            if (element.kind == SyncRegionElement::Kind::Phase) {
                if (!phase(*schedule.findPhase(element.phase), history)) {
                    return false;
                }
            } else if (element.kind == SyncRegionElement::Kind::ChildRegion) {
                if (!region(*schedule.findRegion(element.child), history, depth + 1)) {
                    return false;
                }
            }
        }
        return true;
    }
    bool region(const SyncRegion& owner, History& history, unsigned depth)
    {
        const bool exhaustedRegion = depth > 16 || !charge(history.effects.size() + history.definitions.size() + 1);
        if (exhaustedRegion) {
            limited = true;
            return false;
        }
        const History incoming = history;
        if (owner.kind == SyncRegionKind::Choice) {
            History joined;
            for (const auto& element : owner.elements) {
                if (element.kind != SyncRegionElement::Kind::ChildRegion) {
                    continue;
                }
                History alternative = incoming;
                if (!region(*schedule.findRegion(element.child), alternative, depth + 1)) {
                    return false;
                }
                merge(joined, alternative);
            }
            history = std::move(joined);
        } else if (owner.kind == SyncRegionKind::Loop) {
            if (!elements(owner, history, depth)) {
                return false;
            }
            for (auto& ref : history.effects) {
                const auto* effect = schedule.findAccess(ref.access);
                const auto* physical = schedule.findPhase(effect->phase);
                ref.carried = llvm::is_contained(physical->iterationDomain.loops, owner.id);
            }
            merge(history, incoming);
            if (!elements(owner, history, depth)) {
                return false;
            }
            merge(history, incoming); // Explicit zero-trip bypass, including readers.
        } else if (!elements(owner, history, depth)) {
            return false;
        }
        const bool exhaustedSummary = !charge(history.effects.size() + history.definitions.size() + 1);
        if (exhaustedSummary) {
            return false;
        }
        SyncLocalRegionSummary summary;
        summary.region = owner.id;
        summary.atom = atom->id;
        summary.incoming = snapshot(incoming);
        summary.outgoing = snapshot(history);
        summary.complete = true;
        result.regionSummaries.push_back(std::move(summary));
        return true;
    }
    const StructuredSyncIR& schedule;
    SyncLocalMemoryAnalysis& result;
    std::size_t remaining;
    const SyncLocalStorageAtom* atom = nullptr;
    std::map<SyncPhaseId, std::uint32_t> states;
    std::map<Key, unsigned> indices;
};
} // namespace

LogicalResult mlir::pto::protocol_sync::analyzeStructuredLocalFlow(
    const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result, std::size_t maximumEntries)
{
    result.structuredStatus = SyncLocalStructuredStatus::Unsupported;
    if (!supportsStructuredFrontier(schedule, true)) {
        return success();
    }
    SyncLocalMemoryAnalysis scratch;
    Transfer transfer(schedule, scratch, maximumEntries);
    for (const auto& atom : result.atoms) {
        if (!transfer.run(atom)) {
            if (!transfer.limited) {
                return failure();
            }
            result.structuredStatus = SyncLocalStructuredStatus::LimitExceeded;
            return success();
        }
    }
    result.states = std::move(scratch.states);
    result.regionSummaries = std::move(scratch.regionSummaries);
    result.requirements = std::move(scratch.requirements);
    result.structuredStatus = SyncLocalStructuredStatus::Complete;
    return success();
}
