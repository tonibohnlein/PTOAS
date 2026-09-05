// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LocalMemoryAnalysis.cpp - Sparse per-atom outstanding effects --------===//

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
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

bool collectRegions(const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result)
{
    const SyncPhase* first = nullptr;
    for (const SyncAccess& access : schedule.getAccesses()) {
        if (access.storage.space != AddressSpace::VEC) {
            continue;
        }
        const SyncPhase* phase = schedule.findPhase(access.phase);
        const bool exactPhase = phase && phase->operation && phase->core == SyncPhysicalCore::Vector &&
                                !phase->macroPhase && phase->completion == SyncCompletionKind::PhaseEnd;
        const bool ordinaryPhase =
            exactPhase && phase->iterationDomain.loops.empty() && access.mode != SyncAccessMode::Ordered;
        if (!ordinaryPhase) {
            result.boundary = "local domain needs ordinary non-recurring physical phases";
            return false;
        }
        const bool sameDomain =
            !first || (phase->operation->getBlock() == result.block && sameGuard(first->guard, phase->guard));
        if (!sameDomain) {
            result.boundary = "local domain crosses a block, choice, or physical section";
            return false;
        }
        first = first ? first : phase;
        result.block = phase->operation->getBlock();
        SyncLocalAccessRegion region = recoverLocalAccessRegion(access);
        if (region.precision == SyncRegionPrecision::Unknown) {
            result.boundary = "local access lacks an independently recovered bounded footprint";
            return false;
        }
        result.regions.push_back(region);
    }
    return true;
}

LogicalResult buildAtoms(SyncLocalMemoryAnalysis& result)
{
    // Endpoint sweep: do not form overlap-connected equivalence classes.
    using Endpoint = std::pair<SyncAccessId, bool>;
    std::map<std::uint64_t, SmallVector<Endpoint, 2>> endpoints;
    for (const SyncLocalAccessRegion& region : result.regions) {
        endpoints[region.interval.begin].push_back({region.access, true});
        endpoints[region.interval.begin + region.interval.size].push_back({region.access, false});
    }
    std::set<SyncAccessId> active;
    for (auto point = endpoints.begin(); point != endpoints.end(); ++point) {
        for (const auto& [access, starts] : point->second) {
            if (starts) {
                active.insert(access);
            } else {
                active.erase(access);
            }
        }
        auto next = std::next(point);
        const bool emptySegment = next == endpoints.end() || active.empty();
        if (emptySegment) {
            continue;
        }
        const bool atomIdsExhausted = result.atoms.size() >= kInvalidSyncId;
        if (atomIdsExhausted) {
            return failure();
        }
        SyncLocalStorageAtom atom;
        atom.id = static_cast<std::uint32_t>(result.atoms.size());
        atom.interval = {point->first, next->first - point->first};
        for (SyncAccessId access : active) {
            atom.accesses.push_back(access);
        }
        result.atoms.push_back(std::move(atom));
    }
    return success();
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
    obligation.control = phase->guard.empty() ? SyncControlRelation::MustExecute : SyncControlRelation::SameGuard;
    obligation.iteration = {SyncIterationRelationKind::SameIteration, 0};
    obligation.detail = kind == SyncObligationKind::Reclamation ? "canonical local outstanding read before overwrite" :
                                                                  "canonical local outstanding write before access";
    indices.emplace(key, obligation.id);
    result.requirements.push_back(std::move(obligation));
    return success();
}

LogicalResult buildAtomFlow(
    const StructuredSyncIR& schedule, const SyncLocalStorageAtom& atom, SyncLocalMemoryAnalysis& result,
    std::map<RequirementKey, unsigned>& indices)
{
    std::map<SyncPhaseId, PhaseAccess> phases;
    for (SyncAccessId id : atom.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        if (!access) {
            return failure();
        }
        PhaseAccess& phase = phases[access->phase];
        phase.phase = access->phase;
        if (access->mode != SyncAccessMode::Write) {
            phase.read = std::min(phase.read, id);
        }
        if (access->mode != SyncAccessMode::Read) {
            phase.write = std::min(phase.write, id);
        }
    }
    SyncAccessId writer = kInvalidSyncId;
    std::uint32_t definition = kInvalidSyncId;
    SmallVector<SyncAccessId, 4> readers;
    for (const auto& [phaseId, phase] : phases) {
        const bool writes = phase.write != kInvalidSyncId;
        const SyncAccessId target = writes ? phase.write : phase.read;
        if (failed(appendRequirement(
                schedule, result, indices, SyncObligationKind::Completion, writer, target, atom.id))) {
            return failure();
        }
        const bool stateIdsExhausted = result.states.size() >= kInvalidSyncId;
        if (stateIdsExhausted) {
            return failure();
        }
        const auto stateId = static_cast<std::uint32_t>(result.states.size());
        result.states.push_back(
            {stateId, atom.id, phaseId, definition, phase.read != kInvalidSyncId, writes,
             SyncRegionPrecision::Conservative});
        if (writes) {
            for (SyncAccessId reader : readers) {
                if (failed(appendRequirement(
                        schedule, result, indices, SyncObligationKind::Reclamation, reader, phase.write, atom.id))) {
                    return failure();
                }
            }
            // Retire only the discovery frontier: the old effects remain in the
            // required chain through this write. This is NOT a definite kill;
            // previousDefinition retains the may-reaching memory state.
            readers.clear();
            writer = phase.write;
            definition = stateId;
        } else {
            readers.push_back(phase.read);
        }
    }
    return success();
}

} // namespace

FailureOr<SyncLocalMemoryAnalysis> mlir::pto::protocol_sync::analyzeLocalMemory(const StructuredSyncIR& schedule)
{
    if (!schedule.isFrozen()) {
        return failure();
    }
    SyncLocalMemoryAnalysis result;
    result.scope = schedule.getFunction();
    result.coveredAccesses.resize(schedule.getAccesses().size());
    if (!collectRegions(schedule, result)) {
        result.regions.clear();
        return result;
    }
    if (failed(buildAtoms(result))) {
        return failure();
    }
    std::map<RequirementKey, unsigned> indices;
    for (const SyncLocalStorageAtom& atom : result.atoms) {
        if (failed(buildAtomFlow(schedule, atom, result, indices))) {
            return failure();
        }
    }
    for (const SyncLocalAccessRegion& region : result.regions) {
        result.coveredAccesses.set(region.access);
    }
    return result;
}
