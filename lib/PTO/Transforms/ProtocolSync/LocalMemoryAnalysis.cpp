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

} // namespace

FailureOr<SyncLocalMemoryAnalysis> mlir::pto::protocol_sync::analyzeLocalMemory(
    const StructuredSyncIR& schedule, SyncLocalFlowOptions options)
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
    const bool productionDomain = result.boundary.empty();
    if (productionDomain && failed(analyzeLocalRegionFlow(schedule, result))) {
        return failure();
    }
    if (options.expandState) {
        // Optional expanded provenance must not replace a complete sparse
        // baseline with partial output when the diagnostic budget runs out.
        SyncLocalMemoryAnalysis expanded;
        expanded.atoms = result.atoms;
        if (failed(analyzeLocalRegionFlow(schedule, expanded, options))) {
            return failure();
        }
        result.expandedStateStatus = expanded.expandedStateStatus;
        if (expanded.expandedStateStatus == SyncLocalExpandedStateStatus::Complete) {
            result.states = std::move(expanded.states);
            result.regionSummaries = std::move(expanded.regionSummaries);
            result.requirements = std::move(expanded.requirements);
        }
    }
    for (const SyncLocalAccessRegion& region : result.regions) {
        if (result.boundary.empty()) {
            result.coveredAccesses.set(region.access);
        }
    }
    return result;
}
