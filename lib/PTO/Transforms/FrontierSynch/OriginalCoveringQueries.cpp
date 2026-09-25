// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalCoveringQueries.h"
#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "DirectionalCovering.h"
#include <map>

namespace mlir::pto::frontiersynch {
struct OriginalCoveringQueries::Impl {
    const OriginalStructure& original;
    const OriginalLifetimes& storage;
    const OriginalValueQueries& values;
    OriginalProgramVersion version;
    detail::ControlGraph graph;
    mutable std::map<std::size_t, OriginalValueQualification> loopQualifications;
    detail::DirectionalCovering core;

    Impl(const OriginalStructure& original, const OriginalLifetimes& storage, const OriginalValueQueries& values)
        : original(original), storage(storage), values(values), version(original.version),
          graph(detail::buildControlGraph(original)),
          core(original.body, graph, version,
               [this](std::size_t op, const OriginalAccessSelector& selector) { return matches(op, selector); },
               [this](const Region& region) { return finite(region); })
    {}
    detail::CoveringMatch matches(std::size_t op, const OriginalAccessSelector& selector) const
    {
        using Match = detail::CoveringMatch;
        const auto& payload = original.operations[op];
        const bool knownEngine = payload.instruction && payload.instruction->kPipeValue != PipelineType::PIPE_UNASSIGNED;
        if (selector.engine && knownEngine && unsigned(payload.instruction->kPipeValue) != *selector.engine) {
            return Match::NoHit;
        }
        auto result = Match::NoHit;
        // The complete shared effect import is authoritative. Unknown geometry
        // is a may incidence, not an exclusion; no opcode-specific translator or
        // bounding-allocation full-read assumption is added here.
        for (const auto& effect : payload.accesses) {
            if (effect.cell != selector.cell ||
                (selector.physicalRelation != NoControlId && effect.physicalRelation != NoControlId &&
                 effect.physicalRelation != selector.physicalRelation) ||
                !((selector.read && effect.read) || (selector.write && effect.write))) {
                continue;
            }
            if (selector.physicalRelation != NoControlId && effect.physicalRelation == NoControlId) {
                result = Match::Unknown;
                continue;
            }
            if (selector.write && effect.write && effect.definiteWrite && knownEngine) {
                return Match::Must;
            }
            result = original.cells[selector.cell].unknownRange || !knownEngine ? Match::Unknown : Match::May;
        }
        return result;
    }
    bool finite(const Region& region) const
    {
        const auto found = loopQualifications.find(region.originalOwner);
        if (found != loopQualifications.end()) {
            return found->second.executableAfterPrerequisites();
        }
        OriginalValueQualification qualification;
        if (region.kind == Region::For && region.originalOwner < original.originalSites.size()) {
            if (auto loop = dyn_cast_or_null<scf::ForOp>(original.originalSites[region.originalOwner])) {
                qualification = values.counted(loop);
            } else {
                qualification.obstructions.emplace_back("counted region has no original scf.for identity");
            }
        } else {
            qualification.obstructions.emplace_back("repetition has no supplied finite-exit derivation");
        }
        const bool qualified = qualification.executableAfterPrerequisites();
        // NeedsCompletion retains the exact original control prerequisite. It
        // does NOT grant selected completion. Unlike exact D3, covering does not
        // require invariant participation or an available last-reader guard.
        loopQualifications.emplace(region.originalOwner, std::move(qualification));
        return qualified;
    }
};
OriginalCoveringQueries::OriginalCoveringQueries(
    const OriginalStructure& original, const OriginalLifetimes& storage, const OriginalValueQueries& values)
    : impl(std::make_unique<Impl>(original, storage, values))
{}
OriginalCoveringQueries::~OriginalCoveringQueries() = default;
const CoveringBoundaryStats& OriginalCoveringQueries::stats() const { return impl->core.stats(); }
OriginalCoveringBoundary OriginalCoveringQueries::source(const OriginalInterval& interval) const
{
    return query(interval, CoveringDirection::Source);
}
OriginalCoveringBoundary OriginalCoveringQueries::target(const OriginalInterval& interval) const
{
    return query(interval, CoveringDirection::Target);
}
OriginalCoveringBoundary OriginalCoveringQueries::query(
    const OriginalInterval& interval, CoveringDirection direction) const
{
    OriginalCoveringBoundary result;
    result.interval = interval;
    result.direction = direction;
    // Validate the snapshot, selector, legal interval cuts and derived owner
    // BEFORE consulting even a negative/unknown cached covering answer.
    const auto prepared = impl->storage.prepareInterval(interval.query);
    if (!impl->values.current() || impl->original.version != impl->version || !prepared.valid ||
        prepared.interval != interval) {
        result.obstruction = CoveringBoundary::Obstruction::InvalidInterval;
        result.reason = !prepared.valid ? prepared.reason : "original interval or analysis snapshot changed";
        return result;
    }
    if (interval.query.selector.engine == unsigned(PipelineType::PIPE_UNASSIGNED)) {
        result.obstruction = CoveringBoundary::Obstruction::EngineRequired;
        result.reason = "directional boundary has no assigned original engine";
        return result;
    }
    static_cast<CoveringBoundary&>(result) = impl->core.query(interval, direction);
    for (auto owner : result.loopOwners) {
        const auto found = impl->loopQualifications.find(owner);
        if (found != impl->loopQualifications.end()) {
            result.controlQualifications.push_back({owner, found->second});
        }
    }
    if (result.status == CoveringBoundary::Status::Covering) {
        if (!result.cut || !resolveOriginalCut(impl->original, *result.cut)) {
            result.status = CoveringBoundary::Status::Unknown;
            result.obstruction = CoveringBoundary::Obstruction::CutUnavailable;
            result.reason = "prescribed covering cut is not executable in the unchanged original IR";
            result.cut.reset();
            result.referenceCoverage = false;
            return result;
        }
        // There is no new value to observe: execute true within the existing
        // control at this cut. Original loop prerequisites are separate records.
        result.endpointQualification.status = OriginalValueQualification::Status::Available;
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
