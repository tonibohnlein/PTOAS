// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALREQUESTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALREQUESTS_H

#include "PTO/Transforms/FrontierSynch/OriginalDescriptorSlots.h"
#include "PTO/Transforms/FrontierSynch/OriginalObligations.h"
#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "PTO/Transforms/FrontierSynch/OriginalCoveringQueries.h"
#include <functional>

namespace mlir::pto::frontiersynch {
class ProgramAnalysis;

struct OriginalRequestReference {
    OriginalObligationFamilyId family;
    // nullopt retains the whole unresolved original source expression. It must
    // not be counted as one progress unit or interpreted as an incoming origin.
    std::optional<ObligationOrigin> origin;
    std::size_t sourceExpression = NoControlId;
    std::optional<unsigned> sourceEngine;
    DescriptorPredicate applicability;
    std::optional<OriginalObligationId> obligation() const
    {
        return origin ? std::optional<OriginalObligationId>{{family, *origin}} : std::nullopt;
    }
};
struct OriginalRequestGroup {
    std::size_t id = NoControlId;
    std::vector<OriginalObligationFamilyId> families;
    std::size_t owner = NoControlId, deadline = NoControlId;
    OriginalObligationKey::Role sourceRole = OriginalObligationKey::Role::Writer;
    OriginalObligationKey::Role targetRole = OriginalObligationKey::Role::Reader;
    std::optional<unsigned> sourceEngine, targetEngine;
    std::size_t sourceExpression = NoControlId;
    // The full slot inventory preserves obstructions and obligation references.
    std::vector<OriginalDescriptorSlots::SlotRoot> declaredSlots;
    // Only fully described roots belong to the recipe population (I.4).
    // A binder must still check endpoint uniformity, support and actual paths.
    std::vector<OriginalDescriptorSlots::SlotRoot> descriptors;
};

// Finite, declared original roles. A provider supplies the complete record;
// shifts label links and never allocate a node per dynamic iteration.
struct OriginalSupportRole {
    // Provider-local index, arena zero, universe equal to the original snapshot.
    // Duplicate IDs are rejected across the complete formation provider output.
    DescriptorFactRef id;
    OriginalProgramVersion version;
    OriginalCut source;
    OriginalInterval interval;
    bool qualified = false;
    std::vector<DescriptorSupportLink> links;
};

// A formation-time answer from the original boundary services. All four answers
// are supplied together, including missing covers when the exact answers exist.
// Observations are full results of the common OriginalValueQueries service;
// prerequisites, defining values, arithmetic recipes and failures are retained.
struct OriginalRequestAnswers {
    OriginalRequestAnswers()
    {
        exactTarget.direction = coveringTarget.direction = DescriptorBoundary::Direction::Target;
    }
    DescriptorBoundary exactSource, coveringSource, exactTarget, coveringTarget;
    FixedVisitCorrespondence fixedVisit;
    std::optional<PeriodicUseCorrespondence> periodic;
    std::optional<OriginalCoveringBoundary> sourceCoveringQuery, targetCoveringQuery;
    // Other supported D2/D4 providers retain their relation in occurrence.
    // This qualification concerns original source/consumer uses; it is not
    // one-to-one matching of the descriptor's eventual physical events.
    bool occurrenceQualified = false;
    std::vector<OriginalValueQualification> observations;
    std::vector<DescriptorFactRef> occurrence;
    std::vector<DescriptorSupportLink> support;
    std::vector<OriginalSupportRole> declaredRoles;
    std::vector<std::string> unresolved;
};
using OriginalRequestBoundaryProvider =
    std::function<OriginalRequestAnswers(const ProgramAnalysis&, const OriginalRequestReference&)>;

// Immutable after build(). The native default adapts supported D1 cuts and
// independent covering queries, retaining D2 endpoint limitations explicitly.
// Neither a failed binding nor a residual refresh has a mutation entry point.
class OriginalRequests {
public:
    OriginalRequests(const OriginalRequests&) = delete;
    OriginalRequests& operator=(const OriginalRequests&) = delete;
    bool complete() const { return model.complete() && slots.complete() && error.empty(); }
    const std::string& reason() const { return error; }
    const OriginalDescriptorSlots& descriptors() const { return slots; }
    const std::vector<OriginalRequestGroup>& groups() const
    {
        static const std::vector<OriginalRequestGroup> empty;
        return complete() ? requestGroups : empty;
    }
    const std::vector<std::size_t>& atOriginalSite(std::size_t site) const
    {
        static const std::vector<std::size_t> empty;
        const auto it = deadlines.find(site);
        return complete() && it != deadlines.end() ? it->second : empty;
    }
    const OriginalRequestReference* reference(DescriptorFactRef ref) const
    {
        return complete() && ref.kind == DescriptorFactRef::Kind::Obligation && ref.arena == identity() && ref.universe == universe &&
                       ref.index < references.size() && model.get(references[ref.index].family) ?
                   &references[ref.index] : nullptr;
    }
    const OriginalRequestAnswers* answer(DescriptorFactRef ref) const
    {
        return complete() && ref.arena == identity() && ref.universe == universe && ref.index < answers.size() &&
                       (ref.kind == DescriptorFactRef::Kind::Occurrence ||
                        ref.kind == DescriptorFactRef::Kind::BoundaryQuery) ? &answers[ref.index] : nullptr;
    }
    const OriginalValueQualification* observation(DescriptorFactRef ref) const
    {
        return complete() && ref.kind == DescriptorFactRef::Kind::CompletionPrerequisite &&
                       ref.arena == identity() && ref.universe == universe && ref.index < qualifications.size() ?
                   &qualifications[ref.index] : nullptr;
    }
    // Lazy origin membership remains the original service. A descriptor copy
    // neither registers an ID nor discharges its witnesses or applicable cases.
    ObligationMembership membership(DescriptorFactRef ref, ObligationOrigin origin) const
    {
        const auto* r = reference(ref);
        if (!r || (r->origin && !(*r->origin == origin))) { return {}; }
        return model.membership(r->family, origin);
    }
    std::size_t materializedReferences() const { return references.size(); }
    std::size_t providerQueries() const { return answers.size(); }
    const std::vector<OriginalRequestAnswers>& boundaryAnswers() const
    {
        static const std::vector<OriginalRequestAnswers> empty;
        return complete() ? answers : empty;
    }

private:
    friend class ProgramAnalysis;
    static std::unique_ptr<OriginalRequests> build(
        const ProgramAnalysis&, const OriginalRequestBoundaryProvider& provider = {});
    explicit OriginalRequests(const OriginalObligations& model) : model(model) {}
    std::uintptr_t identity() const { return reinterpret_cast<std::uintptr_t>(this); }
    DescriptorFactRef ref(DescriptorFactRef::Kind kind, std::size_t index) const { return {kind, identity(), index, universe}; }
    const OriginalObligations& model;
    OriginalProgramVersion universe = OriginalProgramVersion::fresh();
    OriginalDescriptorSlots slots;
    std::vector<OriginalRequestReference> references;
    std::vector<OriginalRequestAnswers> answers;
    std::vector<OriginalValueQualification> qualifications;
    std::vector<OriginalRequestGroup> requestGroups;
    std::map<std::size_t, std::vector<std::size_t>> deadlines;
    std::string error;
};
} // namespace mlir::pto::frontiersynch
#endif
