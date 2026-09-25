// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PROGRAMANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PROGRAMANALYSIS_H

#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "PTO/Transforms/FrontierSynch/ReaderFrontiers.h"
#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include "PTO/Transforms/FrontierSynch/OriginalObligations.h"
#include "PTO/Transforms/FrontierSynch/ExactFrontiers.h"
#include "PTO/Transforms/FrontierSynch/OriginalCoveringQueries.h"
#include "PTO/Transforms/FrontierSynch/OriginalRequests.h"
#include "PTO/Transforms/FrontierSynch/OriginalPreparation.h"
#include <optional>

namespace mlir::pto::frontiersynch {

// An original-program interpretation of one required physical ordering edge.
// Unknown qualification retains the edge and every independently established fact.
struct DecodedRequirement {
    OriginalRequirement requirement;
    std::size_t owner = NoControlId;
    std::vector<std::size_t> sourceBankRelations, targetBankRelations;
    OriginalReaderFrontiers readers;
    bool hasReaderFrontier = false;
    std::vector<std::string> unresolved;
    std::optional<OriginalInterval> interval;
    OriginalIntervalRequest continuation;
    // The reader summary has its OWN interval, not the triggering edge's
    // source-to-deadline interval (which can end before the first reader).
    std::optional<OriginalInterval> readerInterval;
};
struct OriginalBoundaryResult {
    enum class Status { Unknown, NoHit, Exact } status = Status::Unknown;
    std::size_t owner = NoControlId, cell = 0;
    std::size_t nonempty = 0, frontier = 0;
    bool guardsAvailableAtReadSites = false;
    // This direction's endpoint qualification, independent of the other frontier.
    OriginalValueQualification endpointQualification;
    // Shared original may-set owned by ProgramAnalysis; no per-requirement copy.
    const std::vector<StorageOrigin>* mayAccesses = nullptr;
    std::string reason;
    std::optional<OriginalInterval> interval;
    std::vector<OriginalCut> cuts;
    // NoControlId means unresolved, not an empty/no-access path.
    std::size_t noHit = NoControlId;
};
struct OriginalEndpointCandidate {
    SourceMilestone position;
    std::size_t predicate = 0;
    bool guardAvailableAtOperation = false;
    bool executableInOriginalIR = false;
    OriginalValueQualification qualification;
};
struct OriginalRequirementId {
    std::size_t deadline = NoControlId, index = 0;
};
struct TypedOriginalRequirement {
    enum class Cause { BranchCondition, LoopBound, WhileCondition, Address } cause = Cause::BranchCondition;
    SourceMilestone source;
    std::size_t deadlineOriginalSite = NoControlId;
    Value requiredValue;
    PipelineType sourceEngine = PipelineType::PIPE_UNASSIGNED;
    bool incomingOrUnknownSource = false;
    bool sourceGapExecutable = false;
    // The original SSA dependency is a requirement, not proof that the source
    // completion is natively available at this dynamic control occurrence.
    bool occurrenceQualified = false;
    OriginalValueQualification::Status availability = OriginalValueQualification::Status::Unresolved;
    OriginalCompletionPrerequisite completion;
    std::vector<std::string> qualificationObstructions;
};
// Borrowed original-effect witnesses. The source/consumer operation retains its
// engine and every translated incidence (memory and physical-selector identity).
// Incoming and typed cases are explicit; no witness asserts acquired completion.
struct OriginalObligationWitness {
    ObligationMembership membership;
    const Cell* physicalCell = nullptr;
    const PhysicalOperation *sourceOperation = nullptr, *consumerOperation = nullptr;
    const std::vector<std::size_t>*sourceEffects = nullptr, *consumerEffects = nullptr;
    const TypedOriginalRequirement* typedRequirement = nullptr;
};
struct OriginalAllQuery {
    std::size_t owner = NoControlId, cell = 0;
    bool read = true, write = true;
    std::optional<PipelineType> engine;
};
struct OriginalOccurrenceInterpretation {
    enum class Status { Unknown, FixedVisit, PeriodicSameRole, PeriodicRoles } status = Status::Unknown;
    std::size_t owner = NoControlId;
    std::size_t source = NoControlId, target = NoControlId;
    // Compatibility scalar summary for uniform-distance links. The full D2
    // record below is authoritative for distinct roles and boundary domains.
    std::size_t period = 0;
    std::size_t firstOrdinalWithLocalPredecessor = 0;
    bool earlierOrdinalsNeedIncomingCase = false;
    // A bank relation is a physical candidate until predecessor/successor
    // occurrence matching, participation and child transport are proved.
    std::vector<std::size_t> sharedBankCandidates;
    std::optional<PeriodicUseCorrespondence> periodic;
    std::string reason;
    std::optional<OriginalInterval> interval;
};
struct OriginalSupportRequest {
    bool hasWriteDelimitedCandidate = false;
    std::size_t producer = NoControlId, reuse = NoControlId, cell = 0;
    std::string reason;
};
// Complete original-only interpretation requested at a construction deadline.
// Each component keeps its own qualification: a may origin or structural
// frontier is never promoted to an exact generation or selected completion.
struct InterpretedRequirement {
    const DecodedRequirement* decoded = nullptr;
    OriginalOccurrenceInterpretation occurrence;
    const StorageLifecycle *sourceUse = nullptr, *targetUse = nullptr;
    // A missing incoming full writer is a separate, unresolved entry path; the
    // alternative-origin may-set does not silently cover it.
    bool mayHaveIncomingGeneration = true;
    OriginalContinuationQuery interval; // Compatibility view; queryInterval is authoritative.
    OriginalSupportRequest support;
    OriginalBoundaryResult firstConflict, lastRelevantUse;
    const std::vector<StorageOrigin>* mayAlternativeSources = nullptr;
    bool alternativeGuardsQualified = false;
    // Shared conditional original-use expression. Acyclic formation alone is
    // not an exact occurrence or executable endpoint qualification.
    const FactoredUseResult* factoredUse = nullptr;
    std::optional<OriginalInterval> queryInterval;
    // This view retains the target's applicability, hazard, old-state source
    // expression and translated-effect witnesses. It is not a new obligation ID
    // and does not certify a source from the conservative marginal relationship.
    FactoredDemandView factoredDemand;
    // D1 is conditional: keep the predicates, endpoint qualifications and the
    // complete alternative family, not only occurrence.status == FixedVisit.
    // Predicate IDs use fixedVisit.alternatives->sources.predicates.
    FixedVisitCorrespondence fixedVisit;
};

// Owns the immutable original structure and its construction-facing query services.
// The unchanged SyncInput and source function must outlive this result. The
// obligation universe is frozen before this result is read. Compatibility pair
// and subscription populations are materialized together when requested.
class ProgramAnalysis {
public:
    ProgramAnalysis(
        const SyncInput& input, OriginalStructure structure, OriginalRequestBoundaryProvider requestBoundaries = {});
    ~ProgramAnalysis();
    ProgramAnalysis(const ProgramAnalysis&) = delete;
    ProgramAnalysis& operator=(const ProgramAnalysis&) = delete;

    // Frozen formation completed. Inspect preparation()->repertoireComplete()
    // and per-query obstructions before relying on a declared exact source.
    bool complete() const;
    const std::string& reason() const;
    const OriginalStructure& structure() const { return original; }
    const SyncInput& translatedInput() const { return input; }
    const OriginalLifetimes& lifetimes() const { return storage; }
    // Also queryable when the only source is an incoming interface and no
    // local-source marginal requirement exists. Input expressions are shared,
    // not enumerated into synthetic source/target pairs.
    const FactoredUseResult& originalUsesAt(std::size_t operation, std::size_t cell) const
    {
        return storage.factoredAt(operation, cell);
    }
    FactoredUseResult applyOriginalUsesAt(std::size_t operation, std::size_t cell, FactoredUseInterface boundary) const
    {
        return storage.applyFactoredAt(operation, cell, std::move(boundary));
    }
    const OccurrenceQueries& occurrences() const { return occurrenceQueries; }
    const OriginalValueQueries& originalValues() const { return values; }
    // The construction-facing universe, indexed by ORIGINAL site (not phase).
    // Shared family roots include payload/control-only deadlines and incoming cases.
    // membership()/origins() expose functional OriginalObligationIds lazily;
    // a family root is not one progress unit for all its independent readers.
    // Use membership/origins/witness queries explicitly; group/descriptor copies
    // do not create IDs. Guarded means membership in the modeled original-use
    // relation, not exact physical geometry or executable endpoint qualification.
    const OriginalObligations& obligations() const;
    // Prepared and frozen in the constructor, before traversal/probing. nullptr
    // means preparation failed or the original snapshot is stale. Unknown
    // boundary components remain in declaredSlots, not the recipe population;
    // their presence is not a preparation error or selected completion.
    const OriginalRequests* requests() const;
    const OriginalPreparation* preparation() const;
    const std::vector<OriginalObligationFamilyId>& obligationsAt(std::size_t originalSite) const;
    OriginalObligationWitness obligationWitness(OriginalObligationFamilyId id, ObligationOrigin source) const;
    OriginalObligationWitness obligationWitness(OriginalObligationId id) const;
    Value obligationGuardValue(std::size_t originalIfSite) const;
    // D1 query of one frozen obligation family, including incoming-only cases.
    // This materializes that family's origins, not source-target products during
    // formation. It neither changes IDs nor discharges the family.
    // For a canonical repeated-program family this is ONLY a local visit view.
    // Its alternatives do not exhaust the global origins or discharge transport.
    std::shared_ptr<const FixedVisitSourceFrontier> fixedSourcesFor(OriginalObligationFamilyId id) const
    {
        const auto* family = obligations().get(id);
        if (!family || !values.current() || family->key.originalVersion != original.version ||
            family->key.kind == OriginalObligationKey::Kind::Typed ||
            (family->key.occurrences != OriginalObligationKey::Occurrences::FixedUseProjection &&
             (!family->expression || !family->expression->complete))) {
            auto unknown = std::make_shared<FixedVisitSourceFrontier>();
            unknown->sources.reason = "D1 requires a current fixed-use physical obligation family";
            return unknown;
        }
        const auto& key = family->key;
        const auto hazard = key.kind == OriginalObligationKey::Kind::RAW ? FactoredUseNode::Hazard::RAW :
                            key.kind == OriginalObligationKey::Kind::WAR ? FactoredUseNode::Hazard::WAR :
                                                                          FactoredUseNode::Hazard::WAW;
        const auto result = occurrenceQueries.fixedSourcesAt(key.consumerOperation, key.cell, hazard);
        if (result->sources.complete && key.occurrences == OriginalObligationKey::Occurrences::FixedUseProjection &&
            result->sources.frame.owner != key.owner) {
            auto unknown = std::make_shared<FixedVisitSourceFrontier>(*result);
            unknown->sources.complete = false;
            unknown->sources.reason = "D1 family needs a qualified enclosing occurrence transport";
            return unknown;
        }
        return result;
    }

    const PhysicalBankCorrespondence& bankRelation(std::size_t index) const;
    // Query a local D2 pairing directly from a lazy obligation ID. Its entry and
    // exit are NOT substituted for the obligation's complete prefix/horizon.
    // This does not enumerate compatibility pairs or alter obligation IDs.
    PeriodicUseCorrespondence periodicUseFor(OriginalObligationId obligation) const;
    const std::vector<StorageOrigin>& alternativeSourcesFor(const DecodedRequirement& requirement) const;
    // Compatibility pair view. These APIs are not the obligation-ID universe and
    // requesting them explicitly charges their expanded marginal pair output.
    const std::vector<OriginalRequirement>& requirementsAt(std::size_t operation) const;
    const std::vector<OriginalRequirementId>& requirementsFromTo(PipelineType source, PipelineType target) const;
    const std::vector<TypedOriginalRequirement>& typedRequirementsAt(std::size_t originalSite) const;
    const std::vector<std::pair<std::size_t, std::size_t>>& typedSubscriptionsAt(std::size_t sourceOperation) const;
    // Decode only an applicable indexed request. The immutable source positions
    // were subscribed before construction; decoding need not precede traversal.
    const DecodedRequirement& decodeAt(std::size_t operation, std::size_t index) const;
    const DecodedRequirement& decodeAt(
        std::size_t operation, std::size_t index, OriginalIntervalRequest continuation) const;
    OriginalIntervalRequest intervalFor(std::size_t operation, std::size_t index) const;
    OriginalIntervalResult prepareInterval(OriginalIntervalRequest request) const;
    // I.3 directional queries over explicitly prepared family intervals. The
    // caller chooses each side's fixed selector and horizon; a triggering edge's
    // interval is not automatically the interval of all source or target uses.
    // These queries neither overwrite exact answers nor generate descriptors,
    // source subscriptions, selected completion or event protocols.
    const OriginalCoveringQueries& coveringQueries() const
    {
        if (!covering) {
            covering = std::make_unique<OriginalCoveringQueries>(original, storage, values);
        }
        return *covering;
    }
    OriginalCoveringBoundary sourceCovering(const OriginalInterval& interval) const
    {
        return coveringQueries().source(interval);
    }
    OriginalCoveringBoundary targetCovering(const OriginalInterval& interval) const
    {
        return coveringQueries().target(interval);
    }
    // Lazily joins the shared original queries for a currently relevant request.
    // No selected ledger, event key, or predicted completion enters this result.
    const InterpretedRequirement& interpretAt(std::size_t operation, std::size_t index) const;
    const InterpretedRequirement& interpretAt(
        std::size_t operation, std::size_t index, OriginalIntervalRequest continuation) const;
    FixedVisitCorrespondence fixedVisitFor(const DecodedRequirement& requirement) const;
    OriginalBoundaryResult firstConflict(const DecodedRequirement& requirement) const;
    OriginalBoundaryResult lastRelevantUse(const DecodedRequirement& requirement) const;
    // D3/I.2 over the FULL original interval key and fixed access selector.
    // readOnly additionally requires an unchanged read episode (no cell writes).
    // These locate original accesses, not D1/D2 source-to-target matches. Exact
    // structural roots and each endpoint's availability are separate results.
    OriginalExactFrontiers exactFrontiers(const OriginalInterval& interval, bool readOnly = false) const;
    OriginalBoundaryResult firstConflict(const OriginalInterval& interval, bool readOnly = false) const;
    OriginalBoundaryResult lastRelevantUse(const OriginalInterval& interval, bool readOnly = false) const;
    const OriginalIntervalParticipation* intervalParticipation(std::size_t id) const;
    OriginalExactFrontierStats exactFrontierStats() const;
    // General may-frontier queries over a caller-qualified original interval.
    // Their Present result does not by itself establish guarded participation.
    PhysicalUseFrontier firstMayUse(const OriginalUseQuery& query) const;
    PhysicalUseFrontier lastMayUse(const OriginalUseQuery& query) const;
    PhysicalUseFrontier firstMayUse(const OriginalInterval& query) const;
    PhysicalUseFrontier lastMayUse(const OriginalInterval& query) const;
    const std::vector<SourceSubscription>& subscriptionsAt(std::size_t operation) const;
    OriginalAccessSummary all(const OriginalAllQuery& query) const;
    OriginalMayAfter mayAfter(const OriginalContinuationQuery& query) const;
    OriginalMayAfter mayAfter(const OriginalInterval& query) const;
    OriginalSupportInterval qualifySupport(const InterpretedRequirement& requirement) const;
    OriginalBoundaryResult qualifiedBoundary(
        const InterpretedRequirement& requirement, const OriginalSupportInterval& support, bool first) const;
    // A qualified original frontier expands to guarded legal payload cuts.
    // Selected-word positions and their causal source snapshots belong to Phase B.
    std::vector<OriginalEndpointCandidate> endpointCandidates(
        const OriginalBoundaryResult& boundary, SourceMilestone::Side side) const;
    ParticipationExpression predicate(std::size_t id) const;
    GuardedReadFrontier readerFrontier(std::size_t id) const;
    bool guardAvailableAt(std::size_t predicateId, std::size_t operation) const;
    OriginalValueQualification guardQualificationAt(
        std::size_t predicateId, std::size_t operation, SourceMilestone::Side side) const;
    std::vector<OriginalParticipationDemand> participationDemands() const;

private:
    void ensureLegacyRequirementIndex() const;
    OriginalBoundaryResult boundary(const DecodedRequirement& requirement, bool first) const;
    OriginalBoundaryResult exactBoundary(const OriginalInterval& interval, bool first, bool readOnly) const;
    OriginalIntervalResult supportInterval(const InterpretedRequirement& requirement) const;
    const SyncInput& input;
    OriginalStructure original;
    OriginalValueQueries values;
    OriginalLifetimes storage;
    OccurrenceQueries occurrenceQueries;
    mutable std::unique_ptr<OriginalCoveringQueries> covering;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace mlir::pto::frontiersynch
#endif
