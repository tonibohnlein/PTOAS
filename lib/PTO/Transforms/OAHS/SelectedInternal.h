// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_SELECTED_INTERNAL_H
#define PTO_OAHS_SELECTED_INTERNAL_H

#include "Control.h"
#include "SelectedLookahead.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <utility>

namespace mlir::pto::oahs::selected {

using Id = std::size_t;
struct Component {
    std::vector<Id> sites, order, entries;
    bool cyclic = false;
};
enum class ProofOutcome { Unknown, Proved, Disproved };
struct OccurrenceCorrespondence {
    ProofOutcome outcome = ProofOutcome::Unknown;
    std::string reason;
    Cut failedAt = NoAnalysisId;
    uint64_t siteEvaluations = 0;
    std::vector<std::pair<Cut, Cut>> pairs;
    bool proved() const { return outcome == ProofOutcome::Proved; }
};
struct Control {
    detail::ControlGraph graph;
    std::vector<std::vector<Id>> predecessors;
    // Construction-only, zero-or-more loop summary traversal. Acceptance uses
    // graph's original edges, not these exit-summary edges or hypotheses.
    std::vector<std::vector<Id>> constructionEdges, headerAccesses;
    std::vector<Component> components;
    std::vector<Id> component, position, frame;
    std::vector<bool> reachable;
    LookaheadIndex lookahead;
    struct LoopEntryFacts {
        Cut entry;
        // A unique first observer payload on every exiting entry path, or no
        // qualified deadline. These are original-program facts, not receipts.
        std::array<Cut, PipeCount> firstConsumer;
        std::array<std::vector<Cut>, PipeCount> firstConsumers;
        std::vector<Cut> sites;
        std::set<Pipe> issuedPipes;
        // Original word positions crossed by moving an acquisition to entry,
        // including the deadline's pre-payload word, excluding entry itself.
        // Their commands are selected-plan facts and must be queried afresh.
        std::array<std::vector<Cut>, PipeCount> crossedWords;
        std::set<Id> issuedClasses;
    };
    std::vector<LoopEntryFacts> loopEntries;
    uint64_t loopEntryPreparationSites = 0;
    // Memo of canonicalCommandCut, and the sites sharing each canonical word,
    // with the component span of each such word. These are facts about the
    // immutable program and control alone: no ledger state enters them, so they
    // are built once instead of rescanned by every boundary query.
    std::vector<Cut> canonicalCut;
    std::vector<std::vector<Id>> wordOccurrences;
    std::vector<std::pair<Id, Id>> wordSpan;
    bool complete = false, validInput = false;
    std::string reason;
    explicit Control(const Program&);
    bool straight(Id, Id) const;
    Cut after(Id) const;
    // The budget bounds analysis work, independently of physical event capacity.
    // Same-word correspondence assumes publication precedes acquisition; exact
    // endpoint order, acquired credit and key legality remain binder obligations.
    const OccurrenceCorrespondence& correspondence(Cut, Cut, Id budget = 65536) const;
    mutable uint64_t occurrenceAnalysisSites = 0;

private:
    OccurrenceCorrespondence pairOccurrences(Cut, Cut, Id budget) const;
    mutable std::map<std::tuple<Cut, Cut, Id>, OccurrenceCorrespondence> correspondences;
};

// An ordered proposal, including provenance, materialized by the ledger once
// for both private validation and commitment. This first contract appends at
// existing word ends; it grants no placement, matching or resource certificate.
struct PacketEndpoint {
    Cut cut;
    Command command;
    EndpointPurpose purpose;
    Id request = NoAnalysisId, acknowledges = NoAnalysisId;
};
using OrderedPacket = std::vector<PacketEndpoint>;

class Ledger {
public:
    // The canonical table belongs to the enclosing Control and outlives this.
    Ledger(const Program&, const std::vector<Cut>&);
    bool initialize(const Commands&, std::string&);
    Id append(Cut, Command, EndpointPurpose, Id request = NoAnalysisId, Id ack = NoAnalysisId);
    Id after(Id, Command, EndpointPurpose, Id request, Id ack);
    const std::vector<Id>& word(Cut) const;
    const SelectedEndpoint& endpoint(Id) const;
    Commands commands() const;
    Commands withPacket(const OrderedPacket&) const;
    std::vector<Id> appendPacket(const OrderedPacket&);
    bool active(Id id) const { return !removed.count(id); }
    void erase(Id);
    void restoreAfter(Id, Id);
    uint64_t version() const { return revision; }
    const std::vector<SelectedEndpoint>& records() const { return endpoints; }
    const std::vector<Cut>& changes() const { return changed; }
    void clearChanges() { changed.clear(); }

private:
    Cut canonical(Cut) const;
    const Program& program;
    const std::vector<Cut>& canonicalCut;
    uint64_t revision = 0;
    std::vector<std::vector<Id>> words;
    std::vector<SelectedEndpoint> endpoints;
    std::vector<Cut> changed;
    std::set<Id> removed;
    Id insert(Cut, Id, Command, EndpointPurpose, Id, Id);
};

// The last original static origin per access class. Only the classes an
// execution actually touched are represented: an absent class is NoAnalysisId,
// exactly as the dense array over cells x pipes x modes expressed it, so the two
// representations describe the same partial map and compare equal iff the maps
// are equal. The dense array was copied at every site evaluation, which
// dominated programs with many storage cells.
class LatestOrigins {
public:
    Id get(Id index) const
    {
        const auto at = std::lower_bound(items.begin(), items.end(), index,
            [](const std::pair<Id, Id>& entry, Id key) { return entry.first < key; });
        return at != items.end() && at->first == index ? at->second : NoAnalysisId;
    }
    void set(Id index, Id origin)
    {
        const auto at = std::lower_bound(items.begin(), items.end(), index,
            [](const std::pair<Id, Id>& entry, Id key) { return entry.first < key; });
        if (at != items.end() && at->first == index) {
            if (origin == NoAnalysisId) {
                items.erase(at);
            } else {
                at->second = origin;
            }
            return;
        }
        if (origin != NoAnalysisId) {
            items.insert(at, {index, origin});
        }
    }
    const std::vector<std::pair<Id, Id>>& entries() const { return items; }
    bool operator==(const LatestOrigins& b) const { return items == b.items; }
    bool operator!=(const LatestOrigins& b) const { return items != b.items; }

private:
    // Sorted by class index, and never holding a NoAnalysisId value, so the
    // representation of a given partial map is unique.
    std::vector<std::pair<Id, Id>> items;
};
struct State {
    FrontierState causal;
    // Last original static origin, or unknown after an unqualified recurrence
    // or an incompatible choice. These names never become runtime predicates.
    LatestOrigins latest;
    std::vector<std::vector<Id>> consumptions;
};
struct Checkpoint {
    State incoming, before, outgoing;
};
struct Replay {
    uint64_t version = 0, evaluations = 0;
    bool success = true;
    Cut failureCut = NoAnalysisId;
    Id failureEndpoint = NoAnalysisId;
    FrontierFailure failure = FrontierFailure::None;
    std::string reason;
    std::vector<Checkpoint> cuts;
    std::map<Id, State> afterEndpoint;
    // Leading components whose cuts hold their actual fixed point rather than
    // the construction-only hypothesis traversal of the active component. Only
    // those may be reused by a later replay; everything else is recomputed.
    std::size_t fixedComponents = 0, reusedComponents = 0;
    Id partialComponent = NoAnalysisId, partialOffset = NoAnalysisId;
    std::vector<State> partialIncoming;
};
struct RecurringRequirement {
    unsigned cell = 0;
    std::vector<unsigned> cells;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<Cut> publications, acquisitions;
    Id owner = NoAnalysisId;
    uint64_t period = 0;
    // Exact storage-cycle qualification supplies occurrence, endpoint and
    // participation correspondence directly. Relationship-derived fallbacks
    // remain eligible for conservative redundancy trials.
    bool qualifiedCycle = false;
    bool storageRelease = false;
};
struct OccurrenceMode {
    Id owner = NoAnalysisId;
    uint64_t period = 0, residue = 0;
    bool previous = false, next = false, valid = false;
    bool operator==(const OccurrenceMode& b) const
    {
        return valid && b.valid && owner == b.owner && period == b.period && residue == b.residue &&
               previous == b.previous && next == b.next;
    }
    bool operator<(const OccurrenceMode& b) const
    {
        return std::tie(owner, period, residue, previous, next, valid) <
               std::tie(b.owner, b.period, b.residue, b.previous, b.next, b.valid);
    }
};
OccurrenceMode occurrenceMode(const Program&, Cut);
enum class RequirementOccurrence : unsigned {
    Acyclic,
    SameVisit,
    PreviousUse,
    RegionEntry,
    RegionContinuation,
    Guarded,
    Unknown,
    Count
};
// Immutable physical access facts, shared by ordinary and recurring clients.
// Deadlines retain every required continuation; none allocates a private event.
struct LifecycleUse {
    unsigned cell = 0, roles = 0;
    StorageOrigin origin;
    Pipe pipe = Pipe::S;
    OccurrenceMode occurrence;
    Cut release = NoAnalysisId;
    std::vector<Cut> deadlines, returns;
};
// One original storage requirement with both of its placement bounds retained.
// This is immutable analysis metadata: it grants no completion receipt, event
// token, or permission to merge requirements that happen to use one pipeline.
struct RequirementFrontier {
    StorageRelationship relationship;
    // Priority/occurrence provenance is enriched once, on demand, because
    // native import probes recurring eligibility for several candidate loops.
    // The source/deadline relationship itself is always indexed eagerly.
    mutable RequirementProvenance provenance;
    mutable bool described = false;
    RequirementOccurrence occurrence = RequirementOccurrence::Unknown;
    Pipe source = Pipe::S, observer = Pipe::S;
    Id access = NoAnalysisId;
    // The earliest ordinary publication cut immediately after the source when
    // it is qualified as one acyclic visit. Recurring/guarded frontiers retain
    // NoAnalysisId until their occurrence qualifier supplies a paired cut.
    Cut publication = NoAnalysisId;
    // The original consumer launch deadline. Several records may deliberately
    // share a pipeline while retaining different publications or deadlines.
    Cut deadline = NoAnalysisId;
    Cut lifecycleRelease = NoAnalysisId;
};
class RequirementFrontiers {
public:
    RequirementFrontiers(const Program&, const Control&, const StorageFrontierAnalysis&);
    bool complete() const { return ready; }
    const std::string& reason() const { return error; }
    const std::vector<RequirementFrontier>& at(Cut site) const;
    std::map<Id, unsigned> reasons(Cut site) const;
    const LifecycleUse& use(Cut site, unsigned cell) const;
    Cut recurringRelease(Cut site, unsigned cell) const;
    std::vector<SelectedLifecycleDemand> demandsAt(
        Cut, const std::vector<FrontierRequirement>&) const;
    std::size_t size() const { return population; }
    std::size_t sourceBoundaries() const { return boundedSources; }
    const std::array<std::size_t, unsigned(RequirementOccurrence::Count)>& occurrenceCounts() const
    {
        return occurrences;
    }

private:
    bool ready = false;
    std::string error;
    std::size_t population = 0, boundedSources = 0;
    std::array<std::size_t, unsigned(RequirementOccurrence::Count)> occurrences{};
    const StorageFrontierAnalysis* storage = nullptr;
    const Program* program = nullptr;
    const Control* control = nullptr;
    mutable std::map<std::pair<Cut, unsigned>, LifecycleUse> uses;
    std::vector<std::vector<RequirementFrontier>> byDeadline;
};
// A storage/control qualifier: it returns requirements and original frontiers,
// not commands or physical key choices. Empty means ordinary F1--F8 applies.
std::vector<RecurringRequirement> qualifyCyclicFrontiers(
    const Program&, const Control&, const RequirementFrontiers&);

struct Group {
    Pipe source = Pipe::S;
    Cut publication = NoAnalysisId;
    std::vector<FrontierRequirement> requirements;
    std::set<Id> coverage;
    bool common = false;
    // A participation-qualified set of alternative early source cuts. It uses
    // one virgin directional key, not one key per branch. Empty means the
    // original single-cut/F7 path. The candidate is tied to the selected map.
    std::vector<Cut> publications = {};
    Id forwardKey = NoAnalysisId;
    uint64_t version = 0;
    Cut entryAcquisition = NoAnalysisId;
    Id entryReturnKey = NoAnalysisId;
    bool entryRepeats = false;
};

class Constructor {
    friend struct ReplayTestAccess;
public:
    explicit Constructor(const Program&);
    SelectedPlan run(const Commands&, bool useRecurring = true);

private:
    const Program& program;
    CausalFrontier frontier;
    Control control;
    StorageFrontierAnalysis storage;
    RequirementFrontiers requirements;
    Ledger ledger;
    SelectedPlan result;
    Replay cache;
    std::vector<bool> finalized;
    std::map<Cut, std::vector<Id>> sourcesAtCut;
    Id activeComponent = 0, activeOffset = 0;
    Cut current = NoAnalysisId;
    // Roles are stable within a selected loop component. A reverse acknowledgment
    // key must not be borrowed as another recurring channel's forward key.
    std::map<std::pair<Pipe, Pipe>, std::pair<Id, Id>> closedBindings;
    std::set<Id> closedKeys, recurringKeys;
    // Contextual state propagation is also needed for one-shot loop-entry
    // receipts. It does not reserve a physical key or establish rearming.
    bool needsContextualReplay = false;
    // Each fallback return records a pending rearming obligation. A later
    // necessary transfer may discharge it; memory requirements remain separate.
    std::map<std::pair<Pipe, Pipe>, std::vector<std::pair<Id, Id>>> pendingRearming;
    std::map<std::pair<Pipe, Pipe>, std::vector<Id>> necessaryReturns;
    // Append-only populations already paired for each direction. Old helpers
    // see only new returns; new helpers see all returns, in the original order.
    std::map<std::pair<Pipe, Pipe>, std::pair<Id, Id>> pairedReturns;
    std::set<Id> requiredReturns;
    bool restoreReturns(Id);
    bool restoreRearming(Id, Cut);
    bool settleRearming(const SelectedDecision&);
    bool returnBeforeUse(Id helperWait, Id necessaryWait);
    void rememberReturn(Id publication, Id acquisition);

    bool fail(SelectedFailure, std::string, Cut = NoAnalysisId);
    State initial() const;
    bool join(State&, const State&);
    bool word(State&, Cut, Replay&);
    bool payload(State&, Cut, Replay&, bool pending = false);
    bool contextualReplay();
    bool fixedComponent(Id, const std::vector<State>&, Replay&, std::vector<State>&);
    bool partialComponent(Id, const std::vector<State>&, Replay&);
    bool partialSite(Id, Cut, std::vector<State>&, Replay&);
    bool replay();
    // The component prefix an update may keep, shared by both replay paths.
    Id reusablePrefix() const;
    bool advance();
    void registerSource();
    void refreshSources(Cut = NoAnalysisId);
    bool update();
    State& currentState();
    std::vector<FrontierRequirement> residual() const;
    std::map<Id, unsigned> reasons(Cut) const;
    std::vector<Group> groups(const std::vector<FrontierRequirement>&, RequirementStage);
    Group sourceGroup(Pipe, const std::vector<FrontierRequirement>&,
                      const std::vector<FrontierRequirement>&);
    std::set<Id> coverage(Cut, Pipe, const std::vector<FrontierRequirement>&) const;
    bool freshBetween(Cut, Cut, Id) const;
    bool sourceFrontier(Pipe, const std::vector<FrontierRequirement>&, Group&) const;
    bool loopEntryFrontier(Pipe, const std::vector<FrontierRequirement>&, Group&);
    bool consume();
    bool bind(Group&, RequirementStage);
    bool edge(Pipe, Pipe, Cut&, bool, SelectedDecision&);
    bool acknowledgment(Pipe, Pipe, Cut&, Id&, SelectedDecision&);
    bool needsCommonAcknowledgment(const State&) const;
    Id reusable(Pipe, Pipe, const State&);
    bool canPublish(const State&, Id) const;
    bool clearInterval(Id, Cut, Cut) const;
    std::vector<Pipe> route(Pipe, Pipe) const;
    bool recurring(const std::vector<RecurringRequirement>&);
    bool finish();
};

Id accessClass(const FrontierRequirement&);
bool identical(const Command&, const Command&);
std::vector<Id> unionIds(const std::vector<Id>&, const std::vector<Id>&);

} // namespace mlir::pto::oahs::selected
#endif
