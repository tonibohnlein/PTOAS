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
#include <tuple>

namespace mlir::pto::oahs::selected {

using Id = std::size_t;
bool publicationMayCross(const Command&, const Command&);
struct Component {
    std::vector<Id> sites, order, entries;
    bool cyclic = false;
};
// A balanced source/receipt relation in the original graph. Pairs identify
// analytical occurrences, including alternative paths and repeated visits.
// Qualification supplies participation only, never completion or key credit.
struct OccurrenceCorrespondence {
    bool qualified = false;
    std::vector<std::pair<Cut, Cut>> pairs;
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
    // First downstream split/exit, through merge-only original continuations.
    // This can balance a delayed return even when a later key use is skipped.
    std::vector<Cut> rearmingBoundary;
    bool acyclic = true;
    std::vector<Pipe> sitePipes;
    LookaheadIndex lookahead;
    struct LoopEntryFacts {
        Cut entry;
        Cut exit, owner;
        std::vector<Cut> exits;
        uint64_t lastVisitDistance = 1;
        // A unique first observer payload on every exiting entry path, or no
        // qualified deadline. These are original-program facts, not receipts.
        std::array<Cut, PipeCount> firstConsumer;
        std::array<std::vector<Cut>, PipeCount> firstConsumers;
        std::vector<Cut> sites;
        // One occurrence per entry, at an invariant input's actual deadline.
        std::vector<Cut> firstInputConsumers;
        std::vector<std::pair<Cut, Cut>> firstWriteFrontiers;
        std::set<Pipe> issuedPipes;
        // Original word positions crossed by moving an acquisition to entry,
        // including the deadline's pre-payload word, excluding entry itself.
        // Their commands are selected-plan facts and must be queried afresh.
        std::array<std::vector<Cut>, PipeCount> crossedWords;
        std::set<Id> issuedClasses;
    };
    std::vector<LoopEntryFacts> loopEntries;
    // Universal placement facts for every reachable occurrence of one emitted
    // entry word. Individual records above remain the physical region view.
    std::vector<LoopEntryFacts> loopEntryFrontiers;
    struct ChoiceFrontier {
        Cut entry;
        Pipe observer;
        std::vector<Cut> consumers, crossedWords;
        std::set<Id> issuedClasses;
    };
    std::vector<ChoiceFrontier> choiceFrontiers;
    std::vector<std::vector<Id>> choicesAtConsumer;
    uint64_t choicePreparationSites = 0;
    std::set<Cut> firstPrefixWords, firstWriteWords;
    std::vector<Cut> finalReadGaps;
    std::map<Cut, Pipe> receiptGaps;
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
    bool sourceCut(Id, Pipe) const;
    bool balancedWords(Cut publication, Cut acquisition) const;
    const OccurrenceCorrespondence& correspondence(Cut publication, Cut acquisition) const;

private:
    mutable std::map<std::pair<Cut, Cut>, OccurrenceCorrespondence> correspondences;
    OccurrenceCorrespondence pairOccurrences(Cut, Cut) const;
    void prepareChoiceFrontiers(const Program&);
    bool appendChoiceOccurrence(const Program&, Cut, ChoiceFrontier&);
};

class Ledger {
public:
    // The canonical table belongs to the enclosing Control and outlives this.
    Ledger(const Program&, const std::vector<Cut>&, const std::vector<std::pair<Id, Id>>&);
    bool initialize(const Commands&, std::string&);
    Id append(Cut, Command, EndpointPurpose, Id request = NoAnalysisId, Id ack = NoAnalysisId);
    Id prepend(Cut, Command, EndpointPurpose, Id request = NoAnalysisId);
    Id after(Id, Command, EndpointPurpose, Id request, Id ack);
    const std::vector<Id>& word(Cut) const;
    const SelectedEndpoint& endpoint(Id) const;
    Commands commands() const;
    bool active(Id id) const { return !removed.count(id); }
    void erase(Id);
    void protectPublicationPrefix(Id);
    // Restricted within-word motion: retain endpoint identity and certify the
    // crossed command suffix before changing the selected word.
    bool movePublicationBefore(Id, Id);
    bool movePublicationTo(Id, Cut, Id, const std::vector<Cut>&);
    bool publicationPrefixesValid() const;
    void restoreAfter(Id, Id);
    Id lastPublicationComponent(Pipe, Pipe, unsigned) const;
    uint64_t version() const { return revision; }
    const std::vector<SelectedEndpoint>& records() const { return endpoints; }
    const std::vector<Cut>& changes() const { return changed; }
    void clearChanges() { changed.clear(); }

private:
    Cut canonical(Cut) const;
    const Program& program;
    const std::vector<Cut>& canonicalCut;
    const std::vector<std::pair<Id, Id>>& wordSpan;
    // Maximum reachable component per active publication, including all word
    // occurrences. Maintained on insert/erase/restore, including private copies.
    std::map<std::tuple<Pipe, Pipe, unsigned>, std::multiset<Id>> publicationComponents;
    void indexPublication(Id, bool);
    uint64_t revision = 0;
    std::vector<std::vector<Id>> words;
    std::vector<SelectedEndpoint> endpoints;
    std::vector<Cut> changed;
    std::set<Id> removed;
    std::map<Cut, Id> protectedPrefixes;
    std::map<Id, std::vector<Id>> publicationPrefixes;
    struct PublicationSpan {
        Id publication;
        Cut original;
        Id continuation;
        // Exact selected word intervals on which the certificate depends.
        // Mutating one invalidates this certificate, including deletion or
        // reordering of an existing endpoint. No cached proof survives it.
        std::map<Cut, std::vector<Id>> words;
    };
    std::vector<PublicationSpan> publicationSpans;
    Id insert(Cut, Id, Command, EndpointPurpose, Id, Id);
};

// Pure ordering query over a selected fragment. It consumes immutable control
// and physical-key identities, not a completion snapshot or a binding policy.
// Requires the constructor's admitted issue-only program and its unique
// CausalFrontier key population; typed effects use separate admission paths.
// Coverage, token legality and future edits require separate certificates.
bool certifyPublicationOrder(const Program&, const Control&, const Ledger&,
    const std::vector<EventIdentity>&, Id publication, Cut target, Id offset,
    std::vector<Cut>& crossed);

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
    // Marks a whole-original-graph pending-payload solve; reuse additionally
    // requires success. A partial traversal cannot certify a sibling.
    bool contextualFixedPoint = false;
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
    std::size_t sharedReturns = 0;
    // Multi-input retained-reader admission requires the staged ledger to
    // cover every payload requirement on this producer. Otherwise removing
    // one local repair could move a remaining repair past a new write.
    std::set<Pipe> repairFreeProducers;
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
// One physical use in original control. Successor requirements retain all
// reader engines and original deadlines; they do not allocate private channels.
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
    // Minimal lifecycle extension: candidate gap after this physical source
    // use, including a same-visit segment of a loop. This is not by itself a
    // proof of the last read, matching participation, or sufficient coverage.
    // Cell, access mode and source occurrence live in relationship/access.
    // This boundary grants neither completion nor event-rearming credit.
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
    const StorageLifecycle& lifetime(Cut site, unsigned cell) const;
    Cut recurringRelease(Cut site, unsigned cell) const;
    std::vector<SelectedLifecycleDemand> demandsAt(
        Cut site, const std::vector<FrontierRequirement>& required) const;
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
    mutable std::map<std::pair<Cut, unsigned>, StorageLifecycle> lifetimes;
    std::vector<std::vector<RequirementFrontier>> byDeadline;
};
// A storage/control qualifier: it returns requirements and original frontiers,
// not commands or physical key choices. Empty means ordinary F1--F8 applies.
std::vector<RecurringRequirement> qualifyCyclicFrontiers(
    const Program&, const Control&, const RequirementFrontiers&,
    bool allowGuardedEpisodes = true, bool movingFrontiers = true,
    bool shareReaderReturns = true, SelectedWork* work = nullptr);

struct Group {
    Pipe source = Pipe::S;
    Cut publication = NoAnalysisId;
    // Motivating demands determine the endpoint boundaries. Coverage may also
    // contain other current residual classes proved by those source prefixes;
    // it does not widen the motivation or grant credit before actual replay.
    std::vector<FrontierRequirement> requirements;
    std::set<Id> coverage;
    bool common = false;
    bool atWordStart = false;
    bool finalReadSource = false;
    bool bindingCertified = false;
    // A participation-qualified set of alternative early source cuts. It uses
    // one virgin directional key, not one key per branch. Empty means the
    // original single-cut/F7 path. The candidate is tied to the selected map.
    std::vector<Cut> publications = {};
    Id forwardKey = NoAnalysisId;
    uint64_t version = 0;
    Cut entryAcquisition = NoAnalysisId;
    Id entryReturnKey = NoAnalysisId;
    bool entryRepeats = false;
    bool choiceAcquisition = false;
};

class Constructor {
    friend struct ReplayTestAccess;
public:
    explicit Constructor(const Program&, SelectedOptions = {});
    SelectedPlan run(const Commands&);

private:
    const Program& program;
    SelectedOptions options;
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
    std::set<Id> closedKeys, recurringKeys, entryProtocolKeys;
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
    bool join(State&, const State&, bool* changed = nullptr);
    bool word(State&, Cut, Replay&);
    bool payload(State&, Cut, Replay&, bool pending = false);
    bool contextualReplay();
    bool fixedComponent(Id, const std::vector<State>&, Replay&, std::vector<State>&);
    bool partialComponent(Id, const std::vector<State>&, Replay&);
    bool partialSite(Id, Cut, std::vector<State>&, Replay&);
    bool replay();
    // The component prefix an update may keep, shared by both replay paths.
    Id reusablePrefix(SelectedReplayTrace* = nullptr);
    std::vector<bool> reusableComponents(SelectedReplayTrace*);
    bool advance();
    void registerSource();
    void refreshSources(Cut = NoAnalysisId);
    bool update();
    State& currentState();
    std::vector<FrontierRequirement> residual() const;
    std::map<Id, unsigned> reasons(Cut) const;
    std::vector<Group> groups(const std::vector<FrontierRequirement>&, RequirementStage);
    Group sourceGroup(Pipe, const std::vector<FrontierRequirement>&,
                      const std::vector<FrontierRequirement>&, const std::set<Id>* promotion = nullptr);
    std::set<Id> coverage(Cut, Pipe, const std::vector<FrontierRequirement>&, bool atStart = false) const;
    Id reusableAtStart(Cut, Pipe, Pipe) const;
    Id helperFreeBinding(const Group&, Pipe) const;
    bool freshBetween(Cut, Cut, Id) const;
    bool finalReadFrontier(Pipe, const std::vector<FrontierRequirement>&, Group&);
    bool finalReadGap(Cut, Pipe, const std::vector<FrontierRequirement>&, Id = NoAnalysisId);
    bool sourceFrontier(Pipe, const std::vector<FrontierRequirement>&, Group&,
                        const std::vector<FrontierRequirement>&, const std::set<Id>*) const;
    bool loopEntryFrontier(Pipe, const std::vector<FrontierRequirement>&, Group&,
                           const std::vector<FrontierRequirement>&, const std::set<Id>*);
    std::set<Id> choiceCoverage(Cut, const Control::ChoiceFrontier&, Pipe,
        const std::vector<FrontierRequirement>&) const;
    bool choiceConsumerFrontier(Pipe, const std::vector<FrontierRequirement>&, Group&,
                                const std::vector<FrontierRequirement>&, const std::set<Id>*);
    bool consume();
    bool bind(Group&, RequirementStage);
    bool preservePublicationPrefixes(SelectedDecision&);
    bool preservePublicationSpan(Id, const SelectedDecision&);
    Cut lifecycleRelease(const SelectedDecision&) const;
    bool publicationGapCovers(Id, Id, const std::vector<FrontierRequirement>&) const;
    bool publicationPositionCovers(Id, Cut, Id, const std::vector<FrontierRequirement>&) const;
    bool publicationEditValid(const Ledger&);
    bool edge(Pipe, Pipe, Cut&, bool, SelectedDecision&, Id certifiedKey = NoAnalysisId);
    bool acknowledgment(Pipe, Pipe, Cut&, Id&, SelectedDecision&);
    bool joinedAcknowledgment(Pipe, Pipe, Cut, Id&, SelectedDecision&);
    bool needsCommonAcknowledgment(const State&) const;
    bool deferCommonRearming(Id key, Id acquisition, const SelectedDecision&);
    Cut deferredReturnCut(Id acquisition, Cut publication, Id key) const;
    void observeDeferredRearming(const SelectedDecision&);
    // Index obligations by the actual consumption endpoint, not engine pair.
    std::map<Id, Id> deferredByAcquisition;
    Id reusable(Pipe, Pipe, const State&);
    bool canPublish(const State&, Id) const;
    bool canPublishAt(Cut, Id) const;
    bool consumptionBeforeNextPublication(Cut, Cut, Id, const Ledger* = nullptr);
    bool availableKey(Id) const;
    Id splitReturnKey(Cut, Id);
    bool inactiveReservation(Cut, Cut, Id);
    bool inactiveClosedReservation(Cut, Cut, Id);
    bool prepareClosedReservation(Cut, Cut, Id);
    bool prepareDormantKey(Cut, Cut, Id);
    bool borrowedInterval(Cut, Cut, Id);
    bool suspendedReturnKey(Id) const;
    bool crossControlReturn(Id, Cut, Id, Id);
    bool clearInterval(Id, Cut, Cut) const;
    std::vector<Pipe> route(Pipe, Pipe) const;
    std::optional<bool> splitRelay(const Group&, Pipe, RequirementStage);
    bool recurring(const std::vector<RecurringRequirement>&);
    bool finish();
};

Id accessClass(const FrontierRequirement&);
bool identical(const Command&, const Command&);
std::vector<Id> unionIds(const std::vector<Id>&, const std::vector<Id>&);

} // namespace mlir::pto::oahs::selected
#endif
