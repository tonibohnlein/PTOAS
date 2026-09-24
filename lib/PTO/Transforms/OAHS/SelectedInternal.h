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
Id keyIndex(const CausalFrontier&, const Command&);
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
struct PublicationBoundary {
    ProofOutcome outcome = ProofOutcome::Unknown;
    Cut word = NoAnalysisId;
    std::string reason;
    bool proved() const { return outcome == ProofOutcome::Proved; }
};
struct Control {
    detail::ControlGraph graph;
    std::vector<std::vector<Id>> predecessors;
    // Construction-only, zero-or-more loop summary traversal. Acceptance uses
    // graph's original edges, not these exit-summary edges or hypotheses.
    std::vector<std::vector<Id>> constructionEdges, headerAccesses;
    std::size_t unsummarizedBackedges = 0, finiteOccurrenceTransitions = 0;
    uint64_t transitionClassificationWork = 0;
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
    // Endpoint sets describe emitted words, not command multiplicities.
    // Repeated/canonical aliases in a query denote the same endpoint.
    const OccurrenceCorrespondence& correspondence(
        const std::vector<Cut>& publications, const std::vector<Cut>& acquisitions, Id budget = 65536) const;
    const PublicationBoundary& publicationAfter(Cut) const;
    mutable uint64_t occurrenceAnalysisSites = 0;
    mutable uint64_t boundaryAnalysisSites = 0;

private:
    PublicationBoundary findPublicationAfter(Cut) const;
    mutable std::map<Cut, PublicationBoundary> publicationBoundaries;
    OccurrenceCorrespondence pairOccurrences(
        const std::vector<Cut>&, const std::vector<Cut>&, Id budget) const;
    mutable std::map<std::tuple<std::vector<Cut>, std::vector<Cut>, Id>,
                     OccurrenceCorrespondence> correspondences;
};

// Stable original-word gap. Missing neighbors mean word beginning/end.
// A proof is tied to the prepared packet revision, not just these identities.
struct WordGap {
    Cut cut = NoAnalysisId;
    Id left = NoAnalysisId, right = NoAnalysisId;
};
struct SourceGapQualification {
    ProofOutcome outcome = ProofOutcome::Unknown;
    WordGap gap;
    Pipe source = Pipe::S, observer = Pipe::S;
    Cut deadline = NoAnalysisId;
    uint64_t version = 0;
    std::vector<FrontierState> prefixes;
    mutable std::vector<bool> futureSites;
    std::string reason;
    bool proved() const { return outcome == ProofOutcome::Proved; }
};
struct PacketEndpoint {
    Cut cut;
    Command command;
    EndpointPurpose purpose;
    Id request = NoAnalysisId, acknowledges = NoAnalysisId;
    std::optional<WordGap> gap = {};
    // Reference to an earlier endpoint in this packet, resolved before commit.
    Id acknowledgesPacket = NoAnalysisId;
    // Restore this inactive identity, preserving all original provenance.
    Id restore = NoAnalysisId;
    // Explicit relocation of an inactive consumption-only WAIT. Ordinary
    // restoration still requires the exact original word and provenance.
    bool relocate = false;
};
using OrderedPacket = std::vector<PacketEndpoint>;
class Ledger;
class PacketView;
class PreparedPacket {
public:
    bool valid() const { return ready; }
    const std::string& reason() const { return error; }
    Id size() const { return ordered.size(); }

private:
    friend class Ledger;
    friend class PacketView;
    bool ready = false;
    std::string error;
    const Ledger* owner = nullptr;
    uint64_t version = 0;
    Id firstEndpoint = NoAnalysisId;
    std::vector<SelectedEndpoint> endpoints;
    std::vector<Id> ordered, restored;
    std::map<Id, SelectedEndpoint> relocated;
    // Frozen original offsets are derived from stable gaps once. Multiple
    // insertions at one gap retain packet order; no word is copied to prepare.
    std::map<Cut, std::map<Id, std::vector<Id>>> insertions;
};

// Publication preservation is an ordering certificate, never completion credit.
// Roots describe original issue/completion dependencies and exact selected event
// identities. Admission roots are immutable even after a permitted repair.
class PublicationSupport {
public:
    struct Probe {
        uint64_t version = 0;
        // Only affected existing source contracts; not own coverage, event
        // legality, or complete normal-class qualification.
        bool preserved = true;
        std::map<Cut, std::vector<Id>> states;
        std::map<std::pair<Id, Cut>, Id> roots;
    };
    PublicationSupport(const Program&, const Control&, const CausalFrontier&);
    Probe inspect(const Ledger&, const PacketView*, const std::vector<Cut>&);
    void accept(const Ledger&, Probe);
    void refresh(const Ledger&);
    const std::vector<SelectedPublicationSupport>& records() const { return contracts; }
    uint64_t sites = 0, commands = 0, checks = 0, nodes = 0;
private:
    enum class Kind { Union, Choice, Publication, Receipt, Issue, Completion };
    using Node = std::tuple<Kind, Id, std::vector<Id>>;
    const Program& program;
    const Control& control;
    const CausalFrontier& frontier;
    std::map<Node, Id> expressions;
    std::vector<std::vector<Id>> outgoing;
    std::map<std::pair<Id, Cut>, Id> byOccurrence;
    std::vector<SelectedPublicationSupport> contracts;
    uint64_t version = 0;
    bool initialized = false;
    Id expression(Kind, Id, std::vector<Id>);
    Id unite(Id, Id);
    std::vector<Id> incoming(Cut, const Probe&);
    void command(std::vector<Id>&, const SelectedEndpoint&, Cut, Probe&);
    void payload(std::vector<Id>&, Cut);
};

class OwnedPacket {
    friend class Constructor;
    friend struct ReplayTestAccess;
    bool qualified = false;
    PreparedPacket prepared;
    std::vector<Id> restoredWaits;
    Id restoredEndpoints = 0;
    std::map<Id, Id> deadlines;
    std::map<Id, std::string> fallbackReasons;
    // Complete restoration effects are included before a recipe is classified.
    std::optional<PublicationSupport::Probe> publicationProbe;
};

class Ledger {
public:
    // The canonical table belongs to the enclosing Control and outlives this.
    Ledger(const Program&, const std::vector<Cut>&);
    bool initialize(const Commands&, std::string&);
    Id append(Cut, Command, EndpointPurpose, Id request = NoAnalysisId, Id ack = NoAnalysisId);
    WordGap tail(Cut) const;
    std::optional<WordGap> gapAfter(Id) const;
    std::map<Id, WordGap> gapsAfter(const std::vector<Id>&) const;
    const std::vector<Id>& word(Cut) const;
    const SelectedEndpoint& endpoint(Id) const;
    Commands commands() const;
    const std::vector<Id>& eventUses(const EventIdentity&) const;
    bool hasDormantUses(const EventIdentity&) const;
    std::optional<PacketEndpoint> restoration(Id, const WordGap&) const;
    std::optional<PacketEndpoint> relocateAcknowledgment(Id, const WordGap&) const;
    PreparedPacket preparePacket(const OrderedPacket&) const;
    std::optional<Commands> withPacket(const PreparedPacket&) const;
    std::optional<PacketView> packetView(const PreparedPacket&) const;
    std::vector<Id> appendPacket(const PreparedPacket&);
    bool active(Id id) const { return !removed.count(id); }
    void erase(Id);
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
    std::map<std::tuple<Pipe, Pipe, unsigned>, std::vector<Id>> byEvent;
    std::map<std::tuple<Pipe, Pipe, unsigned>, Id> dormantEvents;
    void recordEvent(const SelectedEndpoint&);
    void setDormant(Id, bool);
    Id insert(Cut, Id, Command, EndpointPurpose, Id, Id);
};

// Immutable overlay of the exact checked packet. Unchanged words and endpoint
// records stay shared with the ledger; only changed words are materialized.
class PacketView {
    friend class Ledger;
public:
    const std::vector<Id>& word(Cut) const;
    const SelectedEndpoint& endpoint(Id) const;
    uint64_t version() const;
    std::vector<Cut> changedCuts() const;
private:
    const Ledger* ledger = nullptr;
    const PreparedPacket* packet = nullptr;
    const std::vector<Cut>* canonical = nullptr;
    std::map<Cut, std::vector<Id>> words;
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
    // Keys publishable before any occurrence of an indexed selected receipt.
    // An absent key means no visited incoming state proved it publishable.
    std::map<Id, std::set<Id>> beforeReceiptPublishable;
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
    // Original physical uses witnessing the prescribed endpoint milestones.
    std::vector<std::pair<Cut, Cut>> publicationOrigins, acquisitionOrigins;
    mutable std::optional<bool> physicalQualified;
    bool storageRelease = false;
    // Producer deadlines whose preceding local repair can be replaced by this
    // complete cycle. Require scoped residual support before commitment.
    std::vector<Cut> supportSeeds;
};
// Logical roles are shared by exact endpoint identity. A family names its own
// finite support recipe; sharing a role does not activate every other family.
struct RecurringFamily {
    Id owner = NoAnalysisId;
    std::vector<unsigned> cells;
    std::vector<Id> roles;
    std::vector<RecurringRequirement> support;
    std::vector<Cut> deadlines;
};
struct RecurringFrontiers {
    std::vector<RecurringRequirement> roles;
    std::vector<RecurringFamily> families;
    std::map<std::pair<Cut, unsigned>, std::vector<Id>> at;
};
struct OrdinaryProducerSupport {
    Cut consumer = NoAnalysisId;
    Id access = NoAnalysisId;
    Pipe source = Pipe::S;
    std::shared_ptr<const std::vector<Cut>> seeds;
    bool operator<(const OrdinaryProducerSupport& b) const
    {
        const auto a = std::tie(consumer, access, source);
        const auto other = std::tie(b.consumer, b.access, b.source);
        if (a != other) { return a < other; }
        if (seeds == b.seeds) { return false; }
        const std::vector<Cut> empty;
        return (seeds ? *seeds : empty) < (b.seeds ? *b.seeds : empty);
    }
};
struct ProducerSupportScope {
    std::array<std::set<Id>, PipeCount> classes;
    std::array<std::vector<Cut>, PipeCount> seeds;
    std::array<std::shared_ptr<const std::vector<Cut>>, PipeCount> seedHandles;
    std::vector<Cut> consumers;
    std::vector<OrdinaryProducerSupport> ordinary;
};
// Immutable, conditional consequences of one finite two-role induction system.
// These masks are opportunity proofs; only selected replay supplies live credit.
struct RecurringCertificate {
    bool complete = false;
    std::string reason;
    std::map<Cut, std::set<Id>> guaranteed;
    std::map<Cut, std::vector<std::pair<Id, Command::Kind>>> words;
};
struct RecurringPacket {
    OwnedPacket packet;
    std::vector<RecurringRequirement> requests;
    std::vector<Id> keys;
    std::array<std::set<Id>, PipeCount> supportClasses;
    std::vector<bool> supportConsumers;
    Replay evaluated;
    bool localCertificate = false;
    std::map<Cut, std::set<Id>> guaranteed;
    bool normalWords = false;
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
// Exact eligibility at an existing original endpoint, not a synthesized guard.
// NoHit is relative to this candidate occurrence; Unknown retains adjacent may facts.
struct ReaderEndpoint {
    enum class Status { Unknown, NoHit, Exact } status = Status::Unknown;
    Cut site = NoAnalysisId;
    Id observation = NoAnalysisId;
    bool hit() const { return status == Status::Exact; }
};
struct ReaderFrontiers {
    Id owner = NoAnalysisId;
    bool originalInterval = false;
    ReaderIntervalQuery interval;
    const std::vector<Id>* ownerInterfaces = nullptr;
    ReaderEndpoint first, final;
    PhysicalUseSummary preceding, following;
    std::string reason;
    bool proved() const
    {
        return first.status != ReaderEndpoint::Status::Unknown && final.status != ReaderEndpoint::Status::Unknown;
    }
};
// One original storage requirement with both of its placement bounds retained.
// This is immutable analysis metadata: it grants no completion receipt, event
// token, or permission to merge requirements that happen to use one pipeline.
struct RequirementFrontier {
    StorageRelationship relationship;
    // Cheap flags are queried lazily; detailed witness paths are not needed
    // to select a provider. The original relationship is indexed eagerly.
    mutable unsigned reasons = AdditionalOverlap;
    mutable bool classified = false;
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
};
// A request for an original owner to expose an endpoint role. Collection is
// independent of channel selection, event capacity and refinement order.
struct EndpointRequirement {
    enum Role { FirstConsumer, FirstWrite, FinalReader } role = FirstConsumer;
    Id owner = NoAnalysisId;
    StorageOrigin access;
    StorageRelationship relationship;
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
    const ReaderFrontiers& readerBoundaries(Cut site, unsigned cell, Id owner = NoAnalysisId) const;
    const std::vector<EndpointRequirement>& endpoints(Id originalOwner) const;
    bool needsOccurrenceSeparation(Id originalOwner) const;
    uint64_t endpointClassificationWork() const { return endpointWork; }
    const PhysicalUseFrontier& nextUses(
        const std::vector<Cut>& starts, unsigned cell, const std::vector<Cut>& stops) const;
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
    mutable std::map<std::tuple<Id, Cut, unsigned>, ReaderFrontiers> readerFrontiers;
    std::vector<std::vector<RequirementFrontier>> byDeadline;
    mutable bool endpointsIndexed = false;
    mutable uint64_t endpointWork = 0;
    mutable std::map<std::tuple<Cut, unsigned, bool>, bool> separationByUse;
    mutable std::map<Id, std::vector<EndpointRequirement>> byOwner;
    ReaderFrontiers originalReaderBoundaries(Cut, unsigned, Id) const;
    mutable bool pairedReadersIndexed = false;
    mutable std::vector<ObservedLoopOccurrence> readerOwnerInterfaces;
    mutable std::map<std::pair<Id, Cut>, std::vector<Id>> pairedReaderInterfaces;
    using PredicateKey = std::tuple<unsigned, Id, uint64_t>;
    mutable std::map<Cut, std::map<PredicateKey, uint64_t>> readerPredicateFacts;
};
// A storage/control qualifier: it returns requirements and original frontiers,
// not commands or physical key choices. Empty means ordinary F1--F8 applies.
RecurringFrontiers qualifyCyclicFrontiers(
    const Program&, const Control&, const RequirementFrontiers&);

struct Group {
    Pipe source = Pipe::S;
    Cut publication = NoAnalysisId;
    std::vector<FrontierRequirement> requirements, supporting;
    std::set<Id> coverage;
    bool common = false;
    // A participation-qualified set of alternative early source cuts. It uses
    // one virgin directional key, not one key per branch. Empty means the
    // original single-cut/F7 path. The candidate is tied to the selected map.
    std::vector<Cut> publications = {};
    Id forwardKey = NoAnalysisId, repairKey = NoAnalysisId;
    uint64_t version = 0;
    Cut entryAcquisition = NoAnalysisId;
    Id entryReturnKey = NoAnalysisId;
    bool entryRepeats = false;
    std::optional<OwnedPacket> packet;
};

struct DueObligation {
    std::set<Id> classes;
};
struct RealizationSupport {
    enum Kind { Completion, Consumption, Induction, OrdinaryRepair } kind = Completion;
    Id identity = NoAnalysisId;
    Cut deadline = NoAnalysisId;
};
struct CertifiedRealization {
    unsigned placementClass = 0;
    uint64_t version = 0;
    bool known = false;
    Group ordinary;
    Cut sourceMilestone = NoAnalysisId;
    WordGap selectedSourceGap;
    Id supportingReceipt = NoAnalysisId;
    std::optional<RecurringPacket> recurring;
    std::vector<Id> families, newRoles;
    std::vector<RealizationSupport> support;
    std::set<Id> coverage, physicalCoverage, ownCoverage;
    using Order = std::tuple<Id, Id, Cut, Pipe, Pipe, std::vector<Cut>, std::vector<Cut>>;
    Order order;
    std::vector<std::tuple<Pipe, Pipe, std::vector<Cut>, std::vector<Cut>>> shape;
};
Id selectRealization(const std::vector<CertifiedRealization>&);

// Each tracked fallback references its actual consumption, even when several
// generations reuse a key. Activity/completion remain ledger/frontier facts.
struct RearmingObligation {
    Id consumption = NoAnalysisId, forwardKey = NoAnalysisId;
    Id publication = NoAnalysisId, acquisition = NoAnalysisId;
    bool required = false, deferred = false;
    Id supportingReceipt = NoAnalysisId, supportRevision = NoAnalysisId;
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
    PublicationSupport publicationSupport;
    SelectedPlan result;
    Replay cache;
    // Original scope of an admitted producer-support obligation. The classes
    // include every potentially preceding access, not just LatestOrigins.
    // Only stabilized contextual replay can discharge these obligations.
    std::array<std::set<Id>, PipeCount> producerSupportClasses;
    std::vector<bool> producerSupportConsumers;
    std::vector<bool> finalized;
    std::map<Cut, std::vector<Id>> sourcesAtCut;
    Id activeComponent = 0, activeOffset = 0;
    Cut current = NoAnalysisId;
    // Roles are stable within a selected loop component. A reverse acknowledgment
    // key must not be borrowed as another recurring channel's forward key.
    std::map<std::pair<Pipe, Pipe>, std::pair<Id, Id>> closedBindings;
    std::set<Id> closedKeys, recurringKeys;
    RecurringFrontiers recurringFrontiers;
    std::vector<bool> activeFamilies;
    struct RecurringAttempt {
        uint64_t version = 0;
        bool evaluated = false;
        std::set<Cut> improving;
    };
    std::vector<RecurringAttempt> attemptedFamilies;
    std::map<Id, Id> activeRoles;
    bool recurringBaseline = false;
    std::map<Id, ProducerSupportScope> recurringScopes;
    struct SupportLinks {
        uint64_t version = 0;
        bool complete = false;
        std::vector<Id> families;
        std::vector<OrdinaryProducerSupport> ordinary;
        std::string reason;
    };
    std::map<Id, SupportLinks> recurringLinks;
    std::map<std::pair<Pipe, std::vector<Cut>>, std::shared_ptr<const std::vector<Cut>>> producerSeedHandles;
    std::map<std::pair<Pipe, std::shared_ptr<const std::vector<Cut>>>, std::set<Cut>> ordinaryFenceSites;
    std::map<std::tuple<Pipe, std::shared_ptr<const std::vector<Cut>>, Cut>, std::set<Id>> ordinaryFenceMay;
    std::vector<uint64_t> ordinaryVisitStamp;
    uint64_t ordinaryVisitEpoch = 0;
    std::map<std::pair<Id, bool>, RecurringCertificate> recurringCertificates;
    std::map<unsigned, std::vector<std::pair<Id, Id>>> projectionAccesses;
    bool projectionIndexed = false;
    const RecurringCertificate& recurringCertificate(Id, bool normal = false);
    bool qualifyRecurringInterface(const std::vector<Id>&, RecurringPacket&, const ProducerSupportScope&);
    std::optional<std::map<Cut, std::set<Id>>> ordinarySupportGuarantee(
        const ProducerSupportScope&, const PacketView&);

    ProducerSupportScope producerScope(const std::vector<RecurringRequirement>&);
    bool supportsRecurring(Id family, Cut, const FrontierRequirement&) const;
    const SupportLinks& recurringSupportLinks(Id family);
    // Contextual state propagation is also needed for one-shot loop-entry
    // receipts. It does not reserve a physical key or establish rearming.
    bool needsContextualReplay = false;
    // Each fallback return records a pending rearming obligation. A later
    // necessary transfer may discharge it; memory requirements remain separate.
    std::map<Id, RearmingObligation> rearming;
    std::map<std::pair<Pipe, Pipe>, std::vector<Id>> pendingRearming;
    std::map<Id, Id> helperOwners;
    std::map<Id, std::vector<Id>> rearmingByKey;
    std::map<Id, std::set<Id>> deferredByKey, latentReturns;
    std::map<std::pair<Pipe, Pipe>, std::vector<Id>> necessaryReturns;
    // Append-only populations already paired for each direction. Old helpers
    // see only new returns; new helpers see all returns, in the original order.
    std::map<std::pair<Pipe, Pipe>, std::pair<Id, Id>> pairedReturns;
    std::optional<WordGap> restorationDeadline(
        const RearmingObligation&, const OrderedPacket&, Id, const std::vector<Id>&,
        std::map<Cut, std::map<Id, Id>>&, std::string&);
    bool restoreReturns(Id);
    bool restoreRearming(Id, Cut);
    bool settleRearming(const SelectedDecision&);
    bool returnBeforeUse(Id helperWait, Id necessaryWait);
    void rememberReturn(Id publication, Id acquisition);

    bool fail(SelectedFailure, std::string, Cut = NoAnalysisId);
    State initial() const;
    bool join(State&, const State&);
    bool word(State&, Cut, Replay&, const PacketView* = nullptr);
    bool payload(State&, Cut, Replay&, bool pending = false);
    bool contextualReplay();
    Replay evaluateContextual(const PacketView*, const std::array<std::set<Id>, PipeCount>&,
                              const std::vector<bool>&, Id resume);
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
                      const std::vector<FrontierRequirement>&, bool structured = true);
    std::set<Id> sourceHistoryCoverage(
        const std::vector<Cut>&, Pipe, const std::vector<FrontierRequirement>&) const;
    std::set<Id> coverage(Cut, Pipe, const std::vector<FrontierRequirement>&) const;
    bool freshBetween(Cut, Cut, Id) const;
    struct SourceFrontierFacts {
        std::vector<Cut> publications;
        std::set<Id> crossedClasses;
    };
    using SourceFrontierKey = std::tuple<uint64_t, Cut, Pipe, std::set<Id>>;
    uint64_t sourceFrontierVersion = NoAnalysisId;
    std::map<SourceFrontierKey, std::optional<SourceFrontierFacts>> sourceFrontierCache;
    std::optional<SourceFrontierFacts> discoverSourceFrontier(
        Pipe, const std::vector<FrontierRequirement>&);
    bool earlierLoopEntrySource(Pipe, const std::vector<FrontierRequirement>&) const;
    bool sourceFrontier(Pipe, const std::vector<FrontierRequirement>&,
                        const std::vector<FrontierRequirement>&, Group&);
    bool loopEntryFrontier(Pipe, const std::vector<FrontierRequirement>&,
                           const std::vector<FrontierRequirement>&, Group&);
    bool consume();
    bool ensureRecurringBaseline();
    std::vector<DueObligation> normalizeDue(const std::vector<FrontierRequirement>&);
    std::vector<Id> normalCellIdentities;
    std::set<Id> normalizedCoverage(const std::vector<DueObligation>&,
        const std::vector<FrontierRequirement>&, const std::set<Id>&) const;
    std::optional<CertifiedRealization> normalOrdinary(
        Group, const std::vector<FrontierRequirement>&, const std::vector<DueObligation>&,
        bool repair = false, std::optional<WordGap> prescribedGap = {},
        Id prescribedKey = NoAnalysisId, const SourceGapQualification* prescribedFacts = nullptr);
    std::optional<CertifiedRealization> normalAlternative(
        const Group&, const std::vector<FrontierRequirement>&, const std::vector<DueObligation>&);
    std::optional<CertifiedRealization> normalCorridor(
        const Group&, const std::vector<FrontierRequirement>&, const std::vector<DueObligation>&);
    std::optional<CertifiedRealization> normalCommonCut(
        const Group&, const std::vector<FrontierRequirement>&, const std::vector<DueObligation>&);
    void indexSelectedReturns();
    Id indexedReturnEndpoints = 0;
    std::map<std::pair<Pipe, Pipe>, std::map<Id, std::vector<Id>>> selectedReturnGaps;
    std::optional<CertifiedRealization> normalRecurring(
        const std::vector<Id>&, const std::vector<FrontierRequirement>&, const std::vector<DueObligation>&);
    std::optional<bool> selectNormal(const std::vector<DueObligation>&);
    bool preservePublications(const OwnedPacket&);
    bool physicalMilestones(const RecurringRequirement&) const;
    struct SupportClosure {
        bool complete = false;
        std::vector<Id> families, roles;
        ProducerSupportScope scope;
    };
    std::shared_ptr<const SupportClosure> normalSupport(Id);
    uint64_t normalSupportVersion = NoAnalysisId;
    std::map<Id, std::shared_ptr<const SupportClosure>> normalClosures;
    uint64_t publicationReachVersion = NoAnalysisId;
    std::vector<bool> publicationReach;

    bool bind(Group&, RequirementStage);
    std::optional<OwnedPacket> prepareOwnedPacket(const OrderedPacket&, const std::vector<Id>& obligations = {},
                                                   bool placeAtDeadline = true);
    std::optional<OwnedPacket> qualifyOwnedPacket(const OrderedPacket&, bool alwaysCheck = false,
                                                 const std::vector<Id>& obligations = {},
                                                 AnalysisResult* refusal = nullptr);
    bool acceptOwnedPacket(OwnedPacket&, const AnalysisResult&) const;
    bool commitOwnedPacket(const OwnedPacket&, SelectedDecision&);
    bool unownedKey(Id) const;
    bool helperFreeKey(Id) const;
    std::optional<bool> dormantTransfer(Pipe, Pipe, Cut, bool, SelectedDecision&);
    bool commitPacket(const OrderedPacket&, SelectedDecision&);
    bool edge(Pipe, Pipe, Cut&, bool, SelectedDecision&);
    bool acknowledgment(Pipe, Pipe, Cut&, Id&, SelectedDecision&, bool&, OrderedPacket&);
    std::optional<bool> joinedAcknowledgment(Pipe, Pipe, Cut, SelectedDecision&);
    bool terminalCommonCut();
    bool needsCommonAcknowledgment(const State&);
    void deferReturn(Id);
    bool canDeferCommonReturn() const;
    bool latentPublicationAfter(Id anchor, Pipe observer) const;
    Id reusable(Pipe, Pipe, const State&);
    bool canPublish(const State&, Id) const;
    SourceGapQualification sourceGapFacts(
        const WordGap&, Pipe, Pipe, const std::vector<FrontierRequirement>&);
    bool sourceKeyNeighbors(const SourceGapQualification&, Id);
    bool selectedKeyUsesOutside(const std::vector<bool>& futureSites, Id key,
        std::map<Cut, bool>& wordIntersects);
    bool sourceGapKey(const SourceGapQualification&, Id);
    bool fixedBoundaryPacket(const SourceGapQualification&, Group&);
    std::optional<WordGap> earlyPublicationMilestone(Cut, Pipe) const;
    bool clearInterval(Id, Cut, Cut) const;
    std::vector<Pipe> route(Pipe, Pipe) const;
    std::optional<RecurringPacket> prepareRecurring(const std::vector<RecurringRequirement>&, std::string&,
                                                     const ProducerSupportScope* support = nullptr,
                                                     const std::vector<Id>* families = nullptr,
                                                     bool normalOnly = false);
    bool commitRecurring(RecurringPacket&);
    bool activateRecurring();
    bool finish();
};

Id accessClass(const FrontierRequirement&);
bool identical(const Command&, const Command&);
std::vector<Id> unionIds(const std::vector<Id>&, const std::vector<Id>&);

} // namespace mlir::pto::oahs::selected
#endif
