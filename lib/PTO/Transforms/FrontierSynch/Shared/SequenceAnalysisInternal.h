// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared implementation state for regional adapters, queries and endpoint preparation.
#ifndef PTO_FRONTIERSYNCH_SEQUENCEANALYSISINTERNAL_H
#define PTO_FRONTIERSYNCH_SEQUENCEANALYSISINTERNAL_H
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Structured/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "../../InsertSync/SyncEffectRanges.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <map>
#include <array>
#include <numeric>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
bool sameOriginalCoordinates(const RegionalAnalysis& a, const RegionalAnalysis& b);
RegionalAnalysis guardRegionalArm(RegionalAnalysis body, RegionExpressions::Id guard);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> maskRegionalArmPlan(
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepared,
    const std::shared_ptr<RegionExpressions>& arena, RegionExpressions::Id guard);
using Expr = RegionExpressions::Id;
struct Pattern {
    SyncStorageCell range;
    uint32_t type = 0;
    uint64_t residue = 0;
    uint64_t period = 1;
    bool read = false;
    bool write = false;
};
struct RegionalAllocationGroup;
struct OrderCoveredPeriodicResult;
class InvocationPrefixCollection;
struct BoundaryPartition;
struct Child {
    std::shared_ptr<const MathematicalResult> prefixMathematics;
    // Owned proof for this original root sequence only; never a disjoint-effect certificate.
    std::shared_ptr<const OrderCoveredPeriodicResult> orderCovered;
    RegionalAnalysis regional;
    RegionalCost costs;
    scf::ForOp loop;
    Expr trips = 0;
    std::vector<TemplateEndpointAnchor> anchors;
    std::vector<Pattern> patterns;
    SmallVector<std::size_t> dischargedEffects;
    std::shared_ptr<FiniteAnalysis> finite;
    PeriodicAnalysis periodic;
    NumericTemplateEndpoints endpoints;
    std::shared_ptr<const NumericTemplate> numericTemplate;
    std::shared_ptr<const NumericTemplateEndpoints> numericRecipe;
    std::optional<std::size_t> originalNode;
    std::optional<Expr> originalArmGuard;
    // Producer-owned disjoint visit slices; valid only at the retained endpoints.
    std::shared_ptr<const BoundaryPartition> boundaryPartition;
    std::size_t boundarySlice = 0;
};
struct Port {
    uint32_t child = 0;
    uint32_t type = 0;
    Expr ordinal = 0;
    std::vector<Expr> visits;
    RegionalEvent event(PeriodicEventKind kind = PeriodicEventKind::Start) const
    { return {type, ordinal, kind, visits}; }
};
struct Selected { uint32_t port = 0; Expr present = 0; };
struct CellBoundary {
    std::vector<Selected> firstWriters, lastWriters;
    std::map<uint32_t, std::vector<Selected>> firstReaders, lastReaders;
};
struct Crossing { uint32_t source = 0, target = 0; Expr guard = 0; };
struct SequenceNumericalNode;
struct SequenceCrossingPresence;
struct SequenceAnalysisState : std::enable_shared_from_this<SequenceAnalysisState> {
    // Primitive child adapters may capture their original composer's address.
    // An endpoint upgrade owns that immutable source while reusing its children.
    std::shared_ptr<const SequenceAnalysisState> upgradeSource;
    // Closure seals the endpoint-existence invariant of its retained crossings.
    // Source semantics are immutable after closure; only query caches and
    // prepare/prepareWithVisits/prepareFiltered may change. Semantic callbacks
    // (including endpointEventGuard) stay frozen. Adapters construct a new state.
    bool hasCertifiedCrossingPresence() const;
    LogicalResult collectPrefix(InvocationPrefixCollection& collection, std::string& error);
    const ArithmeticRegionalAllocationPolicy allocationPolicy;
    func::FuncOp function;
    const SyncInput* input = nullptr;
    std::shared_ptr<const FiniteOccurrenceInput> finiteInput;
    const ProgramRecognition* program = nullptr;
    std::shared_ptr<PhaseIndex> indexOwner;
    PhaseIndex& index;
    bool indexReady = false;
    // Semantic original-region identity for capability upgrades. Specialized
    // compositions have no such identity and retain their own exact route.
    std::optional<std::size_t> originalRegionId;
    // Diagnostic origin only; does not participate in proof/cache identity.
    std::string constructionOrigin = "direct-composition";
    std::optional<std::size_t> diagnosticRegion;
    Operation* constructionAnchor = nullptr;
    bool completeInvocation = true;
    bool reconstructPrerequisites = true;
    bool requireEndpoints = true;
    SequenceRegionResolver resolveOriginal;
    SmallVector<scf::ForOp> requiredOuterLoops;
    std::shared_ptr<RegionExpressions> arena;
    RegionExpressions& expressions;
    SequenceCost costs;
    std::array<AnalysisCostRecord, 2> crossingMethods;
    std::vector<std::string> crossingObligations;
    std::shared_ptr<NumericalChainMerge> numerical;
    std::shared_ptr<SequenceNumericalNode> numericalTree;
    // Root index IDs map to original event IDs; interior events stay queryable
    // through the hierarchy instead of being copied into every ancestor index.
    std::vector<uint32_t> numericalEvents;
    uint64_t numericalProjectedPorts = 0;
    std::shared_ptr<const SequenceAnalysis> numericalBoundary;
    std::vector<std::pair<uint32_t, PeriodicEventKind>> numericalChainKeys;
    NumericalChainQueryCost numericalQueryCost;
    // Mathematical construction failures only; endpoint attempts own their diagnostics.
    std::string error;
    std::string repeatedAttempt;
    uint64_t boundarySummaryVisits = 0;
    SequenceBridgeWork bridgeWork, nativeFrontierWork;
    // Bounded structural attribution; no query IDs survive a speculative attempt.
    struct CrossingOrderWork {
        uint64_t groups = 0, qualified = 0, dynamicQualified = 0, residual = 0, orderQueries = 0;
    } crossingOrderWork;
    struct CrossingQueryWork {
        uint64_t entries = 0, aliases = 0, suffixQueries = 0, zeroTails = 0, prefixQueries = 0;
        uint64_t correlatedChoices = 0;
        uint64_t incomingRequests = 0, incomingFacts = 0, incomingProofVisits = 0;
        uint64_t incomingArms = 0, incomingCoordinates = 0, incomingResidual = 0;
    } crossingQueryWork;
    void summarizeBoundaries(SequenceAnalysis& result);
    std::vector<Child> children;
    // A symbolic crossing relation cannot be represented as a sampled finite
    // port graph. Retain its exact regional adapter as a separate result.
    std::optional<RegionalAnalysis> relationalResult;
    uint64_t relationalCompositions = 0;
    // Pair-local finite crossing support does not cover the full symbolic effects.
    using StoragePair = std::tuple<uint32_t, std::size_t, uint32_t, std::size_t>;
    std::set<StoragePair> finiteCrossingPairs;
    std::vector<SyncStorageCell> cells;
    std::vector<Port> ports;
    std::map<std::tuple<uint32_t, uint32_t, Expr, std::vector<Expr>>, uint32_t> portIds;
    // Endpoint folding keeps executable selected coordinates, but queries
    // distribute through their original ports instead of asking a compact
    // child to reconstruct reachability for newly invented coordinates.
    // Registered only for new ports: both alternatives have smaller IDs.
    struct PortChoice { Expr choose; uint32_t yes, no; };
    std::map<uint32_t, PortChoice> portChoices;
    using SelectorKey = std::vector<std::pair<uint32_t, Expr>>;
    std::map<SelectorKey, std::vector<Selected>> selectorAlternatives;
    std::vector<Crossing> crossings;
    std::vector<Crossing> nativeValueCrossings;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> crossingIds;
    struct EntryLink { std::size_t source = 0, target = 0; Expr guard = 0; };
    std::map<uint32_t, std::vector<EntryLink>> incoming;
    using EventKey = std::tuple<uint32_t, uint32_t, Expr, PeriodicEventKind, std::vector<Expr>>;
    std::map<std::pair<EventKey, EventKey>, Expr> reachabilityCache;
    // One ordered map of participating children per cell; absent children have no entry.
    std::vector<std::map<uint32_t, CellBoundary>> boundaries;
    using NativeSelectors = std::map<uint32_t, std::vector<Selected>>;
    std::vector<NativeSelectors> nativeFirst, nativeLast;
    uint64_t childPreparationOperations = 0, crossingPreparationOperations = 0;
    explicit SequenceAnalysisState(func::FuncOp f, const SyncInput& i, const ProgramRecognition& p)
        : allocationPolicy(p.allocationPolicy), function(f), input(&i), program(&p),
          indexOwner(std::make_shared<PhaseIndex>()), index(*indexOwner),
          arena(std::make_shared<RegionExpressions>()), expressions(*arena) {}
    SequenceAnalysisState(func::FuncOp f, std::shared_ptr<RegionExpressions> a,
                          std::shared_ptr<PhaseIndex> shared = {},
                          ArithmeticRegionalAllocationPolicy policy = ArithmeticRegionalAllocationPolicy::SingleId)
        : allocationPolicy(policy), function(f), indexOwner(shared ? shared : std::make_shared<PhaseIndex>()),
          index(*indexOwner),
          indexReady(bool(shared)), arena(std::move(a)), expressions(*arena) {}
    Expr yes() { return expressions.boolean(true); }
    Expr no() { return expressions.boolean(false); }
    Expr c(uint64_t value) { return expressions.constant(value); }
    Expr both(Expr a, Expr b) { return expressions.land(a, b); }
    Expr either(Expr a, Expr b) { return expressions.lor(a, b); }
    Expr negate(Expr value) { return expressions.lnot(value); }
    bool fail(StringRef message) { error = message.str(); return false; }
    uint32_t pipe(uint32_t port) const {
        const auto& p = ports[port];
        return static_cast<uint32_t>(children[p.child].anchors[p.type].phase->kPipeValue);
    }
    uint32_t port(uint32_t child, uint32_t type, Expr ordinal, const std::vector<Expr>& visits = {}) {
        if (ports.size() >= UINT32_MAX / 2) { fail("sequence boundary event identity overflow"); return 0; }
        auto key = std::make_tuple(child, type, ordinal, visits);
        auto [position, added] = portIds.emplace(key, ports.size());
        if (added) { ports.push_back({child, type, ordinal, visits}); }
        return position->second;
    }
    uint32_t port(uint32_t child, const RegionalEvent& event) {
        return port(child, event.type, event.ordinal, event.visits);
    }
    Expr before(uint32_t a, uint32_t b) {
        ++costs.selectorComparisons;
        const auto& x = ports[a];
        const auto& y = ports[b];
        if (x.child != y.child) { return expressions.boolean(x.child < y.child); }
        if (!children[x.child].regional.presence && x.visits.empty() && y.visits.empty()) {
            return either(expressions.lt(x.ordinal, y.ordinal),
                both(expressions.eq(x.ordinal, y.ordinal), expressions.boolean(x.type < y.type)));
        }
        auto answer = regionalReferenceBefore(children[x.child].regional, x.event(), y.event());
        if (!answer) { fail("regional reference order unavailable"); return no(); }
        return *answer;
    }
    Expr same(uint32_t a, uint32_t b) {
        const auto& x = ports[a];
        const auto& y = ports[b];
        if (x.child != y.child || x.type != y.type || x.visits.size() != y.visits.size()) { return no(); }
        auto equal = expressions.eq(x.ordinal, y.ordinal);
        for (std::size_t i = 0; i < x.visits.size(); ++i) {
            equal = both(equal, expressions.eq(x.visits[i], y.visits[i]));
        }
        return equal;
    }
    Expr present(uint32_t id) {
        const auto& p = ports[id];
        auto answer = regionalPresence(children[p.child].regional, p.event());
        if (!answer) { fail("regional occurrence presence query unavailable"); return no(); }
        return *answer;
    }
    void crossing(Selected source, Selected target) {
        ++costs.crossingCandidates;
        auto guard = both(source.present, target.present);
        auto key = std::make_pair(source.port, target.port);
        auto [position, added] = crossingIds.emplace(key, crossings.size());
        if (added) { crossings.push_back({source.port, target.port, guard}); }
        else { crossings[position->second].guard = either(crossings[position->second].guard, guard); }
    }
    bool partialStorage() const {
        return llvm::any_of(children, [](const Child& child) { return bool(child.orderCovered); });
    }
    bool orderCoveredChild(const StructureNode& node);
    bool collect(std::size_t rootNode = 0);
    FailureOr<RegionalAnalysis> originalRegion(std::size_t node, std::string& diagnostic);
    bool resolvedChild(std::size_t node);
    bool finiteChild(ArrayRef<Operation*> roots, FiniteStrategy strategy);
    bool explicitChild(const StructureNode& node);
    bool conditionalChild(const StructureNode& node);
    bool loopChild(const StructureNode& node);
    bool repeatedChild(const StructureNode& node, Expr trips);
    bool singletonBoundaryChild(const StructureNode& node, Expr trips);
    bool phasedChild(const StructureNode& node, Expr trips);
    bool boundaryLoop(scf::ForOp loop);
    bool rotatingPatterns(Child& child, const RotatingAnalysis& analysis, const RecognitionResult& recognized);
    bool numericPatterns(Child& child, const NumericTemplate& numeric);
    bool partition();
    void summarize();
    void bindAdapters();
    bool importSummaries(bool requireEndpoints = true);
    void maskFrontier(std::vector<Selected>& values, Expr replaced, SequenceBridgeWork& work);
    void buildNativeLinks();
    void bridges();
    bool valueBridges();
    void canonicalizeCrossings();
    uint32_t selectPort(Expr choose, uint32_t yesPort, uint32_t noPort);
    void normalizeSelectorAlternatives(std::vector<Selected>& values);
    void foldCrossingEndpoints(bool incomingSources);
    void consolidateCrossings(bool incomingSources);
    SequenceEvent event(std::size_t id) const;
    std::optional<Expr> eventReachability(SequenceEvent source, SequenceEvent target);
    std::optional<Expr> eventReachability(std::size_t source, std::size_t target);
    bool closure();
    bool numericalCrossingReduction();
    bool symbolicCrossingReduction();
    bool preferNumericalCrossings();
    std::optional<std::vector<uint32_t>> numericalThresholds(
        SequenceEvent event, bool reverse, NumericalChainQueryCost& cost);
    std::optional<Expr> numericalReachability(
        SequenceEvent source, SequenceEvent target, NumericalChainQueryCost& cost);
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepare(
        std::string& diagnostic, ArrayRef<scf::ForOp> enclosing = {},
        const SequenceEndpointResolver& resolver = {});
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareWithTypeBases(
        std::string& diagnostic, ArrayRef<scf::ForOp> enclosing, ArrayRef<uint32_t> typeBases,
        const SequenceEndpointResolver& resolver = {});
private:
    friend std::optional<SequenceAnalysis> upgradeSequenceEndpoints(
        const SequenceAnalysis&, std::size_t, SequenceRegionResolver);
    // False retains per-crossing qualifications for one mixed ordered batch; no partial plan is adopted.
    FailureOr<bool> prepareFiniteCrossings(PreparedLogicalPlan& plan,
        std::vector<RegionalAllocationGroup>& allocation, ArrayRef<uint32_t> typeOffsets,
        uint32_t& nextRecord, std::vector<uint8_t>& qualified);
    std::shared_ptr<const SequenceCrossingPresence> crossingPresence;
    void sealCrossingPresence();
};
std::optional<int64_t> sequenceInteger(Value value);
} // namespace mlir::pto::frontiersynch
#endif
