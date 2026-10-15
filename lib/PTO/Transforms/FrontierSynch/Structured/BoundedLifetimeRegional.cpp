// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A finite mandatory graph proves transport; local rows answer near queries.
// Neither a demand span nor sampled branch executions prove global queries.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeRegional.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "../Shared/IterationPredicates.h"
#include "../Shared/CountedLoop.h"
#include "PhysicalRefreshLimits.h"
#include "BoundedLifetimeAllocationInternal.h"
#include "BoundedLifetimeAllocationContext.h"
#include "BoundedMandatoryTransport.h"
#include "llvm/ADT/BitVector.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct Use { unsigned payload; bool read, write; };
using TransportGraph = detail::BoundedMandatoryTransport::Graph;
void addLifetimes(TransportGraph& graph, std::vector<std::vector<Use>>& cells,
    const BoundedLifetimeDemandResult& demands, const GuardedRecognition& skeleton, unsigned m)
{
    auto edge = [&](unsigned source, unsigned target) { graph[source].push_back(target); };
    for (auto& uses : cells) {
        llvm::sort(uses, [](const Use& a, const Use& b) { return a.payload < b.payload; });
        std::vector<Use> merged;
        std::map<std::pair<unsigned, Operation*>, unsigned> envelopes;
        bool simultaneous = false;
        for (auto use : uses) {
            if (!merged.empty() && merged.back().payload == use.payload) {
                merged.back().read |= use.read; merged.back().write |= use.write;
                continue;
            }
            auto key = std::make_pair(use.payload / m,
                skeleton.phases[use.payload % m].phase->elementOp);
            simultaneous |= !envelopes.emplace(key, use.payload).second;
            merged.push_back(use);
        }
        // Dropping an atom is safe for this sufficient lower certificate.
        // Simultaneous envelopes require the full producer's raw-pair model.
        if (simultaneous) { continue; }
        auto connect = [&](unsigned source, unsigned target) {
            const auto p = static_cast<uint32_t>(skeleton.phases[source % m].phase->kPipeValue);
            const auto q = static_cast<uint32_t>(skeleton.phases[target % m].phase->kPipeValue);
            if (p != q && !demands.window.storageProtection.protectsScalar(p, q)) {
                edge(2 * source + 1, 2 * target);
            }
        };
        std::optional<unsigned> writer;
        std::map<uint32_t, unsigned> readers;
        for (auto use : merged) {
            const auto pipe = static_cast<uint32_t>(skeleton.phases[use.payload % m].phase->kPipeValue);
            if (use.write) {
                if (writer) { connect(*writer, use.payload); }
                for (auto [readerPipe, reader] : readers) { connect(reader, use.payload); }
                readers.clear(); writer = use.payload;
            } else if (use.read) {
                if (writer) { connect(*writer, use.payload); }
                readers[pipe] = use.payload;
            }
        }
    }
}
bool certifyTransport(const TransportGraph& graph, const std::map<uint32_t, uint32_t>& anchors,
    unsigned m, uint64_t span, uint64_t residue, std::string& error)
{
    for (auto [pipe, source] : anchors) {
        llvm::BitVector reached(graph.size());
        SmallVector<unsigned> work{2 * source + 1};
        while (!work.empty()) {
            auto next = work.pop_back_val();
            if (reached.test(next)) { continue; }
            reached.set(next);
            llvm::append_range(work, graph[next]);
        }
        for (auto [targetPipe, target] : anchors) {
            if (!reached.test(2 * (span * m + target))) {
                error = "mandatory global transport is unproved for pipe " + std::to_string(pipe) +
                    " to " + std::to_string(targetPipe) + " at physical phase " + std::to_string(residue);
                return false;
            }
        }
    }
    return true;
}
// Delete conditional occurrences, then add their sparse physical lifetimes.
// Such an edge is a path in the full sparse graph: intermediate writers split
// it into writer-chain paths, and omitted same-pipe protected hazards are
// carried by native start/completion order. No guessed synchronization edges.
bool mixing(const BoundedLifetimeDemandResult& demands, const BoundedLifetimeRecognition& recognized,
    uint64_t span, std::vector<TransportGraph>& retained, std::string& error)
{
    if (!recognized.physical) { error = "global queries need an absolute physical refresh map"; return false; }
    const auto& physical = *recognized.physical;
    const auto& skeleton = recognized.skeleton;
    const unsigned m = skeleton.phases.size();
    const unsigned count = m * (span + 1);
    std::map<uint32_t, uint32_t> anchors;
    DenseMap<const CompoundInstanceElement*, unsigned> positions;
    for (unsigned site = 0; site < m; ++site) {
        positions[skeleton.phases[site].phase] = site;
        if (demands.unconditional[site]) {
            anchors.try_emplace(static_cast<uint32_t>(skeleton.phases[site].phase->kPipeValue), site);
        }
    }
    for (const auto& site : skeleton.phases) {
        if (!anchors.count(static_cast<uint32_t>(site.phase->kPipeValue))) {
            error = "global queries lack an unconditional occurrence on a participating pipe"; return false;
        }
    }
    for (uint64_t residue = 0; residue < physical.period; ++residue) {
        std::vector<std::vector<unsigned>> graph(2 * count);
        auto edge = [&](unsigned source, unsigned target) { graph[source].push_back(target); };
        std::map<uint32_t, unsigned> previous;
        std::vector<std::vector<Use>> cells(physical.atoms.size());
        for (unsigned visit = 0; visit <= span; ++visit) {
            for (unsigned site = 0; site < m; ++site) {
                if (!demands.unconditional[site]) { continue; }
                const auto payload = visit * m + site;
                const auto pipe = static_cast<uint32_t>(skeleton.phases[site].phase->kPipeValue);
                edge(2 * payload, 2 * payload + 1);
                if (auto found = previous.find(pipe); found != previous.end()) {
                    edge(2 * found->second, 2 * payload);
                    edge(2 * found->second + 1, 2 * payload + 1);
                }
                previous[pipe] = payload;
            }
            for (const auto& projection : physical.accesses) {
                if (projection.phase != (residue + visit) % physical.period) { continue; }
                const auto& access = skeleton.result.accesses[projection.access];
                const auto site = positions.lookup(demands.input->accesses().effects()[access.effect].phase);
                if (demands.unconditional[site]) {
                    cells[projection.atom].push_back({visit * m + site, access.reads, access.writes});
                }
            }
            for (const auto& prerequisite : demands.window.prerequisites) {
                // The source window contains repeated copies. Use only its
                // first-visit original prerequisites, then translate identities.
                if (prerequisite.source >= m || prerequisite.target >= m ||
                    !demands.unconditional[prerequisite.source] || !demands.unconditional[prerequisite.target] ||
                    demands.expressions.constantValue(prerequisite.guard) != 1) {
                    continue;
                }
                edge(2 * (visit * m + prerequisite.source) + 1, 2 * (visit * m + prerequisite.target));
            }
            for (const auto& prerequisite : demands.window.nativePrerequisites) {
                if (prerequisite.source >= m || prerequisite.target >= m ||
                    !demands.unconditional[prerequisite.source] || !demands.unconditional[prerequisite.target] ||
                    demands.expressions.constantValue(prerequisite.guard) != 1) {
                    continue;
                }
                edge(2 * (visit * m + prerequisite.source) + 1, 2 * (visit * m + prerequisite.target));
            }
        }
        addLifetimes(graph, cells, demands, skeleton, m);
        if (!certifyTransport(graph, anchors, m, span, residue, error)) { return false; }
        retained.push_back(std::move(graph));
    }
    return true;
}
struct Queries {
    std::shared_ptr<BoundedLifetimeDemandResult> window;
    std::shared_ptr<const SyncInput> inputOwner;
    std::unique_ptr<detail::BoundedMandatoryTransport> mandatory;
    Id trips;
    std::vector<std::vector<std::pair<Value, bool>>> guards;
    using Key = std::tuple<uint32_t, Id, PeriodicEventKind, uint32_t, Id, PeriodicEventKind>;
    std::map<Key, Id> answers;
    std::map<Id, RegionExpressions::Substitution> substitutions;
    Id bind(Id root, Id base)
    {
        auto& e = window->expressions;
        if (auto found = substitutions.find(base); found != substitutions.end()) {
            return e.substitute(root, found->second);
        }
        std::vector<std::pair<Id, Id>> bindings{{window->base, base}};
        for (auto [id, recipe] : window->predicateBindings) {
            bindings.push_back({id, window->predicates->at(recipe.first, e.add(base, e.constant(recipe.second)))});
        }
        auto inserted = substitutions.try_emplace(base, bindings);
        return e.substitute(root, inserted.first->second);
    }
    std::optional<Id> presence(RegionalEvent event, IterationPredicates& predicates)
    {
        auto& e = window->expressions;
        if ((event.kind != PeriodicEventKind::Start && event.kind != PeriodicEventKind::Completion) ||
            !event.visits.empty() || event.type >= window->window.sites ||
            event.ordinal >= e.size() || e.isBoolean(event.ordinal)) { return std::nullopt; }
        auto present = e.lt(event.ordinal, trips);
        for (auto [condition, takeThen] : guards[event.type]) {
            auto value = predicates.at(condition, event.ordinal);
            present = e.land(present, takeThen ? value : e.lnot(value));
        }
        return present;
    }
    std::optional<Id> presence(RegionalEvent event)
    {
        return presence(event, *window->predicates);
    }
    std::optional<Id> query(RegionalEvent source, RegionalEvent target)
    {
        auto a = presence(source), b = presence(target);
        if (!a || !b) { return std::nullopt; }
        const Key key{source.type, source.ordinal, source.kind, target.type, target.ordinal, target.kind};
        if (auto found = answers.find(key); found != answers.end()) { return found->second; }
        auto& e = window->expressions;
        const auto m = window->window.sites;
        const bool nativeKinds = source.kind == PeriodicEventKind::Start ||
            target.kind == PeriodicEventKind::Completion;
        if (nativeKinds && window->window.payloads[source.type].pipe == window->window.payloads[target.type].pipe) {
            // On one pipe, native start/start and completion/completion chains,
            // plus I->C at the target, decide these kinds exactly. All graph
            // edges are reference-forward, so reverse occurrences cannot reach.
            // C->I remains a genuine demand/query obligation, even on one pipe.
            auto ordered = e.lt(source.ordinal, target.ordinal);
            if (source.type <= target.type) { ordered = e.lor(ordered, e.eq(source.ordinal, target.ordinal)); }
            const auto answer = e.land(e.land(*a, *b), ordered);
            answers.emplace(key, answer);
            return answer;
        }
        auto distance = e.sub(target.ordinal, source.ordinal);
        auto answer = e.boolean(false);
        const auto fixedDistance = e.constantValue(distance);
        const uint64_t first = fixedDistance.value_or(0);
        const uint64_t last = fixedDistance.value_or(window->window.span);
        const uint64_t from = uint64_t{2} * source.type + unsigned(source.kind == PeriodicEventKind::Completion);
        for (uint64_t d = first; d <= last && d <= window->window.span; ++d) {
            const auto to = 2 * (d * m + target.type) + unsigned(target.kind == PeriodicEventKind::Completion);
            // A path in every physical residue is valid for any source ordinal.
            // Keep endpoint presence and forward distance below. The final
            // extended visit, and every uncertified pair, use the exact window.
            if (mandatory && from <= std::numeric_limits<unsigned>::max() &&
                to <= std::numeric_limits<unsigned>::max() &&
                mandatory->alwaysReaches(static_cast<unsigned>(from), static_cast<unsigned>(to))) {
                answer = e.lor(answer, e.eq(distance, e.constant(d)));
                continue;
            }
            auto local = window->analysis.window.query(e, {source.type, source.kind},
                {static_cast<uint32_t>(d * m + target.type), target.kind});
            if (!local) { return std::nullopt; }
            answer = e.lor(answer, e.land(e.eq(distance, e.constant(d)), bind(*local, source.ordinal)));
        }
        // The transport certificate ends one visit before the target. Native
        // start order then reaches any present target site, even if it precedes
        // its own pipe's mandatory anchor within the target visit.
        answer = e.lor(answer, e.lt(e.constant(window->window.span), distance));
        answer = e.land(e.land(*a, *b), e.land(e.le(source.ordinal, target.ordinal), answer));
        answers.emplace(key, answer);
        return answer;
    }
};
bool validateContext(const std::shared_ptr<BoundedLifetimeDemandResult>& demands,
    const BoundedLifetimeRecognition& recognized, const std::shared_ptr<RegionExpressions>& arena,
    const std::shared_ptr<const SyncInput>& inputOwner, std::string& error)
{
    if ((arena && !arena->constructionError().empty()) ||
        (inputOwner && demands && inputOwner.get() != demands->input) || !demands || !arena || !recognized.physical ||
        !demands->input || !demands->function || !demands->loop || recognized.refresh.span > (UINT32_MAX - 2) / 2 ||
        recognized.skeleton.phases.size() != demands->unconditional.size() ||
        recognized.skeleton.phases.size() != demands->anchors.size() || !recognized.physical->period ||
        recognized.physical->period > recognized.refresh.span) {
        error = "bounded global-query context or physical certificate unavailable"; return false;
    }
    for (auto [site, anchor] : llvm::zip(recognized.skeleton.phases, demands->anchors)) {
        if (!site.phase || site.phase != anchor.phase ||
            (site.guard && *site.guard >= recognized.skeleton.guards.size())) {
            error = "bounded query skeleton belongs to another context"; return false;
        }
    }
    for (auto [id, guard] : llvm::enumerate(recognized.skeleton.guards)) {
        if (!guard.condition || !guard.condition.getType().isInteger(1) || (guard.parent && *guard.parent >= id)) {
            error = "bounded query guard ancestry is invalid"; return false;
        }
    }
    for (const auto& projection : recognized.physical->accesses) {
        if (projection.atom >= recognized.physical->atoms.size() ||
            projection.access >= recognized.skeleton.result.accesses.size() ||
            projection.phase >= recognized.physical->period ||
            recognized.skeleton.result.accesses[projection.access].effect >=
                demands->input->accesses().effects().size()) {
            error = "bounded query physical projection has invalid identities"; return false;
        }
        const auto* phase = demands->input->accesses().effects()[
            recognized.skeleton.result.accesses[projection.access].effect].phase;
        if (llvm::none_of(demands->anchors, [&](const auto& anchor) { return anchor.phase == phase; })) {
            error = "bounded query projection has no original occurrence"; return false;
        }
    }
    return CountedLoop::get(demands->loop).has_value();
}
} // namespace
FailureOr<RegionalAnalysis> boundedLifetimeRegionalQueries(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, const BoundedLifetimeRecognition& recognized,
    const PhaseIndex& index, std::shared_ptr<RegionExpressions> arena, std::string& error,
    std::shared_ptr<const SyncInput> inputOwner)
{
    if (!validateContext(demands, recognized, arena, inputOwner, error)) { return failure(); }
    const uint64_t mixingSpan = 2 * recognized.refresh.span + 1;
    if (!detail::physicalRefreshWindowFits(mixingSpan + 1, recognized.skeleton.phases.size(),
            recognized.skeleton.guards.size(), recognized.physical->accesses.size())) {
        error = "global-query window exceeds the supported numerical representation"; return failure();
    }
    std::vector<TransportGraph> mandatory;
    if (!mixing(*demands, recognized, mixingSpan, mandatory, error)) { return failure(); }
    RegionExpressions::Transaction transaction(*arena);
    auto extended = analyzeBoundedLifetimeRegion(demands->function, demands->loop, index,
        *demands->input, recognized, error, arena, mixingSpan + 1);
    if (failed(extended)) { return failure(); }
    auto queries = std::make_shared<Queries>(); queries->window = *extended;
    queries->mandatory = std::make_unique<detail::BoundedMandatoryTransport>(std::move(mandatory));
    queries->inputOwner = std::move(inputOwner);
    queries->trips = CountedLoop::get(demands->loop)->trips(*arena);
    for (const auto& site : recognized.skeleton.phases) {
        std::vector<std::pair<Value, bool>> guards;
        for (auto current = site.guard; current; current = recognized.skeleton.guards[*current].parent) {
            const auto& guard = recognized.skeleton.guards[*current];
            guards.push_back({guard.condition, guard.takeThen});
        }
        queries->guards.push_back(std::move(guards));
    }
    RegionalAnalysis out;
    out.expressions = arena; out.anchors = demands->anchors;
    out.occurrenceLoops.assign(out.anchors.size(), demands->loop);
    out.accessModel = &demands->input->accesses();
    out.gmAliasPolicy = demands->input->memory().gmPolicy();
    out.capabilities.exactQueries = true; out.capabilities.contextualGuards = true;
    // Every guard was normalized at the symbolic window base plus each visit.
    // Without opaque evaluations, coordinate substitution uses the same total
    // scalar expressions and cannot introduce new historical input identities.
    // Fixed distances request a subset of the ordinary window's local queries.
    out.capabilities.substitutionStableQueries = !queries->window->predicates->hasOpaqueEvaluations();
    out.presence = [queries](RegionalEvent event) { return queries->presence(event); };
    out.reachability = [queries](RegionalEvent source, RegionalEvent target) { return queries->query(source, target); };
    SmallVector<scf::ForOp> enclosing;
    for (auto* parent = demands->loop->getParentOp(); parent != demands->function; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) { enclosing.push_back(loop); }
    }
    std::reverse(enclosing.begin(), enclosing.end());
    // This region has flat local event coordinates. Repetition adds its own
    // visit coordinate when importing the plan, as for the other flat children.
    auto owner = std::make_shared<MathematicalResult>();
    owner->input = queries->inputOwner;
    owner->boundedDemands = demands;
    // Cache the requested allocation capability, including an unavailable
    // ordinary export, independently of detached endpoint instantiation.
    auto allocation = std::make_shared<std::optional<std::shared_ptr<RegionalAllocationSummary>>>();
    auto invocation = std::make_shared<std::shared_ptr<detail::BoundedLifetimeAllocationContext>>();
    out.prepareWithVisits = [demands, owner, enclosing, allocation, arena,
        trips = queries->trips, queries, invocation](ArrayRef<scf::ForOp> visits)
        -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        if (visits != ArrayRef<scf::ForOp>(enclosing)) { return failure(); }
        std::string diagnostic;
        auto plan = prepareBoundedLifetimeLogicalResult(demands, diagnostic, false);
        if (succeeded(plan)) {
            (*plan)->mathematicalOwner = owner;
            (*plan)->allocationPreparation =
                [demands, trips, allocation, arena, queries, invocation](PreparedLogicalPlan& prepared) {
                if (!allocation->has_value()) {
                    *invocation = detail::BoundedLifetimeAllocationContext::create(demands, queries->inputOwner);
                    SmallVector<const CompoundInstanceElement*> phases;
                    for (const auto& anchor : demands->anchors) { phases.push_back(anchor.phase); }
                    // Allocation rollback must not leave IDs in retained query caches.
                    auto predicates = std::make_shared<IterationPredicates>(demands->loop, *arena, phases);
                    allocation->emplace(detail::boundedLifetimeRegionalAllocationWithPresence(
                        demands->expressions, *arena, demands->window, demands->unconditional,
                        demands->analysis.sourceDemands, trips,
                        {arena, [queries, predicates](RegionalEvent event) {
                            return queries->presence(event, *predicates);
                        }, *invocation}));
                }
                prepared.regionalAllocation = **allocation;
            };
        }
        return plan;
    };
    if (enclosing.empty()) {
        auto prepare = out.prepareWithVisits;
        out.prepare = [prepare]() { return prepare({}); };
    }
    out.capabilities.endpointRecipes = true;
    out.cost.children = 1; out.cost.numericVisits = mixingSpan + 2;
    out.cost.physicalFragments = recognized.physical->accesses.size();
    out.cost.retainedExpressionNodes = demands->expressions.size();
    out.cost.expressionNodes = arena->size();
    if (!arena->constructionError().empty()) { error = arena->constructionError(); return failure(); }
    transaction.commit();
    return out;
}
} // namespace mlir::pto::frontiersynch
