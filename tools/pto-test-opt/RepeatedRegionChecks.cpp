// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check hierarchical identities through two sequence exports, without insertion.
#include "PTO/Transforms/FrontierSynch/Shared/RepeatedRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RepeatedRegionInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RepeatedAllocation.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalConstantPalette.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/OccurrenceBinding.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RepeatedLaneAllocation.h"
#include <map>
#include <set>
#include <tuple>
#include <algorithm>
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
#include "RepeatedOrderedAllocationChecks.h"
namespace {
using Matrix = std::vector<std::vector<bool>>;
void closure(Matrix& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
bool check(func::FuncOp function, Operation* anchor, scf::ForOp outer, scf::ForOp inner,
           unsigned trips, unsigned steps, unsigned mask, uint64_t& checked, bool symbolicEnable = false)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto zero = e.constant(0);
    const auto enabled = symbolicEnable ? e.input(function.getArgument(1)) : e.boolean(true);
    auto present = [&](bool active) { return e.land(enabled, e.boolean(active)); };
    std::vector<pto::CompoundInstanceElement> phases;
    for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M, pto::PipelineType::PIPE_V}) {
        phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
                            SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
        phases.back().elementOp = anchor;
    }
    fs::RegionalAnalysis body;
    body.expressions = arena; body.capabilities = {true, true, true, false};
    for (std::size_t t = 0; t < phases.size(); ++t) {
        body.anchors.push_back({&phases[t], {}, {anchor->getBlock(), anchor},
                                {anchor->getBlock(), anchor->getNextNode()}});
        body.occurrenceLoops.push_back({});
        fs::RegionalSelector selected{{uint32_t(t), zero, fs::PeriodicEventKind::Start}, present(mask & (1U << t))};
        body.firstPayloads[t].push_back(selected); body.lastPayloads[t].push_back(selected);
    }
    Matrix local(6, std::vector<bool>(6));
    for (unsigned t = 0; t < 3; ++t) {
        if (!(mask & (1U << t))) { continue; }
        local[2*t][2*t] = local[2*t+1][2*t+1] = local[2*t][2*t+1] = true;
    }
    local[1][2] = (mask & 3) == 3; closure(local);
    body.presence = [arena, zero, mask, enabled](fs::RegionalEvent a) -> std::optional<fs::RegionExpressions::Id> {
        if (a.type >= 3 || !a.visits.empty()) { return std::nullopt; }
        return arena->land(enabled,
            arena->land(arena->boolean(mask & (1U << a.type)), arena->eq(a.ordinal, zero)));
    };
    body.reachability = [arena, local, present = body.presence](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        auto pa = present(a), pb = present(b);
        if (!pa || !pb) { return std::nullopt; }
        auto i = 2*a.type + (a.kind == fs::PeriodicEventKind::Completion);
        auto j = 2*b.type + (b.kind == fs::PeriodicEventKind::Completion);
        return arena->land(arena->land(*pa, *pb), arena->boolean(local[i][j]));
    };
    fs::RegionalStorageBoundary cell;
    cell.cell = {pto::AddressSpace::LEFT, 0, 8};
    cell.firstWriters.push_back({{0, zero, fs::PeriodicEventKind::Start}, present(mask & 1)});
    cell.lastWriters = cell.firstWriters;
    cell.lastReaders[1].push_back({{1, zero, fs::PeriodicEventKind::Start}, present(mask & 2)});
    cell.firstReaders[1].push_back({{1, zero, fs::PeriodicEventKind::Start}, present((mask & 3) == 2)});
    body.storageBoundary.push_back(cell);
    auto once = fs::repeatInvariantRegion(function, inner, body, e.constant(steps));
    if (!once.error.empty()) { llvm::errs() << once.error << "\n"; return false; }
    auto twice = fs::repeatInvariantRegion(function, outer, once.regional, e.constant(trips));
    if (!twice.error.empty()) { llvm::errs() << twice.error << "\n"; return false; }
    if (!symbolicEnable && steps && trips && mask &&
        (!once.regional.numerical || !twice.regional.numerical)) {
        llvm::errs() << "numerical invariant repetition did not export its shared index\n"; return false;
    }
    const unsigned count = 3 * trips * steps;
    Matrix graph(2 * count, std::vector<bool>(2 * count));
    for (unsigned i = 0; i < count; ++i) {
        if (!(mask & (1U << (i % 3)))) { continue; }
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = true;
        for (unsigned j = i+1; j < count; ++j) {
            if (!(mask & (1U << (j % 3)))) { continue; }
            if (i%3 == j%3) { graph[2*i][2*j] = graph[2*i+1][2*j+1] = true; }
            if (i%3 < 2 && j%3 < 2 && (i%3 == 0 || j%3 == 0)) { graph[2*i+1][2*j] = true; }
        }
    }
    closure(graph);
    auto event = [&](unsigned vertex) {
        const unsigned occurrence = vertex / 2, visit = occurrence / 3;
        return fs::RegionalEvent{occurrence % 3, zero, vertex % 2 ? fs::PeriodicEventKind::Completion :
            fs::PeriodicEventKind::Start, {e.constant(visit / steps), e.constant(visit % steps)}};
    };
    // Construct the symbolic repeated graph once, then compare both enable
    // valuations with the independently unfolded concrete graph. Factoring
    // the common enable must not make an inactive across-visit path reachable.
    fs::RegionExpressions::Substitution active({{enabled, e.boolean(true)}});
    fs::RegionExpressions::Substitution inactive({{enabled, e.boolean(false)}});
    for (unsigned i = 0; i < 2*count; ++i) {
        for (unsigned j = 0; j < 2*count; ++j) {
            auto got = fs::regionalReachability(twice.regional, event(i), event(j));
            if (!got) { return false; }
            if (twice.regional.numerical) {
                fs::NumericalChainQueryCost work;
                auto indexed = twice.regional.numerical->query(event(i), event(j), work);
                if (!indexed || *indexed != graph[i][j]) {
                    llvm::errs() << "squared regional query mismatch\n"; return false;
                }
            }
            auto concrete = symbolicEnable ? e.substitute(*got, active) : *got;
            if (e.constantValue(concrete) != uint64_t(graph[i][j]) ||
                (symbolicEnable && e.constantValue(e.substitute(*got, inactive)) != 0)) {
                llvm::errs() << "repeat mismatch " << trips << "," << steps << ":" << i << "," << j << "\n";
                return false;
            }
            if (trips == 3 && i < 6 && j / 6 >= i / 6) {
                // Ask the inner uniform relation beyond its finite trip bound.
                // The independent graph includes translated outer invocations;
                // their boundary has the same invariant writer/reader pattern.
                fs::RegionalEvent source{i / 2, zero, i % 2 ? fs::PeriodicEventKind::Completion :
                    fs::PeriodicEventKind::Start};
                fs::RegionalEvent target{(j / 2) % 3, zero, j % 2 ? fs::PeriodicEventKind::Completion :
                    fs::PeriodicEventKind::Start};
                auto uniform = once.state->relativeQuery(source, target, j / 6);
                if (!uniform) { return false; }
                auto value = symbolicEnable ? e.substitute(*uniform, active) : *uniform;
                if (e.constantValue(value) != uint64_t(graph[i][j]) ||
                    (symbolicEnable && e.constantValue(e.substitute(*uniform, inactive)) != 0)) {
                    llvm::errs() << "uniform repeat query mismatch\n"; return false;
                }
            }
            ++checked;
        }
    }
    if (!symbolicEnable && trips == 2 && steps == 2 && mask == 7) {
        auto child = twice.regional;
        child.cost.arithmeticRegions = UINT64_MAX;
        child.cost.boundaryBytes = UINT64_MAX;
        child.cost.retainedExpressionNodes = UINT64_MAX;
        auto joined = fs::composeRegionalSequence(function, arena, {child, child}, false, false);
        // This oracle targets the numerical adapter independently of dispatch
        // cost preferences; preserve its ordinary eligibility checks.
        if (joined.error.empty() && joined.state && !joined.state->numerical &&
            !joined.state->numericalCrossingReduction()) { return false; }
        if (joined.cost.arithmeticRegions != UINT64_MAX || joined.cost.boundaryBytes != UINT64_MAX ||
            joined.cost.retainedExpressionNodes != UINT64_MAX) { return false; }
        auto region = fs::sequenceRegionalResult(joined);
        if (!joined.error.empty() || !region.numerical) {
            llvm::errs() << "squared children were not reused by numerical composition: " << joined.error << "\n";
            return false;
        }
        Matrix doubled(4 * count, std::vector<bool>(4 * count));
        for (unsigned a = 0; a < 2 * count; ++a) {
            doubled[2*a][2*a] = doubled[2*a+1][2*a+1] = doubled[2*a][2*a+1] = true;
            for (unsigned b = a+1; b < 2 * count; ++b) {
                if (a%3 == b%3) { doubled[2*a][2*b] = doubled[2*a+1][2*b+1] = true; }
                if (a%3 < 2 && b%3 < 2 && (a%3 == 0 || b%3 == 0)) { doubled[2*a+1][2*b] = true; }
            }
        }
        closure(doubled);
        auto translated = [&](unsigned vertex) {
            auto result = event(vertex % (2 * count));
            result.type += vertex >= 2 * count ? 3 : 0;
            return result;
        };
        for (unsigned a = 0; a < doubled.size(); ++a) {
            for (unsigned b = 0; b < doubled.size(); ++b) {
                fs::NumericalChainQueryCost work;
                auto answer = region.numerical->query(translated(a), translated(b), work);
                if (!answer || *answer != doubled[a][b]) {
                    llvm::errs() << "numerical composition of squared children differs from unfolded graph\n";
                    return false;
                }
                ++checked;
            }
        }
    }
    auto absent = fs::regionalPresence(twice.regional,
        {0, zero, fs::PeriodicEventKind::Start, {e.constant(trips), zero}});
    return absent && e.constantValue(*absent) == 0 && e.error().empty();
}

// Build actual finite occurrences independently of the producer's phase-bound
// formulas. A/B use distinct pipes; A writes and B reads one shared cell in
// the storage case. Without storage, only each local A->B readiness remains.
bool allocationCheck(func::FuncOp function, Operation* anchor, scf::ForOp loop,
                     unsigned q, unsigned begin, unsigned end, bool storage,
                     bool omitMember, uint64_t& checked, bool cyclePolicy = false, unsigned guardMode = 0,
                     fs::RepeatedLaneProofCost* totalCost = nullptr)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto zero = e.constant(0), yes = e.boolean(true);
    const auto enabled = guardMode ? e.input(function.getArgument(1)) : yes;
    fs::RegionExpressions::Substitution valuation({{enabled, e.boolean(guardMode != 1)}});
    auto evaluate = [&](fs::RegionExpressions::Id expression) {
        return e.constantValue(guardMode ? e.substitute(expression, valuation) : expression);
    };
    const unsigned types = 2*q, periods = (end + q - 1)/q;
    std::vector<pto::CompoundInstanceElement> phases;
    phases.reserve(types);
    fs::RegionalAnalysis body;
    body.expressions = arena; body.capabilities = {true, true, true, false};
    Matrix local(2*types, std::vector<bool>(2*types));
    for (unsigned t = 0; t < types; ++t) {
        const auto pipe = t%2 ? pto::PipelineType::PIPE_M : pto::PipelineType::PIPE_MTE1;
        phases.emplace_back(t, SmallVector<const pto::BaseMemInfo*>{},
                            SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
        phases.back().elementOp = anchor;
        body.anchors.push_back({&phases.back(), {}, {anchor->getBlock(), anchor},
                                {anchor->getBlock(), anchor->getNextNode()}});
        body.occurrenceLoops.push_back({});
        local[2*t][2*t] = local[2*t+1][2*t+1] = local[2*t][2*t+1] = true;
        for (unsigned u = t+1; u < types; ++u) {
            if (t%2 == u%2) { local[2*t][2*u] = local[2*t+1][2*u+1] = true; }
            if ((storage && (!(t%2) || !(u%2))) || (!(t%2) && u == t+1)) {
                local[2*t+1][2*u] = true;
            }
        }
    }
    closure(local);
    for (unsigned parity = 0; parity < 2; ++parity) {
        const auto pipe = static_cast<uint32_t>(phases[parity].kPipeValue);
        body.firstPayloads[pipe].push_back({{parity, zero, fs::PeriodicEventKind::Start}, yes});
        body.lastPayloads[pipe].push_back({{types-2+parity, zero, fs::PeriodicEventKind::Start}, yes});
    }
    body.presence = [arena, types, zero](fs::RegionalEvent a) -> std::optional<fs::RegionExpressions::Id> {
        if (a.type >= types || !a.visits.empty()) { return std::nullopt; }
        return arena->eq(a.ordinal, zero);
    };
    body.reachability = [arena, local, present = body.presence](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        auto pa = present(a), pb = present(b);
        if (!pa || !pb) { return std::nullopt; }
        const auto i = 2*a.type + (a.kind == fs::PeriodicEventKind::Completion);
        const auto j = 2*b.type + (b.kind == fs::PeriodicEventKind::Completion);
        return arena->land(arena->land(*pa, *pb), arena->boolean(local[i][j]));
    };
    if (storage) {
        fs::RegionalStorageBoundary cell;
        cell.cell = {pto::AddressSpace::LEFT, 0, 8};
        cell.firstWriters.push_back({{0, zero, fs::PeriodicEventKind::Start}, yes});
        cell.lastWriters.push_back({{types-2, zero, fs::PeriodicEventKind::Start}, yes});
        cell.lastReaders[static_cast<uint32_t>(phases[1].kPipeValue)].push_back(
            {{types-1, zero, fs::PeriodicEventKind::Start}, yes});
        body.storageBoundary.push_back(std::move(cell));
    }
    auto repeated = fs::repeatInvariantRegion(function, loop, body, e.constant(periods));
    if (!repeated.error.empty()) { return false; }
    auto& state = *repeated.state;
    state.phaseCount = q; state.originalBegin = e.constant(begin); state.originalTrips = e.constant(end);
    for (unsigned t = 0; t < types; ++t) { state.typePhases.push_back(t/2); }
    const auto p = static_cast<uint32_t>(phases[0].kPipeValue);
    const auto r = static_cast<uint32_t>(phases[1].kPipeValue);
    fs::RegionalAllocationSummary child;
    fs::RegionalAllocationGroup group{p, r, storage ? 2U : q, {}};
    struct Spec { unsigned record, source, target, delay; bool active; };
    std::vector<Spec> specs;
    for (unsigned phase = 0; phase < q; ++phase) {
        // The last handoff may cross phases, but remains exactly one pair.
        const auto source = storage && phase && phase+1 == q ? 2*(phase-1) : 2*phase;
        const auto target = 2*phase+1;
        const bool active = !(omitMember && phase == 1) && !(guardMode == 1 && phase == 0);
        fs::RegionalAllocationMember member{phase, 0, storage ? phase%2 : phase,
            {source, zero, fs::PeriodicEventKind::Start},
            {target, zero, fs::PeriodicEventKind::Completion},
            guardMode && phase == 0 ? enabled : e.boolean(active)};
        member.singletonHandoff = true;
        group.members.push_back(member);
        specs.push_back({phase, source, target, 0, active});
    }
    child.groups.push_back(std::move(group));
    if (cyclePolicy) {
        // This redundant, valid A_last(i)->B_first(i+1) handoff has the same
        // direction as child handoffs. It forces their joint cycle policy to
        // handle consumer displacement, including zero-cost within-visit links.
        // Its required order already follows from the independently built
        // storage graph, so adding its recipe must not strengthen query Q.
        state.crossings.push_back({{types-2, zero, fs::PeriodicEventKind::Completion},
            {1, zero, fs::PeriodicEventKind::Start}, yes, false});
    }
    std::vector<std::pair<uint32_t, std::size_t>> crossings;
    for (std::size_t i = 0; i < state.crossings.size(); ++i) {
        const auto& crossing = state.crossings[i];
        if (crossing.native || e.constantValue(crossing.guard) == 0 ||
            crossing.source.type%2 == crossing.target.type%2) { continue; }
        const unsigned record = q + crossings.size();
        crossings.emplace_back(record, i);
        specs.push_back({record, crossing.source.type, crossing.target.type, 1, true});
    }
    if (storage && q > 1 && (begin%q || end%q)) {
        // Without a singleton certificate, a cross-phase member may represent
        // a truncated chain: its nominal endpoints cannot certify that chain.
        auto unqualified = child;
        unqualified.groups.front().members.back().singletonHandoff = false;
        auto input = std::make_shared<fs::RegionalAllocationSummary>(unqualified);
        if (fs::repeatedRegionalAllocation(state, input, crossings)) { return false; }
    }
    auto coalesced = fs::coalesceRegionalAllocation(body, child);
    if (!coalesced) { return false; }
    fs::RepeatedLaneProofCost proofCost;
    auto input = std::make_shared<fs::RegionalAllocationSummary>(child);
    auto summary = cyclePolicy ? fs::periodicLaneAllocation(state, *coalesced, crossings, &proofCost) :
                                 fs::repeatedRegionalAllocation(state, input, crossings);
    if (!summary) {
        llvm::errs() << "allocation producer rejected " << q << ":" << begin << ":" << end
                     << " storage=" << storage << " omit=" << omitMember << " cycle_policy=" << cyclePolicy << "\n";
        return false;
    }
    if (totalCost) {
        totalCost->implicationRequests += proofCost.implicationRequests;
        totalCost->implicationComputations += proofCost.implicationComputations;
        totalCost->constantRequests += proofCost.constantRequests;
        totalCost->constantComputations += proofCost.constantComputations;
    }
    // The concrete graph uses only actual occurrences in the retained interval.
    const unsigned occurrences = 2*(end-begin);
    Matrix graph(2*occurrences, std::vector<bool>(2*occurrences));
    for (unsigned i = 0; i < occurrences; ++i) {
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = true;
        for (unsigned j = i+1; j < occurrences; ++j) {
            if (i%2 == j%2) { graph[2*i][2*j] = graph[2*i+1][2*j+1] = true; }
            if ((storage && (!(i%2) || !(j%2))) || (!(i%2) && j == i+1)) {
                graph[2*i+1][2*j] = true;
            }
        }
    }
    closure(graph);
    struct Handoff { unsigned source, target; uint64_t id; };
    std::map<uint32_t, unsigned> recordsSeen;
    for (const auto& palette : summary->groups) {
        if (!palette.budget || palette.budget > 6) { return false; }
        std::vector<Handoff> handoffs;
        std::vector<std::set<unsigned>> firstByLane(palette.budget), lastByLane(palette.budget);
        for (const auto& member : palette.members) {
            auto spec = std::find_if(specs.begin(), specs.end(), [&](const Spec& s) {
                return s.record == member.record;
            });
            if (spec == specs.end() || !member.tupleRule || member.singletonHandoff ||
                ++recordsSeen[member.record] != 1 ||
                member.tupleRule->coordinateCount != (spec->delay ? 1U : 2U) ||
                !fs::validPhysicalTupleRule(*member.tupleRule, palette.budget)) { return false; }
            std::vector<Handoff> actual;
            for (unsigned visit = 0; visit < periods; ++visit) {
                const auto sourcePhase = visit*q + spec->source/2;
                const auto targetPhase = (visit+spec->delay)*q + spec->target/2;
                if (!spec->active || sourcePhase < begin || sourcePhase >= end ||
                    targetPhase < begin || targetPhase >= end) { continue; }
                const auto& rule = *member.tupleRule;
                uint64_t id = rule.base;
                for (const auto& term : rule.terms) {
                    // Children have identity0 and outer source visit coordinate1;
                    // new crossings use source visit as identity0 instead.
                    uint64_t coordinate = term.coordinate == 0 ? (spec->delay ? visit : 0) : visit;
                    id += term.scale * ((term.stride*coordinate + term.phase)%term.modulus);
                }
                if (id >= palette.budget) { return false; }
                actual.push_back({2*(sourcePhase-begin)+spec->source%2,
                                  2*(targetPhase-begin)+spec->target%2, id});
            }
            if (evaluate(member.active) != uint64_t(!actual.empty())) { return false; }
            if (!actual.empty()) {
                auto first = evaluate(member.firstSource.visits.front());
                auto last = evaluate(member.lastTarget.visits.front());
                if (!first || !last || *first*q + spec->source/2 != actual.front().source/2+begin ||
                    *last*q + spec->target/2 != actual.back().target/2+begin) { return false; }
            }
            std::map<uint64_t, std::pair<unsigned, unsigned>> memberBounds;
            for (const auto& handoff : actual) {
                auto [entry, added] = memberBounds.try_emplace(handoff.id, handoff.source, handoff.target);
                if (!added) { entry->second.second = handoff.target; }
            }
            for (const auto& [lane, bounds] : memberBounds) {
                firstByLane[lane].insert(bounds.first);
                lastByLane[lane].insert(bounds.second);
            }
            handoffs.insert(handoffs.end(), actual.begin(), actual.end());
            ++checked;
        }
        if (cyclePolicy && palette.lanes.empty()) { return false; }
        if (!palette.lanes.empty()) {
            if (palette.lanes.size() != palette.budget) { return false; }
            auto checkSelectors = [&](const auto& selectors, const std::set<unsigned>& expected) {
                std::set<unsigned> actual;
                for (const auto& selector : selectors) {
                    auto present = evaluate(selector.present);
                    if (!present) { return false; }
                    if (!*present) { continue; }
                    if (selector.event.visits.empty()) { return false; }
                    auto visit = evaluate(selector.event.visits.front());
                    if (!visit) { return false; }
                    const auto phase = *visit*q + selector.event.type/2;
                    if (phase < begin || phase >= end) { return false; }
                    actual.insert(2*(phase-begin) + selector.event.type%2);
                }
                return actual == expected;
            };
            for (std::size_t lane = 0; lane < palette.lanes.size(); ++lane) {
                if (!checkSelectors(palette.lanes[lane].firstSources, firstByLane[lane]) ||
                    !checkSelectors(palette.lanes[lane].lastTargets, lastByLane[lane])) { return false; }
            }
        }
        std::sort(handoffs.begin(), handoffs.end(), [](const Handoff& a, const Handoff& b) {
            return std::tie(a.source, a.target) < std::tie(b.source, b.target);
        });
        std::map<uint64_t, unsigned> lastConsumer;
        for (const auto& handoff : handoffs) {
            auto previous = lastConsumer.find(handoff.id);
            if (previous != lastConsumer.end()) {
                // The concrete fixture assigns A/B to distinct pipes by parity.
                // A WAIT is before its consumer's issue and the next SET is
                // after its producer's issue. On the same pipe, native start
                // order suffices, including one payload between WAIT and SET.
                // Across pipes the previous consumer must complete before the
                // next producer starts; keep that stronger event-graph check.
                const bool samePipe = previous->second%2 == handoff.source%2;
                const auto previousEvent = 2*previous->second + (samePipe ? 0 : 1);
                if (!graph[previousEvent][2*handoff.source]) {
                    llvm::errs() << "unsafe repeated ID reuse " << q << ":" << begin << ":" << end
                                 << " consumer=" << previous->second << " producer=" << handoff.source << "\n";
                    return false;
                }
            }
            lastConsumer[handoff.id] = handoff.target;
        }
    }
    return recordsSeen.size() == specs.size() && e.error().empty();
}
// Concrete handoff intervals are the oracle. A common residual predicate may
// guard many sequential lifetimes; overlapping envelopes must retain pairwise
// compatibility, and genuinely overlapping live intervals must stay separate.
bool checkGuardedPaletteBounds(func::FuncOp function, Operation* anchor,
                               unsigned count, unsigned layout, unsigned guardMode)
{
    fs::RegionalAnalysis body;
    body.expressions = std::make_shared<fs::RegionExpressions>();
    body.capabilities.exactQueries = true;
    auto& e = *body.expressions;
    const auto gate = e.input(function.getArgument(1));
    const auto other = e.eq(e.input(function.getArgument(0)), e.constant(0));
    std::vector<pto::CompoundInstanceElement> phases;
    phases.reserve(2);
    for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M}) {
        phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
                            SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
        phases.back().elementOp = anchor;
        body.anchors.push_back({&phases.back(), {}, {anchor->getBlock(), anchor},
                               {anchor->getBlock(), anchor->getNextNode()}});
        body.occurrenceLoops.push_back({});
    }
    std::map<std::pair<uint32_t, uint64_t>, fs::RegionExpressions::Id> presence;
    fs::RegionalAllocationSummary input;
    const unsigned groups = layout == 2 ? 2 : 3;
    for (unsigned g = 0; g < groups; ++g) {
        fs::RegionalAllocationGroup group{static_cast<uint32_t>(phases[0].kPipeValue),
                                         static_cast<uint32_t>(phases[1].kPipeValue), 1, {}};
        const auto active = g % 2 && guardMode ? (guardMode == 1 ? e.lnot(gate) : other) : gate;
        for (unsigned i = 0; i < count; ++i) {
            const uint64_t source = layout == 1 ? i * groups + g : g * count + i;
            const uint64_t target = source + (layout == 2 ? groups : 0);
            group.members.push_back({g * count + i, 0, 0,
                {0, e.constant(source), fs::PeriodicEventKind::Start},
                {1, e.constant(target), fs::PeriodicEventKind::Completion}, active});
            presence.emplace(std::make_pair(0U, source), active);
            presence.emplace(std::make_pair(1U, target), active);
        }
        input.groups.push_back(std::move(group));
    }
    body.presence = [&](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
        auto ordinal = e.constantValue(event.ordinal);
        if (!ordinal) { return {}; }
        auto found = presence.find({event.type, *ordinal});
        return found == presence.end() ? e.boolean(false) : found->second;
    };
    unsigned queries = 0;
    body.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b) {
        ++queries;
        const auto x = *e.constantValue(a.ordinal), y = *e.constantValue(b.ordinal);
        const bool forward = std::make_pair(x, a.type) < std::make_pair(y, b.type) ||
            (x == y && a.type == b.type && a.kind <= b.kind);
        return std::optional(e.land(e.land(*body.presence(a), *body.presence(b)), e.boolean(forward)));
    };
    auto output = fs::coalesceConstantRegionalAllocation(body, input);
    if (!output || (layout == 0 && guardMode == 0 && queries > 4 * groups)) { return false; }
    const unsigned expectedGroups = layout == 2 && guardMode != 1 ? 2 : 1;
    if (output->groups.size() != expectedGroups) { return false; }
    std::set<uint32_t> records;
    for (const auto& group : output->groups) {
        for (const auto& member : group.members) { records.insert(member.record); }
        for (unsigned mask = 0; mask < 4; ++mask) {
            fs::RegionExpressions::Transaction scratch(e);
            fs::RegionExpressions::Substitution values({{gate, e.boolean(mask & 1)},
                {other, e.boolean(mask & 2)}});
            std::vector<std::pair<uint64_t, uint64_t>> live;
            for (const auto& member : group.members) {
                auto active = e.constantValue(e.substitute(member.active, values));
                if (!active) { return false; }
                if (*active) {
                    live.emplace_back(*e.constantValue(member.firstSource.ordinal),
                                      *e.constantValue(member.lastTarget.ordinal));
                }
            }
            std::sort(live.begin(), live.end());
            for (std::size_t i = 1; i < live.size(); ++i) {
                if (live[i - 1].second >= live[i].first) { return false; }
            }
        }
    }
    return records.size() == groups * count;
}
// Compare extrema with direct enumeration of active original occurrences.
// Scrambled provenance does not affect order; a shuffled endpoint inventory
// exercises coordinate sorting; conditional order exercises the fallback,
// and mixed directions retain every bound.
bool checkOrderedExtrema(func::FuncOp function, unsigned count, bool shuffle, bool mixed, bool conditional = false)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto input = e.input(function.getArgument(0)), zero = e.constant(0), yes = e.boolean(true);
    fs::RegionalAnalysis body;
    body.expressions = arena;
    body.anchors.resize(2 * count);
    body.occurrenceLoops.resize(2 * count);
    body.presence = [yes](fs::RegionalEvent) { return std::optional(yes); };
    if (conditional) {
        const auto reversed = e.eq(e.rem(input, e.constant(2)), e.constant(1));
        body.referenceBefore = [&e, reversed](fs::RegionalEvent a, fs::RegionalEvent b) {
            return std::optional(e.select(reversed, e.boolean(a.type > b.type), e.boolean(a.type < b.type)));
        };
    }
    fs::RegionalAllocationGroup group{1, 2, 1, {}};
    for (unsigned i = 0; i < count; ++i) {
        // Repeating eight independently selectable guards keeps long fixtures
        // compact, while allowing every combination in the small oracle.
        auto active = e.eq(e.rem(e.div(input, e.constant(uint64_t(1) << (i % 8))), e.constant(2)), e.constant(1));
        fs::RegionalAllocationMember member{count - i, 0, 0,
            {2 * i, zero, fs::PeriodicEventKind::Start},
            {2 * i + 1, zero, fs::PeriodicEventKind::Completion}, active};
        if (mixed && i % 2) { member.sourcePipe = 2; member.targetPipe = 1; }
        group.members.push_back(std::move(member));
    }
    if (shuffle) { std::reverse(group.members.begin(), group.members.end()); }
    fs::RepeatedExtremaCost cost;
    const auto initialSize = e.size();
    auto result = fs::repeatedAllocationExtrema(body, group, &cost);
    if (!result || cost.candidates != count) { return false; }
    if (!conditional && !shuffle && !mixed &&
        (cost.orderedLists != 2 || cost.comparisons != 2 * (count ? count - 1 : 0) ||
         e.size() - initialSize > 8 * count + 2)) { return false; }
    if (shuffle && !conditional && !mixed && cost.orderedLists != 2) { return false; }
    if (!conditional && !mixed) {
        const auto unique = std::min(count, 8U);
        if (result->first.size() > unique || result->second.size() > unique ||
            cost.duplicateGuards != 2 * (count - unique) ||
            e.size() - initialSize > 8 * unique + 2) { return false; }
    }
    if (mixed && cost.duplicateGuards) { return false; }
    for (unsigned mask = 0; mask < 256; ++mask) {
        std::set<unsigned> expected;
        for (unsigned i = 0; i < count; ++i) {
            if (mask & (1U << (i % 8))) { expected.insert(i); }
        }
        fs::RegionExpressions::Transaction transaction(e);
        fs::RegionExpressions::Substitution bindings({{input, e.constant(mask)}});
        auto selected = [&](const auto& selectors) -> std::optional<std::set<unsigned>> {
            std::set<unsigned> active;
            for (const auto& selector : selectors) {
                auto present = e.constantValue(e.substitute(selector.present, bindings));
                if (!present) { return std::nullopt; }
                if (*present) { active.insert(selector.event.type / 2); }
            }
            return active;
        };
        auto first = selected(result->first), last = selected(result->second);
        auto expectedFirst = expected, expectedLast = expected;
        if (!mixed && !expected.empty()) {
            expectedFirst = {*expected.begin()}; expectedLast = {*expected.rbegin()};
            if (conditional && mask % 2) { std::swap(expectedFirst, expectedLast); }
        }
        if (first != std::optional(expectedFirst) || last != std::optional(expectedLast)) { return false; }
    }
    return true;
}
// Evaluate the shared binding against original coordinates, including a
// quotient outer frame and a fixed anchor coordinate. Neither the site guard
// nor invocation activity belongs to this coordinate-only interpretation.
bool checkOccurrenceBinding(func::FuncOp function, scf::ForOp outer, scf::ForOp inner)
{
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    auto& e = *region.expressions;
    region.anchors.resize(1);
    region.anchors[0].coordinates.push_back({inner, 2});
    region.occurrenceLoops = {inner};
    region.outerLoops = {{outer}};
    region.outerDivisors = {{2}};
    const auto condition = e.input(function.getArgument(1));
    region.endpointEventGuard = [condition](auto) { return std::optional(condition); };
    region.endpointSiteGuard = e.boolean(false);
    region.endpointInvocationGuard = e.boolean(false);
    fs::RegionalEvent event{0, e.constant(2), fs::PeriodicEventKind::Start, {e.constant(1)}};
    std::string error;
    auto constraints = fs::regionalOccurrenceConstraints(region, event, error);
    if (failed(constraints) || constraints->size() != 4) { return false; }
    auto guard = e.boolean(true);
    for (auto value : *constraints) { guard = e.land(guard, value); }
    const auto outerInput = e.input(outer.getInductionVar()), innerInput = e.input(inner.getInductionVar());
    for (uint64_t i = 0; i < 5; ++i) {
        for (uint64_t j = 0; j < 4; ++j) { for (bool enabled : {false, true}) {
            fs::RegionExpressions::Transaction transaction(e);
            fs::RegionExpressions::Substitution bindings({{outerInput, e.constant(i)},
                {innerInput, e.constant(j)}, {condition, e.boolean(enabled)}});
            auto actual = e.constantValue(e.substitute(guard, bindings));
            if (actual != std::optional<uint64_t>(enabled && i / 2 == 1 && j == 2)) { return false; }
        } }
    }
    region.outerDivisors[0][0] = 0;
    if (succeeded(fs::regionalOccurrenceConstraints(region, event, error))) { return false; }
    region.outerDivisors[0][0] = 2;
    auto missingVisit = event; missingVisit.visits.clear();
    if (succeeded(fs::regionalOccurrenceConstraints(region, missingVisit, error))) { return false; }
    region.anchors[0].coordinates[0].loop = {};
    if (succeeded(fs::regionalOccurrenceConstraints(region, event, error))) { return false; }
    region.anchors[0].coordinates.clear();
    region.endpointEventGuard = [](auto) { return std::optional(fs::RegionExpressions::invalid); };
    return failed(fs::regionalOccurrenceConstraints(region, event, error)) &&
        !fs::originalLoopOrdinal(e, {}) && !fs::originalLoopOrdinal(e, inner, 0);
}
// An invariant unknown return edge cannot certify reuse within any of six
// gaps. Its repeated unknown/false proof requests must share work within one
// call, while a second attempt after rollback must perform its own proofs.
bool checkReuseProofIsolation(func::FuncOp function, Operation* anchor, scf::ForOp loop)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        const auto initialSize = e.size();
        {
            fs::RegionExpressions::Transaction transaction(e);
            fs::RepeatedRegionState state;
            state.function = function; state.loop = loop; state.body.expressions = arena;
            state.body.capabilities = {true, true, true, false};
            state.trips = e.constant(8);
            const auto zero = e.constant(0), yes = e.boolean(true), gate = e.input(function.getArgument(1));
            std::vector<pto::CompoundInstanceElement> phases;
            phases.reserve(2);
            for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M}) {
                phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
                                    SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
                phases.back().elementOp = anchor;
                state.body.anchors.push_back({&phases.back(), {}, {anchor->getBlock(), anchor},
                                              {anchor->getBlock(), anchor->getNextNode()}});
                state.body.occurrenceLoops.push_back({});
            }
            state.body.presence = [yes](fs::RegionalEvent) { return std::optional(yes); };
            state.body.reachability = [&e](fs::RegionalEvent a, fs::RegionalEvent b) {
                // A completes before B starts; B does not precede A locally.
                return std::optional(e.boolean(a.type < b.type || (a.type == b.type && a.kind <= b.kind)));
            };
            state.weightedAcross = [gate, yes](const fs::RegionalEvent& a, const fs::RegionalEvent& b,
                                              fs::RegionExpressions::Id) {
                const bool native = a.type == b.type &&
                    !(a.kind == fs::PeriodicEventKind::Completion && b.kind == fs::PeriodicEventKind::Start);
                return std::optional(a.type < b.type || native ? yes : gate);
            };
            fs::RegionalAllocationMember member{0, 0, 0,
                {0, zero, fs::PeriodicEventKind::Start}, {1, zero, fs::PeriodicEventKind::Completion}, yes};
            member.tupleRule = fs::PhysicalTupleRule{1, 0, {}};
            member.singletonHandoff = true;
            fs::RegionalAllocationSummary input;
            input.groups.push_back({static_cast<uint32_t>(phases[0].kPipeValue),
                                    static_cast<uint32_t>(phases[1].kPipeValue), 1, {member}});
            auto duplicate = input.groups.front();
            duplicate.members.front().record = 1;
            input.groups.push_back(std::move(duplicate));
            fs::RepeatedLaneProofCost cost;
            if (fs::periodicLaneAllocation(state, input, {}, &cost) ||
                !cost.implicationComputations || cost.implicationComputations >= cost.implicationRequests ||
                cost.constantComputations != 3 || cost.constantComputations >= cost.constantRequests ||
                !cost.relativeComputations || cost.relativeComputations >= cost.relativeQueries ||
                !cost.proofNodeVisits) { return false; }
            // Exactly the true, false, and unknown-gate consequences are
            // specialized under true. Repeated unknown results are cached too.
            state.weightedAcross = [](const auto&, const auto&, auto) {
                return std::optional<fs::RegionExpressions::Id>();
            };
            fs::RepeatedLaneProofCost unavailable;
            if (fs::periodicLaneAllocation(state, input, {}, &unavailable) ||
                !unavailable.relativeComputations || unavailable.relativeComputations >= unavailable.relativeQueries) {
                return false;
            }
            uint64_t saturated = UINT64_MAX;
            if (!e.implies(yes, yes, &saturated) || e.constantUnder(yes, gate, &saturated) ||
                saturated != UINT64_MAX) { return false; }
        }
        if (e.size() != initialSize) { return false; }
    }
    return true;
}
} // namespace
bool runRepeatedRegionChecks(func::FuncOp function)
{
    Operation* anchor = nullptr;
    function.walk([&](Operation* op) { if (op->hasAttr("test.nested")) { anchor = op; } });
    if (!anchor) { return false; }
    auto inner = anchor->getParentOfType<scf::ForOp>();
    auto outer = inner ? inner->getParentOfType<scf::ForOp>() : scf::ForOp();
    if (!outer) { return false; }
    uint64_t checked = 0;
    for (unsigned t = 0; t <= 3; ++t) {
        for (unsigned k = 0; k <= 4; ++k) { for (unsigned mask = 0; mask < 8; ++mask) {
            if (!check(function, anchor, outer, inner, t, k, mask, checked)) { return false; }
        } }
    }
    if (function.getNumArguments() < 2 || !function.getArgument(1).getType().isInteger(1) ||
        !check(function, anchor, outer, inner, 2, 3, 7, checked, true) ||
        !check(function, anchor, outer, inner, 2, 3, 3, checked, true)) { return false; }
    if (!checkOrderedLaneAllocations(function, anchor, inner) ||
        !checkOccurrenceBinding(function, outer, inner) ||
        !checkReuseProofIsolation(function, anchor, inner)) { return false; }
    for (unsigned count : {0U, 1U, 8U, 128U}) {
        if (!checkOrderedExtrema(function, count, false, false)) { return false; }
    }
    if (!checkOrderedExtrema(function, 128, true, false) ||
        !checkOrderedExtrema(function, 32, false, true) ||
        !checkOrderedExtrema(function, 8, false, false, true)) { return false; }
    for (unsigned guardMode = 0; guardMode < 3; ++guardMode) {
        for (unsigned count : {1U, 8U, 128U}) {
            for (unsigned layout : {0U, 1U}) {
                if (!checkGuardedPaletteBounds(function, anchor, count, layout, guardMode)) { return false; }
            }
        }
        if (!checkGuardedPaletteBounds(function, anchor, 1, 2, guardMode)) { return false; }
    }
    uint64_t allocations = 0;
    fs::RepeatedLaneProofCost proofCost;
    for (unsigned q = 1; q <= 3; ++q) {
        for (unsigned end = 0; end <= 3*q; ++end) {
            for (unsigned begin = 0; begin <= end; ++begin) {
                for (bool storage : {false, true}) { for (bool omit : {false, true}) {
                    if (!allocationCheck(function, anchor, inner, q, begin, end, storage, omit, allocations)) {
                        return false;
                    }
                    if (storage && !omit &&
                        !allocationCheck(function, anchor, inner, q, begin, end, true, false,
                                         allocations, true, 0, &proofCost)) {
                        return false;
                    }
                } }
            }
        }
    }
    // One independently enabled lane must not use an absent middle lane as
    // an implicit successor. Both valuations check the exact same typed scheme.
    for (unsigned guardMode : {1U, 2U}) {
        if (!allocationCheck(function, anchor, inner, 2, 1, 6, true, false,
                             allocations, true, guardMode, &proofCost)) { return false; }
    }
    // Exercise a positive wrap even if a sufficient assignment uses all six
    // labels. Short/partial intervals above separately check missing phases;
    // these longer finite graphs check repeated collisions under both guards.
    for (unsigned q = 1; q <= 3; ++q) {
        for (unsigned begin : {0U, 1U}) {
            for (unsigned guardMode : {1U, 2U}) {
                if (!allocationCheck(function, anchor, inner, q, begin, 8*q, true, false,
                                     allocations, true, guardMode, &proofCost)) { return false; }
            }
        }
    }
    if (!proofCost.implicationComputations ||
        proofCost.implicationComputations >= proofCost.implicationRequests ||
        proofCost.constantComputations > proofCost.constantRequests) { return false; }
    llvm::outs() << "repeated allocation proof requests=" << proofCost.implicationRequests
                 << " computations=" << proofCost.implicationComputations << "\n";
    llvm::outs() << "repeated allocation checked " << allocations << " member envelopes and reuse chains\n";
    llvm::outs() << "repeated region checked " << checked << " event pairs\n";
    return true;
}
