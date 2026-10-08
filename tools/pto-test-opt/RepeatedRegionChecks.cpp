// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check hierarchical identities through two sequence exports, without insertion.
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
#include "PTO/Transforms/FrontierSynch/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RepeatedRegionInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RepeatedAllocation.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RepeatedLaneAllocation.h"
#include <map>
#include <set>
#include <tuple>
#include <algorithm>
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
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
        auto joined = fs::composeRegionalSequence(function, arena,
            {twice.regional, twice.regional}, false, false);
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
                     bool omitMember, uint64_t& checked, bool cyclePolicy = false, unsigned guardMode = 0)
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
        if (fs::repeatedRegionalAllocation(state, unqualified, crossings)) { return false; }
    }
    auto coalesced = fs::coalesceRegionalAllocation(body, child);
    if (!coalesced) { return false; }
    auto summary = cyclePolicy ? fs::periodicLaneAllocation(state, *coalesced, crossings) :
                                 fs::repeatedRegionalAllocation(state, child, crossings);
    if (!summary) {
        llvm::errs() << "allocation producer rejected " << q << ":" << begin << ":" << end
                     << " storage=" << storage << " omit=" << omitMember << " cycle_policy=" << cyclePolicy << "\n";
        return false;
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
    uint64_t allocations = 0;
    for (unsigned q = 1; q <= 3; ++q) {
        for (unsigned end = 0; end <= 3*q; ++end) {
            for (unsigned begin = 0; begin <= end; ++begin) {
                for (bool storage : {false, true}) { for (bool omit : {false, true}) {
                    if (!allocationCheck(function, anchor, inner, q, begin, end, storage, omit, allocations)) {
                        return false;
                    }
                    if (storage && !omit &&
                        !allocationCheck(function, anchor, inner, q, begin, end, true, false, allocations, true)) {
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
                             allocations, true, guardMode)) { return false; }
    }
    // Exercise a positive wrap even if a sufficient assignment uses all six
    // labels. Short/partial intervals above separately check missing phases;
    // these longer finite graphs check repeated collisions under both guards.
    for (unsigned q = 1; q <= 3; ++q) {
        for (unsigned begin : {0U, 1U}) {
            for (unsigned guardMode : {1U, 2U}) {
                if (!allocationCheck(function, anchor, inner, q, begin, 8*q, true, false,
                                     allocations, true, guardMode)) { return false; }
            }
        }
    }
    llvm::outs() << "repeated allocation checked " << allocations << " member envelopes and reuse chains\n";
    llvm::outs() << "repeated region checked " << checked << " event pairs\n";
    return true;
}
