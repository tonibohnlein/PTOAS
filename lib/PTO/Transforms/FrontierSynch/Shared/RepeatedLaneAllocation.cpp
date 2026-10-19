// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
#include "ModuloLaneSpan.h"
#include "RepeatedLaneAllocation.h"
#include "RepeatedRegionInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalLaneExports.h"
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicSharedAllocation.h"
#include <algorithm>
#include <chrono>
#include <map>

namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
constexpr uint64_t maximumPaletteWidth = NativeEventIdCount;
class WorkTimer {
public:
    explicit WorkTimer(uint64_t* elapsed) : elapsed(elapsed)
    {
        if (elapsed) { started = Clock::now(); }
    }
    ~WorkTimer()
    {
        if (!elapsed) { return; }
        const auto duration = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started).count());
        *elapsed += std::min(duration, UINT64_MAX - *elapsed);
    }
private:
    using Clock = std::chrono::steady_clock;
    uint64_t* elapsed;
    Clock::time_point started;
};
// One call owns caches for one append-only arena; existing expression IDs
// retain their meaning. Caches do not survive the call or cross arenas.
// Keep implication and specialization separate to preserve the existing lazy
// proof sequence and its sufficient (rather than complete) answers.
class ReuseProof {
public:
    ReuseProof(RepeatedRegionState& state, RepeatedLaneProofCost* cost) : state(state), e(state.e()), cost(cost) {}
    Id activity(Id end, Id begin)
    {
        count(&RepeatedLaneProofCost::activityRequests);
        const auto key = std::make_pair(end, begin);
        auto found = activities.find(key);
        if (found != activities.end()) { return found->second; }
        count(&RepeatedLaneProofCost::activityComputations);
        WorkTimer timer(cost ? &cost->activityNanoseconds : nullptr);
        auto active = e.land(end, begin);
        activities.emplace(key, active);
        return active;
    }
    bool implies(Id premise, Id consequence)
    {
        if (cost) { increment(cost->implicationRequests); }
        const auto key = std::make_pair(premise, consequence);
        auto found = implications.find(key);
        if (found != implications.end()) { return found->second; }
        if (cost) { increment(cost->implicationComputations); }
        WorkTimer timer(cost ? &cost->proofNanoseconds : nullptr);
        const auto answer = e.implies(premise, consequence, cost ? &cost->proofNodeVisits : nullptr);
        implications.emplace(key, answer);
        return answer;
    }
    std::optional<uint64_t> constantUnder(Id premise, Id expression)
    {
        if (cost) { increment(cost->constantRequests); }
        const auto key = std::make_pair(premise, expression);
        auto found = constants.find(key);
        if (found != constants.end()) { return found->second; }
        if (cost) { increment(cost->constantComputations); }
        WorkTimer timer(cost ? &cost->proofNanoseconds : nullptr);
        const auto answer = e.constantUnder(premise, expression, cost ? &cost->proofNodeVisits : nullptr);
        constants.emplace(key, answer);
        return answer;
    }
    void count(uint64_t RepeatedLaneProofCost::*field, uint64_t amount = 1)
    {
        if (!cost) { return; }
        auto& value = cost->*field;
        value += std::min(amount, UINT64_MAX - value);
    }
    std::optional<Id> relativeQuery(const RegionalEvent& source, const RegionalEvent& target, uint64_t gap)
    {
        count(&RepeatedLaneProofCost::relativeQueries);
        const auto key = std::make_tuple(eventKey(source), eventKey(target), gap);
        auto found = relativeQueries.find(key);
        if (found != relativeQueries.end()) { return found->second; }
        count(&RepeatedLaneProofCost::relativeComputations);
        WorkTimer timer(cost ? &cost->relativeNanoseconds : nullptr);
        auto answer = state.relativeQuery(source, target, gap);
        relativeQueries.emplace(key, answer);
        return answer;
    }
private:
    using EventKey = RepeatedRegionState::EventKey;
    static EventKey eventKey(const RegionalEvent& event)
    {
        return {event.type, event.ordinal, event.kind, event.visits};
    }
    static void increment(uint64_t& value) { if (value != UINT64_MAX) { ++value; } }
    // The state fixes the command graph and invocation. No rollback occurs in
    // this scope; cache false and unavailable results as well as true ones.
    RepeatedRegionState& state;
    RegionExpressions& e;
    RepeatedLaneProofCost* cost;
    // Cache before conjunction normalization: the implication cache is keyed
    // by its result and cannot avoid reconstructing the same activity at each
    // candidate gap. Keep ordered operands to preserve the existing IDs.
    std::map<std::pair<Id, Id>, Id> activities;
    std::map<std::pair<Id, Id>, bool> implications;
    std::map<std::pair<Id, Id>, std::optional<uint64_t>> constants;
    std::map<std::tuple<EventKey, EventKey, uint64_t>, std::optional<Id>> relativeQueries;
};
struct Lane {
    RegionalAllocationGroup group;
    RepeatedAllocationExtrema bounds;
    uint64_t delay = 0;
    bool child = true;
    std::optional<std::pair<RegionalEvent, RegionalEvent>> orderedPhase;
};

// A unit lane may contain many handoffs. Retain unconditional extrema before
// invocation clipping; these describe one complete phase of the repeated word.
std::optional<std::pair<RegionalEvent, RegionalEvent>> orderedPhase(
    const RegionalAnalysis& body, const RegionalAllocationGroup& group)
{
    if (group.terminalCleanup || group.members.empty()) { return std::nullopt; }
    auto& e = *body.expressions;
    auto first = group.members.front().firstSource, last = group.members.front().lastTarget;
    for (const auto& member : group.members) {
        auto sourcePresent = regionalPresence(body, member.firstSource);
        auto targetPresent = regionalPresence(body, member.lastTarget);
        if (e.constantValue(member.active) != 1 || !sourcePresent || !targetPresent ||
            e.constantValue(*sourcePresent) != 1 || e.constantValue(*targetPresent) != 1) { return std::nullopt; }
        auto earlier = regionalReferenceBefore(body, member.firstSource, first);
        auto later = regionalReferenceBefore(body, last, member.lastTarget);
        if (!earlier || !later) { return std::nullopt; }
        const auto a = e.constantValue(*earlier), b = e.constantValue(*later);
        if (!a || !b) { return std::nullopt; }
        if (*a) { first = member.firstSource; }
        if (*b) { last = member.lastTarget; }
    }
    return std::make_pair(std::move(first), std::move(last));
}
bool addLane(RepeatedRegionState& state, std::vector<Lane>& lanes,
             RegionalAllocationGroup group, uint64_t delay, bool child, RepeatedLaneProofCost* cost)
{
    if (group.budget != 1 || group.members.empty()) { return false; }
    for (const auto& member : group.members) {
        if (child && (!member.tupleRule || !validPhysicalTupleRule(*member.tupleRule, 1) ||
                      !canRestrictRepeatedAllocationMember(state, member))) { return false; }
    }
    auto phase = orderedPhase(state.body, group);
    // Intersect each member's invocation with the retained phase interval
    // before constructing the reuse graph. A clipped-away handoff still needs
    // a record mapping, but must not become a phantom vertex in a cycle.
    for (auto& member : group.members) {
        auto clipped = member;
        if (!liftRepeatedAllocationEnvelope(state, clipped, delay)) { return false; }
        member.active = clipped.active;
    }
    auto bounds = repeatedAllocationExtrema(state.body, group, cost ? &cost->extrema : nullptr);
    if (!bounds || bounds->first.empty() != bounds->second.empty()) { return false; }
    lanes.push_back({std::move(group), std::move(*bounds), delay, child, std::move(phase)});
    return true;
}

bool addCrossings(RepeatedRegionState& state, std::vector<Lane>& lanes,
                  ArrayRef<std::pair<uint32_t, std::size_t>> crossings, RepeatedLaneProofCost* cost)
{
    for (const auto& [record, index] : crossings) {
        if (index >= state.crossings.size()) { return false; }
        const auto& crossing = state.crossings[index];
        if (crossing.source.type >= state.body.anchors.size() ||
            crossing.target.type >= state.body.anchors.size()) { return false; }
        const auto& source = state.body.anchors[crossing.source.type];
        const auto& target = state.body.anchors[crossing.target.type];
        if (!source.phase || !target.phase) { return false; }
        auto p = static_cast<uint32_t>(source.phase->kPipeValue);
        auto q = static_cast<uint32_t>(target.phase->kPipeValue);
        if (p == q) { continue; }
        auto first = crossing.source, last = crossing.target;
        first.kind = PeriodicEventKind::Start;
        last.kind = PeriodicEventKind::Completion;
        RegionalAllocationGroup group{p, q, 1, {{record, 0, 0, first, last, crossing.guard}}};
        if (!addLane(state, lanes, std::move(group), 1, false, cost)) { return false; }
    }
    return true;
}

std::optional<uint64_t> reuseCost(RepeatedRegionState& state, const Lane& from, const Lane& to, ReuseProof& proof)
{
    auto& e = state.e();
    // Costs propose an assignment only. A separate all-pair check below
    // certifies reuse when intervening handoffs have different activity.
    // Restrict to certified nonnegative source displacements. This is a
    // sufficient matrix, not the set of every possible hardware assignment.
    for (uint64_t gap = from.delay; gap <= maximumPaletteWidth; ++gap) {
        proof.count(&RepeatedLaneProofCost::candidateGaps);
        bool proven = true;
        for (const auto& end : from.bounds.second) {
            for (const auto& begin : to.bounds.first) {
                proof.count(&RepeatedLaneProofCost::boundaryPairs);
                auto active = proof.activity(end.present, begin.present);
                if (proof.implies(active, e.boolean(false))) { continue; }
                auto consumer = end.event, producer = begin.event;
                const auto& anchors = state.body.anchors;
                if (consumer.type >= anchors.size() || producer.type >= anchors.size() ||
                    !anchors[consumer.type].phase || !anchors[producer.type].phase) { return std::nullopt; }
                // Match regionalHandoffReuse: a WAIT precedes its payload's
                // start, while a SET follows its payload's issue on that pipe.
                // Native start order therefore suffices on one pipe, including
                // the same occurrence; cross-pipe reuse requires completion.
                producer.kind = PeriodicEventKind::Start;
                consumer.kind = anchors[consumer.type].phase->kPipeValue ==
                    anchors[producer.type].phase->kPipeValue ?
                    PeriodicEventKind::Start : PeriodicEventKind::Completion;
                // The WAIT endpoint of an outer handoff is already one period
                // later than its SET. Subtract that consumer delay exactly once.
                auto query = proof.relativeQuery(consumer, producer, gap - from.delay);
                if (!query || !(proof.implies(active, *query) || proof.constantUnder(active, active) == 0 ||
                                proof.constantUnder(active, *query) == 1)) { proven = false; break; }
            }
            if (!proven) { break; }
        }
        if (proven) { return gap; }
    }
    return std::nullopt;
}

// Export each physical label's actual first/last member occurrences after
// clipping. A member's source visits form one interval; its consumer is delayed
// by a fixed number of repeats. Only the at-most-eight physical labels expand.
void appendBounds(RepeatedRegionState& state, RegionalAllocationGroup& result,
                  const RegionalAllocationMember& member, uint64_t delay,
                  uint64_t base, uint64_t width, uint64_t offset, uint64_t stride)
{
    if (width == 1) {
        result.lanes[base].firstSources.push_back({member.firstSource, member.active});
        result.lanes[base].lastTargets.push_back({member.lastTarget, member.active});
        return;
    }
    auto& e = state.e();
    const auto first = member.firstSource.visits.front();
    const auto last = e.sub(member.lastTarget.visits.front(), e.constant(delay));
    const detail::ModuloLaneDomain domain(stride, offset, width);
    for (uint64_t lane = 0; lane < width; ++lane) {
        const auto span = domain.span(e, first, last, member.active, lane);
        auto source = member.firstSource, target = member.lastTarget;
        source.visits.front() = span.first;
        target.visits.front() = e.add(span.last, e.constant(delay));
        result.lanes[base + lane].firstSources.push_back({std::move(source), span.present});
        result.lanes[base + lane].lastTargets.push_back({std::move(target), span.present});
    }
}

bool appendLane(RepeatedRegionState& state, RegionalAllocationGroup& result,
                const Lane& lane, uint64_t base, uint64_t width, uint64_t offset, uint64_t stride = 1)
{
    for (auto member : lane.group.members) {
        auto direction = regionalAllocationDirection(lane.group, member);
        member.sourcePipe = direction.first;
        member.targetPipe = direction.second;
        auto arity = lane.child ? member.tupleRule->coordinateCount : 0;
        if (arity == UINT64_MAX) { return false; }
        PhysicalTupleRule rule{arity + 1, base,
            {{lane.child ? uint64_t(1) : uint64_t(0), stride, offset, width, 1}}};
        member.tupleRule = std::move(rule);
        member.stride = 0;
        member.phase = 0;
        if (!liftRepeatedAllocationEnvelope(state, member, lane.delay)) { return false; }
        appendBounds(state, result, member, lane.delay, base, width, offset, stride);
        result.members.push_back(std::move(member));
    }
    return true;
}

// The formula (visit+offset)%width collides first at this positive gap.
// Larger collisions are safe through native start order at the destination
// source site. Its activity is immutable across repeats, and a witness between
// two retained occurrences stays inside the contiguous clipped interval.
bool compatible(const PeriodicReuseMatrix& costs, std::size_t from, std::size_t to,
                uint64_t fromOffset, uint64_t toOffset, uint64_t width)
{
    if (!costs[from][to]) { return false; }
    auto gap = (fromOffset + width - toOffset) % width;
    if (!gap) {
        // Distinct phases using one label in the same visit require an actual
        // command ordering; unrelated absent intermediates cannot supply it.
        if (from != to && *costs[from][to] != 0 && (!costs[to][from] || *costs[to][from] != 0)) {
            return false;
        }
        gap = width;
    }
    return *costs[from][to] <= gap;
}

bool certifyAssignment(const PeriodicReuseMatrix& costs, const PeriodicSharedAllocation& assignment)
{
    if (assignment.status != PeriodicSharedAllocationStatus::Success ||
        !assignment.budget || assignment.budget > maximumPaletteWidth ||
        assignment.phases.size() != costs.size()) { return false; }
    for (std::size_t i = 0; i < costs.size(); ++i) {
        const auto& from = assignment.phases[i];
        for (std::size_t j = 0; j < costs.size(); ++j) {
            const auto& to = assignment.phases[j];
            if (from.cycle != to.cycle) { continue; }
            if (from.laneCount != to.laneCount || !from.laneCount ||
                from.offset >= from.laneCount || to.offset >= to.laneCount ||
                !compatible(costs, i, j, from.offset, to.offset, from.laneCount)) { return false; }
        }
    }
    return true;
}

// A deterministic polynomial fallback when the cheapest cycle cover has an
// uncertified skipped phase (or conditional zero edges form a candidate cycle).
// Try at most NativeEventIdCount widths and offsets; this is sufficient, not a minimum.
std::optional<std::pair<uint64_t, std::vector<uint64_t>>> greedyAssignment(const PeriodicReuseMatrix& costs)
{
    for (uint64_t width = 1; width <= maximumPaletteWidth; ++width) {
        std::vector<uint64_t> offsets;
        for (std::size_t i = 0; i < costs.size(); ++i) {
            std::optional<uint64_t> selected;
            for (uint64_t offset = 0; offset < width; ++offset) {
                bool valid = compatible(costs, i, i, offset, offset, width);
                for (std::size_t j = 0; valid && j < i; ++j) {
                    valid = compatible(costs, i, j, offset, offsets[j], width) &&
                            compatible(costs, j, i, offsets[j], offset, width);
                }
                if (valid) { selected = offset; break; }
            }
            if (!selected) { break; }
            offsets.push_back(*selected);
        }
        if (offsets.size() == costs.size()) { return std::make_pair(width, std::move(offsets)); }
    }
    return std::nullopt;
}

// Keep independent palettes when some pairs have no certified order. Each
// insertion reruns the direct all-pair proof; a missing intermediate is never
// used as a witness. The bounded offset search costs O(NativeEventIdCount^2 n^3) in total.
std::optional<PeriodicSharedAllocation> partitionAssignment(const PeriodicReuseMatrix& costs, ReuseProof& proof)
{
    if (auto together = greedyAssignment(costs)) {
        PeriodicSharedAllocation result;
        result.budget = together->first;
        result.phases.resize(costs.size());
        for (std::size_t i = 0; i < costs.size(); ++i) {
            result.phases[i].laneCount = result.budget;
            result.phases[i].offset = together->second[i];
        }
        return result;
    }
    struct Palette {
        std::vector<std::size_t> members;
        uint64_t width = 0;
        std::vector<uint64_t> offsets;
    };
    std::vector<Palette> palettes;
    for (std::size_t i = 0; i < costs.size(); ++i) {
        if (!costs[i][i]) { return std::nullopt; }
        auto increase = std::max(uint64_t(1), *costs[i][i]);
        if (increase > maximumPaletteWidth) { return std::nullopt; }
        std::optional<std::size_t> chosen;
        Palette next{{i}, increase, {0}};
        for (std::size_t g = 0; g < palettes.size(); ++g) {
            auto members = palettes[g].members; members.push_back(i);
            proof.count(&RepeatedLaneProofCost::partitionAttempts);
            PeriodicReuseMatrix subset(members.size(), std::vector<std::optional<uint64_t>>(members.size()));
            for (std::size_t a = 0; a < members.size(); ++a) {
                for (std::size_t b = 0; b < members.size(); ++b) {
                    proof.count(&RepeatedLaneProofCost::partitionMatrixEntries);
                    subset[a][b] = costs[members[a]][members[b]];
                }
            }
            auto placed = greedyAssignment(subset);
            if (!placed || placed->first < palettes[g].width ||
                placed->first - palettes[g].width >= increase) { continue; }
            increase = placed->first - palettes[g].width;
            chosen = g;
            next = {std::move(members), placed->first, std::move(placed->second)};
        }
        if (chosen) { palettes[*chosen] = std::move(next); }
        else { palettes.push_back(std::move(next)); }
    }
    PeriodicSharedAllocation result;
    result.phases.resize(costs.size());
    for (std::size_t g = 0; g < palettes.size(); ++g) {
        const auto& palette = palettes[g];
        if (palette.width > maximumPaletteWidth - result.budget) { return std::nullopt; }
        for (std::size_t i = 0; i < palette.members.size(); ++i) {
            auto& phase = result.phases[palette.members[i]];
            phase.cycle = g; phase.laneBegin = result.budget;
            phase.laneCount = palette.width; phase.offset = palette.offsets[i];
        }
        result.budget += palette.width;
    }
    return result;
}

struct OrderedLaneResult {
    RepeatedLaneAllocationStatus status = RepeatedLaneAllocationStatus::NotApplicable;
    std::optional<RegionalAllocationGroup> assignment;
    std::shared_ptr<const RepeatedLaneCapacityWitness> overCapacity;
};
struct OrderedPhase { std::size_t lane; std::vector<uint64_t> key; };

// Check a fixed-size finite antichain, not just failure of a sufficient reuse
// search. All selected spans must actually occur in this complete invocation.
std::shared_ptr<const RepeatedLaneCapacityWitness> capacityWitness(
    RepeatedRegionState& state, ArrayRef<Lane> lanes, ArrayRef<OrderedPhase> phases,
    uint64_t first, ReuseProof& proof)
{
    auto trips = state.e().constantValue(state.trips);
    if (!trips || !state.body.capabilities.exactQueries) { return {}; }
    const uint64_t count = phases.size(), delay = lanes[phases.front().lane].delay;
    const uint64_t lastVisit = (first + maximumPaletteWidth) / count;
    if (delay >= *trips || lastVisit >= *trips - delay) { return {}; }
    auto result = std::make_shared<RepeatedLaneCapacityWitness>();
    result->expressions = state.body.expressions;
    result->sourcePipe = lanes[phases.front().lane].group.sourcePipe;
    result->targetPipe = lanes[phases.front().lane].group.targetPipe;
    if (result->sourcePipe == result->targetPipe) { return {}; }
    result->phasesPerVisit = count; result->consumerDelay = delay; result->admittedTrips = *trips;
    for (uint64_t i = first; i <= first + maximumPaletteWidth; ++i) {
        const auto& lane = lanes[phases[i % count].lane];
        RepeatedLaneCapacityWitness::Span span{lane.orderedPhase->first, lane.orderedPhase->second,
                                              i / count, i / count + delay, {}};
        span.source.kind = PeriodicEventKind::Start;
        span.target.kind = PeriodicEventKind::Completion;
        for (const auto& member : lane.group.members) { span.records.push_back(member.record); }
        result->spans.push_back(std::move(span));
    }
    for (std::size_t i = 0; i < result->spans.size(); ++i) {
        for (std::size_t j = i + 1; j < result->spans.size(); ++j) {
            const auto& a = result->spans[i]; const auto& b = result->spans[j];
            if (a.targetVisit > b.sourceVisit) { continue; }
            proof.count(&RepeatedLaneProofCost::orderedWitnessQueries);
            auto query = proof.relativeQuery(a.target, b.source, b.sourceVisit - a.targetVisit);
            if (!query || state.e().constantValue(*query) != 0) { return {}; }
            result->negativeReuse.push_back(*query);
        }
    }
    return result;
}

// The admitted word has ordered publications and ordered consumptions, all
// present on the same complete generation interval. Native publication order
// makes each exact reuse set a suffix; consumption order makes its first
// reusable position monotone. Truncating the word preserves these witnesses.
// Unknown queries or mixed/partial phases retain the general matrix route.
OrderedLaneResult allocateOrderedLanes(
    RepeatedRegionState& state, ArrayRef<Lane> lanes, ArrayRef<std::size_t> live, ReuseProof& proof)
{
    proof.count(&RepeatedLaneProofCost::orderedAttempts);
    if (live.empty() || live.size() > UINT32_MAX || state.phaseCount != 1 ||
        state.originalBegin != RegionExpressions::invalid ||
        state.originalTrips != RegionExpressions::invalid) { return {}; }
    auto& e = state.e();
    const auto delay = lanes[live.front()].delay;
    std::vector<OrderedPhase> phases;
    for (const auto index : live) {
        const auto& lane = lanes[index];
        if (!lane.orderedPhase || lane.delay != delay) { return {}; }
        const auto& source = lane.orderedPhase->first;
        std::vector<uint64_t> key;
        for (const auto coordinate : source.visits) {
            auto value = e.constantValue(coordinate);
            if (!value) { return {}; }
            key.push_back(*value);
        }
        auto ordinal = e.constantValue(source.ordinal);
        if (!ordinal) { return {}; }
        key.push_back(*ordinal); key.push_back(source.type);
        if (!phases.empty() && phases.front().key.size() != key.size()) { return {}; }
        phases.push_back({index, std::move(key)});
    }
    std::sort(phases.begin(), phases.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    for (std::size_t i = 1; i < phases.size(); ++i) {
        const auto& a = *lanes[phases[i - 1].lane].orderedPhase;
        const auto& b = *lanes[phases[i].lane].orderedPhase;
        auto sources = regionalReferenceBefore(state.body, a.first, b.first);
        auto targets = regionalReferenceBefore(state.body, a.second, b.second);
        if (!sources || !targets || e.constantValue(*sources) != 1 || e.constantValue(*targets) != 1) {
            return {};
        }
    }
    proof.count(&RepeatedLaneProofCost::orderedPhases, phases.size());
    const uint64_t count = phases.size();
    uint64_t cursor = 1, width = 1;
    for (uint64_t i = 0; i < count; ++i) {
        cursor = std::max(cursor, i + 1);
        bool found = false;
        for (; cursor <= i + maximumPaletteWidth; ++cursor) {
            if (cursor / count < delay) { continue; }
            auto consumer = lanes[phases[i].lane].orderedPhase->second;
            auto producer = lanes[phases[cursor % count].lane].orderedPhase->first;
            consumer.kind = PeriodicEventKind::Completion;
            producer.kind = PeriodicEventKind::Start;
            proof.count(&RepeatedLaneProofCost::orderedReuseQueries);
            auto query = proof.relativeQuery(consumer, producer, cursor / count - delay);
            if (!query) { return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}}; }
            auto answer = e.constantValue(*query);
            if (!answer) { return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}}; }
            if (*answer) { found = true; break; }
        }
        if (!found) {
            auto witness = capacityWitness(state, lanes, phases, i, proof);
            if (!witness) { return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}}; }
            proof.count(&RepeatedLaneProofCost::orderedOverCapacity);
            return {RepeatedLaneAllocationStatus::CertifiedOverCapacity, {}, std::move(witness)};
        }
        width = std::max(width, cursor - i);
    }
    RegionalAllocationGroup result{lanes.front().group.sourcePipe, lanes.front().group.targetPipe, width, {}};
    result.lanes.resize(width);
    for (uint64_t i = 0; i < count; ++i) {
        if (!appendLane(state, result, lanes[phases[i].lane], 0, width, i % width, count % width)) {
            return {};
        }
    }
    for (const auto& lane : lanes) {
        if (lane.bounds.first.empty() && !appendLane(state, result, lane, 0, 1, 0)) { return {}; }
    }
    proof.count(&RepeatedLaneProofCost::orderedAccepted);
    return {RepeatedLaneAllocationStatus::CertifiedAssignment, std::move(result), {}};
}

OrderedLaneResult allocateLanes(
    RepeatedRegionState& state, ArrayRef<Lane> lanes, ReuseProof& proof)
{
    std::vector<std::size_t> live;
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        if (!lanes[i].bounds.first.empty()) { live.push_back(i); }
    }
    const auto n = live.size();
    proof.count(&RepeatedLaneProofCost::lanes, lanes.size());
    for (const auto& lane : lanes) {
        proof.count(&RepeatedLaneProofCost::firstAlternatives, lane.bounds.first.size());
        proof.count(&RepeatedLaneProofCost::lastAlternatives, lane.bounds.second.size());
    }
    auto ordered = allocateOrderedLanes(state, lanes, live, proof);
    if (ordered.status == RepeatedLaneAllocationStatus::CertifiedAssignment ||
        ordered.status == RepeatedLaneAllocationStatus::CertifiedOverCapacity) { return ordered; }
    if (ordered.status == RepeatedLaneAllocationStatus::UnresolvedQuery) {
        proof.count(&RepeatedLaneProofCost::orderedUnresolved);
    }
    if (n > UINT32_MAX || (n && n > std::vector<std::optional<uint64_t>>().max_size() / n)) {
        return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}};
    }
    PeriodicReuseMatrix costs(n, std::vector<std::optional<uint64_t>>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            proof.count(&RepeatedLaneProofCost::matrixPairs);
            costs[i][j] = reuseCost(state, lanes[live[i]], lanes[live[j]], proof);
        }
    }
    auto assignment = allocatePeriodicShared(costs);
    if (!certifyAssignment(costs, assignment)) {
        auto partitioned = partitionAssignment(costs, proof);
        if (!partitioned || (n && !certifyAssignment(costs, *partitioned))) {
            return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}};
        }
        assignment = std::move(*partitioned);
    }
    // The existing typed record mapping requires a positive placeholder budget
    // even when every guarded record is absent; no command uses that label.
    const auto budget = std::max(uint64_t(1), assignment.budget);
    RegionalAllocationGroup result{lanes.front().group.sourcePipe, lanes.front().group.targetPipe, budget, {}};
    result.lanes.resize(budget);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& phase = assignment.phases[i];
        if (!appendLane(state, result, lanes[live[i]], phase.laneBegin, phase.laneCount, phase.offset)) {
            return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}};
        }
    }
    // Absent records execute no commands and consume no additional lane.
    for (const auto& lane : lanes) {
        if (lane.bounds.first.empty() && !appendLane(state, result, lane, 0, 1, 0)) {
            return {RepeatedLaneAllocationStatus::UnresolvedQuery, {}, {}};
        }
    }
    return {RepeatedLaneAllocationStatus::CertifiedAssignment, std::move(result), {}};
}
} // namespace

RepeatedLaneAllocationResult periodicLaneAllocation(
    RepeatedRegionState& state, const RegionalAllocationSummary& coalesced,
    ArrayRef<std::pair<uint32_t, std::size_t>> crossings, RepeatedLaneProofCost* cost)
{
    ReuseProof proof(state, cost);
    std::vector<Lane> lanes;
    for (const auto& group : coalesced.groups) {
        for (auto lane : constantAllocationLanes(group)) {
            if (!addLane(state, lanes, std::move(lane), 0, true, cost)) { return {}; }
        }
    }
    if (!addCrossings(state, lanes, crossings, cost)) { return {}; }
    auto result = std::make_shared<RegionalAllocationSummary>();
    if (lanes.empty()) { return {RepeatedLaneAllocationStatus::CertifiedAssignment, std::move(result), {}}; }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<Lane>> directions;
    for (auto& lane : lanes) {
        const auto direction = regionalAllocationDirection(lane.group, lane.group.members.front());
        if (llvm::any_of(lane.group.members, [&](const auto& member) {
            return regionalAllocationDirection(lane.group, member) != direction;
        })) { return {}; }
        directions[direction].push_back(std::move(lane));
    }
    for (auto& direction : directions) {
        auto allocated = allocateLanes(state, direction.second, proof);
        if (!allocated.assignment) { return {allocated.status, {}, std::move(allocated.overCapacity)}; }
        result->groups.push_back(std::move(*allocated.assignment));
    }
    return {RepeatedLaneAllocationStatus::CertifiedAssignment, std::move(result), {}};
}
} // namespace mlir::pto::frontiersynch
