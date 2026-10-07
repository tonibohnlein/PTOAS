// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/VaryingRotatingAllocation.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Kind = VaryingAllocationRecordKind;
using Record = VaryingAllocationRecord;
constexpr uint64_t maxWidth = 6;
bool add(uint64_t a, uint64_t b, uint64_t& out)
{
    if (b > UINT64_MAX - a) { return false; }
    out = a + b;
    return true;
}
bool multiply(uint64_t a, uint64_t b, uint64_t& out)
{
    if (b && a > UINT64_MAX / b) { return false; }
    out = a * b;
    return true;
}
std::optional<uint64_t> length(const AffineRotatingVisits& visits, uint64_t v)
{
    uint64_t result;
    if (!multiply(visits.slope, v, result) || !add(result, visits.intercept, result)) { return std::nullopt; }
    return result;
}
struct InternalPhase { const Record* record; uint64_t offset; };
struct BuilderState {
    const AffineRotatingVisits& visits;
    RegionExpressions& e;
    Id trips;
    const VaryingBoundaryReuseQuery& query;
    Id c(uint64_t value) { return e.constant(value); }
    Id scaled(Id value, uint64_t scale)
    {
        Id result = c(0);
        while (scale) {
            if (scale & 1) { result = e.add(result, value); }
            scale >>= 1;
            if (scale) { value = e.add(value, value); }
        }
        return result;
    }
    Id lengthAt(Id visit) { return e.add(scaled(visit, visits.slope), c(visits.intercept)); }
    RegionalEvent event(BoundaryOccurrence occurrence, Id visit, PeriodicEventKind kind)
    {
        auto ordinal = occurrence.tail ? e.sub(lengthAt(visit), c(occurrence.coordinate)) : c(occurrence.coordinate);
        return {occurrence.type, ordinal, kind, {visit}};
    }
    RegionalAllocationGroup group(const Record& record, uint64_t budget)
    {
        return {visits.child.quotient.payloads[record.source].pipe,
            visits.child.quotient.payloads[record.target].pipe, budget, {}};
    }
    void direction(RegionalAllocationMember& member, const Record& record)
    {
        member.sourcePipe = visits.child.quotient.payloads[record.source].pipe;
        member.targetPipe = visits.child.quotient.payloads[record.target].pipe;
    }
    // First v for K(v)>d+residue. Positive slope makes this set a suffix.
    std::optional<uint64_t> firstVisit(uint64_t distance, uint64_t residue)
    {
        uint64_t threshold;
        if (!add(distance, residue, threshold) || !add(threshold, 1, threshold)) { return std::nullopt; }
        if (threshold <= visits.intercept) { return 0; }
        auto delta = threshold - visits.intercept;
        return delta / visits.slope + (delta % visits.slope != 0);
    }
    // One arithmetic progression of source visits, with a delayed consumer.
    // Keep inactive arithmetic defined; callers observe coordinates only under active.
    std::optional<std::pair<Id, Id>> finalVisit(uint64_t first, uint64_t period, uint64_t delay)
    {
        uint64_t threshold;
        if (!period || !add(first, delay, threshold)) { return std::nullopt; }
        auto active = e.lt(c(threshold), trips);
        auto end = e.select(active, e.sub(e.sub(trips, c(delay)), c(1)), c(first));
        auto last = e.sub(end, e.rem(e.sub(end, c(first)), c(period)));
        return std::make_pair(active, last);
    }
    RegionalAllocationMember internalMember(const Record& record, PhysicalTupleRule rule, uint64_t first)
    {
        auto nonempty = e.lt(c(first), trips);
        auto last = e.select(nonempty, e.sub(trips, c(1)), c(first));
        RegionalAllocationMember member{record.record, 0, 0,
            {record.source, c(0), PeriodicEventKind::Start, {c(first)}},
            {record.target, e.sub(lengthAt(last), c(1)), PeriodicEventKind::Completion, {last}}, nonempty};
        member.tupleRule = std::move(rule);
        direction(member, record);
        return member;
    }
    bool internalSpans(RegionalAllocationGroup& out, const InternalPhase& phase, uint64_t width, uint64_t banks)
    {
        const auto& record = *phase.record;
        for (uint64_t bank = 0; bank < banks; ++bank) {
            for (uint64_t lane = 0; lane < width; ++lane) {
                const auto residue = (lane + width - phase.offset) % width;
                auto first = firstVisit(record.displacement, residue);
                if (!first) { return false; }
                const auto delta = (bank + banks - *first % banks) % banks;
                if (!add(*first, delta, *first)) { return false; }
                auto final = finalVisit(*first, banks, 0);
                if (!final) { return false; }
                const auto active = final->first, lastVisit = final->second;
                auto end = e.select(active,
                    e.sub(e.sub(lengthAt(lastVisit), c(record.displacement)), c(1)), c(residue));
                auto lastSource = e.sub(end, e.rem(e.sub(end, c(residue)), c(width)));
                auto& span = out.lanes[bank * width + lane];
                span.firstSources.push_back(
                    {{record.source, c(residue), PeriodicEventKind::Start, {c(*first)}}, active});
                span.lastTargets.push_back({{record.target, e.add(lastSource, c(record.displacement)),
                    PeriodicEventKind::Completion, {lastVisit}}, active});
            }
        }
        return true;
    }
    bool wholeRecordReuse(const Record& record, uint64_t banks)
    {
        uint64_t contexts;
        if (!add(visits.startup, visits.period, contexts)) { return false; }
        for (uint64_t v = 0; v < contexts; ++v) {
            if (v < visits.startup) {
                auto count = length(visits, v);
                if (!count) { return false; }
                if (*count <= record.displacement) { continue; }
            }
            auto proven = query({record.target, 1, true}, PeriodicEventKind::Completion,
                {record.source, 0, false}, PeriodicEventKind::Start, v, banks);
            if (!proven || !*proven) { return false; }
        }
        return true;
    }
    // B|selectorPeriod makes the last-consumer residue constant in each long
    // class. B<=refresh places every attachment inside the existing cutoff proof.
    bool sharedReuse(ArrayRef<InternalPhase> phases, uint64_t width, uint64_t banks)
    {
        uint64_t contexts;
        if (!add(visits.startup, visits.period, contexts)) { return false; }
        for (uint64_t v = 0; v < contexts; ++v) {
            uint64_t next;
            if (!add(v, banks, next)) { return false; }
            auto count = length(visits, v), nextCount = length(visits, next);
            if (!count || !nextCount) { return false; }
            for (uint64_t lane = 0; lane < width; ++lane) {
                for (const auto& from : phases) {
                    const auto residue = (lane + width - from.offset) % width;
                    uint64_t firstConsumer;
                    if (!add(residue, from.record->displacement, firstConsumer)) { return false; }
                    if (*count <= firstConsumer) {
                        if (v >= visits.startup) { return false; }
                        continue;
                    }
                    const auto r = (*count - 1) % width, t = firstConsumer % width;
                    const auto tail = 1 + (r >= t ? r - t : width - (t - r));
                    for (const auto& to : phases) {
                        const auto head = (lane + width - to.offset) % width;
                        uint64_t threshold;
                        if (!add(head, to.record->displacement, threshold)) { return false; }
                        if (*nextCount <= threshold) {
                            if (next >= visits.startup) { return false; }
                            continue;
                        }
                        auto proof = query({from.record->target, tail, true}, PeriodicEventKind::Completion,
                            {to.record->source, head, false}, PeriodicEventKind::Start, v, banks);
                        if (!proof || !*proof) { return false; }
                    }
                }
            }
        }
        return true;
    }
    std::optional<RegionalAllocationGroup> internalGroup(ArrayRef<InternalPhase> phases,
                                                        uint64_t width, bool shared)
    {
        if (phases.empty() || !width || width > maxWidth) { return std::nullopt; }
        for (uint64_t banks = 1; banks <= maxWidth / width; ++banks) {
            if (!(shared ? sharedReuse(phases, width, banks) : wholeRecordReuse(*phases.front().record, banks))) {
                continue;
            }
            auto out = group(*phases.front().record, width * banks);
            out.lanes.resize(out.budget);
            for (const auto& phase : phases) {
                auto first = firstVisit(phase.record->displacement, 0);
                if (!first) { return std::nullopt; }
                PhysicalTupleRule rule{2, 0, {{0, 1, phase.offset, width, 1}, {1, 1, 0, banks, width}}};
                out.members.push_back(internalMember(*phase.record, std::move(rule), *first));
                if (!internalSpans(out, phase, width, banks)) { return std::nullopt; }
            }
            return out;
        }
        return std::nullopt;
    }
    std::optional<RegionalAllocationGroup> dedicated(const Record& record)
    {
        auto threshold = visits.child.quotient.eventThreshold(
            {record.target, PeriodicEventKind::Completion}, {record.source, PeriodicEventKind::Start});
        uint64_t width;
        if (threshold.error != PeriodicQueryError::None || !threshold.displacement ||
            !add(record.displacement, *threshold.displacement, width)) { return std::nullopt; }
        width = std::max(uint64_t(1), width);
        const InternalPhase phase{&record, 0};
        return internalGroup({phase}, width, false);
    }
    std::optional<RegionalAllocationGroup> crossing(const Record& record)
    {
        const bool singleton = record.kind == Kind::SingletonCrossing;
        uint64_t firstTarget = record.targetVisit, width = 1;
        if (singleton) {
            if (!firstTarget) { return std::nullopt; }
        } else {
            if (!add(visits.startup, record.targetVisit ? record.targetVisit : visits.period, firstTarget)) {
                return std::nullopt;
            }
            uint64_t representative;
            if (!add(visits.startup, visits.period, representative) ||
                !add(representative, record.targetVisit, representative)) { return std::nullopt; }
            for (; width <= maxWidth; ++width) {
                uint64_t gap;
                if (!multiply(visits.period / std::gcd(visits.period, width), width, gap)) { continue; }
                auto proof = query(record.consumption, PeriodicEventKind::Completion,
                    record.publication, PeriodicEventKind::Start, representative, gap - 1);
                if (proof && *proof) { break; }
            }
            if (width > maxWidth) { return std::nullopt; }
        }
        const auto first = firstTarget - 1;
        auto final = singleton ? std::make_optional(std::make_pair(e.lt(c(firstTarget), trips), c(first))) :
            finalVisit(first, visits.period, 1);
        if (!final) { return std::nullopt; }
        auto out = group(record, width);
        RegionalAllocationMember member{record.record, 0, 0,
            event(record.publication, c(first), PeriodicEventKind::Start),
            event(record.consumption, e.add(final->second, c(1)), PeriodicEventKind::Completion), final->first};
        member.tupleRule = PhysicalTupleRule{2, 0, singleton ? std::vector<PhysicalCoordinateTerm>{} :
            std::vector<PhysicalCoordinateTerm>{{1, 1, 0, width, 1}}};
        member.singletonHandoff = singleton;
        direction(member, record);
        out.members.push_back(std::move(member));
        out.lanes.resize(width);
        if (singleton) {
            out.lanes[0].firstSources.push_back({out.members[0].firstSource, final->first});
            out.lanes[0].lastTargets.push_back({out.members[0].lastTarget, final->first});
            return out;
        }
        uint64_t period;
        if (!multiply(visits.period / std::gcd(visits.period, width), width, period)) { return std::nullopt; }
        // At most six physical labels are inspected, never a runtime trip range.
        const auto candidates = width / std::gcd(visits.period, width);
        for (uint64_t k = 0; k < candidates; ++k) {
            uint64_t v;
            if (!multiply(k, visits.period, v) || !add(first, v, v)) { return std::nullopt; }
            auto span = finalVisit(v, period, 1);
            if (!span) { return std::nullopt; }
            auto& lane = out.lanes[v % width];
            lane.firstSources.push_back({event(record.publication, c(v), PeriodicEventKind::Start), span->first});
            lane.lastTargets.push_back({event(record.consumption, e.add(span->second, c(1)),
                PeriodicEventKind::Completion), span->first});
        }
        return out;
    }
};
} // namespace
std::optional<std::vector<VaryingAllocationRecord>> enumerateVaryingAllocationRecords(
    const AffineRotatingVisits& visits, std::string& error)
{
    error.clear();
    std::vector<Record> out;
    const auto& child = visits.child.quotient;
    if (!visits.error.empty() || !visits.child.error.empty() || !child.error.empty() || !visits.slope ||
        !visits.period || visits.startupCrossings.size() != visits.startup ||
        visits.suffixCrossings.size() != visits.period) {
        error = "varying allocation requires a complete affine boundary certificate";
        return std::nullopt;
    }
    auto append = [&](Record record) {
        if (out.size() >= UINT32_MAX || record.source >= child.payloads.size() ||
            record.target >= child.payloads.size()) { return false; }
        record.record = static_cast<uint32_t>(out.size());
        out.push_back(std::move(record));
        return true;
    };
    for (auto id : child.retained) {
        if (id >= child.generators.size()) { error = "invalid varying child record"; return std::nullopt; }
        const auto& r = child.generators[id];
        if (!append({0, r.source, r.target, Kind::Internal, r.displacement, {}, {}, 0, id})) {
            error = "varying allocation record identity overflow"; return std::nullopt;
        }
    }
    auto crossings = [&](ArrayRef<BoundaryDemand> records, uint64_t target, Kind kind) {
        for (const auto& r : records) {
            if (!append({0, r.source.type, r.target.type, kind, 0, r.source, r.target, target})) { return false; }
        }
        return true;
    };
    for (uint64_t v = 1; v < visits.startup; ++v) {
        if (!crossings(visits.startupCrossings[v], v, Kind::SingletonCrossing)) {
            error = "varying startup record identity overflow"; return std::nullopt;
        }
    }
    if (!crossings(visits.seam, visits.startup, Kind::SingletonCrossing)) {
        error = "varying seam record identity overflow"; return std::nullopt;
    }
    for (uint64_t phase = 0; phase < visits.period; ++phase) {
        if (!crossings(visits.suffixCrossings[phase], phase, Kind::SuffixCrossing)) {
            error = "varying suffix record identity overflow"; return std::nullopt;
        }
    }
    return out;
}
std::shared_ptr<RegionalAllocationSummary> buildVaryingRotatingAllocation(
    const RegionalAnalysis& region, const AffineRotatingVisits& visits, Id trips,
    ArrayRef<VaryingAllocationRecord> records, const VaryingBoundaryReuseQuery& query, std::string& error)
{
    error.clear();
    if (!region.expressions || trips >= region.expressions->size() || region.expressions->isBoolean(trips) ||
        !query || !visits.slope || !visits.period || !visits.child.period ||
        !visits.error.empty() || !visits.child.error.empty() ||
        region.anchors.size() != visits.child.quotient.payloads.size()) {
        error = "varying allocation requires an exact numerical boundary query and original coordinate frame";
        return {};
    }
    BuilderState builder{visits, *region.expressions, trips, query};
    auto out = std::make_shared<RegionalAllocationSummary>();
    std::map<uint32_t, const Record*> internal;
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        if (r.record != i || r.source >= region.anchors.size() || r.target >= region.anchors.size() ||
            (r.kind == Kind::SuffixCrossing && r.targetVisit >= visits.period)) {
            error = "varying allocation record table does not match its original sites";
            return {};
        }
        if (r.kind == Kind::Internal && !internal.emplace(r.childRecord, &r).second) {
            error = "duplicate varying internal allocation record";
            return {};
        }
    }
    std::set<uint32_t> allocated;
    auto shared = buildPeriodicSharedAssignment(visits.child.quotient);
    if (shared) {
        for (std::size_t cycle = 0; cycle < shared->allocation.cycles.size(); ++cycle) {
            const auto width = shared->allocation.cycles[cycle].laneCount;
            if (!width || width > maxWidth || width > visits.child.refresh || visits.child.period % width ||
                visits.child.cutoff <= 2 * width - 1) { continue; }
            // d<=B below gives d+residue<=2B-1<cutoff. Therefore every
            // member/lane exists throughout each long class; only exact short
            // startup checks may omit an absent member.
            std::vector<InternalPhase> phases;
            bool valid = true;
            for (std::size_t p = 0; p < shared->allocation.phases.size(); ++p) {
                const auto& phase = shared->allocation.phases[p];
                if (phase.cycle != cycle) { continue; }
                auto found = internal.find(shared->records[p]);
                if (found == internal.end() || found->second->displacement > width || phase.offset >= width) {
                    valid = false; break;
                }
                phases.push_back({found->second, phase.offset});
            }
            if (!valid) { continue; }
            auto group = builder.internalGroup(phases, width, true);
            if (!group) { continue; }
            for (const auto& phase : phases) { allocated.insert(phase.record->record); }
            out->groups.push_back(std::move(*group));
        }
    }
    for (const auto& record : records) {
        const auto& payloads = visits.child.quotient.payloads;
        if (payloads[record.source].pipe == payloads[record.target].pipe || allocated.count(record.record)) {
            continue;
        }
        auto group = record.kind == Kind::Internal ? builder.dedicated(record) : builder.crossing(record);
        if (!group) {
            error = "varying record " + std::to_string(record.record) +
                " has no sufficient reuse certificate with a palette of at most six lanes; no scarcity claim";
            return {};
        }
        out->groups.push_back(std::move(*group));
    }
    // M2 proves each inner lane is a chain even for a truncated child visit.
    // Positive affine length makes its visit activity a suffix, so L-step bank
    // proofs compose without skipping an inactive intermediate use. Whole-record
    // fallback uses the record's own handoffs for the same bridging argument.
    // Each original record occurs exactly once. M3 may share these palettes
    // using the supplied exact lane spans. This producer makes no pool minimum claim.
    return out;
}
} // namespace mlir::pto::frontiersynch
