// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite-lifetime oracle for the unconditional envelope fast path.
#ifndef PTO_COALESCING_ENVELOPE_CHECKS_H
#define PTO_COALESCING_ENVELOPE_CHECKS_H
#include <map>
#include <set>
namespace {
bool coalescingEnvelopeChecks(mlir::Operation* anchor, mlir::Value coordinate)
{
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    auto& e = *region.expressions;
    mlir::pto::CompoundInstanceElement sourcePhase(0, {}, {}, mlir::pto::PipelineType::PIPE_MTE2,
                                                   anchor->getName());
    mlir::pto::CompoundInstanceElement targetPhase(1, {}, {}, mlir::pto::PipelineType::PIPE_V,
                                                   anchor->getName());
    sourcePhase.elementOp = targetPhase.elementOp = anchor;
    region.anchors = {{&sourcePhase, {}, {}, {}}, {&targetPhase, {}, {}, {}}};
    region.occurrenceLoops.resize(2);
    region.capabilities.exactQueries = true;
    region.presence = [&](fs::RegionalEvent) { return e.boolean(true); };
    region.referenceBefore = [&](fs::RegionalEvent a, fs::RegionalEvent b) { return e.lt(a.ordinal, b.ordinal); };
    uint64_t queries = 0;
    region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b) {
        ++queries;
        auto x = e.constantValue(a.ordinal), y = e.constantValue(b.ordinal);
        return e.boolean(x && y && (*x < *y || (*x == *y &&
            (a.kind == fs::PeriodicEventKind::Start || b.kind == fs::PeriodicEventKind::Completion))));
    };
    const auto yes = e.boolean(true), guard = e.eq(e.input(coordinate), e.constant(0)), other = e.lnot(guard);
    const auto p = static_cast<uint32_t>(sourcePhase.kPipeValue);
    const auto q = static_cast<uint32_t>(targetPhase.kPipeValue);
    auto member = [&](uint32_t id, uint64_t first, uint64_t last, fs::RegionExpressions::Id active) {
        fs::RegionalAllocationMember out{id, 0, 0,
            {0, e.constant(first), fs::PeriodicEventKind::Start},
            {1, e.constant(last), fs::PeriodicEventKind::Completion}, active};
        out.tupleRule = fs::PhysicalTupleRule{1, 0, {}};
        return out;
    };
    auto check = [&](fs::RegionalAllocationSummary input, std::size_t palettes, uint64_t limit) {
        for (auto& group : input.groups) {
            // Request the constant-lane route, with its original certified bounds.
            group.lanes.resize(1);
            for (const auto& m : group.members) {
                group.lanes[0].firstSources.push_back({m.firstSource, m.active});
                group.lanes[0].lastTargets.push_back({m.lastTarget, m.active});
            }
        }
        queries = 0;
        auto result = fs::coalesceRegionalAllocation(region, input);
        if (!result || result->groups.size() != palettes || (limit && queries > limit)) { return false; }
        std::map<uint32_t, fs::RegionalAllocationMember> original;
        for (const auto& group : input.groups) {
            for (const auto& m : group.members) { original.emplace(m.record, m); }
        }
        std::set<uint32_t> retained;
        for (const auto& group : result->groups) {
            if (group.budget != 1 || group.sourcePipe != p || group.targetPipe != q) { return false; }
            for (const auto& m : group.members) {
                auto found = original.find(m.record);
                if (found == original.end() || !retained.insert(m.record).second ||
                    m.active != found->second.active || m.firstSource.ordinal != found->second.firstSource.ordinal ||
                    m.lastTarget.ordinal != found->second.lastTarget.ordinal) { return false; }
            }
            // Enumerate both guard valuations. A reused label must have disjoint
            // closed publication/consumption intervals; this does not call the
            // production reachability or envelope code.
            for (bool value : {false, true}) {
                auto active = [&](const auto& m) {
                    return m.active == yes || (m.active == guard && value) || (m.active == other && !value);
                };
                for (std::size_t a = 0; a < group.members.size(); ++a) {
                    for (std::size_t b = a + 1; b < group.members.size(); ++b) {
                        const auto& x = group.members[a]; const auto& y = group.members[b];
                        if (!active(x) || !active(y)) { continue; }
                        auto xf = e.constantValue(x.firstSource.ordinal), xl = e.constantValue(x.lastTarget.ordinal);
                        auto yf = e.constantValue(y.firstSource.ordinal), yl = e.constantValue(y.lastTarget.ordinal);
                        if (!xf || !xl || !yf || !yl || !(*xl < *yf || *yl < *xf)) { return false; }
                    }
                }
            }
        }
        return retained.size() == original.size() && e.error().empty();
    };
    for (uint32_t size : {32U, 128U, 512U}) {
        for (bool reverse : {false, true}) {
            fs::RegionalAllocationSummary input;
            for (uint32_t i = 0; i < size; ++i) {
                const uint32_t visit = reverse ? size - i - 1 : i;
                input.groups.push_back({p, q, 1, {member((37 * i) % size, 2 * visit, 2 * visit + 1, yes)}});
            }
            if (!check(std::move(input), 1, 4 * size)) { return false; }
        }
    }
    fs::RegionalAllocationSummary interleaved;
    interleaved.groups = {{p, q, 1, {member(0, 0, 1, yes), member(1, 4, 5, yes)}},
                          {p, q, 1, {member(2, 2, 3, yes)}}};
    if (!check(interleaved, 1, 0)) { return false; }
    fs::RegionalAllocationSummary overlap;
    overlap.groups = {{p, q, 1, {member(0, 0, 5, yes)}}, {p, q, 1, {member(1, 2, 3, yes)}}};
    if (!check(overlap, 2, 0)) { return false; }
    overlap.groups[1].members[0] = member(1, 0, 5, yes);
    if (!check(overlap, 2, 0)) { return false; }
    fs::RegionalAllocationSummary conditional;
    conditional.groups = {{p, q, 1, {member(0, 0, 1, guard), member(1, 4, 5, yes)}},
                          {p, q, 1, {member(2, 0, 1, other)}}};
    return check(conditional, 1, 0);
}
} // namespace
#endif
