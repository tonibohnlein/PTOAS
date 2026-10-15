// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite lane enumeration and event closure for allocation frames.
#ifndef PTO_PERIODIC_LANE_BOUNDARY_CHECKS_H
#define PTO_PERIODIC_LANE_BOUNDARY_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Structured/VaryingRotatingAllocation.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/VaryingBoundaryQuery.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <map>
#include <tuple>
namespace mlir::pto::test {
namespace lane_boundary {
namespace fs = frontiersynch;
using Occ = fs::BoundaryOccurrence;
using Kind = fs::PeriodicEventKind;
using Position = std::pair<uint64_t, uint32_t>;
using LaneKey = std::pair<uint32_t, uint64_t>;
inline bool failedCheck(unsigned line)
{
    llvm::errs() << "periodic lane-boundary mismatch at line " << line << "\n";
    return false;
}
inline fs::VaryingBoundaryMatrix referenceProduct(
    const fs::VaryingBoundaryMatrix& left, const fs::VaryingBoundaryMatrix& right)
{
    auto result = left;
    for (std::size_t i = 0; i < left.size(); ++i) {
        for (std::size_t j = 0; j < left.size(); ++j) {
            bool reaches = false;
            for (std::size_t k = 0; k < left.size(); ++k) { reaches |= left[i][k] && right[k][j]; }
            result[i][j] = reaches;
        }
    }
    return result;
}
inline bool powerConstruction()
{
    using Matrix = fs::VaryingBoundaryMatrix;
    const Matrix identity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const Matrix nilpotent{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
    const Matrix permutation{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
    const Matrix branch{{1, 1, 0}, {0, 0, 1}, {0, 0, 1}};
    if (referenceProduct(permutation, branch) == referenceProduct(branch, permutation)) {
        return failedCheck(__LINE__);
    }
    std::vector<std::vector<Matrix>> cases{{}, {identity}, {nilpotent}, {permutation}, {permutation, branch}};
    for (unsigned count : {3, 16, 64}) { cases.emplace_back(count, identity); }
    for (const auto& suffix : cases) {
        uint64_t rotations = 0, squares = 0;
        auto powers = fs::buildVaryingBoundaryPowers(suffix, rotations, squares);
        if (rotations != 3 * suffix.size() || powers.size() != suffix.size()) { return failedCheck(__LINE__); }
        uint64_t expectedSquares = 0;
        for (std::size_t phase = 0; phase < suffix.size(); ++phase) {
            auto expected = identity;
            for (std::size_t step = 0; step < suffix.size(); ++step) {
                expected = referenceProduct(expected, suffix[(phase + step) % suffix.size()]);
            }
            const auto& column = powers[phase];
            if (column.empty() || column.size() > 64) { return failedCheck(__LINE__); }
            expectedSquares += column.size() < 64 ? column.size() : 63;
            for (unsigned bit = 0; bit < 64; ++bit) {
                if (column[std::min<std::size_t>(bit, column.size() - 1)] != expected) {
                    return failedCheck(__LINE__);
                }
                expected = referenceProduct(expected, expected);
            }
        }
        if (squares != expectedSquares) { return failedCheck(__LINE__); }
        if (suffix.size() == 1 && suffix.front() == permutation && powers.front().size() != 64) {
            return failedCheck(__LINE__);
        }
        if (!suffix.empty() && suffix.front() == identity && squares != suffix.size()) {
            return failedCheck(__LINE__);
        }
    }
    return true;
}
inline bool spans(const fs::RotatingBoundaryCertificate& boundary, uint64_t length)
{
    auto type = boundary.select(length);
    std::map<LaneKey, std::vector<std::pair<Position, Position>>> expected;
    for (const auto& rule : boundary.laneRules) {
        const auto& record = boundary.quotient.generators[rule.record];
        for (uint64_t lane = 0; lane < rule.width; ++lane) { expected.try_emplace({rule.group, lane}); }
        for (uint64_t n = 0; n < length && record.displacement < length - n; ++n) {
            const auto lane = (llvm::APInt(128, rule.stride) * llvm::APInt(128, n) +
                llvm::APInt(128, rule.phase)).urem(llvm::APInt(128, rule.width)).getZExtValue();
            expected[{rule.group, lane}].push_back({{n, record.source}, {n + record.displacement, record.target}});
        }
    }
    if (expected.size() != type.lanes.size() || !type.index.error.empty()) { return failedCheck(__LINE__); }
    for (const auto& [lane, uses] : expected) {
        const auto found = type.lanes.find(lane);
        if (found == type.lanes.end()) { return failedCheck(__LINE__); }
        const auto& actual = found->second;
        if (bool(actual.firstPublication) != !uses.empty() || bool(actual.lastConsumption) != !uses.empty()) {
            return failedCheck(__LINE__);
        }
        if (uses.empty()) { continue; }
        auto first = uses.front().first, last = uses.front().second;
        for (const auto& use : uses) { first = std::min(first, use.first); last = std::max(last, use.second); }
        if (Position(actual.firstPublication->at(length), actual.firstPublication->type) != first ||
            Position(actual.lastConsumption->at(length), actual.lastConsumption->type) != last) {
            return failedCheck(__LINE__);
        }
        for (auto occurrence : {*actual.firstPublication, *actual.lastConsumption}) {
            for (auto kind : {Kind::Start, Kind::Completion}) {
                if (!type.eventIds.count({occurrence.type, occurrence.at(length), kind})) {
                    return failedCheck(__LINE__);
                }
            }
        }
    }
    return true;
}
inline bool geometry()
{
    auto quotient = fs::analyzePeriodicDemands({{0}, {1}, {0}, {1}},
        {{0, 1, 0}, {2, 3, 0}, {1, 0, 2}, {3, 2, 2}});
    auto original = fs::buildRotatingBoundaryCertificate(std::move(quotient), {}, {}, {});
    if (!original.error.empty()) { return failedCheck(__LINE__); }
    std::vector<uint32_t> records;
    for (auto id : original.quotient.retained) {
        const auto& r = original.quotient.generators[id];
        if (original.quotient.payloads[r.source].pipe == 0 && original.quotient.payloads[r.target].pipe == 1) {
            records.push_back(id);
        }
    }
    if (records.size() != 2) { return failedCheck(__LINE__); }
    unsigned checked = 0;
    for (uint64_t width = 1; width <= 6; ++width) {
        for (uint64_t stride = 0; stride <= 2 * width; ++stride) {
            for (uint64_t phase = 0; phase < width; ++phase) {
                std::vector<fs::BoundaryLaneRule> rules{{records[0], 0, stride, phase, width},
                    {records[1], 0, stride, (phase + 1) % width, width}};
                auto extended = fs::extendRotatingLaneBoundary(original, rules);
                if (!extended.error.empty() || extended.refresh < original.refresh ||
                    extended.period % original.period) { return failedCheck(__LINE__); }
                for (uint64_t length = 0; length < 32; ++length) {
                    if (!spans(extended, length)) { return false; }
                    ++checked;
                }
            }
        }
    }
    auto wide = fs::extendRotatingLaneBoundary(original, {{records[0], 0, UINT64_MAX, 5, 6}});
    if (!wide.error.empty() || !spans(wide, 29)) { return failedCheck(__LINE__); }
    for (const auto& rule : std::vector<fs::BoundaryLaneRule>{
             {records[0], 0, 1, 0, 0}, {records[0], 0, 1, 6, 6}, {UINT32_MAX, 0, 1, 0, 2}}) {
        if (fs::extendRotatingLaneBoundary(original, {rule}).error.empty()) { return failedCheck(__LINE__); }
    }
    auto mixed = std::vector<fs::BoundaryLaneRule>{{records[0], 0, 1, 0, 2}, {records[1], 0, 1, 0, 3}};
    if (fs::extendRotatingLaneBoundary(original, mixed).error.empty() ||
        fs::extendRotatingLaneBoundary(original, {{records[0], 0, 1, 0, UINT64_MAX}}, 256, UINT64_MAX).error.empty() ||
        fs::extendRotatingLaneBoundary(original, {{records[0], 0, 1, 0, 3}}, 2).error.empty() ||
        original.period != 1 || original.refresh != 1 || !original.laneRules.empty()) { return failedCheck(__LINE__); }
    llvm::outs() << "lane geometry: " << checked << " exact finite enumerations passed\n";
    return true;
}
struct FiniteGraph {
    std::vector<unsigned> starts, lengths;
    std::vector<llvm::BitVector> reach;
    explicit FiniteGraph(const fs::AffineRotatingVisits& visits, unsigned count)
    {
        struct Payload { unsigned pipe, slot; bool write; };
        std::vector<Payload> payloads;
        for (unsigned v = 0; v < count; ++v) {
            starts.push_back(payloads.size()); lengths.push_back(visits.slope * v + visits.intercept);
            for (unsigned n = 0; n < lengths.back(); ++n) {
                payloads.push_back({0, n % 2, true}); payloads.push_back({1, n % 2, false});
            }
        }
        std::vector<std::vector<unsigned>> outgoing(2 * payloads.size());
        std::map<unsigned, unsigned> previous;
        for (unsigned i = 0; i < payloads.size(); ++i) {
            outgoing[2 * i].push_back(2 * i + 1);
            if (auto found = previous.find(payloads[i].pipe); found != previous.end()) {
                outgoing[2 * found->second].push_back(2 * i);
                outgoing[2 * found->second + 1].push_back(2 * i + 1);
            }
            previous[payloads[i].pipe] = i;
            for (unsigned j = 0; j < i; ++j) {
                if (payloads[i].slot == payloads[j].slot && (payloads[i].write || payloads[j].write)) {
                    outgoing[2 * j + 1].push_back(2 * i);
                }
            }
        }
        reach.assign(outgoing.size(), llvm::BitVector(outgoing.size()));
        for (std::size_t end = outgoing.size(); end; --end) {
            auto& row = reach[end - 1];
            for (auto target : outgoing[end - 1]) { row.set(target); row |= reach[target]; }
            row.set(end - 1);
        }
    }
    std::optional<unsigned> id(Occ occurrence, Kind kind, unsigned visit) const
    {
        if (visit >= lengths.size() || (occurrence.tail && occurrence.coordinate > lengths[visit]) ||
            (!occurrence.tail && occurrence.coordinate >= lengths[visit])) { return std::nullopt; }
        return 2 * (starts[visit] + 2 * occurrence.at(lengths[visit]) + occurrence.type) + unsigned(kind);
    }
};
inline bool queries(uint64_t slope, uint64_t intercept)
{
    const std::vector<fs::RotatingFragment> effects{
        {0, 0, 0, 2, 1, 0, false, true, 0}, {1, 0, 0, 2, 1, 0, true, false, 0}};
    auto original = std::make_shared<const fs::AffineRotatingVisits>(fs::analyzeAffineRotatingVisits(
        fs::buildRotatingBoundaryCertificate({{0}, {1}}, effects, {{0, 0, 0}, {0, 0, 1}}), slope, intercept));
    if (!original->error.empty()) { return failedCheck(__LINE__); }
    std::vector<fs::BoundaryLaneRule> rules;
    uint32_t group = 0;
    for (auto id : original->child.quotient.retained) { rules.push_back({id, group++, 1, 1, 3}); }
    auto refined = std::make_shared<const fs::AffineRotatingVisits>(
        fs::extendAffineRotatingLaneBoundary(*original, rules));
    std::string error;
    auto frame = fs::VaryingBoundaryFrame::build(original, refined, error);
    if (!frame || !error.empty() || refined->startup <= original->startup || refined->child.period % 3) {
        return failedCheck(__LINE__);
    }
    fs::VaryingBoundaryQuery query(*frame);
    if (!query.error().empty()) { return failedCheck(__LINE__); }
    auto fullFrame = *frame;
    for (auto& column : fullFrame.powers) {
        while (column.size() < 64) { column.push_back(referenceProduct(column.back(), column.back())); }
    }
    fs::VaryingBoundaryQuery fullQuery(fullFrame);
    for (auto a : frame->ports) {
        for (auto b : frame->ports) {
            for (uint64_t gap : {uint64_t(0), uint64_t(1) << 32, uint64_t(1) << 63}) {
                const auto actual = query.query(a.occurrence, a.kind, b.occurrence, b.kind, refined->startup, gap);
                const auto expected = fullQuery.query(
                    a.occurrence, a.kind, b.occurrence, b.kind, refined->startup, gap);
                if (!actual || actual != expected) { return failedCheck(__LINE__); }
            }
        }
    }
    const unsigned visits = refined->startup + 2 * refined->period + 4;
    FiniteGraph graph(*original, visits);
    std::vector<fs::VaryingBoundaryPort> ports = frame->ports;
    for (uint32_t type = 0; type < 2; ++type) {
        ports.push_back({{type, refined->child.refresh - 1, false}, Kind::Start});
        ports.push_back({{type, refined->child.refresh, true}, Kind::Completion});
    }
    unsigned checked = 0;
    for (unsigned visit = 0; visit < visits; ++visit) {
        for (unsigned gap : {0U, 1U, 2U, 5U}) {
            if (gap >= visits - visit) { continue; }
            for (auto a : ports) {
                for (auto b : ports) {
                    const auto x = graph.id(a.occurrence, a.kind, visit);
                    const auto y = graph.id(b.occurrence, b.kind, visit + gap);
                    const bool expected = x && y && graph.reach[*x].test(*y);
                    const auto actual = query.query(a.occurrence, a.kind, b.occurrence, b.kind, visit, gap);
                    if (!actual || *actual != expected) { return failedCheck(__LINE__); }
                    ++checked;
                }
            }
        }
    }
    // Existing records/crossings remain unchanged when proof classes refine.
    for (unsigned v = 0; v < visits; ++v) {
        const auto& a = original->crossingInto(v); const auto& b = refined->crossingInto(v);
        if (a.size() != b.size()) { return failedCheck(__LINE__); }
        for (unsigned i = 0; i < a.size(); ++i) {
            if (std::tie(a[i].source.type, a[i].source.coordinate, a[i].source.tail,
                         a[i].target.type, a[i].target.coordinate, a[i].target.tail) !=
                std::tie(b[i].source.type, b[i].source.coordinate, b[i].source.tail,
                         b[i].target.type, b[i].target.coordinate, b[i].target.tail)) { return failedCheck(__LINE__); }
        }
    }
    // Exercise the public allocation boundary with an adversarial capability:
    // all contexts qualify except one original-congruent refined residue.
    // A successful representative must not hide that missing universal proof.
    fs::RegionalAnalysis regional;
    regional.expressions = std::make_shared<fs::RegionExpressions>();
    regional.anchors.resize(original->child.quotient.payloads.size());
    auto records = fs::enumerateVaryingAllocationRecords(*original, error);
    if (!records) { return failedCheck(__LINE__); }
    const auto trips = regional.expressions->constant(visits);
    fs::VaryingBoundaryReuseQuery available = [](auto, auto, auto, auto, auto, auto) {
        return std::optional(true);
    };
    if (!fs::buildVaryingRotatingAllocation(regional, *original, trips, *records, available, error, refined.get())) {
        return failedCheck(__LINE__);
    }
    auto crossing = std::find_if(records->begin(), records->end(), [](const auto& record) {
        return record.kind == fs::VaryingAllocationRecordKind::SuffixCrossing;
    });
    if (crossing == records->end()) { return failedCheck(__LINE__); }
    const auto firstTarget = original->startup + (crossing->targetVisit ? crossing->targetVisit : original->period);
    const auto rejectedVisit = firstTarget + original->period;
    unsigned rejected = 0;
    fs::VaryingBoundaryReuseQuery missingResidue = [&](auto, auto, auto, auto, uint64_t visit, auto) {
        if (visit == rejectedVisit) { ++rejected; return std::optional<bool>(); }
        return std::optional(true);
    };
    const auto missing = fs::buildVaryingRotatingAllocation(
        regional, *original, trips, *records, missingResidue, error, refined.get());
    if (missing ||
        !rejected) { return failedCheck(__LINE__); }
    auto unrelated = std::make_shared<fs::AffineRotatingVisits>(*refined);
    unrelated->child.quotient.payloads[0].pipe = 4;
    if (fs::VaryingBoundaryFrame::build(original, unrelated, error) ||
        fs::buildVaryingRotatingAllocation(regional, *original, trips, *records, available, error, unrelated.get())) {
        return failedCheck(__LINE__);
    }
    llvm::outs() << "lane frame: " << checked << " independent event queries passed\n";
    return true;
}
} // namespace lane_boundary
inline bool checkPeriodicLaneBoundaries()
{
    return lane_boundary::powerConstruction() && lane_boundary::geometry() && lane_boundary::queries(1, 0) &&
        lane_boundary::queries(1, 2) && lane_boundary::queries(2, 1);
}
} // namespace mlir::pto::test
#endif
