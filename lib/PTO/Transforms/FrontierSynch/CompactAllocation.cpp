// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using System = DifferenceBoundSystem;
using Pair = std::pair<std::size_t, std::size_t>;
DictionaryAttr encode(PeriodicAllocation& allocation, int64_t plan, MLIRContext* context, StringRef strategy)
{
    NamedAttrList attributes(encodeCyclicAllocation(allocation, plan, context));
    attributes.set("strategy", StringAttr::get(context, strategy));
    return attributes.getDictionary(context);
}
bool orderedReuse(const ArithmeticDemandAnalysis& analysis,
                  const ArithmeticRelationKey& a, const System& x,
                  const ArithmeticRelationKey& b, const System& y)
{
    if (a.parameterResidues != b.parameterResidues) { return true; }
    const unsigned s = a.source.residues.size(), t = a.target.residues.size();
    if (b.source.residues.size() != s || b.target.residues.size() != t) { return false; }
    const unsigned params = 2 * (s + t), dimensions = params + analysis.parameterCount;
    SmallVector<unsigned> left, right, query;
    for (unsigned i = 0; i < s + t; ++i) { left.push_back(i); right.push_back(s + t + i); }
    for (unsigned i = 0; i < t; ++i) { query.push_back(s + i); }
    for (unsigned i = 0; i < s; ++i) { query.push_back(s + t + i); }
    for (unsigned i = 0; i < analysis.parameterCount; ++i) {
        left.push_back(params + i); right.push_back(params + i); query.push_back(params + i);
    }
    auto lx = x.remap(dimensions, left), ry = y.remap(dimensions, right);
    if (failed(lx) || failed(ry)) { return false; }
    auto pair = lx->intersect(*ry);
    if (failed(pair)) { return false; }
    auto from = a.target, to = b.source;
    from.event = ArithmeticEvent::Completion;
    to.event = ArithmeticEvent::Start;
    auto required = analysis.requiredOrder.find({from, to, a.parameterResidues});
    SmallVector<DifferenceBoundConstraint> prefix;
    for (unsigned i = 0; i < s; ++i) {
        // Original coordinates are period*q+residue; compare lexicographically.
        auto atoms = prefix;
        BoundInteger limit = BoundInteger(static_cast<int64_t>(b.source.residues[i])) -
                             BoundInteger(static_cast<int64_t>(a.source.residues[i])) - BoundInteger(1);
        atoms.push_back({i + 1, s + t + i + 1,
            floorDiv(limit, BoundInteger(static_cast<int64_t>(analysis.period)))});
        auto order = System::create(dimensions, atoms);
        if (failed(order)) { return false; }
        auto candidate = pair->intersect(*order);
        if (failed(candidate)) { return false; }
        if (!candidate->isEmpty()) {
            auto projected = candidate->project(query);
            if (failed(projected) || required == analysis.requiredOrder.end()) { return false; }
            if (!llvm::any_of(required->second, [&](const auto& cover) { return projected->isSubsetOf(cover); })) {
                auto missing = subtractDifferenceBoundUnions(projected->dimensions(), {*projected}, required->second);
                if (failed(missing) || !missing->empty()) { return false; }
            }
        }
        if (a.source.residues[i] != b.source.residues[i]) { break; }
        prefix.push_back({i + 1, s + t + i + 1, BoundInteger(0)});
        prefix.push_back({s + t + i + 1, i + 1, BoundInteger(0)});
    }
    // No coordinates means one occurrence: no two distinct same-site sources.
    return true;
}
} // namespace
DictionaryAttr arithmeticAllocationCertificate(const ArithmeticDemandAnalysis& analysis,
    ArrayRef<uint32_t> pipes, const std::map<Pair, int64_t>& records, int64_t plan, MLIRContext* context)
{
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.period || analysis.period > INT64_MAX) {
        return {};
    }
    using Entry = std::pair<const ArithmeticRelationKey*, const System*>;
    std::map<Pair, std::vector<Entry>> families;
    for (const auto& [key, pieces] : analysis.minimumDemands) {
        if (key.source.site >= pipes.size() || key.target.site >= pipes.size()) { return {}; }
        if (pipes[key.source.site] == pipes[key.target.site]) { continue; }
        for (const auto& piece : pieces) {
            if (!piece.isEmpty()) { families[{key.source.site, key.target.site}].push_back({&key, &piece}); }
        }
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<int64_t>> groups;
    for (const auto& [family, entries] : families) {
        auto record = records.find(family);
        if (record == records.end()) { return {}; }
        for (const auto& a : entries) {
            for (const auto& b : entries) {
                if (!orderedReuse(analysis, *a.first, *a.second, *b.first, *b.second)) { return {}; }
            }
        }
        groups[{pipes[family.first], pipes[family.second]}].push_back(record->second);
    }
    PeriodicAllocation allocation;
    for (const auto& [pipes, ids] : groups) {
        DirectedAllocation direction;
        direction.sourcePipe = pipes.first; direction.targetPipe = pipes.second;
        direction.uniformBudget = ids.size();
        for (auto id : ids) {
            if (id < 0 || static_cast<uint64_t>(id) > UINT32_MAX) { return {}; }
            DirectedHandoff handoff; handoff.record = id;
            direction.handoffs.push_back(handoff);
        }
        allocation.directions.push_back(std::move(direction));
    }
    return encode(allocation, plan, context, "dedicated-families");
}
DictionaryAttr guardedAllocationCertificate(const GuardedRotatingAnalysis& analysis,
                                             int64_t plan, MLIRContext* context)
{
    if (!analysis.error.empty() || !analysis.expressions || !analysis.periodic.error.empty() ||
        analysis.generators.size() != analysis.periodic.retained.size()) { return {}; }
    auto& arena = *analysis.expressions;
    struct Record { uint32_t id; uint64_t gap; };
    std::map<std::pair<uint32_t, uint32_t>, std::vector<Record>> groups;
    for (uint32_t r = 0; r < analysis.generators.size(); ++r) {
        auto retained = analysis.periodic.retained[r];
        if (arena.implies(retained, arena.boolean(false))) { continue; }
        const auto& edge = analysis.generators[r];
        auto p = analysis.payloads[edge.source].pipe, q = analysis.payloads[edge.target].pipe;
        if (p == q) { continue; }
        std::vector<uint32_t> mapping(analysis.payloads.size(), UINT32_MAX);
        std::vector<PeriodicPayload> payloads;
        for (uint32_t i = 0; i < analysis.payloads.size(); ++i) {
            if (arena.implies(retained, analysis.payloads[i].presence)) {
                mapping[i] = payloads.size();
                payloads.push_back({analysis.payloads[i].pipe});
            }
        }
        auto distance = arena.constantUnder(retained, edge.displacement);
        if (!distance || mapping[edge.source] == UINT32_MAX || mapping[edge.target] == UINT32_MAX) { return {}; }
        std::vector<PeriodicRecord> generators;
        for (const auto& candidate : analysis.generators) {
            auto d = arena.constantUnder(retained, candidate.displacement);
            if (d && mapping[candidate.source] != UINT32_MAX && mapping[candidate.target] != UINT32_MAX &&
                (arena.implies(retained, candidate.active) || arena.constantUnder(retained, candidate.active) == 1)) {
                generators.push_back({mapping[candidate.source], mapping[candidate.target], *d});
            }
        }
        std::vector<PeriodicRecord> native;
        for (const auto& candidate : analysis.periodic.nativePrerequisites) {
            auto d = arena.constantUnder(retained, candidate.displacement);
            if (d && mapping[candidate.source] != UINT32_MAX && mapping[candidate.target] != UINT32_MAX &&
                arena.implies(retained, candidate.active)) {
                native.push_back({mapping[candidate.source], mapping[candidate.target], *d});
            }
        }
        auto witness = analyzePeriodicDemands(payloads, generators, native);
        if (!witness.error.empty()) { return {}; }
        auto reuse = witness.eventThreshold({mapping[edge.target], PeriodicEventKind::Completion},
                                             {mapping[edge.source], PeriodicEventKind::Start});
        if (reuse.error != PeriodicQueryError::None || !reuse.displacement ||
            *reuse.displacement > UINT64_MAX - *distance) { return {}; }
        groups[{p,q}].push_back({r, std::max(uint64_t(1), *distance + *reuse.displacement)});
    }
    PeriodicAllocation allocation;
    for (const auto& [pipes, records] : groups) {
        uint64_t gap = 1;
        for (const auto& record : records) { gap = std::max(gap, record.gap); }
        if (records.size() > UINT64_MAX / gap) { return {}; }
        DirectedAllocation direction;
        direction.sourcePipe = pipes.first; direction.targetPipe = pipes.second;
        direction.uniformBudget = records.size() * gap;
        for (const auto& record : records) {
            DirectedHandoff handoff; handoff.record = record.id;
            direction.handoffs.push_back(handoff);
        }
        allocation.directions.push_back(std::move(direction));
    }
    return encode(allocation, plan, context, "guarded-record-cycles");
}
} // namespace mlir::pto::frontiersynch
